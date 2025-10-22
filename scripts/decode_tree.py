import os, json, argparse, torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from models.student.gpt2_mtp import GPT2MTPStudent

def load_student(ckpt_dir, model_name="gpt2", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    meta_path = os.path.join(ckpt_dir, "mtp_meta.json")

    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        offsets = tuple(meta.get("offsets", [1,2]))
        base_name = meta.get("model_name", model_name)
        print(f"[INFO] Loaded metadata: base model = {base_name}, offsets = {offsets}")
    else:
        offsets = (1,2)
        base_name = model_name
        print(f"[WARN] No metadata found, falling back to model_name={model_name}")

    s = GPT2MTPStudent(model_name=base_name, offsets=offsets, device=device, lora=False)
    print(f"[INFO] Backbone skeleton built: hidden_size={s.config.hidden_size}, vocab={s.config.vocab_size}")

    base = AutoModelForCausalLM.from_pretrained(base_name).to(device)
    s.backbone = PeftModel.from_pretrained(base, ckpt_dir)
    lora_params = [n for n,p in s.backbone.named_parameters() if "lora" in n]
    print(f"[INFO] LoRA params loaded: {len(lora_params)} parameters with 'lora' in name")

    mtp_path = os.path.join(ckpt_dir, "mtp_head.pt")
    state = torch.load(mtp_path, map_location=device, weights_only=False)
    s.mtp_head.load_state_dict(state, strict=True)
    print(f"[INFO] Loaded MTP head weights from {mtp_path}")
    
    for off, head in s.mtp_head.heads.items():
        wnorm = head[0].weight.norm().item()
        print(f"   - offset {off}: first linear weight norm = {wnorm:.4f}")

    s.to(device)
    s.eval()
    return s, s.tokenizer

def sample_topk(logits, k=0, temperature=1.0):
    if temperature != 1.0:
        logits = logits / temperature
    if k and k > 0:
        v, i = torch.topk(logits, k=k, dim=-1)
        p = torch.softmax(v, dim=-1)
        j = torch.multinomial(p, 1)
        return i.gather(-1, j).squeeze(-1)
    return torch.argmax(logits, dim=-1)

@torch.no_grad()
def parallel_decode(student, tokenizer, prompt, max_new_tokens=32, 
                   draft_top_k=0, draft_temperature=1.0, 
                   verify_top_k=1, verbose=False):
    
    device = next(student.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)
    
    # Initial forward pass
    out = student.backbone(input_ids=input_ids, attention_mask=attn, 
                          use_cache=True, output_hidden_states=True, 
                          return_dict=True)
    past = out.past_key_values
    last_h = out.hidden_states[-1][:, -1:, :]  # [B, 1, D]
    
    # Ensure last_h is 3D [B, S, D]
    if last_h.dim() == 4:
        last_h = last_h.squeeze(1)
    
    eos = tokenizer.eos_token_id
    gen = []
    
    # Statistics tracking
    stats = {
        'both_accepted': 0,
        'first_only': 0,
        'both_rejected': 0,
        'total_tokens': 0
    }
    
    num_iterations = (max_new_tokens + 1) // 2
    
    for step in range(num_iterations):
        # Ensure last_h is always 3D [B, S, D]
        if last_h.dim() == 4:
            last_h = last_h.squeeze(1)
        
        # Draft BOTH tokens from current state
        mtp = student.mtp_head(last_h, last_only=True)
        log1 = mtp[1][:, -1, :]
        log2 = mtp[2][:, -1, :]
        
        prop1 = sample_topk(log1, k=draft_top_k, temperature=draft_temperature)
        prop2 = sample_topk(log2, k=draft_top_k, temperature=draft_temperature)
        
        # Create candidate sequence [prop1, prop2]
        candidates = torch.cat([prop1.unsqueeze(0), prop2.unsqueeze(0)], dim=1)  # [1, 2]
        
        # ⭐ SAVE the past BEFORE verification
        past_before_verify = past
        
        # SINGLE backbone forward pass for BOTH tokens (parallel verification)
        out = student.backbone(input_ids=candidates, 
                              past_key_values=past,
                              use_cache=True,
                              output_hidden_states=True,
                              return_dict=True)
        
        logits = out.logits  # [1, 2, vocab_size]
        
        # Verify both tokens
        if verify_top_k == 1:
            backbone_tok1 = torch.argmax(logits[:, 0, :], dim=-1)
            backbone_tok2 = torch.argmax(logits[:, 1, :], dim=-1)
            ok1 = (prop1.item() == backbone_tok1.item())
            ok2 = (prop2.item() == backbone_tok2.item())
        else:
            _, idx1 = torch.topk(logits[:, 0, :], k=verify_top_k, dim=-1)
            _, idx2 = torch.topk(logits[:, 1, :], k=verify_top_k, dim=-1)
            ok1 = prop1.item() in idx1[0].tolist()
            ok2 = prop2.item() in idx2[0].tolist()
            backbone_tok1 = torch.argmax(logits[:, 0, :], dim=-1)
            backbone_tok2 = torch.argmax(logits[:, 1, :], dim=-1)
        
        # Accept longest valid prefix
        if ok1 and ok2:
            # Both tokens accepted - use the state from verification
            gen.extend([prop1.item(), prop2.item()])
            stats['both_accepted'] += 1
            stats['total_tokens'] += 2
            past = out.past_key_values
            last_h = out.hidden_states[-1][:, -1:, :]
            if last_h.dim() == 4:
                last_h = last_h.squeeze(1)
            
            if verbose:
                print(f"[{len(gen)-1}] ✓✓ both accepted: "
                      f"'{tokenizer.decode([prop1.item()])}' '{tokenizer.decode([prop2.item()])}'")
            
            if prop2.item() == eos:
                break
                
        elif ok1:
            # Only first token accepted
            gen.append(prop1.item())
            stats['first_only'] += 1
            stats['total_tokens'] += 1
            # ⭐ Use past_before_verify and add just the accepted token
            out_single = student.backbone(input_ids=prop1.unsqueeze(0).unsqueeze(0),
                                         past_key_values=past_before_verify,
                                         use_cache=True,
                                         output_hidden_states=True,
                                         return_dict=True)
            past = out_single.past_key_values
            last_h = out_single.hidden_states[-1][:, -1:, :]
            if last_h.dim() == 4:
                last_h = last_h.squeeze(1)
            
            if verbose:
                print(f"[{len(gen)}] ✓✗ first accepted: "
                      f"'{tokenizer.decode([prop1.item()])}', rejected '{tokenizer.decode([prop2.item()])}'")
            
            if prop1.item() == eos:
                break
                
        else:
            # Both rejected, use backbone's first prediction
            tok = backbone_tok1
            gen.append(tok.item())
            stats['both_rejected'] += 1
            stats['total_tokens'] += 1
            # ⭐ Use past_before_verify and add the corrected token
            out_single = student.backbone(input_ids=tok.unsqueeze(0).unsqueeze(0),
                                         past_key_values=past_before_verify,
                                         use_cache=True,
                                         output_hidden_states=True,
                                         return_dict=True)
            past = out_single.past_key_values
            last_h = out_single.hidden_states[-1][:, -1:, :]
            if last_h.dim() == 4:
                last_h = last_h.squeeze(1)
            
            if verbose:
                print(f"[{len(gen)}] ✗✗ both rejected: "
                      f"used backbone '{tokenizer.decode([tok.item()])}' "
                      f"(drafts: '{tokenizer.decode([prop1.item()])}', '{tokenizer.decode([prop2.item()])}')")
            
            if tok.item() == eos:
                break
        
        if len(gen) >= max_new_tokens:
            break
    
    out_ids = torch.cat([input_ids[0], torch.tensor(gen, device=device)], dim=0)
    decoded = tokenizer.decode(out_ids, skip_special_tokens=True)
    
    return decoded, stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--draft_top_k", type=int, default=0)
    ap.add_argument("--draft_temperature", type=float, default=1.0)
    ap.add_argument("--verify_top_k", type=int, default=1)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    
    s, tok = load_student(args.ckpt, args.model_name)
    
    print(f"\n[CONFIG] max_new_tokens={args.max_new_tokens}, draft_top_k={args.draft_top_k}, "
          f"draft_temp={args.draft_temperature}, verify_top_k={args.verify_top_k}\n")
    
    try:
        while True:
            q = input(">>> ").strip()
            if not q or q.lower() in {"exit","quit",":q"}:
                break
            txt, stats = parallel_decode(s, tok, q, 
                                max_new_tokens=args.max_new_tokens,
                                draft_top_k=args.draft_top_k, 
                                draft_temperature=args.draft_temperature, 
                                verify_top_k=args.verify_top_k,
                                verbose=args.verbose)
            print(txt)
            
            # Print statistics
            parallel_tokens = stats['both_accepted'] * 2
            total_tokens = stats['total_tokens']
            parallel_rate = (parallel_tokens / total_tokens * 100) if total_tokens > 0 else 0
            
            print(f"\n📊 Stats: {total_tokens} tokens | "
                  f"✓✓ {stats['both_accepted']} | "
                  f"✓✗ {stats['first_only']} | "
                  f"✗✗ {stats['both_rejected']} | "
                  f"Parallel: {parallel_tokens}/{total_tokens} ({parallel_rate:.1f}%)\n")
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()@torch.no_grad()
