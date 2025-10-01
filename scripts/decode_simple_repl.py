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

    # build student skeleton
    s = GPT2MTPStudent(model_name=base_name, offsets=offsets, device=device, lora=False)
    print(f"[INFO] Backbone skeleton built: hidden_size={s.config.hidden_size}, vocab={s.config.vocab_size}")

    # load lora adapters
    base = AutoModelForCausalLM.from_pretrained(base_name).to(device)
    s.backbone = PeftModel.from_pretrained(base, ckpt_dir)
    lora_params = [n for n,p in s.backbone.named_parameters() if "lora" in n]
    print(f"[INFO] LoRA params loaded: {len(lora_params)} parameters with 'lora' in name")

    # load MTP head
    mtp_path = os.path.join(ckpt_dir, "mtp_head.pt")
    state = torch.load(mtp_path, map_location=device)
    s.mtp_head.load_state_dict(state, strict=True)
    print(f"[INFO] Loaded MTP head weights from {mtp_path}")
    # quick sanity check on weight norms
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
        t = i.gather(-1, j).squeeze(-1)
        return t
    return torch.argmax(logits, dim=-1)

@torch.no_grad()
def decode_once(student, tokenizer, prompt, steps=4, draft_top_k=0, draft_temperature=1.0, verify_top_k=3):
    device = next(student.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask", None)
    out = student.backbone(input_ids=input_ids, attention_mask=attn, use_cache=True, output_hidden_states=True, return_dict=True)
    past = out.past_key_values
    last_h = out.hidden_states[-1][:, -1:, :]
    eos = tokenizer.eos_token_id
    gen = []
    for _ in range(steps):
        mtp = student.mtp_head(last_h, last_only=True)
        log1 = mtp[1][:, -1, :]
        prop1 = sample_topk(log1, k=draft_top_k, temperature=draft_temperature)
        out1 = student.backbone(input_ids=prop1.unsqueeze(0), past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
        nxt1 = out1.logits[:, -1, :]
        if verify_top_k == 1:
            ok1 = (torch.argmax(nxt1, dim=-1).item() == prop1.item())
        else:
            _, idx = torch.topk(nxt1, k=verify_top_k, dim=-1)
            ok1 = prop1.item() in idx[0].tolist()
        tok1 = prop1 if ok1 else torch.argmax(nxt1, dim=-1)
        gen.append(tok1.item())
        past = out1.past_key_values if ok1 else student.backbone(input_ids=tok1.unsqueeze(0), past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True).past_key_values
        last_h = out1.hidden_states[-1][:, -1:, :] if ok1 else student.backbone(input_ids=tok1.unsqueeze(0), past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1:, :]
        if tok1.item() == eos:
            break
        log2 = mtp[2][:, -1, :]
        prop2 = sample_topk(log2, k=draft_top_k, temperature=draft_temperature)
        out2 = student.backbone(input_ids=prop2.unsqueeze(0), past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True)
        nxt2 = out2.logits[:, -1, :]
        if verify_top_k == 1:
            ok2 = (torch.argmax(nxt2, dim=-1).item() == prop2.item())
        else:
            _, idx2 = torch.topk(nxt2, k=verify_top_k, dim=-1)
            ok2 = prop2.item() in idx2[0].tolist()
        tok2 = prop2 if ok2 else torch.argmax(nxt2, dim=-1)
        gen.append(tok2.item())
        past = out2.past_key_values if ok2 else student.backbone(input_ids=tok2.unsqueeze(0), past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True).past_key_values
        last_h = out2.hidden_states[-1][:, -1:, :] if ok2 else student.backbone(input_ids=tok2.unsqueeze(0), past_key_values=past, use_cache=True, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1:, :]
        if tok2.item() == eos:
            break
    out_ids = torch.cat([input_ids[0], torch.tensor(gen, device=device)], dim=0)
    return tokenizer.decode(out_ids, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="gpt2")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--draft_top_k", type=int, default=0)
    ap.add_argument("--draft_temperature", type=float, default=1.0)
    ap.add_argument("--verify_top_k", type=int, default=3)
    args = ap.parse_args()
    s, tok = load_student(args.ckpt, args.model_name)
    try:
        while True:
            q = input(">>> ").strip()
            if not q or q.lower() in {"exit","quit",":q"}:
                break
            txt = decode_once(s, tok, q, steps=args.steps, draft_top_k=args.draft_top_k, draft_temperature=args.draft_temperature, verify_top_k=args.verify_top_k)
            print(txt)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
