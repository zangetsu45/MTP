import os, json, torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

class MultiTokenHead(nn.Module):
    def __init__(self, d_model, vocab_size, offsets=(1, 2)):
        super().__init__()
        self.offsets = offsets
        self.heads = nn.ModuleDict()
        for off in offsets:
            self.heads[str(off)] = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, vocab_size)
            )
    def forward(self, hidden_states, last_only=False):
        hs = hidden_states[:, -1:, :] if last_only else hidden_states
        B, S, D = hs.shape
        x = hs.reshape(B * S, D)
        out = {}
        for off in self.offsets:
            o = self.heads[str(off)](x).reshape(B, S, -1)
            out[off] = o
        return out

class GPT2MTPStudent(nn.Module):
    def __init__(self, model_name="gpt2", offsets=(1, 2), device=None, lora=True, lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_targets=("c_attn","c_proj")):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)
        if lora:
            cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none", target_modules=list(lora_targets), task_type=TaskType.CAUSAL_LM)
            self.backbone = get_peft_model(self.backbone, cfg)
        self.config = self.backbone.config
        d_model = self.config.hidden_size
        vocab_size = self.config.vocab_size
        self.mtp_head = MultiTokenHead(d_model, vocab_size, offsets)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.to(self.device)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone.transformer(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden_states = outputs.last_hidden_state
        lm_logits = self.backbone.lm_head(hidden_states)
        mtp_logits_full = self.mtp_head(hidden_states, last_only=False)
        mtp_logits_last = {k: v[:, -1, :] for k, v in mtp_logits_full.items()}
        return {"lm_logits": lm_logits, "mtp_logits_full": mtp_logits_full, "mtp_logits": mtp_logits_last}

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(save_directory)
        else:
            torch.save(self.backbone.state_dict(), os.path.join(save_directory, "backbone.bin"))
        torch.save(self.mtp_head.state_dict(), os.path.join(save_directory, "mtp_head.pt"))
        with open(os.path.join(save_directory, "mtp_meta.json"), "w") as f:
            json.dump({"model_name": self.model_name, "offsets": list(self.mtp_head.offsets)}, f)
