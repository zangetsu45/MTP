import os, json, torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------
# MEDUSA head: SiLU -> +residual -> vocab
#   p_t^(k) = softmax(W2^(k) * ( SiLU(W1^(k) * h_t) + h_t ))
#   with init: W1^(k) = 0, W2^(k) = LM head
# -----------------------
class _MedusaSingleHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, lm_head: nn.Module | None):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_model, bias=True)
        # mirror LM head bias presence if available
        use_bias_w2 = bool(getattr(lm_head, "bias", None) is not None) if lm_head is not None else False
        self.w2 = nn.Linear(d_model, vocab_size, bias=use_bias_w2)
        self._init_from_lm_head(lm_head)

    def _init_from_lm_head(self, lm_head: nn.Module | None):
        # W1 = 0 so SiLU(W1*h) == 0 -> output starts identical to LM head
        with torch.no_grad():
            self.w1.weight.zero_()
            if self.w1.bias is not None:
                self.w1.bias.zero_()

            if lm_head is not None and hasattr(lm_head, "weight"):
                if self.w2.weight.shape == lm_head.weight.shape:
                    self.w2.weight.copy_(lm_head.weight.detach())
                else:
                    # fallback if shapes differ (shouldn't for standard CausalLM)
                    nn.init.xavier_uniform_(self.w2.weight)
                # bias
                if self.w2.bias is not None:
                    if getattr(lm_head, "bias", None) is not None and self.w2.bias.shape == lm_head.bias.shape:
                        self.w2.bias.copy_(lm_head.bias.detach())
                    else:
                        self.w2.bias.zero_()
            else:
                nn.init.xavier_uniform_(self.w2.weight)
                if self.w2.bias is not None:
                    self.w2.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B*S, D]
        z = F.silu(self.w1(h)) + h      # residual before vocab proj
        return self.w2(z)               # [B*S, V]


class MultiTokenHead(nn.Module):
    def __init__(self, d_model, vocab_size, offsets=(1, 2), lm_head: nn.Module | None = None):
        super().__init__()
        self.offsets = tuple(int(o) for o in offsets)
        self.heads = nn.ModuleDict()
        for off in self.offsets:
            self.heads[str(off)] = _MedusaSingleHead(d_model, vocab_size, lm_head)

    def forward(self, hidden_states: torch.Tensor, last_only: bool = False):
        """
        hidden_states: [B, S, D] (last layer of backbone)
        returns dict {k: [B, S_or_1, V]}
        """
        hs = hidden_states[:, -1:, :] if last_only else hidden_states
        B, S, D = hs.shape
        x = hs.reshape(B * S, D)
        out = {}
        for off in self.offsets:
            logits = self.heads[str(off)](x).reshape(B, S, -1)
            out[off] = logits
        return out


class GPT2MTPStudent(nn.Module):
    def __init__(
        self,
        model_name="gpt2",
        offsets=(1, 2),
        device=None,
        lora=True,
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        lora_targets=("c_attn","c_proj"),
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.backbone = AutoModelForCausalLM.from_pretrained(model_name)

        if lora:
            cfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=list(lora_targets),
                task_type=TaskType.CAUSAL_LM,
            )
            self.backbone = get_peft_model(self.backbone, cfg)

        self.config = self.backbone.config
        d_model = self.config.hidden_size
        vocab_size = self.config.vocab_size

        # IMPORTANT: pass the LM head so MEDUSA heads can copy W2 from it
        self.mtp_head = MultiTokenHead(d_model, vocab_size, offsets, lm_head=self.backbone.lm_head)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.to(self.device)

    def forward(self, input_ids, attention_mask=None):
        # Ask the CausalLM model for hidden states
        outputs = self.backbone.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state            # [B, S, D]
        lm_logits = self.backbone.lm_head(hidden_states)     # [B, S, V]

        # MEDUSA heads (residual + SiLU) over the same hidden states
        mtp_logits_full = self.mtp_head(hidden_states, last_only=False)
        mtp_logits_last = {k: v[:, -1, :] for k, v in mtp_logits_full.items()}
        return {
            "lm_logits": lm_logits,
            "mtp_logits_full": mtp_logits_full,
            "mtp_logits": mtp_logits_last
        }

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self.backbone, "save_pretrained"):
            self.backbone.save_pretrained(save_directory)
        else:
            torch.save(self.backbone.state_dict(), os.path.join(save_directory, "backbone.bin"))
        torch.save(self.mtp_head.state_dict(), os.path.join(save_directory, "mtp_head.pt"))
        with open(os.path.join(save_directory, "mtp_meta.json"), "w") as f:
            json.dump(
                {"model_name": self.model_name, "offsets": list(self.mtp_head.offsets)},
                f
            )
