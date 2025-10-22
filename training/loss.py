import torch
import torch.nn as nn
import torch.nn.functional as F

def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None):
    if mask is None:
        return x.mean()
    m = mask.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def ce_loss(lm_logits: torch.Tensor, input_ids: torch.Tensor, pad_token_id: int):
    """
    Standard next-token CE (still useful for eval or non-distill runs).
    """
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    )

# -------------------------------
# KL loss for self-distillation (backbone)
# -------------------------------

def kl_next_token_loss(student_logits: torch.Tensor,
                       teacher_logits: torch.Tensor,
                       attention_mask: torch.Tensor | None = None,
                       temperature: float = 1.0) -> torch.Tensor:
    """
    KL( p_teacher || p_student ) on next-token distributions (aligned at t+1).
      student_logits, teacher_logits: [B, S, V]
      returns scalar mean loss
    """
    s = student_logits[:, :-1, :].contiguous()   # [B, S-1, V]
    t = teacher_logits[:, :-1, :].contiguous()   # [B, S-1, V]

    s_logp = F.log_softmax(s / temperature, dim=-1)
    t_prob = F.softmax(t / temperature, dim=-1)

    # tokenwise KL; sum over vocab -> [B, S-1]
    kl = F.kl_div(s_logp, t_prob, reduction="none").sum(dim=-1)

    if attention_mask is not None:
        # align with next-token positions
        m = attention_mask[:, 1:].contiguous().float()  # [B, S-1]
        kl = _masked_mean(kl, m)
    else:
        kl = kl.mean()

    # standard T^2 scaling for distillation
    return (temperature ** 2) * kl

# -------------------------------
# MEDUSA heads
# -------------------------------

def medusa_heads_loss(mtp_logits_full: dict,
                      input_ids: torch.Tensor,
                      pad_token_id: int,
                      offsets: tuple[int, ...],
                      attention_mask: torch.Tensor | None = None,
                      lambda_base: float = 0.8,
                      label_smoothing: float = 0.0):
    """
    MEDUSA-1 heads loss:
      L = Σ_k λ_k * CE_k,  with λ_k = (lambda_base)^k
    """
    total = 0.0
    for k in offsets:
        logits = mtp_logits_full[k][:, :-k, :].contiguous()
        labels = input_ids[:, k:].contiguous()

        if attention_mask is not None:
            mask = attention_mask[:, k:].float()
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_token_id,
                reduction="none",
                label_smoothing=label_smoothing
            )
            ce = ce.view(labels.shape)
            head_loss = _masked_mean(ce, mask)
        else:
            head_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=pad_token_id,
                reduction="mean",
                label_smoothing=label_smoothing
            )

        total += (lambda_base ** k) * head_loss

    return total

# -------------------------------
# Public APIs
# -------------------------------

def medusa1_loss(out_student: dict,
                 input_ids: torch.Tensor,
                 pad_token_id: int,
                 offsets: tuple[int, ...],
                 attention_mask: torch.Tensor | None = None,
                 lambda_base: float = 0.8,
                 mtp_label_smoothing: float = 0.0):
    """MEDUSA-1: Frozen backbone (heads only)."""
    return medusa_heads_loss(
        out_student["mtp_logits_full"], input_ids, pad_token_id,
        offsets, attention_mask=attention_mask,
        lambda_base=lambda_base, label_smoothing=mtp_label_smoothing
    )


def medusa2_loss(out_student: dict,
                 teacher_logits: torch.Tensor,                  # <-- NEW: teacher required
                 input_ids: torch.Tensor,
                 pad_token_id: int,
                 offsets: tuple[int, ...],
                 attention_mask: torch.Tensor | None = None,
                 lambda0: float = 1.0,
                 lambda_base: float = 0.8,
                 temperature: float = 1.0,                      # <-- NEW: distill temperature
                 mtp_label_smoothing: float = 0.0):
    """
    MEDUSA-2 with self-distillation (Section 2.3.2):
      L = KL(p_teacher || p_student) + λ0 * L_MEDUSA-1
    Backbone uses KL to match teacher next-token distribution.
    Heads use weighted CE like MEDUSA-1.
    """
    LLM_distill = kl_next_token_loss(
        out_student["lm_logits"], teacher_logits,
        attention_mask=attention_mask, temperature=temperature
    )

    L_heads = medusa_heads_loss(
        out_student["mtp_logits_full"], input_ids, pad_token_id,
        offsets, attention_mask=attention_mask,
        lambda_base=lambda_base, label_smoothing=mtp_label_smoothing
    )
    return LLM_distill + (lambda0 * L_heads)
