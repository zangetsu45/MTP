import torch
import torch.nn as nn
import torch.nn.functional as F

def ce_loss(lm_logits, input_ids, pad_token_id):
    shift_logits = lm_logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return nn.CrossEntropyLoss(ignore_index=pad_token_id)(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

def kd_loss(student_logits, teacher_logits, attention_mask=None, temperature=1.0):
    s = student_logits[:, :-1, :].contiguous()
    t = teacher_logits[:, :-1, :].contiguous()
    s = F.log_softmax(s / temperature, dim=-1)
    t = F.softmax(t / temperature, dim=-1)
    kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
    if attention_mask is not None:
        m = attention_mask[:, 1:].contiguous().float()
        kl = (kl * m).sum() / (m.sum().clamp_min(1.0))
    else:
        kl = kl.mean()
    return (temperature ** 2) * kl

def mtp_ce_loss(mtp_logits_full, input_ids, pad_token_id, offsets, attention_mask=None, gamma=0.5):
    losses = []
    for off in offsets:
        logits = mtp_logits_full[off][:, :-off, :].contiguous()
        labels = input_ids[:, off:, :].contiguous() if input_ids.dim() == 3 else input_ids[:, off:].contiguous()
        if attention_mask is not None:
            mask = attention_mask[:, off:].contiguous()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=pad_token_id, reduction="none")
            loss = (loss.view(labels.shape) * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=pad_token_id)
        weight = gamma ** (off - 1)
        losses.append(weight * loss)
    return sum(losses) / max(1, len(losses))

def mtp_kd_loss(mtp_logits_full, teacher_logits, attention_mask, offsets, temperature=1.0, gamma=0.5):
    losses = []
    for off in offsets:
        s = mtp_logits_full[off][:, :-off, :].contiguous()
        t = teacher_logits[:, off-1:-1, :].contiguous()
        s = F.log_softmax(s / temperature, dim=-1)
        t = F.softmax(t / temperature, dim=-1)
        kl = F.kl_div(s, t, reduction="none").sum(dim=-1)
        if attention_mask is not None:
            m = attention_mask[:, off:].contiguous().float()
            kl = (kl * m).sum() / m.sum().clamp_min(1.0)
        else:
            kl = kl.mean()
        weight = gamma ** (off - 1)
        losses.append((temperature ** 2) * weight * kl)
    return sum(losses) / max(1, len(losses))

def total_loss(out_student, input_ids, pad_token_id, offsets, teacher_logits=None, attention_mask=None, alpha=0.5, beta_mtp=1.0, beta_mtp_kd=0.5, temperature=1.0):
    ce = ce_loss(out_student["lm_logits"], input_ids, pad_token_id)
    if teacher_logits is None:
        kd = 0.0
        mtp_kd = 0.0
    else:
        kd = kd_loss(out_student["lm_logits"], teacher_logits, attention_mask=attention_mask, temperature=temperature)
        mtp_kd = mtp_kd_loss(out_student["mtp_logits_full"], teacher_logits, attention_mask, offsets, temperature=temperature)
    mtp = mtp_ce_loss(out_student["mtp_logits_full"], input_ids, pad_token_id, offsets, attention_mask=attention_mask)
    return alpha * ce + (1 - alpha) * kd + beta_mtp * mtp + beta_mtp_kd * mtp_kd
