import os, math, torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from training.loss import medusa1_loss, medusa2_loss
from tqdm import tqdm

class MedusaTrainer:
    """
    MEDUSA trainer with:
      - Stage 1 (MEDUSA-1): heads-only, backbone frozen
      - Stage 2 (MEDUSA-2 self-distill): joint training with KL(backbone) + λ0 * CE(heads)

    Requirements:
      - model has .backbone and .mtp_head (with .offsets)
      - teacher provides next-token logits, kept frozen (eval mode)
    """
    def __init__(
        self,
        model,                      # student model (with .backbone and .mtp_head)
        train_loader,
        val_loader,
        pad_token_id,
        *,
        teacher=None,               # frozen teacher model (required for MEDUSA-2 self-distill)
        temperature=1.0,            # distillation temperature
        epochs=5,
        lr_backbone=5e-6,           # smaller LR for backbone
        lr_heads=5e-4,              # larger LR for MEDUSA heads
        weight_decay=0.01,
        grad_accum=1,
        max_grad_norm=1.0,
        use_amp=True,
        lambda0=1.0,                # weight for heads loss in MEDUSA-2
        lambda_base=0.8,            # λ^k decay for heads
        warmup_epochs=1,            # heads-only (MEDUSA-1) training epochs
        lambda0_warmup=False,       # if True: gradually increase λ0 after warmup
        save_dir="experiments/logs/run_medusa",
        ckpt_root="checkpoints",
        tokenizer=None
    ):
        self.model = model
        self.teacher = teacher
        self.temperature = temperature

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_id = pad_token_id
        self.device = next(model.parameters()).device

        # Ensure teacher is frozen & on the right device
        if self.teacher is not None:
            try:
                self.teacher.to(self.device)
            except Exception:
                pass
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)

        self.epochs = epochs
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp
        self.lambda0 = lambda0
        self.lambda_base = lambda_base
        self.warmup_epochs = warmup_epochs
        self.lambda0_warmup = lambda0_warmup

        # Differential LR param groups (keep names so we can toggle LR later)
        self.lr_backbone = lr_backbone
        self.lr_heads = lr_heads
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.mtp_head.parameters())
        self.optimizer = AdamW([
            {"params": backbone_params, "lr": lr_backbone, "name": "backbone"},
            {"params": head_params, "lr": lr_heads, "name": "heads"},
        ], weight_decay=weight_decay)

        total_steps = (len(train_loader) * epochs) // max(1, grad_accum)
        warmup = max(0, total_steps // 20)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps
        )
        self.scaler = GradScaler(enabled=use_amp)

        # Dirs
        os.makedirs(save_dir, exist_ok=True)
        run_name = os.path.basename(os.path.normpath(save_dir)) or "run"
        self.ckpt_dir = os.path.join(ckpt_root, run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    # ---------- freeze / unfreeze ----------
    def _freeze_backbone(self):
        for p in self.model.backbone.parameters():
            p.requires_grad_(False)

    def _unfreeze_backbone(self):
        for p in self.model.backbone.parameters():
            p.requires_grad_(True)

    def _set_backbone_lr(self, lr: float):
        """Optionally zero backbone LR in warmup for extra safety."""
        for g in self.optimizer.param_groups:
            if g.get("name") == "backbone":
                g["lr"] = lr

    # ---------- checkpoint helpers ----------
    def _save(self, epoch, val_loss, best=False):
        tag = "best" if best else f"epoch_{epoch:03d}"
        out_dir = os.path.join(self.ckpt_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        # Model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(out_dir)
        else:
            torch.save(self.model.state_dict(), os.path.join(out_dir, "pytorch_model.bin"))

        # Tokenizer
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(out_dir)
            except Exception:
                pass

        # Trainer state
        torch.save({
            "epoch": epoch,
            "val_loss": float(val_loss),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
        }, os.path.join(out_dir, "training_state.pt"))
        tqdm.write(f"[checkpoint] {'Best' if best else f'Epoch {epoch}'} saved to {out_dir}")

    # ---------- training loop ----------
    def train(self):
        best_val = float("inf")
        global_step = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            running = 0.0

            # λ₀ warmup schedule
            if self.lambda0_warmup and epoch > self.warmup_epochs:
                frac = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
                current_lambda0 = self.lambda0 * min(1.0, frac)
            else:
                current_lambda0 = self.lambda0 if epoch > self.warmup_epochs else 0.0

            # Stage selection
            if epoch <= self.warmup_epochs:
                stage = "MEDUSA-1 (heads only)"
                self._freeze_backbone()
                self._set_backbone_lr(0.0)            # optional safety
                use_medusa2 = False
            else:
                stage = f"MEDUSA-2 (joint/self-distill, λ0={current_lambda0:.2f})"
                if self.teacher is None:
                    raise RuntimeError("MEDUSA-2 self-distillation requires a teacher model (teacher is None).")
                self._unfreeze_backbone()
                self._set_backbone_lr(self.lr_backbone)
                use_medusa2 = True

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch} | {stage}", leave=True)
            for step, batch in enumerate(loop, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                with autocast(enabled=self.use_amp):
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # Compute loss
                if not use_medusa2:
                    # MEDUSA-1: heads loss only (can be under AMP)
                    with autocast(enabled=self.use_amp):
                        loss = medusa1_loss(
                            out_student=out,
                            input_ids=input_ids,
                            pad_token_id=self.pad_id,
                            offsets=tuple(self.model.mtp_head.offsets),
                            attention_mask=attention_mask,
                            lambda_base=self.lambda_base,
                        ) / self.grad_accum
                else:
                    # MEDUSA-2: self-distillation (KL on backbone in fp32 + λ0*heads CE)
                    with torch.no_grad():
                        if hasattr(self.teacher, "get_logits") and callable(getattr(self.teacher, "get_logits")):
                            teacher_logits = self.teacher.get_logits(input_ids, attention_mask)
                        else:
                            teacher_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
                            teacher_logits = getattr(teacher_out, "logits", teacher_out["logits"])

                    # Compute total loss in fp32 for numerical stability of KL
                    with autocast(enabled=False):
                        out_fp32 = dict(out)
                        # Ensure KL uses fp32 logits
                        out_fp32["lm_logits"] = out["lm_logits"].float()
                        loss = medusa2_loss(
                            out_student=out_fp32,
                            teacher_logits=teacher_logits.float(),
                            input_ids=input_ids,
                            pad_token_id=self.pad_id,
                            offsets=tuple(self.model.mtp_head.offsets),
                            attention_mask=attention_mask,
                            lambda0=current_lambda0,
                            lambda_base=self.lambda_base,
                            temperature=self.temperature,
                        ) / self.grad_accum

                # Backward + step
                self.scaler.scale(loss).backward()

                if step % self.grad_accum == 0:
                    # Unscale before clipping for correct grad-norm behavior under AMP
                    self.scaler.unscale_(self.optimizer)
                    # Clip only trainable params
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running += loss.item() * self.grad_accum
                lr_backbone_now = next((g["lr"] for g in self.optimizer.param_groups if g.get("name") == "backbone"), 0.0)
                loop.set_postfix({"loss": f"{running/step:.4f}", "lr_bb": f"{lr_backbone_now:.2e}"})

            # Validation
            val_loss, val_ppl = self.evaluate()
            tqdm.write(f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")

            self._save(epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                self._save(epoch, val_loss, best=True)

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
            shift_logits = out["lm_logits"][:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_id,
                reduction="mean"
            )
            total += loss.item()
            count += 1
        avg = total / max(1, count)
        ppl = math.exp(min(20, avg))
        return avg, ppl
