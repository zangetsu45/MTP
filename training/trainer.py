import os, math, torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from training.loss import total_loss
from tqdm import tqdm

class KDTrainer:
    def __init__(
        self,
        student,
        teacher,
        train_loader,
        val_loader,
        pad_token_id,
        lr=5e-5,
        weight_decay=0.01,
        epochs=1,
        alpha=0.5,
        temperature=1.0,
        beta_mtp=1.0,
        beta_mtp_kd=0.5,
        grad_accum=1,
        max_grad_norm=1.0,
        use_amp=True,
        save_dir="experiments/logs/run1",
        tokenizer=None,                 # optional: saved if provided
        ckpt_root="checkpoints"         # base folder for all checkpoints
    ):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_id = pad_token_id
        self.device = next(student.parameters()).device
        self.epochs = epochs
        self.alpha = alpha
        self.temperature = temperature
        self.beta_mtp = beta_mtp
        self.beta_mtp_kd = beta_mtp_kd
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.use_amp = use_amp

        # Logging dir (unchanged)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Checkpoint root: checkpoints/<run_name>/
        run_name = os.path.basename(os.path.normpath(save_dir)) or "run"
        self.ckpt_dir = os.path.join(ckpt_root, run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tokenizer = tokenizer

        self.optimizer = AdamW(self.student.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = (len(self.train_loader) * epochs) // max(1, grad_accum)
        warmup = max(0, total_steps // 20)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps
        )
        self.scaler = GradScaler(enabled=use_amp)

    def _save_checkpoint(self, epoch, val_loss):
        """
        Saves:
          - model weights (prefer HF save_pretrained if available, else state_dict)
          - tokenizer (if provided)
          - optimizer / scheduler / scaler states for resume
          - a small metadata file (epoch, val_loss)
        """
        epoch_dir = os.path.join(self.ckpt_dir, f"epoch_{epoch:03d}")
        os.makedirs(epoch_dir, exist_ok=True)

        # Model
        saved_with_hf = False
        if hasattr(self.student, "save_pretrained") and callable(getattr(self.student, "save_pretrained")):
            try:
                self.student.save_pretrained(epoch_dir)
                saved_with_hf = True
            except Exception:
                saved_with_hf = False
        if not saved_with_hf:
            torch.save(self.student.state_dict(), os.path.join(epoch_dir, "pytorch_model.bin"))

        # Tokenizer (optional)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(epoch_dir)
            except Exception:
                pass

        # Trainer state for resume
        torch.save(
            {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
            },
            os.path.join(epoch_dir, "training_state.pt")
        )
        tqdm.write(f"[checkpoint] Saved epoch {epoch} to {epoch_dir}")

    def _save_best(self, epoch, val_loss):
        """
        Save/update the 'best' snapshot in checkpoints/<run_name>/best/
        """
        best_dir = os.path.join(self.ckpt_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Model
        saved_with_hf = False
        if hasattr(self.student, "save_pretrained") and callable(getattr(self.student, "save_pretrained")):
            try:
                self.student.save_pretrained(best_dir)
                saved_with_hf = True
            except Exception:
                saved_with_hf = False
        if not saved_with_hf:
            torch.save(self.student.state_dict(), os.path.join(best_dir, "pytorch_model.bin"))

        # Tokenizer (optional)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(best_dir)
            except Exception:
                pass

        # Trainer state
        torch.save(
            {
                "epoch": epoch,
                "val_loss": float(val_loss),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
            },
            os.path.join(best_dir, "training_state.pt")
        )
        tqdm.write(f"[checkpoint] New best (val_loss={val_loss:.4f}) saved to {best_dir}")

    def train(self):
        best_val = float("inf")
        global_step = 0
        for epoch in range(1, self.epochs + 1):
            self.student.train()
            self.optimizer.zero_grad(set_to_none=True)
            running = 0.0
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=True)
            for step, batch in enumerate(loop, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                with autocast(enabled=self.use_amp):
                    out = self.student(input_ids=input_ids, attention_mask=attention_mask)
                    teacher_logits = None
                    if self.teacher is not None:
                        with torch.no_grad():
                            teacher_logits = self.teacher.get_logits(input_ids, attention_mask)

                    loss = total_loss(
                        out_student=out,
                        input_ids=input_ids,
                        pad_token_id=self.pad_id,
                        offsets=tuple(self.student.mtp_head.offsets),
                        teacher_logits=teacher_logits,
                        attention_mask=attention_mask,
                        alpha=self.alpha,
                        beta_mtp=self.beta_mtp,
                        beta_mtp_kd=self.beta_mtp_kd,
                        temperature=self.temperature,
                    ) / self.grad_accum

                self.scaler.scale(loss).backward()

                if step % self.grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running += loss.item() * self.grad_accum
                lr = self.scheduler.get_last_lr()[0]
                loop.set_postfix({"loss": f"{running/step:.4f}", "lr": f"{lr:.2e}"})

            # Validation & checkpointing
            val_loss, val_ppl = self.evaluate()
            tqdm.write(f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}")

            # Save this epoch
            self._save_checkpoint(epoch, val_loss)

            # Save best
            if val_loss < best_val:
                best_val = val_loss
                self._save_best(epoch, val_loss)

    @torch.no_grad()
    def evaluate(self):
        self.student.eval()
        total, count = 0.0, 0
        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            out = self.student(input_ids=input_ids, attention_mask=attention_mask)
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
