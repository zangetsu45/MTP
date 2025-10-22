# training/trainer.py (Updated for LoRA Self-Distillation Trick)

import os, math, torch
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from training.loss import medusa1_loss, medusa2_loss
from tqdm import tqdm
# Ensure peft is installed: pip install peft
from peft import PeftModel # Used for type hinting and checks

class MedusaTrainer:
    """
    MEDUSA trainer using LoRA self-distillation trick (no separate teacher model).
      - Stage 1 (MEDUSA-1): heads-only, backbone LoRA adapters frozen
      - Stage 2 (MEDUSA-2 self-distill): joint training with KL(backbone_no_adapter || backbone_with_adapter) + λ0 * CE(heads)

    Requirements:
      - model.backbone must be a PEFT model (e.g., QLoRA Llama)
      - model has .mtp_head (with .offsets)
    """
    def __init__(
        self,
        model,                      # student PEFT model (e.g., QLoRA Llama + Medusa heads)
        train_loader,
        val_loader,
        pad_token_id,
        *,
        # teacher=None,             # <-- REMOVED
        temperature=1.0,            # distillation temperature
        epochs=5,
        lr_backbone=5e-6,           # LR for LoRA adapters
        lr_heads=5e-4,              # LR for MEDUSA heads
        weight_decay=0.01,
        grad_accum=1,
        max_grad_norm=1.0,
        use_amp=True,               # Should be True for QLoRA
        lambda0=1.0,                # weight for heads loss in MEDUSA-2
        lambda_base=0.8,            # λ^k decay for heads
        warmup_epochs=1,            # heads-only (MEDUSA-1) training epochs
        lambda0_warmup=False,       # if True: gradually increase λ0 after warmup
        save_dir="experiments/logs/run_medusa",
        ckpt_root="checkpoints",
        tokenizer=None
    ):
        self.model = model
        # self.teacher = teacher    # <-- REMOVED
        self.temperature = temperature

        # --- Sanity Check: Ensure backbone is a PEFT model ---
        if not isinstance(model.backbone, PeftModel):
             raise TypeError("model.backbone must be a PEFT model (e.g., from get_peft_model) "
                             "to use the LoRA self-distillation trick.")
        # -----------------------------------------------------

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.pad_id = pad_token_id
        # Get device from model parameters (works with device_map="auto")
        try:
             self.device = next(model.parameters()).device
        except StopIteration:
             print("[warn] Could not determine model device automatically.")
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.epochs = epochs
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        # Force AMP if QLoRA is likely used (check backbone dtype)
        if any(p.dtype == torch.uint8 or p.dtype == torch.int8 for p in model.backbone.parameters()):
            print("[info] Detected quantized backbone. Forcing use_amp=True for QLoRA.")
            self.use_amp = True
        else:
            self.use_amp = use_amp

        self.lambda0 = lambda0
        self.lambda_base = lambda_base
        self.warmup_epochs = warmup_epochs
        self.lambda0_warmup = lambda0_warmup

        # Differential LR param groups
        self.lr_backbone = lr_backbone
        self.lr_heads = lr_heads

        # --- Optimizer Setup for PEFT ---
        # Get only parameters that require gradients
        backbone_trainable_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
        head_params = list(model.mtp_head.parameters()) # Medusa heads are always fully trained

        if not backbone_trainable_params:
             print("[warn] No trainable parameters found in model.backbone. LoRA adapters might not be set up correctly or are frozen.")
             # Create a dummy group to avoid optimizer errors if backbone is fully frozen
             param_groups = [{"params": head_params, "lr": lr_heads, "name": "heads"}]
        else:
            param_groups = [
                {"params": backbone_trainable_params, "lr": lr_backbone, "name": "backbone"},
                {"params": head_params, "lr": lr_heads, "name": "heads"},
            ]
        # --------------------------------

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)

        total_steps = (len(train_loader) * epochs) // max(1, grad_accum)
        warmup = max(0, total_steps // 20)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup,
            num_training_steps=total_steps
        )
        self.scaler = GradScaler(enabled=self.use_amp) # Use self.use_amp determined above

        # Dirs (Unchanged)
        os.makedirs(save_dir, exist_ok=True)
        run_name = os.path.basename(os.path.normpath(save_dir)) or "run"
        self.ckpt_dir = os.path.join(ckpt_root, run_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.save_dir = save_dir
        self.tokenizer = tokenizer

    # ---------- freeze / unfreeze LoRA Adapters ----------
    def _freeze_backbone(self):
        """Freezes LoRA adapters in the backbone."""
        for name, param in self.model.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = False

    def _unfreeze_backbone(self):
        """Unfreezes LoRA adapters in the backbone."""
        for name, param in self.model.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True

    def _set_backbone_lr(self, lr: float):
        """Sets the LR for the 'backbone' parameter group (LoRA adapters)."""
        for g in self.optimizer.param_groups:
            if g.get("name") == "backbone":
                g["lr"] = lr

    # ---------- checkpoint helpers ----------
    def _save(self, epoch, val_loss, best=False):
        tag = "best" if best else f"epoch_{epoch:03d}"
        out_dir = os.path.join(self.ckpt_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        # --- Saving PEFT Model + Custom Head ---
        print(f"Saving model components to {out_dir}...")
        # Save PEFT adapter state + config (saves only adapters, not base model)
        self.model.backbone.save_pretrained(out_dir)

        # Save the custom Medusa head separately
        torch.save(self.model.mtp_head.state_dict(), os.path.join(out_dir, "mtp_head.pt"))
        # --------------------------------------

        # Save Tokenizer (Unchanged)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(out_dir)
            except Exception as e:
                 print(f"[warn] Failed to save tokenizer: {e}")

        # Save Trainer state (Unchanged)
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
            # Ensure Medusa head is trainable even if backbone adapters are frozen
            self.model.mtp_head.train()
            self.optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0 # Use a more descriptive name

            # λ₀ warmup schedule (Unchanged)
            if self.lambda0_warmup and epoch > self.warmup_epochs:
                frac = (epoch - self.warmup_epochs) / max(1, self.epochs - self.warmup_epochs)
                current_lambda0 = self.lambda0 * min(1.0, frac)
            else:
                current_lambda0 = self.lambda0 if epoch > self.warmup_epochs else 0.0

            # Stage selection (Handles freezing/unfreezing LoRA adapters)
            if epoch <= self.warmup_epochs:
                stage = "MEDUSA-1 (heads only, LoRA frozen)"
                self._freeze_backbone() # Freezes LoRA adapters
                self._set_backbone_lr(0.0)
                use_medusa2 = False
            else:
                stage = f"MEDUSA-2 (LoRA self-distill, λ0={current_lambda0:.2f})"
                # No teacher check needed as we use the self-distill trick
                self._unfreeze_backbone() # Unfreezes LoRA adapters
                self._set_backbone_lr(self.lr_backbone)
                use_medusa2 = True

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch} | {stage}", leave=True)
            for step, batch in enumerate(loop, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # --- STUDENT FORWARD PASS (with adapters enabled) ---
                # This needs AMP enabled if using QLoRA
                with autocast(enabled=self.use_amp):
                    out_student = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # --- COMPUTE LOSS ---
                if not use_medusa2:
                    # MEDUSA-1: heads loss only
                    # Can compute under AMP
                    with autocast(enabled=self.use_amp):
                         loss = medusa1_loss(
                             out_student=out_student,
                             input_ids=input_ids,
                             pad_token_id=self.pad_id,
                             offsets=tuple(self.model.mtp_head.offsets),
                             attention_mask=attention_mask,
                             lambda_base=self.lambda_base,
                         ) / self.grad_accum
                else:
                    # MEDUSA-2: LoRA self-distillation
                    # --- TEACHER FORWARD PASS (same model, adapters disabled) ---
                    teacher_logits = None
                    with torch.no_grad():
                        # Use the disable_adapter context manager from PEFT
                        with self.model.backbone.disable_adapter():
                             # Forward pass using only the base model (quantized if QLoRA)
                             # Run teacher pass under autocast for efficiency, even if base is quantized
                             with autocast(enabled=self.use_amp):
                                  teacher_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                             # Teacher logits are only the standard LM head output
                             # Accessing .lm_logits directly might fail if model returns dict/tuple
                             teacher_logits = teacher_out.get("lm_logits") if isinstance(teacher_out, dict) else None
                             if teacher_logits is None:
                                  # Fallback for tuple or other outputs
                                  try:
                                      # Assuming standard HF CausalLMOutput format
                                      teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else getattr(teacher_out, 'logits', None)
                                  except (IndexError, AttributeError):
                                      pass # Keep teacher_logits as None if access fails

                             if teacher_logits is None:
                                  raise RuntimeError("Could not retrieve 'lm_logits' or equivalent "
                                                     "from teacher forward pass within disable_adapter context.")


                    # Compute total loss in fp32 for stability
                    with autocast(enabled=False):
                        # Make a copy to avoid modifying original student output dict
                        out_student_fp32 = dict(out_student)
                        out_student_fp32["lm_logits"] = out_student["lm_logits"].float()
                        
                        loss = medusa2_loss(
                            out_student=out_student_fp32,
                            teacher_logits=teacher_logits.float(), # Ensure teacher logits are also fp32 for KL
                            input_ids=input_ids,
                            pad_token_id=self.pad_id,
                            offsets=tuple(self.model.mtp_head.offsets),
                            attention_mask=attention_mask,
                            lambda0=current_lambda0,
                            lambda_base=self.lambda_base,
                            temperature=self.temperature,
                        ) / self.grad_accum

                # --- Backward + Step ---
                # Check for NaN/Inf loss before scaling
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[warn] Step {global_step}: Detected NaN/Inf loss ({loss.item()}). Skipping optimizer step.")
                    self.optimizer.zero_grad(set_to_none=True) # Clear potentially bad gradients
                    continue # Skip step

                self.scaler.scale(loss).backward()

                if step % self.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    # Clip grads of only trainable params (LoRA adapters + Medusa heads)
                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    if trainable_params:
                        # Use total_norm to check for exploding grads after unscaling
                        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                        if torch.isnan(total_norm) or torch.isinf(total_norm):
                             print(f"[warn] Step {global_step}: Detected NaN/Inf gradient norm ({total_norm}). Skipping optimizer step.")
                             self.optimizer.zero_grad(set_to_none=True) # Clear bad gradients
                             continue # Skip step
                    else:
                        print(f"[warn] Step {global_step}: No trainable parameters found for gradient clipping.")


                    self.scaler.step(self.optimizer)
                    self.scaler.update() # Update scaler only if step was successful
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                running_loss += loss.item() * self.grad_accum
                lr_backbone_now = next((g["lr"] for g in self.optimizer.param_groups if g.get("name") == "backbone"), 0.0)
                loop.set_postfix({"loss": f"{running_loss/step:.4f}", "lr_bb": f"{lr_backbone_now:.2e}"})

            # --- Validation ---
            val_loss, val_ppl = self.evaluate()
            tqdm.write(f"Epoch {epoch} Val Loss={val_loss:.4f} | Val PPL={val_ppl:.2f}")

            self._save(epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                tqdm.write(f"*** New best validation loss: {best_val:.4f} ***")
                self._save(epoch, val_loss, best=True)

    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        total_tokens = 0 # Use token count for more stable perplexity if needed

        eval_loop = tqdm(self.val_loader, desc="Evaluating", leave=False)
        for batch in eval_loop:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast(enabled=self.use_amp):
                 outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                 # Use lm_logits from the standard forward pass (with adapters if enabled)
                 logits = outputs.get("lm_logits") if isinstance(outputs, dict) else outputs[0]
                 shift_logits = logits[:, :-1, :].contiguous()

            shift_labels = input_ids[:, 1:].contiguous()

            # Calculate loss in fp32 for stability
            loss = F.cross_entropy(
                shift_logits.float().view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_id,
                reduction='sum' # Sum loss over non-ignored tokens
            )

            # Count non-ignored tokens for averaging
            num_tokens = (shift_labels != self.pad_id).sum().item()
            
            if num_tokens > 0:
                total_loss += loss.item()
                total_tokens += num_tokens
            eval_loop.set_postfix({"batch_loss": f"{(loss.item()/num_tokens):.4f}" if num_tokens > 0 else "0.00"})


        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        try:
            # Calculate perplexity: exp(average cross-entropy loss)
            ppl = math.exp(min(avg_loss, 20)) # Cap avg_loss to prevent overflow
        except OverflowError:
            ppl = float('inf')

        self.model.train() # Set model back to training mode
        return avg_loss, ppl