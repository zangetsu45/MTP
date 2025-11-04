# training/trainer.py (Updated for LoRA Self-Distillation Trick & NameError/AttributeError fixes)

import os
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from training.loss import medusa1_loss, medusa2_loss # Ensure these are in the correct relative path
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
        tokenizer=None,
    ):
        self.model = model
        self.temperature = temperature
        # self.resume_from_checkpoint = resume_from_checkpoint # <-- REMOVED THIS LINE (NameError FIX)

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
             # Find the first parameter that is not None to get the device
             first_param = next(p for p in model.parameters() if p is not None)
             self.device = first_param.device
        except StopIteration:
             print("[warn] Could not determine model device automatically (no parameters found?). Defaulting.")
             self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.epochs = epochs
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm

        # --- CORRECTED AMP Check ---
        # Force AMP if QLoRA/quantization is detected (check for uint8 or int8)
        # Removed torch.float4 which caused AttributeError
        if any(p.dtype in [torch.uint8, torch.int8] for _, p in model.backbone.named_parameters()):
            print("[info] Detected quantized backbone parameters (uint8/int8). Forcing use_amp=True.")
            self.use_amp = True
        else:
            # Use the provided use_amp flag if not quantized
            self.use_amp = use_amp
        # --- END CORRECTION ---


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

        param_groups = []
        if backbone_trainable_params:
             param_groups.append({"params": backbone_trainable_params, "lr": lr_backbone, "name": "backbone"})
        else:
             print("[warn] No trainable parameters found in model.backbone. LoRA adapters might not be set up correctly or are frozen.")

        # Always add head params, even if backbone is frozen initially
        param_groups.append({"params": head_params, "lr": lr_heads, "name": "heads"})
        # --------------------------------

        self.optimizer = AdamW(param_groups, weight_decay=weight_decay)

        total_steps = (len(train_loader) * epochs) // max(1, grad_accum)
        # Calculate warmup steps based on epochs, not just a fraction of total steps
        warmup_steps = (len(train_loader) * self.warmup_epochs) // max(1, grad_accum)
        # Ensure warmup steps don't exceed total steps if warmup_epochs >= epochs
        warmup_steps = min(warmup_steps, total_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
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
        print("[info] Freezing backbone LoRA adapters...")
        count = 0
        for name, param in self.model.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = False
                count += 1
        # print(f"Froze {count} LoRA parameters.") # Optional debug

    def _unfreeze_backbone(self):
        """Unfreezes LoRA adapters in the backbone."""
        print("[info] Unfreezing backbone LoRA adapters...")
        count = 0
        for name, param in self.model.backbone.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
                count += 1
        # print(f"Unfroze {count} LoRA parameters.") # Optional debug

    def _set_backbone_lr(self, lr: float):
        """Sets the LR for the 'backbone' parameter group (LoRA adapters)."""
        updated = False
        for g in self.optimizer.param_groups:
            if g.get("name") == "backbone":
                g["lr"] = lr
                updated = True
        # if not updated and lr > 0: # Optional warning
        #     print("[warn] 'backbone' parameter group not found in optimizer. Could not set LR.")


    # ---------- checkpoint helpers ----------
    def _save(self, epoch, val_loss, best=False):
        tag = "best" if best else f"epoch_{epoch:03d}"
        out_dir = os.path.join(self.ckpt_dir, tag)
        os.makedirs(out_dir, exist_ok=True)

        # --- Saving PEFT Model + Custom Head ---
        print(f"Saving model components to {out_dir}...")
        try:
            # Save PEFT adapter state + config (saves only adapters, not base model)
            self.model.backbone.save_pretrained(out_dir)

            # Save the custom Medusa head separately
            torch.save(self.model.mtp_head.state_dict(), os.path.join(out_dir, "mtp_head.pt"))
        except Exception as e:
            print(f"[error] Failed during model saving: {e}")
            return # Avoid saving corrupted state if model save fails
        # --------------------------------------

        # Save Tokenizer (Unchanged)
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            try:
                self.tokenizer.save_pretrained(out_dir)
            except Exception as e:
                 print(f"[warn] Failed to save tokenizer: {e}")

        # Save Trainer state (Unchanged)
        try:
            torch.save({
                "epoch": epoch,
                "val_loss": float(val_loss),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
            }, os.path.join(out_dir, "training_state.pt"))
            tqdm.write(f"[checkpoint] {'Best' if best else f'Epoch {epoch}'} saved successfully to {out_dir}")
        except Exception as e:
            print(f"[error] Failed to save training state: {e}")


    def _find_latest_checkpoint(self):
        """Finds the latest 'epoch_XXX' checkpoint in self.ckpt_dir."""
        if not os.path.isdir(self.ckpt_dir):
            return None

        # Get all subdirectories that start with 'epoch_'
        try:
            epoch_dirs = [
                d for d in os.listdir(self.ckpt_dir)
                if os.path.isdir(os.path.join(self.ckpt_dir, d)) and d.startswith("epoch_")
            ]
        except OSError:
            return None

        if not epoch_dirs:
            return None

        # Extract epoch numbers and find the max
        latest_epoch = -1
        latest_epoch_dir = None
        for d in epoch_dirs:
             try:
                  # Extract number after the last underscore
                  epoch_num = int(d.split('_')[-1])
                  if epoch_num > latest_epoch:
                       latest_epoch = epoch_num
                       latest_epoch_dir = d
             except (ValueError, IndexError):
                  continue # Ignore directories not matching the pattern like 'best'

        if latest_epoch_dir:
            return os.path.join(self.ckpt_dir, latest_epoch_dir)
        else:
            return None


    def _load_checkpoint(self, path):
        """Loads a checkpoint from the specified directory."""
        if not os.path.isdir(path):
            print(f"[warn] Checkpoint path not found, starting from scratch: {path}")
            return 1 # Start from epoch 1

        try:
            # 1. Load trainer state FIRST (to get epoch number)
            state_path = os.path.join(path, "training_state.pt")
            if not os.path.exists(state_path):
                 print(f"[warn] training_state.pt not found in {path}, cannot resume training state. Starting epoch 1.")
                 # Attempt to load model weights anyway, but start from epoch 1
                 start_epoch = 1
                 load_trainer_state = False
            else:
                state = torch.load(state_path, map_location='cpu') # Load to CPU first
                # Resume on the *next* epoch after the one that was saved
                start_epoch = state.get("epoch", 0) + 1
                load_trainer_state = True

            # 2. Load model weights
            # Load PEFT adapters (this loads into the existing QLoRA model structure)
            print(f"[info] Loading adapters from checkpoint: {path}...")
            # Ensure the adapter name matches if needed, default is usually fine
            self.model.backbone.load_adapter(path, adapter_name="default")

            # Load Medusa head
            head_path = os.path.join(path, "mtp_head.pt")
            if os.path.exists(head_path):
                print("[info] Loading MTP head weights...")
                # Load state dict, moving tensors to the correct device
                head_state_dict = torch.load(head_path, map_location=self.device)
                self.model.mtp_head.load_state_dict(head_state_dict)
            else:
                 print(f"[error] mtp_head.pt not found in {path}. Cannot resume model state completely.")
                 raise FileNotFoundError(f"mtp_head.pt missing from checkpoint: {path}")

            # 3. Load optimizer, scheduler, scaler states if available
            if load_trainer_state:
                 print("[info] Loading optimizer, scheduler, and scaler states...")
                 self.optimizer.load_state_dict(state["optimizer_state"])
                 self.scheduler.load_state_dict(state["scheduler_state"])
                 # Load scaler state safely
                 if "scaler_state" in state and self.scaler is not None:
                     self.scaler.load_state_dict(state["scaler_state"])

                 # Move optimizer states to the correct device if needed
                 for state_dict in self.optimizer.state.values():
                     for k, v in state_dict.items():
                         if isinstance(v, torch.Tensor):
                             state_dict[k] = v.to(self.device)

            tqdm.write(f"[info] Resumed training state from checkpoint: {path} (starting Epoch {start_epoch})")
            return start_epoch

        except Exception as e:
            print(f"[error] Failed to load checkpoint from {path}: {e}. Starting from scratch.")
            # Reset optimizer just in case
            self.optimizer.zero_grad(set_to_none=True)
            # Potentially reset scheduler/scaler too if needed
            return 1


    # ---------- training loop ----------
    def train(self):
        best_val = float("inf")
        global_step = 0 # Track global steps if needed

        # --- RESUME LOGIC ---
        start_epoch = 1
        latest_checkpoint_path = self._find_latest_checkpoint()

        if latest_checkpoint_path:
            tqdm.write(f"[info] Found latest checkpoint: {latest_checkpoint_path}")
            try:
                # _load_checkpoint returns the *next* epoch number to start from
                start_epoch = self._load_checkpoint(latest_checkpoint_path)
                # Estimate global step based on resumed epoch and loader length
                global_step = (start_epoch - 1) * (len(self.train_loader) // self.grad_accum)
            except Exception as e:
                print(f"[warn] Failed to load latest checkpoint, starting from scratch. Error: {e}")
                start_epoch = 1
        else:
            tqdm.write("[info] No checkpoint found. Starting from scratch.")

        # --- MAIN TRAINING LOOP ---
        for epoch in range(start_epoch, self.epochs + 1):
            self.model.train()
            # Ensure Medusa head is trainable even if backbone adapters are frozen
            self.model.mtp_head.train()
            self.optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0

            # λ₀ warmup schedule (Unchanged)
            if self.lambda0_warmup and epoch > self.warmup_epochs:
                # Ensure division by zero is avoided if epochs == warmup_epochs
                num_joint_epochs = max(1, self.epochs - self.warmup_epochs)
                frac = (epoch - self.warmup_epochs) / num_joint_epochs
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
                self._unfreeze_backbone() # Unfreezes LoRA adapters
                self._set_backbone_lr(self.lr_backbone)
                use_medusa2 = True

            loop = tqdm(self.train_loader, total=len(self.train_loader), desc=f"Epoch {epoch} | {stage}", leave=True)
            for step, batch in enumerate(loop, 1):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # --- STUDENT FORWARD PASS (with adapters enabled) ---
                with autocast(enabled=self.use_amp):
                    out_student = self.model(input_ids=input_ids, attention_mask=attention_mask)

                # --- COMPUTE LOSS ---
                loss_value = None # Initialize loss_value
                if not use_medusa2:
                    # MEDUSA-1: heads loss only
                    with autocast(enabled=self.use_amp):
                         loss_value = medusa1_loss(
                             out_student=out_student,
                             input_ids=input_ids,
                             pad_token_id=self.pad_id,
                             offsets=tuple(self.model.mtp_head.offsets),
                             attention_mask=attention_mask,
                             lambda_base=self.lambda_base,
                         )
                else:
                    # MEDUSA-2: LoRA self-distillation
                    # --- TEACHER FORWARD PASS (same model, adapters disabled) ---
                    teacher_logits = None
                    with torch.no_grad():
                        with self.model.backbone.disable_adapter():
                             with autocast(enabled=self.use_amp):
                                  teacher_out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                             # Robustly get lm_logits
                             teacher_logits = teacher_out.get("lm_logits") if isinstance(teacher_out, dict) else None
                             if teacher_logits is None:
                                  try:
                                      teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else getattr(teacher_out, 'logits', None)
                                  except (IndexError, AttributeError): pass
                             if teacher_logits is None:
                                  raise RuntimeError("Could not retrieve teacher logits.")

                    # Compute total loss in fp32 for stability
                    with autocast(enabled=False):
                        # Ensure student outputs are properly handled for fp32 conversion
                        out_student_fp32 = {}
                        for k, v in out_student.items():
                            if torch.is_tensor(v):
                                out_student_fp32[k] = v.clone() # Clone tensors
                            else:
                                out_student_fp32[k] = v # Copy other types

                        if "lm_logits" not in out_student_fp32:
                             print(f"[error] 'lm_logits' missing from student output at step {step}.")
                             continue

                        out_student_fp32["lm_logits"] = out_student_fp32["lm_logits"].float()

                        loss_value = medusa2_loss(
                            out_student=out_student_fp32,
                            teacher_logits=teacher_logits.float(),
                            input_ids=input_ids,
                            pad_token_id=self.pad_id,
                            offsets=tuple(self.model.mtp_head.offsets),
                            attention_mask=attention_mask,
                            lambda0=current_lambda0,
                            lambda_base=self.lambda_base,
                            temperature=self.temperature,
                        )

                # --- Loss Scaling for Grad Accum ---
                if loss_value is None:
                    print(f"[error] Loss calculation failed at epoch {epoch}, step {step}.")
                    continue # Skip this step

                loss = loss_value / self.grad_accum # Scale loss *before* backward

                # --- Backward + Step ---
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[warn] Step {global_step}: Detected NaN/Inf loss ({loss.item()}). Skipping backward and optimizer step.")
                    # Detach loss before clearing gradients if it's NaN/Inf
                    loss = loss.detach()
                    self.optimizer.zero_grad(set_to_none=True)
                    continue

                self.scaler.scale(loss).backward()

                # --- START: MODIFIED GRADIENT STEP BLOCK ---
                if (step + 1) % self.grad_accum == 0 or (step + 1) == len(self.train_loader):
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)

                    trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                    perform_step = True # Assume we will step
                    total_norm = float('inf') # Initialize norm
                    
                    if trainable_params:
                        total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.max_grad_norm)
                        # Check for NaN/Inf gradients after clipping
                        if torch.isnan(total_norm) or torch.isinf(total_norm):
                             print(f"[warn] Step {global_step}: Detected NaN/Inf gradient norm ({total_norm}) after unscaling. Skipping optimizer step.")
                             perform_step = False # Don't step
                    elif epoch > self.warmup_epochs:
                        # This is a problem: joint training but no trainable params
                        print(f"[warn] Step {global_step}: No trainable parameters found during joint training stage. Skipping step.")
                        perform_step = False
                    # else:
                        # This is fine: Stage 1 and no trainable backbone params is expected.
                        # Head params are separate and should be in trainable_params.
                        # If trainable_params is empty even in stage 1, something is wrong,
                        # but perform_step=False is still the right action.

                    
                    # Perform step only if gradients are valid and there are params
                    if perform_step:
                        self.scaler.step(self.optimizer)
                        self.scheduler.step()
                        global_step += 1
                    
                    # ALWAYS update the scaler. This resets its state and prevents the crash.
                    self.scaler.update() 
                    
                    # Always clear grads
                    self.optimizer.zero_grad(set_to_none=True)
                # --- END: MODIFIED GRADIENT STEP BLOCK ---


                # Log using the unscaled, accumulated loss item
                running_loss += loss.item() * self.grad_accum
                # Fetch LR safely
                lr_backbone_now = 0.0
                for g in self.optimizer.param_groups:
                    if g.get("name") == "backbone":
                         lr_backbone_now = g['lr']
                         break
                loop.set_postfix({"loss": f"{running_loss / (step + 1):.4f}", "lr_bb": f"{lr_backbone_now:.2e}"})


            # --- Validation ---
            val_loss, val_ppl = self.evaluate()
            tqdm.write(f"Epoch {epoch} Val Loss={val_loss:.4f} | Val PPL={val_ppl:.2f}")

            # Save checkpoint regardless of validation performance (for resuming)
            self._save(epoch, val_loss, best=False)

            # Save best model based on validation loss
            if val_loss < best_val:
                best_val = val_loss
                tqdm.write(f"*** New best validation loss: {best_val:.4f} ***")
                self._save(epoch, val_loss, best=True) # Overwrite 'best' checkpoint

        tqdm.write("🏁 Training finished.")


    # ---------- evaluation ----------
    @torch.no_grad()
    def evaluate(self):
        self.model.eval() # Set model to evaluation mode
        total_loss = 0.0
        total_tokens = 0

        eval_loop = tqdm(self.val_loader, desc="Evaluating", leave=False)
        for batch in eval_loop:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with autocast(enabled=self.use_amp):
                 outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                 # Use lm_logits from the standard forward pass (with adapters if enabled)
                 logits = outputs.get("lm_logits") if isinstance(outputs, dict) else None
                 if logits is None:
                      try: logits = outputs[0] if isinstance(outputs, tuple) else getattr(outputs, 'logits', None)
                      except (IndexError, AttributeError): pass
                 if logits is None:
                     print("[error] Could not get logits during evaluation.")
                     continue # Skip batch if logits are missing

                 shift_logits = logits[:, :-1, :].contiguous()

            shift_labels = input_ids[:, 1:].contiguous()

            # Calculate loss in fp32 for stability
            loss = F.cross_entropy(
                shift_logits.float().view(-T1, shift_logits.size(-1)), # Cast logits to fp32
                shift_labels.view(-1),
                ignore_index=self.pad_id,
                reduction='sum' # Sum loss over non-ignored tokens
            )

            # Count non-ignored tokens for averaging
            num_tokens = (shift_labels != self.pad_id).sum().item()

            if num_tokens > 0:
                total_loss += loss.item()
                total_tokens += num_tokens
                eval_loop.set_postfix({"batch_loss": f"{(loss.item()/num_tokens):.4f}"})
            else:
                 eval_loop.set_postfix({"batch_loss": "0.00"})


        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        try:
            # Calculate perplexity: exp(average cross-entropy loss)
            ppl = math.exp(min(avg_loss, 700)) # Increase cap slightly for fp32 loss
        except OverflowError:
            ppl = float('inf')

        self.model.train() # Set model back to training mode
        return avg_loss, ppl