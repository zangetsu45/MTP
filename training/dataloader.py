# main_llama.py

import argparse, torch, os, random, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.student.medusa_llama_student import MedusaLlamaStudent  # <-- UPDATED
from training.trainer import MedusaTrainer
from training.dataloader import get_dataloaders

def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    # (Optional) determinism knobs
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main():
    p = argparse.ArgumentParser()
    # --- Model Config ---
    p.add_argument("--student_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    p.add_argument("--teacher_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                       help="Teacher model for distillation. Use same as student for self-distill.")
    p.add_argument("--tokenizer_name", type=str, default=None, 
                       help="Defaults to student_name")
    p.add_argument("--offsets", type=int, nargs="*", default=[1, 2, 3])

    # --- QLoRA / LoRA Config ---
    p.add_argument('--qlora', action=argparse.BooleanOptionalAction, default=True,
                       help="Enable QLoRA (4-bit quantization)")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_targets", type=str, nargs="*", default=["q_proj", "v_proj"])

    # --- Data Config ---
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--batch_size", type=int, default=8)

    # --- Training Config ---
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default="experiments/logs/medusa_llama_run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", 
                       help="Use AMP. (Forced True if QLoRA is enabled)")

    # --- Medusa Loss Config ---
    p.add_argument("--temperature", type=float, default=1.0,   # KL temperature
                       help="Distillation temperature")
    p.add_argument("--lr_backbone", type=float, default=5e-6)  # small LR
    p.add_argument("--lr_heads", type=float, default=5e-4)     # larger LR
    p.add_argument("--warmup_epochs", type=int, default=1,     # MEDUSA-1 heads-only
                       help="Epochs for Stage 1 (heads-only) training")
    p.add_argument("--lambda0", type=float, default=1.0,       # weight on heads loss in MEDUSA-2
                       help="Weight for Medusa heads loss in Stage 2")
    p.add_argument("--lambda_base", type=float, default=0.8,   # per-head decay base
                       help="Decay factor for Medusa head losses (lambda^k)")
    p.add_argument("--lambda0_warmup", action="store_true",    # ramp λ0 after warmup
                       help="Gradually ramp up lambda0 during Stage 2")

    # --- optional: MEDUSA-1 only (no teacher; warmup == all epochs) ---
    p.add_argument("--medusa1_only", action="store_true",
                       help="Run MEDUSA-1 (heads-only) training for all epochs")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    # QLoRA requires AMP/fp16
    use_amp = True if args.qlora else args.fp16
    if args.qlora and not args.fp16:
        print("[warn] QLoRA requires AMP. Forcing --fp16 (use_amp=True).")

    # --- Load Tokenizer ---
    tokenizer_name = args.tokenizer_name or args.student_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Load Data ---
    train_loader, val_loader, pad_id = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=tokenizer_name,
        batch_size=args.batch_size,
    )

    # --- Load Student ---
    student = MedusaLlamaStudent(
        model_name=args.student_name, 
        offsets=tuple(args.offsets),
        lora=True, # Assuming LoRA is always desired with this student
        qlora=args.qlora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_targets=args.lora_targets
    )

    # --- Load Teacher ---
    device_type = next(student.parameters()).device.type
    teacher = None
    warmup_epochs = args.warmup_epochs
    
    if args.medusa1_only:
        warmup_epochs = args.epochs  # entire run is heads-only
        print("[info] Running MEDUSA-1 training only (all epochs). Teacher model will not be loaded.")
    else:
        print(f"[info] Loading teacher model: {args.teacher_name}")
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher_name,
            torch_dtype=torch.bfloat16, # Load in bfloat16 for VRAM efficiency
        )
        # Trainer will move to device, freeze, and set to eval()

    # Guard: if no teacher but joint stage requested, force MEDUSA-1 only
    if teacher is None and warmup_epochs < args.epochs:
        print("[warn] No teacher loaded but joint stage requested; "
              "setting warmup_epochs=epochs to run MEDUSA-1 only.")
        warmup_epochs = args.epochs

    # --- Log run config ---
    print("=== Run config ===")
    print(f"student_name:   {args.student_name} (QLoRA: {args.qlora})")
    print(f"teacher_name:   {args.teacher_name if teacher is not None else 'None (MEDUSA-1 only)'}")
    print(f"offsets:        {args.offsets}")
    print(f"epochs:         {args.epochs}  (warmup_epochs={warmup_epochs})")
    print(f"lrs:            backbone={args.lr_backbone}, heads={args.lr_heads}")
    print(f"lambda0:        {args.lambda0}  (lambda0_warmup={args.lambda0_warmup})")
    print(f"lambda_base:    {args.lambda_base}")
    print(f"temperature:    {args.temperature}")
    print(f"use_amp (fp16): {use_amp}")
    print("==================")

    # --- Initialize Trainer ---
    trainer = MedusaTrainer(
        model=student,
        teacher=teacher,                      # None if MEDUSA-1 only
        tokenizer=tokenizer,                  # Pass tokenizer for saving
        temperature=args.temperature,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_token_id=pad_id,
        epochs=args.epochs,
        lr_backbone=args.lr_backbone,
        lr_heads=args.lr_heads,
        weight_decay=args.weight_decay,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        use_amp=use_amp and torch.cuda.is_available(),
        lambda0=args.lambda0,
        lambda_base=args.lambda_base,
        warmup_epochs=warmup_epochs,
        lambda0_warmup=args.lambda0_warmup,
        save_dir=args.save_dir,
    )

    trainer.train()

if __name__ == "__main__":
    main()