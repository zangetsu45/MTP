import argparse, torch, os, random, numpy as np
from models.student.gpt2_mtp import GPT2MTPStudent
from models.teacher.gpt2_large import GPT2TeacherLarge
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
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--student_name", type=str, default="gpt2")  # <-- NEW
    p.add_argument("--tokenizer_name", type=str, default=None)  # default: use student_name
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default="experiments/logs/medusa_run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--offsets", type=int, nargs="*", default=[1, 2, 3])

    # —— self-distill & training knobs ——
    p.add_argument("--teacher_name", type=str, default="gpt2-medium")
    p.add_argument("--temperature", type=float, default=1.0)   # KL temperature
    p.add_argument("--lr_backbone", type=float, default=5e-6)  # small LR
    p.add_argument("--lr_heads", type=float, default=5e-4)     # larger LR
    p.add_argument("--warmup_epochs", type=int, default=1)     # MEDUSA-1 heads-only
    p.add_argument("--lambda0", type=float, default=1.0)       # weight on heads loss in MEDUSA-2
    p.add_argument("--lambda_base", type=float, default=0.8)   # per-head decay base
    p.add_argument("--lambda0_warmup", action="store_true")    # ramp λ0 after warmup

    # —— optional: MEDUSA-1 only (no teacher; warmup == all epochs) ——
    p.add_argument("--medusa1_only", action="store_true")      # <-- NEW
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer_name = args.tokenizer_name or args.student_name
    train_loader, val_loader, pad_id = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=tokenizer_name,
        batch_size=args.batch_size,
    )

    # Student with MEDUSA heads
    student = GPT2MTPStudent(model_name=args.student_name, offsets=tuple(args.offsets))

    # Teacher for self-distillation (only if not MEDUSA-1 only)
    device_type = next(student.parameters()).device.type
    teacher = None
    warmup_epochs = args.warmup_epochs
    if args.medusa1_only:
        warmup_epochs = args.epochs  # entire run is heads-only
    else:
        teacher = GPT2TeacherLarge(model_name=args.teacher_name, device=device_type)

    # Guard: if no teacher but joint stage requested, force MEDUSA-1 only
    if teacher is None and warmup_epochs < args.epochs:
        print("[warn] No teacher provided but joint stage requested; "
              "setting warmup_epochs=epochs to run MEDUSA-1 only.")
        warmup_epochs = args.epochs

    # Log run config
    print("=== Run config ===")
    print(f"student_name:   {args.student_name}")
    print(f"teacher_name:   {args.teacher_name if teacher is not None else 'None (MEDUSA-1 only)'}")
    print(f"offsets:        {args.offsets}")
    print(f"epochs:         {args.epochs}  (warmup_epochs={warmup_epochs})")
    print(f"lrs:            backbone={args.lr_backbone}, heads={args.lr_heads}")
    print(f"lambda0:        {args.lambda0}  (lambda0_warmup={args.lambda0_warmup})")
    print(f"lambda_base:    {args.lambda_base}")
    print(f"temperature:    {args.temperature}")
    print(f"fp16:           {args.fp16}")
    print("==================")

    trainer = MedusaTrainer(
        model=student,
        teacher=teacher,                      # None if MEDUSA-1 only
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
        use_amp=args.fp16 and torch.cuda.is_available(),
        lambda0=args.lambda0,
        lambda_base=args.lambda_base,
        warmup_epochs=warmup_epochs,
        lambda0_warmup=args.lambda0_warmup,
        save_dir=args.save_dir,
    )

    trainer.train()

if __name__ == "__main__":
    main()
