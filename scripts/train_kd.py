import argparse, torch, os, random, numpy as np
from models.student.gpt2_mtp import GPT2MTPStudent
from models.teacher.gpt2_large import GPT2TeacherLarge
from training.dataloader import get_dataloaders
from training.trainer import KDTrainer

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--tokenizer_name", type=str, default="gpt2")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--save_dir", type=str, default="experiments/logs/run1")
    p.add_argument("--no_teacher", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--offsets", type=int, nargs="*", default=[1,2])
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    train_loader, val_loader, pad_id = get_dataloaders(
        data_dir=args.data_dir,
        tokenizer_name=args.tokenizer_name,
        batch_size=args.batch_size,
    )

    student = GPT2MTPStudent(model_name="gpt2", offsets=tuple(args.offsets))
    teacher = None if args.no_teacher else GPT2TeacherLarge(model_name="gpt2-medium", device=next(student.parameters()).device.type)

    trainer = KDTrainer(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        pad_token_id=pad_id,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        alpha=args.alpha,
        temperature=args.temperature,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.fp16,
        save_dir=args.save_dir,
    )

    trainer.train()

if __name__ == "__main__":
    main()
