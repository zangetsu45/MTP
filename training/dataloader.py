# training/dataloader.py
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

def _make_collate_fn(pad_token_id):
    def collate(batch):
        input_ids = torch.tensor([x["input_ids"] for x in batch], dtype=torch.long)
        if "attention_mask" in batch[0]:
            attention_mask = torch.tensor([x["attention_mask"] for x in batch], dtype=torch.long)
        else:
            attention_mask = (input_ids != pad_token_id).long()
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return collate

def get_dataloaders(data_dir="data/processed", tokenizer_name="gpt2", batch_size=8, num_workers=2, pin_memory=True, drop_last=True, shuffle=True):
    ds = load_from_disk(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate_fn = _make_collate_fn(tokenizer.pad_token_id)
    train_loader = DataLoader(
        ds["train"],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ds["test"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, tokenizer.pad_token_id

if __name__ == "__main__":
    tl, vl, pad_id = get_dataloaders()
    b = next(iter(tl))
    print(b["input_ids"].shape, b["attention_mask"].shape, pad_id)
