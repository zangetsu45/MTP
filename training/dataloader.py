# training/dataloader.py

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer


def _make_collate_fn(pad_token_id: int):
    """
    Collate that stacks 'input_ids' and provides/derives 'attention_mask'.
    Works for both packed and fixed-length datasets.
    """
    def collate(batch):
        # batch is a list of dicts
        input_ids = [b["input_ids"] for b in batch]
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        if "attention_mask" in batch[0]:
            attention_mask = [b["attention_mask"] for b in batch]
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        else:
            # derive mask from padding if not provided
            attention_mask = (input_ids != pad_token_id).long()

        return {"input_ids": input_ids, "attention_mask": attention_mask}
    return collate


def _seed_worker(worker_id: int):
    """
    Make dataloader workers deterministic (optional but helpful for reproducibility).
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _get_splits(ds: DatasetDict):
    """
    Return (train, val) datasets; handle 'test' vs 'validation' naming.
    """
    if "train" not in ds:
        raise ValueError("Expected a 'train' split in the saved dataset.")
    train = ds["train"]
    if "test" in ds:
        val = ds["test"]
    elif "validation" in ds:
        val = ds["validation"]
    else:
        raise ValueError("Expected a 'test' or 'validation' split in the saved dataset.")
    return train, val


def get_dataloaders(
    data_dir: str = "data/processed",
    tokenizer_name: str = "gpt2",
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = True,
    shuffle: bool = True,
):
    """
    Load dataset saved by scripts/preprocess_data.py and return train/val DataLoaders
    plus the pad_token_id.

    - Works for packed (concat+chunk) or fixed-length preprocessed datasets.
    - If attention_mask is missing, it is derived from pad_token_id.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} not found. Run scripts/preprocess_data.py first.")

    ds = load_from_disk(data_dir)
    if not isinstance(ds, DatasetDict):
        raise ValueError(f"Expected a DatasetDict at {data_dir}, got: {type(ds)}")

    train_ds, val_ds = _get_splits(ds)

    # Tokenizer for pad id (handles GPT-2 vs Llama/Mistral)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_token_id = tokenizer.pad_token_id
    collate_fn = _make_collate_fn(pad_token_id)

    # DataLoader worker settings
    generator = torch.Generator()
    generator.manual_seed(42)
    persistent = num_workers > 0
    # prefetch_factor only applies when num_workers > 0; use smaller value to reduce RAM spikes
    dl_kwargs = {}
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2
        dl_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
        **dl_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=collate_fn,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=generator,
        **dl_kwargs,
    )
    return train_loader, val_loader, pad_token_id


if __name__ == "__main__":
    tl, vl, pad_id = get_dataloaders()
    b = next(iter(tl))
    print(b["input_ids"].shape, b["attention_mask"].shape, pad_id)
