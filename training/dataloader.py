# training/dataloader.py

import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer

def get_dataloaders(data_dir, tokenizer_name, batch_size):
    """Loads the processed dataset and creates DataLoaders."""

    print(f"Loading processed data from {data_dir}...")
    try:
        dataset_dict = load_from_disk(data_dir)
    except FileNotFoundError:
        print(f"Error: Processed data not found at {data_dir}.")
        print(f"Please run 'python preprocess_data.py' first.")
        exit(1)

    dataset_dict.set_format(type="torch", columns=["input_ids", "attention_mask"])

    train_ds = dataset_dict["train"]
    val_ds = dataset_dict["test"]

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4, # You can adjust this
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4, # You can adjust this
        pin_memory=True
    )

    # Get pad_token_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    print(f"Loaders ready. Train: {len(train_ds)} samples, Val: {len(val_ds)} samples.")
    return train_loader, val_loader, pad_id