# scripts/preprocess_data.py

import os
import argparse
from typing import List
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer


def normalize_dataset(ds: Dataset, name: str = "") -> Dataset:
    """Drop empty/whitespace-only examples."""
    ds = ds.filter(lambda x: x.get("text") is not None and len(x["text"].strip()) > 0)
    print(f"{name} after cleaning: {len(ds)} samples")
    return ds


def preprocess_and_save(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-v1",
    dataset_split: str = "train",
    tokenizer_name: str = "gpt2",
    output_dir: str = "data/processed",
    block_size: int = 1024,
    sample_size: int | None = None,
    pack: bool = True,                       # concat & chunk (recommended)
    eos_between_documents: bool = True,      # add EOS between docs when packing
    add_special_tokens_when_not_packing: bool = True,  # add BOS/EOS for fixed-length mode
    num_proc: int | None = None,             # parallelism for map
):
    os.makedirs(output_dir, exist_ok=True)
    num_proc = num_proc or max(1, min(8, (os.cpu_count() or 1)))

    # 1) Load dataset
    print(f"Loading dataset: {dataset_name} / {dataset_config} / split={dataset_split} ...")
    dataset = load_dataset(dataset_name, dataset_config, split=dataset_split)

    # Optional downsample
    if sample_size:
        sample_size = min(sample_size, len(dataset))
        dataset = dataset.shuffle(seed=42).select(range(sample_size))
        print(f"Downsampled to {sample_size} examples.")

    # 2) Normalize / clean
    dataset = normalize_dataset(dataset, name=f"{dataset_name}:{dataset_config}")

    # 3) Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # Ensure PAD exists (Llama/Mistral often have no pad)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Use right-padding for causal LM batches
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass

    eos_id = tokenizer.eos_token_id

    # 4) Tokenize
    # For packing: tokenize WITHOUT padding/truncation; later we concat & chunk to block_size
    # For fixed-length: tokenize WITH truncation/padding to block_size (and optionally force EOS at end)
    def tokenize_fn(examples):
        if pack:
            enc = tokenizer(
                examples["text"],
                add_special_tokens=False,   # avoid inserting BOS/EOS mid-pack
                truncation=False,
                padding=False,
            )
        else:
            enc = tokenizer(
                examples["text"],
                add_special_tokens=add_special_tokens_when_not_packing,
                truncation=True,
                padding="max_length",
                max_length=block_size,
            )
            # Optionally ensure EOS ends the sequence (helpful for GPT-2 style)
            if eos_id is not None:
                for ids in enc["input_ids"]:
                    if ids[-1] != eos_id:
                        ids[-1] = eos_id
        return enc

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "text"],  # keep text for pack step
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # 5) Pack or keep fixed-length
    if pack:
        # Optionally append EOS between documents so boundaries are explicit
        def pack_fn(examples):
            all_ids: List[int] = []
            for ids in examples["input_ids"]:
                if eos_between_documents and eos_id is not None:
                    # add EOS if not present to mark boundary
                    if len(ids) == 0 or ids[-1] != eos_id:
                        ids = ids + [eos_id]
                all_ids.extend(ids)

            # Trim to multiple of block_size
            total_len = (len(all_ids) // block_size) * block_size
            all_ids = all_ids[:total_len]

            # Chunk
            input_ids = [all_ids[i : i + block_size] for i in range(0, total_len, block_size)]
            attention_mask = [[1] * block_size for _ in range(len(input_ids))]
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        print("Packing (concatenate + chunk) to fixed blocks...")
        # We remove 'text' after packing to avoid keeping raw text in disk dump
        tokenized = tokenized.map(
            pack_fn,
            batched=True,
            remove_columns=["text", "input_ids", "attention_mask"],  # remove originals if present
            num_proc=num_proc,
            desc="Packing",
        )
    else:
        # Ensure attention_mask exists (it does when padding="max_length")
        if "attention_mask" not in tokenized.column_names:
            print("attention_mask missing; creating default masks...")
            def ensure_mask(examples):
                am = [[1 if t != tokenizer.pad_token_id else 0 for t in ids] for ids in examples["input_ids"]]
                return {"attention_mask": am}
            tokenized = tokenized.map(
                ensure_mask, batched=True, num_proc=num_proc, desc="Building attention_mask"
            )
        # Remove raw text
        if "text" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["text"])

    # 6) Train/val split
    print("Creating train/val split (5%)...")
    dataset_dict = tokenized.train_test_split(test_size=0.05, seed=42)

    # 7) Save
    print(f"Saving tokenized dataset to {output_dir} ...")
    dataset_dict.save_to_disk(output_dir)
    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for MEDUSA (self-distill friendly)")

    # Dataset selection
    parser.add_argument("--dataset_name", type=str, default="wikitext")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1")
    parser.add_argument("--dataset_split", type=str, default="train")

    # Tokenizer / IO
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer/model name")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Where to save processed data")

    # Sequence shaping
    parser.add_argument("--block_size", type=int, default=1024, help="Sequence length for training")
    parser.add_argument("--no_pack", action="store_true", help="Disable concat+chunk packing (use fixed-length pad)")
    parser.add_argument("--no_eos_between_docs", action="store_true",
                        help="When packing, do NOT append EOS between documents")
    parser.add_argument("--no_special_tokens_when_not_packing", action="store_true",
                        help="When not packing, do NOT add special tokens (BOS/EOS)")

    # Misc
    parser.add_argument("--sample_size", type=int, default=None, help="Optional limit for quick tests")
    parser.add_argument("--num_proc", type=int, default=None, help="Parallel processes for map")

    args = parser.parse_args()

    preprocess_and_save(
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        dataset_split=args.dataset_split,
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        block_size=args.block_size,
        sample_size=args.sample_size,
        pack=not args.no_pack,
        eos_between_documents=not args.no_eos_between_docs,
        add_special_tokens_when_not_packing=not args.no_special_tokens_when_not_packing,
        num_proc=args.num_proc,
    )
