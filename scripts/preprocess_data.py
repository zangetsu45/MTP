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
    # --- UPDATED DEFAULTS FOR COMPUTE BUDGET ---
    dataset_name: str = "wikitext",
    dataset_config: str | None = "wikitext-103-v1", # Use WikiText-103
    dataset_split: str = "train",
    # Keep tokenizer matching the target Llama model
    tokenizer_name: str = "NousResearch/Llama-2-7b-chat-hf",
    # --- END UPDATED DEFAULTS ---
    output_dir: str = "data/processed_wikitext103", # Suggest specific output dir
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
    dataset = normalize_dataset(dataset, name=f"{dataset_name}:{dataset_config or ''}")

    # 3) Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    # --- THIS LOGIC IS ALREADY PERFECT FOR LLAMA ---
    # Ensure PAD exists (Llama/Mistral often have no pad)
    if tokenizer.pad_token is None:
        print("Tokenizer has no pad_token; setting pad_token = eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    # Use right-padding for causal LM batches
    try:
        tokenizer.padding_side = "right"
    except Exception:
        pass
    # --- END LLAMA-COMPATIBLE LOGIC ---

    eos_id = tokenizer.eos_token_id

    # 4) Tokenize
    # This logic is model-agnostic and correct for both packing/not-packing
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
            # Optionally ensure EOS ends the sequence
            if eos_id is not None:
                for ids in enc["input_ids"]:
                    # Check if the last token is pad_token or eos_token before replacing
                    if ids[-1] != eos_id and ids[-1] != tokenizer.pad_token_id:
                         # Replace only if it's neither EOS nor PAD
                         ids[-1] = eos_id
                    elif len(ids) > 1 and ids[-2] != eos_id and ids[-1] == tokenizer.pad_token_id:
                         # If ends in PAD, try putting EOS before PAD if possible
                         ids[-1] = eos_id # Replace last PAD with EOS
        return enc

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=[c for c in dataset.column_names if c != "text"],  # keep text for pack step
        num_proc=num_proc,
        desc="Tokenizing",
    )

    # 5) Pack or keep fixed-length (This logic is also model-agnostic and correct)
    if pack:
        # Optionally append EOS between documents so boundaries are explicit
        def pack_fn(examples):
            all_ids: List[int] = []
            num_docs = len(examples["input_ids"])
            for i, ids in enumerate(examples["input_ids"]):
                all_ids.extend(ids) # Add token ids
                # Add EOS between documents, except after the very last one
                if eos_between_documents and eos_id is not None and i < num_docs - 1:
                     all_ids.append(eos_id)

            # Trim trailing tokens to make length a multiple of block_size
            total_len = (len(all_ids) // block_size) * block_size
            all_ids = all_ids[:total_len]

            # Chunk into blocks
            input_ids = [all_ids[i : i + block_size] for i in range(0, total_len, block_size)]
            # Packed sequences always have full attention
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
    else: # Fixed-length padding/truncation case
        # Ensure attention_mask exists (it should when padding="max_length")
        if "attention_mask" not in tokenized.column_names:
            print("attention_mask missing; creating default masks based on padding...")
            def ensure_mask(examples):
                am = [[1 if t != tokenizer.pad_token_id else 0 for t in ids] for ids in examples["input_ids"]]
                return {"attention_mask": am}
            tokenized = tokenized.map(
                ensure_mask, batched=True, num_proc=num_proc, desc="Building attention_mask"
            )
        # Remove raw text column if it exists
        if "text" in tokenized.column_names:
            tokenized = tokenized.remove_columns(["text"])

    # 6) Train/val split (if the original split wasn't 'train')
    # If we loaded 'train', we need to create a validation split.
    # If we loaded 'validation' or 'test', we might just use that directly (not typical).
    if dataset_split == "train":
        print("Creating train/val split (e.g., 5% validation)...")
        # Ensure enough data for a split, adjust test_size if needed
        test_fraction = 0.05
        if len(tokenized) * test_fraction < 1:
             print("[warn] Dataset too small for 5% validation split. Using 1 example for validation.")
             test_fraction = max(1, len(tokenized)) # Use at least 1 example
             
        dataset_dict = tokenized.train_test_split(test_size=test_fraction, seed=42, shuffle=True)
    else:
         # If user loaded 'validation' or 'test', wrap it in a DatasetDict
         # Assuming 'train' needs to exist for the dataloader, create a dummy one or raise error
         print(f"[warn] Loaded split '{dataset_split}'. Creating DatasetDict format.")
         # This case might need more specific handling depending on use case
         dataset_dict = DatasetDict({
              dataset_split: tokenized,
              'train': tokenized.select([0]) # Dummy train split - adjust as needed
         })


    # 7) Save
    print(f"Saving tokenized dataset to {output_dir} ...")
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    dataset_dict.save_to_disk(output_dir)
    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess datasets for MEDUSA (self-distill friendly)")

    # --- UPDATED DEFAULTS ---
    # Dataset selection (changed back to WikiText-103)
    parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="Hugging Face dataset name (e.g., wikitext, openwebtext)")
    parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1",
                        help="Dataset config name (e.g., wikitext-103-v1)")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split to process (usually 'train')")

    # Tokenizer / IO (Keep matching the target Llama model)
    parser.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-chat-hf",
                        help="HF tokenizer/model name (must match your student model)")
    parser.add_argument("--output_dir", type=str, default="data/processed_wikitext103", # Updated default
                        help="Where to save processed data")
    # --- END UPDATED DEFAULTS ---

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

    # Call the main function with parsed arguments
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