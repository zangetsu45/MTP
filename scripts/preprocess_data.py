# scripts/preprocess_data.py

import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def normalize_dataset(ds, name=""):
    # drop empty lines (common in WikiText)
    ds = ds.filter(lambda x: x.get("text") is not None and len(x["text"].strip()) > 0)
    print(f"{name} after cleaning: {len(ds)} samples")
    return ds


def preprocess_and_save(tokenizer_name="gpt2", output_dir="data/processed",
                        max_length=512, sample_size=None):
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load dataset
    print("Loading WikiText-103...")
    dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")

    # Optionally downsample if sample_size is given
    if sample_size:
        dataset = dataset.shuffle(seed=42).select(range(min(sample_size, len(dataset))))

    # 2. Normalize dataset
    dataset = normalize_dataset(dataset, "WikiText")

    # 3. Load tokenizer
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Tokenize function
    def tokenize_fn(examples):
        encodings = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Ensure EOS at the last position
        for ids in encodings["input_ids"]:
            if ids[-1] != tokenizer.eos_token_id:
                ids[-1] = tokenizer.eos_token_id
        return encodings

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # 5. Train/val split
    dataset_dict = tokenized_dataset.train_test_split(test_size=0.05, seed=42)

    # 6. Save
    print(f"Saving tokenized dataset to {output_dir}...")
    dataset_dict.save_to_disk(output_dir)

    print("✅ Preprocessing complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess WikiText-103 for KD training")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Where to save processed data")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--sample_size", type=int, default=None, help="Optional limit for testing")
    args = parser.parse_args()

    preprocess_and_save(
        tokenizer_name=args.tokenizer,
        output_dir=args.output_dir,
        max_length=args.max_length,
        sample_size=args.sample_size,
    )
