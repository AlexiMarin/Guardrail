"""Shared dataset loading/tokenization for the 3 training scripts."""

from pathlib import Path

import datasets


def load_split(data_dir: str, split: str) -> datasets.Dataset:
    """Load one curated split written by data/build_dataset.py (train/val/test/unseen_attacks)."""
    path = Path(data_dir) / f"{split}.parquet"
    return datasets.Dataset.from_parquet(str(path))


def tokenize_dataset(dataset: datasets.Dataset, tokenizer, max_length: int) -> datasets.Dataset:
    def _tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    remove_cols = [c for c in ("text", "source") if c in dataset.column_names]
    tokenized = dataset.map(_tokenize, batched=True, remove_columns=remove_cols)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")
    return tokenized
