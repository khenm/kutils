""" "HuggingFace Datasets integration (requires the optional `hf` extra)."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset


def load_hf_dataset(
    name: str,
    config: str | None = None,
    *,
    split: str = "train",
    cache_dir: str | None = None,
    streaming: bool = False,
    revision: str | None = None,
):
    """Load a dataset split from the Hub (thin `datasets.load_dataset`
    wrapper)."""
    from datasets import load_dataset  # pyright: ignore[reportAttributeAccessIssue]

    return load_dataset(
        name,
        config,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        revision=revision,
    )


class HFDatasetAdapter(Dataset):
    """Adapt a `datasets.Dataset` to the torch `Dataset` API.

    `transform` is applied to the raw input value before it becomes a tensor.
    """

    def __init__(
        self,
        hf_dataset: Any,
        input_key: str,
        label_key: str,
        transform=None,
    ):
        self.hf_dataset = hf_dataset
        self.input_key = input_key
        self.label_key = label_key
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.hf_dataset[idx]
        x = row[self.input_key]
        if self.transform is not None:
            x = self.transform(x)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        y = torch.as_tensor(row[self.label_key])
        return x, y
