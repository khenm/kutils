"""Thin HuggingFace Datasets integration.

Load real data from the HuggingFace Hub via `datasets.load_dataset` instead
of hand-rolled downloaders. Requires the optional `hf` extra (`datasets`).
"""

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
    """Load a dataset split from the HuggingFace Hub.

    Thin wrapper around `datasets.load_dataset` so callers don't need the
    import at every call site, and so the signature stays stable if the
    underlying library's changes.
    """
    from datasets import load_dataset

    return load_dataset(
        name,
        config,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        revision=revision,
    )


class HFDatasetAdapter(Dataset):
    """Adapt a HuggingFace `datasets.Dataset` to the torch `Dataset` API.

    Args:
        hf_dataset: A `datasets.Dataset` (not a streaming/`IterableDataset`).
        input_key: Column holding the model input.
        label_key: Column holding the target label.
        transform: Optional callable applied to the raw input value before
            it's turned into a tensor (e.g. a torchvision transform, a
            tokenizer call).

    Example:
        >>> ds = load_hf_dataset("cifar10", split="train")
        >>> train_set = HFDatasetAdapter(ds, input_key="img", label_key="label")
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
