"""Tests for kutils.datasets.hf.HFDatasetAdapter (no `datasets` needed)."""

import torch

from kutils.datasets.hf import HFDatasetAdapter


class FakeHFDataset:
    """Duck-typed stand-in for a HuggingFace `datasets.Dataset`."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _rows(n=4):
    return [{"image": torch.randn(3, 4), "label": i % 2} for i in range(n)]


def test_adapter_len_and_getitem():
    adapter = HFDatasetAdapter(FakeHFDataset(_rows(4)), input_key="image", label_key="label")
    assert len(adapter) == 4
    x, y = adapter[2]
    assert x.shape == (3, 4)
    assert y.item() in (0, 1)


def test_adapter_applies_transform():
    rows = _rows(1)
    adapter = HFDatasetAdapter(
        FakeHFDataset(rows), input_key="image", label_key="label", transform=lambda t: t * 2
    )
    x, _ = adapter[0]
    assert torch.equal(x, rows[0]["image"] * 2)


def test_adapter_converts_non_tensor_inputs():
    rows = [{"image": [[1.0, 2.0], [3.0, 4.0]], "label": 1}]
    adapter = HFDatasetAdapter(FakeHFDataset(rows), input_key="image", label_key="label")
    x, y = adapter[0]
    assert x.shape == (2, 2)
    assert y.item() == 1
