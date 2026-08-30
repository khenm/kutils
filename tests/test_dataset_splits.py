"""Tests for kutils.datasets.utils splits."""

import pytest
import torch
from torch.utils.data import TensorDataset

from kutils.datasets.utils import train_val_test_split


def _arange_dataset(n=100):
    data = torch.arange(n)
    return TensorDataset(data, data)


def test_splits_are_disjoint_and_complete():
    train, val, test = train_val_test_split(_arange_dataset(100))
    train_idx = set(train.indices)
    val_idx = set(val.indices)
    test_idx = set(test.indices)
    assert train_idx & val_idx == set()
    assert train_idx & test_idx == set()
    assert val_idx & test_idx == set()
    assert len(train) + len(val) + len(test) == 100


def test_test_split_is_fixed_across_seeds():
    _, _, test_a = train_val_test_split(_arange_dataset(100), seed=1)
    _, _, test_b = train_val_test_split(_arange_dataset(100), seed=2)
    assert test_a.indices == test_b.indices


def test_train_split_varies_with_seed():
    train_a, _, _ = train_val_test_split(_arange_dataset(100), seed=1)
    train_b, _, _ = train_val_test_split(_arange_dataset(100), seed=2)
    assert train_a.indices != train_b.indices


def test_too_small_dataset_raises():
    with pytest.raises(ValueError):
        train_val_test_split(_arange_dataset(2), val_ratio=0.5, test_ratio=0.5)


@pytest.mark.parametrize(
    ("val_ratio", "test_ratio", "expected_lengths"),
    [
        (0.0, 0.1, (90, 0, 10)),
        (0.1, 0.0, (90, 10, 0)),
        (0.0, 0.0, (100, 0, 0)),
    ],
)
def test_zero_ratio_does_not_remove_samples(val_ratio, test_ratio, expected_lengths):
    splits = train_val_test_split(_arange_dataset(100), val_ratio=val_ratio, test_ratio=test_ratio)
    assert tuple(map(len, splits)) == expected_lengths
