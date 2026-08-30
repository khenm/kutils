"""Tests for checkpoint hashing and loading."""

import pytest
import torch
import torch.nn as nn

from kutils.models.checkpoints import hash_checkpoint, load_checkpoint


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_hash_checkpoint_stable(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(Tiny().state_dict(), path)
    assert hash_checkpoint(path) == hash_checkpoint(path)
    assert len(hash_checkpoint(path)) == 64


def test_load_checkpoint_roundtrip(tmp_path):
    path = tmp_path / "model.pt"
    model = Tiny()
    state = model.state_dict()
    torch.save(state, path)

    digest = load_checkpoint(model, path)
    assert digest == hash_checkpoint(path)
    assert torch.allclose(model.fc.weight, state["fc.weight"])
    assert torch.allclose(model.fc.bias, state["fc.bias"])


def test_load_checkpoint_strict_rejects_unexpected_keys(tmp_path):
    path = tmp_path / "model.pt"
    state = Tiny().state_dict()
    state["extra"] = torch.randn(1)
    torch.save(state, path)

    with pytest.raises(ValueError, match="does not match the model"):
        load_checkpoint(Tiny(), path)


def test_load_checkpoint_non_strict_allows_extra_keys(tmp_path):
    path = tmp_path / "model.pt"
    state = Tiny().state_dict()
    state["extra"] = torch.randn(1)
    torch.save(state, path)

    model = Tiny()
    load_checkpoint(model, path, strict=False)
    assert model.fc.weight.shape == (2, 4)
