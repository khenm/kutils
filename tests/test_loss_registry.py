"""Tests for LossRegistry registration and isolation."""

import pytest
import torch.nn as nn

from kutils.losses.base import LossRegistry


def test_get_default_loss():
    assert isinstance(LossRegistry.get("cross_entropy"), nn.CrossEntropyLoss)


def test_register_and_get():
    LossRegistry.register("custom", nn.MSELoss())
    assert isinstance(LossRegistry.get("custom"), nn.MSELoss)


def test_unknown_loss_raises():
    with pytest.raises(KeyError):
        LossRegistry.get("does_not_exist")


def test_clear_removes_defaults():
    LossRegistry.clear()
    with pytest.raises(KeyError):
        LossRegistry.get("cross_entropy")


def test_reset_restores_defaults_and_drops_custom():
    LossRegistry.register("custom", nn.MSELoss())
    LossRegistry.clear()
    LossRegistry.reset()
    assert isinstance(LossRegistry.get("cross_entropy"), nn.CrossEntropyLoss)
    with pytest.raises(KeyError):
        LossRegistry.get("custom")
