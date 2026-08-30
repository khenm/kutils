"""Shared test fixtures."""

import pytest

from kutils.losses.base import LossRegistry


@pytest.fixture(autouse=True)
def isolated_loss_registry():
    """Reset LossRegistry before and after each test."""
    LossRegistry.reset()
    yield
    LossRegistry.reset()
