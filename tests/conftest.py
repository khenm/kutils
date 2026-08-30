"""Shared test fixtures."""

import pytest

from kutils.losses.base import LossRegistry
from kutils.models.registry import reset_model_registry


@pytest.fixture(autouse=True)
def isolated_loss_registry():
    """Reset LossRegistry before and after each test."""
    LossRegistry.reset()
    yield
    LossRegistry.reset()


@pytest.fixture(autouse=True)
def isolated_model_registry():
    """Reset the model registry before and after each test (drops paper
    registrations, restores built-in adapter families)."""
    reset_model_registry()
    yield
    reset_model_registry()
