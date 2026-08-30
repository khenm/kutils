"""lab_utils top-level package.

Public names are lazy-loaded (PEP 562 module __getattr__) so that importing
a lightweight submodule — e.g. `lab_utils.style`, which only needs
matplotlib/numpy — doesn't force-import torch/lightning via this file.
Accessing `lab_utils.BaseModel` etc. still works exactly as before; it's
just resolved on first use instead of at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lab_utils.models.base import BaseModel
    from lab_utils.stats import bootstrap_ci, permutation_test, summarize
    from lab_utils.training import (
        FabricTrainer,
        HubPushCallback,
        StandardRecipe,
        TrainingConfig,
    )

__all__ = [
    "BaseModel",
    "FabricTrainer",
    "HubPushCallback",
    "StandardRecipe",
    "TrainingConfig",
    "bootstrap_ci",
    "permutation_test",
    "summarize",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BaseModel": ("lab_utils.models.base", "BaseModel"),
    "FabricTrainer": ("lab_utils.training", "FabricTrainer"),
    "HubPushCallback": ("lab_utils.training", "HubPushCallback"),
    "StandardRecipe": ("lab_utils.training", "StandardRecipe"),
    "TrainingConfig": ("lab_utils.training", "TrainingConfig"),
    "bootstrap_ci": ("lab_utils.stats", "bootstrap_ci"),
    "permutation_test": ("lab_utils.stats", "permutation_test"),
    "summarize": ("lab_utils.stats", "summarize"),
}


def __getattr__(name: str):
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY.keys()))
