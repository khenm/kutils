"""kutils top-level package.

Public names are lazy-loaded (PEP 562 module __getattr__) so that importing
a lightweight submodule — e.g. `kutils.style`, which only needs
matplotlib/numpy — doesn't force-import torch/lightning via this file.
Accessing `kutils.BaseModel` etc. still works exactly as before; it's
just resolved on first use instead of at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kutils.models.base import BaseModel
    from kutils.stats import bootstrap_ci, permutation_test, summarize
    from kutils.training import (
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
    "BaseModel": ("kutils.models.base", "BaseModel"),
    "FabricTrainer": ("kutils.training", "FabricTrainer"),
    "HubPushCallback": ("kutils.training", "HubPushCallback"),
    "StandardRecipe": ("kutils.training", "StandardRecipe"),
    "TrainingConfig": ("kutils.training", "TrainingConfig"),
    "bootstrap_ci": ("kutils.stats", "bootstrap_ci"),
    "permutation_test": ("kutils.stats", "permutation_test"),
    "summarize": ("kutils.stats", "summarize"),
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
