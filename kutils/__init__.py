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
    from kutils.models.registry import ModelRegistry
    from kutils.models.schemas import (
        ModelInfo,
        ModelSpec,
        RepresentationModel,
        RepresentationOutput,
    )
    from kutils.stats import bootstrap_ci, permutation_test, summarize
    from kutils.training import (
        FabricTrainer,
        HubPushCallback,
        StandardRecipe,
        TaskRecipe,
        TrainingConfig,
    )

__all__ = [
    "BaseModel",
    "FabricTrainer",
    "HubPushCallback",
    "ModelInfo",
    "ModelRegistry",
    "ModelSpec",
    "PretrainedBackbone",
    "RepresentationModel",
    "RepresentationOutput",
    "StandardRecipe",
    "TaskRecipe",
    "TrainingConfig",
    "bootstrap_ci",
    "build_model",
    "permutation_test",
    "register_adapter",
    "register_provider",
    "summarize",
]

_LAZY: dict[str, tuple[str, str]] = {
    "BaseModel": ("kutils.models.base", "BaseModel"),
    "FabricTrainer": ("kutils.training", "FabricTrainer"),
    "HubPushCallback": ("kutils.training", "HubPushCallback"),
    "ModelInfo": ("kutils.models.schemas", "ModelInfo"),
    "ModelRegistry": ("kutils.models.registry", "ModelRegistry"),
    "ModelSpec": ("kutils.models.schemas", "ModelSpec"),
    "PretrainedBackbone": ("kutils.models.pretrained", "PretrainedBackbone"),
    "RepresentationModel": ("kutils.models.schemas", "RepresentationModel"),
    "RepresentationOutput": ("kutils.models.schemas", "RepresentationOutput"),
    "StandardRecipe": ("kutils.training", "StandardRecipe"),
    "TaskRecipe": ("kutils.training", "TaskRecipe"),
    "TrainingConfig": ("kutils.training", "TrainingConfig"),
    "bootstrap_ci": ("kutils.stats", "bootstrap_ci"),
    "build_model": ("kutils.models.factory", "build_model"),
    "permutation_test": ("kutils.stats", "permutation_test"),
    "register_adapter": ("kutils.models.registry", "register_adapter"),
    "register_provider": ("kutils.models.registry", "register_provider"),
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
