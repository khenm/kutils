"""Local provider: build a model from a dotted path and load a checkpoint.

The architecture is the paper's own, referenced as a dotted import path
`<module>:<ClassName>` in the spec's capability table (`architecture`); the
checkpoint is a local state-dict file loaded with strict key validation.
"""

from __future__ import annotations

import importlib
from typing import Any

from kutils.models.checkpoints import load_checkpoint
from kutils.models.schemas import ModelSpec


def load_local(spec: ModelSpec) -> tuple[Any, None]:
    """`(spec) -> (model, None)`.

    Requires `capability.architecture = "<module>:<ClassName>"` (importable
    from the paper's environment) and optionally `checkpoint` (a local
    state-dict file, matched strictly against the model keys).
    """
    architecture = spec.capability.get("architecture")
    if not isinstance(architecture, str) or ":" not in architecture:
        raise ValueError(
            "provider='local' requires [model.capability] architecture = "
            "'<module>:<ClassName>'"
        )
    module_name, _, class_name = architecture.partition(":")
    try:
        model_cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not resolve architecture {architecture!r}: {exc}") from exc

    model = model_cls()
    if spec.checkpoint:
        load_checkpoint(model, spec.checkpoint)
    model.eval()
    return model, None
