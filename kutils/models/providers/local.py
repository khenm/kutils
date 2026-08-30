"""Local provider: build a model from a dotted path and load a checkpoint.

The architecture is the paper's own, referenced as a dotted import path
`<module>:<ClassName>` in the spec's capability table (`architecture`);
remaining capability entries are passed to the constructor as keyword
arguments, so parametrized architectures work without a paper-side
provider. The checkpoint is a local state-dict file loaded with strict key
validation.
"""

from __future__ import annotations

import importlib
from typing import Any

from kutils.models.checkpoints import load_checkpoint
from kutils.models.schemas import ModelSpec

# Capability keys consumed by kutils itself (metadata / training flags);
# they are never forwarded to the model constructor.
_RESERVED_CAPABILITY_KEYS = frozenset(
    {
        "architecture",
        "depth",
        "width",
        "parameters",
        "pretraining_samples",
        "embedding_dimension",
        "objective_class",
        "pretraining_dataset",
        "pretrained",
        "freeze_backbone",
    }
)


def load_local(spec: ModelSpec) -> tuple[Any, None]:
    """`(spec) -> (model, None)`.

    Requires `capability.architecture = "<module>:<ClassName>"` (importable
    from the paper's environment); every other capability entry (minus the
    reserved metadata keys) is passed to the constructor. `checkpoint` is a
    local state-dict file, matched strictly against the model keys.
    """
    architecture = spec.capability.get("architecture")
    if not isinstance(architecture, str) or ":" not in architecture:
        raise ValueError(
            "provider='local' requires [model.capability] architecture = '<module>:<ClassName>'"
        )
    module_name, _, class_name = architecture.partition(":")
    try:
        model_cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Could not resolve architecture {architecture!r}: {exc}") from exc

    constructor_kwargs = {
        key: value for key, value in spec.capability.items() if key not in _RESERVED_CAPABILITY_KEYS
    }
    try:
        model = model_cls(**constructor_kwargs)
    except TypeError as exc:
        raise ValueError(
            f"Architecture {architecture!r} rejected constructor kwargs "
            f"{sorted(constructor_kwargs)}: {exc}"
        ) from exc

    if spec.checkpoint:
        load_checkpoint(model, spec.checkpoint)
    model.eval()
    return model, None
