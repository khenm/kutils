"""Checkpoint loading: state dicts from local files (torch or safetensors)
with key validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kutils.models.checkpoints.hashing import hash_checkpoint


def load_checkpoint(model: Any, path: str | Path, *, strict: bool = True) -> str:
    """Load a state dict into `model` in place and return the file's sha256.

    Accepts torch pickles (any extension) and `.safetensors`. `strict=True`
    raises on missing or unexpected keys — a checkpoint for a different
    architecture is a silent bug otherwise.
    """
    path = Path(path)
    digest = hash_checkpoint(path)

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state: Any = load_file(str(path))
    else:
        import torch

        state = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(state, dict) and set(state) == {"state_dict"}:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path}: expected a state dict, got {type(state).__name__}")

    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise ValueError(
            f"{path}: state dict does not match the model "
            f"(missing={sorted(missing)[:8]}, unexpected={sorted(unexpected)[:8]})"
        )
    return digest
