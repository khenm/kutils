"""RNG seeding for reproducible runs."""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch
from lightning import seed_everything


def set_seed(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs (CPU + CUDA) and DataLoader workers,
    via Lightning's `seed_everything(workers=True)`. Call once at the top of
    an entry point."""
    seed_everything(seed, workers=True, verbose=False)


def capture_rng_state() -> dict[str, Any]:
    """Capture torch/NumPy/Python RNG state as a checkpoint-safe dict
    (numpy state as lists, so `weights_only=True` loads work)."""
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": _numpy_state_to_dict(np.random.get_state()),
        "random": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = {
            str(device): torch.cuda.get_rng_state(device)
            for device in range(torch.cuda.device_count())
        }
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore state from `capture_rng_state`; missing entries (e.g. no
    CUDA on the resuming machine) are skipped."""
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(_dict_to_numpy_state(state["numpy"]))
    if "random" in state:
        random.setstate(state["random"])
    if "cuda" in state and torch.cuda.is_available():
        for device, tensor in state["cuda"].items():
            try:
                torch.cuda.set_rng_state(tensor, device=int(device))
            except (IndexError, RuntimeError):
                # Different GPU count/index on the resuming machine: skip.
                continue


def _numpy_state_to_dict(state: tuple) -> dict[str, Any]:
    """Convert np.random.get_state() (ndarray inside) to a plain dict."""
    bit_generator, key, pos, has_gauss, cached_gaussian = state
    return {
        "bit_generator": bit_generator,
        "key": key.tolist(),
        "pos": int(pos),
        "has_gauss": bool(has_gauss),
        "cached_gaussian": float(cached_gaussian) if cached_gaussian is not None else None,
    }


def _dict_to_numpy_state(state: dict[str, Any]) -> tuple:
    """Inverse of `_numpy_state_to_dict`."""
    key = np.asarray(state["key"], dtype=np.uint32)
    return (
        state["bit_generator"],
        key,
        state["pos"],
        state["has_gauss"],
        state["cached_gaussian"],
    )
