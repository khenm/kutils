""" "Generic artifact caching: save/load arbitrary objects to disk and
memoize expensive computations (`save_artifact`/`load_artifact`/`cached`)."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

SUPPORTED_SUFFIXES = frozenset({".npy", ".npz", ".json", ".pkl"})


def fingerprint(*args: Any, **kwargs: Any) -> str:
    """Short, stable hash of the given inputs (JSON-serializable, else
    `str()`). Include every input that affects the result — an incomplete
    fingerprint causes a silent, incorrect cache hit."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_artifact(path: str | Path, obj: Any) -> Path:
    """Save `obj` to `path`, choosing a serializer from the file suffix.

    Suffix -> expected `obj`:
        .npy  a single array-like (cast with np.asarray if not already one)
        .npz  a dict of array-likes
        .json anything JSON-serializable (falls back to str() for the rest)
        .pkl  anything else

    Creates parent directories as needed. Raises ValueError on an
    unsupported suffix or a .npz value that isn't a dict.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix

    if suffix == ".npy":
        np.save(path, obj if isinstance(obj, np.ndarray) else np.asarray(obj))
    elif suffix == ".npz":
        if not isinstance(obj, dict):
            raise ValueError(f".npz artifacts must be a dict of arrays, got {type(obj).__name__}")
        np.savez(path, **{k: np.asarray(v) for k, v in obj.items()})  # pyright: ignore[reportArgumentType]
    elif suffix == ".json":
        path.write_text(json.dumps(obj, indent=2, default=str))
    elif suffix == ".pkl":
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(
            f"Unsupported artifact suffix: {suffix!r} "
            f"(expected one of {sorted(SUPPORTED_SUFFIXES)})"
        )

    return path


def load_artifact(path: str | Path) -> Any:
    """Load an artifact previously written by `save_artifact`."""
    path = Path(path)
    suffix = path.suffix

    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {k: data[k] for k in data.files}
    if suffix == ".json":
        return json.loads(path.read_text())
    if suffix == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(
        f"Unsupported artifact suffix: {suffix!r} (expected one of {sorted(SUPPORTED_SUFFIXES)})"
    )


def cached(
    key: Any,
    compute_fn: Callable[[], Any],
    *,
    cache_dir: str | Path,
    name: str,
    ext: str = ".pkl",
) -> Any:
    """Compute once, reuse forever: on a hit, load `cache_dir/<name>-<digest><ext>`
    and return without calling `compute_fn`; on a miss, run and cache it. `key`
    must include everything that determines the result (see `fingerprint`)."""

    digest = fingerprint(key)
    path = Path(cache_dir) / f"{name}-{digest}{ext}"
    if path.exists():
        return load_artifact(path)
    result = compute_fn()
    save_artifact(path, result)
    return result
