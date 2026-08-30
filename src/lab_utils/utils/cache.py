"""Generic artifact caching.

Save and load arbitrary objects to disk, and memoize expensive computations
keyed by whatever inputs determine their result. Not tied to metrics, plots,
or any particular research domain: use it for anything worth computing once
and reusing later — a distance matrix, a preprocessed split, a fitted
decomposition, a training-metrics history, a model's intermediate outputs.

Typical use:

    from lab_utils.utils.cache import cached

    def _compute():
        return expensive_pairwise_distances(embeddings)

    dist = cached(
        {"dataset": config.dataset_name, "metric": "cosine"},
        _compute,
        cache_dir=run_output_dir / "cache",
        name="pairwise_dist",
        ext=".npy",
    )
"""

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
    """Short, stable hash of the given inputs.

    Inputs must survive `json.dumps(..., default=str)` — numbers, strings,
    bools, None, lists/tuples, dicts, and anything with a meaningful
    `str()` (e.g. Path). Positional-arg order matters; keyword-arg order
    does not.

    Callers are responsible for including every input that affects the
    result: an incomplete fingerprint causes a silent, incorrect cache hit
    later.
    """
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
    """Compute once, reuse forever.

    `key` is hashed via `fingerprint` into a short digest; the result lives
    at `cache_dir/<name>-<digest><ext>`. On a hit, the file is loaded and
    returned without calling `compute_fn`. On a miss, `compute_fn()` runs,
    its result is saved, and then returned.

    `key` should be everything that determines the result (config values,
    data identifiers, hyperparameters) — see the warning on `fingerprint`.

    Pick `ext` to match what `compute_fn` returns: ".npy" for a single
    array, ".npz" for a dict of arrays, ".json" for plain data, ".pkl"
    (default) for anything else picklable.
    """
    digest = fingerprint(key)
    path = Path(cache_dir) / f"{name}-{digest}{ext}"
    if path.exists():
        return load_artifact(path)
    result = compute_fn()
    save_artifact(path, result)
    return result
