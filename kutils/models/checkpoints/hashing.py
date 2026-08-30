"""Checkpoint hashing: stable sha256 of checkpoint files for provenance and
cache keys."""

from __future__ import annotations

import hashlib
from pathlib import Path


def hash_checkpoint(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """sha256 hex digest of a checkpoint file (chunked, so large weights
    don't balloon memory)."""
    path = Path(path)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
