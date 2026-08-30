"""OpenCLIP provider (optional `open_clip_torch` package, paper-owned)."""

from __future__ import annotations

from typing import Any

import torch

from kutils.models.schemas import ModelSpec


def load_open_clip(spec: ModelSpec) -> tuple[Any, Any]:
    """`(spec) -> (CLIP model, processor)`.

    `spec.checkpoint` is the pretrained tag (e.g. a LAION tag). The
    processor closure mirrors a HF-style processor: `processor(images=...,
    text=..., return_tensors="pt")` -> `{"pixel_values", "input_ids"}`. The
    optional `open_clip_torch` package must be installed by the paper that
    uses this provider — kutils never depends on it.
    """
    try:
        import open_clip
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'open_clip' provider requires the optional 'open_clip_torch' "
            "package; add it to your own pyproject when you use this provider."
        ) from exc

    model, _, _preprocess = open_clip.create_model_and_transforms(
        spec.model_id, pretrained=spec.checkpoint
    )
    preprocess: Any = _preprocess  # open_clip stubs type it as Compose; it is callable

    def processor(images=None, text=None, return_tensors="pt"):
        out: dict[str, Any] = {}
        if images is not None:
            batch = images if isinstance(images, (list, tuple)) else [images]
            out["pixel_values"] = torch.stack([preprocess(img) for img in batch])
        if text is not None:
            out["input_ids"] = open_clip.tokenize(text)
        return out

    return model, processor
