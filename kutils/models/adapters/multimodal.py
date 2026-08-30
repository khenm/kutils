"""Multimodal (CLIP-style two-tower) adapter."""

from __future__ import annotations

from typing import Any

from kutils.models.adapters.base import BaseAdapter
from kutils.models.schemas import RepresentationOutput


class MultimodalAdapter(BaseAdapter):
    """Backend conventions: `encode_image(pixels)` / `encode_text(ids)`
    return pooled embeddings (global only; no token/spatial structure)."""

    def preprocess(self, samples: Any) -> dict[str, Any]:
        if self.processor is None:
            raise NotImplementedError("This backend has no processor; use encode")
        if isinstance(samples, dict):
            return self.processor(
                images=samples.get("image"), text=samples.get("text"), return_tensors="pt"
            )
        return self.processor(images=samples, return_tensors="pt")

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        if "pixel_values" in batch:
            embedding = self.model.encode_image(batch["pixel_values"])
            return RepresentationOutput(global_embedding=embedding)
        if "input_ids" in batch:
            embedding = self.model.encode_text(batch["input_ids"])
            return RepresentationOutput(global_embedding=embedding)
        raise ValueError(
            "Batch must contain 'pixel_values' (image) or 'input_ids' (text), "
            f"got keys: {sorted(batch)}"
        )
