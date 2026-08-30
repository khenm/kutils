"""CNN adapter — spatial feature maps."""

from __future__ import annotations

from typing import Any

from kutils.models.adapters.base import BaseAdapter, batch_tensor
from kutils.models.schemas import RepresentationOutput


class CNNAdapter(BaseAdapter):
    """Backend conventions: `model(x)` -> spatial feature map
    `[B, C, H, W]`. Global embedding = mean-pooled map; the spatial map is
    kept in `spatial_features` (structure preserved, never erased)."""

    def preprocess(self, samples: Any) -> dict[str, Any]:
        if self.processor is None:
            raise NotImplementedError("This backend has no processor; use encode_tensor")
        return self.processor(images=samples, return_tensors="pt")

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        features = self.model(batch_tensor(batch))
        if features.ndim == 4:
            return RepresentationOutput(
                global_embedding=features.mean(dim=(2, 3)),
                spatial_features=features,
            )
        return RepresentationOutput(global_embedding=features)
