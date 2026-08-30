"""Vision transformer adapter — HF-style ViT or timm-style feature models."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from kutils.models.adapters.base import BaseAdapter, select_layers
from kutils.models.schemas import RepresentationOutput


class VisionTransformerAdapter(BaseAdapter):
    """Backend output conventions:

    - HF-style: `model(**batch)` -> object with `.last_hidden_state` (and
      `.hidden_states` when requested); global embedding = [CLS] token.
    - timm-style: `model(x)` -> pooled feature tensor (``num_classes=0``).

    `layers` is ignored for the timm-style branch (no hidden states).
    """

    def preprocess(self, samples: Any) -> dict[str, Any]:
        if self.processor is None:
            raise NotImplementedError("This backend has no processor; use encode_tensor")
        return self.processor(images=samples, return_tensors="pt")

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        output = self.model(**batch, output_hidden_states=layers is not None)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            return RepresentationOutput(
                global_embedding=hidden[:, 0],
                token_embeddings=hidden,
                layer_outputs=select_layers(getattr(output, "hidden_states", None), layers),
                metadata={"backend": "huggingface_style"},
            )
        return RepresentationOutput(global_embedding=output, metadata={"backend": "timm_style"})

    def encode_tensor(self, x: Tensor) -> RepresentationOutput:
        output = self.model(x)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            return RepresentationOutput(global_embedding=hidden[:, 0], token_embeddings=hidden)
        return RepresentationOutput(global_embedding=output)
