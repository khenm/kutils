"""Text transformer adapter — HF-style encoder models."""

from __future__ import annotations

from typing import Any

from torch import Tensor

from kutils.models.adapters.base import BaseAdapter, select_layers
from kutils.models.schemas import RepresentationOutput


class TextTransformerAdapter(BaseAdapter):
    """Backend conventions: `model(**batch)` -> object with
    `.last_hidden_state`; global embedding = [CLS] token (index 0)."""

    def preprocess(self, samples: Any) -> dict[str, Any]:
        if self.processor is None:
            raise NotImplementedError("This backend has no processor; use encode_tensor")
        return self.processor(text=samples, return_tensors="pt", truncation=True, padding=True)

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        output = self.model(**batch, output_hidden_states=layers is not None)
        hidden = output.last_hidden_state
        return RepresentationOutput(
            global_embedding=hidden[:, 0],
            token_embeddings=hidden,
            layer_outputs=select_layers(getattr(output, "hidden_states", None), layers),
        )

    def encode_tensor(self, x: Tensor) -> RepresentationOutput:
        output = self.model(x)  # tensor of input ids
        hidden = output.last_hidden_state
        return RepresentationOutput(global_embedding=hidden[:, 0], token_embeddings=hidden)
