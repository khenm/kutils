"""Custom adapter — the base for paper-registered adapters and simple
backends: `model(x)` -> global embedding. Subclass to translate a custom
architecture's output into `RepresentationOutput`."""

from __future__ import annotations

from typing import Any

from kutils.models.adapters.base import BaseAdapter, batch_tensor
from kutils.models.schemas import RepresentationOutput


class CustomAdapter(BaseAdapter):
    """Default behavior: treat the backend output as the global embedding.
    Papers subclass this for their own architectures."""

    def preprocess(self, samples: Any) -> dict[str, Any]:
        if self.processor is None:
            return samples  # type: ignore[return-value]
        return self.processor(samples)

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        return self.encode_tensor(batch_tensor(batch))
