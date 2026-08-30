"""Text transformer adapter — HF-style encoder models."""

from __future__ import annotations

from typing import Any

from kutils.models.adapters.base import HFSequenceAdapter


class TextTransformerAdapter(HFSequenceAdapter):
    """Backend conventions: `model(**batch)` -> object with
    `.last_hidden_state`; global embedding = [CLS] token (index 0)."""

    def preprocess(self, samples: Any) -> dict[str, Any]:
        if self.processor is None:
            raise NotImplementedError("This backend has no processor; use encode_tensor")
        return self.processor(text=samples, return_tensors="pt", truncation=True, padding=True)
