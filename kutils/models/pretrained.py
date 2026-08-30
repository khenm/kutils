"""Pretrained backbone (built through the model factory) plus a task head,
with the same checkpointing/HF-Hub behavior as a from-scratch model."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kutils.models.base import BaseModel
from kutils.models.factory import build_model
from kutils.models.schemas import ModelSpec


class PretrainedBackbone(BaseModel):
    """Pretrained encoder (via `build_model`) plus a linear head.

    The backbone is constructed through the model factory (provider loader +
    adapter), so any registered provider works. `freeze_backbone=True`
    freezes the backbone (probing/linear-eval)."""

    def __init__(
        self,
        model_spec: ModelSpec,
        output_dim: int,
        freeze_backbone: bool = False,
    ):
        if not isinstance(model_spec, ModelSpec):
            raise TypeError(f"model_spec must be a ModelSpec, got {type(model_spec).__name__}")
        super().__init__()
        self.adapter = build_model(model_spec)
        self.backbone: Any = self.adapter.model
        hidden_dim = self.adapter.model_info().embedding_dimension
        if hidden_dim is None:
            raise ValueError(
                f"Could not determine the embedding dimension for {model_spec.model_id!r}; "
                "set [model.capability] embedding_dimension/width in the spec."
            )
        self.head = nn.Linear(hidden_dim, output_dim)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter.encode_tensor(x).global_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode(x)
        return self.head(features)
