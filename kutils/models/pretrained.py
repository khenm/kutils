""" "Pretrained backbone (transformers/timm) plus a task head, with the same
checkpointing/HF-Hub behavior as a from-scratch model. Requires the
`backbones` extra."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from kutils.models.base import BaseModel


class PretrainedBackbone(BaseModel):
    """Pretrained encoder (transformers/timm) plus a linear head.

    `freeze_backbone=True` freezes the backbone (probing/linear-eval).
    """

    def __init__(
        self,
        backbone: str,
        model_name: str,
        output_dim: int,
        freeze_backbone: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_kind = backbone
        self.model_name = model_name
        self.backbone: Any  # set in the branches below (duck-typed per backend)

        hidden_dim: int
        if backbone == "transformers":
            from transformers import AutoModel

            self.backbone = AutoModel.from_pretrained(model_name)
            hidden_dim = int(self.backbone.config.hidden_size)
        elif backbone == "timm":
            import timm

            self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
            hidden_dim = int(self.backbone.num_features)
        else:
            raise ValueError(f"Unknown backbone kind: {backbone!r} (use 'transformers' or 'timm')")

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Linear(hidden_dim, output_dim)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone_kind == "transformers":
            # Assumes token-classification-style input; callers with a
            # different modality should encode upstream and pass
            # inputs_embeds, or subclass and override `_encode`.
            out = self.backbone(x)
            return out.last_hidden_state[:, 0]  # [CLS]-style pooling
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._encode(x)
        return self.head(features)
