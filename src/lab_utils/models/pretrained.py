"""Wrap a pretrained backbone (transformers or timm) as a lab_utils model.

Use this instead of hand-writing a from-scratch architecture when a paper
fine-tunes or probes an existing pretrained model. `PretrainedBackbone`
still inherits `save_pretrained`/`from_pretrained`/`push_to_hub` from
`BaseModel` (via `PyTorchModelHubMixin`), so checkpointing and the
`FabricTrainer` save/load path work identically to a from-scratch model —
only the backbone's own weights come from the Hub / timm instead of random
init.

Requires the optional `backbones` extra: `transformers` and/or `timm`.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from lab_utils.models.base import BaseModel


class PretrainedBackbone(BaseModel):
    """A pretrained encoder (HF `transformers` or `timm`) plus a linear head.

    Args:
        backbone: "transformers" to load via `AutoModel.from_pretrained`, or
            "timm" to load via `timm.create_model`.
        model_name: The Hub / timm model identifier, e.g.
            "distilbert-base-uncased" or "vit_base_patch16_224.augreg_in21k".
        output_dim: Size of the task head's output.
        freeze_backbone: If True, backbone params are frozen (only the head
            trains) — cheap probing/linear-eval instead of full fine-tuning.
        pretrained: For the timm backend, whether to load pretrained weights
            (always True for transformers, which has no random-init mode
            here).

    Example:
        >>> model = PretrainedBackbone(
        ...     backbone="timm", model_name="vit_base_patch16_224.augreg_in21k",
        ...     output_dim=10, freeze_backbone=True,
        ... )
        >>> model.save_pretrained("checkpoints/run/final")  # from BaseModel
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
