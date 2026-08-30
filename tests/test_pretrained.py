"""Tests for lab_utils.models.pretrained.PretrainedBackbone.

Backend-dependent tests skip when the optional `backbones` extra
(transformers/timm) isn't installed; the transformers branch is tested with
a stubbed AutoModel so nothing is downloaded.
"""

import pytest
import torch
import torch.nn as nn

from lab_utils.models.pretrained import PretrainedBackbone


def test_unknown_backbone_raises():
    with pytest.raises(ValueError, match="Unknown backbone"):
        PretrainedBackbone(backbone="nope", model_name="x", output_dim=10)


def test_timm_backbone_forward():
    pytest.importorskip("timm")
    model = PretrainedBackbone(
        backbone="timm", model_name="resnet18", output_dim=10, pretrained=False
    )
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 10)


def test_transformers_backbone_forward_with_stub(monkeypatch):
    pytest.importorskip("transformers")

    class StubConfig:
        hidden_size = 32

    class StubOutput:
        last_hidden_state = torch.randn(2, 5, 32)

    class StubModel(nn.Module):
        config = StubConfig()

        def forward(self, x):
            return StubOutput()

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda name: StubModel())
    model = PretrainedBackbone(backbone="transformers", model_name="stub", output_dim=10)
    out = model(torch.randn(2, 5, 32))
    assert out.shape == (2, 10)


def test_freeze_backbone_freezes_encoder_not_head(monkeypatch):
    pytest.importorskip("transformers")

    class StubConfig:
        hidden_size = 8

    class StubOutput:
        last_hidden_state = torch.randn(2, 3, 8)

    class StubModel(nn.Module):
        config = StubConfig()

        def forward(self, x):
            return StubOutput()

    monkeypatch.setattr("transformers.AutoModel.from_pretrained", lambda name: StubModel())
    model = PretrainedBackbone(
        backbone="transformers", model_name="stub", output_dim=4, freeze_backbone=True
    )
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
