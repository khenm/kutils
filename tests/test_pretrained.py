"""Tests for kutils.models.pretrained.PretrainedBackbone (factory-based).

The backbone is built through the model factory with a stubbed provider, so
nothing is downloaded and no optional backend is required.
"""

from typing import Any, cast

import pytest
import torch
import torch.nn as nn

from kutils.models.pretrained import PretrainedBackbone
from kutils.models.registry import ModelRegistry
from kutils.models.schemas import ModelSpec


class StubOutput:
    def __init__(self, hidden):
        self.last_hidden_state = torch.randn(2, 5, hidden)


class StubBackbone(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = cast(Any, type("C", (), {"hidden_size": hidden})())
        self.encoder = nn.Linear(hidden, hidden)

    def forward(self, x, **kwargs):
        return StubOutput(self.config.hidden_size)


def stub_loader(spec):
    return StubBackbone(), None


def make_spec(**overrides):
    fields: dict[str, Any] = {
        "provider": "stub",
        "model_id": "stub/backbone",
        "family": "text_transformer",
    }
    fields.update(overrides)
    return ModelSpec(**fields)


@pytest.fixture
def stub_registry(monkeypatch):
    reg = ModelRegistry()
    reg.register_provider("stub", stub_loader)
    monkeypatch.setattr("kutils.models.registry._default_registry", reg)
    return reg


def test_type_error_on_non_spec():
    with pytest.raises(TypeError, match="ModelSpec"):
        PretrainedBackbone("not-a-spec", output_dim=10)  # type: ignore[arg-type]


def test_forward_shape(stub_registry):
    model = PretrainedBackbone(make_spec(), output_dim=10)
    out = model(torch.randn(2, 5, 8))
    assert out.shape == (2, 10)


def test_freeze_backbone_freezes_encoder_not_head(stub_registry):
    model = PretrainedBackbone(make_spec(), output_dim=4, freeze_backbone=True)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_unknown_provider_raises():
    with pytest.raises(KeyError, match="Unknown model provider"):
        PretrainedBackbone(ModelSpec(provider="nope", model_id="x"), output_dim=10)
