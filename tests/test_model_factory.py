"""Tests for the model factory: provider -> adapter wiring."""

import pytest
import torch
import torch.nn as nn

from kutils.models.factory import build_model, resolve_family
from kutils.models.registry import ModelRegistry
from kutils.models.schemas import ModelSpec, RepresentationModel


class StubOutput:
    def __init__(self, hidden):
        self.last_hidden_state = torch.randn(2, 5, hidden)


class StubModel(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.config = type("Config", (), {"hidden_size": hidden})()
        self._p = nn.Parameter(torch.randn(hidden, hidden))

    def forward(self, x, **kwargs):
        return StubOutput(self.config.hidden_size)


def stub_loader(spec):
    return StubModel(), None


def make_spec(**overrides):
    fields = {"provider": "stub", "model_id": "stub/model", "family": "text_transformer"}
    fields.update(overrides)
    return ModelSpec(**fields)


def test_build_model_wires_provider_and_adapter():
    reg = ModelRegistry()
    reg.register_provider("stub", stub_loader)
    adapter = build_model(make_spec(), registry=reg)
    assert isinstance(adapter, RepresentationModel)
    out = adapter.encode_tensor(torch.randint(0, 10, (2, 5)))
    assert out.global_embedding.shape == (2, 8)


def test_unknown_provider_raises():
    with pytest.raises(KeyError, match="Unknown model provider"):
        build_model(make_spec(provider="nope"))


def test_type_error_on_non_spec():
    with pytest.raises(TypeError, match="ModelSpec"):
        build_model("not a spec")  # type: ignore[arg-type]


def test_resolve_family_defaults():
    assert resolve_family(make_spec(family=None, provider="timm")) == "vision_transformer"
    assert resolve_family(make_spec(family=None, provider="torchvision")) == "cnn"
    assert resolve_family(make_spec(family=None, provider="open_clip")) == "multimodal"
    assert resolve_family(make_spec(family=None, provider="local")) == "custom"
    assert (
        resolve_family(make_spec(family=None, provider="huggingface", modality="text"))
        == "text_transformer"
    )
    assert (
        resolve_family(make_spec(family=None, provider="huggingface", modality="image"))
        == "vision_transformer"
    )
    assert resolve_family(make_spec(family="custom")) == "custom"


def test_unknown_provider_without_default_family_raises():
    with pytest.raises(ValueError, match="no default adapter family"):
        resolve_family(make_spec(family=None, provider="weird"))
