"""Tests for provider loaders.

Backends are optional and generally not installed in the base test
environment, so the actionable-error path is what runs here; tests skip when
a backend happens to be installed. Real-download paths are
`@pytest.mark.integration` (off by default).
"""

import importlib.util
import sys
import types

import pytest
import torch
import torch.nn as nn

from kutils.models.providers.huggingface import load_huggingface
from kutils.models.providers.local import load_local
from kutils.models.providers.open_clip import load_open_clip
from kutils.models.providers.timm import load_timm
from kutils.models.providers.torchvision import load_torchvision
from kutils.models.registry import get_model_registry
from kutils.models.schemas import ModelSpec

LOCAL_MODULE = "kutils_test_local_models"


class LocalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)

    def forward(self, x):
        return self.fc(x)


class ParamModel(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        self.fc = nn.Linear(dim, 2)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def fake_local_module():
    """A fake importable module hosting the local-provider test models."""
    module = types.ModuleType(LOCAL_MODULE)
    module.LocalModel = LocalModel
    module.ParamModel = ParamModel
    sys.modules[LOCAL_MODULE] = module
    yield module
    del sys.modules[LOCAL_MODULE]


def _skip_if_installed(name):
    if importlib.util.find_spec(name) is not None:
        pytest.skip(f"{name} is installed; error-path test not applicable")


def spec(**overrides):
    fields = {"provider": "x", "model_id": "x/model"}
    fields.update(overrides)
    return ModelSpec(**fields)


def test_default_registry_registers_providers():
    reg = get_model_registry()
    for name in ("huggingface", "timm", "open_clip", "torchvision", "local"):
        assert name in reg.providers


def test_huggingface_missing_backend_error():
    _skip_if_installed("transformers")
    with pytest.raises(ImportError, match="backbones"):
        load_huggingface(spec(provider="huggingface"))


def test_timm_missing_backend_error():
    _skip_if_installed("timm")
    with pytest.raises(ImportError, match="backbones"):
        load_timm(spec(provider="timm"))


def test_open_clip_missing_backend_error():
    _skip_if_installed("open_clip")
    with pytest.raises(ImportError, match="open_clip_torch"):
        load_open_clip(spec(provider="open_clip"))


def test_torchvision_missing_backend_error():
    _skip_if_installed("torchvision")
    with pytest.raises(ImportError, match="torchvision"):
        load_torchvision(spec(provider="torchvision"))


def test_local_requires_architecture():
    with pytest.raises(ValueError, match="architecture"):
        load_local(spec(provider="local"))


def test_local_loads_checkpoint(tmp_path, fake_local_module):
    path = tmp_path / "model.pt"
    torch.save(LocalModel().state_dict(), path)
    s = ModelSpec(
        provider="local",
        model_id="x",
        checkpoint=str(path),
        capability={"architecture": f"{LOCAL_MODULE}:LocalModel"},
    )
    model, processor = load_local(s)
    assert processor is None
    assert isinstance(model, LocalModel)
    assert model(torch.randn(1, 4)).shape == (1, 2)


def test_local_forwards_constructor_kwargs(fake_local_module):
    s = ModelSpec(
        provider="local",
        model_id="x",
        capability={"architecture": f"{LOCAL_MODULE}:ParamModel", "dim": 6},
    )
    model, _ = load_local(s)
    assert model(torch.randn(1, 6)).shape == (1, 2)


def test_local_reserved_keys_not_forwarded(fake_local_module):
    s = ModelSpec(
        provider="local",
        model_id="x",
        capability={
            "architecture": f"{LOCAL_MODULE}:ParamModel",
            "depth": 12,
            "freeze_backbone": True,
        },
    )
    model, _ = load_local(s)
    assert model.fc.in_features == 4  # default dim; reserved keys ignored


def test_local_rejected_kwargs_raise(fake_local_module):
    s = ModelSpec(
        provider="local",
        model_id="x",
        capability={"architecture": f"{LOCAL_MODULE}:ParamModel", "nope": 1},
    )
    with pytest.raises(ValueError, match="rejected constructor kwargs"):
        load_local(s)


@pytest.mark.integration
def test_integration_timm_forward():
    pytest.importorskip("timm")
    model, _ = load_timm(
        spec(provider="timm", model_id="resnet18", capability={"pretrained": False})
    )
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 512)


@pytest.mark.integration
def test_integration_huggingface_forward():
    pytest.importorskip("transformers")
    model, _ = load_huggingface(
        spec(provider="huggingface", model_id="hf-internal-testing/tiny-random-BertModel")
    )
    assert model.num_parameters() > 0
