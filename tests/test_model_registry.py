"""Tests for kutils.models.registry."""

import pytest

from kutils.models.registry import ModelRegistry, get_model_registry


def loader(spec):
    return object(), None


def test_register_and_get_provider():
    reg = ModelRegistry()
    reg.register_provider("stub", loader)
    assert reg.get_provider("stub") is loader


def test_unknown_provider_raises():
    with pytest.raises(KeyError, match="Unknown model provider"):
        get_model_registry().get_provider("nope")


def test_invalid_provider_registration_raises():
    reg = ModelRegistry()
    with pytest.raises(ValueError, match="non-empty string"):
        reg.register_provider("", loader)
    with pytest.raises(ValueError, match="must be callable"):
        reg.register_provider("stub", "not-callable")  # type: ignore[arg-type]


def test_builtin_families_available():
    reg = ModelRegistry()
    for family in ("vision_transformer", "text_transformer", "multimodal", "cnn", "custom"):
        assert family in reg.families


def test_register_and_get_adapter():
    reg = ModelRegistry()
    reg.register_adapter("my_family", lambda m, p, s: m)
    assert reg.get_adapter("my_family") is not None


def test_reset_drops_custom_keeps_builtins():
    reg = ModelRegistry()
    reg.register_provider("stub", loader)
    reg.register_adapter("my_family", lambda m, p, s: m)
    reg.reset()
    with pytest.raises(KeyError):
        reg.get_provider("stub")
    with pytest.raises(KeyError):
        reg.get_adapter("my_family")
    assert "vision_transformer" in reg.families  # built-ins restored
