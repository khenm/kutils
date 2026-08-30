"""Model registries: named provider loaders and adapter families.

Both are plain extension points — papers register their own providers and
adapters (e.g. for their own architectures) without forking kutils. Mirrors
the `LossRegistry` pattern: a module-level default registry plus `reset()`
for test isolation.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from kutils.models.schemas import ModelSpec, RepresentationModel

ProviderLoader = Callable[[ModelSpec], tuple[Any, Any]]
"""`(spec) -> (backend_model, processor)` — construction only, never
extraction or evaluation."""

AdapterFactory = Callable[[Any, Any, ModelSpec], RepresentationModel]
"""`(backend_model, processor, spec) -> adapted model`."""


class ModelRegistry:
    """Registry of named provider loaders and adapter families."""

    def __init__(self) -> None:
        self._providers: dict[str, ProviderLoader] = {}
        self._adapters: dict[str, AdapterFactory] = {}
        self.register_defaults()

    # -- providers ---------------------------------------------------------

    def register_provider(self, name: str, loader: ProviderLoader) -> None:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Provider name must be a non-empty string, got {name!r}")
        if not callable(loader):
            raise ValueError(f"Provider loader for {name!r} must be callable")
        self._providers[name] = loader

    def get_provider(self, name: str) -> ProviderLoader:
        try:
            return self._providers[name]
        except KeyError:
            raise KeyError(
                f"Unknown model provider: {name!r}. Registered: {self.providers}"
            ) from None

    @property
    def providers(self) -> list[str]:
        return sorted(self._providers)

    # -- adapters ----------------------------------------------------------

    def register_adapter(self, family: str, factory: AdapterFactory) -> None:
        if not isinstance(family, str) or not family:
            raise ValueError(f"Adapter family must be a non-empty string, got {family!r}")
        if not callable(factory):
            raise ValueError(f"Adapter factory for {family!r} must be callable")
        self._adapters[family] = factory

    def get_adapter(self, family: str) -> AdapterFactory:
        try:
            return self._adapters[family]
        except KeyError:
            raise KeyError(
                f"Unknown adapter family: {family!r}. Registered: {self.families}"
            ) from None

    @property
    def families(self) -> list[str]:
        return sorted(self._adapters)

    # -- lifecycle ---------------------------------------------------------

    def register_defaults(self) -> None:
        """Register the built-in adapter families and provider loaders.
        Imported lazily to keep module imports cheap and cycle-free; the
        backends themselves are imported lazily inside each loader, so no
        optional package is required at import time."""
        from kutils.models.adapters.cnn import CNNAdapter
        from kutils.models.adapters.custom import CustomAdapter
        from kutils.models.adapters.multimodal import MultimodalAdapter
        from kutils.models.adapters.text_transformer import TextTransformerAdapter
        from kutils.models.adapters.vision_transformer import VisionTransformerAdapter

        self._adapters = {
            "vision_transformer": VisionTransformerAdapter,
            "text_transformer": TextTransformerAdapter,
            "multimodal": MultimodalAdapter,
            "cnn": CNNAdapter,
            "custom": CustomAdapter,
        }

        from kutils.models.providers.huggingface import load_huggingface
        from kutils.models.providers.local import load_local
        from kutils.models.providers.open_clip import load_open_clip
        from kutils.models.providers.timm import load_timm
        from kutils.models.providers.torchvision import load_torchvision

        self._providers = {
            "huggingface": load_huggingface,
            "timm": load_timm,
            "open_clip": load_open_clip,
            "torchvision": load_torchvision,
            "local": load_local,
        }

    def reset(self) -> None:
        """Drop paper registrations and restore built-ins (test isolation)."""
        self._providers.clear()
        self.register_defaults()


_default_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """The process-wide default registry."""
    return _default_registry


def register_provider(name: str, loader: ProviderLoader) -> None:
    _default_registry.register_provider(name, loader)


def register_adapter(family: str, factory: AdapterFactory) -> None:
    _default_registry.register_adapter(family, factory)


def reset_model_registry() -> None:
    _default_registry.reset()
