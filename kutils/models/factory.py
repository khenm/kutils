"""Model factory: construct an adapted model from a `ModelSpec`.

The factory only wires a provider loader to an adapter; it contains no
experimental logic."""

from __future__ import annotations

from kutils.models.registry import ModelRegistry, get_model_registry
from kutils.models.schemas import ModelSpec, RepresentationModel

# Adapter family defaulted per provider when the spec doesn't say.
_FAMILY_BY_PROVIDER: dict[str, str] = {
    "timm": "vision_transformer",
    "torchvision": "cnn",
    "open_clip": "multimodal",
    "local": "custom",
}


def resolve_family(spec: ModelSpec) -> str:
    """The adapter family for a spec: an explicit `family` wins, else a
    provider default (huggingface picks by modality — text models are text
    transformers, everything else is a vision transformer)."""
    if spec.family:
        return spec.family
    if spec.provider == "huggingface":
        return "text_transformer" if spec.modality == "text" else "vision_transformer"
    try:
        return _FAMILY_BY_PROVIDER[spec.provider]
    except KeyError:
        raise ValueError(
            f"Provider {spec.provider!r} has no default adapter family; "
            "set [model] family in the spec."
        ) from None


def build_model(spec: ModelSpec, *, registry: ModelRegistry | None = None) -> RepresentationModel:
    """Construct + adapt a model from a spec: provider loader -> adapter."""
    if not isinstance(spec, ModelSpec):
        raise TypeError(f"Expected a ModelSpec, got {type(spec).__name__}")
    reg = registry or get_model_registry()
    loader = reg.get_provider(spec.provider)
    model, processor = loader(spec)
    adapter_factory = reg.get_adapter(resolve_family(spec))
    return adapter_factory(model, processor, spec)
