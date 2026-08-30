"""timm provider (optional `backbones` extra)."""

from __future__ import annotations

from typing import Any

from kutils.models.schemas import ModelSpec


def load_timm(spec: ModelSpec) -> tuple[Any, None]:
    """`(spec) -> (timm model with no head, None)`.

    Pretrained weights are used unless the spec's capability table sets
    `pretrained = false`. The `backbones` extra (timm) must be installed.
    """
    try:
        import timm
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'timm' provider requires the optional `backbones` extra (timm); "
            "install it to use this provider."
        ) from exc

    pretrained = spec.capability.get("pretrained", True)
    if not isinstance(pretrained, bool):
        raise ValueError("capability 'pretrained' must be a bool (TOML `true`/`false`)")
    model = timm.create_model(spec.model_id, pretrained=pretrained, num_classes=0)
    return model, None
