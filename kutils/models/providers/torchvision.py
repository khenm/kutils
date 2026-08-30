"""torchvision provider (optional `torchvision` package, paper-owned)."""

from __future__ import annotations

from typing import Any, cast

from kutils.models.schemas import ModelSpec


def load_torchvision(spec: ModelSpec) -> tuple[Any, None]:
    """`(spec) -> (torchvision model without its head, None)`.

    The classification head is stripped so the model outputs features
    (spatial maps for CNNs). `spec.checkpoint` may be "default" (the
    default pretrained weights). The optional `torchvision` package must be
    installed by the paper that uses this provider.
    """
    try:
        import torchvision
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'torchvision' provider requires the optional 'torchvision' "
            "package; add it to your own pyproject when you use this provider."
        ) from exc

    weights = None
    if spec.checkpoint and spec.checkpoint.lower() in ("default", "imagenet"):
        weights = cast(Any, torchvision.models.get_model_weights(spec.model_id)).DEFAULT
    model = torchvision.models.get_model(spec.model_id, weights=weights)

    import torch.nn as nn

    if hasattr(model, "fc"):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier"):
        model.classifier = nn.Identity()
    elif hasattr(model, "heads"):
        model.heads = nn.Identity()
    return model, None
