"""HuggingFace `transformers` provider (optional `backbones` extra)."""

from __future__ import annotations

from typing import Any

from kutils.models.schemas import ModelSpec


def load_huggingface(spec: ModelSpec) -> tuple[Any, Any]:
    """`(spec) -> (AutoModel, AutoProcessor)` from the Hub.

    `revision`, `trust_remote_code` and `dtype` (a torch dtype name such as
    "bfloat16") are forwarded. A model without a processor gets
    `processor=None`. The `backbones` extra (transformers) must be
    installed; the adapter family is chosen by `spec.family` (or modality).
    """
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "The 'huggingface' provider requires the optional `backbones` extra "
            "(transformers); install it to use this provider."
        ) from exc

    kwargs: dict[str, Any] = {}
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if spec.dtype:
        import torch

        dtype = getattr(torch, spec.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(
                f"Unknown dtype {spec.dtype!r} in spec; use a torch dtype name such as 'bfloat16'"
            )
        kwargs["torch_dtype"] = dtype

    model = AutoModel.from_pretrained(spec.model_id, **kwargs)
    try:
        processor = AutoProcessor.from_pretrained(spec.model_id, **kwargs)
    except Exception:  # noqa: BLE001 - some models have no processor
        processor = None
    return model, processor
