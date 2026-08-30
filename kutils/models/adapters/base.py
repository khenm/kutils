"""Base adapter: wraps a backend model + processor + spec behind the
`RepresentationModel` contract. Subclasses translate backend-specific output
conventions into `RepresentationOutput`; they never erase structure."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from kutils.models.schemas import ModelInfo, ModelSpec, RepresentationOutput


def select_layers(hidden_states: Any, layers: list[int] | None) -> dict[str, Tensor]:
    """Pick layer outputs from a HF-style `hidden_states` tuple.

    Index 0 is the embedding layer, so layer `i` (0-based, or negative from
    the end, `-1` = last) is returned as `layer_<i>`. Returns an empty dict
    when `layers` is None/empty or `hidden_states` is unavailable.
    """
    if not layers or hidden_states is None:
        return {}
    n = len(hidden_states)
    selected: dict[str, Tensor] = {}
    for i in layers:
        idx = i if i >= 0 else n + i
        if idx < 0 or idx >= n:
            raise IndexError(f"Layer index {i} out of range (have {n} hidden states)")
        selected[f"layer_{i}"] = hidden_states[idx]
    return selected


def batch_tensor(batch: dict[str, Any]) -> Tensor:
    """The single input tensor of a preprocessed batch, preferring
    conventional keys (pixel_values / images / image)."""
    for key in ("pixel_values", "images", "image"):
        value = batch.get(key)
        if torch.is_tensor(value):
            return value
    tensors = [v for v in batch.values() if torch.is_tensor(v)]
    if len(tensors) == 1:
        return tensors[0]
    if not tensors:
        raise ValueError("Batch contains no tensor input")
    raise ValueError(f"Batch contains multiple tensors; specify an input key: {sorted(batch)}")


class BaseAdapter:
    """Wraps a backend model + processor + spec behind the
    `RepresentationModel` contract."""

    def __init__(self, model: Any, processor: Any, spec: ModelSpec):
        self.model = model
        self.processor = processor
        self.spec = spec

    # -- contract ---------------------------------------------------------

    def preprocess(self, samples: Any) -> dict[str, Any]:
        raise NotImplementedError

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        raise NotImplementedError

    def encode_tensor(self, x: Tensor) -> RepresentationOutput:
        """Raw-tensor path for training-style callers: tensor in, uniform
        output out. Default treats the backend output as the global
        embedding."""
        return RepresentationOutput(global_embedding=self.model(x))

    # -- metadata ---------------------------------------------------------

    def model_info(self) -> ModelInfo:
        raw = dict(self.spec.capability)
        return ModelInfo(
            model_id=self.spec.model_id,
            provider=self.spec.provider,
            architecture=raw.get("architecture"),
            modality=self.spec.modality,
            objective_class=raw.get("objective_class"),
            output=self.spec.output,
            parameter_count=self._count_parameters(),
            embedding_dimension=self._infer_embedding_dimension(raw),
            pretraining_dataset=raw.get("pretraining_dataset"),
            checkpoint_id=self.spec.checkpoint,
            revision=self.spec.revision,
            capability_variables=dict(self.spec.capability),
        )

    def _count_parameters(self) -> int | None:
        parameters = getattr(self.model, "parameters", None)
        if parameters is None:
            return None
        return sum(p.numel() for p in parameters())

    def _infer_embedding_dimension(self, raw: dict[str, Any]) -> int | None:
        for key in ("embedding_dimension", "width"):
            value = raw.get(key)
            if isinstance(value, int) and value > 0:
                return value
        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int):
            return hidden_size
        num_features = getattr(self.model, "num_features", None)
        if isinstance(num_features, int):
            return num_features
        return None


class HFSequenceAdapter(BaseAdapter):
    """HF-style encoder path shared by the vision/text transformer adapters.

    Convention: `model(**batch)` (or `model(x)` on the raw-tensor path)
    returns an object with `.last_hidden_state` (plus `.hidden_states` when
    requested); the global embedding is the first token. Subclasses only
    provide the `preprocess` convention (images vs text)."""

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        output = self.model(**batch, output_hidden_states=layers is not None)
        return self._encode_hf(output, layers)

    def encode_tensor(self, x: Tensor) -> RepresentationOutput:
        output = self.model(x)
        return self._encode_hf(output, None)

    def _encode_hf(self, output: Any, layers: list[int] | None) -> RepresentationOutput:
        hidden = output.last_hidden_state
        return RepresentationOutput(
            global_embedding=hidden[:, 0],
            token_embeddings=hidden,
            layer_outputs=select_layers(getattr(output, "hidden_states", None), layers),
        )
