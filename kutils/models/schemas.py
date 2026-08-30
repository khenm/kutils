"""Model-spec schemas: declarative model description (`ModelSpec`), scientific
metadata (`ModelInfo`), the uniform output every adapter returns
(`RepresentationOutput`), and the adapter contract (`RepresentationModel`).

A model spec is a single strict TOML file (see `ModelSpec.from_toml`):
unknown keys raise, mirroring the research-lab template config philosophy.
"""

from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from kutils.utils.cache import fingerprint

_SPEC_FIELDS = frozenset(
    {
        "name",
        "provider",
        "model_id",
        "family",
        "checkpoint",
        "revision",
        "dtype",
        "modality",
        "output",
        "trust_remote_code",
    }
)

_CAPABILITY_SCALARS = (int, float, str, bool)


@dataclass(frozen=True)
class ModelSpec:
    """Declarative description of a model.

    `provider` decides **how to build** the model, `model_id` decides
    **which** model it is, `output` decides **how it will be used** — the
    three never collapse into one field. `capability` holds raw measures
    (depth, width, parameters, ...) verbatim; derived ordering is the
    experiment's business, not the spec's.
    """

    provider: str
    model_id: str
    name: str | None = None
    family: str | None = None
    checkpoint: str | None = None
    revision: str | None = None
    dtype: str | None = None
    modality: str | None = None
    output: str | None = None
    trust_remote_code: bool = False
    capability: dict[str, int | float | str | bool] = field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path) -> ModelSpec:
        """Parse a strict spec file. Only a `[model]` table is allowed
        (plus `[model.capability]`); unknown keys and missing required keys
        raise ValueError."""
        path = Path(path)
        with open(path, "rb") as f:
            data = tomllib.load(f)

        if set(data) - {"model"}:
            raise ValueError(
                f"{path}: unknown top-level section(s) {sorted(set(data) - {'model'})} "
                "(expected only [model])"
            )
        model = data.get("model")
        if not isinstance(model, dict):
            raise ValueError(f"{path}: missing required [model] section")

        for key in model:
            if key not in _SPEC_FIELDS and key != "capability":
                raise ValueError(
                    f"Unknown model spec key {key!r} in {path}. Valid keys: {sorted(_SPEC_FIELDS)}"
                )
        missing = [key for key in ("provider", "model_id") if key not in model]
        if missing:
            raise ValueError(f"{path}: missing required key(s): {missing}")

        capability = model.get("capability", {})
        if not isinstance(capability, dict):
            raise ValueError(f"{path}: [model.capability] must be a table")
        for key, value in capability.items():
            if not isinstance(value, _CAPABILITY_SCALARS):
                raise ValueError(
                    f"{path}: capability {key!r} must be a scalar "
                    f"(int/float/str/bool), got {type(value).__name__}"
                )

        return cls(
            name=model.get("name", path.stem),
            provider=model["provider"],
            model_id=model["model_id"],
            family=model.get("family"),
            checkpoint=model.get("checkpoint"),
            revision=model.get("revision"),
            dtype=model.get("dtype"),
            modality=model.get("modality"),
            output=model.get("output"),
            trust_remote_code=model.get("trust_remote_code", False),
            capability=capability,
        )

    def digest(self) -> str:
        """Short, stable hash of the full spec — the key for any cache that
        depends on the model definition."""
        return fingerprint(self.__dict__)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ModelInfo:
    """Scientific + technical metadata about an adapted model.

    Raw capability measures are preserved verbatim in
    `capability_variables`; any derived capability ordering lives in
    `derived_order` and is the experiment's justification — never guessed
    from model names like "base"/"large".
    """

    model_id: str
    provider: str
    architecture: str | None = None
    modality: str | None = None
    objective_class: str | None = None
    output: str | None = None
    parameter_count: int | None = None
    embedding_dimension: int | None = None
    pretraining_dataset: str | None = None
    checkpoint_id: str | None = None
    revision: str | None = None
    capability_variables: dict[str, Any] = field(default_factory=dict)
    derived_order: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class RepresentationOutput:
    """Uniform output of any adapted model.

    Standardizes *access* without erasing structure: each field is populated
    only when the model actually produces that kind of representation, so
    transformers (tokens), CNNs (spatial maps) and pooled backbones stay
    distinguishable. Pooling/normalization into a single comparison object
    is a later, explicit, scientific decision — never made here.
    """

    global_embedding: Tensor | None = None
    token_embeddings: Tensor | None = None
    spatial_features: Tensor | None = None
    logits: Tensor | None = None
    layer_outputs: dict[str, Tensor] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class RepresentationModel(Protocol):
    """The adapter contract: analysis and training code only ever sees this,
    never raw backend outputs."""

    model: Any
    """The wrapped backend module (duck-typed per backend)."""

    def preprocess(self, samples: Any) -> dict[str, Any]:
        """Turn semantic samples (PIL images, strings, ...) into a
        model-compatible batch dict."""
        ...

    def encode(
        self, batch: dict[str, Any], layers: list[int] | None = None
    ) -> RepresentationOutput:
        """Run the model on a preprocessed batch and return the uniform
        representation output. `layers` optionally requests layer outputs."""
        ...

    def model_info(self) -> ModelInfo:
        """Scientific + technical metadata about this model."""
        ...
