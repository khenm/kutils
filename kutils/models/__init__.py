"""kutils.models: base model, pretrained backbone, and the model zoo
(specs, registry, factory, adapters, checkpoints)."""

from kutils.models.base import BaseModel
from kutils.models.factory import build_model
from kutils.models.pretrained import PretrainedBackbone
from kutils.models.registry import (
    ModelRegistry,
    get_model_registry,
    register_adapter,
    register_provider,
    reset_model_registry,
)
from kutils.models.schemas import (
    ModelInfo,
    ModelSpec,
    RepresentationModel,
    RepresentationOutput,
)

__all__ = [
    "BaseModel",
    "ModelInfo",
    "ModelRegistry",
    "ModelSpec",
    "PretrainedBackbone",
    "RepresentationModel",
    "RepresentationOutput",
    "build_model",
    "get_model_registry",
    "register_adapter",
    "register_provider",
    "reset_model_registry",
]
