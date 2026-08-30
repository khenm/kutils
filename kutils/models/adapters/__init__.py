"""Adapter families: translate backend output conventions into the uniform
`RepresentationOutput`. Adapters standardize *access* without erasing
structure — token sequences, spatial maps and pooled embeddings stay
distinguishable."""

from kutils.models.adapters.base import BaseAdapter, batch_tensor, select_layers
from kutils.models.adapters.cnn import CNNAdapter
from kutils.models.adapters.custom import CustomAdapter
from kutils.models.adapters.multimodal import MultimodalAdapter
from kutils.models.adapters.text_transformer import TextTransformerAdapter
from kutils.models.adapters.vision_transformer import VisionTransformerAdapter

__all__ = [
    "BaseAdapter",
    "CNNAdapter",
    "CustomAdapter",
    "MultimodalAdapter",
    "TextTransformerAdapter",
    "VisionTransformerAdapter",
    "batch_tensor",
    "select_layers",
]
