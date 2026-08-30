"""Tests for adapter families: backend conventions -> RepresentationOutput."""

import pytest
import torch
import torch.nn as nn

from kutils.models.adapters.base import batch_tensor, select_layers
from kutils.models.adapters.cnn import CNNAdapter
from kutils.models.adapters.multimodal import MultimodalAdapter
from kutils.models.adapters.text_transformer import TextTransformerAdapter
from kutils.models.adapters.vision_transformer import VisionTransformerAdapter
from kutils.models.schemas import ModelSpec


def spec(**overrides):
    base = {"provider": "stub", "model_id": "stub/model", "modality": "image"}
    base.update(overrides)
    return ModelSpec(**base)


class HFOutput:
    def __init__(self, hidden, n_layers):
        self.last_hidden_state = torch.randn(2, 5, hidden)
        self.hidden_states = [torch.randn(2, 5, hidden) for _ in range(n_layers)]


class HFModel(nn.Module):
    def __init__(self, hidden=8, n_layers=3):
        super().__init__()
        self.config = type("C", (), {"hidden_size": hidden})()
        self.n_layers = n_layers

    def forward(self, x=None, **kwargs):
        return HFOutput(self.config.hidden_size, self.n_layers)


class CLIPModel(nn.Module):
    def encode_image(self, pixels):
        return torch.randn(pixels.shape[0], 4)

    def encode_text(self, ids):
        return torch.randn(ids.shape[0], 4)


class CNNModel(nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 16, 7, 7)


class TimmModel(nn.Module):
    def forward(self, x):
        return torch.randn(x.shape[0], 8)


def test_text_transformer_encode_tensor():
    adapter = TextTransformerAdapter(HFModel(), None, spec(modality="text"))
    out = adapter.encode_tensor(torch.randint(0, 10, (2, 5)))
    assert out.global_embedding.shape == (2, 8)
    assert out.token_embeddings.shape == (2, 5, 8)


def test_text_transformer_layer_outputs():
    adapter = TextTransformerAdapter(HFModel(n_layers=3), None, spec(modality="text"))
    out = adapter.encode({"input_ids": torch.randint(0, 10, (2, 5))}, layers=[-1])
    assert "layer_-1" in out.layer_outputs


def test_vision_transformer_timm_style():
    adapter = VisionTransformerAdapter(TimmModel(), None, spec())
    out = adapter.encode_tensor(torch.randn(2, 3, 32, 32))
    assert out.global_embedding.shape == (2, 8)
    assert out.token_embeddings is None


def test_vision_transformer_hf_style():
    adapter = VisionTransformerAdapter(HFModel(), None, spec())
    out = adapter.encode_tensor(torch.randn(2, 3, 32, 32))
    assert out.global_embedding.shape == (2, 8)
    assert out.token_embeddings.shape == (2, 5, 8)


def test_cnn_spatial_and_pooled():
    adapter = CNNAdapter(CNNModel(), None, spec())
    out = adapter.encode({"images": torch.randn(2, 3, 32, 32)})
    assert out.spatial_features.shape == (2, 16, 7, 7)
    assert out.global_embedding.shape == (2, 16)


def test_multimodal_image_and_text():
    adapter = MultimodalAdapter(CLIPModel(), None, spec(modality="multimodal"))
    img = adapter.encode({"pixel_values": torch.randn(2, 3, 32, 32)})
    assert img.global_embedding.shape == (2, 4)
    txt = adapter.encode({"input_ids": torch.randint(0, 10, (2, 5))})
    assert txt.global_embedding.shape == (2, 4)
    with pytest.raises(ValueError, match="pixel_values"):
        adapter.encode({"foo": torch.randn(2)})


def test_model_info_infers_embedding_dimension():
    adapter = TextTransformerAdapter(HFModel(hidden=16), None, spec(modality="text"))
    info = adapter.model_info()
    assert info.embedding_dimension == 16
    assert info.parameter_count is not None
    assert info.model_id == "stub/model"


def test_select_layers_negative_index():
    hidden = [torch.randn(2, 3) for _ in range(4)]
    out = select_layers(hidden, [-1])
    assert "layer_-1" in out
    assert out["layer_-1"] is hidden[3]
    with pytest.raises(IndexError):
        select_layers(hidden, [-10])


def test_batch_tensor_prefers_conventional_key():
    batch = {"images": torch.randn(2), "labels": torch.tensor([1])}
    assert batch_tensor(batch).shape == (2,)
