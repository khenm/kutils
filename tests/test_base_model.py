"""Tests for lab_utils.models.base.BaseModel."""

import torch.nn as nn

from lab_utils.models.base import BaseModel


class TinyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(4, 2)

    def forward(self, x):
        return self.net(x)


def test_generate_model_card_includes_provenance():
    model = TinyModel()
    card = model.generate_model_card(
        run_name="260830-1402-baseline",
        config={"seed": 42, "max_epochs": 10, "dataset_name": "cifar10"},
        metrics={"test_accuracy": 0.9},
        git={"commit": "e116c651d3253c123603e8a73f260f71da127609", "dirty": False},
        lab_utils_commit="abc123def456",
    )
    text = card.text
    assert "Training provenance" in text
    assert "seed: 42" in text
    assert "test_accuracy: 0.9" in text
    assert "e116c651d3" in text
    assert "abc123def4" in text


def test_generate_model_card_degrades_gracefully():
    card = TinyModel().generate_model_card()
    assert card.text is not None
    assert "Training provenance" not in card.text
