"""Tests for kutils.models.components (MLP, ResidualBlock)."""

import torch

from kutils.models.components import MLP, ResidualBlock


def test_mlp_forward_shape():
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4)
    assert model(torch.randn(3, 8)).shape == (3, 4)


def test_mlp_num_layers_and_dropout():
    model = MLP(in_dim=8, hidden_dim=16, out_dim=4, num_layers=3, dropout=0.5)
    model.train()
    out = model(torch.randn(3, 8))
    assert out.shape == (3, 4)
    assert model.training


def test_residual_block_forward_shape():
    block = ResidualBlock(dim=16)
    x = torch.randn(3, 16)
    out = block(x)
    assert out.shape == x.shape


def test_residual_block_is_residual():
    block = ResidualBlock(dim=16)
    # Zeroing the inner net weights makes the block ~identity (plus LayerNorm).
    with torch.no_grad():
        for p in block.net.parameters():
            p.zero_()
    x = torch.randn(3, 16)
    out = block(x)
    assert torch.allclose(out, block.norm(x), atol=1e-5)
