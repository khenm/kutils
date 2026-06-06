import torch
import torch.nn as nn


class LossRegistry:
    """Registry of named loss functions."""

    _registry: dict[str, nn.Module] = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "bce": nn.BCEWithLogitsLoss(),
        "cosine_embedding": nn.CosineEmbeddingLoss(),
    }

    @classmethod
    def get(cls, name: str) -> nn.Module:
        if name not in cls._registry:
            raise KeyError(f"Unknown loss: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def register(cls, name: str, loss: nn.Module) -> None:
        cls._registry[name] = loss


class MultiLoss(nn.Module):
    """Weighted sum of multiple loss functions."""

    def __init__(self, losses: dict[str, tuple[nn.Module, float]]):
        super().__init__()
        self.losses = nn.ModuleDict({name: loss for name, (loss, _) in losses.items()})
        self.weights = {name: weight for name, (_, weight) in losses.items()}

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.tensor(0.0, device=outputs.device)
        components: dict[str, float] = {}
        for name in self.losses:
            value = self.losses[name](outputs, targets) * self.weights[name]
            total = total + value
            components[name] = value.item()
        return total, components
