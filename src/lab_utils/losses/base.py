import torch.nn as nn


class LossRegistry:
    """Registry of named loss functions: `register` adds, `clear` empties
    (test isolation), `reset` restores the defaults."""

    _defaults: dict[str, nn.Module] = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        "bce": nn.BCEWithLogitsLoss(),
        "cosine_embedding": nn.CosineEmbeddingLoss(),
    }
    _registry: dict[str, nn.Module] = dict(_defaults)

    @classmethod
    def get(cls, name: str) -> nn.Module:
        if name not in cls._registry:
            raise KeyError(f"Unknown loss: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name]

    @classmethod
    def register(cls, name: str, loss: nn.Module) -> None:
        cls._registry[name] = loss

    @classmethod
    def clear(cls) -> None:
        """Remove every registered loss."""
        cls._registry.clear()

    @classmethod
    def reset(cls) -> None:
        """Restore the default loss set."""
        cls._registry = dict(cls._defaults)
