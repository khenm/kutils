import torch
from torch.utils.data import Dataset, random_split


def train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Randomly split a dataset into train and validation sets."""
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def collate_list(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Default collate: stack tensors from list of (x, y) tuples."""
    xs, ys = zip(*batch)
    return torch.stack(xs), torch.stack(ys)
