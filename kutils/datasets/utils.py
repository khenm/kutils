import torch
from torch.utils.data import Dataset, Subset


def train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    test_seed: int = 12345,
) -> tuple[Subset, Subset, Subset]:
    """Split into train/val/test (original index space). The test draw uses
    the fixed `test_seed` (same examples every run); train/val use the run
    `seed`."""
    n = len(dataset)  # pyright: ignore[reportArgumentType]  # torch Dataset isn't Sized in stubs
    test_size = max(1, int(n * test_ratio))
    val_size = max(1, int(n * val_ratio))
    train_size = n - test_size - val_size
    if train_size <= 0:
        raise ValueError("dataset too small for the requested val/test ratios")

    # Fixed test draw; the rest keeps a deterministic order for the next stage.
    test_gen = torch.Generator().manual_seed(test_seed)
    perm = torch.randperm(n, generator=test_gen)
    test_idx = perm[:test_size].tolist()
    rest_idx = perm[test_size:].tolist()

    # Seeded train/val draw over the rest.
    train_gen = torch.Generator().manual_seed(seed)
    perm_rest = torch.randperm(len(rest_idx), generator=train_gen)
    train_idx = [rest_idx[i] for i in perm_rest[:train_size].tolist()]
    val_idx = [rest_idx[i] for i in perm_rest[train_size:].tolist()]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )


def collate_list(batch: list) -> tuple[torch.Tensor, torch.Tensor]:
    """Default collate: stack tensors from list of (x, y) tuples."""
    xs, ys = zip(*batch, strict=True)
    return torch.stack(xs), torch.stack(ys)
