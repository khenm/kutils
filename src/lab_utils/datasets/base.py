import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset with common utilities."""

    def __init__(self):
        self.data = []
        self.labels = []

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]

    @property
    def num_classes(self) -> int:
        return len(set(self.labels))

    @property
    def input_shape(self) -> tuple:
        if len(self.data) > 0:
            return tuple(self.data[0].shape)
        return ()
