import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


class BaseModel(nn.Module, PyTorchModelHubMixin):
    """Base model with HF Hub integration.

    Every model gets save_pretrained / from_pretrained / push_to_hub for free.
    """

    def __init__(self):
        super().__init__()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
