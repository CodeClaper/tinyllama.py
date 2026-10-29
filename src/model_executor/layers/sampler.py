import torch
from torch import nn


class Sampler(nn.Module):
    """Samples the next tokens form the model's outputs."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    
    def forwad(
        self,
        embedding: torch.Tensor,
        hidden_stats: torch.Tensor,
    ):
        pass
