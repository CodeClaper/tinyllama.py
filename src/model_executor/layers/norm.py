

import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps


    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        out = torch.empty_like(x)
        return out;
    

