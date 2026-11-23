

from typing import Optional

import torch
from torch import nn

from cscr import norm as _C


class RMSNorm(nn.Module):
    """
    Root mean square normalization.

    Computes x -> w * x / sqrt(E[x^2] + eps) where w is the learned weight.
    Refer to https://arxiv.org/abs/1910.07467
    """

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
        residual: Optional[torch.Tensor] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is not None:
            _C.fused_add_rms_norm(x, residual, self.weight, self.eps)
            return x, residual
        return _C.rms_norm(x, self.weight, self.eps)
    

