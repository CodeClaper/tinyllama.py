import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """Rotary position embedding (RoPE) for query and key tensors."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        if rotary_dim > head_size:
            raise ValueError("rotary_dim cannot be larger than head_size")
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        inv_freq = 1.0 / (
            base ** (
                torch.arange(0, rotary_dim, 2, dtype=torch.float32)
                / rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = query.dtype
        positions = positions.to(device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().unsqueeze(1)
        sin = emb.sin().unsqueeze(1)

        def _apply(x: torch.Tensor) -> torch.Tensor:
            x_rot = x[..., : self.rotary_dim]
            x_pass = x[..., self.rotary_dim :]
            x_rot = x_rot.float() * cos + self._rotate_half(x_rot.float()) * sin
            return torch.cat((x_rot.to(dtype), x_pass), dim=-1)

        return _apply(query), _apply(key)
