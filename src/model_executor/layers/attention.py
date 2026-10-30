import torch
from torch import nn

from cscr import attention as _C


class PagedAttention(nn.Module):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:

        """PagedAttention decode forward.

        Args:
            query: ``[num_seqs, num_heads, head_dim]``.
            key_cache: ``[num_blocks, num_kv_heads, block_size, head_dim]``.
            value_cache: same shape as ``key_cache``.
            block_tables: ``[num_seqs, max_blocks_per_seq]`` int32.
            seq_lens: ``[num_seqs]`` int32.
            scale: softmax scale, usually ``head_dim ** -0.5``.

        Returns:
            ``[num_seqs, num_heads, head_dim]``.
        """
        return _C.paged_attention(
            query, key_cache, value_cache, block_tables, seq_lens, float(self.scale)
        )


class PagedAttentionWithRoPE(PagedAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
    ):
        super().__init__(num_heads, head_size, scale)
        self.rotary_dim = rotary_dim
        self.max_position = max_position
