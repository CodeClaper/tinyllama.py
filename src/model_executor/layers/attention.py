import torch
from torch import nn

from cscr import attention as _C
from cscr import pos_encoding as _PE

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
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            scale=self.scale,
        )


class PagedAttentionWithRoPE(PagedAttention):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
    ):
        super().__init__(num_heads, head_size, scale)

        if rotary_dim <= 0 or rotary_dim % 2 != 0:
            raise ValueError("rotary_dim must be a positive even number")
        if rotary_dim > head_size:
            raise ValueError("rotary_dim must not exceed head_size")

        self.rotary_dim = rotary_dim

        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2) / rotary_dim))
        t = torch.arange(max_position).float()
        freqs = torch.einsum("i,j -> ij", t, inv_freq.float())
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)

        torch_dtype = torch.get_default_dtype()
        cache = cache.to(torch_dtype)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _write_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        positions: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
    ) -> None:
        num_seqs = positions.numel()
        num_kv_heads = key_cache.size(1)
        head_size = key_cache.size(3)
        block_size = key_cache.size(2)

        positions = positions.to(torch.long)
        logical_blocks = torch.div(positions, block_size, rounding_mode="floor")
        slots = positions % block_size

        rows = torch.arange(num_seqs, device=positions.device)
        physical_blocks = block_tables.to(torch.long)[rows, logical_blocks]
        valid = physical_blocks >= 0
        if not valid.any():
            return

        physical_blocks = physical_blocks[valid]
        slots = slots[valid]
        rows = rows[valid]

        keys = key.view(num_seqs, num_kv_heads, head_size)[rows].to(key_cache.dtype)
        values = value.view(num_seqs, num_kv_heads, head_size)[rows].to(
            value_cache.dtype
        )

        key_cache[physical_blocks, :, slots] = keys
        value_cache[physical_blocks, :, slots] = values

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        if query.dim() != 3:
            raise ValueError("query must be [num_tokens, num_heads, head_dim]")
        if key.dim() != 3:
            raise ValueError("key must be [num_tokens, num_kv_heads, head_dim]")
        if value.dim() != 3:
            raise ValueError("value must be [num_tokens, num_kv_heads, head_dim]")

        num_tokens, num_heads, head_size = query.shape
        if head_size != self.head_size:
            raise ValueError("query head_dim must match head_size")
        if num_tokens != block_tables.size(0):
            raise NotImplementedError(
                "PagedAttentionWithRoPE only supports decode: "
                "there must be one token per sequence"
            )

        query_2d = query.reshape(num_tokens, num_heads * head_size).contiguous()
        key_2d = key.reshape(num_tokens, -1).contiguous()

        cos_sin_cache = self.cos_sin_cache.to(
            device=query.device, dtype=query.dtype
        )
        query_2d = _PE.rotary_embedding_neox(
            positions, query_2d, key_2d, self.head_size, cos_sin_cache
        )

        self._write_kv_cache(
            key_2d, value, positions, key_cache, value_cache, block_tables
        )

        query = query_2d.view(num_tokens, num_heads, head_size)
        return super().forward(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
        )
