from pathlib import Path
from typing import Any

import torch
from torch import nn

_EXTENSION: Any | None = None


def _get_extension() -> Any:
    """Compile the C++/CUDA sources on first use and cache the module."""
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        root = Path(__file__).resolve().parents[3]
        _EXTENSION = load(
            name="tinyllama_attention",
            sources=[
                str(root / "cscr" / "attention.cpp"),
                str(root / "cscr" / "attention_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    return _EXTENSION


def paged_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
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
    return _get_extension().paged_attention(
        query, key_cache, value_cache, block_tables, seq_lens, float(scale)
    )


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
        return paged_attention(
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
    ):
        super().__init__(num_heads, head_size, scale)
        self.rotary_dim = rotary_dim
        self.max_position = max_position
