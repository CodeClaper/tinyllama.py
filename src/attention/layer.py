from typing import Optional, List

import torch
import torch.nn as nn

from src.attention.backends.abstract import AttentionMetadata
from src.attention.selector import get_attn_backend
from src.defs import KVCache

class Attention(nn.Module):
    """
    Attention layer.

    This class takes query, key, and value tensor as input.
    The input tensors can either contain prompt tokens or 
    generation tokens.

    1. Store the input key and value tensors in the KV cache.
    2. Perform (multi-head/multi-query/grouped-query) attention.
    3. Return the ouput tensor.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.backend = get_attn_backend(torch.get_default_dtype())
        impl_cls = self.backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads=num_heads, 
            head_size=head_size, 
            scale=scale, 
            num_kv_heads=num_kv_heads, 
            alibi_slopes=alibi_slopes, 
            sliding_window=sliding_window
        )


    def forward(
        self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: KVCache,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        return self.impl.forward(query, key, value, kv_cache, attn_metadata)
