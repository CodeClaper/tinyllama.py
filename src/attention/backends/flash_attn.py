from dataclasses import dataclass

import torch

from typing import Optional, List, Tuple, Type
from src.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from src.attention.paged_attn import PagedAttention, PagedAttentionMetadata

@dataclass
class FlashAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """
    Metadata for FlashAttentionBacked.

    Args:
        is_prompt: Input sequences can only contain all prompts or all decoding. True if all sequences are prompts.
        prompt_lens: The prompt length per sequence.
        prompt_lens_tensor: The prompt_lens store as a tensor.
        num_prompt_tokens: The number of prompt tokens, not include padding.
        num_prompt_tokens: The number of generation tokens, not include padding.
    """

    is_prompt: bool
    prompt_lens: Optional[List[int]]
    prompt_lens_tensor: Optional[torch.Tensor]
    num_prompt_tokens: int
    num_generation_tokens: int
    max_subquery_len: Optional[int]
    max_prompt_len: Optional[int]
    subquery_start_loc: Optional[torch.Tensor]
    seq_start_loc: Optional[torch.Tensor]
    use_cuda_graph: bool


class FlashAttentionImpl(AttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        self.alibi_slopes = (
            torch.tensor(alibi_slopes, dtype=torch.float32)
            if alibi_slopes is not None
            else None
        )
        self.sliding_window = (
            (sliding_window, sliding_window) 
            if sliding_window is not None 
            else (-1, -1)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: FlashAttentionMetadata,
    ) -> torch.Tensor:
        """
        Forward pass with FlashAttention and PagedAttention.

        Args: 
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        
        key_cache: Optional[torch.Tensor] = None
        value_cache: Optional[torch.Tensor] = None
        if kv_cache is not None:
            key_cache, value_cache = PagedAttention.split_kv_cache(kv_cache, self.num_kv_heads, self.head_size)
            # Reshape the input keys and values and store them in the cache.,
            PagedAttention.write_to_paged_cache(key, value, key_cache, value_cache, attn_metadata.slot_mapping, attn_metadata.kv_cache_dtype)
        
        if attn_metadata.is_prompt:
            output = PagedAttention.forward_prefix(
                query=query,
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=attn_metadata.block_tables,
                subquery_start_loc=attn_metadata.subquery_start_loc,
                prompt_lens_tensor=attn_metadata.prompt_lens_tensor,
                context_lens=attn_metadata.context_lens,
                max_subquery_len=attn_metadata.max_context_len,
                alibi_slopes=self.alibi_slopes
            )
        else:
            if key_cache is None or value_cache is None:
                raise ValueError("kv_cache is required for decoding")
            output = PagedAttention.forward_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=attn_metadata.block_tables,
                context_lens=attn_metadata.context_lens,
                max_context_len=attn_metadata.max_context_len,
                kv_cache_dtype=attn_metadata.kv_cache_dtype,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                alibi_slopes=self.alibi_slopes,
            )

        return output.view(num_tokens, hidden_size)


class FlashAttentionBacked(AttentionBackend):
    
    @staticmethod
    def get_impl_cls() -> Type[FlashAttentionImpl]:
        return FlashAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)




