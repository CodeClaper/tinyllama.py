import torch

from typing import Optional, List, Tuple, Type
from src.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from src.model_executor.layers.attention import PagedAttention


class FlashAttentionMetadata(AttentionMetadata):
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
        attn_metadata: AttentionMetadata,
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
        raise NotADirectoryError


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
        return (2, num_blocks, block_size * num_kv_heads * head_size)
