import torch

from typing import Optional, List
from src.attention.backends.abstract import AttentionMetadata


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
