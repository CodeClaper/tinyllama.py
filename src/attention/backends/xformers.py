import torch

from typing import List, Optional
from src.attention.backends.abstract import AttentionBackend, AttentionMetadata
from src.attention.paged_attn import PagedAttentionMetadata


class XFormersAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """
    Metadata for XFormersAttentionBackend
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


class XFormersBackend(AttentionBackend):
    pass
