
import torch
from torch import nn


class PagedAttention(nn.Module):

    def __init__(
        self,
        num_head: int,
        head_size: int,
        scale: float,
    ):
        super().__init__()
        self.num_heads = num_head
        self.head_size = head_size
        self.scale = scale

    def forward(
        self,
        query: torch.Tensor,                    #[num_tokens, num_heads, heads_dim]
        key: torch.Tensor,                      #[num_tokens, num_heads, heads_dim]
        value: torch.Tensor,                    #[num_tokens, num_heads, heads_dim]
        key_cache: torch.Tensor | None,      #[num_blocks, num_heads, heads_dim/x, block_size, x]
        value_cache: torch.Tensor | None,    #[num_blocks, num_heads, heads_dim, block_size]
    ) -> torch.Tensor:
        output = torch.empty_like(query)
        return output;



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

    def forward(
        self,
        query: torch.Tensor,                    #[num_tokens, num_heads, heads_dim]
        key: torch.Tensor,                      #[num_tokens, num_heads, heads_dim]
        value: torch.Tensor,                    #[num_tokens, num_heads, heads_dim]
        key_cache: torch.Tensor | None,      #[num_blocks, num_heads, heads_dim/x, block_size, x]
        value_cache: torch.Tensor | None,    #[num_blocks, num_heads, heads_dim, block_size]
    ) -> torch.Tensor:
        return super().forward(
            query=query,
            key=key,
            value=value,
            key_cache=key_cache,
            value_cache=value_cache,
        );
