from dataclasses import dataclass

import torch
from typing import Optional, Tuple
from cscr import cache as _C
from cscr import attention as _A
from src.defs import KVCache


@dataclass
class PagedAttentionMetadata:
    """Metadata for PagedAttention."""
    
    slot_mapping: torch.Tensor
    context_lens: Optional[torch.Tensor]
    max_context_len: Optional[int]
    block_tables: Optional[torch.Tensor]
    kv_cache_dtype: str

class PagedAttention:

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def split_kv_cache(
        kv_cache: KVCache,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        key_cache = kv_cache[0]
        x = 16 // key_cache.element_size()
        num_blocks = key_cache.shape[1]
        key_cache = key_cache.view(num_blocks, num_kv_heads, head_size // x, -1, x)

        value_cache = kv_cache[1]
        value_cache = value_cache.view(num_blocks, num_kv_heads, head_size, -1)

        return (key_cache, value_cache)
    

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
    ) -> None:
        _C.reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten(),
            kv_cache_dtype,
        )

    @staticmethod
    def forward_decode(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        block_size = value_cache.shape[3]
        return _A.paged_attention(
            query,
            key_cache,
            value_cache,
            num_kv_heads,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
            kv_cache_dtype,
        )

    @staticmethod
    def forward_prefix(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        block_tables: Optional[torch.Tensor],
        subquery_start_loc: Optional[torch.Tensor],
        prompt_lens_tensor: Optional[torch.Tensor],
        context_lens: Optional[torch.Tensor],
        max_subquery_len: int,
        alibi_slopes: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Attention for a chunk of prompt tokens.

        The prompt of each sequence is split into subqueries. A subquery
        attends to the cached prefix (``context_lens[i]`` tokens already
        stored in ``key_cache``/``value_cache``) and to the current chunk of
        keys/values causally.

        Args:
            query: ``[num_query_tokens, num_heads, head_size]``.
            key: ``[num_query_tokens, num_kv_heads, head_size]``.
            value: ``[num_query_tokens, num_kv_heads, head_size]``.
            key_cache: ``[num_blocks, num_kv_heads, head_size / x, block_size, x]``.
            value_cache: ``[num_blocks, num_kv_heads, head_size, block_size]``.
            block_tables: ``[num_seqs, max_blocks_per_seq]`` int32.
            subquery_start_loc: ``[num_seqs + 1]`` cumulative start offsets of
                each subquery inside the packed query.
            prompt_lens_tensor: ``[num_seqs]`` full prompt lengths, including
                the cached prefix and the current subquery.
            context_lens: ``[num_seqs]`` number of tokens already cached.
            max_subquery_len: maximum number of tokens in a single subquery.
            alibi_slopes: optional ``[num_heads]`` float32 slopes.

        Returns:
            ``[num_query_tokens, num_heads, head_size]``.
        """
        if query.dim() != 3:
            raise ValueError("query must be [num_tokens, num_heads, head_size]")
        if key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                "key and value must be [num_tokens, num_kv_heads, head_size]"
            )
        if key.shape != value.shape:
            raise ValueError("key and value must have the same shape")
        if key.size(0) != query.size(0):
            raise ValueError("query, key and value must have the same number of tokens")

        num_tokens, num_heads, head_size = query.shape
        num_kv_heads = key.size(1)
        if key.size(2) != head_size:
            raise ValueError("query and key must have the same head_size")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")

        if (key_cache is None) != (value_cache is None):
            raise ValueError(
                "key_cache and value_cache must both be provided or both be None"
            )

        if key_cache is not None and value_cache is not None:
            if key_cache.dim() != 5 or value_cache.dim() != 4:
                raise ValueError(
                    "key_cache and value_cache must use the paged cache layout"
                )
            x = key_cache.size(4)
            block_size = value_cache.size(3)
            if x <= 0 or head_size % x != 0:
                raise ValueError("head_size must be divisible by key_cache.size(4)")
            if key_cache.size(1) != num_kv_heads:
                raise ValueError("key_cache num_kv_heads must match key")
            if value_cache.size(1) != num_kv_heads:
                raise ValueError("value_cache num_kv_heads must match key")
            if key_cache.size(2) != head_size // x:
                raise ValueError("key_cache head_size / x must match query")
            if key_cache.size(3) != block_size:
                raise ValueError("key_cache and value_cache block_size must match")
            if value_cache.size(2) != head_size:
                raise ValueError("value_cache head_size must match query")
            if key_cache.size(0) != value_cache.size(0):
                raise ValueError(
                    "key_cache and value_cache must have the same number of blocks"
                )
        else:
            # Without a KV cache there is no cached prefix to attend to.
            block_size = 0

        num_seqs = block_tables.size(0)
        if block_tables.dim() != 2:
            raise ValueError("block_tables must be [num_seqs, max_blocks_per_seq]")
        if subquery_start_loc.numel() != num_seqs + 1:
            raise ValueError("subquery_start_loc must be [num_seqs + 1]")
        if prompt_lens_tensor.numel() != num_seqs:
            raise ValueError("prompt_lens_tensor must be [num_seqs]")
        if context_lens.numel() != num_seqs:
            raise ValueError("context_lens must be [num_seqs]")
        if max_subquery_len < 0:
            raise ValueError("max_subquery_len must be non-negative")

        alibi = None
        if alibi_slopes is not None:
            alibi = alibi_slopes.to(device=query.device, dtype=torch.float32)
            if alibi.numel() != num_heads:
                raise ValueError("alibi_slopes must have one value per query head")

        subquery_start = subquery_start_loc.tolist()
        prompt_lens = prompt_lens_tensor.tolist()
        context_lens_list = context_lens.tolist()

        output = torch.empty_like(query)
        scale = head_size**-0.5
        group_size = num_heads // num_kv_heads
        max_blocks_per_seq = block_tables.size(1)

        for seq_idx in range(num_seqs):
            start = int(subquery_start[seq_idx])
            end = int(subquery_start[seq_idx + 1])
            chunk_len = end - start
            if chunk_len < 0:
                raise ValueError("subquery_start_loc must be non-decreasing")
            if chunk_len == 0:
                continue
            if chunk_len > max_subquery_len:
                raise ValueError(
                    "max_subquery_len is smaller than a subquery in the batch"
                )

            ctx_len = int(context_lens_list[seq_idx])
            prompt_len = int(prompt_lens[seq_idx])
            if ctx_len < 0 or ctx_len > prompt_len:
                raise ValueError("context_lens must be in [0, prompt_len]")
            if chunk_len != prompt_len - ctx_len:
                raise ValueError(
                    "subquery length must equal prompt_len - context_len"
                )
            if start < 0 or end > num_tokens:
                raise ValueError("subquery_start_loc is out of range")

            if ctx_len > 0:
                if key_cache is None or value_cache is None:
                    raise ValueError(
                        "context_lens must be zero when no KV cache is provided"
                    )
                if ctx_len > max_blocks_per_seq * block_size:
                    raise ValueError("block_tables cannot cover the cached prefix")
                num_prefix_blocks = (ctx_len + block_size - 1) // block_size
                blocks = block_tables[seq_idx, :num_prefix_blocks].long()
                if (blocks < 0).any():
                    raise ValueError(
                        "block_tables contains an invalid block for the cached prefix"
                    )
                k_prefix = key_cache[blocks]
                k_prefix = k_prefix.permute(0, 3, 1, 2, 4).reshape(
                    -1, num_kv_heads, head_size
                )[:ctx_len]
                v_prefix = value_cache[blocks]
                v_prefix = v_prefix.permute(0, 3, 1, 2).reshape(
                    -1, num_kv_heads, head_size
                )[:ctx_len]
            else:
                k_prefix = key.new_empty((0, num_kv_heads, head_size))
                v_prefix = value.new_empty((0, num_kv_heads, head_size))

            k_all = torch.cat((k_prefix, key[start:end]), dim=0).float()
            v_all = torch.cat((v_prefix, value[start:end]), dim=0).float()
            if group_size > 1:
                k_all = k_all.repeat_interleave(group_size, dim=1)
                v_all = v_all.repeat_interleave(group_size, dim=1)

            query_chunk = query[start:end].float()
            scores = torch.einsum("qhd,khd->hqk", query_chunk, k_all)
            scores *= scale

            total_len = ctx_len + chunk_len
            query_pos = torch.arange(
                ctx_len, total_len, device=query.device, dtype=torch.long
            )
            key_pos = torch.arange(
                total_len, device=query.device, dtype=torch.long
            )
            causal = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
            scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))

            if alibi is not None:
                scores = scores + alibi[:, None, None] * (
                    key_pos.unsqueeze(0) - query_pos.unsqueeze(1)
                )

            probs = torch.softmax(scores, dim=-1)
            chunk_output = torch.einsum("hqk,khd->qhd", probs, v_all)
            output[start:end] = chunk_output.to(query.dtype)

        return output

