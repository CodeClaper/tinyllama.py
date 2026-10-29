import pytest
import torch

from src.model_executor.layers.attention import paged_attention


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 4), (4, 2), (8, 1)])
def test_paged_attention_matches_reference(dtype, num_heads, num_kv_heads):
    torch.manual_seed(0)
    device = "cuda"
    num_seqs = 3
    head_dim = 64
    block_size = 8
    seq_lens_list = [13, 1, 16]
    scale = head_dim**-0.5

    max_blocks = (max(seq_lens_list) + block_size - 1) // block_size
    total_blocks = num_seqs * max_blocks

    block_tables = torch.full(
        (num_seqs, max_blocks), -1, dtype=torch.int32, device=device
    )
    for i in range(num_seqs):
        start = i * max_blocks
        block_tables[i] = torch.arange(
            start, start + max_blocks, dtype=torch.int32, device=device
        )
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    key_cache = torch.randn(
        total_blocks, num_kv_heads, block_size, head_dim, dtype=dtype, device=device
    )
    value_cache = torch.randn_like(key_cache)
    query = torch.randn(num_seqs, num_heads, head_dim, dtype=dtype, device=device)

    out = paged_attention(
        query, key_cache, value_cache, block_tables, seq_lens, scale
    )

    ref = torch.empty_like(out)
    group_size = num_heads // num_kv_heads
    for seq, seq_len in enumerate(seq_lens_list):
        blocks = block_tables[seq].long()
        keys = key_cache[blocks].permute(0, 2, 1, 3).reshape(
            -1, num_kv_heads, head_dim
        )[:seq_len]
        values = value_cache[blocks].permute(0, 2, 1, 3).reshape(
            -1, num_kv_heads, head_dim
        )[:seq_len]
        keys = keys.repeat_interleave(group_size, dim=1)
        values = values.repeat_interleave(group_size, dim=1)

        scores = torch.einsum("hd,lhd->hl", query[seq].float(), keys.float()) * scale
        probs = torch.softmax(scores, dim=-1)
        ref[seq] = torch.einsum("hl,lhd->hd", probs, values.float()).to(dtype)

    if dtype == torch.float32:
        atol = rtol = 1e-4
    else:
        atol = rtol = 3e-2
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
