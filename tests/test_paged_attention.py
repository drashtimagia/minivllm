import pytest
import torch

from minivllm.attention import AttentionMetadata
from minivllm.attention.naive import NaiveBackend, causal_attention
from minivllm.kv_cache import PagedKVCache

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 4, 2, 16
BLOCK_SIZE = 4
NUM_BLOCKS = 32


def _paged_metadata(
    block_tables: list[list[int]], query_lens: list[int], context_lens: list[int]
) -> AttentionMetadata:
    slots: list[int] = []
    for table, q_len, ctx in zip(block_tables, query_lens, context_lens):
        for p in range(ctx - q_len, ctx):
            slots.append(table[p // BLOCK_SIZE] * BLOCK_SIZE + p % BLOCK_SIZE)
    width = max(len(t) for t in block_tables)
    padded = [t + [0] * (width - len(t)) for t in block_tables]
    return AttentionMetadata(
        query_lens=query_lens,
        context_lens=context_lens,
        slot_mapping=torch.tensor(slots),
        block_tables=torch.tensor(padded, dtype=torch.int32),
        context_lens_tensor=torch.tensor(context_lens, dtype=torch.int32),
        block_size=BLOCK_SIZE,
    )


def _cache() -> PagedKVCache:
    return PagedKVCache(1, NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, torch.float32, torch.device("cpu"), NaiveBackend())


def test_paged_naive_matches_contiguous_attention_with_scrambled_blocks() -> None:
    torch.manual_seed(0)
    # Blocks are deliberately non-adjacent and interleaved between sequences, with a mix
    # of a fresh prefill, a decode step, and a prefill after a cached prefix.
    block_tables = [[7, 2, 19], [30, 5, 11, 0], [23, 14]]
    query_lens = [9, 1, 3]
    context_lens = [9, 14, 8]
    full_k = [torch.randn(n, NUM_KV_HEADS, HEAD_DIM) for n in context_lens]
    full_v = [torch.randn(n, NUM_KV_HEADS, HEAD_DIM) for n in context_lens]
    q = torch.randn(sum(query_lens), NUM_HEADS, HEAD_DIM)
    cache = _cache()
    # Cached prefixes are written first, as earlier steps would have done.
    for i, (table, q_len, ctx) in enumerate(zip(block_tables, query_lens, context_lens)):
        prefix = ctx - q_len
        if prefix:
            meta = _paged_metadata([table], [prefix], [prefix])
            cache.set_metadata(meta)
            cache.k[0].flatten(0, 1)[meta.slot_mapping] = full_k[i][:prefix]
            cache.v[0].flatten(0, 1)[meta.slot_mapping] = full_v[i][:prefix]
    meta = _paged_metadata(block_tables, query_lens, context_lens)
    cache.set_metadata(meta)
    new_k = torch.cat([full_k[i][c - n : c] for i, (n, c) in enumerate(zip(query_lens, context_lens))])
    new_v = torch.cat([full_v[i][c - n : c] for i, (n, c) in enumerate(zip(query_lens, context_lens))])
    out = cache.attend(0, q, new_k, new_v)
    offset = 0
    for i, q_len in enumerate(query_lens):
        expected = causal_attention(q[offset : offset + q_len], full_k[i], full_v[i])
        torch.testing.assert_close(out[offset : offset + q_len], expected)
        offset += q_len


@pytest.mark.gpu
def test_flash_backend_matches_naive() -> None:
    from minivllm.attention.flash import FlashBackend

    torch.manual_seed(0)
    device, dtype = torch.device("cuda"), torch.float16
    k_cache = torch.randn(NUM_BLOCKS, BLOCK_SIZE, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=dtype)
    v_cache = torch.randn_like(k_cache)
    meta = _paged_metadata([[7, 2, 19], [30, 5, 11, 0], [23, 14]], [9, 1, 3], [9, 14, 8])
    meta = AttentionMetadata(
        meta.query_lens, meta.context_lens, meta.slot_mapping.to(device), meta.block_tables.to(device),
        meta.context_lens_tensor.to(device), meta.block_size,
    )
    q = torch.randn(13, NUM_HEADS, HEAD_DIM, device=device, dtype=dtype)
    expected = NaiveBackend().attend(q, k_cache, v_cache, meta)
    actual = FlashBackend().attend(q, k_cache, v_cache, meta)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
