import torch
import torch.nn.functional as F

from minivllm.attention import AttentionMetadata


class NaiveBackend:
    """Reference backend: for each sequence, gather its context out of the paged cache
    into a contiguous tensor and run SDPA. Slow, but easy to verify by reading."""

    def attend(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        flat_k = k_cache.flatten(0, 1)
        flat_v = v_cache.flatten(0, 1)
        offset = 0
        for i, (q_len, ctx) in enumerate(zip(meta.query_lens, meta.context_lens)):
            slots = context_slots(meta.block_tables[i], ctx, meta.block_size)
            rows = slice(offset, offset + q_len)
            out[rows] = causal_attention(q[rows], flat_k[slots], flat_v[slots])
            offset += q_len
        return out


def context_slots(block_table: torch.Tensor, context_len: int, block_size: int) -> torch.Tensor:
    """Flat cache slots of a sequence's tokens 0..context_len-1, in order. Gathering by
    token rather than by whole block copies exactly context_len tokens, which matters
    when one block is as large as the whole sequence (the contiguous layout)."""
    positions = torch.arange(context_len, device=block_table.device)
    return block_table[positions // block_size].long() * block_size + positions % block_size


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention for one sequence whose last q_len tokens are the queries.
    q: [q_len, heads, dim]; k, v: [context_len, kv_heads, dim], context_len >= q_len."""
    q_len, context_len = q.shape[0], k.shape[0]
    mask = None
    if 1 < q_len < context_len:
        # Queries sit at the end of the context, so the causal diagonal is offset by the
        # number of cached tokens before them; is_causal assumes no such offset.
        query_pos = torch.arange(context_len - q_len, context_len, device=q.device)
        mask = torch.arange(context_len, device=q.device)[None, :] <= query_pos[:, None]
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1),
        k.transpose(0, 1),
        v.transpose(0, 1),
        attn_mask=mask,
        is_causal=q_len > 1 and q_len == context_len,
        enable_gqa=True,
    )
    return out.transpose(0, 1)
