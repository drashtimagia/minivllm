from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn.functional as F


class CacheAttention(Protocol):
    """What the model needs from a KV cache: store this step's keys and values for a
    layer and return attention output for this step's queries. Tokens arrive flattened;
    each implementation knows how they group into sequences."""

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...


def causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Attention for one sequence whose last q_len tokens are the queries.
    q: [q_len, heads, dim]; k, v: [context_len, kv_heads, dim], context_len >= q_len."""
    q_len, context_len = q.shape[0], k.shape[0]
    mask = None
    if q_len > 1 and q_len < context_len:
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


@dataclass(frozen=True)
class SlotBatch:
    """Where each sequence of the current step lives in a SlotKVCache. The step's
    tokens are the concatenation of each sequence's query_len newest tokens."""

    slots: list[int]
    query_lens: list[int]
    context_lens: list[int]


class SlotKVCache:
    """One contiguous region of max_len tokens per sequence slot, reserved when the
    sequence is admitted whether or not it ever grows that long."""

    def __init__(
        self,
        num_layers: int,
        num_slots: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_layers, num_slots, max_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.batch: SlotBatch | None = None

    def set_batch(self, batch: SlotBatch) -> None:
        if max(batch.context_lens) > self.max_len:
            raise ValueError(f"context of {max(batch.context_lens)} exceeds slot size {self.max_len}")
        self.batch = batch

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if self.batch is None:
            raise RuntimeError("set_batch must be called before the forward pass")
        out = torch.empty_like(q)
        offset = 0
        for slot, q_len, ctx in zip(self.batch.slots, self.batch.query_lens, self.batch.context_lens):
            rows = slice(offset, offset + q_len)
            self.k[layer_idx, slot, ctx - q_len : ctx] = k[rows]
            self.v[layer_idx, slot, ctx - q_len : ctx] = v[rows]
            out[rows] = causal_attention(
                q[rows], self.k[layer_idx, slot, :ctx], self.v[layer_idx, slot, :ctx]
            )
            offset += q_len
        return out


class PaddedKVCache:
    """KV cache for a static batch: one row per sequence, all rows advance together, and
    each row is reserved up front for max_len tokens. Rows are left padded, so the cache
    keeps a mask of which slots hold padding."""

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_layers, batch_size, max_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.is_real = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        self.batch_size = batch_size
        self.max_len = max_len
        self.num_cached = 0

    def mark_padding(self, row: int, num_pad: int) -> None:
        self.is_real[row, :num_pad] = False

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        num_tokens, num_heads, head_dim = q.shape
        num_new = num_tokens // self.batch_size
        start, end = self.num_cached, self.num_cached + num_new
        if end > self.max_len:
            raise ValueError(f"step needs {end} cache slots; capacity is {self.max_len}")
        self.k[layer_idx, :, start:end] = k.view(self.batch_size, num_new, *k.shape[1:])
        self.v[layer_idx, :, start:end] = v.view(self.batch_size, num_new, *v.shape[1:])
        out = F.scaled_dot_product_attention(
            q.view(self.batch_size, num_new, num_heads, head_dim).transpose(1, 2),
            self.k[layer_idx, :, :end].transpose(1, 2),
            self.v[layer_idx, :, :end].transpose(1, 2),
            attn_mask=self._mask(start, end),
            enable_gqa=True,
        )
        return out.transpose(1, 2).reshape(num_tokens, num_heads, head_dim)

    def _mask(self, start: int, end: int) -> torch.Tensor:
        query_pos = torch.arange(start, end, device=self.is_real.device)
        key_pos = torch.arange(end, device=self.is_real.device)
        causal = key_pos[None, :] <= query_pos[:, None]
        allowed = causal[None] & self.is_real[:, None, :end]
        # A padding query on the left edge has no real key to its left, and a row of all
        # False makes softmax produce NaN. Letting each query see itself keeps the row
        # finite; real queries already see themselves, and nothing reads padding outputs.
        allowed[:, torch.arange(end - start), query_pos] = True
        return allowed[:, None]

    def advance(self, num_tokens: int) -> None:
        self.num_cached += num_tokens
