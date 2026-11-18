from typing import Protocol

import torch
import torch.nn.functional as F

from minivllm.attention import AttentionBackend, AttentionMetadata


class CacheAttention(Protocol):
    """What the model needs from a KV cache: store this step's keys and values for a
    layer and return attention output for this step's queries. Tokens arrive flattened;
    each implementation knows how they group into sequences."""

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor: ...


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


def kv_bytes_per_block(
    num_layers: int, block_size: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype
) -> int:
    return 2 * num_layers * block_size * num_kv_heads * head_dim * dtype.itemsize


class PagedKVCache:
    """Per-layer pools of fixed-size blocks, [num_blocks, block_size, num_kv_heads,
    head_dim], shared by all sequences. Which blocks belong to which sequence is
    described by the block tables in the step's AttentionMetadata."""

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        backend: AttentionBackend,
    ) -> None:
        shape = (num_layers, num_blocks, block_size, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.backend = backend
        self.meta: AttentionMetadata | None = None

    def set_metadata(self, meta: AttentionMetadata) -> None:
        self.meta = meta

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        if self.meta is None:
            raise RuntimeError("set_metadata must be called before the forward pass")
        # Viewing a layer's pool as one flat run of slots turns "write token t of
        # sequence s" into a single scatter through slot_mapping.
        self.k[layer_idx].flatten(0, 1)[self.meta.slot_mapping] = k
        self.v[layer_idx].flatten(0, 1)[self.meta.slot_mapping] = v
        return self.backend.attend(q, self.k[layer_idx], self.v[layer_idx], self.meta)
