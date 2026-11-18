import torch
import torch.nn.functional as F


class ContiguousKVCache:
    """KV cache for one sequence, reserved up front for max_len tokens."""

    def __init__(
        self,
        num_layers: int,
        max_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        shape = (num_layers, max_len, num_kv_heads, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.max_len = max_len
        self.num_cached = 0

    def attend(
        self, layer_idx: int, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        num_new = q.shape[0]
        end = self.num_cached + num_new
        if end > self.max_len:
            raise ValueError(f"sequence of {end} tokens exceeds KV cache capacity {self.max_len}")
        if self.num_cached > 0 and num_new > 1:
            raise ValueError("multi-token steps on top of a cached prefix are not supported")
        self.k[layer_idx, self.num_cached : end] = k
        self.v[layer_idx, self.num_cached : end] = v
        keys = self.k[layer_idx, :end]
        values = self.v[layer_idx, :end]
        # SDPA wants [batch, heads, tokens, head_dim]. With no cached prefix the query and
        # key ranges coincide, so the top-left aligned is_causal mask is exactly right; a
        # single decode token may see every cached key, so it needs no mask at all.
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            keys.transpose(0, 1)[None],
            values.transpose(0, 1)[None],
            is_causal=self.num_cached == 0 and num_new > 1,
            enable_gqa=True,
        )
        return out[0].transpose(0, 1)

    def advance(self, num_tokens: int) -> None:
        self.num_cached += num_tokens
