import torch

from minivllm.attention import AttentionMetadata


class FlashBackend:
    """Paged attention through flash_attn_with_kvcache, which reads keys and values
    straight from the block tables instead of gathering them. CUDA only."""

    def __init__(self) -> None:
        try:
            from flash_attn import flash_attn_with_kvcache
        except ImportError as e:
            raise ImportError("the flash backend needs the flash-attn package and a CUDA GPU") from e
        self._kernel = flash_attn_with_kvcache

    def attend(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor:
        if meta.is_decode_only:
            return self._kernel(
                q[:, None],
                k_cache,
                v_cache,
                cache_seqlens=meta.context_lens_tensor,
                block_table=meta.block_tables,
                causal=True,
            )[:, 0]
        out = torch.empty_like(q)
        decode_rows = [i for i, n in enumerate(meta.query_lens) if n == 1]
        starts = torch.tensor([0] + meta.query_lens, device=q.device).cumsum(0)[:-1]
        if decode_rows:
            # Every decode query has length 1, so they batch into one kernel call.
            idx = torch.tensor(decode_rows, device=q.device)
            token_rows = starts[idx]
            out[token_rows] = self._kernel(
                q[token_rows][:, None],
                k_cache,
                v_cache,
                cache_seqlens=meta.context_lens_tensor[idx],
                block_table=meta.block_tables[idx],
                causal=True,
            )[:, 0]
        for i, q_len in enumerate(meta.query_lens):
            if q_len == 1:
                continue
            # Prefills have different lengths; one call each avoids padding them. The
            # kernel's causal mask is bottom-right aligned, so queries that follow a
            # cached prefix see that prefix.
            rows = slice(int(starts[i]), int(starts[i]) + q_len)
            out[rows] = self._kernel(
                q[rows][None],
                k_cache,
                v_cache,
                cache_seqlens=meta.context_lens_tensor[i : i + 1],
                block_table=meta.block_tables[i : i + 1],
                causal=True,
            )[0]
        return out
