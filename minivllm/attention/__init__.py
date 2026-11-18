from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class AttentionMetadata:
    """Layout of one step's tokens. The step's queries are the concatenation, per
    sequence, of its query_len newest tokens; after this step's keys and values are
    written, sequence i has context_lens[i] tokens in the cache, stored in the blocks
    listed in block_tables[i]."""

    query_lens: list[int]
    context_lens: list[int]
    # [num_tokens]: flat cache slot (block_id * block_size + offset) for each new token.
    slot_mapping: torch.Tensor
    # [num_seqs, max_blocks_per_seq], int32, rows padded with 0 past each sequence's end.
    block_tables: torch.Tensor
    # [num_seqs], int32: context_lens as a tensor, which the flash kernel takes.
    context_lens_tensor: torch.Tensor
    block_size: int
    # Every query is a single token, so attention reduces to one batched kernel call
    # that reads only the tensors above; this is the shape CUDA graphs can capture.
    is_decode_only: bool = False


class AttentionBackend(Protocol):
    """Computes attention for the step's queries against keys and values that have
    already been written to the paged cache. k_cache, v_cache: [num_blocks, block_size,
    num_kv_heads, head_dim] for one layer."""

    def attend(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        meta: AttentionMetadata,
    ) -> torch.Tensor: ...
