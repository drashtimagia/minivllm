from dataclasses import dataclass

import torch

from minivllm.attention import AttentionMetadata
from minivllm.kv_cache import PagedKVCache
from minivllm.models.qwen2 import Qwen2ForCausalLM


@dataclass(frozen=True)
class DecodeInputs:
    input_ids: list[int]
    positions: list[int]
    slot_mapping: list[int]
    block_tables: list[list[int]]
    context_lens: list[int]


def graph_buckets(max_batch_size: int) -> list[int]:
    """Batch sizes that get a captured graph: powers of two, plus the cap itself. A
    decode batch runs in the smallest bucket that holds it, padded up to that size."""
    buckets = []
    size = 1
    while size < max_batch_size:
        buckets.append(size)
        size *= 2
    return buckets + [max_batch_size]


def pick_bucket(buckets: list[int], batch_size: int) -> int | None:
    return next((b for b in buckets if b >= batch_size), None)


def pad_decode_inputs(inputs: DecodeInputs, bucket: int, scratch_block: int, block_size: int, table_width: int) -> DecodeInputs:
    """Fill a batch up to bucket rows. A graph always processes bucket rows, so the
    extra rows must be harmless: they read and write only a scratch block that no
    sequence owns, and every block table is padded to the same width."""
    extra = bucket - len(inputs.input_ids)
    scratch_table = [scratch_block] * table_width
    return DecodeInputs(
        input_ids=inputs.input_ids + [0] * extra,
        positions=inputs.positions + [0] * extra,
        slot_mapping=inputs.slot_mapping + [scratch_block * block_size] * extra,
        block_tables=[t + [scratch_block] * (table_width - len(t)) for t in inputs.block_tables]
        + [scratch_table] * extra,
        context_lens=inputs.context_lens + [1] * extra,
    )


class DecodeGraphs:
    """One CUDA graph per batch-size bucket for the decode forward pass (model plus
    vocabulary projection). A graph replays recorded kernel launches on fixed memory
    addresses, which removes the per-step Python and launch overhead that dominates a
    small decode step. Inputs are copied into static buffers before each replay."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        kv_cache: PagedKVCache,
        buckets: list[int],
        table_width: int,
        block_size: int,
        scratch_block: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.kv_cache = kv_cache
        self.buckets = buckets
        self.table_width = table_width
        self.block_size = block_size
        self.scratch_block = scratch_block
        self.device = device
        largest = max(buckets)
        self.input_ids = torch.zeros(largest, dtype=torch.long, device=device)
        self.positions = torch.zeros(largest, dtype=torch.long, device=device)
        self.slot_mapping = torch.zeros(largest, dtype=torch.long, device=device)
        self.block_tables = torch.zeros(largest, table_width, dtype=torch.int32, device=device)
        self.context_lens = torch.ones(largest, dtype=torch.int32, device=device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.logits: dict[int, torch.Tensor] = {}

    @torch.inference_mode()
    def capture(self) -> None:
        pool = None
        # Largest first: later graphs then reuse the memory pool the first one sized.
        for bucket in sorted(self.buckets, reverse=True):
            self._load(pad_decode_inputs(DecodeInputs([], [], [], [], []), bucket, self.scratch_block, self.block_size, self.table_width))
            self.kv_cache.set_metadata(self._metadata(bucket))
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                self._forward(bucket)  # warm-up: lazy initialisation must not happen during capture
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.logits[bucket] = self._forward(bucket)
            pool = graph.pool()
            self.graphs[bucket] = graph

    @torch.inference_mode()
    def run(self, inputs: DecodeInputs) -> torch.Tensor:
        batch = len(inputs.input_ids)
        bucket = pick_bucket(self.buckets, batch)
        if bucket is None:
            raise ValueError(f"batch of {batch} exceeds the largest captured graph ({max(self.buckets)})")
        self._load(pad_decode_inputs(inputs, bucket, self.scratch_block, self.block_size, self.table_width))
        self.graphs[bucket].replay()
        return self.logits[bucket][:batch]

    def _load(self, padded: DecodeInputs) -> None:
        n = len(padded.input_ids)
        for buffer, values in (
            (self.input_ids, padded.input_ids),
            (self.positions, padded.positions),
            (self.slot_mapping, padded.slot_mapping),
            (self.block_tables, padded.block_tables),
            (self.context_lens, padded.context_lens),
        ):
            buffer[:n].copy_(torch.tensor(values, dtype=buffer.dtype, device=self.device))

    def _metadata(self, bucket: int) -> AttentionMetadata:
        return AttentionMetadata(
            query_lens=[1] * bucket,
            context_lens=[],
            slot_mapping=self.slot_mapping[:bucket],
            block_tables=self.block_tables[:bucket],
            context_lens_tensor=self.context_lens[:bucket],
            block_size=self.block_size,
            is_decode_only=True,
        )

    def _forward(self, bucket: int) -> torch.Tensor:
        hidden = self.model(self.input_ids[:bucket], self.positions[:bucket], self.kv_cache)
        return self.model.compute_logits(hidden)
