import torch

from minivllm.attention import AttentionMetadata
from minivllm.attention.flash import FlashBackend
from minivllm.attention.naive import NaiveBackend
from minivllm.config import EngineConfig
from minivllm.cuda_graphs import DecodeGraphs, DecodeInputs, graph_buckets
from minivllm.kv_cache import PagedKVCache, kv_bytes_per_block
from minivllm.models.qwen2 import Qwen2ForCausalLM
from minivllm.sampler import sample
from minivllm.scheduler import SchedulerOutput


class ModelRunner:
    """Turns a SchedulerOutput into tensors, runs the model, and samples one token per
    sequence that has reached the end of its tokens. It does not decide what runs; it
    only executes what it is given."""

    def __init__(self, config: EngineConfig, model: Qwen2ForCausalLM) -> None:
        self.config = config
        self.model = model
        self.device = torch.device(config.device)
        self.rng = torch.Generator(device=self.device).manual_seed(config.seed)
        mc = model.config
        bytes_per_block = kv_bytes_per_block(
            mc.num_hidden_layers, config.block_size, mc.num_key_value_heads, mc.head_dim, config.torch_dtype
        )
        self.num_blocks = int(config.kv_budget_gb * 1e9 // bytes_per_block)
        min_blocks = -(-config.max_model_len // config.block_size)
        if self.num_blocks < min_blocks:
            raise ValueError(
                f"kv_budget_gb={config.kv_budget_gb} gives {self.num_blocks} blocks, but one "
                f"max_model_len={config.max_model_len} sequence needs {min_blocks}"
            )
        backend = FlashBackend() if config.attention_backend == "flash" else NaiveBackend()
        # Padding rows of a graph batch write somewhere; one extra block outside the
        # block manager's range keeps those writes away from every sequence.
        self.scratch_block = self.num_blocks
        self.kv_cache = PagedKVCache(
            num_layers=mc.num_hidden_layers,
            num_blocks=self.num_blocks + (1 if config.enable_cuda_graphs else 0),
            block_size=config.block_size,
            num_kv_heads=mc.num_key_value_heads,
            head_dim=mc.head_dim,
            dtype=config.torch_dtype,
            device=self.device,
            backend=backend,
        )
        self.graphs = self._capture_graphs() if config.enable_cuda_graphs else None

    def _capture_graphs(self) -> DecodeGraphs:
        cap = min(self.config.cuda_graph_max_batch_size, self.config.max_num_seqs)
        graphs = DecodeGraphs(
            self.model,
            self.kv_cache,
            graph_buckets(cap),
            table_width=-(-self.config.max_model_len // self.config.block_size),
            block_size=self.config.block_size,
            scratch_block=self.scratch_block,
            device=self.device,
        )
        graphs.capture()
        return graphs

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput) -> list[int | None]:
        sampling_rows = [
            i
            for i, (seq, n) in enumerate(zip(sched.seqs, sched.num_scheduled_tokens))
            if seq.num_computed_tokens + n == seq.num_tokens
        ]
        logits = self._forward(sched, sampling_rows)
        tokens: list[int | None] = [None] * len(sched.seqs)
        if logits is not None:
            sampled = sample(logits, [sched.seqs[i].params for i in sampling_rows], self.rng).tolist()
            for row, token in zip(sampling_rows, sampled):
                tokens[row] = token
        return tokens

    def _forward(self, sched: SchedulerOutput, sampling_rows: list[int]) -> torch.Tensor | None:
        if self._can_replay_graph(sched, sampling_rows):
            assert self.graphs is not None
            return self.graphs.run(self._decode_inputs(sched))
        input_ids: list[int] = []
        positions: list[int] = []
        for seq, n in zip(sched.seqs, sched.num_scheduled_tokens):
            start = seq.num_computed_tokens
            input_ids += seq.token_ids[start : start + n]
            positions += range(start, start + n)
        self.kv_cache.set_metadata(self._metadata(sched, positions))
        hidden = self.model(
            torch.tensor(input_ids, device=self.device),
            torch.tensor(positions, device=self.device),
            self.kv_cache,
        )
        if not sampling_rows:
            return None
        last_token_rows = torch.tensor(sched.num_scheduled_tokens, device=self.device).cumsum(0) - 1
        return self.model.compute_logits(hidden[last_token_rows[sampling_rows]])

    def _can_replay_graph(self, sched: SchedulerOutput, sampling_rows: list[int]) -> bool:
        # Only pure decode batches have the fixed shape a graph needs; anything with a
        # prompt chunk in it runs eagerly.
        if self.graphs is None or len(sampling_rows) != len(sched.seqs):
            return False
        return all(n == 1 for n in sched.num_scheduled_tokens) and len(sched.seqs) <= max(self.graphs.buckets)

    def _decode_inputs(self, sched: SchedulerOutput) -> DecodeInputs:
        bs = self.config.block_size
        positions = [seq.num_computed_tokens for seq in sched.seqs]
        return DecodeInputs(
            input_ids=[seq.token_ids[p] for seq, p in zip(sched.seqs, positions)],
            positions=positions,
            slot_mapping=[t[p // bs] * bs + p % bs for t, p in zip(sched.block_tables, positions)],
            block_tables=sched.block_tables,
            context_lens=[p + 1 for p in positions],
        )

    def _metadata(self, sched: SchedulerOutput, positions: list[int]) -> AttentionMetadata:
        block_size = self.config.block_size
        query_lens = sched.num_scheduled_tokens
        # Token at position p of a sequence lives at offset p % block_size of block
        # p // block_size of that sequence's table; flattening the pool makes that the
        # single index block_id * block_size + offset.
        slots: list[int] = []
        offset = 0
        for table, q_len in zip(sched.block_tables, query_lens):
            slots += [table[p // block_size] * block_size + p % block_size for p in positions[offset : offset + q_len]]
            offset += q_len
        # A chunk of a long prompt attends only to tokens up to the end of that chunk.
        context_lens = [seq.num_computed_tokens + n for seq, n in zip(sched.seqs, query_lens)]
        width = max(len(t) for t in sched.block_tables)
        padded = [t + [0] * (width - len(t)) for t in sched.block_tables]
        return AttentionMetadata(
            query_lens=query_lens,
            context_lens=context_lens,
            slot_mapping=torch.tensor(slots, device=self.device),
            block_tables=torch.tensor(padded, dtype=torch.int32, device=self.device),
            context_lens_tensor=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_size=block_size,
            is_decode_only=all(n == 1 for n in query_lens),
        )
