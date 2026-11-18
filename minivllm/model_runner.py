import torch

from minivllm.attention import AttentionMetadata
from minivllm.attention.flash import FlashBackend
from minivllm.attention.naive import NaiveBackend
from minivllm.config import EngineConfig
from minivllm.kv_cache import PagedKVCache, kv_bytes_per_block
from minivllm.models.qwen2 import Qwen2ForCausalLM
from minivllm.sampler import sample
from minivllm.scheduler import SchedulerOutput


class ModelRunner:
    """Turns a SchedulerOutput into tensors, runs the model, and samples one token per
    sequence. It does not decide what runs; it only executes what it is given."""

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
        self.kv_cache = PagedKVCache(
            num_layers=mc.num_hidden_layers,
            num_blocks=self.num_blocks,
            block_size=config.block_size,
            num_kv_heads=mc.num_key_value_heads,
            head_dim=mc.head_dim,
            dtype=config.torch_dtype,
            device=self.device,
            backend=backend,
        )

    @torch.inference_mode()
    def execute(self, sched: SchedulerOutput) -> list[int]:
        input_ids: list[int] = []
        positions: list[int] = []
        query_lens: list[int] = []
        for seq in sched.seqs:
            start = seq.num_computed_tokens
            input_ids += seq.token_ids[start:]
            positions += range(start, seq.num_tokens)
            query_lens.append(seq.num_tokens - start)
        self.kv_cache.set_metadata(self._metadata(sched, positions, query_lens))
        hidden = self.model(
            torch.tensor(input_ids, device=self.device),
            torch.tensor(positions, device=self.device),
            self.kv_cache,
        )
        last_token_rows = torch.tensor(query_lens, device=self.device).cumsum(0) - 1
        logits = self.model.compute_logits(hidden[last_token_rows])
        return sample(logits, [seq.params for seq in sched.seqs], self.rng).tolist()

    def _metadata(
        self, sched: SchedulerOutput, positions: list[int], query_lens: list[int]
    ) -> AttentionMetadata:
        block_size = self.config.block_size
        # Token at position p of a sequence lives at offset p % block_size of block
        # p // block_size of that sequence's table; flattening the pool makes that the
        # single index block_id * block_size + offset.
        slots: list[int] = []
        offset = 0
        for table, q_len in zip(sched.block_tables, query_lens):
            slots += [table[p // block_size] * block_size + p % block_size for p in positions[offset : offset + q_len]]
            offset += q_len
        context_lens = [seq.num_tokens for seq in sched.seqs]
        width = max(len(t) for t in sched.block_tables)
        padded = [t + [0] * (width - len(t)) for t in sched.block_tables]
        return AttentionMetadata(
            query_lens=query_lens,
            context_lens=context_lens,
            slot_mapping=torch.tensor(slots, device=self.device),
            block_tables=torch.tensor(padded, dtype=torch.int32, device=self.device),
            context_lens_tensor=torch.tensor(context_lens, dtype=torch.int32, device=self.device),
            block_size=block_size,
        )
