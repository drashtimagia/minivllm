import torch

from minivllm.config import EngineConfig
from minivllm.kv_cache import SlotBatch, SlotKVCache
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
        self.kv_cache = SlotKVCache(
            num_layers=mc.num_hidden_layers,
            num_slots=config.max_num_seqs,
            max_len=config.max_model_len,
            num_kv_heads=mc.num_key_value_heads,
            head_dim=mc.head_dim,
            dtype=config.torch_dtype,
            device=self.device,
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
        context_lens = [seq.num_tokens for seq in sched.seqs]
        self.kv_cache.set_batch(SlotBatch(sched.cache_slots, query_lens, context_lens))
        hidden = self.model(
            torch.tensor(input_ids, device=self.device),
            torch.tensor(positions, device=self.device),
            self.kv_cache,
        )
        last_token_rows = torch.tensor(query_lens, device=self.device).cumsum(0) - 1
        logits = self.model.compute_logits(hidden[last_token_rows])
        return sample(logits, [seq.params for seq in sched.seqs], self.rng).tolist()
