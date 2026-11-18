import torch

from minivllm.kv_cache import PaddedKVCache
from minivllm.models.qwen2 import Qwen2ForCausalLM
from minivllm.sampler import sample
from minivllm.sequence import SamplingParams


class StaticBatch:
    """A fixed group of requests that is prefilled together and then decoded in lockstep
    until every row is finished. Rows that finish early keep occupying a slot (and keep
    being computed on) until the longest one is done; that waste is what continuous
    batching removes, and this class exists as the baseline that shows it."""

    def __init__(
        self,
        model: Qwen2ForCausalLM,
        prompts: list[list[int]],
        params: list[SamplingParams],
        eos_token_ids: frozenset[int],
        max_model_len: int,
        rng: torch.Generator,
    ) -> None:
        if len(prompts) != len(params) or not prompts:
            raise ValueError("need one SamplingParams per prompt and at least one prompt")
        self.model = model
        self.params = params
        self.eos_token_ids = eos_token_ids
        self.rng = rng
        self.device = model.lm_head.weight.device
        self.prompt_len = max(len(p) for p in prompts)
        self.max_new = [min(p.max_tokens, max_model_len - len(prompt)) for p, prompt in zip(params, prompts)]
        if min(self.max_new) < 1:
            raise ValueError(f"a prompt leaves no room to generate within max_model_len={max_model_len}")
        self.pad_lens = [self.prompt_len - len(p) for p in prompts]
        self.cache = self._new_cache(capacity=self.prompt_len + max(self.max_new))
        self.outputs: list[list[int]] = [[] for _ in prompts]
        self.finished = [False] * len(prompts)
        self._pending_prompts: list[list[int]] | None = prompts

    @property
    def done(self) -> bool:
        return all(self.finished)

    @torch.inference_mode()
    def step(self) -> list[int | None]:
        """Advance every row by one token. Returns the new token per row, or None for a
        row that had already finished before this step."""
        if self._pending_prompts is not None:
            hidden = self._prefill(self._pending_prompts)
            self._pending_prompts = None
        else:
            hidden = self._decode()
        tokens = sample(self.model.compute_logits(hidden), self.params, self.rng).tolist()
        return [self._record(row, token) for row, token in enumerate(tokens)]

    def abort(self, row: int) -> None:
        self.finished[row] = True

    def _prefill(self, prompts: list[list[int]]) -> torch.Tensor:
        # Left padding puts every prompt's last token in the final column, so one slice
        # picks the logits rows and every row's next token lands in the same cache column.
        pad_id = min(self.eos_token_ids)
        ids = [[pad_id] * pad + p for pad, p in zip(self.pad_lens, prompts)]
        positions = [[0] * pad + list(range(len(p))) for pad, p in zip(self.pad_lens, prompts)]
        for row, pad in enumerate(self.pad_lens):
            self.cache.mark_padding(row, pad)
        hidden = self._forward(ids, positions)
        return hidden.view(len(prompts), self.prompt_len, -1)[:, -1]

    def _decode(self) -> torch.Tensor:
        # Finished rows still run: static batching cannot shrink the batch mid-flight.
        last = [out[-1] for out in self.outputs]
        step = self.cache.num_cached - self.prompt_len
        positions = [[prompt_len + step] for prompt_len in self._prompt_lens()]
        return self._forward([[t] for t in last], positions)

    def _forward(self, ids: list[list[int]], positions: list[list[int]]) -> torch.Tensor:
        ids_t = torch.tensor(ids, device=self.device).flatten()
        pos_t = torch.tensor(positions, device=self.device).flatten()
        hidden = self.model(ids_t, pos_t, self.cache)
        self.cache.advance(len(ids[0]))
        return hidden

    def _record(self, row: int, token: int) -> int | None:
        if self.finished[row]:
            return None
        self.outputs[row].append(token)
        hit_eos = token in self.eos_token_ids and not self.params[row].ignore_eos
        if hit_eos or len(self.outputs[row]) >= self.max_new[row]:
            self.finished[row] = True
        return token

    def _prompt_lens(self) -> list[int]:
        return [self.prompt_len - pad for pad in self.pad_lens]

    def _new_cache(self, capacity: int) -> PaddedKVCache:
        mc = self.model.config
        return PaddedKVCache(
            num_layers=mc.num_hidden_layers,
            batch_size=len(self.params),
            max_len=capacity,
            num_kv_heads=mc.num_key_value_heads,
            head_dim=mc.head_dim,
            dtype=self.model.lm_head.weight.dtype,
            device=self.device,
        )

