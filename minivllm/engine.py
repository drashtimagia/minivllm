import time

from minivllm.block_manager import BlockManager
from minivllm.config import EngineConfig
from minivllm.model_files import LoadedModel, load_model
from minivllm.model_runner import ModelRunner
from minivllm.scheduler import Scheduler
from minivllm.sequence import EngineStats, RequestOutput, SamplingParams, Sequence
from minivllm.static_batch import StaticEngine
from minivllm.validation import check_prompt_fits


class Engine:
    """Continuous batching: requests join and leave the running batch between steps."""

    def __init__(self, config: EngineConfig, loaded: LoadedModel) -> None:
        self.config = config
        self.tokenizer = loaded.tokenizer
        self.eos_token_ids = loaded.eos_token_ids
        self.runner = ModelRunner(config, loaded.model)
        block_manager = BlockManager(
            self.runner.num_blocks, config.block_size, config.enable_prefix_caching
        )
        self.scheduler = Scheduler(
            block_manager, config.max_num_seqs, config.max_num_batched_tokens, config.enable_chunked_prefill
        )

    def add_request(self, request_id: str, prompt_token_ids: list[int], params: SamplingParams) -> None:
        check_prompt_fits(prompt_token_ids, self.config.max_model_len)
        self.scheduler.add(Sequence(request_id, list(prompt_token_ids), params, time.monotonic()))

    def abort_request(self, request_id: str) -> bool:
        return self.scheduler.abort(request_id) is not None

    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_unfinished()

    def stats(self) -> EngineStats:
        bm = self.scheduler.block_manager
        return EngineStats(
            num_running=len(self.scheduler.running),
            num_waiting=len(self.scheduler.waiting),
            num_blocks=bm.num_blocks,
            num_free_blocks=bm.num_free_blocks,
            prefix_hit_rate=bm.hit_rate,
        )

    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        if sched is None:
            return []
        tokens = self.runner.execute(sched)
        outputs = []
        for seq, n, token in zip(sched.seqs, sched.num_scheduled_tokens, tokens, strict=True):
            seq.num_computed_tokens += n
            self.scheduler.register_computed(seq)
            if token is None:
                continue
            seq.append_token(token, self.eos_token_ids, self.config.max_model_len)
            reason = seq.status.value if seq.finished else None
            outputs.append(RequestOutput(seq.request_id, [token], seq.finished, reason))
        self.scheduler.free_finished()
        return outputs


AnyEngine = Engine | StaticEngine


def create_engine(config: EngineConfig) -> AnyEngine:
    loaded = load_model(config)
    if config.scheduler == "static":
        return StaticEngine(config, loaded)
    return Engine(config, loaded)


def generate(
    engine: AnyEngine, prompts: list[list[int]], params: list[SamplingParams]
) -> list[list[int]]:
    """Offline helper: submit every prompt, step until all finish, return outputs in order."""
    ids = [f"offline-{i}" for i in range(len(prompts))]
    for request_id, prompt, p in zip(ids, prompts, params, strict=True):
        engine.add_request(request_id, prompt, p)
    outputs: dict[str, list[int]] = {request_id: [] for request_id in ids}
    while engine.has_unfinished_requests():
        for out in engine.step():
            outputs[out.request_id] += out.new_token_ids
    return [outputs[request_id] for request_id in ids]
