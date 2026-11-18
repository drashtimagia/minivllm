import pytest
from transformers import PreTrainedTokenizerBase

from minivllm.engine import Engine
from minivllm.model_files import LoadedModel
from minivllm.scheduler import SchedulerOutput
from minivllm.sequence import SamplingParams
from tests.prompts import LONG_ARTICLE
from tests.test_engine import _collect, _engine

PARAMS = SamplingParams(temperature=0.0, max_tokens=16)
SYSTEM = "You are a careful assistant. Reference text:\n" + LONG_ARTICLE[:1500]
QUESTIONS = ["What is the main idea?", "Name one bottleneck mentioned.", "Summarize in five words."]


def _prompts(tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    return [
        list(
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
        )
        for q in QUESTIONS
    ]


def _record_prefill_starts(engine: Engine) -> list[list[int]]:
    """For every prefill step, how many tokens of each sequence were already cached."""
    starts: list[list[int]] = []
    original = engine.runner.execute

    def spy(sched: SchedulerOutput) -> list[int]:
        if sched.is_prefill:
            starts.append([s.num_computed_tokens for s in sched.seqs])
        return original(sched)

    engine.runner.execute = spy  # type: ignore[method-assign]
    return starts


def _run_sequentially(engine: Engine, prompts: list[list[int]]) -> list[list[int]]:
    results = []
    for i, prompt in enumerate(prompts):
        engine.add_request(f"r{i}", prompt, PARAMS)
        out: dict[str, list[int]] = {}
        while engine.has_unfinished_requests():
            _collect(out, engine.step())
        results.append(out[f"r{i}"])
    return results


def _common_prefix_len(a: list[int], b: list[int]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def test_shared_system_prompt_gives_identical_outputs(
    loaded: LoadedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompts = _prompts(tokenizer)
    off = _run_sequentially(_engine(loaded), prompts)
    on_engine = _engine(loaded, prefix_caching=True)
    assert _run_sequentially(on_engine, prompts) == off
    assert on_engine.scheduler.block_manager.hit_rate > 0


def test_second_request_prefills_only_new_tokens(
    loaded: LoadedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompts = _prompts(tokenizer)
    engine = _engine(loaded, prefix_caching=True)
    starts = _record_prefill_starts(engine)
    _run_sequentially(engine, prompts)
    block_size = engine.config.block_size
    assert starts[0] == [0]
    for i in (1, 2):
        shared = _common_prefix_len(prompts[0], prompts[i])
        assert shared > 10 * block_size, "the test prompts must share a long prefix"
        assert starts[i] == [shared // block_size * block_size]


def test_concurrent_requests_with_shared_prefix_match_without_caching(
    loaded: LoadedModel, tokenizer: PreTrainedTokenizerBase
) -> None:
    prompts = _prompts(tokenizer)
    results = {}
    for caching in (False, True):
        engine = _engine(loaded, prefix_caching=caching)
        out: dict[str, list[int]] = {}
        for i, prompt in enumerate(prompts):
            engine.add_request(f"r{i}", prompt, PARAMS)
        while engine.has_unfinished_requests():
            _collect(out, engine.step())
        results[caching] = out
    assert results[True] == results[False]


@pytest.mark.parametrize("caching", [False, True])
def test_blocks_all_returned_after_requests_finish(
    loaded: LoadedModel, tokenizer: PreTrainedTokenizerBase, caching: bool
) -> None:
    engine = _engine(loaded, prefix_caching=caching)
    _run_sequentially(engine, _prompts(tokenizer))
    bm = engine.scheduler.block_manager
    assert bm.num_free_blocks == bm.num_blocks
    assert not bm.block_tables
    assert all(c == 0 for c in bm.ref_counts)
