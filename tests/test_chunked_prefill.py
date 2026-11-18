import pytest

from minivllm.engine import Engine, generate
from minivllm.model_files import LoadedModel
from minivllm.sequence import RequestOutput, SamplingParams
from tests.test_engine import GREEDY_SHORT, SHORT, SHORT_PROMPTS, SUBSET, _collect, _engine

CHUNK = 64


def test_chunked_outputs_match_hf_including_prompts_longer_than_the_budget(
    loaded: LoadedModel, prompt_ids: list[list[int]], hf_outputs: list[list[int]]
) -> None:
    assert max(len(prompt_ids[i]) for i in SUBSET) > 10 * CHUNK
    engine = _engine(loaded, chunk_budget=CHUNK)
    outputs = generate(engine, [prompt_ids[i] for i in SUBSET], [GREEDY_SHORT] * len(SUBSET))
    for i, out in zip(SUBSET, outputs):
        assert out == hf_outputs[i][:SHORT], f"prompt {i}"


def test_a_long_prompt_is_prefilled_over_several_steps(
    loaded: LoadedModel, prompt_ids: list[list[int]]
) -> None:
    engine = _engine(loaded, chunk_budget=CHUNK)
    engine.add_request("long", prompt_ids[17], GREEDY_SHORT)
    first_token_step = None
    for step in range(1, 40):
        if any(o.request_id == "long" for o in engine.step()):
            first_token_step = step
            break
    expected_steps = -(-len(prompt_ids[17]) // CHUNK)
    assert first_token_step == expected_steps


def test_running_decode_keeps_producing_tokens_while_a_long_prompt_prefills(
    loaded: LoadedModel, prompt_ids: list[list[int]]
) -> None:
    engine = _engine(loaded, chunk_budget=CHUNK)
    engine.add_request("short", prompt_ids[0], SamplingParams(temperature=0.0, max_tokens=40, ignore_eos=True))
    engine.step()
    engine.add_request("long", prompt_ids[17], GREEDY_SHORT)
    tokens_for_short = 0
    long_started = False
    for _ in range(5):
        outs: list[RequestOutput] = engine.step()
        tokens_for_short += sum(len(o.new_token_ids) for o in outs if o.request_id == "short")
        long_started |= any(o.request_id == "long" for o in outs)
    assert tokens_for_short == 5, "the decode advanced on every step"
    assert not long_started, "the long prompt was still being prefilled"


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_chunked_with_preemption_and_prefix_caching_matches_hf(
    loaded: LoadedModel, prompt_ids: list[list[int]], hf_outputs: list[list[int]], prefix_caching: bool
) -> None:
    engine = _engine(loaded, num_blocks=12, prefix_caching=prefix_caching, chunk_budget=32)
    for i in SHORT_PROMPTS:
        engine.add_request(str(i), prompt_ids[i], GREEDY_SHORT)
    outputs: dict[str, list[int]] = {}
    while engine.has_unfinished_requests():
        _collect(outputs, engine.step())
    assert engine.scheduler.num_preemptions > 0
    for i in SHORT_PROMPTS:
        assert outputs[str(i)] == hf_outputs[i][:SHORT], f"prompt {i}"
    bm = engine.scheduler.block_manager
    assert bm.num_free_blocks == bm.num_blocks


def test_chunked_staggered_arrivals_match_hf(
    loaded: LoadedModel, prompt_ids: list[list[int]], hf_outputs: list[list[int]]
) -> None:
    engine: Engine = _engine(loaded, chunk_budget=CHUNK, prefix_caching=True)
    outputs: dict[str, list[int]] = {}
    arrivals = {0: SUBSET[:2], 2: SUBSET[2:4], 5: SUBSET[4:]}
    step = 0
    while step <= max(arrivals) or engine.has_unfinished_requests():
        for i in arrivals.get(step, []):
            engine.add_request(str(i), prompt_ids[i], GREEDY_SHORT)
        _collect(outputs, engine.step())
        step += 1
    for i in SUBSET:
        assert outputs[str(i)] == hf_outputs[i][:SHORT], f"prompt {i}"
