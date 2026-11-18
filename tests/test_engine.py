import pytest

from minivllm.config import EngineConfig
from minivllm.engine import Engine, generate
from minivllm.model_files import LoadedModel
from minivllm.sequence import RequestOutput, SamplingParams
from tests.conftest import TEST_MODEL

SHORT = 16
GREEDY_SHORT = SamplingParams(temperature=0.0, max_tokens=SHORT)
SUBSET = [0, 2, 7, 13, 17, 19]


def _engine(loaded: LoadedModel, max_num_seqs: int = 16) -> Engine:
    config = EngineConfig(
        model=TEST_MODEL, max_model_len=1024, max_num_batched_tokens=2048, max_num_seqs=max_num_seqs
    )
    return Engine(config, loaded)


def _collect(outputs: dict[str, list[int]], step_outputs: list[RequestOutput]) -> None:
    for out in step_outputs:
        outputs.setdefault(out.request_id, []).extend(out.new_token_ids)


def test_staggered_arrivals_match_hf(
    loaded: LoadedModel, prompt_ids: list[list[int]], hf_outputs: list[list[int]]
) -> None:
    engine = _engine(loaded)
    outputs: dict[str, list[int]] = {}
    arrivals = {0: SUBSET[:2], 3: SUBSET[2:4], 7: SUBSET[4:]}
    step = 0
    while step <= max(arrivals) or engine.has_unfinished_requests():
        for i in arrivals.get(step, []):
            engine.add_request(str(i), prompt_ids[i], GREEDY_SHORT)
        _collect(outputs, engine.step())
        step += 1
    for i in SUBSET:
        # Greedy decoding is deterministic, so the first SHORT tokens of HF's 64 token
        # reference are exactly what a SHORT token request must produce.
        assert outputs[str(i)] == hf_outputs[i][:SHORT], f"prompt {i}"


def test_abort_mid_generation_frees_everything(
    loaded: LoadedModel, prompt_ids: list[list[int]]
) -> None:
    engine = _engine(loaded, max_num_seqs=2)
    long_params = SamplingParams(temperature=0.0, max_tokens=200, ignore_eos=True)
    for i in range(4):
        engine.add_request(f"r{i}", prompt_ids[i], long_params)
    outputs: dict[str, list[int]] = {}
    for _ in range(5):
        _collect(outputs, engine.step())
    running = [s.request_id for s in engine.scheduler.running]
    waiting = [s.request_id for s in engine.scheduler.waiting]
    assert running and waiting
    for request_id in running + waiting:
        assert engine.abort_request(request_id)
    assert not engine.has_unfinished_requests()
    assert sorted(engine.scheduler.free_slots) == list(range(2))
    assert not engine.scheduler.slot_of
    assert engine.step() == []
    assert all(len(tokens) < 200 for tokens in outputs.values())


def test_batched_outputs_equal_each_request_alone(
    loaded: LoadedModel, prompt_ids: list[list[int]]
) -> None:
    prompts = [prompt_ids[i] for i in SUBSET]
    alone = [generate(_engine(loaded), [p], [GREEDY_SHORT])[0] for p in prompts]
    together = generate(_engine(loaded), prompts, [GREEDY_SHORT] * len(prompts))
    assert together == alone


@pytest.mark.parametrize("bad_len", [0, 1024])
def test_rejects_prompts_that_cannot_generate(loaded: LoadedModel, bad_len: int) -> None:
    with pytest.raises(ValueError):
        _engine(loaded).add_request("x", [1] * bad_len, GREEDY_SHORT)
