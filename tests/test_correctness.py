import dataclasses

import pytest
import torch

from minivllm.config import EngineConfig
from minivllm.engine import Engine, generate
from minivllm.model_files import LoadedModel
from minivllm.sequence import SamplingParams
from minivllm.static_batch import StaticEngine
from tests.conftest import MAX_NEW_TOKENS, TEST_MODEL
from tests.prompts import PROMPTS

GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)


BASE_CONFIG = EngineConfig(model=TEST_MODEL, dtype="float32", device="cpu")


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_static_batch_matches_hf(
    loaded: LoadedModel, prompt_ids: list[list[int]], hf_outputs: list[list[int]], batch_size: int
) -> None:
    engine = StaticEngine(
        dataclasses.replace(BASE_CONFIG, scheduler="static", max_num_seqs=batch_size), loaded
    )
    assert_matches_hf(generate(engine, prompt_ids, [GREEDY] * len(prompt_ids)), hf_outputs)


def test_continuous_batching_matches_hf(
    loaded: LoadedModel, prompt_ids: list[list[int]], hf_outputs: list[list[int]]
) -> None:
    engine = Engine(BASE_CONFIG, loaded)
    assert_matches_hf(generate(engine, prompt_ids, [GREEDY] * len(prompt_ids)), hf_outputs)


def assert_matches_hf(outputs: list[list[int]], reference: list[list[int]]) -> None:
    mismatches = [
        f"{PROMPTS[i].name}: {_first_divergence(ours, ref)}"
        for i, (ours, ref) in enumerate(zip(outputs, reference, strict=True))
        if ours != ref
    ]
    assert not mismatches, "\n".join(mismatches)


def _first_divergence(ours: list[int], reference: list[int]) -> str:
    for i, (a, b) in enumerate(zip(ours, reference)):
        if a != b:
            return f"first mismatch at token {i}: ours={a} hf={b}"
    return f"length differs: ours={len(ours)} hf={len(reference)}"


def test_reference_is_not_trivial(hf_outputs: list[list[int]]) -> None:
    # Guards against a broken reference (for example every output stopping at once)
    # that would make the comparisons above pass vacuously.
    assert sum(len(o) for o in hf_outputs) > len(hf_outputs) * MAX_NEW_TOKENS // 2
    assert torch.tensor([len(o) > 1 for o in hf_outputs]).all()
