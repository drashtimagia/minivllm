import pytest
import torch

from minivllm.config import EngineConfig
from minivllm.generate import Generator
from minivllm.sequence import SamplingParams
from tests.conftest import MAX_NEW_TOKENS, TEST_MODEL
from tests.prompts import PROMPTS

GREEDY = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)


@pytest.fixture(scope="module")
def generator(hf_outputs: list[list[int]]) -> Generator:
    # Depends on hf_outputs so the HF reference model is loaded, used and freed before
    # this model is loaded; two fp32 copies at once do not fit comfortably in 8 GB.
    return Generator(EngineConfig(model=TEST_MODEL, dtype="float32", device="cpu"))


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_static_batch_matches_hf(
    generator: Generator, prompt_ids: list[list[int]], hf_outputs: list[list[int]], batch_size: int
) -> None:
    outputs: list[list[int]] = []
    for start in range(0, len(prompt_ids), batch_size):
        chunk = prompt_ids[start : start + batch_size]
        outputs += generator.generate(chunk, [GREEDY] * len(chunk))
    assert_matches_hf(outputs, hf_outputs)


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
    # that would make the comparison above pass vacuously.
    assert sum(len(o) for o in hf_outputs) > len(hf_outputs) * MAX_NEW_TOKENS // 2
    assert torch.tensor([len(o) > 1 for o in hf_outputs]).all()
