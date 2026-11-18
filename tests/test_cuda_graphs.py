import pytest
import torch

from minivllm.config import EngineConfig
from minivllm.cuda_graphs import DecodeInputs, graph_buckets, pad_decode_inputs, pick_bucket
from minivllm.engine import Engine, generate
from minivllm.model_files import load_model
from minivllm.sequence import SamplingParams
from tests.conftest import TEST_MODEL


def test_buckets_are_powers_of_two_plus_the_cap() -> None:
    assert graph_buckets(64) == [1, 2, 4, 8, 16, 32, 64]
    assert graph_buckets(48) == [1, 2, 4, 8, 16, 32, 48]
    assert graph_buckets(1) == [1]


def test_batch_runs_in_the_smallest_bucket_that_holds_it() -> None:
    buckets = graph_buckets(32)
    assert [pick_bucket(buckets, n) for n in (1, 2, 3, 9, 32)] == [1, 2, 4, 16, 32]
    assert pick_bucket(buckets, 33) is None


def test_padding_rows_only_touch_the_scratch_block() -> None:
    inputs = DecodeInputs(
        input_ids=[7, 8], positions=[5, 20], slot_mapping=[37, 90], block_tables=[[2, 5], [1, 4, 6]], context_lens=[6, 21]
    )
    padded = pad_decode_inputs(inputs, bucket=4, scratch_block=99, block_size=16, table_width=4)
    assert padded.input_ids == [7, 8, 0, 0] and padded.positions == [5, 20, 0, 0]
    assert padded.slot_mapping == [37, 90, 99 * 16, 99 * 16]
    assert padded.block_tables == [[2, 5, 99, 99], [1, 4, 6, 99], [99] * 4, [99] * 4]
    assert padded.context_lens == [6, 21, 1, 1]
    assert all(len(row) == 4 for row in padded.block_tables)


@pytest.mark.gpu
def test_graph_decode_matches_eager_decode(prompt_ids: list[list[int]]) -> None:
    common = dict(model=TEST_MODEL, device="cuda", dtype="bfloat16", attention_backend="flash", max_model_len=1024)
    loaded = load_model(EngineConfig(**common))  # type: ignore[arg-type]
    params = [SamplingParams(temperature=0.0, max_tokens=32)] * 6
    prompts = prompt_ids[:6]
    eager = generate(Engine(EngineConfig(**common), loaded), prompts, params)  # type: ignore[arg-type]
    graphs = generate(Engine(EngineConfig(**common, enable_cuda_graphs=True), loaded), prompts, params)  # type: ignore[arg-type]
    assert graphs == eager
