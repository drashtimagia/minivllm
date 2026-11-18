import pytest

from minivllm.config import EngineConfig


def test_token_budget_below_model_len_needs_chunked_prefill() -> None:
    with pytest.raises(ValueError, match="enable-chunked-prefill"):
        EngineConfig(max_model_len=2048, max_num_batched_tokens=256)
    EngineConfig(max_model_len=2048, max_num_batched_tokens=256, enable_chunked_prefill=True)


def test_chunked_budget_must_cover_one_token_per_sequence() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        EngineConfig(max_num_batched_tokens=8, max_num_seqs=16, enable_chunked_prefill=True)


def test_cuda_graphs_need_cuda_and_flash() -> None:
    with pytest.raises(ValueError, match="cuda"):
        EngineConfig(enable_cuda_graphs=True)
    with pytest.raises(ValueError, match="flash"):
        EngineConfig(enable_cuda_graphs=True, device="cuda")
    EngineConfig(enable_cuda_graphs=True, device="cuda", attention_backend="flash")
