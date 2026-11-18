from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from minivllm.config import EngineConfig
from minivllm.model_files import LoadedModel, load_eos_token_ids, load_model, load_tokenizer, resolve_model_dir
from tests.hf_reference import hf_greedy_outputs
from tests.prompts import PROMPTS, encode_prompt

TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 64


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="needs a CUDA GPU")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def model_dir() -> Path:
    return resolve_model_dir(TEST_MODEL)


@pytest.fixture(scope="session")
def tokenizer(model_dir: Path) -> PreTrainedTokenizerBase:
    return load_tokenizer(model_dir)


@pytest.fixture(scope="session")
def eos_token_ids(model_dir: Path) -> frozenset[int]:
    return load_eos_token_ids(model_dir)


@pytest.fixture(scope="session")
def prompt_ids(tokenizer: PreTrainedTokenizerBase) -> list[list[int]]:
    return [encode_prompt(tokenizer, p) for p in PROMPTS]


@pytest.fixture(scope="session")
def hf_outputs(
    model_dir: Path, prompt_ids: list[list[int]], eos_token_ids: frozenset[int]
) -> list[list[int]]:
    return hf_greedy_outputs(model_dir, prompt_ids, MAX_NEW_TOKENS, eos_token_ids)


@pytest.fixture(scope="session")
def loaded(hf_outputs: list[list[int]]) -> LoadedModel:
    # Depends on hf_outputs so the HF reference model is loaded, used and freed before
    # this model is loaded; two fp32 copies at once do not fit comfortably in 8 GB.
    return load_model(EngineConfig(model=TEST_MODEL, dtype="float32", device="cpu"))
