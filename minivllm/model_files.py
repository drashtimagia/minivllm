import json
from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from minivllm.config import EngineConfig
from minivllm.models.qwen2 import Qwen2ForCausalLM, load_qwen2

WEIGHT_FILE_PATTERNS = ["*.json", "*.safetensors", "merges.txt", "vocab.json"]


def resolve_model_dir(model: str) -> Path:
    """Accept either a local checkpoint directory or a Hugging Face repo id."""
    local = Path(model).expanduser()
    if local.is_dir():
        return local
    return Path(snapshot_download(model, allow_patterns=WEIGHT_FILE_PATTERNS))


def load_tokenizer(model_dir: Path) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(model_dir)


def load_eos_token_ids(model_dir: Path) -> frozenset[int]:
    # Instruct checkpoints list several stop tokens (end of turn and end of text) in
    # generation_config.json, while config.json often names only one.
    for name in ("generation_config.json", "config.json"):
        path = model_dir / name
        if path.exists():
            eos = json.loads(path.read_text()).get("eos_token_id")
            if eos is not None:
                return frozenset(eos if isinstance(eos, list) else [eos])
    raise ValueError(f"no eos_token_id found in {model_dir}")


@dataclass(frozen=True)
class LoadedModel:
    model: Qwen2ForCausalLM
    tokenizer: PreTrainedTokenizerBase
    eos_token_ids: frozenset[int]


def load_model(config: EngineConfig) -> LoadedModel:
    model_dir = resolve_model_dir(config.model)
    return LoadedModel(
        model=load_qwen2(model_dir, config.torch_dtype, torch.device(config.device), config.max_model_len),
        tokenizer=load_tokenizer(model_dir),
        eos_token_ids=load_eos_token_ids(model_dir),
    )
