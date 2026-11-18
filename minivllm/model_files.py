import json
from pathlib import Path

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
