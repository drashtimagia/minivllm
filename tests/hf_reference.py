import gc
import hashlib
import json
from pathlib import Path

import torch
import transformers

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "hf_reference"


def hf_greedy_outputs(
    model_dir: Path, prompts: list[list[int]], max_new_tokens: int, eos_token_ids: frozenset[int]
) -> list[list[int]]:
    """Greedy continuations from HF transformers, one prompt at a time so that no padding
    is involved. Results are cached on disk because loading a second copy of the model
    and generating takes longer than the rest of the test."""
    key = _cache_key(model_dir, prompts, max_new_tokens, eos_token_ids)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    outputs = _generate(model_dir, prompts, max_new_tokens, eos_token_ids)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outputs))
    return outputs


def _cache_key(
    model_dir: Path, prompts: list[list[int]], max_new_tokens: int, eos_token_ids: frozenset[int]
) -> str:
    payload = json.dumps(
        {
            "model_dir": str(model_dir.resolve()),
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "prompts": prompts,
            "max_new_tokens": max_new_tokens,
            "eos": sorted(eos_token_ids),
        }
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@torch.inference_mode()
def _generate(
    model_dir: Path, prompts: list[list[int]], max_new_tokens: int, eos_token_ids: frozenset[int]
) -> list[list[int]]:
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()
    outputs = []
    for prompt in prompts:
        input_ids = torch.tensor([prompt])
        # The checkpoint's generation_config turns on sampling and a repetition penalty;
        # every one of those is switched off so the reference is plain greedy decoding.
        generated = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            repetition_penalty=1.0,
            eos_token_id=sorted(eos_token_ids),
            pad_token_id=sorted(eos_token_ids)[0],
        )
        outputs.append(generated[0, len(prompt) :].tolist())
    del model
    gc.collect()
    return outputs
