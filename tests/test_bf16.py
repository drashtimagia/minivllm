import gc
from pathlib import Path

import pytest
import torch
import transformers

from minivllm.kv_cache import PaddedKVCache
from minivllm.models.qwen2 import load_qwen2

PROMPT_INDICES = [1, 3, 7, 13]
MIN_COSINE = 0.999
MIN_TOP1_AGREEMENT = 0.75


def _our_last_logits(model_dir: Path, prompts: list[list[int]]) -> list[torch.Tensor]:
    model = load_qwen2(model_dir, torch.bfloat16, torch.device("cpu"), max_positions=256)
    cfg = model.config
    outs = []
    with torch.inference_mode():
        for ids in prompts:
            cache = PaddedKVCache(cfg.num_hidden_layers, 1, len(ids), cfg.num_key_value_heads, cfg.head_dim,
                                  torch.bfloat16, torch.device("cpu"))
            hidden = model(torch.tensor(ids), torch.arange(len(ids)), cache)
            outs.append(model.compute_logits(hidden[-1:])[0].float())
    del model
    gc.collect()
    return outs


def _hf_last_logits(model_dir: Path, prompts: list[list[int]]) -> list[torch.Tensor]:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16).eval()
    with torch.inference_mode():
        outs = [model(torch.tensor([ids])).logits[0, -1].float() for ids in prompts]
    del model
    gc.collect()
    return outs


def test_bf16_logits_stay_close_to_hf(model_dir: Path, prompt_ids: list[list[int]]) -> None:
    """bf16 reorders floating point work, so greedy tokens may legitimately differ at a
    near-tie. Closeness of the whole logit vector is the right check, with the argmax
    required to agree on most prompts."""
    prompts = [prompt_ids[i][:60] for i in PROMPT_INDICES]
    ours = _our_last_logits(model_dir, prompts)
    reference = _hf_last_logits(model_dir, prompts)
    cosines = [torch.nn.functional.cosine_similarity(a, b, dim=0).item() for a, b in zip(ours, reference)]
    assert min(cosines) > MIN_COSINE, cosines
    agree = sum(int(a.argmax() == b.argmax()) for a, b in zip(ours, reference)) / len(prompts)
    assert agree >= MIN_TOP1_AGREEMENT, f"top-1 agreed on only {agree:.0%} of prompts"
