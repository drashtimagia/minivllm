import json
from pathlib import Path

import pytest
import torch
import transformers

from minivllm.config import EngineConfig
from minivllm.engine import Engine, generate
from minivllm.kv_cache import PaddedKVCache
from minivllm.model_files import LoadedModel
from minivllm.models.qwen2 import Qwen2Config, load_qwen2
from minivllm.sequence import SamplingParams

VOCAB = 128
NEW_TOKENS = 12


def _tiny_llama(tmp_path: Path, tie: bool) -> tuple[Path, transformers.LlamaForCausalLM]:
    cfg = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        rope_theta=10000.0, rms_norm_eps=1e-5, tie_word_embeddings=tie,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(cfg).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)
    return tmp_path, model


@pytest.mark.parametrize("tie", [False, True])
def test_llama_logits_match_hf(tmp_path: Path, tie: bool) -> None:
    model_dir, hf = _tiny_llama(tmp_path, tie)
    ours = load_qwen2(model_dir, torch.float32, torch.device("cpu"), max_positions=128)
    assert not ours.config.attention_bias
    ids = torch.randint(0, VOCAB, (23,), generator=torch.Generator().manual_seed(1))
    cfg = ours.config
    cache = PaddedKVCache(cfg.num_hidden_layers, 1, 23, cfg.num_key_value_heads, cfg.head_dim,
                          torch.float32, torch.device("cpu"))
    with torch.inference_mode():
        mine = ours.compute_logits(ours(ids, torch.arange(23), cache))
        reference = hf(ids[None]).logits[0]
    torch.testing.assert_close(mine, reference, atol=1e-4, rtol=1e-4)


def test_llama_greedy_generation_through_the_engine_matches_hf(tmp_path: Path) -> None:
    model_dir, hf = _tiny_llama(tmp_path, tie=False)
    model = load_qwen2(model_dir, torch.float32, torch.device("cpu"), max_positions=128)
    config = EngineConfig(model=str(model_dir), max_model_len=64, max_num_batched_tokens=64, kv_budget_gb=0.001)
    # No tokenizer exists for a random model, and no token ends generation.
    loaded = LoadedModel(model=model, tokenizer=None, eos_token_ids=frozenset())  # type: ignore[arg-type]
    prompts = [torch.randint(0, VOCAB, (n,), generator=torch.Generator().manual_seed(n)).tolist() for n in (5, 17, 30)]
    params = [SamplingParams(temperature=0.0, max_tokens=NEW_TOKENS, ignore_eos=True)] * len(prompts)
    outputs = generate(Engine(config, loaded), prompts, params)
    with torch.inference_mode():
        for prompt, out in zip(prompts, outputs):
            ids = torch.tensor([prompt])
            ref = hf.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=NEW_TOKENS,
                              do_sample=False, pad_token_id=0, eos_token_id=None)
            assert out == ref[0, len(prompt):].tolist()


def test_rope_scaling_is_rejected_loudly(tmp_path: Path) -> None:
    model_dir, _ = _tiny_llama(tmp_path, tie=False)
    raw = json.loads((model_dir / "config.json").read_text())
    raw["rope_scaling"] = {"rope_type": "llama3", "factor": 8.0}
    (model_dir / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="rope_scaling"):
        Qwen2Config.from_model_dir(model_dir)
