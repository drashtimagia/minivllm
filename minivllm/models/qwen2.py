import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from minivllm.kv_cache import CacheAttention


@dataclass(frozen=True)
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> "Qwen2Config":
        raw = json.loads((model_dir / "config.json").read_text())
        if raw["model_type"] != "qwen2":
            raise ValueError(f"expected a qwen2 checkpoint, got model_type={raw['model_type']!r}")
        return cls(**{name: raw[name] for name in cls.__dataclass_fields__})


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The variance is computed in fp32 even for bf16 weights; HF does the same and
        # matching its rounding order is what lets greedy outputs agree token for token.
        input_dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, max_positions: int) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        freqs = torch.arange(max_positions, dtype=torch.float)[:, None] * inv_freq[None, :]
        # Rotating the two halves of head_dim (not interleaved pairs) is the HF/Qwen layout.
        angles = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[positions].to(q.dtype)[:, None, :]
        sin = self.sin[positions].to(q.dtype)[:, None, :]
        return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Qwen2Attention(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        rotary: RotaryEmbedding,
        kv_cache: CacheAttention,
    ) -> torch.Tensor:
        num_tokens = hidden.shape[0]
        q = self.q_proj(hidden).view(num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(num_tokens, self.num_kv_heads, self.head_dim)
        q, k = rotary(positions, q, k)
        out = kv_cache.attend(self.layer_idx, q, k, v)
        return self.o_proj(out.reshape(num_tokens, self.num_heads * self.head_dim))


class Qwen2MLP(nn.Module):
    def __init__(self, config: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Qwen2Attention(config, layer_idx)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Qwen2MLP(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        rotary: RotaryEmbedding,
        kv_cache: CacheAttention,
    ) -> torch.Tensor:
        hidden = hidden + self.self_attn(positions, self.input_layernorm(hidden), rotary, kv_cache)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config, max_positions: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta, max_positions)

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, kv_cache: CacheAttention
    ) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden = layer(positions, hidden, self.rotary, kv_cache)
        return self.norm(hidden)


class Qwen2ForCausalLM(nn.Module):
    """Tokens are always passed flattened as [num_tokens] so that batched, padded and
    variable-length callers share one forward; the KV cache knows how to regroup them."""

    def __init__(self, config: Qwen2Config, max_positions: int) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config, max_positions)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self, input_ids: torch.Tensor, positions: torch.Tensor, kv_cache: CacheAttention
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_cache)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        # Callers pass only the rows they sample from, which skips the vocab projection
        # (the largest matmul in a 0.5B model) for every prompt token but the last.
        return self.lm_head(hidden)


def load_qwen2(
    model_dir: Path, dtype: torch.dtype, device: torch.device, max_positions: int
) -> Qwen2ForCausalLM:
    config = Qwen2Config.from_model_dir(model_dir)
    with torch.device("meta"):
        model = Qwen2ForCausalLM(config, max_positions)
    state = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        state.update(load_file(shard))
    if config.tie_word_embeddings:
        state.pop("lm_head.weight", None)
    # assign=True swaps the meta parameters for the loaded tensors instead of copying,
    # so peak memory is one copy of the weights, not two.
    model.load_state_dict({k: v.to(dtype) for k, v in state.items()}, strict=False, assign=True)
    if config.tie_word_embeddings:
        model.lm_head.weight = model.model.embed_tokens.weight
    _check_all_weights_loaded(model, state.keys())
    # Buffers built on meta have no data; rebuild the rotary tables on the real device.
    model.model.rotary = RotaryEmbedding(config.head_dim, config.rope_theta, max_positions)
    return model.to(device).eval()


def _check_all_weights_loaded(model: nn.Module, loaded: Iterable[str]) -> None:
    loaded_names = set(loaded)
    expected = {name for name, _ in model.named_parameters()}
    missing = expected - loaded_names
    unexpected = loaded_names - expected
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}")
