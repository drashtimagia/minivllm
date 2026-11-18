import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass(frozen=True)
class EngineConfig:
    model: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "Hugging Face repo id or local checkpoint directory"},
    )
    dtype: str = field(
        default="float32", metadata={"help": "weight and activation dtype", "choices": list(DTYPES)}
    )
    device: str = field(default="cpu", metadata={"help": "torch device, e.g. cpu or cuda"})
    max_model_len: int = field(
        default=2048, metadata={"help": "max prompt plus generated tokens per sequence"}
    )
    seed: int = field(default=0, metadata={"help": "seed for the sampling generator"})
    scheduler: str = field(
        default="continuous",
        metadata={
            "help": "static: fixed batches run to completion; continuous: join and leave per step",
            "choices": ["static", "continuous"],
        },
    )
    block_size: int = field(default=16, metadata={"help": "tokens per KV cache block"})
    kv_budget_gb: float = field(
        default=1.0, metadata={"help": "memory for the paged KV cache, in GB (1e9 bytes)"}
    )
    enable_prefix_caching: bool = field(
        default=False, metadata={"help": "reuse KV blocks of identical prompt prefixes across requests"}
    )
    attention_backend: str = field(
        default="naive",
        metadata={"help": "naive: gather + SDPA; flash: flash_attn_with_kvcache (CUDA)", "choices": ["naive", "flash"]},
    )
    host: str = field(default="127.0.0.1", metadata={"help": "server bind address"})
    port: int = field(default=8000, metadata={"help": "server port"})
    served_model_name: str = field(
        default="", metadata={"help": "model id reported by the API; defaults to --model"}
    )
    disconnect_poll_interval_s: float = field(
        default=0.5, metadata={"help": "how often a waiting request checks that its client is still connected"}
    )
    enable_chunked_prefill: bool = field(
        default=False, metadata={"help": "split long prompts into chunks that share a step with decodes"}
    )
    enable_cuda_graphs: bool = field(
        default=False, metadata={"help": "replay captured CUDA graphs for decode steps (needs cuda and flash)"}
    )
    cuda_graph_max_batch_size: int = field(
        default=64, metadata={"help": "largest decode batch that has a captured graph"}
    )
    max_num_seqs: int = field(default=16, metadata={"help": "max sequences in one batch"})
    max_num_batched_tokens: int = field(
        default=4096, metadata={"help": "max prompt tokens prefilled in one step"}
    )

    def __post_init__(self) -> None:
        # Without chunked prefill a prompt is prefilled in one step, so the per-step
        # token budget must fit the longest allowed prompt or that prompt never runs.
        if not self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_model_len ({self.max_model_len}) unless --enable-chunked-prefill is set"
            )
        # Every running sequence needs one token of budget per step to decode.
        if self.enable_chunked_prefill and self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must be >= "
                f"max_num_seqs ({self.max_num_seqs}) with chunked prefill"
            )
        if self.enable_cuda_graphs and not (self.device.startswith("cuda") and self.attention_backend == "flash"):
            raise ValueError("--enable-cuda-graphs needs --device cuda and --attention-backend flash")

    @property
    def torch_dtype(self) -> torch.dtype:
        return DTYPES[self.dtype]

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        for f in dataclasses.fields(cls):
            flag = "--" + f.name.replace("_", "-")
            kwargs: dict[str, Any] = {"default": f.default, "help": f.metadata["help"]}
            if f.type is bool:
                kwargs["action"] = argparse.BooleanOptionalAction
            else:
                kwargs["type"] = f.type
                if "choices" in f.metadata:
                    kwargs["choices"] = f.metadata["choices"]
            parser.add_argument(flag, **kwargs)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "EngineConfig":
        return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})
