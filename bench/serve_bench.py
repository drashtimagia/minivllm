import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import torch

from bench.common import RESULTS_DIR, kv_blocks_for_budget, write_result

SYSTEMS = (
    "static", "continuous", "paged", "paged_prefix", "paged_chunked", "paged_graphs", "paged_chunked_graphs",
    "vllm", "vllm_eager", "vllm_prefix",
)
WORKLOADS = ("sharegpt", "agent")
VLLM_BLOCK_SIZE = 16
READY_TIMEOUT_S = 900
READY_POLL_S = 2.0


def build_server_cmd(system: str, args: argparse.Namespace) -> list[str]:
    if system.startswith("vllm"):
        cmd = [
            "vllm", "serve", args.model, "--dtype", args.dtype, "--port", str(args.port),
            "--max-model-len", str(args.max_model_len), "--max-num-batched-tokens", str(args.max_model_len),
            "--max-num-seqs", str(args.max_num_seqs), "--block-size", str(VLLM_BLOCK_SIZE),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            "--num-gpu-blocks-override",
            str(kv_blocks_for_budget(args.model, getattr(torch, args.dtype), VLLM_BLOCK_SIZE, args.kv_budget_gb)),
            "--enable-prefix-caching" if system == "vllm_prefix" else "--no-enable-prefix-caching",
        ]
        return cmd + (["--enforce-eager"] if system == "vllm_eager" else [])
    # One block as large as a whole sequence is the contiguous reservation that
    # continuous batching used before paging was added.
    block_size = {"continuous": args.max_model_len}.get(system, VLLM_BLOCK_SIZE)
    chunked = "chunked" in system
    cmd = [
        sys.executable, "-m", "minivllm.server", "--model", args.model, "--dtype", args.dtype,
        "--device", "cuda", "--port", str(args.port), "--served-model-name", args.model,
        "--max-model-len", str(args.max_model_len),
        "--max-num-batched-tokens", str(args.chunk_budget if chunked else args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs), "--kv-budget-gb", str(args.kv_budget_gb),
        "--block-size", str(block_size), "--attention-backend", "flash",
    ]
    if system == "static":
        return cmd + ["--scheduler", "static"]
    flags = {
        "paged_prefix": ["--enable-prefix-caching"],
        "paged_chunked": ["--enable-chunked-prefill"],
        "paged_graphs": ["--enable-cuda-graphs"],
        "paged_chunked_graphs": ["--enable-chunked-prefill", "--enable-cuda-graphs"],
    }
    return cmd + flags.get(system, [])


def build_client_cmd(workload: str, rate: str, args: argparse.Namespace, result_dir: Path) -> list[str]:
    cmd = [
        "vllm", "bench", "serve", "--backend", "openai", "--base-url", f"http://127.0.0.1:{args.port}",
        "--endpoint", "/v1/completions", "--model", args.model, "--tokenizer", args.model,
        "--num-prompts", str(args.num_prompts), "--request-rate", rate, "--seed", str(args.seed),
        "--ignore-eos", "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
        "--save-result", "--result-dir", str(result_dir), "--result-filename", "client.json",
    ]
    if workload == "sharegpt":
        return cmd + [
            "--dataset-name", "sharegpt", "--dataset-path", args.dataset_path,
            "--sharegpt-output-len", str(args.output_len),
        ]
    return cmd + [
        "--dataset-name", "prefix_repetition",
        "--prefix-repetition-prefix-len", str(args.agent_prefix_len),
        "--prefix-repetition-suffix-len", str(args.agent_suffix_len),
        "--prefix-repetition-num-prefixes", "1",
        "--prefix-repetition-output-len", str(args.output_len),
    ]


def wait_until_ready(port: int, server: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"server exited early with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5):
                return
        except OSError:
            time.sleep(READY_POLL_S)
    raise TimeoutError(f"server not healthy after {READY_TIMEOUT_S}s")


def run_one(system: str, workload: str, rate: str, args: argparse.Namespace) -> Path:
    # A fresh server per run keeps one run's prefix cache and warmed state from
    # leaking into the next.
    server_cmd = build_server_cmd(system, args)
    server = subprocess.Popen(server_cmd, start_new_session=True)
    try:
        wait_until_ready(args.port, server)
        with tempfile.TemporaryDirectory() as tmp:
            client_cmd = build_client_cmd(workload, rate, args, Path(tmp))
            subprocess.run(client_cmd, check=True)
            client_result: dict[str, Any] = json.loads((Path(tmp) / "client.json").read_text())
    finally:
        server.terminate()
        server.wait(timeout=60)
    config = {
        "system": system, "workload": workload, "request_rate": rate, "model": args.model, "dtype": args.dtype,
        "max_model_len": args.max_model_len, "max_num_seqs": args.max_num_seqs, "kv_budget_gb": args.kv_budget_gb,
        "output_len": args.output_len, "num_prompts": args.num_prompts, "seed": args.seed,
        "server_cmd": server_cmd, "client_cmd": client_cmd,
    }
    return write_result(
        args.out_dir, f"{workload}_{system}_rate{rate}.json", "serving", config, client_result, args.allow_dirty
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run vllm bench serve against one system, once per request rate")
    p.add_argument("--system", choices=SYSTEMS, required=True)
    p.add_argument("--workload", choices=WORKLOADS, required=True)
    p.add_argument("--rates", default="1,2,4,8,16,32,inf", help="comma separated request rates (req/s)")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--kv-budget-gb", type=float, default=12.0)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--chunk-budget", type=int, default=2048, help="tokens per step for the chunked systems")
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--num-prompts", type=int, default=500)
    p.add_argument("--dataset-path", help="ShareGPT json file (sharegpt workload)")
    p.add_argument("--agent-prefix-len", type=int, default=3072, help="shared system prompt tokens (agent workload)")
    p.add_argument("--agent-suffix-len", type=int, default=128, help="varied user turn tokens (agent workload)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args(argv)
    if args.workload == "sharegpt" and not args.dataset_path:
        p.error("--dataset-path is required for the sharegpt workload")
    return args


def main() -> None:
    args = parse_args()
    for rate in args.rates.split(","):
        print(run_one(args.system, args.workload, rate, args))


if __name__ == "__main__":
    main()
