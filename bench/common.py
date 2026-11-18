import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

from minivllm.kv_cache import kv_bytes_per_block
from minivllm.model_files import resolve_model_dir
from minivllm.models.qwen2 import Qwen2Config

RESULTS_DIR = Path(__file__).resolve().parent / "results"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_state() -> tuple[str, bool]:
    commit = _run(["git", "rev-parse", "HEAD"])
    if commit is None:
        raise RuntimeError("not inside a git checkout; results must record a commit hash")
    return commit, bool(_run(["git", "status", "--porcelain"]))


def hardware() -> dict[str, Any]:
    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    return {
        "gpu": gpu.splitlines() if gpu else None,
        "cpu": platform.processor() or platform.machine(),
        "platform": platform.platform(),
    }


def software() -> dict[str, Any]:
    def version(package: str) -> str | None:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": version("vllm"),
        "flash_attn": version("flash-attn"),
    }


def write_result(
    out_dir: Path, filename: str, kind: str, config: dict[str, Any], result: dict[str, Any], allow_dirty: bool
) -> Path:
    """Results are only worth comparing if they can be tied to the code that produced
    them, so a run from a modified working tree is refused unless explicitly allowed
    (and then marked dirty)."""
    commit, dirty = git_state()
    if dirty and not allow_dirty:
        raise RuntimeError("the working tree has uncommitted changes; commit them or pass --allow-dirty")
    payload = {
        "meta": {
            "kind": kind,
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "commit": commit,
            "dirty": dirty,
            "hardware": hardware(),
            "software": software(),
            "argv": sys.argv,
        },
        "config": config,
        "result": result,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps(payload, indent=2))
    return path


def kv_blocks_for_budget(model: str, dtype: torch.dtype, block_size: int, kv_budget_gb: float) -> int:
    """How many KV blocks a byte budget buys, so that vLLM can be given the same pool
    size as --kv-budget-gb gives minivllm."""
    cfg = Qwen2Config.from_model_dir(resolve_model_dir(model))
    per_block = kv_bytes_per_block(cfg.num_hidden_layers, block_size, cfg.num_key_value_heads, cfg.head_dim, dtype)
    return int(kv_budget_gb * 1e9 // per_block)


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(results_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"no result files in {results_dir}; run the benchmarks first")
    return [json.loads(p.read_text()) for p in paths]


def rate_key(rate: str) -> float:
    return float("inf") if rate == "inf" else float(rate)


def of_kind(results: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [r for r in results if r["meta"]["kind"] == kind]


def serving_series(results: list[dict[str, Any]], workload: str, system: str, metric: str) -> list[tuple[str, float]]:
    """(request rate, metric) points for one system on one workload, slowest rate first."""
    points = [
        (r["config"]["request_rate"], float(r["result"][metric]))
        for r in of_kind(results, "serving")
        if r["config"]["workload"] == workload and r["config"]["system"] == system
    ]
    return sorted(points, key=lambda point: rate_key(point[0]))
