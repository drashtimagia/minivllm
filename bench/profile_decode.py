import argparse
import dataclasses
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from bench.common import RESULTS_DIR, write_result
from bench.max_concurrency import random_prompts
from minivllm.config import EngineConfig
from minivllm.engine import Engine
from minivllm.model_files import load_model
from minivllm.sequence import SamplingParams

TOP_OPS = 25


def profile_one_decode_step(
    engine: Engine, prompts: list[list[int]], device: str
) -> list[dict[str, float | int | str]]:
    params = SamplingParams(temperature=0.0, max_tokens=10_000, ignore_eos=True)
    for i, prompt in enumerate(prompts):
        engine.add_request(f"r{i}", prompt, params)
    engine.step()  # prefill: admits every sequence
    if len(engine.scheduler.running) != len(prompts):
        raise RuntimeError(f"expected {len(prompts)} running sequences, got {len(engine.scheduler.running)}")
    engine.step()  # warm up the first decode step so one-time costs are not profiled
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.startswith("cuda") else [])
    with profile(activities=activities) as prof:
        engine.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
    sort_key = "self_cuda_time_total" if device.startswith("cuda") else "self_cpu_time_total"
    events = sorted(prof.key_averages(), key=lambda e: getattr(e, sort_key), reverse=True)[:TOP_OPS]
    return [
        {
            "name": e.key,
            "count": e.count,
            "self_cpu_us": e.self_cpu_time_total,
            "self_device_us": getattr(e, "self_device_time_total", 0.0),
        }
        for e in events
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="torch.profiler summary of one decode step")
    EngineConfig.add_cli_args(p)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--prompt-len", type=int, default=512)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    base = EngineConfig.from_cli_args(args)
    config = dataclasses.replace(
        base,
        max_num_seqs=args.batch_size,
        max_num_batched_tokens=max(base.max_model_len, args.batch_size * args.prompt_len),
    )
    loaded = load_model(config)
    prompts = random_prompts(args.batch_size, args.prompt_len, loaded.model.config.vocab_size, config.seed)
    ops = profile_one_decode_step(Engine(config, loaded), prompts, config.device)
    path = write_result(
        args.out_dir, f"profile_decode_batch{args.batch_size}.json", "profile_decode",
        {**dataclasses.asdict(config), "batch_size": args.batch_size, "prompt_len": args.prompt_len},
        {"top_ops": ops}, args.allow_dirty,
    )
    print(path)


if __name__ == "__main__":
    main()
