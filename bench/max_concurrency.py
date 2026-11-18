import argparse
import dataclasses
import random
from pathlib import Path

import torch

from bench.common import RESULTS_DIR, write_result
from minivllm.config import EngineConfig
from minivllm.engine import Engine
from minivllm.model_files import load_model
from minivllm.sequence import SamplingParams


def random_prompts(num: int, length: int, vocab_size: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(vocab_size) for _ in range(length)] for _ in range(num)]


def measure_peak_running(engine: Engine, prompts: list[list[int]], output_len: int) -> dict[str, int]:
    params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)
    for i, prompt in enumerate(prompts):
        engine.add_request(f"r{i}", prompt, params)
    peak = steps = 0
    while engine.has_unfinished_requests():
        engine.step()
        steps += 1
        peak = max(peak, len(engine.scheduler.running))
    return {"peak_running_seqs": peak, "steps": steps, "num_preemptions": engine.scheduler.num_preemptions}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Peak number of sequences running at once under a fixed KV budget, "
        "for a contiguous reservation per sequence versus paged blocks"
    )
    EngineConfig.add_cli_args(p)
    p.add_argument("--num-requests", type=int, default=256)
    p.add_argument("--prompt-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--paged-block-size", type=int, default=16)
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    base = EngineConfig.from_cli_args(args)
    # Admission is limited by memory, not by the sequence cap, so the cap must not bind.
    base = dataclasses.replace(base, max_num_seqs=args.num_requests, max_num_batched_tokens=base.max_model_len)
    loaded = load_model(base)
    prompts = random_prompts(args.num_requests, args.prompt_len, loaded.model.config.vocab_size, base.seed)
    layouts = {"contiguous": base.max_model_len, "paged": args.paged_block_size}
    for name, block_size in layouts.items():
        config = dataclasses.replace(base, block_size=block_size)
        result = measure_peak_running(Engine(config, loaded), prompts, args.output_len)
        written = write_result(
            args.out_dir, f"max_concurrency_{name}.json", "max_concurrency",
            {**dataclasses.asdict(config), "layout": name, "num_requests": args.num_requests,
             "prompt_len": args.prompt_len, "output_len": args.output_len},
            result, args.allow_dirty,
        )
        print(written)
        if base.device.startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
