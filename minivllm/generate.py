import argparse

import torch

from minivllm.config import EngineConfig
from minivllm.model_files import load_eos_token_ids, load_tokenizer, resolve_model_dir
from minivllm.models.qwen2 import Qwen2ForCausalLM, load_qwen2
from minivllm.sequence import SamplingParams
from minivllm.static_batch import StaticBatch


class Generator:
    """Offline generation with static batching: each call runs one batch to completion."""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        model_dir = resolve_model_dir(config.model)
        self.tokenizer = load_tokenizer(model_dir)
        self.eos_token_ids = load_eos_token_ids(model_dir)
        self.model: Qwen2ForCausalLM = load_qwen2(
            model_dir, config.torch_dtype, self.device, config.max_model_len
        )
        self.rng = torch.Generator(device=self.device).manual_seed(config.seed)

    def generate(self, prompts: list[list[int]], params: list[SamplingParams]) -> list[list[int]]:
        batch = StaticBatch(
            self.model, prompts, params, self.eos_token_ids, self.config.max_model_len, self.rng
        )
        while not batch.done:
            batch.step()
        return batch.outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate completions for a batch of prompts")
    EngineConfig.add_cli_args(parser)
    parser.add_argument("--prompt", action="append", required=True, help="repeat for a batch")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    generator = Generator(EngineConfig.from_cli_args(args))
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    prompts = [generator.tokenizer.encode(p) for p in args.prompt]
    for output in generator.generate(prompts, [params] * len(prompts)):
        print(generator.tokenizer.decode(output))


if __name__ == "__main__":
    main()
