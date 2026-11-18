import argparse

import torch

from minivllm.config import EngineConfig
from minivllm.kv_cache import ContiguousKVCache
from minivllm.model_files import load_eos_token_ids, load_tokenizer, resolve_model_dir
from minivllm.models.qwen2 import Qwen2ForCausalLM, load_qwen2
from minivllm.sampler import sample
from minivllm.sequence import SamplingParams


class Generator:
    """Runs one request at a time: prefill the prompt, then decode token by token."""

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

    @torch.inference_mode()
    def generate(self, prompt_ids: list[int], params: SamplingParams) -> list[int]:
        max_new = min(params.max_tokens, self.config.max_model_len - len(prompt_ids))
        if max_new < 1:
            raise ValueError(
                f"prompt has {len(prompt_ids)} tokens; max_model_len is {self.config.max_model_len}"
            )
        cache = self._new_cache()
        output: list[int] = []
        next_input = prompt_ids
        while len(output) < max_new:
            token = self._step(next_input, cache, params)
            output.append(token)
            if token in self.eos_token_ids and not params.ignore_eos:
                break
            next_input = [token]
        return output

    def _step(self, input_ids: list[int], cache: ContiguousKVCache, params: SamplingParams) -> int:
        ids = torch.tensor(input_ids, device=self.device)
        positions = torch.arange(cache.num_cached, cache.num_cached + len(input_ids), device=self.device)
        hidden = self.model(ids, positions, cache)
        cache.advance(len(input_ids))
        logits = self.model.compute_logits(hidden[-1:])
        return int(sample(logits, [params], self.rng)[0])

    def _new_cache(self) -> ContiguousKVCache:
        mc = self.model.config
        return ContiguousKVCache(
            num_layers=mc.num_hidden_layers,
            max_len=self.config.max_model_len,
            num_kv_heads=mc.num_key_value_heads,
            head_dim=mc.head_dim,
            dtype=self.config.torch_dtype,
            device=self.device,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a completion for one prompt")
    EngineConfig.add_cli_args(parser)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    generator = Generator(EngineConfig.from_cli_args(args))
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    output = generator.generate(generator.tokenizer.encode(args.prompt), params)
    print(generator.tokenizer.decode(output))


if __name__ == "__main__":
    main()
