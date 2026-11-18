import argparse

from minivllm.config import EngineConfig
from minivllm.engine import create_engine, generate
from minivllm.sequence import SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate completions for a batch of prompts")
    EngineConfig.add_cli_args(parser)
    parser.add_argument("--prompt", action="append", required=True, help="repeat for a batch")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    args = parser.parse_args()
    engine = create_engine(EngineConfig.from_cli_args(args))
    tokenizer = engine.tokenizer
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
    prompts = [tokenizer.encode(p) for p in args.prompt]
    for output in generate(engine, prompts, [params] * len(prompts)):
        print(tokenizer.decode(output))


if __name__ == "__main__":
    main()
