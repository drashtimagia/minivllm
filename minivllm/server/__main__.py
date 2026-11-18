import argparse

import uvicorn

from minivllm.config import EngineConfig
from minivllm.engine import create_engine
from minivllm.server.app import create_app
from minivllm.server.async_engine import AsyncEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an OpenAI compatible API")
    EngineConfig.add_cli_args(parser)
    config = EngineConfig.from_cli_args(parser.parse_args())
    engine = create_engine(config)
    app = create_app(config, AsyncEngine(engine), engine.tokenizer)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
