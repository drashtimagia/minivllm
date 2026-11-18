import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import openai
import pytest
import uvicorn
from transformers import PreTrainedTokenizerBase

from minivllm.config import EngineConfig
from minivllm.engine import Engine
from minivllm.model_files import LoadedModel
from minivllm.server.app import create_app
from minivllm.server.async_engine import AsyncEngine
from tests.conftest import TEST_MODEL

NEW_TOKENS = 16
MODEL_LEN = 1024
SHORT_COMPLETION_PROMPT = 0
SHORT_CHAT_PROMPT = 2


@dataclass
class RunningServer:
    base_url: str
    engine: Engine
    client: openai.OpenAI


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="module")
def server(loaded: LoadedModel) -> Iterator[RunningServer]:
    config = EngineConfig(
        model=TEST_MODEL, max_model_len=MODEL_LEN, max_num_batched_tokens=MODEL_LEN, disconnect_poll_interval_s=0.1
    )
    engine = Engine(config, loaded)
    app = create_app(config, AsyncEngine(engine), loaded.tokenizer)
    port = _free_port()
    uvicorn_server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=uvicorn_server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 30
    while not uvicorn_server.started:
        assert time.monotonic() < deadline, "server did not start"
        time.sleep(0.05)
    base_url = f"http://127.0.0.1:{port}"
    yield RunningServer(base_url, engine, openai.OpenAI(base_url=f"{base_url}/v1", api_key="unused"))
    uvicorn_server.should_exit = True
    thread.join(timeout=30)


def _expected_text(tokenizer: PreTrainedTokenizerBase, hf_outputs: list[list[int]], index: int) -> str:
    return tokenizer.decode(hf_outputs[index][:NEW_TOKENS], skip_special_tokens=True)


def _wait_until_idle(engine: Engine, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while engine.has_unfinished_requests():
        assert time.monotonic() < deadline, "engine still has requests after the client left"
        time.sleep(0.1)
    bm = engine.scheduler.block_manager
    assert bm.num_free_blocks == bm.num_blocks
    assert not bm.block_tables


def test_models_and_health(server: RunningServer) -> None:
    assert [m.id for m in server.client.models.list().data] == [TEST_MODEL]
    assert httpx.get(f"{server.base_url}/health").json() == {"status": "ok"}


def test_completion(server: RunningServer, tokenizer: PreTrainedTokenizerBase, hf_outputs: list[list[int]]) -> None:
    from tests.prompts import PROMPTS

    result = server.client.completions.create(
        model=TEST_MODEL, prompt=PROMPTS[SHORT_COMPLETION_PROMPT].text, max_tokens=NEW_TOKENS, temperature=0
    )
    assert result.choices[0].text == _expected_text(tokenizer, hf_outputs, SHORT_COMPLETION_PROMPT)
    assert result.choices[0].finish_reason == "length"
    assert result.usage is not None and result.usage.completion_tokens == NEW_TOKENS


def test_completion_stream(
    server: RunningServer, tokenizer: PreTrainedTokenizerBase, hf_outputs: list[list[int]]
) -> None:
    from tests.prompts import PROMPTS

    stream = server.client.completions.create(
        model=TEST_MODEL,
        prompt=PROMPTS[SHORT_COMPLETION_PROMPT].text,
        max_tokens=NEW_TOKENS,
        temperature=0,
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks = list(stream)
    text = "".join(c.choices[0].text for c in chunks if c.choices)
    assert text == _expected_text(tokenizer, hf_outputs, SHORT_COMPLETION_PROMPT)
    assert [c.choices[0].finish_reason for c in chunks if c.choices][-1] == "length"
    assert chunks[-1].usage is not None and chunks[-1].usage.completion_tokens == NEW_TOKENS


def test_chat_completion(
    server: RunningServer, tokenizer: PreTrainedTokenizerBase, hf_outputs: list[list[int]]
) -> None:
    from tests.prompts import PROMPTS

    messages = [{"role": "user", "content": PROMPTS[SHORT_CHAT_PROMPT].text}]
    result = server.client.chat.completions.create(
        model=TEST_MODEL, messages=messages, max_tokens=NEW_TOKENS, temperature=0
    )
    expected = _expected_text(tokenizer, hf_outputs, SHORT_CHAT_PROMPT)
    assert result.choices[0].message.content == expected
    assert result.choices[0].message.role == "assistant"


def test_chat_completion_stream(
    server: RunningServer, tokenizer: PreTrainedTokenizerBase, hf_outputs: list[list[int]]
) -> None:
    from tests.prompts import PROMPTS

    messages = [{"role": "user", "content": PROMPTS[SHORT_CHAT_PROMPT].text}]
    chunks = list(
        server.client.chat.completions.create(
            model=TEST_MODEL, messages=messages, max_tokens=NEW_TOKENS, temperature=0, stream=True
        )
    )
    text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
    assert text == _expected_text(tokenizer, hf_outputs, SHORT_CHAT_PROMPT)
    assert chunks[0].choices[0].delta.role == "assistant"
    assert chunks[-1].choices[0].finish_reason in ("length", "stop")


def test_chat_without_max_tokens_generates_until_stop(server: RunningServer) -> None:
    result = server.client.chat.completions.create(
        model=TEST_MODEL, messages=[{"role": "user", "content": "Say hello in one word."}], temperature=0
    )
    assert result.choices[0].finish_reason == "stop"


def test_request_over_max_model_len_is_a_clear_400(server: RunningServer) -> None:
    with pytest.raises(openai.BadRequestError) as info:
        server.client.completions.create(model=TEST_MODEL, prompt="hello", max_tokens=MODEL_LEN + 1)
    assert info.value.status_code == 400
    assert "maximum context length is 1024" in str(info.value)
    with pytest.raises(openai.BadRequestError):
        server.client.completions.create(model=TEST_MODEL, prompt="word " * 2000, max_tokens=1)


def test_unknown_model_is_404(server: RunningServer) -> None:
    with pytest.raises(openai.NotFoundError):
        server.client.completions.create(model="nope", prompt="hi", max_tokens=1)


def test_unsupported_parameter_is_rejected(server: RunningServer) -> None:
    response = httpx.post(
        f"{server.base_url}/v1/completions", json={"model": TEST_MODEL, "prompt": "hi", "n": 2}
    )
    assert response.status_code == 422


def test_metrics_endpoint(server: RunningServer) -> None:
    server.client.completions.create(model=TEST_MODEL, prompt="hello", max_tokens=4, temperature=0)
    text = httpx.get(f"{server.base_url}/metrics").text
    for name in (
        "minivllm_num_running_seqs",
        "minivllm_num_waiting_seqs",
        "minivllm_kv_cache_usage_ratio",
        "minivllm_prefix_cache_hit_rate",
        "minivllm_time_to_first_token_seconds_bucket",
        "minivllm_time_per_output_token_seconds_bucket",
    ):
        assert name in text, name
    ttft_count = next(l for l in text.splitlines() if l.startswith("minivllm_time_to_first_token_seconds_count"))
    assert float(ttft_count.split()[-1]) >= 1


def test_streaming_disconnect_frees_blocks(server: RunningServer) -> None:
    body = {"model": TEST_MODEL, "prompt": "Tell me a long story.", "max_tokens": 900, "stream": True, "ignore_eos": True}
    with httpx.stream("POST", f"{server.base_url}/v1/completions", json=body, timeout=60) as response:
        next(response.iter_lines())
        assert server.engine.has_unfinished_requests()
    _wait_until_idle(server.engine)


def test_non_streaming_disconnect_frees_blocks(server: RunningServer) -> None:
    body = {"model": TEST_MODEL, "prompt": "Tell me a long story.", "max_tokens": 900, "ignore_eos": True}
    with pytest.raises(httpx.ReadTimeout):
        httpx.post(f"{server.base_url}/v1/completions", json=body, timeout=1.0)
    _wait_until_idle(server.engine)
