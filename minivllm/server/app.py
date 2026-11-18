import json
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST
from transformers import PreTrainedTokenizerBase

from minivllm.config import EngineConfig
from minivllm.sequence import RequestOutput, SamplingParams
from minivllm.server.async_engine import AsyncEngine
from minivllm.server.metrics import Metrics
from minivllm.server.protocol import ChatCompletionRequest, CompletionRequest
from minivllm.server.service import IncrementalDetokenizer, RequestError, Service

DONE = "data: [DONE]\n\n"


def create_app(
    config: EngineConfig, async_engine: AsyncEngine, tokenizer: PreTrainedTokenizerBase
) -> FastAPI:
    service = Service(config, async_engine, tokenizer, Metrics())

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        async_engine.start()
        yield
        await async_engine.stop()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(RequestError)
    async def request_error_handler(_: Request, error: RequestError) -> JSONResponse:
        body = {"error": {"message": error.message, "type": error.error_type, "code": error.status}}
        return JSONResponse(body, status_code=error.status)

    @app.get("/health")
    async def health() -> JSONResponse:
        if async_engine.failure is not None:
            return JSONResponse({"status": "engine stopped"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        card = {"id": service.model_name, "object": "model", "created": int(time.time()), "owned_by": "minivllm"}
        return {"object": "list", "data": [card]}

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(service.metrics.render(async_engine.engine.stats()), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/completions", response_model=None)
    async def completions(body: CompletionRequest, request: Request) -> Response:
        service.check_model(body.model)
        prompt_ids = tokenizer.encode(body.prompt) if isinstance(body.prompt, str) else list(body.prompt)
        service.check_length(len(prompt_ids), body.max_tokens)
        params = _sampling_params(body, body.max_tokens)
        request_id = f"cmpl-{uuid.uuid4().hex}"
        outputs = service.generate(request, request_id, prompt_ids, params)
        meta = _Meta(request_id, service.model_name, int(time.time()), len(prompt_ids))
        if body.stream:
            include_usage = body.stream_options is not None and body.stream_options.include_usage
            return _sse(_completion_stream(outputs, tokenizer, meta, include_usage))
        text, finish_reason, num_completion = await _collect(outputs, tokenizer)
        choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}
        return JSONResponse(_response("text_completion", meta, [choice], num_completion))

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(body: ChatCompletionRequest, request: Request) -> Response:
        service.check_model(body.model)
        prompt_ids = service.encode_chat([m.model_dump() for m in body.messages])
        max_tokens = body.max_completion_tokens or body.max_tokens or service.default_max_tokens(len(prompt_ids))
        service.check_length(len(prompt_ids), max_tokens)
        params = _sampling_params(body, max_tokens)
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        outputs = service.generate(request, request_id, prompt_ids, params)
        meta = _Meta(request_id, service.model_name, int(time.time()), len(prompt_ids))
        if body.stream:
            include_usage = body.stream_options is not None and body.stream_options.include_usage
            return _sse(_chat_stream(outputs, tokenizer, meta, include_usage))
        text, finish_reason, num_completion = await _collect(outputs, tokenizer)
        message = {"role": "assistant", "content": text}
        choice = {"index": 0, "message": message, "finish_reason": finish_reason}
        return JSONResponse(_response("chat.completion", meta, [choice], num_completion))

    return app


class _Meta:
    def __init__(self, request_id: str, model: str, created: int, num_prompt_tokens: int) -> None:
        self.request_id = request_id
        self.model = model
        self.created = created
        self.num_prompt_tokens = num_prompt_tokens


def _sampling_params(body: CompletionRequest | ChatCompletionRequest, max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=body.temperature, top_p=body.top_p, max_tokens=max_tokens, ignore_eos=body.ignore_eos
    )


def _usage(meta: _Meta, num_completion: int) -> dict[str, int]:
    return {
        "prompt_tokens": meta.num_prompt_tokens,
        "completion_tokens": num_completion,
        "total_tokens": meta.num_prompt_tokens + num_completion,
    }


def _response(kind: str, meta: _Meta, choices: list[dict[str, Any]], num_completion: int) -> dict[str, Any]:
    return {
        "id": meta.request_id,
        "object": kind,
        "created": meta.created,
        "model": meta.model,
        "choices": choices,
        "usage": _usage(meta, num_completion),
    }


async def _collect(
    outputs: AsyncIterator[RequestOutput], tokenizer: PreTrainedTokenizerBase
) -> tuple[str, str | None, int]:
    token_ids: list[int] = []
    finish_reason: str | None = None
    async for out in outputs:
        token_ids += out.new_token_ids
        finish_reason = out.finish_reason
    return tokenizer.decode(token_ids, skip_special_tokens=True), finish_reason, len(token_ids)


def _sse(stream: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream")


def _event(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _chunk(kind: str, meta: _Meta, choices: list[dict[str, Any]], usage: dict[str, int] | None = None) -> str:
    payload: dict[str, Any] = {
        "id": meta.request_id, "object": kind, "created": meta.created, "model": meta.model, "choices": choices,
    }
    if usage is not None:
        payload["usage"] = usage
    return _event(payload)


async def _completion_stream(
    outputs: AsyncIterator[RequestOutput], tokenizer: PreTrainedTokenizerBase, meta: _Meta, include_usage: bool
) -> AsyncIterator[str]:
    detokenizer = IncrementalDetokenizer(tokenizer)
    num_completion = 0
    async for out in outputs:
        num_completion += len(out.new_token_ids)
        text = detokenizer.push(out.new_token_ids, out.finished)
        if text or out.finished:
            choice = {"index": 0, "text": text, "logprobs": None, "finish_reason": out.finish_reason}
            yield _chunk("text_completion", meta, [choice])
    if include_usage:
        yield _chunk("text_completion", meta, [], _usage(meta, num_completion))
    yield DONE


async def _chat_stream(
    outputs: AsyncIterator[RequestOutput], tokenizer: PreTrainedTokenizerBase, meta: _Meta, include_usage: bool
) -> AsyncIterator[str]:
    detokenizer = IncrementalDetokenizer(tokenizer)
    num_completion = 0
    role = {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
    yield _chunk("chat.completion.chunk", meta, [role])
    async for out in outputs:
        num_completion += len(out.new_token_ids)
        text = detokenizer.push(out.new_token_ids, out.finished)
        if text:
            yield _chunk("chat.completion.chunk", meta, [{"index": 0, "delta": {"content": text}, "finish_reason": None}])
        if out.finished:
            yield _chunk("chat.completion.chunk", meta, [{"index": 0, "delta": {}, "finish_reason": out.finish_reason}])
    if include_usage:
        yield _chunk("chat.completion.chunk", meta, [], _usage(meta, num_completion))
    yield DONE
