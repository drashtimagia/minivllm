import asyncio
import time
from collections.abc import AsyncIterator

from fastapi import Request
from transformers import PreTrainedTokenizerBase

from minivllm.config import EngineConfig
from minivllm.sequence import RequestOutput, SamplingParams
from minivllm.server.async_engine import AsyncEngine
from minivllm.server.metrics import Metrics


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400, error_type: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.error_type = error_type


class IncrementalDetokenizer:
    """Turns a growing list of token ids into text deltas. The whole list is decoded
    each time because a token's text can depend on its neighbours (multi-byte
    characters are split across tokens); that is quadratic in output length, which is
    negligible next to a model forward at these lengths."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        self._ids: list[int] = []
        self._emitted = ""

    def push(self, new_ids: list[int], finished: bool) -> str:
        self._ids += new_ids
        text = self._tokenizer.decode(self._ids, skip_special_tokens=True)
        # A trailing U+FFFD means the last character is still incomplete; it is held
        # back until later tokens complete it, unless generation has ended.
        if not finished and text.endswith("�"):
            return ""
        delta = text[len(self._emitted) :]
        self._emitted = text
        return delta


class Service:
    def __init__(
        self,
        config: EngineConfig,
        async_engine: AsyncEngine,
        tokenizer: PreTrainedTokenizerBase,
        metrics: Metrics,
    ) -> None:
        self.config = config
        self.async_engine = async_engine
        self.tokenizer = tokenizer
        self.metrics = metrics
        self.model_name = config.served_model_name or config.model

    def check_model(self, model: str) -> None:
        if model != self.model_name:
            raise RequestError(f"The model `{model}` does not exist.", 404, "NotFoundError")

    def encode_chat(self, messages: list[dict[str, str]]) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=False
            )
        )

    def check_length(self, prompt_len: int, max_tokens: int) -> None:
        limit = self.config.max_model_len
        if prompt_len < 1:
            raise RequestError("The prompt must contain at least one token.")
        if prompt_len + max_tokens > limit:
            raise RequestError(
                f"This model's maximum context length is {limit} tokens. However, you requested "
                f"{max_tokens} output tokens and your prompt contains {prompt_len} tokens, which "
                f"exceeds the limit by {prompt_len + max_tokens - limit} tokens."
            )

    def default_max_tokens(self, prompt_len: int) -> int:
        remaining = self.config.max_model_len - prompt_len
        if remaining < 1:
            raise RequestError(
                f"This model's maximum context length is {self.config.max_model_len} tokens, and "
                f"your prompt contains {prompt_len} tokens, leaving no room to generate."
            )
        return remaining

    async def generate(
        self,
        http_request: Request,
        request_id: str,
        prompt_token_ids: list[int],
        params: SamplingParams,
    ) -> AsyncIterator[RequestOutput]:
        """Yield the request's outputs as the engine produces them. If the client goes
        away, or this generator is closed early, the request is aborted so the engine
        frees its blocks."""
        queue = self.async_engine.submit(request_id, prompt_token_ids, params)
        interval = self.config.disconnect_poll_interval_s
        start = last_check = time.monotonic()
        first_token_time: float | None = None
        num_tokens = 0
        finished = False
        try:
            while not finished:
                try:
                    item: RequestOutput | Exception | None = await asyncio.wait_for(queue.get(), interval)
                except TimeoutError:
                    item = None
                now = time.monotonic()
                if now - last_check >= interval:
                    last_check = now
                    if await http_request.is_disconnected():
                        return
                if item is None:
                    continue
                if isinstance(item, Exception):
                    raise item
                if first_token_time is None:
                    first_token_time = now
                    self.metrics.ttft.observe(now - start)
                num_tokens += len(item.new_token_ids)
                finished = item.finished
                if finished and num_tokens > 1:
                    self.metrics.tpot.observe((now - first_token_time) / (num_tokens - 1))
                yield item
        finally:
            if not finished:
                self.async_engine.abort(request_id)
