from pydantic import BaseModel, ConfigDict, Field


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_usage: bool = False


class _SamplingFields(BaseModel):
    # Unsupported OpenAI parameters are rejected rather than silently ignored.
    model_config = ConfigDict(extra="forbid")
    model: str
    temperature: float = Field(default=1.0, ge=0)
    top_p: float = Field(default=1.0, gt=0, le=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    ignore_eos: bool = False


class CompletionRequest(_SamplingFields):
    prompt: str | list[int]
    max_tokens: int = Field(default=16, ge=1)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class ChatCompletionRequest(_SamplingFields):
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
