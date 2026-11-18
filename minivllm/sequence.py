import enum
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 16
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_STOPPED = "stop"
    FINISHED_LENGTH = "length"
    FINISHED_ABORTED = "abort"

    @property
    def finished(self) -> bool:
        return self in (
            SequenceStatus.FINISHED_STOPPED,
            SequenceStatus.FINISHED_LENGTH,
            SequenceStatus.FINISHED_ABORTED,
        )


@dataclass
class Sequence:
    request_id: str
    prompt_token_ids: list[int]
    params: SamplingParams
    arrival_time: float
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    # How many of this sequence's tokens already have keys and values in the cache.
    num_computed_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def finished(self) -> bool:
        return self.status.finished

    def append_token(self, token: int, eos_token_ids: frozenset[int], max_model_len: int) -> None:
        if self.status is not SequenceStatus.RUNNING:
            raise RuntimeError(f"{self.request_id}: append_token while {self.status}")
        self.output_token_ids.append(token)
        if token in eos_token_ids and not self.params.ignore_eos:
            self.status = SequenceStatus.FINISHED_STOPPED
        elif len(self.output_token_ids) >= self.params.max_tokens or self.num_tokens >= max_model_len:
            self.status = SequenceStatus.FINISHED_LENGTH


@dataclass(frozen=True)
class RequestOutput:
    request_id: str
    new_token_ids: list[int]
    finished: bool
    finish_reason: str | None


@dataclass(frozen=True)
class EngineStats:
    num_running: int
    num_waiting: int
    num_blocks: int
    num_free_blocks: int
    prefix_hit_rate: float
