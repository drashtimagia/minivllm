from collections import deque
from dataclasses import dataclass

from minivllm.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerOutput:
    seqs: list[Sequence]
    is_prefill: bool
    # Phase 3 keeps one contiguous cache region per sequence; slot i belongs to seqs[i].
    cache_slots: list[int]


class Scheduler:
    """Decides, once per engine step, which sequences run and whether the step is a
    prefill or a decode. Prefill is prioritized: if a waiting sequence can be admitted,
    the step prefills new sequences and running ones wait one step. This keeps the
    batch as full as possible, at the cost of a decode stall whenever requests arrive."""

    def __init__(self, max_num_seqs: int, max_num_batched_tokens: int) -> None:
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.free_slots = list(range(max_num_seqs))
        self.slot_of: dict[str, int] = {}

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput | None:
        admitted = self._admit()
        if admitted:
            return SchedulerOutput(admitted, True, [self.slot_of[s.request_id] for s in admitted])
        if self.running:
            return SchedulerOutput(
                list(self.running), False, [self.slot_of[s.request_id] for s in self.running]
            )
        return None

    def _admit(self) -> list[Sequence]:
        admitted: list[Sequence] = []
        budget = self.max_num_batched_tokens
        while self.waiting and self.free_slots and self.waiting[0].num_tokens <= budget:
            seq = self.waiting.popleft()
            budget -= seq.num_tokens
            seq.status = SequenceStatus.RUNNING
            self.slot_of[seq.request_id] = self.free_slots.pop()
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    def free_finished(self) -> list[Sequence]:
        finished = [s for s in self.running if s.finished]
        for seq in finished:
            self.free_slots.append(self.slot_of.pop(seq.request_id))
        self.running = [s for s in self.running if not s.finished]
        return finished

    def abort(self, request_id: str) -> Sequence | None:
        for seq in self.waiting:
            if seq.request_id == request_id:
                self.waiting.remove(seq)
                seq.status = SequenceStatus.FINISHED_ABORTED
                return seq
        for seq in self.running:
            if seq.request_id == request_id:
                seq.status = SequenceStatus.FINISHED_ABORTED
                self.free_finished()
                return seq
        return None
