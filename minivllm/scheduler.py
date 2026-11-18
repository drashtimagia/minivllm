from collections import deque
from dataclasses import dataclass

from minivllm.block_manager import BlockManager
from minivllm.sequence import Sequence, SequenceStatus


@dataclass(frozen=True)
class SchedulerOutput:
    seqs: list[Sequence]
    is_prefill: bool
    # block_tables[i] lists the physical blocks of seqs[i], in token order.
    block_tables: list[list[int]]


class Scheduler:
    """Decides, once per engine step, which sequences run and whether the step is a
    prefill or a decode. Prefill is prioritized: if a waiting sequence can be admitted,
    the step prefills new sequences and running ones wait one step. This keeps the
    batch as full as possible, at the cost of a decode stall whenever requests arrive.

    When a decode step cannot get a block for every running sequence, the most recently
    admitted sequence is preempted by recompute: its blocks are freed and it goes back
    to the front of the waiting queue, to be prefilled again from prompt plus the tokens
    it already generated."""

    def __init__(
        self, block_manager: BlockManager, max_num_seqs: int, max_num_batched_tokens: int
    ) -> None:
        self.block_manager = block_manager
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput | None:
        admitted = self._admit()
        if admitted:
            return self._output(admitted, is_prefill=True)
        self._reserve_decode_blocks()
        if self.running:
            return self._output(list(self.running), is_prefill=False)
        return None

    def _admit(self) -> list[Sequence]:
        admitted: list[Sequence] = []
        budget = self.max_num_batched_tokens
        bm = self.block_manager
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            # Each running sequence may need one new block on its next decode step;
            # leaving that headroom stops a fresh admission from being preempted at
            # once by the sequences it just crowded out.
            headroom = len(self.running)
            token_ids = seq.token_ids
            uncached = len(token_ids) - len(bm.match_prefix(token_ids)) * bm.block_size
            if uncached > budget or not bm.can_allocate(token_ids, reserve=headroom):
                break
            self.waiting.popleft()
            budget -= uncached
            seq.num_computed_tokens = bm.allocate(seq.request_id, token_ids)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    def _reserve_decode_blocks(self) -> None:
        bm = self.block_manager
        for seq in list(self.running):
            if seq.status is not SequenceStatus.RUNNING:
                continue
            while not bm.can_grow(seq.request_id, seq.num_tokens):
                victim = self.running[-1]
                self._preempt(victim)
                if victim is seq:
                    break
            else:
                bm.grow(seq.request_id, seq.num_tokens)

    def _preempt(self, seq: Sequence) -> None:
        self.running.remove(seq)
        self.block_manager.free(seq.request_id)
        seq.num_computed_tokens = 0
        seq.status = SequenceStatus.WAITING
        self.waiting.appendleft(seq)
        self.num_preemptions += 1

    def _output(self, seqs: list[Sequence], is_prefill: bool) -> SchedulerOutput:
        tables = [list(self.block_manager.block_tables[s.request_id]) for s in seqs]
        return SchedulerOutput(seqs, is_prefill, tables)

    def register_computed(self, seq: Sequence) -> None:
        self.block_manager.register_full_blocks(seq.request_id, seq.token_ids, seq.num_computed_tokens)

    def free_finished(self) -> list[Sequence]:
        finished = [s for s in self.running if s.finished]
        for seq in finished:
            self.block_manager.free(seq.request_id)
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
