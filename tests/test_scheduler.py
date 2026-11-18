import random

from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence, SequenceStatus

EOS = frozenset({0})
MAX_LEN = 1000


def _seq(request_id: str, prompt_len: int = 4, max_tokens: int = 8) -> Sequence:
    return Sequence(request_id, [1] * prompt_len, SamplingParams(max_tokens=max_tokens), 0.0)


def _run_step(scheduler: Scheduler) -> None:
    sched = scheduler.schedule()
    assert sched is not None
    for seq in sched.seqs:
        seq.num_computed_tokens = seq.num_tokens
        seq.append_token(5, EOS, MAX_LEN)
    scheduler.free_finished()


def test_admits_up_to_max_num_seqs() -> None:
    scheduler = Scheduler(max_num_seqs=2, max_num_batched_tokens=100)
    for i in range(3):
        scheduler.add(_seq(f"r{i}"))
    sched = scheduler.schedule()
    assert sched is not None and sched.is_prefill
    assert [s.request_id for s in sched.seqs] == ["r0", "r1"]
    assert len(scheduler.waiting) == 1


def test_prefill_respects_token_budget() -> None:
    scheduler = Scheduler(max_num_seqs=8, max_num_batched_tokens=10)
    for i in range(3):
        scheduler.add(_seq(f"r{i}", prompt_len=4))
    sched = scheduler.schedule()
    assert sched is not None
    assert [s.request_id for s in sched.seqs] == ["r0", "r1"]


def test_prefill_has_priority_over_decode() -> None:
    scheduler = Scheduler(max_num_seqs=4, max_num_batched_tokens=100)
    scheduler.add(_seq("a"))
    _run_step(scheduler)
    scheduler.add(_seq("b"))
    sched = scheduler.schedule()
    assert sched is not None and sched.is_prefill
    assert [s.request_id for s in sched.seqs] == ["b"]


def test_decode_runs_all_running() -> None:
    scheduler = Scheduler(max_num_seqs=4, max_num_batched_tokens=100)
    for i in range(3):
        scheduler.add(_seq(f"r{i}"))
    _run_step(scheduler)
    sched = scheduler.schedule()
    assert sched is not None and not sched.is_prefill
    assert len(sched.seqs) == 3
    assert len(set(sched.cache_slots)) == 3


def test_abort_waiting_and_running_frees_slots() -> None:
    scheduler = Scheduler(max_num_seqs=1, max_num_batched_tokens=100)
    scheduler.add(_seq("run"))
    scheduler.add(_seq("wait"))
    _run_step(scheduler)
    assert scheduler.abort("wait") is not None
    seq = scheduler.abort("run")
    assert seq is not None and seq.status is SequenceStatus.FINISHED_ABORTED
    assert not scheduler.has_unfinished()
    assert scheduler.free_slots == [0]
    assert scheduler.abort("missing") is None


def test_randomized_add_step_abort_never_leaks_slots() -> None:
    rng = random.Random(0)
    max_num_seqs = 6
    scheduler = Scheduler(max_num_seqs=max_num_seqs, max_num_batched_tokens=64)
    next_id = 0
    for _ in range(5000):
        op = rng.random()
        if op < 0.3:
            scheduler.add(_seq(f"r{next_id}", rng.randint(1, 30), rng.randint(1, 10)))
            next_id += 1
        elif op < 0.4 and next_id:
            scheduler.abort(f"r{rng.randrange(next_id)}")
        elif scheduler.has_unfinished():
            _run_step(scheduler)
        slots = [scheduler.slot_of[s.request_id] for s in scheduler.running]
        assert len(slots) == len(set(slots))
        assert sorted(slots + scheduler.free_slots) == list(range(max_num_seqs))
    while scheduler.has_unfinished():
        _run_step(scheduler)
    assert sorted(scheduler.free_slots) == list(range(max_num_seqs))
    assert not scheduler.slot_of
