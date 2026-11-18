import random

from minivllm.block_manager import BlockManager
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence, SequenceStatus

EOS = frozenset({0})
MAX_LEN = 1000
BLOCK_SIZE = 4


def _seq(request_id: str, prompt_len: int = 4, max_tokens: int = 8) -> Sequence:
    return Sequence(request_id, [1] * prompt_len, SamplingParams(max_tokens=max_tokens), 0.0)


def _scheduler(num_blocks: int = 64, max_num_seqs: int = 4, budget: int = 100) -> Scheduler:
    return Scheduler(BlockManager(num_blocks, BLOCK_SIZE), max_num_seqs, budget)


def _run_step(scheduler: Scheduler) -> None:
    sched = scheduler.schedule()
    if sched is not None:
        for seq in sched.seqs:
            seq.num_computed_tokens = seq.num_tokens
            seq.append_token(5, EOS, MAX_LEN)
    scheduler.free_finished()


def _assert_blocks_consistent(scheduler: Scheduler) -> None:
    bm = scheduler.block_manager
    assert set(bm.block_tables) == {s.request_id for s in scheduler.running}
    owned = [b for t in bm.block_tables.values() for b in t]
    assert sorted(owned + list(bm.free_blocks)) == list(range(bm.num_blocks))
    for seq in scheduler.running:
        assert len(bm.block_tables[seq.request_id]) >= bm.blocks_needed(seq.num_computed_tokens)


def test_admits_up_to_max_num_seqs() -> None:
    scheduler = _scheduler(max_num_seqs=2)
    for i in range(3):
        scheduler.add(_seq(f"r{i}"))
    sched = scheduler.schedule()
    assert sched is not None and sched.is_prefill
    assert [s.request_id for s in sched.seqs] == ["r0", "r1"]
    assert len(scheduler.waiting) == 1


def test_prefill_respects_token_budget() -> None:
    scheduler = _scheduler(max_num_seqs=8, budget=10)
    for i in range(3):
        scheduler.add(_seq(f"r{i}", prompt_len=4))
    sched = scheduler.schedule()
    assert sched is not None
    assert [s.request_id for s in sched.seqs] == ["r0", "r1"]


def test_prefill_has_priority_over_decode() -> None:
    scheduler = _scheduler()
    scheduler.add(_seq("a"))
    _run_step(scheduler)
    scheduler.add(_seq("b"))
    sched = scheduler.schedule()
    assert sched is not None and sched.is_prefill
    assert [s.request_id for s in sched.seqs] == ["b"]


def test_decode_grows_block_table_at_boundary() -> None:
    scheduler = _scheduler()
    scheduler.add(_seq("a", prompt_len=BLOCK_SIZE))
    _run_step(scheduler)
    sched = scheduler.schedule()
    assert sched is not None and not sched.is_prefill
    assert len(sched.block_tables[0]) == 2


def test_preempts_most_recently_admitted_and_requeues_it_first() -> None:
    scheduler = _scheduler(num_blocks=4, max_num_seqs=3)
    for name in "ab":
        scheduler.add(_seq(name, prompt_len=BLOCK_SIZE, max_tokens=100))
    for _ in range(20):
        _run_step(scheduler)
        if scheduler.num_preemptions:
            break
    # Two sequences cannot both outgrow two blocks each in a 4 block pool: b, the later
    # admission, is the one that goes, and a keeps its blocks.
    assert scheduler.num_preemptions == 1
    assert [s.request_id for s in scheduler.running] == ["a"]
    b = scheduler.waiting[0]
    assert b.request_id == "b"
    assert b.status is SequenceStatus.WAITING and b.num_computed_tokens == 0
    assert b.output_token_ids, "generated tokens are kept for the recompute prefill"
    assert "b" not in scheduler.block_manager.block_tables
    _assert_blocks_consistent(scheduler)


def test_abort_frees_blocks() -> None:
    scheduler = _scheduler(max_num_seqs=1)
    scheduler.add(_seq("run"))
    scheduler.add(_seq("wait"))
    _run_step(scheduler)
    assert scheduler.abort("wait") is not None
    seq = scheduler.abort("run")
    assert seq is not None and seq.status is SequenceStatus.FINISHED_ABORTED
    assert not scheduler.has_unfinished()
    assert scheduler.block_manager.num_free_blocks == 64
    assert scheduler.abort("missing") is None


def test_randomized_add_step_abort_never_leaks_blocks() -> None:
    rng = random.Random(0)
    scheduler = _scheduler(num_blocks=24, max_num_seqs=6, budget=64)
    next_id = 0
    for _ in range(5000):
        op = rng.random()
        if op < 0.3:
            scheduler.add(_seq(f"r{next_id}", rng.randint(1, 30), rng.randint(1, 30)))
            next_id += 1
        elif op < 0.4 and next_id:
            scheduler.abort(f"r{rng.randrange(next_id)}")
        elif scheduler.has_unfinished():
            _run_step(scheduler)
        _assert_blocks_consistent(scheduler)
    for _ in range(100_000):
        if not scheduler.has_unfinished():
            break
        _run_step(scheduler)
    assert not scheduler.has_unfinished()
    assert scheduler.block_manager.num_free_blocks == 24
    assert scheduler.num_preemptions > 0


def _chunked_scheduler(num_blocks: int = 64, max_num_seqs: int = 4, budget: int = 8) -> Scheduler:
    return Scheduler(BlockManager(num_blocks, BLOCK_SIZE), max_num_seqs, budget, enable_chunked_prefill=True)


def _run_chunked_step(scheduler: Scheduler) -> list[tuple[str, int]]:
    sched = scheduler.schedule()
    if sched is None:
        return []
    assert sum(sched.num_scheduled_tokens) <= scheduler.max_num_batched_tokens
    for seq, n in zip(sched.seqs, sched.num_scheduled_tokens):
        seq.num_computed_tokens += n
        if seq.num_computed_tokens == seq.num_tokens:
            seq.append_token(5, EOS, MAX_LEN)
    scheduler.free_finished()
    return [(s.request_id, n) for s, n in zip(sched.seqs, sched.num_scheduled_tokens)]


def test_chunked_prefill_splits_a_long_prompt_across_steps() -> None:
    scheduler = _chunked_scheduler(budget=8)
    scheduler.add(_seq("long", prompt_len=30, max_tokens=2))
    steps = [_run_chunked_step(scheduler) for _ in range(4)]
    assert steps == [[("long", 8)], [("long", 8)], [("long", 8)], [("long", 6)]]
    seq = scheduler.running[0]
    assert len(seq.output_token_ids) == 1, "only the last chunk samples a token"


def test_decode_shares_a_step_with_a_prompt_chunk() -> None:
    scheduler = _chunked_scheduler(budget=8)
    scheduler.add(_seq("a", prompt_len=4, max_tokens=50))
    _run_chunked_step(scheduler)
    scheduler.add(_seq("b", prompt_len=30))
    step = _run_chunked_step(scheduler)
    assert step == [("a", 1), ("b", 7)], "the decode goes first and the chunk takes the rest"


def test_prefill_first_would_have_stalled_that_decode() -> None:
    scheduler = Scheduler(BlockManager(64, BLOCK_SIZE), 4, 100)
    scheduler.add(_seq("a", prompt_len=4, max_tokens=50))
    _run_step(scheduler)
    scheduler.add(_seq("b", prompt_len=30))
    sched = scheduler.schedule()
    assert sched is not None and [s.request_id for s in sched.seqs] == ["b"]


def test_chunked_randomized_never_leaks_and_respects_budget() -> None:
    rng = random.Random(7)
    scheduler = _chunked_scheduler(num_blocks=24, max_num_seqs=6, budget=12)
    next_id = 0
    for _ in range(5000):
        op = rng.random()
        if op < 0.3:
            scheduler.add(_seq(f"r{next_id}", rng.randint(1, 40), rng.randint(1, 20)))
            next_id += 1
        elif op < 0.4 and next_id:
            scheduler.abort(f"r{rng.randrange(next_id)}")
        else:
            _run_chunked_step(scheduler)
        _assert_blocks_consistent(scheduler)
    for _ in range(100_000):
        if not scheduler.has_unfinished():
            break
        _run_chunked_step(scheduler)
    assert not scheduler.has_unfinished()
    assert scheduler.block_manager.num_free_blocks == 24
