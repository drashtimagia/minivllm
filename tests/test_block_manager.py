import random

import pytest

from minivllm.block_manager import BlockManager


def _assert_no_leaks(bm: BlockManager) -> None:
    owned = [b for table in bm.block_tables.values() for b in table]
    assert len(owned) == len(set(owned)), "a block is owned twice"
    assert not set(owned) & set(bm.free_blocks), "a block is both owned and free"
    assert sorted(owned + list(bm.free_blocks)) == list(range(bm.num_blocks))


def test_allocate_rounds_up_to_whole_blocks() -> None:
    bm = BlockManager(num_blocks=10, block_size=4)
    bm.allocate("a", 5)
    assert len(bm.block_tables["a"]) == 2
    assert bm.num_free_blocks == 8


def test_grow_adds_block_only_at_boundary() -> None:
    bm = BlockManager(num_blocks=10, block_size=4)
    bm.allocate("a", 4)
    bm.grow("a", 4)
    assert len(bm.block_tables["a"]) == 1
    bm.grow("a", 5)
    assert len(bm.block_tables["a"]) == 2


def test_free_returns_all_blocks() -> None:
    bm = BlockManager(num_blocks=4, block_size=4)
    bm.allocate("a", 9)
    bm.free("a")
    assert bm.num_free_blocks == 4
    _assert_no_leaks(bm)


def test_over_allocation_fails_loudly() -> None:
    bm = BlockManager(num_blocks=2, block_size=4)
    assert not bm.can_allocate(9)
    with pytest.raises(RuntimeError):
        bm.allocate("a", 9)
    bm.allocate("a", 8)
    assert not bm.can_grow("a", 9)
    with pytest.raises(RuntimeError):
        bm.grow("a", 9)


def test_randomized_10k_ops_never_leak() -> None:
    rng = random.Random(1234)
    bm = BlockManager(num_blocks=64, block_size=16)
    lengths: dict[str, int] = {}
    next_id = 0
    for _ in range(10_000):
        op = rng.random()
        if op < 0.35:
            n = rng.randint(1, 200)
            if bm.can_allocate(n):
                seq_id = f"s{next_id}"
                next_id += 1
                bm.allocate(seq_id, n)
                lengths[seq_id] = n
        elif op < 0.75 and lengths:
            seq_id = rng.choice(list(lengths))
            n = lengths[seq_id] + rng.randint(1, 20)
            if bm.can_grow(seq_id, n):
                bm.grow(seq_id, n)
                lengths[seq_id] = n
            else:
                # Out of blocks: free the newest sequence, as preemption would.
                victim = max(lengths, key=lambda s: int(s[1:]))
                bm.free(victim)
                del lengths[victim]
        elif lengths:
            seq_id = rng.choice(list(lengths))
            bm.free(seq_id)
            del lengths[seq_id]
        _assert_no_leaks(bm)
        for seq_id, n in lengths.items():
            assert len(bm.block_tables[seq_id]) == bm.blocks_needed(n)
    for seq_id in list(lengths):
        bm.free(seq_id)
    assert bm.num_free_blocks == bm.num_blocks
