import random

import pytest

from minivllm.block_manager import BlockManager

BLOCK_SIZE = 4


def _tokens(n: int, start: int = 0) -> list[int]:
    return list(range(start, start + n))


def _assert_consistent(bm: BlockManager) -> None:
    counts = [0] * bm.num_blocks
    for table in bm.block_tables.values():
        for b in table:
            counts[b] += 1
    assert counts == bm.ref_counts
    free = list(bm.free_blocks) + list(bm.cached_free_blocks)
    assert len(free) == len(set(free)), "a block is in the free pools twice"
    assert all(bm.ref_counts[b] == 0 for b in free), "a block in use is in a free pool"
    assert sorted(free + [b for b in range(bm.num_blocks) if bm.ref_counts[b] > 0]) == list(range(bm.num_blocks))
    for h, b in bm.hash_to_block.items():
        assert bm.block_hashes[b] == h
    assert all(bm.block_hashes[b] is not None for b in bm.cached_free_blocks)
    assert not any(bm.block_hashes[b] is not None for b in bm.free_blocks)


def test_allocate_rounds_up_to_whole_blocks() -> None:
    bm = BlockManager(10, BLOCK_SIZE)
    assert bm.allocate("a", _tokens(5)) == 0
    assert len(bm.block_tables["a"]) == 2
    assert bm.num_free_blocks == 8


def test_grow_adds_block_only_at_boundary() -> None:
    bm = BlockManager(10, BLOCK_SIZE)
    bm.allocate("a", _tokens(4))
    bm.grow("a", 4)
    assert len(bm.block_tables["a"]) == 1
    bm.grow("a", 5)
    assert len(bm.block_tables["a"]) == 2


def test_free_returns_all_blocks() -> None:
    bm = BlockManager(4, BLOCK_SIZE)
    bm.allocate("a", _tokens(9))
    bm.free("a")
    assert bm.num_free_blocks == 4
    _assert_consistent(bm)


def test_over_allocation_fails_loudly() -> None:
    bm = BlockManager(2, BLOCK_SIZE)
    assert not bm.can_allocate(_tokens(9))
    with pytest.raises(RuntimeError):
        bm.allocate("a", _tokens(9))
    bm.allocate("a", _tokens(8))
    assert not bm.can_grow("a", 9)
    with pytest.raises(RuntimeError):
        bm.grow("a", 9)


def _run(bm: BlockManager, seq_id: str, tokens: list[int]) -> int:
    cached = bm.allocate(seq_id, tokens)
    bm.register_full_blocks(seq_id, tokens, len(tokens))
    return cached


def test_second_request_reuses_full_prefix_blocks() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=True)
    prefix = _tokens(10)
    assert _run(bm, "a", prefix + [100, 101]) == 0
    cached = _run(bm, "b", prefix + [200, 201, 202])
    assert cached == 8, "two full blocks of the 10 token shared prefix"
    assert bm.block_tables["a"][:2] == bm.block_tables["b"][:2]
    assert bm.block_tables["a"][2] != bm.block_tables["b"][2]
    assert bm.ref_counts[bm.block_tables["a"][0]] == 2
    _assert_consistent(bm)


def test_last_token_is_never_served_from_cache() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=True)
    tokens = _tokens(8)
    _run(bm, "a", tokens)
    bm.free("a")
    assert _run(bm, "b", tokens) == 4, "the block holding the last token must be recomputed"


def test_hash_covers_whole_prefix_not_just_block() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=True)
    _run(bm, "a", [1, 2, 3, 4, 9, 9, 9, 9, 0])
    assert _run(bm, "b", [7, 7, 7, 7, 9, 9, 9, 9, 0]) == 0, "same second block, different prefix"


def test_freed_blocks_stay_cached_and_are_found_again() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=True)
    tokens = _tokens(13)
    _run(bm, "a", tokens)
    bm.free("a")
    assert len(bm.cached_free_blocks) == 3
    assert _run(bm, "b", tokens) == 12
    assert not bm.cached_free_blocks or all(bm.ref_counts[b] == 0 for b in bm.cached_free_blocks)
    _assert_consistent(bm)


def test_eviction_takes_oldest_cached_block_and_only_when_needed() -> None:
    bm = BlockManager(4, BLOCK_SIZE, enable_prefix_caching=True)
    _run(bm, "old", _tokens(5, 0))
    bm.free("old")
    _run(bm, "new", _tokens(5, 100))
    bm.free("new")
    assert len(bm.cached_free_blocks) == 2 and len(bm.free_blocks) == 2
    bm.allocate("x", _tokens(8, 500))
    assert len(bm.cached_free_blocks) == 2, "never-used blocks are taken before cached ones"
    bm.allocate("y", _tokens(1, 900))
    assert bm.match_prefix(_tokens(5, 0) + [0]) == [], "the oldest cached prefix was evicted"
    assert len(bm.match_prefix(_tokens(5, 100) + [0])) == 1
    _assert_consistent(bm)


def test_eviction_never_takes_a_block_in_use() -> None:
    bm = BlockManager(3, BLOCK_SIZE, enable_prefix_caching=True)
    _run(bm, "a", _tokens(9))
    assert bm.num_free_blocks == 0
    assert not bm.can_allocate(_tokens(1, 50))
    with pytest.raises(RuntimeError):
        bm.allocate("b", _tokens(1, 50))
    _assert_consistent(bm)


def test_shared_block_survives_until_last_user_frees_it() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=True)
    prefix = _tokens(8)
    _run(bm, "a", prefix + [1])
    _run(bm, "b", prefix + [2])
    shared = bm.block_tables["a"][0]
    bm.free("a")
    assert bm.ref_counts[shared] == 1 and shared not in bm.cached_free_blocks
    bm.free("b")
    assert shared in bm.cached_free_blocks
    _assert_consistent(bm)


def test_hit_rate_counts_cached_prompt_tokens() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=True)
    _run(bm, "a", _tokens(9))
    _run(bm, "b", _tokens(9))
    assert bm.num_queried_tokens == 18 and bm.num_hit_tokens == 8
    assert bm.hit_rate == pytest.approx(8 / 18)


def test_disabled_caching_never_shares() -> None:
    bm = BlockManager(16, BLOCK_SIZE, enable_prefix_caching=False)
    _run(bm, "a", _tokens(9))
    assert _run(bm, "b", _tokens(9)) == 0
    assert not set(bm.block_tables["a"]) & set(bm.block_tables["b"])


@pytest.mark.parametrize("caching", [False, True])
def test_randomized_10k_ops_never_leak(caching: bool) -> None:
    rng = random.Random(1234)
    bm = BlockManager(40, BLOCK_SIZE, enable_prefix_caching=caching)
    seqs: dict[str, list[int]] = {}
    next_id = 0
    for _ in range(10_000):
        op = rng.random()
        if op < 0.35:
            # A tiny vocabulary and a few shared stems make prefix hits common.
            stem = rng.choice([[1, 1, 1, 1, 2, 2, 2, 2], [3, 3, 3, 3], []])
            tokens = stem + [rng.randint(0, 2) for _ in range(rng.randint(1, 40))]
            if bm.can_allocate(tokens):
                seq_id = f"s{next_id}"
                next_id += 1
                bm.allocate(seq_id, tokens)
                bm.register_full_blocks(seq_id, tokens, len(tokens))
                seqs[seq_id] = tokens
        elif op < 0.75 and seqs:
            seq_id = rng.choice(list(seqs))
            seqs[seq_id].append(rng.randint(0, 2))
            n = len(seqs[seq_id])
            if bm.can_grow(seq_id, n):
                bm.grow(seq_id, n)
                bm.register_full_blocks(seq_id, seqs[seq_id], n)
            else:
                victim = max(seqs, key=lambda s: int(s[1:]))
                bm.free(victim)
                del seqs[victim]
        elif seqs:
            seq_id = rng.choice(list(seqs))
            bm.free(seq_id)
            del seqs[seq_id]
        _assert_consistent(bm)
        for seq_id, tokens in seqs.items():
            assert len(bm.block_tables[seq_id]) >= bm.blocks_needed(len(tokens)) - 1
    for seq_id in list(seqs):
        bm.free(seq_id)
    assert bm.num_free_blocks == bm.num_blocks
    _assert_consistent(bm)
