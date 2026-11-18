from collections import OrderedDict, deque


class BlockManager:
    """Owns the physical KV cache blocks. Each sequence gets a block table: the list of
    physical block ids holding its tokens in order, block_size tokens per block. Only the
    last block of a sequence can be partially filled.

    With prefix caching, a full block is identified by hash(parent block hash, its
    tokens), so a hash names a whole prefix, not just one block's contents. Blocks are
    reference counted: sequences with the same prefix share the same physical blocks.
    A block whose count drops to zero keeps its contents and stays findable by hash in
    an LRU pool, and is only reused (evicted) when no never-used block is left."""

    def __init__(self, num_blocks: int, block_size: int, enable_prefix_caching: bool = False) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError(f"need num_blocks >= 1 and block_size >= 1, got {num_blocks}, {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.block_tables: dict[str, list[int]] = {}
        self.ref_counts = [0] * num_blocks
        self.block_hashes: list[int | None] = [None] * num_blocks
        self.hash_to_block: dict[int, int] = {}
        self.free_blocks: deque[int] = deque(range(num_blocks))
        # Blocks with ref count zero that still hold a registered prefix, oldest first.
        self.cached_free_blocks: OrderedDict[int, None] = OrderedDict()
        self._seq_hashes: dict[str, list[int]] = {}
        self.num_queried_tokens = 0
        self.num_hit_tokens = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks) + len(self.cached_free_blocks)

    @property
    def hit_rate(self) -> float:
        return self.num_hit_tokens / self.num_queried_tokens if self.num_queried_tokens else 0.0

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def match_prefix(self, token_ids: list[int]) -> list[int]:
        """Blocks already holding the longest cached prefix of token_ids. The block that
        contains the last token is never matched: the sequence must run at least one
        token through the model to get logits to sample from."""
        if not self.enable_prefix_caching:
            return []
        matched: list[int] = []
        parent: int | None = None
        for i in range((len(token_ids) - 1) // self.block_size):
            parent = self._block_hash(parent, token_ids[i * self.block_size : (i + 1) * self.block_size])
            block = self.hash_to_block.get(parent)
            if block is None:
                break
            matched.append(block)
        return matched

    def can_allocate(self, token_ids: list[int], reserve: int = 0) -> bool:
        matched = self.match_prefix(token_ids)
        fresh = self.blocks_needed(len(token_ids)) - len(matched)
        # A matched block that is sitting in the free pool stops being free once reused.
        revived = sum(1 for b in matched if self.ref_counts[b] == 0)
        return fresh + revived + reserve <= self.num_free_blocks

    def allocate(self, seq_id: str, token_ids: list[int]) -> int:
        """Give the sequence blocks for token_ids. Returns how many leading tokens are
        already in the cache and need no computation."""
        if seq_id in self.block_tables:
            raise RuntimeError(f"{seq_id} already has blocks")
        if not self.can_allocate(token_ids):
            raise RuntimeError(f"{seq_id}: not enough free blocks for {len(token_ids)} tokens")
        matched = self.match_prefix(token_ids)
        for block in matched:
            self._acquire(block)
        fresh = self.blocks_needed(len(token_ids)) - len(matched)
        self.block_tables[seq_id] = matched + [self._take_fresh_block() for _ in range(fresh)]
        self._seq_hashes[seq_id] = []
        self.num_queried_tokens += len(token_ids)
        self.num_hit_tokens += len(matched) * self.block_size
        return len(matched) * self.block_size

    def can_grow(self, seq_id: str, num_tokens: int) -> bool:
        return self._extra_blocks(seq_id, num_tokens) <= self.num_free_blocks

    def grow(self, seq_id: str, num_tokens: int) -> None:
        """Make sure the sequence has room for num_tokens tokens in total."""
        extra = self._extra_blocks(seq_id, num_tokens)
        if extra > self.num_free_blocks:
            raise RuntimeError(f"{seq_id}: not enough free blocks to grow to {num_tokens} tokens")
        self.block_tables[seq_id] += [self._take_fresh_block() for _ in range(extra)]

    def register_full_blocks(self, seq_id: str, token_ids: list[int], num_computed: int) -> None:
        """Make the sequence's newly completed blocks findable by later requests. Only
        blocks whose keys and values are fully computed may be registered."""
        if not self.enable_prefix_caching:
            return
        hashes = self._seq_hashes[seq_id]
        table = self.block_tables[seq_id]
        for i in range(len(hashes), num_computed // self.block_size):
            parent = hashes[-1] if hashes else None
            h = self._block_hash(parent, token_ids[i * self.block_size : (i + 1) * self.block_size])
            hashes.append(h)
            # When two sequences computed the same prefix concurrently, the first to
            # register wins and the duplicate block stays an ordinary private block.
            if h not in self.hash_to_block:
                self.hash_to_block[h] = table[i]
                self.block_hashes[table[i]] = h

    def free(self, seq_id: str) -> None:
        # Freed back to front so that, among blocks of one sequence, the deepest part of
        # the prefix is evicted first; a shorter prefix stays useful to more requests.
        for block in reversed(self.block_tables.pop(seq_id)):
            self.ref_counts[block] -= 1
            if self.ref_counts[block] == 0:
                self._release(block)
        self._seq_hashes.pop(seq_id)

    def _block_hash(self, parent: int | None, tokens: list[int]) -> int:
        return hash((parent, tuple(tokens)))

    def _acquire(self, block: int) -> None:
        if self.ref_counts[block] == 0:
            del self.cached_free_blocks[block]
        self.ref_counts[block] += 1

    def _release(self, block: int) -> None:
        h = self.block_hashes[block]
        if h is not None and self.hash_to_block.get(h) == block:
            self.cached_free_blocks[block] = None
        else:
            self.block_hashes[block] = None
            self.free_blocks.append(block)

    def _take_fresh_block(self) -> int:
        if self.free_blocks:
            block = self.free_blocks.popleft()
        else:
            block, _ = self.cached_free_blocks.popitem(last=False)
            evicted_hash = self.block_hashes[block]
            if evicted_hash is None:
                raise RuntimeError(f"cached free block {block} has no hash")
            del self.hash_to_block[evicted_hash]
            self.block_hashes[block] = None
        self.ref_counts[block] = 1
        return block

    def _extra_blocks(self, seq_id: str, num_tokens: int) -> int:
        return max(0, self.blocks_needed(num_tokens) - len(self.block_tables[seq_id]))
