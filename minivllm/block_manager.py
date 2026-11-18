from collections import deque


class BlockManager:
    """Owns the physical KV cache blocks. Each sequence gets a block table: the list of
    physical block ids holding its tokens in order, block_size tokens per block. Only the
    last block of a sequence can be partially filled."""

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if num_blocks < 1 or block_size < 1:
            raise ValueError(f"need num_blocks >= 1 and block_size >= 1, got {num_blocks}, {block_size}")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks: deque[int] = deque(range(num_blocks))
        self.block_tables: dict[str, list[int]] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def blocks_needed(self, num_tokens: int) -> int:
        return -(-num_tokens // self.block_size)

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.num_free_blocks

    def allocate(self, seq_id: str, num_tokens: int) -> None:
        if seq_id in self.block_tables:
            raise RuntimeError(f"{seq_id} already has blocks")
        if not self.can_allocate(num_tokens):
            raise RuntimeError(f"{seq_id}: not enough free blocks for {num_tokens} tokens")
        self.block_tables[seq_id] = [self.free_blocks.popleft() for _ in range(self.blocks_needed(num_tokens))]

    def can_grow(self, seq_id: str, num_tokens: int) -> bool:
        return self._extra_blocks(seq_id, num_tokens) <= self.num_free_blocks

    def grow(self, seq_id: str, num_tokens: int) -> None:
        """Make sure the sequence has room for num_tokens tokens in total."""
        extra = self._extra_blocks(seq_id, num_tokens)
        if extra > self.num_free_blocks:
            raise RuntimeError(f"{seq_id}: not enough free blocks to grow to {num_tokens} tokens")
        self.block_tables[seq_id] += [self.free_blocks.popleft() for _ in range(extra)]

    def free(self, seq_id: str) -> None:
        self.free_blocks.extend(self.block_tables.pop(seq_id))

    def _extra_blocks(self, seq_id: str, num_tokens: int) -> int:
        return max(0, self.blocks_needed(num_tokens) - len(self.block_tables[seq_id]))
