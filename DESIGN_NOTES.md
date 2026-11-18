# Design notes

One entry per decision: what was chosen, why, and what else was considered.

## Phase 1: model and naive generation

### Module and parameter names mirror the HF checkpoint
`models/qwen2.py` uses the same attribute names as the safetensors keys
(`model.layers.N.self_attn.q_proj`, ...), so loading is a single `load_state_dict` with
an explicit check that nothing is missing or unexpected.
Alternatives: a renaming table (more code to read, no benefit), or fused QKV and
gate/up projections like vLLM (fewer kernel launches; deferred until a profile shows the
separate projections matter).

### Tokens are passed flattened as `[num_tokens]`
The model forward takes `input_ids` and `positions` as 1D tensors and never sees a batch
dimension. The KV cache object regroups tokens per sequence inside attention. This is
how vLLM structures its forward, and it means padded batches (Phase 2), variable-length
continuous batches (Phase 3), and paged batches (Phase 4) reuse the same model code.
Alternative: `[batch, seq_len]` like HF. Simpler for the padded case, but every later
phase would need either padding or a second forward.

### The vocabulary projection runs only on rows that are sampled
`compute_logits` is separate from `forward`, and callers pass only the last hidden state
of each sequence. For a 780 token prompt this skips 779 rows of a 896 x 151936 matmul.
Alternative: compute logits for all tokens like HF; wasted work with no consumer.

### RMSNorm and RoPE follow HF's rounding order
RMSNorm computes the variance in fp32 and casts back before multiplying by the weight.
RoPE uses the "rotate half" layout with a precomputed fp32 cos/sin table indexed by
position. Matching the order of operations is what makes greedy outputs agree with HF
token for token, rather than merely being close in logit space.

### Correctness is checked in fp32 on CPU
The gate compares against HF `generate` with sampling, top-k, top-p and the checkpoint's
repetition penalty all explicitly disabled, both sides in fp32. bf16 would make
near-tie argmax flips likely over 64 tokens and turn a correctness test into a flaky one.
HF reference outputs are cached under `.cache/hf_reference/`, keyed by model directory,
prompts, transformers and torch versions, because loading a second fp32 copy of the
model dominates test time on an 8 GB machine.

### Sampling: greedy rows are decided by argmax, not by temperature -> 0
A batch can mix greedy and sampled requests. Greedy rows take `argmax` of the raw logits;
sampled rows are scaled by temperature, filtered by top-p (smallest prefix whose mass
reaches `top_p`, top token always kept), and drawn with `torch.multinomial` from one
seeded generator. Alternative: per-request generators for reproducible seeds per request;
not needed yet.

## Phase 2: static batching baseline

### Left padding with an explicit boolean mask
Prompts are left padded so every row's last prompt token is in the same column; the
logits for the whole batch are then one slice, and each decode step writes every row's
new key and value into the same cache column. The mask combines causality with a
per-row "is real token" mask over the cache. Pad positions get position id 0; their
outputs are never read.
Alternative: right padding. It keeps positions simple but scatters each row's next
token to a different cache column, which complicates the lockstep decode.

### Padding queries attend to themselves
A pad query at the left edge has no real key at or before it, and a fully masked row
makes softmax return NaN, which then propagates through the residual stream of that
row. Allowing each query to see its own key keeps the row finite. Real queries already
see themselves, so their output is unchanged.

### The cache is sized per batch
A static batch reserves `longest_prompt + max(max_tokens)` slots per row, all up front.
That reservation, plus finished rows that keep running until the longest row is done,
are the costs continuous batching and paging remove, which is why this path exists.

## Phase 3: continuous batching

### Prefill-first scheduling
Each step is either a prefill of newly admitted sequences or a decode of all running
ones, never mixed. If anything can be admitted, the step prefills and running sequences
wait one step. This is vLLM's default policy before chunked prefill: it keeps the batch
full, at the cost of a decode stall when requests arrive (addressed in Phase 8).

### One cache slot per sequence, reserved at admission
Phase 3 gives each running sequence a contiguous max_model_len region. It is the
simplest thing that lets sequences join and leave; the waste from over-reservation is
what Phase 4's paging removes.

### Engine and StaticEngine share one interface
Both expose add_request, abort_request, step, has_unfinished_requests, so the offline
helper, tests and benchmarks switch between them with `--scheduler`.

## Phase 4: paged KV cache

### Block size 16
Block size trades internal fragmentation against table size and gather overhead. A
sequence wastes on average half a block, so 16 wastes about 8 tokens per sequence, small
next to hundreds of tokens, while keeping block tables short. 16 is also vLLM's default.
It is a flag (`--block-size`). The naive backend works with any size. Whether
`flash_attn_with_kvcache` accepts a page size of 16 depends on the installed flash-attn
version; this has not been checked on a GPU yet, and the flash backend leaves that
error to the kernel.

### Slot mapping
A layer's pool is `[num_blocks, block_size, kv_heads, head_dim]`. Token at position p of
a sequence is stored in block `table[p // block_size]` at offset `p % block_size`.
Viewing the pool as `[num_blocks * block_size, ...]` makes that one flat index,
`block_id * block_size + offset`. The model runner builds a `slot_mapping` of those
indices for the step's new tokens, and the cache writes keys and values with a single
scatter per layer. Attention reads through the block tables (gather for naive, directly
in the kernel for flash).

### Preemption: recompute, most recently admitted first
When a decode step cannot give every running sequence the block it needs, the last
sequence in the running list (the most recently admitted) is freed and put at the front
of the waiting queue. Its generated tokens are kept, so on readmission one prefill over
prompt plus output rebuilds the cache. The oldest sequences keep their progress, so the
system always advances. Alternative: swapping blocks to CPU memory, which avoids
recompute but needs a second block pool and copy engine; recompute is simpler and is
vLLM's default for single-sequence requests.

### Admission headroom
A waiting sequence is admitted only if its blocks plus one spare block per running
sequence are free. Without it, a fresh admission takes the last free blocks, the next
decode step cannot grow the older sequences, and the new sequence is preempted
immediately, repeating every step.

### Memory budget
`--kv-budget-gb` sets the pool size; the number of blocks is the budget divided by the
bytes of one block across all layers (keys and values). The engine refuses to start if
the pool cannot hold one `max_model_len` sequence, since such a sequence could never
finish. Setting a tiny budget on a small model is how tests force preemption.

### One model forward path for all cache types
The model only calls `kv_cache.attend(layer, q, k, v)`. `PaddedKVCache` (static baseline)
and `PagedKVCache` implement it; the attention backend (naive or flash) sits behind the
paged cache and receives the pool, the block tables and the step layout.

## Phase 5: prefix caching

### Hash chain over full blocks only
A block is registered under `hash((parent_hash, block_tokens))`, where the parent hash
is that of the previous block (None for the first). The hash therefore names the whole
prefix up to and including the block, so two blocks with identical tokens but different
histories never collide on purpose. Partial blocks are never registered: they are still
being written. Python's built-in `hash` on a tuple of ints is a 64 bit value; an
accidental collision would silently serve the wrong keys and values. vLLM offers sha256
for that reason; the odds here are negligible for the workloads in this project, and a
cryptographic hash is a one-line change in `_block_hash` if that ever matters.

### The block holding the last token is always recomputed
Matching stops before the block that contains a prompt's final token. Even a fully
cached prompt must run one token through the model to produce logits to sample from, and
recomputing the final block keeps the prefill code path identical for hits and misses.

### Ref counts, and an LRU pool for blocks nobody is using
A block shared by several sequences has a ref count above one. At zero it is not wiped:
it moves to an ordered pool of cached free blocks and can still be matched by hash.
Allocation takes never-used blocks first and only then evicts the oldest cached block,
so eviction happens only when memory is needed and can never touch a block with a
non-zero ref count. When a sequence is freed its blocks are released back to front, so
the deepest part of a prefix is evicted before the shared start of it.

### Registration happens after the step that computed the block
Blocks are registered by the scheduler after the engine records a step's tokens as
computed, never at admission. Two requests with the same new prefix admitted in the
same step both compute it; the first to register wins and the other block stays
private. Alternative: let the second wait for the first. That would add scheduling
state for a rare case.

### Preempted sequences get cache hits on themselves
Preemption frees a sequence's blocks but their contents stay in the cached pool, so
when it is readmitted its prompt blocks are usually still there and the recompute
prefill covers only what was evicted in the meantime.

### Token budget counts only uncached tokens
`max_num_batched_tokens` limits the tokens actually run through the model in a prefill
step, so a prefix hit lets more requests be admitted in the same step.

### Hit rate
The block manager counts prompt tokens it was asked to place and how many were served
from the cache. Readmission after preemption is counted like any other admission.

## Phase 6: server

### One engine loop; handlers only enqueue
`AsyncEngine` runs a single asyncio task. Request handlers call `submit` and `abort`,
which append to an operation queue and wake the loop. The loop applies queued
operations between steps and runs `engine.step` on a one-thread executor so the event
loop stays responsive while the model runs. The engine is therefore only ever touched
by one party at a time and needs no locks. Alternative: a lock around the engine, which
would let handlers call `add_request` directly but would block them for the length of a
step. The `/metrics` handler reads queue lengths and counters while a step may be
running; those are plain reads of integers and list lengths, and a slightly stale
value is acceptable for a gauge.

### Each request owns an asyncio queue
The loop dispatches every step's outputs to the queue of the request they belong to.
If the loop dies, every waiting queue is handed the exception and `/health` returns
503, so clients fail instead of hanging.

### Disconnects abort the request
The generator that feeds a response checks `is_disconnected` at most once per
`--disconnect-poll-interval-s` and aborts the request if the client is gone. Its
`finally` block also aborts when the generator is closed early, which is how Starlette
reports a dropped streaming connection. Both paths end in `abort_request`, which frees
the sequence's blocks. Two tests (streaming and non-streaming) check that the block pool
is full again after the client leaves.

### Unsupported parameters are errors, not no-ops
Request models forbid unknown fields, so `n`, `stop`, `logprobs` and similar return
422 instead of being accepted and ignored. Supported beyond the basics: `ignore_eos`
and `stream_options.include_usage`, because `vllm bench serve` sends them.

### Length errors follow the OpenAI shape
If prompt tokens plus `max_tokens` exceed `--max-model-len`, the response is a 400 with
the usual `error.message` body naming both numbers. Chat requests that give no
`max_tokens` generate up to the remaining context.

### Streaming detokenization
Text deltas come from decoding all generated ids and emitting the new suffix. A trailing
U+FFFD means a multi-byte character is incomplete, so it is held back until the next
token or the end of the stream. Special tokens (end of turn) are skipped in output text
but still counted in `usage.completion_tokens`.

### Metrics
TTFT is arrival to first generated token; TPOT is the mean gap between tokens after
the first, observed once per finished request. KV usage counts only blocks held by
running sequences, since cached free blocks can be taken at any time.

### Dockerfile is untested
Docker is not installed on the development machine, so the image has not been built.
It installs the package with the server extra on `python:3.11-slim` and starts
`python -m minivllm.server`. The pip `torch` wheel on Linux includes CUDA support, but
that has not been checked either.
