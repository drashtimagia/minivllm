# Project: minivllm

Build an LLM inference engine from scratch that implements and measures the core techniques
behind vLLM: continuous batching, paged KV cache, and prefix caching. It serves an
OpenAI compatible API and is benchmarked against real vLLM on the same hardware.

This is a portfolio project for ML infrastructure roles. The code must be clear enough that
every line can be explained in an interview. Optimize for clarity and correctness first,
then performance, and only with measurements. First, save this entire prompt as CLAUDE.md
at the repo root so every future session follows it.

## How to work
- Work autonomously through all phases below, in order. Do not stop to ask me questions.
  When a decision is not covered here, choose the option that is simplest to explain and
  closest to how vLLM does it, then record it in DESIGN_NOTES.md.
- Each phase has a gate. Do not start the next phase until the gate passes. If a gate fails,
  debug and fix the root cause. Never weaken, skip, or delete a test to make it pass.
- Commit as you go: one logical change per commit, and at least one commit per phase.
- If no GPU is available, finish everything that runs on CPU, write the GPU code and bench
  scripts fully, mark GPU tests as skipped, and list exactly what still needs a GPU run in
  the final report. Never fabricate or estimate results.
- When all phases are done, write REPORT.md: what was built, test results, benchmark
  results, known limitations, shortcuts taken, and what to do next.

## Stack
Python 3.11, PyTorch, FastAPI, HF tokenizers and safetensors. Models: Qwen2.5 0.5B Instruct
for CPU development, Qwen2.5 1.5B Instruct for GPU benchmarks. Target GPU: one L4 or A10 (24GB).

## Architecture (do not blur these boundaries)
- sequence.py: Sequence, SequenceStatus, SamplingParams. Plain data plus state transitions.
- block_manager.py: owns physical KV blocks, block tables, ref counts, prefix hash map.
  Knows nothing about the model; deals only in block ids.
- scheduler.py: waiting/running queues, admission, preemption. Talks only to block_manager.
  Returns a SchedulerOutput describing what to run this step.
- model_runner.py: turns a SchedulerOutput into tensors (input_ids, positions, slot_mapping,
  block_tables), runs the model, samples. Makes no scheduling decisions.
- models/: hand written model definitions (Qwen2, Llama) loading HF safetensors. No
  transformers model classes at runtime; tokenizer only.
- attention/: backends behind one small interface. `naive` (gather blocks + SDPA) is the
  reference; `flash` is the fast path. Every backend must match `naive` numerically.
- engine.py: add_request, abort_request, step. No HTTP. Usable as a library.
- server/: FastAPI app. Engine runs in one background loop; requests communicate through
  per request asyncio queues. No model or scheduling logic here.

## Code rules
- Type hints everywhere. Dataclasses for config and state.
- No abstractions for things that do not exist yet: no plugin registries, factories, or
  base classes with one implementation. The attention backend interface is the one exception.
- Comments explain why, never what. No docstrings that restate the function name.
- No broad try/except. Fail loudly with clear messages.
- No dead code, no commented out code, no vague TODOs.
- Small readable functions. If a function needs section comments, split it.
- All config lives in one EngineConfig dataclass, exposed as CLI flags. No magic numbers.

## Testing rules
- tests/test_correctness.py is the master gate: greedy outputs must match HF transformers
  generate token for token on 20 varied prompts (short, long, multilingual, code), 64 new
  tokens each. It must pass at the end of every phase.
- Block manager and scheduler get pure CPU unit tests that run in under 10 seconds, including
  randomized alloc/free/preempt sequences asserting no blocks leak.
- GPU only tests are marked and skipped without a GPU.

## Benchmark and honesty rules
- Never write a performance number anywhere unless it came from a script in bench/ and the
  raw JSON is committed under bench/results/ with hardware, model, dtype, commit hash, flags.
- vLLM comparisons use the same GPU, model, dtype, max tokens, KV budget, and the same
  client (`vllm bench serve`).
- Report losses as plainly as wins, and explain the gap to vLLM using profiler data.

## Writing rules
- No emojis, no marketing words (blazing, powerful, seamless, robust, cutting edge,
  production ready, revolutionary). No filler intros or summaries.
- Do not document features that do not exist.
- README.md is factual and short: one paragraph on what it is, architecture diagram
  (Mermaid), quickstart, supported features, benchmark tables and charts generated from
  bench/results, and a link to DESIGN_NOTES.md. Leave a clearly marked empty section
  "## Why I built this" for me to write.
- DESIGN_NOTES.md: one short entry per decision with the reason and alternatives considered.
- Commit messages: imperative, specific, under 72 chars.

## Phases

### Phase 1: model and naive generation
Implement models/qwen2.py by hand (RMSNorm, RoPE, GQA attention, SwiGLU MLP) loading
Qwen2.5 0.5B and 1.5B weights. Contiguous KV cache, single request prefill + decode loop,
greedy / temperature / top p sampling. Write the correctness test.
Gate: correctness test passes on CPU.

### Phase 2: static batching baseline
Left padding, attention mask, batch runs until its longest sequence finishes. Baseline for
benchmarks only, behind --scheduler static.
Gate: correctness test passes at batch sizes 1, 4, 16.

### Phase 3: continuous batching
Introduce Sequence, Scheduler, Engine. Requests join and leave between steps. Prefill and
decode as separate forward calls. Per sequence contiguous cache for now.
Gate: tests pass for staggered arrivals, abort mid generation frees everything, and batched
outputs equal each request run alone.

### Phase 4: paged KV cache
Preallocated per layer block tensors [num_blocks, block_size, num_kv_heads, head_dim],
block_size configurable (default 16). BlockManager with free list and per sequence block
tables; model_runner builds slot_mapping and block_tables. Naive backend first, then a
flash backend using flash_attn_with_kvcache with block_table (GPU only). Preemption by
recompute when blocks run out, preempting the most recently admitted sequence. Add
--kv-budget-gb so memory pressure can be forced on small models.
Gate: 10k op randomized block manager fuzz shows no leaks, preemption under a tiny budget
still yields correct outputs, correctness passes on both backends, DESIGN_NOTES.md covers
block size, preemption policy, and slot mapping.

### Phase 5: prefix caching
Hash only full blocks as hash(parent_hash, block_tokens). Ref counted blocks; blocks at
ref count zero move to an LRU cached free pool and are evicted only when needed. On
admission, reuse the longest matching prefix and prefill only the uncached tail. Track hit
rate. Behind --enable-prefix-caching.
Gate: shared system prompt gives identical outputs with caching on and off, the second
request prefills only new tokens, eviction never frees a block in use.

### Phase 6: server
FastAPI with /v1/completions, /v1/chat/completions (streaming via SSE and non streaming),
/v1/models, /health, and /metrics in Prometheus format (running/waiting seqs, KV block
usage, prefix hit rate, TTFT and TPOT histograms). Client disconnect aborts the request and
frees its blocks. Requests over max model length return a clear 400. Dockerfile and one
command launch: python -m minivllm.server --model ...
Gate: integration test using the official openai Python SDK passes for streaming and
non streaming, and a disconnect test confirms blocks are freed.

### Phase 7: benchmarks
bench/ scripts running `vllm bench serve` with ShareGPT and Poisson arrivals at rates
1, 2, 4, 8, 16, 32, inf against: static, continuous, paged, paged + prefix cache, and real
vLLM. Agent workload: N requests sharing a 2k to 4k token system prompt with varied user
turns, prefix caching on vs off. A torch.profiler run of one decode step at batch 32 with a
summary of where time goes. Charts generated from the JSON: throughput vs rate, p99 TTFT vs
rate, agent workload TTFT, max concurrent sequences at a fixed memory budget.
Gate: all raw results and charts committed with metadata, README tables generated from them.

### Phase 8: optimizations
CUDA graphs for decode at fixed batch size buckets, then chunked prefill. Rerun the phase 7
scripts before and after each one and commit both results.
Gate: correctness test still passes and before/after numbers are in bench/results.

Then write REPORT.md and finish.
