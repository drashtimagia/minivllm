# minivllm report

Commit at the time of writing: see `git log`. All test results below are from a CPU only
machine (Apple A18 Pro, 8 GB RAM). No GPU was available.

## What was built

| Phase | What | Gate | State |
|---|---|---|---|
| 1 | Hand written Qwen2, sampler, single request generation | Correctness test on CPU | Passed |
| 2 | Static batching with left padding | Correctness at batch sizes 1, 4, 16 | Passed |
| 3 | Sequence, scheduler, model runner, engine; continuous batching | Staggered arrivals, abort frees everything, batched equals alone | Passed |
| 4 | Paged KV cache, block manager, preemption by recompute, `--kv-budget-gb`, naive and flash backends | Fuzz with no leaks, preemption under a tiny budget, correctness on the naive backend | Passed on CPU; flash backend not run |
| 5 | Prefix caching with ref counts and an LRU pool | Identical outputs on and off, second request prefills only the tail, eviction never frees a block in use | Passed |
| 6 | OpenAI compatible server, metrics, disconnect aborts | Official `openai` SDK streaming and non streaming, disconnect frees blocks | Passed |
| 7 | Benchmark scripts, chart and table generators | Raw results, charts and README tables committed | Scripts done; no results, gate open |
| 8 | Chunked prefill, CUDA graph decode | Correctness still passes; before/after numbers in `bench/results` | Chunked prefill passed; CUDA graphs unrun; no numbers, gate open |

## Test results

The last full run of the suite on CPU: 89 passed, 2 skipped. The two skipped tests are
GPU only: the flash backend against the naive backend, and CUDA graph decode against
eager decode. The master correctness test compares greedy output with HF transformers
generate token for token on 20 prompts (short, long, multilingual, code), 64 new tokens
each, in float32. It passes for static batching at batch sizes 1, 4 and 16, continuous
batching, and chunked prefill.

The server tests start a real uvicorn server and use the official `openai` SDK.
Block manager fuzz tests run 10,000 random operations with and without prefix caching.

## Benchmark results

None. `bench/results/` is empty. No performance claim is made anywhere in this
repository.

## What still needs a GPU

Run on one L4 or A10 (24 GB) from a clean commit, with `flash-attn` and `vllm` installed:

1. `python -m pytest -m gpu`: the flash backend against naive, and CUDA graphs against
   eager. Both have never executed. Open questions are whether `flash_attn_with_kvcache`
   accepts a block size of 16, and whether it can be captured in a CUDA graph.
2. `bench/run_all.sh`: the full comparison with vLLM, the concurrency measurement and
   the decode profile.
3. `bench/run_phase8.sh`: before and after for chunked prefill and CUDA graphs.
4. Commit `bench/results`, then regenerate the README tables with
   `python -m bench.make_tables --readme README.md`.

The explanation of any gap to vLLM has to come from the decode profile and the result
files; it is not written here because nothing has been measured.

## Known limitations

- Qwen2 checkpoints are the only ones run end to end. Llama-architecture support (no
  rope scaling) is tested against a tiny random-weight HF Llama, not a real checkpoint,
  and Llama 3 checkpoints are rejected because they use rope scaling.
- `vllm bench serve` and `vllm serve` flags were checked against vLLM's source tree, not
  against an installed vLLM.
- The Dockerfile has never been built.
- The static baseline ignores `--kv-budget-gb`; it sizes its cache per batch.
- The server reads engine counters from the event loop while a step may be running on a
  worker thread. Gauge values can be one step stale.
- `/metrics` KV usage counts only blocks held by running sequences.
- vLLM runs with CUDA graphs and chunked prefill by default; the benchmark includes an
  eager vLLM configuration, but the feature sets still differ.
- Token-for-token agreement with HF is checked in float32 only. In bfloat16 the test
  checks logit closeness (cosine similarity and top-1 agreement), since reordered
  floating point work can legitimately flip a near-tie.

## Shortcuts taken

- The flash backend runs one kernel call per prompt chunk instead of one varlen call.
- Streaming detokenization decodes the full output each step, which is quadratic.
- The scheduler admits new work only if one spare block per running sequence is free,
  a simple rule rather than a tuned watermark.
- Greedy and sampled rows share one seeded generator; there are no per request seeds.

## What to do next

1. Do the GPU runs above and fix whatever they find in the flash and graph paths.
2. Write the "Why I built this" section of the README.
3. Replace the per prompt flash calls with one varlen call and measure it.
4. Run a real Llama checkpoint without rope scaling, and add rope scaling for Llama 3.
