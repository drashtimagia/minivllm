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
