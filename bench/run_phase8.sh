#!/usr/bin/env bash
# Before/after runs for the Phase 8 optimizations. "paged" is the before; each other
# system turns on one optimization (or both). All run at the same commit on the same
# machine, so the result files are directly comparable. Needs the same setup as
# run_all.sh and a clean working tree.
set -euo pipefail

SHAREGPT="${SHAREGPT:-ShareGPT_V3_unfiltered_cleaned_split.json}"
AGENT_RATE="${AGENT_RATE:-4}"

for system in paged paged_chunked paged_graphs paged_chunked_graphs; do
  python -m bench.serve_bench --system "$system" --workload sharegpt --dataset-path "$SHAREGPT"
  python -m bench.serve_bench --system "$system" --workload agent --rates "$AGENT_RATE"
done

python -m bench.profile_decode --device cuda --dtype bfloat16 --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 --kv-budget-gb 12 --attention-backend flash --batch-size 32 --enable-cuda-graphs
python -m bench.make_charts
python -m bench.make_tables
