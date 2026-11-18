#!/usr/bin/env bash
# Runs the full Phase 7 matrix on one GPU machine, then renders charts and tables.
# Needs: this repo installed with the server and bench extras, flash-attn, vllm, and a
# committed (clean) working tree. Results are written to bench/results/.
set -euo pipefail

SHAREGPT="${SHAREGPT:-ShareGPT_V3_unfiltered_cleaned_split.json}"
AGENT_RATE="${AGENT_RATE:-4}"

if [ ! -f "$SHAREGPT" ]; then
  echo "ShareGPT file not found: $SHAREGPT" >&2
  echo "Download it with:" >&2
  echo "  wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json" >&2
  exit 1
fi

for system in static continuous paged paged_prefix vllm vllm_eager; do
  python -m bench.serve_bench --system "$system" --workload sharegpt --dataset-path "$SHAREGPT"
done

for system in paged paged_prefix vllm vllm_prefix; do
  python -m bench.serve_bench --system "$system" --workload agent --rates "$AGENT_RATE"
done

python -m bench.max_concurrency --device cuda --dtype bfloat16 --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 --kv-budget-gb 12 --attention-backend flash
python -m bench.profile_decode --device cuda --dtype bfloat16 --model Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 4096 --kv-budget-gb 12 --attention-backend flash --batch-size 32

python -m bench.make_charts
python -m bench.make_tables
