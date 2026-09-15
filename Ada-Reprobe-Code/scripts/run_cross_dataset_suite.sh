#!/usr/bin/env bash
set -euo pipefail

# Run one predeclared target/dataset slice sequentially. At most two copies of
# this runner should be active, and every launch rechecks memory below 60%.

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 GPU_ID {base|primary|altgsm} {svamp|multiarith|asdiv} [SEEDS]" >&2
  exit 2
fi
gpu_id="$1"
target="$2"
dataset="$3"
seeds="${4:-2040,2041,2042}"

if [[ ! "$gpu_id" =~ ^[0-3]$ ]]; then
  echo "GPU_ID must be one of 0,1,2,3" >&2
  exit 2
fi
if [[ ! "$dataset" =~ ^(svamp|multiarith|asdiv)$ ]]; then
  echo "dataset must be svamp, multiarith, or asdiv" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
hf_home="$project_dir/shared_cache/huggingface"
base_model="$hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
source_head="$repo_dir/auto_res/runs/src_current_l21_fixed_seed42/2026-09-04/20-50-29/model"

case "$target" in
  base)
    adapter_args=()
    ;;
  primary)
    adapter="$hf_home/hub/models--ehzawad--qwen3-1.7b-gsm8k-grpo/snapshots/83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8"
    adapter_args=(--adapter "$adapter")
    ;;
  altgsm)
    adapter="$hf_home/hub/models--jeypiii--Qwen3-1.7B-gsm8k/snapshots/13af138668f8f532686b74a11130c7b4cfe97bda"
    adapter_args=(--adapter "$adapter")
    ;;
  *)
    echo "target must be base, primary, or altgsm" >&2
    exit 2
    ;;
esac

preflight_memory() {
  local used total percent
  while true; do
    IFS=',' read -r used total < <(
      nvidia-smi --id="$gpu_id" --query-gpu=memory.used,memory.total \
        --format=csv,noheader,nounits | tr -d ' '
    )
    percent=$((100 * used / total))
    if (( percent < 60 )); then
      echo "GPU $gpu_id preflight: ${used}/${total} MiB (${percent}%)"
      return
    fi
    echo "GPU $gpu_id is at ${percent}% memory; checking again in 15s" >&2
    sleep 15
  done
}

IFS=',' read -r -a seed_array <<< "$seeds"
for seed in "${seed_array[@]}"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid seed: $seed" >&2
    exit 2
  fi
  run_name="natural_bestofn_${target}_${dataset}_n100x8_seed${seed}_cap512"
  output_dir="$repo_dir/auto_res/runs/$run_name"
  if [[ -f "$output_dir/summary.json" ]]; then
    echo "Skipping completed $run_name"
    continue
  fi
  {
    echo "$(date --iso-8601=seconds) launch/resume $run_name on physical GPU $gpu_id"
    preflight_memory
    (
      cd "$repo_dir"
      CUDA_VISIBLE_DEVICES="$gpu_id" HF_HOME="$hf_home" PYTHONPATH=. \
        .venv/bin/python auto_res/scripts/evaluate_natural_best_of_n.py \
        --base-model "$base_model" "${adapter_args[@]}" \
        --head "source=$source_head" --dataset "$dataset" \
        --output-dir "auto_res/runs/$run_name" \
        --num-questions 100 --num-samples 8 --question-batch-size 4 \
        --score-batch-size 8 --max-new-tokens 512 --temperature 0.7 \
        --top-p 0.95 --top-k 20 --seed "$seed" --resume
    )
  } 2>&1 | tee -a "$repo_dir/auto_res/logs/${run_name}.log"
done
