#!/usr/bin/env bash
set -euo pipefail

# Reproducible sequential runner for native two-parameter calibration sweeps.
# At most one Python training process is alive per invocation; launch no more
# than two invocations concurrently to respect the experiment resource policy.

if [[ $# -ne 3 ]]; then
  echo "usage: $0 GPU_ID {finance|gsm} N:SEED[,N:SEED...]" >&2
  exit 2
fi

gpu_id="$1"
target="$2"
plan="$3"

if [[ ! "$gpu_id" =~ ^[0-3]$ ]]; then
  echo "GPU_ID must be one of 0,1,2,3" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
hf_home="$project_dir/shared_cache/huggingface"
base_model="$hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
source_head="$repo_dir/auto_res/runs/src_current_l21_fixed_seed42/2026-09-04/20-50-29/model"

case "$target" in
  finance)
    adapter="$hf_home/hub/models--fotapol--qwen3-1.7b-financial-qa-lora/snapshots/8433fcf8142e3db64f38df1f6eeaa38bf2e92651"
    ;;
  gsm)
    adapter="$hf_home/hub/models--ehzawad--qwen3-1.7b-gsm8k-grpo/snapshots/83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8"
    ;;
  *)
    echo "target must be finance or gsm" >&2
    exit 2
    ;;
esac

wait_for_memory() {
  local used total percent
  while true; do
    IFS=',' read -r used total < <(
      nvidia-smi --id="$gpu_id" \
        --query-gpu=memory.used,memory.total \
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

IFS=',' read -r -a jobs <<< "$plan"
for job in "${jobs[@]}"; do
  IFS=':' read -r support_size seed <<< "$job"
  if [[ ! "$support_size" =~ ^[0-9]+$ || ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "invalid plan entry: $job" >&2
    exit 2
  fi

  run_name="${target}_cal_unweighted_n${support_size}_seed${seed}"
  if find "$repo_dir/auto_res/runs/$run_name" -name eval_metrics.json \
      -type f -print -quit 2>/dev/null | grep -q .; then
    echo "Skipping completed $run_name"
    continue
  fi

  wait_for_memory
  echo "Starting $run_name on physical GPU $gpu_id"
  (
    cd "$repo_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    HF_HOME="$hf_home" \
    PYTHONPATH=. \
      .venv/bin/python train_luh/run_train_luh.py \
      --config-path ../auto_res/configs \
      --config-name train_claim_hs \
      "model.pretrained_model_name_or_path=$base_model" \
      "model.adapter_path=$adapter" \
      "ue_layer.path=$source_head" \
      ue_layer.trainable_scope=calibration \
      ue_layer.adaptation.bridge_rank=0 \
      ue_layer.adaptation.calibration=true \
      ue_layer.pos_weight=1 \
      "+dataset.train_subset=$support_size" \
      "seed=$seed" \
      "output_dir=auto_res/runs/$run_name" \
      training_arguments.num_train_epochs=30 \
      training_arguments.learning_rate=0.05 \
      training_arguments.gradient_accumulation_steps=1 \
      training_arguments.per_device_train_batch_size=4 \
      training_arguments.eval_strategy=no
  )
done
