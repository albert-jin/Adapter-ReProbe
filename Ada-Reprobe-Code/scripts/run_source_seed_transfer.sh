#!/usr/bin/env bash
set -euo pipefail

# Train one native layer-21 source probe, then evaluate it zero-shot on both
# adapters. Launch at most two instances concurrently.

if [[ $# -ne 2 ]]; then
  echo "usage: $0 GPU_ID SEED" >&2
  exit 2
fi
gpu_id="$1"
seed="$2"
if [[ ! "$gpu_id" =~ ^[0-3]$ || ! "$seed" =~ ^[0-9]+$ ]]; then
  echo "GPU_ID must be 0..3 and SEED must be an integer" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
hf_home="$project_dir/shared_cache/huggingface"
base_model="$hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
gsm_adapter="$hf_home/hub/models--ehzawad--qwen3-1.7b-gsm8k-grpo/snapshots/83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8"
finance_adapter="$hf_home/hub/models--fotapol--qwen3-1.7b-financial-qa-lora/snapshots/8433fcf8142e3db64f38df1f6eeaa38bf2e92651"

wait_for_memory() {
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

source_name="src_current_l21_fixed_seed${seed}"
source_root="$repo_dir/auto_res/runs/$source_name"
if ! find "$source_root" -path '*/model/weights.pth' -type f -print -quit \
    2>/dev/null | grep -q .; then
  wait_for_memory
  (
    cd "$repo_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" HF_HOME="$hf_home" PYTHONPATH=. \
      .venv/bin/python train_luh/run_train_luh.py \
      --config-path ../auto_res/configs --config-name train_claim_hs \
      "model.pretrained_model_name_or_path=$base_model" \
      "output_dir=auto_res/runs/$source_name" \
      "seed=$seed" \
      training_arguments.num_train_epochs=10 \
      training_arguments.learning_rate=0.0003 \
      training_arguments.gradient_accumulation_steps=8 \
      training_arguments.per_device_train_batch_size=4 \
      training_arguments.eval_strategy=no \
      ue_layer.pos_weight=5.6 \
      'ue_layer.head_cfg.feature_extractor.0.layer_nums=[21]'
  )
fi

source_head="$(find "$source_root" -path '*/model/weights.pth' -type f \
  -printf '%T@ %h\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$source_head" ]]; then
  echo "Could not resolve trained source head under $source_root" >&2
  exit 1
fi

evaluate_target() {
  local target_name="$1"
  local adapter="$2"
  local output_name="${source_name}_zero_${target_name}"
  if find "$repo_dir/auto_res/runs/$output_name" -name eval_metrics.json \
      -type f -print -quit 2>/dev/null | grep -q .; then
    echo "Skipping completed $output_name"
    return
  fi
  wait_for_memory
  (
    cd "$repo_dir"
    CUDA_VISIBLE_DEVICES="$gpu_id" HF_HOME="$hf_home" PYTHONPATH=. \
      .venv/bin/python train_luh/run_train_luh.py \
      --config-path ../auto_res/configs --config-name eval_official_uhead \
      "model.pretrained_model_name_or_path=$base_model" \
      "model.adapter_path=$adapter" \
      "ue_layer.path=$source_head" \
      "seed=$seed" \
      "output_dir=auto_res/runs/$output_name"
  )
}

evaluate_target gsm "$gsm_adapter"
evaluate_target finance "$finance_adapter"
