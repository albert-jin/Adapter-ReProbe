#!/usr/bin/env bash
set -euo pipefail

# Train bias-only and/or temperature-only native calibration variants with
# exactly the same optimizer protocol as the completed two-parameter curve.

if [[ $# -lt 2 || $# -gt 5 || ! "$1" =~ ^[0-3]$ || ! "$2" =~ ^(gsm|finance)$ ]]; then
  echo "usage: $0 GPU_ID {gsm|finance} [both|bias|temperature] [SIZES] [SEEDS]" >&2
  exit 2
fi
gpu_id="$1"
target="$2"
scope_selector="${3:-both}"
size_selector="${4:-10,25,50,100}"
seed_selector="${5:-42,43,44}"
case "$scope_selector" in
  both)
    scopes=(calibration_bias calibration_temperature)
    ;;
  bias)
    scopes=(calibration_bias)
    ;;
  temperature)
    scopes=(calibration_temperature)
    ;;
  *)
    echo "scope must be both, bias, or temperature" >&2
    exit 2
    ;;
esac
IFS=',' read -r -a sizes <<< "$size_selector"
IFS=',' read -r -a seeds <<< "$seed_selector"
if (( ${#sizes[@]} == 0 || ${#seeds[@]} == 0 )); then
  echo "SIZES and SEEDS must be non-empty comma-separated lists" >&2
  exit 2
fi
for size in "${sizes[@]}"; do
  if [[ ! "$size" =~ ^(10|25|50|100)$ ]]; then
    echo "invalid support size: $size" >&2
    exit 2
  fi
done
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^(42|43|44)$ ]]; then
    echo "invalid seed: $seed" >&2
    exit 2
  fi
done

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
hf_home="$project_dir/shared_cache/huggingface"
base_model="$hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
source_head="$repo_dir/auto_res/runs/src_current_l21_fixed_seed42/2026-09-04/20-50-29/model"
if [[ "$target" == gsm ]]; then
  adapter="$hf_home/hub/models--ehzawad--qwen3-1.7b-gsm8k-grpo/snapshots/83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8"
else
  adapter="$hf_home/hub/models--fotapol--qwen3-1.7b-financial-qa-lora/snapshots/8433fcf8142e3db64f38df1f6eeaa38bf2e92651"
fi

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

for scope in "${scopes[@]}"; do
  for size in "${sizes[@]}"; do
    for seed in "${seeds[@]}"; do
      run_name="${target}_${scope}_n${size}_seed${seed}"
      if find "$repo_dir/auto_res/runs/$run_name" -name eval_metrics.json \
          -type f -print -quit 2>/dev/null | grep -q .; then
        echo "Skipping completed $run_name"
        continue
      fi
      {
        echo "$(date --iso-8601=seconds) launch $run_name on physical GPU $gpu_id"
        preflight_memory
        (
          cd "$repo_dir"
          CUDA_VISIBLE_DEVICES="$gpu_id" HF_HOME="$hf_home" PYTHONPATH=. \
            .venv/bin/python train_luh/run_train_luh.py \
            --config-path ../auto_res/configs --config-name train_claim_hs \
            "model.pretrained_model_name_or_path=$base_model" \
            "model.adapter_path=$adapter" \
            "ue_layer.path=$source_head" \
            "ue_layer.trainable_scope=$scope" \
            ue_layer.adaptation.bridge_rank=0 \
            ue_layer.adaptation.calibration=true \
            ue_layer.pos_weight=1 \
            "+dataset.train_subset=$size" \
            "seed=$seed" \
            "output_dir=auto_res/runs/$run_name" \
            training_arguments.num_train_epochs=30 \
            training_arguments.learning_rate=0.05 \
            training_arguments.gradient_accumulation_steps=1 \
            training_arguments.per_device_train_batch_size=4 \
            training_arguments.eval_strategy=no
        )
      } 2>&1 | tee -a "$repo_dir/auto_res/logs/${run_name}.log"
    done
  done
done
