#!/usr/bin/env bash
set -euo pipefail

# Export frozen native source-verifier logits for the complete adaptation split.
# A strict 60% preflight threshold implements the shared-GPU safety policy.

if [[ $# -lt 1 || $# -gt 2 || ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: $0 GPU_ID [gsm|finance|both]" >&2
  exit 2
fi

gpu_id="$1"
target="${2:-both}"
if [[ ! "$target" =~ ^(gsm|finance|both)$ ]]; then
  echo "target must be gsm, finance, or both" >&2
  exit 2
fi
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
hf_home="$project_dir/shared_cache/huggingface"
base_model="$hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
source_head="$repo_dir/auto_res/runs/src_current_l21_fixed_seed42/2026-09-04/20-50-29/model"
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

export_target() {
  local name="$1"
  local adapter="$2"
  local output_root="$repo_dir/auto_res/runs/calibration_logits_${name}_seed42"
  if find "$output_root" -name claim_predictions_train.npz -type f -print -quit \
      2>/dev/null | grep -q .; then
    echo "Skipping completed calibration logit export for $name"
    return
  fi
  {
    echo "$(date --iso-8601=seconds) launch $name logit export on physical GPU $gpu_id"
    wait_for_memory
    (
      cd "$repo_dir"
      CUDA_VISIBLE_DEVICES="$gpu_id" HF_HOME="$hf_home" PYTHONPATH=. \
        .venv/bin/python train_luh/run_train_luh.py \
        --config-path ../auto_res/configs --config-name eval_official_uhead \
        "model.pretrained_model_name_or_path=$base_model" \
        "model.adapter_path=$adapter" \
        "ue_layer.path=$source_head" \
        "output_dir=auto_res/runs/calibration_logits_${name}_seed42" \
        do_eval=false do_predict=true +dataset.prediction_split=train
    )
  } 2>&1 | tee -a "$repo_dir/auto_res/logs/calibration_logits_${name}_seed42.log"
}

if [[ "$target" == gsm || "$target" == both ]]; then
  export_target gsm "$gsm_adapter"
fi
if [[ "$target" == finance || "$target" == both ]]; then
  export_target finance "$finance_adapter"
fi
