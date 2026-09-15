#!/usr/bin/env bash
set -euo pipefail

# Export compact held-out claim logits for trajectory-level selective-risk
# analysis. The three evaluations run sequentially on one authorized GPU.

if [[ $# -ne 1 || ! "$1" =~ ^[0-3]$ ]]; then
  echo "usage: $0 GPU_ID (0..3)" >&2
  exit 2
fi

gpu_id="$1"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
hf_home="$project_dir/shared_cache/huggingface"
base_model="$hf_home/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
gsm_adapter="$hf_home/hub/models--ehzawad--qwen3-1.7b-gsm8k-grpo/snapshots/83adf766d2d6dbbd3e9b67aa6858cf55e1b124c8"
finance_adapter="$hf_home/hub/models--fotapol--qwen3-1.7b-financial-qa-lora/snapshots/8433fcf8142e3db64f38df1f6eeaa38bf2e92651"
source_head="$repo_dir/auto_res/runs/src_current_l21_fixed_seed42/2026-09-04/20-50-29/model"

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

export_one() {
  local name="$1"
  local adapter="$2"
  local output_root="$repo_dir/auto_res/runs/$name"
  if find "$output_root" -name claim_predictions.npz -type f -print -quit \
      2>/dev/null | grep -q .; then
    echo "Skipping completed $name"
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
      "output_dir=auto_res/runs/$name" \
      do_eval=false do_predict=true
  )
}

export_one selective_source_seed42 null
export_one selective_gsm_zero_seed42 "$gsm_adapter"
export_one selective_finance_zero_seed42 "$finance_adapter"
