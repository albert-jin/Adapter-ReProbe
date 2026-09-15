#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 {base|primary|altgsm} {svamp|multiarith|asdiv} [SEEDS]" >&2
  exit 2
fi
target="$1"
dataset="$2"
seeds="${3:-2040,2041,2042}"
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
project_dir="$(dirname "$repo_dir")"
tokenizer="$project_dir/shared_cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

record_args=()
IFS=',' read -r -a seed_array <<< "$seeds"
for seed in "${seed_array[@]}"; do
  records="auto_res/runs/natural_bestofn_${target}_${dataset}_n100x8_seed${seed}_cap512/samples.jsonl"
  if [[ ! -f "$repo_dir/$records" ]]; then
    echo "Missing $records" >&2
    exit 1
  fi
  record_args+=(--records "$records")
  (
    cd "$repo_dir"
    .venv/bin/python auto_res/scripts/analyze_natural_bestofn.py \
      --records "$records" \
      --output-prefix "auto_res/results/natural_bestofn_${target}_${dataset}_seed${seed}_cap512" \
      --budgets 1,2,4,8 --bootstrap-repetitions 10000 --seed "$seed"
  ) > "$repo_dir/auto_res/logs/analyze_${target}_${dataset}_seed${seed}.log"
done

prefix="auto_res/results/natural_bestofn_${target}_${dataset}_cap512_replicates"
(
  cd "$repo_dir"
  PYTHONPATH=. .venv/bin/python auto_res/scripts/analyze_natural_replicates.py \
    "${record_args[@]}" --output-prefix "$prefix" \
    --budget 8 --bootstrap-repetitions 20000 --seed 2042
  PYTHONPATH=. .venv/bin/python auto_res/scripts/analyze_consensus_ambiguity.py \
    "${record_args[@]}" \
    --output-prefix "auto_res/results/natural_bestofn_${target}_${dataset}_cap512_ambiguity" \
    --budget 8 --bootstrap-repetitions 20000 --seed 2042
  HF_HOME="$project_dir/shared_cache/huggingface" \
    .venv/bin/python auto_res/scripts/audit_natural_bestofn.py \
    "${record_args[@]}" --tokenizer "$tokenizer" --max-new-tokens 512 \
    --output-prefix "auto_res/results/natural_bestofn_${target}_${dataset}_cap512_audit"
)
