#!/usr/bin/env python3
"""Collect Hydra-native ReProbe runs into tidy, paper-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml


METRICS = (
    "eval_average_precision",
    "eval_roc_auc",
    "eval_f1",
    "eval_best_f1",
    "eval_brier",
    "eval_ece_10",
    "eval_loss",
)


def nested(mapping, *keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def adapter_name(path):
    path = str(path or "")
    if "financial-qa" in path:
        return "finance_r16"
    if "gsm8k-grpo" in path:
        return "gsm8k_grpo_r32"
    return "base"


def collect(runs_dir: Path):
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/*/*/eval_metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / ".hydra" / "config.yaml"
        if not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        train_metrics_path = run_dir / "train_metrics.json"
        train_metrics = (
            json.loads(train_metrics_path.read_text(encoding="utf-8"))
            if train_metrics_path.exists()
            else {}
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        adaptation = nested(config, "ue_layer", "adaptation", default={}) or {}
        row = {
            "run_name": run_dir.parents[1].name,
            "run_dir": str(run_dir.resolve()),
            "date": run_dir.parent.name,
            "time": run_dir.name,
            "target": adapter_name(nested(config, "model", "adapter_path")),
            "adapter_path": nested(config, "model", "adapter_path"),
            "source_head": nested(config, "ue_layer", "path"),
            "train_scope": nested(config, "ue_layer", "trainable_scope", default="all"),
            "calibration": bool(adaptation.get("calibration", False)),
            "bridge_rank": int(adaptation.get("bridge_rank", 0) or 0),
            "pos_weight": nested(config, "ue_layer", "pos_weight"),
            "train_examples": nested(config, "dataset", "train_subset", default=0),
            "seed": config.get("seed", 42),
            "epochs": nested(config, "training_arguments", "num_train_epochs"),
            "learning_rate": nested(config, "training_arguments", "learning_rate"),
            "do_train": bool(config.get("do_train", False)),
            "train_runtime": train_metrics.get("train_runtime"),
            "train_samples_per_second": train_metrics.get("train_samples_per_second"),
            "train_steps_per_second": train_metrics.get("train_steps_per_second"),
        }
        row.update({metric: metrics.get(metric) for metric in METRICS})
        rows.append(row)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No native results found for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("auto_res/runs"))
    parser.add_argument(
        "--output", type=Path, default=Path("auto_res/results/native_experiment_summary.csv")
    )
    args = parser.parse_args()
    rows = collect(args.runs_dir)
    write_csv(args.output, rows)
    summary = {
        "runs_dir": str(args.runs_dir.resolve()),
        "num_completed_native_runs": len(rows),
        "output": str(args.output.resolve()),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
