#!/usr/bin/env python3
"""Aggregate native calibration runs and compare them with zero-shot transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = [
    "eval_average_precision",
    "eval_roc_auc",
    "eval_f1",
    "eval_best_f1",
    "eval_brier",
    "eval_ece_10",
]
EFFICIENCY_METRICS = ["train_runtime"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("auto_res/results/native_experiment_summary.csv")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("auto_res/results/native_calibration_curve.csv")
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input)
    calibrated = data[
        (data["train_scope"] == "calibration") & data["calibration"].astype(bool)
    ].copy()
    if calibrated.empty:
        raise RuntimeError("No calibration runs found")

    grouped = calibrated.groupby(
        ["target", "pos_weight", "train_examples"], dropna=False, sort=True
    )
    rows = []
    for keys, frame in grouped:
        target, pos_weight, train_examples = keys
        row = {
            "target": target,
            "pos_weight": pos_weight,
            "train_examples": int(train_examples),
            "num_seeds": int(len(frame)),
            "seeds": ",".join(str(int(seed)) for seed in sorted(frame["seed"].unique())),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1) if len(frame) > 1 else 0.0
        for metric in EFFICIENCY_METRICS:
            row[f"{metric}_mean"] = frame[metric].mean()
            row[f"{metric}_std"] = frame[metric].std(ddof=1) if len(frame) > 1 else 0.0

        zero = data[
            (data["target"] == target)
            & (data["train_scope"] == "none")
            & (~data["calibration"].astype(bool))
            & (data["source_head"].astype(str).str.contains("src_current_l21_fixed_seed42"))
        ]
        if not zero.empty:
            zero = zero.iloc[0]
            for metric in METRICS:
                row[f"zero_shot_{metric}"] = zero[metric]
                row[f"delta_{metric}"] = row[f"{metric}_mean"] - zero[metric]
        rows.append(row)

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    summary = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "num_groups": len(output),
        "num_runs": len(calibrated),
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
