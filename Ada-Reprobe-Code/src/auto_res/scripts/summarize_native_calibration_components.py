#!/usr/bin/env python3
"""Aggregate native one- and two-scalar calibration component runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRICS = (
    "eval_average_precision",
    "eval_roc_auc",
    "eval_brier",
    "eval_ece_10",
    "eval_loss",
    "train_runtime",
)
SCOPE_NAMES = {
    "calibration_temperature": "temperature_only",
    "calibration_bias": "bias_only",
    "calibration": "temperature_bias",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("auto_res/results/native_experiment_summary.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("auto_res/results/native_calibration_components.csv"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    calibrated = data[
        data["train_scope"].isin(SCOPE_NAMES)
        & data["calibration"].astype(bool)
        & (data["pos_weight"] == 1)
        & data["train_examples"].isin([10, 25, 50, 100])
    ].copy()
    calibrated["method"] = calibrated["train_scope"].map(SCOPE_NAMES)
    rows = []
    for keys, frame in calibrated.groupby(
        ["target", "method", "train_examples"], sort=True
    ):
        target, method, train_examples = keys
        row = {
            "target": target,
            "method": method,
            "train_examples": int(train_examples),
            "num_seeds": int(frame["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in sorted(frame["seed"].unique())),
        }
        for metric in METRICS:
            values = frame[metric].dropna()
            row[f"{metric}_mean"] = values.mean() if len(values) else None
            row[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else 0.0
        rows.append(row)
    output = pd.DataFrame(rows).sort_values(["target", "train_examples", "method"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    expected = {
        (target, method, size)
        for target in ("gsm8k_grpo_r32", "finance_r16")
        for method in SCOPE_NAMES.values()
        for size in (10, 25, 50, 100)
    }
    observed = {
        (row.target, row.method, int(row.train_examples))
        for row in output.itertuples()
        if int(row.num_seeds) == 3
    }
    payload = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "expected_complete_cells": len(expected),
        "observed_complete_cells": len(expected & observed),
        "missing_cells": sorted([list(value) for value in expected - observed]),
        "complete": expected <= observed,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
