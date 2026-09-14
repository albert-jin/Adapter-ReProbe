#!/usr/bin/env python3
"""Aggregate fixed-protocol natural Best-of-N curves across generation seeds."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


METRICS = ("accuracy", "delta_random", "delta_majority")


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    std = float(values.std(ddof=1))
    half_width = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
    return mean, std, mean - half_width, mean + half_width


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--curve", action="append", type=Path, required=True)
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("auto_res/results/natural_bestofn_replicates"),
    )
    args = parser.parse_args()

    rows = []
    for path in args.curve:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source_curve"] = str(path.resolve())
                rows.append(row)
    keys = {(int(row["budget_n"]), row["method"]) for row in rows}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["budget_n"]), row["method"])].append(row)

    aggregate = []
    for budget, method in sorted(keys):
        group = grouped[(budget, method)]
        seeds = sorted(int(row["generation_seed"]) for row in group)
        output = {
            "budget_n": budget,
            "method": method,
            "n_generation_seeds": len(group),
            "generation_seeds": ";".join(str(seed) for seed in seeds),
        }
        for metric in METRICS:
            mean, std, low, high = interval([float(row[metric]) for row in group])
            output.update(
                {
                    f"{metric}_mean": mean,
                    f"{metric}_std": std,
                    f"{metric}_t_ci95_low": low,
                    f"{metric}_t_ci95_high": high,
                }
            )
        aggregate.append(output)

    per_seed_path = args.output_prefix.parent / (args.output_prefix.name + ".per_seed.csv")
    aggregate_path = args.output_prefix.parent / (args.output_prefix.name + ".aggregate.csv")
    summary_path = args.output_prefix.parent / (args.output_prefix.name + ".summary.json")
    write_csv(per_seed_path, rows)
    write_csv(aggregate_path, aggregate)
    summary = {
        "curves": [str(path.resolve()) for path in args.curve],
        "generation_seeds": sorted({int(row["generation_seed"]) for row in rows}),
        "per_seed_csv": str(per_seed_path.resolve()),
        "aggregate_csv": str(aggregate_path.resolve()),
        "note": "Intervals are Student-t intervals across generation seeds.",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
