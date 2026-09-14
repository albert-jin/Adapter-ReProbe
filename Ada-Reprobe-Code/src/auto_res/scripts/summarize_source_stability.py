#!/usr/bin/env python3
"""Summarize paired source/adapter results across native probe seeds.

The target metrics are paired with the source checkpoint trained at the same
seed.  In addition to raw metrics, the output therefore reports the AP change
and retention ratio induced by each adapter.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t


METRICS = (
    "eval_average_precision",
    "eval_roc_auc",
    "eval_f1",
    "eval_best_f1",
    "eval_brier",
    "eval_ece_10",
)


def latest_metrics(run_dir: Path) -> tuple[Path, dict]:
    candidates = list(run_dir.glob("*/*/eval_metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No eval_metrics.json under {run_dir}")
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    return path, json.loads(path.read_text(encoding="utf-8"))


def discover_runs(runs_dir: Path) -> dict[tuple[int, str], Path]:
    discovered = {}
    native_pattern = re.compile(
        r"src_current_l21_fixed_seed(?P<seed>\d+)"
        r"(?:_zero_(?P<target>gsm|finance))?$"
    )
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = native_pattern.fullmatch(run_dir.name)
        if match:
            target = match.group("target") or "source"
            discovered[(int(match.group("seed")), target)] = run_dir

    # Seed 42 target evaluations predate the standardized multi-seed names.
    legacy = {
        (42, "gsm"): runs_dir / "l21_zero_gsm8k_grpo",
        (42, "finance"): runs_dir / "l21_zero_finance",
    }
    for key, run_dir in legacy.items():
        if run_dir.exists() and key not in discovered:
            discovered[key] = run_dir
    return discovered


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows to write to {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    mean = float(array.mean())
    if len(array) < 2:
        return {
            "n": len(array),
            "mean": mean,
            "std": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
        }
    std = float(array.std(ddof=1))
    half_width = float(t.ppf(0.975, df=len(array) - 1) * std / np.sqrt(len(array)))
    return {
        "n": len(array),
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("auto_res/runs"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("auto_res/results/source_seed_stability"),
    )
    args = parser.parse_args()

    discovered = discover_runs(args.runs_dir)
    seeds = sorted(seed for seed, target in discovered if target == "source")
    required = {(seed, target) for seed in seeds for target in ("source", "gsm", "finance")}
    missing = sorted(required - set(discovered))
    if missing:
        raise RuntimeError(f"Missing paired evaluations: {missing}")

    rows = []
    for seed in seeds:
        source_path, source_metrics = latest_metrics(discovered[(seed, "source")])
        source_ap = float(source_metrics["eval_average_precision"])
        for target in ("source", "gsm", "finance"):
            path, metrics = latest_metrics(discovered[(seed, target)])
            row = {
                "seed": seed,
                "target": target,
                "run_name": discovered[(seed, target)].name,
                "metrics_path": str(path.resolve()),
            }
            row.update({metric: float(metrics[metric]) for metric in METRICS})
            target_ap = row["eval_average_precision"]
            row["ap_delta_from_paired_source"] = target_ap - source_ap
            row["ap_retention_from_paired_source"] = target_ap / source_ap
            rows.append(row)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["target"]].append(row)
    aggregate_rows = []
    aggregate_metrics = METRICS + (
        "ap_delta_from_paired_source",
        "ap_retention_from_paired_source",
    )
    for target in ("source", "gsm", "finance"):
        for metric in aggregate_metrics:
            stats = summarize([row[metric] for row in grouped[target]])
            aggregate_rows.append({"target": target, "metric": metric, **stats})

    per_seed_path = args.output_prefix.with_suffix(".csv")
    aggregate_path = args.output_prefix.parent / (
        args.output_prefix.name + ".aggregate.csv"
    )
    summary_path = args.output_prefix.with_suffix(".summary.json")
    write_csv(per_seed_path, rows)
    write_csv(aggregate_path, aggregate_rows)
    payload = {
        "seeds": seeds,
        "num_seeds": len(seeds),
        "per_seed_csv": str(per_seed_path.resolve()),
        "aggregate_csv": str(aggregate_path.resolve()),
        "aggregate": {
            target: {
                metric: summarize([row[metric] for row in grouped[target]])
                for metric in aggregate_metrics
            }
            for target in ("source", "gsm", "finance")
        },
        "note": "95% intervals are Student-t intervals over paired training seeds.",
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
