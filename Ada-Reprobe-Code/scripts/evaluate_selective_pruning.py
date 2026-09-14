#!/usr/bin/env python3
"""Evaluate claim and trajectory selective-risk utility from saved logits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import expit


def aurc(labels, scores):
    order = np.argsort(scores, kind="stable")
    cumulative_risk = np.cumsum(labels[order]) / np.arange(1, len(labels) + 1)
    return float(cumulative_risk.mean())


def oracle_aurc(labels):
    return aurc(labels, labels)


def risk_curve(labels, scores, level, name):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(scores, kind="stable")
    rows = []
    for coverage in np.linspace(0.1, 1.0, 10):
        retained = max(1, int(np.ceil(coverage * len(labels))))
        retained_labels = labels[order[:retained]]
        rows.append(
            {
                "method": name,
                "level": level,
                "coverage": float(coverage),
                "retained": retained,
                "risk": float(retained_labels.mean()),
                "clean_accuracy": float(1.0 - retained_labels.mean()),
            }
        )
    return rows


def trajectory_arrays(data):
    example_ids = data["example_indices"].astype(np.int64)
    labels = data["labels"].astype(np.int8)
    scores = expit(data["logits"].astype(float))
    unique_ids = np.unique(example_ids)
    trajectory_labels = np.asarray(
        [labels[example_ids == idx].max() for idx in unique_ids], dtype=np.int8
    )
    trajectory_scores = np.asarray(
        [scores[example_ids == idx].max() for idx in unique_ids], dtype=float
    )
    return trajectory_labels, trajectory_scores


def detection_recall(labels, scores, reject_fraction):
    reject_count = max(1, int(np.ceil(reject_fraction * len(labels))))
    rejected = np.argsort(scores, kind="stable")[-reject_count:]
    positives = int(labels.sum())
    return float(labels[rejected].sum() / positives) if positives else float("nan")


def evaluate(name, path):
    data = np.load(path)
    claim_labels = data["labels"].astype(np.int8)
    claim_scores = expit(data["logits"].astype(float))
    trajectory_labels, trajectory_scores = trajectory_arrays(data)
    summary = {
        "method": name,
        "prediction_path": str(Path(path).resolve()),
        "num_claims": int(len(claim_labels)),
        "claim_error_rate": float(claim_labels.mean()),
        "claim_aurc": aurc(claim_labels, claim_scores),
        "claim_oracle_aurc": oracle_aurc(claim_labels),
        "num_trajectories": int(len(trajectory_labels)),
        "trajectory_error_rate": float(trajectory_labels.mean()),
        "trajectory_aurc": aurc(trajectory_labels, trajectory_scores),
        "trajectory_oracle_aurc": oracle_aurc(trajectory_labels),
    }
    for fraction in (0.1, 0.2, 0.3):
        key = int(fraction * 100)
        summary[f"trajectory_error_recall_at_reject_{key}"] = detection_recall(
            trajectory_labels, trajectory_scores, fraction
        )
        retained_count = len(trajectory_labels) - max(
            1, int(np.ceil(fraction * len(trajectory_labels)))
        )
        retained = np.argsort(trajectory_scores, kind="stable")[:retained_count]
        summary[f"trajectory_clean_accuracy_after_reject_{key}"] = float(
            1.0 - trajectory_labels[retained].mean()
        )
    curves = risk_curve(claim_labels, claim_scores, "claim", name)
    curves += risk_curve(trajectory_labels, trajectory_scores, "trajectory", name)
    return summary, curves


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="Named artifact in the form NAME=path/to/claim_predictions.npz",
    )
    parser.add_argument(
        "--output-prefix", type=Path, default=Path("auto_res/results/selective_pruning")
    )
    args = parser.parse_args()

    summaries = []
    curves = []
    for item in args.prediction:
        if "=" not in item:
            raise ValueError(f"prediction must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        summary, curve = evaluate(name, Path(path))
        summaries.append(summary)
        curves.extend(curve)

    write_csv(args.output_prefix.with_suffix(".summary.csv"), summaries)
    write_csv(args.output_prefix.with_suffix(".curve.csv"), curves)
    args.output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(summaries, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
