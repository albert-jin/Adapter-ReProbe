#!/usr/bin/env python3
"""Evaluate claim separability across hidden-state layers with grouped splits."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(args.features)
    labels = data["labels"].astype(np.int64)
    example_ids = data["example_ids"].astype(np.int64)

    # Exactly match datasets.train_test_split(seed=42): split trajectories,
    # never individual claims.
    unique_examples = np.unique(example_ids)
    split = Dataset.from_dict({"example_id": unique_examples.tolist()}).train_test_split(
        test_size=args.test_fraction,
        seed=args.seed,
    )
    test_examples = set(split["test"]["example_id"])
    test_mask = np.asarray([example_id in test_examples for example_id in example_ids])
    train_mask = ~test_mask

    rows = []
    for pooling in ("previous", "current"):
        features = data[pooling]
        for layer in range(features.shape[1]):
            classifier = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=args.c,
                    class_weight="balanced",
                    max_iter=args.max_iter,
                    random_state=args.seed,
                    solver="liblinear",
                ),
            )
            classifier.fit(features[train_mask, layer].astype(np.float32), labels[train_mask])
            probabilities = classifier.predict_proba(
                features[test_mask, layer].astype(np.float32)
            )[:, 1]
            row = {
                "pooling": pooling,
                "layer": layer,
                "average_precision": average_precision_score(labels[test_mask], probabilities),
                "roc_auc": roc_auc_score(labels[test_mask], probabilities),
                "brier": brier_score_loss(labels[test_mask], probabilities),
                "train_claims": int(train_mask.sum()),
                "test_claims": int(test_mask.sum()),
                "test_positive_rate": float(labels[test_mask].mean()),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    best_ap = max(rows, key=lambda row: row["average_precision"])
    best_roc = max(rows, key=lambda row: row["roc_auc"])
    summary = {
        "features": str(Path(args.features).resolve()),
        "seed": args.seed,
        "split_protocol": "trajectory_grouped_90/10",
        "best_average_precision": best_ap,
        "best_roc_auc": best_roc,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
