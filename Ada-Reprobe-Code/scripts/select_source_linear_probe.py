#!/usr/bin/env python3
"""Select a sparse source probe on validation data and evaluate once on test."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from datasets import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--c-values", default="0.01,0.1,1.0")
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def split_examples(example_ids, validation_fraction, test_fraction, seed):
    unique_examples = np.unique(example_ids).tolist()
    full = Dataset.from_dict({"example_id": unique_examples})
    heldout_fraction = validation_fraction + test_fraction
    first = full.train_test_split(test_size=heldout_fraction, seed=seed)
    relative_test_fraction = test_fraction / heldout_fraction
    second = first["test"].train_test_split(test_size=relative_test_fraction, seed=seed)
    return {
        "train": set(first["train"]["example_id"]),
        "validation": set(second["train"]["example_id"]),
        "test": set(second["test"]["example_id"]),
    }


def metrics(labels, probabilities):
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
    }


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    data = np.load(args.features)
    labels = data["labels"].astype(np.int64)
    example_ids = data["example_ids"].astype(np.int64)
    partitions = split_examples(
        example_ids, args.validation_fraction, args.test_fraction, args.seed
    )
    masks = {
        name: np.asarray([example_id in ids for example_id in example_ids])
        for name, ids in partitions.items()
    }
    c_values = [float(value) for value in args.c_values.split(",")]

    rows = []
    best = None
    for pooling in ("previous", "current"):
        features = data[pooling]
        for layer in range(features.shape[1]):
            for c_value in c_values:
                classifier = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        random_state=args.seed,
                        solver="liblinear",
                    ),
                )
                classifier.fit(
                    features[masks["train"], layer].astype(np.float32),
                    labels[masks["train"]],
                )
                probabilities = classifier.predict_proba(
                    features[masks["validation"], layer].astype(np.float32)
                )[:, 1]
                row = {
                    "pooling": pooling,
                    "layer": layer,
                    "c": c_value,
                    **metrics(labels[masks["validation"]], probabilities),
                }
                rows.append(row)
                if best is None or row["average_precision"] > best[0]["average_precision"]:
                    best = (row, classifier)
                print(json.dumps(row), flush=True)

    selected, classifier = best
    pooling = selected["pooling"]
    layer = selected["layer"]
    test_probabilities = classifier.predict_proba(
        data[pooling][masks["test"], layer].astype(np.float32)
    )[:, 1]
    test_metrics = metrics(labels[masks["test"]], test_probabilities)

    csv_path = prefix.with_suffix(".validation_scan.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    model_path = prefix.with_suffix(".joblib")
    joblib.dump(
        {
            "classifier": classifier,
            "pooling": pooling,
            "layer": layer,
            "c": selected["c"],
            "split_seed": args.seed,
            "partitions": {name: sorted(ids) for name, ids in partitions.items()},
        },
        model_path,
    )
    summary = {
        "features": str(Path(args.features).resolve()),
        "selection_protocol": "80/10/10 trajectory-grouped split; maximize validation AP",
        "split_seed": args.seed,
        "partition_examples": {name: len(ids) for name, ids in partitions.items()},
        "partition_claims": {name: int(mask.sum()) for name, mask in masks.items()},
        "partition_positive_rate": {
            name: float(labels[mask].mean()) for name, mask in masks.items()
        },
        "selected": selected,
        "untouched_test": test_metrics,
        "model_path": str(model_path.resolve()),
        "scan_path": str(csv_path.resolve()),
    }
    summary_path = prefix.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
