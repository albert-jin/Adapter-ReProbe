#!/usr/bin/env python3
"""Ablate scalar calibration components on fixed native verifier logits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


METHODS = (
    "zero_shot",
    "temperature_only",
    "bias_only",
    "temperature_bias",
    "constant_prevalence",
)


def expected_calibration_error(labels, probabilities, bins=10):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(
        np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1
    )
    result = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            result += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(result)


def bernoulli_nll(labels, logits):
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def fit_scalar_calibrator(logits, labels, method):
    """Fit the exact logit transform while keeping temperature positive."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if method == "temperature_only":
        initial = np.zeros(1)
        bounds = [(-8.0, 8.0)]

        def transform(params, values):
            return values / np.exp(np.clip(params[0], -10.0, 10.0))

    elif method == "bias_only":
        initial = np.zeros(1)
        bounds = [(-20.0, 20.0)]

        def transform(params, values):
            return values + params[0]

    elif method == "temperature_bias":
        initial = np.zeros(2)
        bounds = [(-8.0, 8.0), (-20.0, 20.0)]

        def transform(params, values):
            return values / np.exp(np.clip(params[0], -10.0, 10.0)) + params[1]

    else:
        raise ValueError(f"Unsupported fitted method: {method}")

    fitted = minimize(
        lambda params: bernoulli_nll(labels, transform(params, logits)),
        initial,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"{method} calibration failed: {fitted.message}")
    return fitted.x, transform


def calibrated_probabilities(method, train_logits, train_labels, test_logits):
    if method == "zero_shot":
        return expit(test_logits), {"log_temperature": 0.0, "bias": 0.0}
    if method == "constant_prevalence":
        # Jeffreys smoothing avoids infinite logits for a small one-class support set.
        probability = (float(np.sum(train_labels)) + 0.5) / (len(train_labels) + 1.0)
        return np.full(len(test_logits), probability), {
            "constant_probability": probability
        }
    parameters, transform = fit_scalar_calibrator(train_logits, train_labels, method)
    transformed = transform(parameters, np.asarray(test_logits, dtype=np.float64))
    if method == "temperature_only":
        fitted = {"log_temperature": float(parameters[0]), "bias": 0.0}
    elif method == "bias_only":
        fitted = {"log_temperature": 0.0, "bias": float(parameters[0])}
    else:
        fitted = {
            "log_temperature": float(parameters[0]),
            "bias": float(parameters[1]),
        }
    return expit(transformed), fitted


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10": expected_calibration_error(labels, probabilities),
        "nll": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def integer_sequence_sha256(values):
    payload = json.dumps(
        sorted(int(value) for value in values), separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--sizes", default="10,25,50,100")
    parser.add_argument("--seeds", default="42,43,44")
    return parser.parse_args()


def main():
    args = parse_args()
    train = np.load(args.train_predictions)
    test = np.load(args.test_predictions)
    train_examples = np.asarray(train["example_indices"], dtype=np.int64)
    train_labels = np.asarray(train["labels"], dtype=np.int8)
    train_logits = np.asarray(train["logits"], dtype=np.float64)
    test_labels = np.asarray(test["labels"], dtype=np.int8)
    test_logits = np.asarray(test["logits"], dtype=np.float64)
    if not (len(train_examples) == len(train_labels) == len(train_logits)):
        raise ValueError("Training prediction arrays have inconsistent lengths")
    if not (len(test_labels) == len(test_logits)):
        raise ValueError("Test prediction arrays have inconsistent lengths")

    unique_examples = np.unique(train_examples)
    sizes = [int(value) for value in args.sizes.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    rows = []
    support_sha256 = {}
    for size in sizes:
        if size > len(unique_examples):
            raise ValueError(f"Requested {size} support trajectories; only {len(unique_examples)} exist")
        for seed in seeds:
            rng = np.random.default_rng(seed)
            support_examples = rng.choice(unique_examples, size=size, replace=False)
            support_mask = np.isin(train_examples, support_examples)
            support_labels = train_labels[support_mask]
            support_logits = train_logits[support_mask]
            support_fingerprint = integer_sequence_sha256(support_examples)
            support_sha256[f"n{size}_seed{seed}"] = support_fingerprint
            for method in METHODS:
                probabilities, parameters = calibrated_probabilities(
                    method, support_logits, support_labels, test_logits
                )
                row = {
                    "target": args.target,
                    "method": method,
                    "adaptation_examples": size,
                    "seed": seed,
                    "support_claims": int(support_mask.sum()),
                    "support_positive_rate": float(support_labels.mean()),
                    "support_trajectory_sha256": support_fingerprint,
                    **metrics(test_labels, probabilities),
                    **parameters,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)

    raw_path = args.output_prefix.with_suffix(".csv")
    write_csv(raw_path, rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["adaptation_examples"])].append(row)
    aggregate = []
    metric_names = ("average_precision", "roc_auc", "brier", "ece_10", "nll")
    for (method, size), group in grouped.items():
        item = {
            "target": args.target,
            "method": method,
            "adaptation_examples": size,
            "runs": len(group),
        }
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in group], dtype=float)
            item[f"{metric_name}_mean"] = float(values.mean())
            item[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(item)
    aggregate.sort(key=lambda row: (row["adaptation_examples"], row["method"]))
    aggregate_path = args.output_prefix.with_suffix(".aggregate.csv")
    write_csv(aggregate_path, aggregate)
    payload = {
        "target": args.target,
        "train_predictions": str(args.train_predictions.resolve()),
        "test_predictions": str(args.test_predictions.resolve()),
        "prediction_sha256": {
            "train": sha256(args.train_predictions),
            "test": sha256(args.test_predictions),
        },
        "support_unit": "trajectory",
        "sizes": sizes,
        "seeds": seeds,
        "methods": list(METHODS),
        "support_trajectory_sha256": support_sha256,
        "num_train_trajectories": int(len(unique_examples)),
        "num_train_claims": int(len(train_labels)),
        "num_test_claims": int(len(test_labels)),
        "raw_csv": str(raw_path.resolve()),
        "aggregate_csv": str(aggregate_path.resolve()),
    }
    args.output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
