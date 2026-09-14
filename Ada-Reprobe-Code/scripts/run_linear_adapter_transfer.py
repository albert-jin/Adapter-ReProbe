#!/usr/bin/env python3
"""Run few-shot Adapter-ReProbe transfer on cached paired claim features.

The low-rank bridge is label-free: paired prompts are replayed through the base
and target checkpoints, and a reduced-rank residual map predicts the base
representation from the target representation. Calibration-only and
bridge+calibration variants consume the claim labels of the same support set.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--source-probe", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", default="10,25,50,100,200")
    parser.add_argument("--seeds", default="11,22,33,44,55")
    parser.add_argument("--ranks", default="4,8,16,32")
    parser.add_argument("--bridge-alpha", type=float, default=100.0)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--max-iter", type=int, default=2000)
    return parser.parse_args()


def expected_calibration_error(labels, probabilities, bins=10):
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, bins - 1)
    result = 0.0
    for index in range(bins):
        mask = assignments == index
        if np.any(mask):
            result += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(result)


def metrics(labels, probabilities):
    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-8, 1 - 1e-8)
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1_curve = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best_index = int(np.nanargmax(f1_curve))
    return {
        "average_precision": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "ece_10": expected_calibration_error(labels, probabilities),
        "accuracy_05": float(accuracy_score(labels, probabilities >= 0.5)),
        "f1_05": float(f1_score(labels, probabilities >= 0.5, zero_division=0)),
        "best_f1": float(f1_curve[best_index]),
        "best_f1_threshold": float(thresholds[best_index]) if best_index < len(thresholds) else 1.0,
    }


def source_logits(model, standardized_features):
    return model.named_steps["logisticregression"].decision_function(standardized_features)


def fit_calibrator(logits, labels):
    if len(np.unique(labels)) < 2:
        return None
    calibrator = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs")
    calibrator.fit(np.asarray(logits).reshape(-1, 1), labels)
    return calibrator


def calibrated_probabilities(calibrator, logits):
    if calibrator is None:
        return expit(logits)
    return calibrator.predict_proba(np.asarray(logits).reshape(-1, 1))[:, 1]


class LowRankPairedBridge:
    def __init__(self, rank, alpha, max_pairs, random_state):
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.max_pairs = int(max_pairs)
        self.random_state = int(random_state)

    def fit(self, target, source):
        target = np.asarray(target, dtype=np.float32)
        source = np.asarray(source, dtype=np.float32)
        if len(target) > self.max_pairs:
            rng = np.random.default_rng(self.random_state)
            selected = rng.choice(len(target), self.max_pairs, replace=False)
            target = target[selected]
            source = source[selected]
        delta = source - target
        self.target_mean_ = target.mean(axis=0)
        self.delta_mean_ = delta.mean(axis=0)
        centered_delta = delta - self.delta_mean_
        effective_rank = min(self.rank, len(target) - 1, target.shape[1])
        if effective_rank <= 0:
            self.basis_ = np.zeros((0, target.shape[1]), dtype=np.float32)
            self.regressor_ = None
            return self
        _, _, self.basis_ = randomized_svd(
            centered_delta,
            n_components=effective_rank,
            random_state=self.random_state,
        )
        coefficients = centered_delta @ self.basis_.T
        self.regressor_ = Ridge(
            alpha=self.alpha,
            fit_intercept=False,
            solver="lsqr",
        ).fit(target - self.target_mean_, coefficients)
        return self

    def transform(self, target):
        target = np.asarray(target, dtype=np.float32)
        if self.regressor_ is None:
            predicted_delta = np.broadcast_to(self.delta_mean_, target.shape)
        else:
            coefficients = self.regressor_.predict(target - self.target_mean_)
            predicted_delta = self.delta_mean_ + coefficients @ self.basis_
        return target + predicted_delta


def append_row(rows, target, method, size, seed, rank, elapsed, labels, probabilities, **extra):
    row = {
        "target": target,
        "method": method,
        "adaptation_examples": int(size),
        "seed": seed,
        "bridge_rank": rank,
        "adaptation_seconds": float(elapsed),
        **metrics(labels, probabilities),
        **extra,
    }
    rows.append(row)
    print(json.dumps(row), flush=True)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = np.load(args.source_features)
    target = np.load(args.target_features)
    for key in ("labels", "example_ids", "claim_ids"):
        if not np.array_equal(source[key], target[key]):
            raise ValueError(f"Source/target caches differ in {key}; paired transfer is unsafe")

    bundle = joblib.load(args.source_probe)
    model = bundle["classifier"]
    scaler = model.named_steps["standardscaler"]
    pooling = bundle["pooling"]
    layer = int(bundle["layer"])
    c_value = float(bundle["c"])
    labels = source["labels"].astype(np.int64)
    example_ids = source["example_ids"].astype(np.int64)
    partitions = {name: set(ids) for name, ids in bundle["partitions"].items()}
    train_ids = np.asarray(sorted(partitions["train"]), dtype=np.int64)
    train_mask = np.isin(example_ids, train_ids)
    test_mask = np.isin(example_ids, list(partitions["test"]))

    source_raw = source[pooling][:, layer].astype(np.float32)
    target_raw = target[pooling][:, layer].astype(np.float32)
    source_z = scaler.transform(source_raw)
    target_z = scaler.transform(target_raw)
    test_labels = labels[test_mask]
    rows = []

    source_base_prob = expit(source_logits(model, source_z[test_mask]))
    append_row(
        rows,
        "base",
        "source_probe",
        0,
        "fixed",
        0,
        0.0,
        test_labels,
        source_base_prob,
        pooling=pooling,
        layer=layer,
        source_c=c_value,
    )
    zero_logits = source_logits(model, target_z[test_mask])
    zero_prob = expit(zero_logits)
    append_row(
        rows,
        args.target_name,
        "zero_shot",
        0,
        "fixed",
        0,
        0.0,
        test_labels,
        zero_prob,
        pooling=pooling,
        layer=layer,
        source_c=c_value,
    )

    full_start = time.time()
    full_target = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=args.max_iter,
            random_state=bundle["split_seed"],
            solver="liblinear",
        ),
    )
    full_target.fit(target_raw[train_mask], labels[train_mask])
    full_prob = full_target.predict_proba(target_raw[test_mask])[:, 1]
    append_row(
        rows,
        args.target_name,
        "full_target_retrain",
        len(train_ids),
        "fixed",
        0,
        time.time() - full_start,
        test_labels,
        full_prob,
        pooling=pooling,
        layer=layer,
        source_c=c_value,
    )

    sizes = [int(value) for value in args.sizes.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    ranks = [int(value) for value in args.ranks.split(",")]
    for size in sizes:
        if size > len(train_ids):
            continue
        for seed in seeds:
            rng = np.random.default_rng(seed)
            support_ids = rng.choice(train_ids, size=size, replace=False)
            support_mask = np.isin(example_ids, support_ids)
            support_labels = labels[support_mask]

            start = time.time()
            mean_delta = (source_z[support_mask] - target_z[support_mask]).mean(axis=0)
            mean_test_z = target_z[test_mask] + mean_delta
            mean_logits = source_logits(model, mean_test_z)
            append_row(
                rows, args.target_name, "mean_shift", size, seed, 0,
                time.time() - start, test_labels, expit(mean_logits),
                support_claims=int(support_mask.sum()), pooling=pooling, layer=layer,
            )

            start = time.time()
            calibrator = fit_calibrator(
                source_logits(model, target_z[support_mask]), support_labels
            )
            append_row(
                rows, args.target_name, "calibration", size, seed, 0,
                time.time() - start, test_labels,
                calibrated_probabilities(calibrator, zero_logits),
                support_claims=int(support_mask.sum()), pooling=pooling, layer=layer,
            )

            start = time.time()
            fewshot = None
            if len(np.unique(support_labels)) >= 2:
                fewshot = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c_value,
                        class_weight="balanced",
                        max_iter=args.max_iter,
                        random_state=seed,
                        solver="liblinear",
                    ),
                )
                fewshot.fit(target_raw[support_mask], support_labels)
                fewshot_prob = fewshot.predict_proba(target_raw[test_mask])[:, 1]
                append_row(
                    rows, args.target_name, "target_from_scratch", size, seed, 0,
                    time.time() - start, test_labels, fewshot_prob,
                    support_claims=int(support_mask.sum()), pooling=pooling, layer=layer,
                )

            for rank in ranks:
                start = time.time()
                bridge = LowRankPairedBridge(
                    rank=rank,
                    alpha=args.bridge_alpha,
                    max_pairs=args.max_pairs,
                    random_state=seed,
                ).fit(target_z[support_mask], source_z[support_mask])
                bridged_support = bridge.transform(target_z[support_mask])
                bridged_test = bridge.transform(target_z[test_mask])
                bridged_support_logits = source_logits(model, bridged_support)
                bridged_test_logits = source_logits(model, bridged_test)
                elapsed = time.time() - start
                append_row(
                    rows, args.target_name, "paired_low_rank_bridge", size, seed, rank,
                    elapsed, test_labels, expit(bridged_test_logits),
                    support_claims=int(support_mask.sum()), pooling=pooling, layer=layer,
                    bridge_alpha=args.bridge_alpha, max_pairs=args.max_pairs,
                )
                start = time.time()
                bridge_calibrator = fit_calibrator(bridged_support_logits, support_labels)
                append_row(
                    rows, args.target_name, "bridge_calibration", size, seed, rank,
                    elapsed + time.time() - start, test_labels,
                    calibrated_probabilities(bridge_calibrator, bridged_test_logits),
                    support_claims=int(support_mask.sum()), pooling=pooling, layer=layer,
                    bridge_alpha=args.bridge_alpha, max_pairs=args.max_pairs,
                )

    fieldnames = sorted({key for row in rows for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    aggregate = []
    groups = {}
    for row in rows:
        key = (row["target"], row["method"], row["adaptation_examples"], row["bridge_rank"])
        groups.setdefault(key, []).append(row)
    metric_names = ("average_precision", "roc_auc", "brier", "ece_10", "best_f1")
    for key, group in groups.items():
        summary = {
            "target": key[0],
            "method": key[1],
            "adaptation_examples": key[2],
            "bridge_rank": key[3],
            "runs": len(group),
        }
        for metric_name in metric_names:
            values = np.asarray([row[metric_name] for row in group], dtype=float)
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        aggregate.append(summary)
    aggregate.sort(key=lambda row: (row["target"], row["adaptation_examples"], row["method"], row["bridge_rank"]))
    aggregate_path = output_path.with_suffix(".aggregate.csv")
    with aggregate_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(aggregate[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(aggregate)

    zero_ap = next(row["average_precision"] for row in rows if row["method"] == "zero_shot")
    full_ap = next(row["average_precision"] for row in rows if row["method"] == "full_target_retrain")
    best_adapted = max(
        (row for row in aggregate if row["method"] not in {"zero_shot", "full_target_retrain", "source_probe"}),
        key=lambda row: row["average_precision_mean"],
    )
    summary = {
        "target": args.target_name,
        "source_probe": str(Path(args.source_probe).resolve()),
        "pooling": pooling,
        "layer": layer,
        "source_c": c_value,
        "zero_shot_average_precision": zero_ap,
        "full_target_average_precision": full_ap,
        "best_adapted": best_adapted,
        "best_recovery_fraction_of_gap": (
            (best_adapted["average_precision_mean"] - zero_ap) / (full_ap - zero_ap)
            if abs(full_ap - zero_ap) > 1e-12 else None
        ),
        "raw_results": str(output_path.resolve()),
        "aggregate_results": str(aggregate_path.resolve()),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
