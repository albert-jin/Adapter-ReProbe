#!/usr/bin/env python3
"""Assemble complete checkpoint-by-dataset Best-of-N result cells."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


REPORT_METHODS = (
    "random_choice_expected",
    "majority_vote",
    "source",
    "source_rank_weighted_vote",
    "source_plurality_tiebreak_vote",
    "pass_at_n_oracle",
)


def parse_cell(value):
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "cell must have the form CHECKPOINT=DATASET=POOLED_JSON"
        )
    return parts[0], parts[1], Path(parts[2])


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", action="append", type=parse_cell, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--primary-checkpoint", default="primary")
    parser.add_argument(
        "--primary-datasets", default="gsm8k,svamp,multiarith,asdiv"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    source_paths = {}
    for checkpoint, dataset, path in args.cell:
        payload = json.loads(path.read_text())
        if payload.get("dataset", dataset) != dataset:
            raise ValueError(
                f"Cell says {dataset}, but {path} reports {payload.get('dataset')}"
            )
        by_method = {row["method"]: row for row in payload["rows"]}
        missing = set(REPORT_METHODS) - set(by_method)
        if missing:
            raise ValueError(f"{path} is missing methods {sorted(missing)}")
        source_paths[f"{checkpoint}/{dataset}"] = str(path.resolve())
        for method in REPORT_METHODS:
            result = by_method[method]
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "dataset": dataset,
                    "method": method,
                    "num_question_seed_units": payload["num_question_seed_units"],
                    "num_unique_questions": payload["num_unique_questions"],
                    "accuracy": result["accuracy"],
                    "accuracy_ci95_low": result["accuracy_cluster_ci95_low"],
                    "accuracy_ci95_high": result["accuracy_cluster_ci95_high"],
                    "delta_random": result["delta_random"],
                    "delta_random_ci95_low": result[
                        "delta_random_cluster_ci95_low"
                    ],
                    "delta_random_ci95_high": result[
                        "delta_random_cluster_ci95_high"
                    ],
                    "delta_majority": result["delta_majority"],
                    "delta_majority_ci95_low": result[
                        "delta_majority_cluster_ci95_low"
                    ],
                    "delta_majority_ci95_high": result[
                        "delta_majority_cluster_ci95_high"
                    ],
                }
            )
    rows.sort(key=lambda row: (row["checkpoint"], row["dataset"], row["method"]))
    raw_path = args.output_prefix.with_suffix(".csv")
    write_csv(raw_path, rows)

    primary_datasets = args.primary_datasets.split(",")
    primary_rows = {
        (row["dataset"], row["method"]): row
        for row in rows
        if row["checkpoint"] == args.primary_checkpoint
    }
    missing_cells = [
        (dataset, method)
        for dataset in primary_datasets
        for method in REPORT_METHODS
        if (dataset, method) not in primary_rows
    ]
    if missing_cells:
        raise ValueError(f"Primary suite is incomplete: {missing_cells}")
    signs = {
        dataset: float(primary_rows[(dataset, "source")]["delta_random"])
        for dataset in primary_datasets
    }
    macro = []
    for method in REPORT_METHODS:
        values = [
            float(primary_rows[(dataset, method)]["accuracy"])
            for dataset in primary_datasets
        ]
        deltas = [
            float(primary_rows[(dataset, method)]["delta_random"])
            for dataset in primary_datasets
        ]
        macro.append(
            {
                "method": method,
                "datasets": len(primary_datasets),
                "macro_accuracy": float(np.mean(values)),
                "macro_delta_random": float(np.mean(deltas)),
            }
        )
    macro_path = args.output_prefix.parent / (args.output_prefix.name + ".macro.csv")
    write_csv(macro_path, macro)
    payload = {
        "primary_checkpoint": args.primary_checkpoint,
        "primary_datasets": primary_datasets,
        "direct_source_delta_random_by_dataset": signs,
        "positive_dataset_count": int(sum(value > 0 for value in signs.values())),
        "nonnegative_dataset_count": int(sum(value >= 0 for value in signs.values())),
        "required_positive_dataset_count": 3,
        "majority_positive_gate_passed": sum(value > 0 for value in signs.values()) >= 3,
        "macro_rows": macro,
        "cells": source_paths,
        "matrix_csv": str(raw_path.resolve()),
        "macro_csv": str(macro_path.resolve()),
    }
    args.output_prefix.with_suffix(".summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
