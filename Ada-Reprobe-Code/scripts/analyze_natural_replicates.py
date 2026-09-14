#!/usr/bin/env python3
"""Pool natural Best-of-N replicates with question-clustered uncertainty."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

try:
    from auto_res.scripts.analyze_natural_bestofn import (
        majority_correct,
        verifier_consensus_correct,
    )
except ModuleNotFoundError:  # Direct execution from auto_res/scripts.
    from analyze_natural_bestofn import majority_correct, verifier_consensus_correct


def load_units(paths, budget):
    units = []
    question_sets = {}
    run_metadata = {}
    for path in paths:
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        summary_path = path.parent / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing run summary for {path}: {summary_path}")
        summary = json.loads(summary_path.read_text())
        generation_seed = int(summary["seed"])
        dataset_name = str(
            summary.get("dataset", records[0].get("dataset", "gsm8k"))
        )
        if dataset_name.startswith("gsm8k"):
            dataset_name = "gsm8k"
        run_metadata[generation_seed] = {
            "dataset": dataset_name,
            "base_model": summary.get("base_model"),
            "adapter": summary.get("adapter"),
            "dataset_repo": summary.get("dataset_repo"),
            "dataset_revision": summary.get("dataset_revision"),
            "dataset_filter_rule": summary.get("dataset_filter_rule"),
            "dataset_total_rows": summary.get("dataset_total_rows"),
            "dataset_eligible_rows": summary.get("dataset_eligible_rows"),
            "dataset_excluded_rows": summary.get("dataset_excluded_rows"),
            "dataset_eligible_sha256": summary.get("dataset_eligible_sha256"),
            "dataset_sample_sha256": summary.get("dataset_sample_sha256"),
        }
        grouped = defaultdict(list)
        for row in records:
            grouped[int(row["question_id"])].append(row)
        question_sets[generation_seed] = set(grouped)

        for question_id, rows in grouped.items():
            rows = sorted(rows, key=lambda row: row["sample_id"])[:budget]
            if len(rows) != budget:
                raise ValueError(
                    f"Seed {generation_seed}, question {question_id} has "
                    f"{len(rows)} candidates; expected {budget}"
                )
            source_risks = [row["verifier_logits"]["source"] for row in rows]
            bridge_names = [
                name for name in rows[0]["verifier_logits"] if name.startswith("bridge")
            ]
            outcomes = {
                "first_sample": float(rows[0]["correct"]),
                "random_choice_expected": float(
                    np.mean([row["correct"] for row in rows])
                ),
                "majority_vote": float(majority_correct(rows)),
                "source": float(rows[int(np.argmin(source_risks))]["correct"]),
                "source_rank_weighted_vote": float(
                    verifier_consensus_correct(rows, source_risks, "rank_weighted")
                ),
                "source_plurality_tiebreak_vote": float(
                    verifier_consensus_correct(rows, source_risks, "plurality_tiebreak")
                ),
                "pass_at_n_oracle": float(any(row["correct"] for row in rows)),
            }
            if len(bridge_names) >= 2:
                bridge_risks = [
                    float(
                        np.mean(
                            [row["verifier_logits"][name] for name in bridge_names]
                        )
                    )
                    for row in rows
                ]
                outcomes.update(
                    {
                        "bridge_ensemble": float(
                            rows[int(np.argmin(bridge_risks))]["correct"]
                        ),
                        "bridge_ensemble_rank_weighted_vote": float(
                            verifier_consensus_correct(
                                rows, bridge_risks, "rank_weighted"
                            )
                        ),
                        "bridge_ensemble_plurality_tiebreak_vote": float(
                            verifier_consensus_correct(
                                rows, bridge_risks, "plurality_tiebreak"
                            )
                        ),
                    }
                )
            units.append(
                {
                    "generation_seed": generation_seed,
                    "question_id": question_id,
                    **outcomes,
                }
            )
    return units, question_sets, run_metadata


def cluster_bootstrap(units, methods, repetitions, seed):
    grouped = defaultdict(list)
    for unit in units:
        grouped[unit["question_id"]].append(unit)
    cluster_ids = np.asarray(sorted(grouped))
    rng = np.random.default_rng(seed)
    draws = {method: [] for method in methods}
    delta_random = {method: [] for method in methods}
    delta_majority = {method: [] for method in methods}
    for _ in range(repetitions):
        sampled_ids = rng.choice(cluster_ids, len(cluster_ids), replace=True)
        sample = [unit for question_id in sampled_ids for unit in grouped[int(question_id)]]
        random_accuracy = np.mean([unit["random_choice_expected"] for unit in sample])
        majority_accuracy = np.mean([unit["majority_vote"] for unit in sample])
        for method in methods:
            accuracy = np.mean([unit[method] for unit in sample])
            draws[method].append(accuracy)
            delta_random[method].append(accuracy - random_accuracy)
            delta_majority[method].append(accuracy - majority_accuracy)
    return draws, delta_random, delta_majority


def interval(values):
    return np.quantile(np.asarray(values, dtype=float), [0.025, 0.975]).tolist()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    units, question_sets, run_metadata = load_units(args.records, args.budget)
    dataset_names = {metadata["dataset"] for metadata in run_metadata.values()}
    if len(dataset_names) != 1:
        raise ValueError(f"A replicate analysis must contain one dataset, got {dataset_names}")
    dataset_name = next(iter(dataset_names))
    methods = [
        key
        for key in units[0]
        if key not in {"generation_seed", "question_id"}
    ]
    draws, delta_random, delta_majority = cluster_bootstrap(
        units, methods, args.bootstrap_repetitions, args.seed
    )
    unique_questions = set.union(*question_sets.values())
    random_accuracy = float(np.mean([unit["random_choice_expected"] for unit in units]))
    majority_accuracy = float(np.mean([unit["majority_vote"] for unit in units]))
    rows = []
    for method in methods:
        accuracy = float(np.mean([unit[method] for unit in units]))
        accuracy_ci = interval(draws[method])
        random_ci = interval(delta_random[method])
        majority_ci = interval(delta_majority[method])
        rows.append(
            {
                "budget_n": args.budget,
                "method": method,
                "num_question_seed_units": len(units),
                "num_unique_questions": len(unique_questions),
                "accuracy": accuracy,
                "accuracy_cluster_ci95_low": accuracy_ci[0],
                "accuracy_cluster_ci95_high": accuracy_ci[1],
                "delta_random": accuracy - random_accuracy,
                "delta_random_cluster_ci95_low": random_ci[0],
                "delta_random_cluster_ci95_high": random_ci[1],
                "delta_majority": accuracy - majority_accuracy,
                "delta_majority_cluster_ci95_low": majority_ci[0],
                "delta_majority_cluster_ci95_high": majority_ci[1],
            }
        )

    pooled_path = args.output_prefix.parent / (args.output_prefix.name + ".pooled.csv")
    summary_path = args.output_prefix.parent / (args.output_prefix.name + ".pooled.json")
    write_csv(pooled_path, rows)
    overlaps = {
        f"{left}-{right}": len(question_sets[left] & question_sets[right])
        for left, right in combinations(sorted(question_sets), 2)
    }
    payload = {
        "records": [str(path.resolve()) for path in args.records],
        "record_sha256": {str(path.resolve()): sha256(path) for path in args.records},
        "budget_n": args.budget,
        "generation_seeds": sorted(question_sets),
        "dataset": dataset_name,
        "run_metadata": run_metadata,
        "num_question_seed_units": len(units),
        "num_unique_questions": len(unique_questions),
        "pairwise_question_overlaps": overlaps,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": f"{dataset_name} source question_id cluster",
        "pooled_csv": str(pooled_path.resolve()),
        "rows": rows,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
