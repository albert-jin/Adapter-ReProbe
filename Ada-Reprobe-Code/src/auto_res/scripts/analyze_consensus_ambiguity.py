#!/usr/bin/env python3
"""Measure when a frozen verifier helps as a conservative plurality tie-breaker."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

try:
    from auto_res.scripts.analyze_natural_bestofn import numerically_equal
except ModuleNotFoundError:  # Direct execution from auto_res/scripts.
    from analyze_natural_bestofn import numerically_equal


def answer_groups(rows):
    """Return parsed-answer groups in first-candidate insertion order."""
    groups = {}
    for index, row in enumerate(rows):
        value = row["predicted_answer"]
        if value is None:
            continue
        groups.setdefault(round(float(value), 8), []).append(index)
    return groups


def ambiguity_unit(rows, head="source"):
    """Compare ordinary plurality with verifier-only tie breaking."""
    groups = answer_groups(rows)
    if not groups:
        return {
            "ambiguous": False,
            "intervened": False,
            "majority_correct": False,
            "tiebreak_correct": False,
        }
    largest = max(map(len, groups.values()))
    tied = [key for key, indices in groups.items() if len(indices) == largest]
    majority_key = tied[0]
    risks = [float(row["verifier_logits"][head]) for row in rows]
    tiebreak_key = min(tied, key=lambda key: min(risks[i] for i in groups[key]))
    reference = rows[0]["reference_answer"]
    return {
        "ambiguous": len(tied) > 1,
        "intervened": tiebreak_key != majority_key,
        "majority_correct": numerically_equal(majority_key, reference),
        "tiebreak_correct": numerically_equal(tiebreak_key, reference),
    }


def load_units(paths, budget, head):
    units = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        summary = json.loads((path.parent / "summary.json").read_text())
        dataset_name = str(summary.get("dataset", "gsm8k"))
        if dataset_name.startswith("gsm8k"):
            dataset_name = "gsm8k"
        grouped = defaultdict(list)
        for row in records:
            grouped[int(row["question_id"])].append(row)
        for question_id, rows in grouped.items():
            rows = sorted(rows, key=lambda row: row["sample_id"])[:budget]
            if len(rows) != budget:
                raise ValueError(
                    f"{path.parent.name} question {question_id} has {len(rows)} "
                    f"candidates; expected {budget}"
                )
            units.append(
                {
                    "run_name": path.parent.name,
                    "generation_seed": int(summary["seed"]),
                    "dataset": dataset_name,
                    "question_id": question_id,
                    **ambiguity_unit(rows, head),
                }
            )
    return units


def summarize(units):
    total = len(units)
    ambiguous = [unit for unit in units if unit["ambiguous"]]
    wins = sum(unit["tiebreak_correct"] and not unit["majority_correct"] for unit in units)
    losses = sum(unit["majority_correct"] and not unit["tiebreak_correct"] for unit in units)
    discordant = wins + losses
    return {
        "num_question_units": total,
        "num_unique_questions": len({unit["question_id"] for unit in units}),
        "num_ambiguous": len(ambiguous),
        "ambiguity_fraction": len(ambiguous) / total,
        "num_intervened": sum(unit["intervened"] for unit in units),
        "intervention_fraction": sum(unit["intervened"] for unit in units) / total,
        "majority_accuracy": float(np.mean([unit["majority_correct"] for unit in units])),
        "tiebreak_accuracy": float(np.mean([unit["tiebreak_correct"] for unit in units])),
        "delta_majority": float(
            np.mean(
                [unit["tiebreak_correct"] - unit["majority_correct"] for unit in units]
            )
        ),
        "ambiguous_majority_accuracy": (
            float(np.mean([unit["majority_correct"] for unit in ambiguous]))
            if ambiguous
            else None
        ),
        "ambiguous_tiebreak_accuracy": (
            float(np.mean([unit["tiebreak_correct"] for unit in ambiguous]))
            if ambiguous
            else None
        ),
        "wins_vs_majority": wins,
        "losses_vs_majority": losses,
        "mcnemar_exact_p": (
            float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
        ),
    }


def cluster_bootstrap(units, repetitions, seed):
    by_question = defaultdict(list)
    for unit in units:
        by_question[unit["question_id"]].append(unit)
    question_ids = np.asarray(sorted(by_question))
    rng = np.random.default_rng(seed)
    deltas = []
    ambiguous_deltas = []
    for _ in range(repetitions):
        sampled = rng.choice(question_ids, len(question_ids), replace=True)
        draw = [unit for question_id in sampled for unit in by_question[int(question_id)]]
        differences = [
            unit["tiebreak_correct"] - unit["majority_correct"] for unit in draw
        ]
        deltas.append(np.mean(differences))
        ambiguous = [difference for difference, unit in zip(differences, draw) if unit["ambiguous"]]
        if ambiguous:
            ambiguous_deltas.append(np.mean(ambiguous))
    return {
        "delta_majority_cluster_ci95": np.quantile(deltas, [0.025, 0.975]).tolist(),
        "ambiguous_delta_cluster_ci95": (
            np.quantile(ambiguous_deltas, [0.025, 0.975]).tolist()
            if ambiguous_deltas
            else None
        ),
    }


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
    parser.add_argument("--head", default="source")
    parser.add_argument("--bootstrap-repetitions", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2028)
    args = parser.parse_args()

    units = load_units(args.records, args.budget, args.head)
    by_run = defaultdict(list)
    for unit in units:
        by_run[unit["run_name"]].append(unit)
    per_run = []
    for run_name, run_units in by_run.items():
        per_run.append(
            {
                "run_name": run_name,
                "generation_seed": run_units[0]["generation_seed"],
                "dataset": run_units[0]["dataset"],
                **summarize(run_units),
            }
        )
    pooled = summarize(units)
    pooled.update(cluster_bootstrap(units, args.bootstrap_repetitions, args.seed))
    dataset_names = {unit["dataset"] for unit in units}
    if len(dataset_names) != 1:
        raise ValueError(f"Ambiguity analysis must contain one dataset, got {dataset_names}")
    dataset_name = next(iter(dataset_names))

    csv_path = args.output_prefix.with_suffix(".per_run.csv")
    json_path = args.output_prefix.with_suffix(".summary.json")
    write_csv(csv_path, per_run)
    payload = {
        "records": [str(path.resolve()) for path in args.records],
        "record_sha256": {str(path.resolve()): sha256(path) for path in args.records},
        "budget_n": args.budget,
        "head": args.head,
        "dataset": dataset_name,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_unit": f"{dataset_name} source question_id cluster",
        "per_run_csv": str(csv_path.resolve()),
        "per_run": per_run,
        "pooled": pooled,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
