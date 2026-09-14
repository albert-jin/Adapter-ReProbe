#!/usr/bin/env python3
"""Create paper-ready Best-of-N budget curves from fixed generated samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, rankdata


def numerically_equal(left, right):
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)


def majority_correct(rows):
    parsed = [row["predicted_answer"] for row in rows if row["predicted_answer"] is not None]
    if not parsed:
        return False
    rounded = [round(float(value), 8) for value in parsed]
    majority = Counter(rounded).most_common(1)[0][0]
    return numerically_equal(majority, rows[0]["reference_answer"])


def answer_groups(rows):
    """Group candidates by parsed answer without merging parse failures."""
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        value = row["predicted_answer"]
        key = ("missing", index) if value is None else ("number", round(float(value), 8))
        groups[key].append(index)
    return groups


def verifier_consensus_correct(rows, risks, mode):
    """Combine answer agreement with within-question verifier rankings."""
    risks = np.asarray(risks, dtype=float)
    groups = answer_groups(rows)
    if mode == "rank_weighted":
        # Best (lowest-risk) candidates receive the largest weight. Average
        # ranks make the rule deterministic and fair when bf16 logits tie.
        reliability = len(rows) + 1.0 - rankdata(risks, method="average")
        group_score = {
            key: float(reliability[indices].sum()) for key, indices in groups.items()
        }
        best_score = max(group_score.values())
        candidates = [key for key, score in group_score.items() if score == best_score]
    elif mode == "plurality_tiebreak":
        largest = max(len(indices) for indices in groups.values())
        candidates = [key for key, indices in groups.items() if len(indices) == largest]
    else:
        raise ValueError(f"Unknown verifier-consensus mode: {mode}")
    best_key = min(candidates, key=lambda key: float(risks[groups[key]].min()))
    if best_key[0] == "missing":
        return False
    return numerically_equal(best_key[1], rows[0]["reference_answer"])


def bootstrap(values, random_choice, majority, seed, repetitions):
    values = np.asarray(values, dtype=float)
    random_choice = np.asarray(random_choice, dtype=float)
    majority = np.asarray(majority, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))

    def interval(array):
        return np.quantile(array, [0.025, 0.975]).tolist()

    return {
        "accuracy_ci95": interval(values[indices].mean(axis=1)),
        "delta_random": float((values - random_choice).mean()),
        "delta_random_ci95": interval((values - random_choice)[indices].mean(axis=1)),
        "delta_majority": float((values - majority).mean()),
        "delta_majority_ci95": interval((values - majority)[indices].mean(axis=1)),
    }


def exact_paired_test(selected, majority):
    selected = np.asarray(selected, dtype=np.int8)
    majority = np.asarray(majority, dtype=np.int8)
    wins = int(np.sum((selected == 1) & (majority == 0)))
    losses = int(np.sum((selected == 0) & (majority == 1)))
    discordant = wins + losses
    p_value = float(binomtest(wins, discordant, 0.5).pvalue) if discordant else 1.0
    return wins, losses, p_value


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--budgets", default="1,2,4,8")
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    records = [
        json.loads(line)
        for line in args.records.read_text(encoding="utf-8").splitlines()
        if line
    ]
    run_summary_path = args.records.parent / "summary.json"
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.exists()
        else {}
    )
    generation_seed = int(run_summary.get("seed", args.seed))
    grouped = defaultdict(list)
    for row in records:
        grouped[int(row["question_id"])].append(row)
    groups = [sorted(rows, key=lambda row: row["sample_id"]) for rows in grouped.values()]
    head_names = list(records[0]["verifier_logits"])
    bridge_names = [name for name in head_names if name.startswith("bridge")]
    budgets = [int(value) for value in args.budgets.split(",")]
    max_samples = min(len(rows) for rows in groups)
    if any(budget < 1 or budget > max_samples for budget in budgets):
        raise ValueError(f"Budgets must be within 1..{max_samples}")

    result_rows = []
    for budget_index, budget in enumerate(budgets):
        prefixes = [rows[:budget] for rows in groups]
        first = np.asarray([rows[0]["correct"] for rows in prefixes], dtype=float)
        random_choice = np.asarray(
            [np.mean([row["correct"] for row in rows]) for rows in prefixes], dtype=float
        )
        majority = np.asarray([majority_correct(rows) for rows in prefixes], dtype=float)
        oracle = np.asarray([any(row["correct"] for row in rows) for rows in prefixes], dtype=float)
        outcomes = {
            "first_sample": first,
            "random_choice_expected": random_choice,
            "majority_vote": majority,
            "pass_at_n_oracle": oracle,
        }
        for name in head_names:
            outcomes[name] = np.asarray(
                [min(rows, key=lambda row: row["verifier_logits"][name])["correct"] for rows in prefixes],
                dtype=float,
            )
            for mode in ("rank_weighted", "plurality_tiebreak"):
                outcomes[f"{name}_{mode}_vote"] = np.asarray(
                    [
                        verifier_consensus_correct(
                            rows,
                            [row["verifier_logits"][name] for row in rows],
                            mode,
                        )
                        for rows in prefixes
                    ],
                    dtype=float,
                )
        if len(bridge_names) > 1:
            ensemble_risks = [
                [
                    np.mean([row["verifier_logits"][name] for name in bridge_names])
                    for row in rows
                ]
                for rows in prefixes
            ]
            outcomes["bridge_ensemble"] = np.asarray(
                [
                    min(
                        rows,
                        key=lambda row: np.mean([row["verifier_logits"][name] for name in bridge_names]),
                    )["correct"]
                    for rows in prefixes
                ],
                dtype=float,
            )
            for mode in ("rank_weighted", "plurality_tiebreak"):
                outcomes[f"bridge_ensemble_{mode}_vote"] = np.asarray(
                    [
                        verifier_consensus_correct(rows, risks, mode)
                        for rows, risks in zip(prefixes, ensemble_risks)
                    ],
                    dtype=float,
                )

        for method_index, (method, values) in enumerate(outcomes.items()):
            stats = bootstrap(
                values,
                random_choice,
                majority,
                args.seed + 100 * budget_index + method_index,
                args.bootstrap_repetitions,
            )
            wins, losses, p_value = exact_paired_test(values, majority)
            result_rows.append(
                {
                    "generation_seed": generation_seed,
                    "budget_n": budget,
                    "method": method,
                    "num_questions": len(groups),
                    "accuracy": float(values.mean()),
                    "accuracy_ci95_low": stats["accuracy_ci95"][0],
                    "accuracy_ci95_high": stats["accuracy_ci95"][1],
                    "delta_random": stats["delta_random"],
                    "delta_random_ci95_low": stats["delta_random_ci95"][0],
                    "delta_random_ci95_high": stats["delta_random_ci95"][1],
                    "delta_majority": stats["delta_majority"],
                    "delta_majority_ci95_low": stats["delta_majority_ci95"][0],
                    "delta_majority_ci95_high": stats["delta_majority_ci95"][1],
                    "wins_vs_majority": wins,
                    "losses_vs_majority": losses,
                    "mcnemar_exact_p": p_value,
                }
            )

    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    write_csv(csv_path, result_rows)
    payload = {
        "records": str(args.records.resolve()),
        "generation_seed": generation_seed,
        "num_questions": len(groups),
        "num_samples": len(records),
        "budgets": budgets,
        "head_names": head_names,
        "bridge_ensemble_members": bridge_names,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "rows": result_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
