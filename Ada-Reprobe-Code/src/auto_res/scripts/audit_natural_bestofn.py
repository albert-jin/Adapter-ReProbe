#!/usr/bin/env python3
"""Audit parsing, requested answer format, and generation-length truncation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer


CAP_FIELDS = (
    "run_name",
    "dataset",
    "generation_seed",
    "question_id",
    "sample_id",
    "token_count",
    "correct",
    "reference_answer",
    "predicted_answer",
    "question",
    "response_json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", action="append", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    per_seed = []
    capped_rows = []
    for path in args.records:
        summary_path = path.parent / "summary.json"
        run_summary = json.loads(summary_path.read_text())
        generation_seed = int(run_summary["seed"])
        dataset_name = str(run_summary.get("dataset", "gsm8k"))
        if dataset_name.startswith("gsm8k"):
            dataset_name = "gsm8k"
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        explicit = 0
        parsed = 0
        capped = 0
        capped_correct = 0
        capped_explicit = 0
        capped_correct_without_explicit = 0
        for row in records:
            has_explicit = "final answer:" in row["response"].lower()
            explicit += has_explicit
            parsed += row["predicted_answer"] is not None
            token_count = len(
                tokenizer(row["response"], add_special_tokens=False)["input_ids"]
            )
            if token_count >= args.max_new_tokens:
                capped += 1
                capped_correct += bool(row["correct"])
                capped_explicit += has_explicit
                capped_correct_without_explicit += bool(row["correct"]) and not has_explicit
                capped_rows.append(
                    {
                        "run_name": path.parent.name,
                        "dataset": dataset_name,
                        "generation_seed": generation_seed,
                        "question_id": row["question_id"],
                        "sample_id": row["sample_id"],
                        "token_count": token_count,
                        "correct": row["correct"],
                        "reference_answer": row["reference_answer"],
                        "predicted_answer": row["predicted_answer"],
                        "question": row["question"],
                        # JSON escaping keeps one CSV row per sample while
                        # preserving newlines and trailing spaces exactly.
                        "response_json": json.dumps(
                            row["response"], ensure_ascii=False
                        ),
                    }
                )
        per_seed.append(
            {
                "run_name": path.parent.name,
                "records": str(path.resolve()),
                "generation_seed": generation_seed,
                "dataset": dataset_name,
                "dataset_filter_rule": run_summary.get("dataset_filter_rule"),
                "dataset_total_rows": run_summary.get("dataset_total_rows"),
                "dataset_eligible_rows": run_summary.get("dataset_eligible_rows"),
                "dataset_excluded_rows": run_summary.get("dataset_excluded_rows"),
                "dataset_eligible_sha256": run_summary.get(
                    "dataset_eligible_sha256"
                ),
                "dataset_sample_sha256": run_summary.get("dataset_sample_sha256"),
                "num_samples": len(records),
                "num_parsed": parsed,
                "num_explicit_final_answer": explicit,
                "num_at_token_cap": capped,
                "num_correct_at_token_cap": capped_correct,
                "num_at_token_cap_with_explicit_final_answer": capped_explicit,
                "num_correct_at_token_cap_without_explicit_final_answer": (
                    capped_correct_without_explicit
                ),
            }
        )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    capped_path = args.output_prefix.parent / (args.output_prefix.name + ".capped.csv")
    with capped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAP_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(capped_rows)
    aggregate = {
        "num_samples": sum(row["num_samples"] for row in per_seed),
        "num_parsed": sum(row["num_parsed"] for row in per_seed),
        "num_explicit_final_answer": sum(
            row["num_explicit_final_answer"] for row in per_seed
        ),
        "num_at_token_cap": sum(row["num_at_token_cap"] for row in per_seed),
        "num_correct_at_token_cap": sum(
            row["num_correct_at_token_cap"] for row in per_seed
        ),
        "num_at_token_cap_with_explicit_final_answer": sum(
            row["num_at_token_cap_with_explicit_final_answer"] for row in per_seed
        ),
        "num_correct_at_token_cap_without_explicit_final_answer": sum(
            row["num_correct_at_token_cap_without_explicit_final_answer"]
            for row in per_seed
        ),
    }
    payload = {
        "tokenizer": str(Path(args.tokenizer).resolve()),
        "max_new_tokens": args.max_new_tokens,
        "per_seed": per_seed,
        "aggregate": aggregate,
        "capped_samples_csv": str(capped_path.resolve()),
        "manual_review": (
            "Automated audit only. Manually review every capped-and-correct row "
            "before treating the associated selection aggregate as formal."
        ),
    }
    summary_path = args.output_prefix.parent / (args.output_prefix.name + ".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
