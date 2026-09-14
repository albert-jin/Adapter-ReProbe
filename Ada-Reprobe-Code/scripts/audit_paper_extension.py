#!/usr/bin/env python3
"""Audit completeness of the predeclared paper-extension experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


GENERATION_CELLS = (
    ("primary", "svamp"),
    ("primary", "multiarith"),
    ("primary", "asdiv"),
    ("base", "svamp"),
    ("altgsm", "svamp"),
)
SEEDS = (2040, 2041, 2042)
CALIBRATION_TARGETS = ("gsm", "finance")
CALIBRATION_SCOPES = ("calibration_bias", "calibration_temperature")
SUPPORT_SIZES = (10, 25, 50, 100)
CALIBRATION_SEEDS = (42, 43, 44)
EXPECTED_GENERATION_CONFIG = {
    "num_questions": 100,
    "num_samples": 8,
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.95,
    "top_k": 20,
    "seeding_protocol": "fixed_original_batch_v2",
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def latest_file(root, name):
    matches = sorted(root.glob(f"*/*/{name}"))
    return matches[-1] if matches else None


def audit_generation(runs_dir, results_dir):
    rows = []
    for checkpoint, dataset in GENERATION_CELLS:
        for seed in SEEDS:
            run_name = (
                f"natural_bestofn_{checkpoint}_{dataset}_n100x8_seed{seed}_cap512"
            )
            root = runs_dir / run_name
            summary_path = root / "summary.json"
            config_path = root / "run_config.json"
            records_path = root / "samples.jsonl"
            errors = []
            summary = {}
            config = {}
            records = []
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                errors.append(f"summary:{type(exc).__name__}")
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                errors.append(f"config:{type(exc).__name__}")
            try:
                records = [
                    json.loads(line)
                    for line in records_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            except (FileNotFoundError, json.JSONDecodeError) as exc:
                errors.append(f"records:{type(exc).__name__}")

            counts = Counter(int(row["question_id"]) for row in records)
            if len(records) != 800:
                errors.append(f"records={len(records)}")
            if len(counts) != 100:
                errors.append(f"questions={len(counts)}")
            if counts and set(counts.values()) != {8}:
                errors.append("nonuniform_candidate_count")
            if summary and (
                summary.get("num_questions") != 100
                or summary.get("num_samples") != 800
            ):
                errors.append("summary_count_mismatch")
            for key, expected in EXPECTED_GENERATION_CONFIG.items():
                if config and config.get(key) != expected:
                    errors.append(f"config_{key}={config.get(key)!r}")
            if config and config.get("dataset") != dataset:
                errors.append(f"config_dataset={config.get('dataset')!r}")
            if dataset == "asdiv" and config:
                if config.get("dataset_eligible_rows") != 2097:
                    errors.append("asdiv_eligible_rows")
                if not str(config.get("dataset_filter_rule", "")).startswith(
                    "strict_scalar_numeric_v1"
                ):
                    errors.append("asdiv_filter_rule")
            rows.append(
                {
                    "checkpoint": checkpoint,
                    "dataset": dataset,
                    "seed": seed,
                    "run_name": run_name,
                    "records": len(records),
                    "questions": len(counts),
                    "complete": not errors,
                    "errors": ";".join(errors),
                    "random_accuracy": summary.get("sample_accuracy"),
                    "source_accuracy": summary.get("source_selected_accuracy"),
                }
            )

    analysis_files = []
    for checkpoint, dataset in GENERATION_CELLS:
        prefix = results_dir / f"natural_bestofn_{checkpoint}_{dataset}_cap512"
        for suffix in (
            "_replicates.pooled.json",
            "_audit.summary.json",
            "_ambiguity.summary.json",
        ):
            analysis_files.append(Path(str(prefix) + suffix))
    analysis_files.extend(
        [
            results_dir / "natural_bestofn_replicates.pooled.json",
            results_dir / "natural_bestofn_audit.summary.json",
            results_dir
            / "natural_bestofn_primary_asdiv_cap512_manual_capped_review.csv",
            results_dir
            / "natural_bestofn_base_svamp_cap512_manual_capped_review.csv",
            results_dir
            / "natural_bestofn_altgsm_svamp_cap512_manual_capped_review.csv",
        ]
    )
    missing_analyses = [str(path) for path in analysis_files if not path.exists()]
    return rows, missing_analyses


def audit_calibration(runs_dir):
    rows = []
    # The completed two-scalar curve and the new one-scalar matched-optimizer runs.
    scopes = (
        ("temperature_bias", "cal_unweighted"),
        ("bias_only", "calibration_bias"),
        ("temperature_only", "calibration_temperature"),
    )
    for target in CALIBRATION_TARGETS:
        for method, stem in scopes:
            for size in SUPPORT_SIZES:
                for seed in CALIBRATION_SEEDS:
                    root = runs_dir / f"{target}_{stem}_n{size}_seed{seed}"
                    metrics_path = latest_file(root, "eval_metrics.json")
                    config_path = latest_file(root, ".hydra/config.yaml")
                    errors = []
                    if metrics_path is None:
                        errors.append("missing_eval_metrics")
                    if config_path is None:
                        errors.append("missing_config")
                    rows.append(
                        {
                            "target": target,
                            "method": method,
                            "support_size": size,
                            "seed": seed,
                            "run_root": str(root),
                            "complete": not errors,
                            "errors": ";".join(errors),
                        }
                    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=Path("auto_res/runs"))
    parser.add_argument("--results-dir", type=Path, default=Path("auto_res/results"))
    parser.add_argument("--figures-dir", type=Path, default=Path("auto_res/figures"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("auto_res/results/paper_extension_completeness"),
    )
    args = parser.parse_args()

    generation, missing_analyses = audit_generation(args.runs_dir, args.results_dir)
    calibration = audit_calibration(args.runs_dir)
    report_files = [
        args.results_dir / "benchmark_matrix.summary.json",
        args.results_dir / "native_calibration_components.summary.json",
    ]
    for stem in (
        "cross_dataset_natural_utility",
        "svamp_cross_checkpoint_utility",
        "calibration_component_ablation",
    ):
        report_files.extend(
            [args.figures_dir / f"{stem}.png", args.figures_dir / f"{stem}.pdf"]
        )
    missing_reports = [str(path) for path in report_files if not path.exists()]
    generation_path = Path(str(args.output_prefix) + ".generation.csv")
    calibration_path = Path(str(args.output_prefix) + ".calibration.csv")
    write_csv(generation_path, generation)
    write_csv(calibration_path, calibration)
    payload = {
        "generation_runs_expected": len(generation),
        "generation_runs_complete": sum(bool(row["complete"]) for row in generation),
        "calibration_runs_expected": len(calibration),
        "calibration_runs_complete": sum(bool(row["complete"]) for row in calibration),
        "missing_analysis_files": missing_analyses,
        "missing_report_files": missing_reports,
        "generation_csv": str(generation_path.resolve()),
        "calibration_csv": str(calibration_path.resolve()),
    }
    payload["complete"] = (
        payload["generation_runs_complete"] == payload["generation_runs_expected"]
        and payload["calibration_runs_complete"] == payload["calibration_runs_expected"]
        and not missing_analyses
        and not missing_reports
    )
    summary_path = Path(str(args.output_prefix) + ".summary.json")
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
