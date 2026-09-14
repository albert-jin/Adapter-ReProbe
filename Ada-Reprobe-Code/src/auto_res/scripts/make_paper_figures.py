#!/usr/bin/env python3
"""Render deterministic static figures from the collected experiment tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "base": "#4C78A8",
    "gsm": "#F58518",
    "finance": "#54A24B",
    "alternate": "#72B7B2",
    "negative": "#E45756",
}


def save(fig, output_dir, name):
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def value(native, run_name, metric):
    row = native[native["run_name"] == run_name]
    if row.empty:
        raise KeyError(f"Missing run {run_name}")
    return float(row.iloc[-1][metric])


def stability_value(stability, target, metric, statistic="mean"):
    row = stability[(stability["target"] == target) & (stability["metric"] == metric)]
    if row.empty:
        raise KeyError(f"Missing stability result {target}/{metric}")
    return float(row.iloc[0][statistic])


def core_ap_figure(native, stability, output_dir):
    labels = [
        "Official\nsource",
        "Sparse L21\nsource (3 seeds)",
        "GSM frozen\n(3 seeds)",
        "GSM\nfull retrain",
        "Finance frozen\n(3 seeds)",
        "Finance\nfull retrain",
    ]
    scores = [
        value(native, "official_uhead_fixed_alignment", "eval_average_precision"),
        stability_value(stability, "source", "eval_average_precision"),
        stability_value(stability, "gsm", "eval_average_precision"),
        value(native, "gsm_full_target_l21_seed42", "eval_average_precision"),
        stability_value(stability, "finance", "eval_average_precision"),
        value(native, "finance_full_target_l21_seed42", "eval_average_precision"),
    ]
    errors = [
        0.0,
        stability_value(stability, "source", "eval_average_precision", "std"),
        stability_value(stability, "gsm", "eval_average_precision", "std"),
        0.0,
        stability_value(stability, "finance", "eval_average_precision", "std"),
        0.0,
    ]
    colors = [
        COLORS["base"], COLORS["base"], COLORS["gsm"], COLORS["gsm"],
        COLORS["finance"], COLORS["finance"],
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    bars = ax.bar(
        np.arange(len(scores)), scores, yerr=errors, capsize=3,
        color=colors, edgecolor="white",
    )
    ax.set_ylabel("Average precision (AP)")
    ax.set_ylim(0.0, max(scores) + 0.08)
    ax.set_xticks(np.arange(len(scores)), labels)
    ax.axhline(0.151899, color="#777777", linestyle=":", linewidth=1.2, label="Positive rate")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.009, f"{score:.3f}", ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save(fig, output_dir, "core_average_precision")


def calibration_figure(calibration, native, output_dir):
    data = calibration[calibration["pos_weight"] == 1].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.7), sharex=True)
    targets = [
        ("gsm8k_grpo_r32", "GSM8K-GRPO", COLORS["gsm"], "l21_zero_gsm8k_grpo"),
        ("finance_r16", "Finance-QA", COLORS["finance"], "l21_zero_finance"),
    ]
    for target, label, color, zero_run in targets:
        frame = data[data["target"] == target].sort_values("train_examples")
        if frame.empty:
            continue
        for ax, metric, ylabel in [
            (axes[0], "eval_brier", "Brier score (lower is better)"),
            (axes[1], "eval_ece_10", "ECE-10 (lower is better)"),
        ]:
            ax.errorbar(
                frame["train_examples"],
                frame[f"{metric}_mean"],
                yerr=frame[f"{metric}_std"],
                marker="o",
                linewidth=2,
                capsize=3,
                label=label,
                color=color,
            )
            zero = value(native, zero_run, metric)
            ax.axhline(zero, color=color, linestyle=":", alpha=0.65)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Calibration trajectories")
            ax.set_xticks(sorted(data["train_examples"].unique()))
            ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    save(fig, output_dir, "calibration_label_efficiency")


def layer_transfer_figure(layer_path, output_dir):
    data = pd.read_csv(layer_path)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.plot(data["layer"], data["source_val_ap"], label="Source validation", color=COLORS["base"])
    ax.plot(data["layer"], data["gsm_test_ap"], label="GSM target test", color=COLORS["gsm"])
    ax.plot(data["layer"], data["finance_test_ap"], label="Finance target test", color=COLORS["finance"])
    ax.set_xlabel("Hidden-state output index")
    ax.set_ylabel("Average precision (AP)")
    ax.set_xticks(range(0, int(data["layer"].max()) + 1, 4))
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="lower center")
    save(fig, output_dir, "layer_transfer_profile")


def bridge_figure(native, output_dir):
    frame = native[native["run_name"].str.startswith("gsm_paired_bridge_r8_n100_s")].copy()
    if frame.empty:
        return
    frame["support_seed"] = frame["run_name"].str.extract(r"_s(\d+)$").astype(int)
    frame = frame.sort_values("support_seed")
    zero = value(native, "l21_zero_gsm8k_grpo", "eval_average_precision")
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(
        frame["support_seed"].astype(str),
        frame["eval_average_precision"],
        color=COLORS["negative"],
    )
    ax.axhline(zero, color="#222222", linestyle="--", label=f"Zero-shot ({zero:.3f})")
    ax.set_ylim(0.34, 0.41)
    ax.set_xlabel("Unlabeled support seed")
    ax.set_ylabel("Average precision (AP)")
    for bar, score in zip(bars, frame["eval_average_precision"]):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.002, f"{score:.3f}", ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save(fig, output_dir, "paired_bridge_seed_instability")


def selective_figure(curve_path, output_dir):
    if not curve_path.exists():
        return
    data = pd.read_csv(curve_path)
    data = data[data["level"] == "trajectory"]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for name, frame in data.groupby("method"):
        ax.plot(frame["coverage"], frame["clean_accuracy"], marker="o", label=name)
    ax.set_xlabel("Retained trajectory coverage")
    ax.set_ylabel("Clean-trajectory accuracy")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    save(fig, output_dir, "selective_trajectory_pruning")


def bestofn_figure(curve_path, output_dir):
    if not curve_path.exists():
        return
    data = pd.read_csv(curve_path)
    aggregate = "accuracy_mean" in data.columns
    accuracy_column = "accuracy_mean" if aggregate else "accuracy"
    methods = [
        ("random_choice_expected", "Random candidate", "#9D9D9D", "--"),
        ("majority_vote", "Majority vote", "#B279A2", "-"),
        ("source", "Direct verifier selection", COLORS["base"], "--"),
        ("source_rank_weighted_vote", "Verifier-weighted consensus", COLORS["base"], "-"),
        ("bridge_ensemble_rank_weighted_vote", "Bridge-weighted consensus", COLORS["negative"], "-"),
        ("pass_at_n_oracle", "Pass@N oracle", "#222222", ":"),
    ]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for method, label, color, linestyle in methods:
        frame = data[data["method"] == method].sort_values("budget_n")
        if frame.empty:
            continue
        if aggregate:
            ax.errorbar(
                frame["budget_n"], frame[accuracy_column],
                yerr=frame["accuracy_std"], marker="o", linewidth=2,
                capsize=3, linestyle=linestyle, color=color, label=label,
            )
        else:
            ax.plot(
                frame["budget_n"], frame[accuracy_column], marker="o", linewidth=2,
                linestyle=linestyle, color=color, label=label,
            )
    ax.set_xlabel("Candidates per question (N)")
    ax.set_ylabel("GSM8K answer accuracy")
    ax.set_xticks(sorted(data["budget_n"].unique()))
    ax.set_ylim(0.45, 0.95)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, output_dir, "natural_bestofn_budget_curve")


def pooled_accuracy(path, method):
    data = pd.read_csv(path)
    row = data[data["method"] == method]
    if row.empty:
        raise KeyError(f"Missing {method} in {path}")
    return float(row.iloc[0]["accuracy"])


def pooled_accuracy_interval(path, method):
    data = pd.read_csv(path)
    row = data[data["method"] == method]
    if row.empty:
        raise KeyError(f"Missing {method} in {path}")
    row = row.iloc[0]
    value = float(row["accuracy"])
    return value, (
        value - float(row["accuracy_cluster_ci95_low"]),
        float(row["accuracy_cluster_ci95_high"]) - value,
    )


def cross_checkpoint_utility_figure(results_dir, output_dir):
    paths = [
        results_dir / "natural_bestofn_replicates.pooled.csv",
        results_dir / "natural_bestofn_base_cap512_replicates.pooled.csv",
        results_dir / "natural_bestofn_altgsm_cap512_replicates.pooled.csv",
    ]
    if not all(path.exists() for path in paths):
        return
    labels = ["Primary GSM LoRA", "Base", "Independent GSM LoRA"]
    methods = [
        ("random_choice_expected", "Random expected", "#A0A0A0"),
        ("source", "Direct source verifier", COLORS["base"]),
        ("majority_vote", "Majority", "#B279A2"),
        ("source_plurality_tiebreak_vote", "Verifier tie-break", COLORS["alternate"]),
    ]
    values = np.zeros((len(paths), len(methods)))
    errors = np.zeros((2, len(paths), len(methods)))
    for checkpoint_index, path in enumerate(paths):
        for method_index, (method, _, _) in enumerate(methods):
            value, interval = pooled_accuracy_interval(path, method)
            values[checkpoint_index, method_index] = value
            errors[:, checkpoint_index, method_index] = interval
    x = np.arange(len(labels))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    for index, (_, method_label, color) in enumerate(methods):
        positions = x + (index - 1.5) * width
        bars = ax.bar(
            positions,
            values[:, index],
            width,
            yerr=errors[:, :, index],
            capsize=2,
            error_kw={"elinewidth": 0.8},
            label=method_label,
            color=color,
        )
        if index == 1:
            for dataset_index, (bar, direct, random_expected) in enumerate(
                zip(bars, values[:, 1], values[:, 0])
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    direct + 0.008,
                    f"+{100 * (direct - random_expected):.1f}pt",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x, labels)
    ax.set_ylabel("GSM8K answer accuracy (N=8)")
    ax.set_ylim(0.55, 0.93)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    save(fig, output_dir, "cross_checkpoint_natural_utility")


def generation_cap_sensitivity_figure(results_dir, output_dir):
    audit256_path = results_dir / "natural_bestofn_checkpoint_extension_audit.summary.json"
    audit512_path = (
        results_dir / "natural_bestofn_checkpoint_extension_cap512_audit.summary.json"
    )
    if not audit256_path.exists() or not audit512_path.exists():
        return
    audits = [json.loads(audit256_path.read_text()), json.loads(audit512_path.read_text())]
    run_names = [
        ["natural_bestofn_base_n100x8_seed2030", "natural_bestofn_altgsm_n100x8_seed2030"],
        [
            "natural_bestofn_base_n100x8_seed2030_cap512",
            "natural_bestofn_altgsm_n100x8_seed2030_cap512",
        ],
    ]
    curves = [
        [
            results_dir / "natural_bestofn_base_seed2030.csv",
            results_dir / "natural_bestofn_altgsm_seed2030.csv",
        ],
        [
            results_dir / "natural_bestofn_base_seed2030_cap512.csv",
            results_dir / "natural_bestofn_altgsm_seed2030_cap512.csv",
        ],
    ]
    capped = np.zeros((2, 2))
    accuracy = np.zeros((2, 2))
    for cap_index, audit in enumerate(audits):
        by_run = {row["run_name"]: row for row in audit["per_seed"]}
        for checkpoint_index, run_name in enumerate(run_names[cap_index]):
            row = by_run[run_name]
            capped[cap_index, checkpoint_index] = (
                100 * row["num_at_token_cap"] / row["num_samples"]
            )
            accuracy[cap_index, checkpoint_index] = pooled_accuracy(
                curves[cap_index][checkpoint_index], "random_choice_expected"
            )
    x = np.arange(2)
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.7))
    labels = ["Base", "Independent GSM LoRA"]
    for cap_index, (cap, color) in enumerate([(256, COLORS["negative"]), (512, COLORS["base"])]):
        positions = x + (cap_index - 0.5) * width
        axes[0].bar(positions, capped[cap_index], width, label=f"{cap}-token cap", color=color)
        axes[1].bar(positions, accuracy[cap_index], width, label=f"{cap}-token cap", color=color)
    axes[0].set_ylabel("Responses at token cap (%)")
    axes[1].set_ylabel("Random-candidate expected accuracy")
    axes[1].set_ylim(0.45, 0.85)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False)
    save(fig, output_dir, "generation_cap_sensitivity")


def cross_dataset_utility_figure(results_dir, output_dir):
    matrix_path = results_dir / "benchmark_matrix.csv"
    if not matrix_path.exists():
        return
    data = pd.read_csv(matrix_path)
    data = data[data["checkpoint"] == "primary"]
    datasets = ["gsm8k", "svamp", "multiarith", "asdiv"]
    if not set(datasets) <= set(data["dataset"]):
        return
    methods = [
        ("random_choice_expected", "Random candidate", "#A0A0A0"),
        ("source", "Direct source verifier", COLORS["base"]),
    ]
    values = np.zeros((len(datasets), len(methods)))
    errors = np.zeros((2, len(datasets), len(methods)))
    for dataset_index, dataset in enumerate(datasets):
        frame = data[data["dataset"] == dataset]
        for method_index, (method, _, _) in enumerate(methods):
            row = frame[frame["method"] == method].iloc[0]
            values[dataset_index, method_index] = float(row["accuracy"])
            errors[:, dataset_index, method_index] = [
                float(row["accuracy"]) - float(row["accuracy_ci95_low"]),
                float(row["accuracy_ci95_high"]) - float(row["accuracy"]),
            ]
    x = np.arange(len(datasets))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for method_index, (_, label, color) in enumerate(methods):
        positions = x + (method_index - 0.5) * width
        bars = ax.bar(
            positions, values[:, method_index], width,
            yerr=errors[:, :, method_index],
            capsize=3, color=color, label=label,
        )
        if method_index == 1:
            for dataset_index, (bar, direct, random_expected) in enumerate(
                zip(bars, values[:, 1], values[:, 0])
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    direct + errors[1, dataset_index, 1] + 0.015,
                    f"{100 * (direct - random_expected):+.1f}pt",
                    ha="center", va="bottom", fontsize=8,
                )
    ax.set_xticks(x, ["GSM8K", "SVAMP", "MultiArith", "ASDiv"])
    ax.set_ylabel("Answer accuracy (N=8)")
    ax.set_ylim(0.0, 1.10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2)
    save(fig, output_dir, "cross_dataset_natural_utility")


def svamp_checkpoint_figure(results_dir, output_dir):
    matrix_path = results_dir / "benchmark_matrix.csv"
    if not matrix_path.exists():
        return
    data = pd.read_csv(matrix_path)
    data = data[data["dataset"] == "svamp"]
    checkpoints = ["base", "primary", "altgsm"]
    if not set(checkpoints) <= set(data["checkpoint"]):
        return
    labels = ["Base", "Primary GSM LoRA", "Independent GSM LoRA"]
    methods = [
        ("random_choice_expected", "Random candidate", "#A0A0A0"),
        ("source", "Direct source verifier", COLORS["base"]),
        ("majority_vote", "Majority vote", "#B279A2"),
    ]
    values = np.zeros((len(checkpoints), len(methods)))
    errors = np.zeros((2, len(checkpoints), len(methods)))
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        frame = data[data["checkpoint"] == checkpoint]
        for method_index, (method, _, _) in enumerate(methods):
            row = frame[frame["method"] == method].iloc[0]
            values[checkpoint_index, method_index] = float(row["accuracy"])
            errors[:, checkpoint_index, method_index] = [
                float(row["accuracy"]) - float(row["accuracy_ci95_low"]),
                float(row["accuracy_ci95_high"]) - float(row["accuracy"]),
            ]
    x = np.arange(len(checkpoints))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for method_index, (_, label, color) in enumerate(methods):
        positions = x + (method_index - 1) * width
        bars = ax.bar(
            positions, values[:, method_index], width, color=color, label=label,
            yerr=errors[:, :, method_index], capsize=3,
        )
        if method_index == 1:
            for checkpoint_index, (bar, direct, random_expected) in enumerate(
                zip(bars, values[:, 1], values[:, 0])
            ):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    direct + errors[1, checkpoint_index, 1] + 0.012,
                    f"{100 * (direct - random_expected):+.1f}pt",
                    ha="center", va="bottom", fontsize=8,
                )
    ax.set_xticks(x, labels)
    ax.set_ylabel("SVAMP answer accuracy (N=8)")
    ax.set_ylim(0.0, 1.08)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=False, ncol=3, fontsize=9, loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
    )
    save(fig, output_dir, "svamp_cross_checkpoint_utility")


def calibration_component_figure(results_dir, native, output_dir):
    component_path = results_dir / "native_calibration_components.csv"
    if not component_path.exists():
        return
    data = pd.read_csv(component_path)
    targets = [
        ("gsm8k_grpo_r32", "GSM8K-GRPO", "l21_zero_gsm8k_grpo"),
        ("finance_r16", "Finance-QA", "l21_zero_finance"),
    ]
    styles = [
        ("temperature_only", "Temperature only", "--", "o"),
        ("bias_only", "Bias only", "-", "s"),
        ("temperature_bias", "Temperature + bias", ":", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True)
    for ax, (target, title, zero_run) in zip(axes, targets):
        frame = data[data["target"] == target]
        if frame.empty:
            plt.close(fig)
            return
        for method, label, linestyle, marker in styles:
            method_frame = frame[frame["method"] == method].sort_values("train_examples")
            ax.errorbar(
                method_frame["train_examples"], method_frame["eval_brier_mean"],
                yerr=method_frame["eval_brier_std"], marker=marker,
                linestyle=linestyle, linewidth=2, capsize=3, label=label,
            )
        zero = value(native, zero_run, "eval_brier")
        ax.axhline(
            zero, color="#777777", alpha=0.8, linewidth=1.4,
            label="Frozen zero-shot",
        )
        ax.set_title(title)
        ax.set_xlabel("Calibration trajectories")
        ax.set_ylabel("Brier score (lower is better)")
        ax.set_xticks([10, 25, 50, 100])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, output_dir, "calibration_component_ablation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("auto_res/results"))
    parser.add_argument("--output-dir", type=Path, default=Path("auto_res/figures"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.facecolor": "white"})

    native = pd.read_csv(args.results_dir / "native_experiment_summary.csv")
    calibration = pd.read_csv(args.results_dir / "native_calibration_curve.csv")
    stability = pd.read_csv(args.results_dir / "source_seed_stability.aggregate.csv")
    core_ap_figure(native, stability, args.output_dir)
    calibration_figure(calibration, native, args.output_dir)
    layer_transfer_figure(args.results_dir / "layer_transfer_drift.csv", args.output_dir)
    bridge_figure(native, args.output_dir)
    selective_figure(args.results_dir / "selective_pruning.curve.csv", args.output_dir)
    aggregate_curve = args.results_dir / "natural_bestofn_replicates.aggregate.csv"
    single_curve = args.results_dir / "natural_bestofn_budget_curve.csv"
    bestofn_figure(
        aggregate_curve if aggregate_curve.exists() else single_curve,
        args.output_dir,
    )
    cross_checkpoint_utility_figure(args.results_dir, args.output_dir)
    generation_cap_sensitivity_figure(args.results_dir, args.output_dir)
    cross_dataset_utility_figure(args.results_dir, args.output_dir)
    svamp_checkpoint_figure(args.results_dir, args.output_dir)
    calibration_component_figure(args.results_dir, native, args.output_dir)


if __name__ == "__main__":
    main()
