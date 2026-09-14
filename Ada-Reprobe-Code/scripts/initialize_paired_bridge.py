#!/usr/bin/env python3
"""Initialize an end-to-end probe bridge from unlabeled paired replay features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from transformers import AutoConfig

from luh import AutoUncertaintyHead


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--source-features", required=True)
    parser.add_argument("--target-features", required=True)
    parser.add_argument("--output-head", required=True)
    parser.add_argument("--pooling", default="current", choices=("current", "previous"))
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--support-examples", type=int, default=100)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--support-seed", type=int, default=11)
    return parser.parse_args()


def main():
    args = parse_args()
    source = np.load(args.source_features)
    target = np.load(args.target_features)
    for key in ("labels", "example_ids", "claim_ids"):
        if not np.array_equal(source[key], target[key]):
            raise ValueError(f"Source/target caches differ in {key}")

    example_ids = source["example_ids"].astype(np.int64)
    split = Dataset.from_dict({"id": np.unique(example_ids).tolist()}).train_test_split(
        test_size=0.1, seed=args.split_seed
    )
    train_ids = np.asarray(split["train"]["id"], dtype=np.int64)
    if args.support_examples > len(train_ids):
        raise ValueError("support-examples exceeds the source training split")
    rng = np.random.default_rng(args.support_seed)
    support_ids = rng.choice(train_ids, size=args.support_examples, replace=False)
    train_mask = np.isin(example_ids, train_ids)
    support_mask = np.isin(example_ids, support_ids)

    source_raw = source[args.pooling][:, args.layer].astype(np.float32)
    target_raw = target[args.pooling][:, args.layer].astype(np.float32)
    scaler = StandardScaler().fit(source_raw[train_mask])
    source_z = scaler.transform(source_raw[support_mask])
    target_z = scaler.transform(target_raw[support_mask])
    if len(source_z) > args.max_pairs:
        selected = rng.choice(len(source_z), args.max_pairs, replace=False)
        source_z = source_z[selected]
        target_z = target_z[selected]

    delta = source_z - target_z
    target_mean = target_z.mean(axis=0)
    delta_mean = delta.mean(axis=0)
    centered_delta = delta - delta_mean
    effective_rank = min(args.rank, len(delta) - 1, delta.shape[1])
    _, singular_values, basis = randomized_svd(
        centered_delta,
        n_components=effective_rank,
        random_state=args.support_seed,
    )
    coefficients = centered_delta @ basis.T
    regressor = Ridge(
        alpha=args.ridge_alpha,
        fit_intercept=False,
        solver="lsqr",
    ).fit(target_z - target_mean, coefficients)

    # In standardized row-vector notation:
    #   z' = z + z @ A + b_z,
    # where A = down.T @ up.T. Convert the same affine map back to raw hidden
    # state coordinates so it can be inserted before the frozen neural probe.
    down_z = regressor.coef_.astype(np.float32)  # [rank, hidden]
    up_z = basis.T.astype(np.float32)  # [hidden, rank]
    standardized_bias = delta_mean - (target_mean @ down_z.T) @ up_z.T
    scale = scaler.scale_.astype(np.float32)
    mean = scaler.mean_.astype(np.float32)
    down_raw = down_z / scale[None, :]
    up_raw = scale[:, None] * up_z
    raw_bias = scale * standardized_bias - (mean @ down_raw.T) @ up_raw.T

    config = AutoConfig.from_pretrained(args.base_model)
    base_stub = SimpleNamespace(config=config)
    head = AutoUncertaintyHead.from_pretrained(args.source_head, base_stub)
    head.configure_adaptation(
        bridge_rank=effective_rank,
        bridge_alpha=float(effective_rank),
        bridge_dropout=0.0,
        calibration=False,
    )
    bridge = head.representation_bridge
    with torch.no_grad():
        bridge.down.weight.copy_(torch.from_numpy(down_raw))
        bridge.up.weight.copy_(torch.from_numpy(up_raw))
        bridge.bias.copy_(torch.from_numpy(raw_bias))

    # Numerical audit on the exact paired claim summaries used for fitting.
    raw_tensor = torch.from_numpy(target_raw[support_mask][: min(64, support_mask.sum())])
    with torch.no_grad():
        bridged_raw = bridge(raw_tensor).numpy()
    direct_z = scaler.transform(raw_tensor.numpy())
    direct_z = direct_z + (
        direct_z @ down_z.T @ up_z.T + standardized_bias
    )
    direct_raw = scaler.inverse_transform(direct_z)
    max_factorization_error = float(np.max(np.abs(bridged_raw - direct_raw)))
    if max_factorization_error > 2e-3:
        raise RuntimeError(
            f"Raw bridge factorization error too large: {max_factorization_error}"
        )

    output_head = Path(args.output_head)
    head.save(output_head)
    predicted = target_z + (
        (target_z - target_mean) @ down_z.T @ up_z.T + delta_mean
    )
    summary = {
        "method": "unlabeled paired reduced-rank affine bridge",
        "source_head": str(Path(args.source_head).resolve()),
        "source_features": str(Path(args.source_features).resolve()),
        "target_features": str(Path(args.target_features).resolve()),
        "pooling": args.pooling,
        "layer": args.layer,
        "support_examples": args.support_examples,
        "support_claims_used": int(len(source_z)),
        "support_seed": args.support_seed,
        "rank": effective_rank,
        "ridge_alpha": args.ridge_alpha,
        "pre_alignment_mse_standardized": float(np.mean((target_z - source_z) ** 2)),
        "post_alignment_mse_standardized": float(np.mean((predicted - source_z) ** 2)),
        "max_factorization_error": max_factorization_error,
        "singular_values": singular_values.tolist(),
        "output_head": str(output_head.resolve()),
    }
    (output_head / "paired_bridge_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
