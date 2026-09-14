#!/usr/bin/env python3
"""Cache per-claim, per-layer Qwen hidden-state summaries.

The output stores both current-token and previous-token summaries.  The latter
matches the historical ReProbe ``[:, :-1]`` feature/mask convention, while the
former tests whether semantic claim states are more discriminative.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from luh.alignment import claim_token_positions, locate_reply_start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", default="rediska0123/train_gsm8k_Qwen3-1.7B")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter,
            is_trainable=False,
            cache_dir=args.cache_dir,
        )
    model.eval()

    dataset = load_dataset(args.dataset, cache_dir=args.cache_dir)[args.split]
    if args.max_examples > 0:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))

    current_features: list[np.ndarray] = []
    previous_features: list[np.ndarray] = []
    labels: list[int] = []
    example_ids: list[int] = []
    claim_ids: list[int] = []
    reply_starts: list[int] = []
    claim_lengths: list[int] = []
    metadata: list[dict] = []
    start_time = time.time()

    for batch_start in range(0, len(dataset), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(dataset))
        rows = [dataset[index] for index in range(batch_start, batch_end)]
        encoded = tokenizer.pad(
            {"input_ids": [row["input_ids"] for row in rows]},
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        # [B, layers including embedding, T, H], kept in fp16 on CPU.
        hidden = torch.stack(outputs.hidden_states, dim=1).to(torch.float16).cpu()
        max_length = encoded["input_ids"].shape[1]

        for local_index, row in enumerate(rows):
            example_index = batch_start + local_index
            padding = max_length - len(row["input_ids"])
            reply_start = locate_reply_start(row["input_ids"], row["reply"], tokenizer)
            for claim_index, (claim, label) in enumerate(zip(row["claims"], row["verified"])):
                positions = claim_token_positions(
                    row["input_ids"],
                    reply_start,
                    claim["aligned_token_ids"],
                    tokenizer.all_special_ids,
                )
                padded_positions = torch.tensor(
                    [padding + position for position in positions], dtype=torch.long
                )
                previous_positions = (padded_positions - 1).clamp_min(0)
                current = hidden[local_index, :, padded_positions, :].float().mean(dim=1)
                previous = hidden[local_index, :, previous_positions, :].float().mean(dim=1)
                current_features.append(current.to(torch.float16).numpy())
                previous_features.append(previous.to(torch.float16).numpy())
                labels.append(int(label))
                example_ids.append(example_index)
                claim_ids.append(claim_index)
                reply_starts.append(reply_start)
                claim_lengths.append(len(positions))
                metadata.append(
                    {
                        "example_id": example_index,
                        "claim_id": claim_index,
                        "label": int(label),
                        "question": row["question"],
                        "claim_text": claim.get("claim_text", claim.get("sentence", "")),
                    }
                )

        elapsed = time.time() - start_time
        print(
            f"cached examples {batch_end}/{len(dataset)} | claims {len(labels)} | "
            f"elapsed {elapsed:.1f}s",
            flush=True,
        )
        del outputs, hidden

    np.savez(
        output_path,
        current=np.stack(current_features),
        previous=np.stack(previous_features),
        labels=np.asarray(labels, dtype=np.int8),
        example_ids=np.asarray(example_ids, dtype=np.int32),
        claim_ids=np.asarray(claim_ids, dtype=np.int16),
        reply_starts=np.asarray(reply_starts, dtype=np.int16),
        claim_lengths=np.asarray(claim_lengths, dtype=np.int16),
    )
    metadata_path = output_path.with_suffix(".metadata.jsonl")
    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in metadata:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "dataset": args.dataset,
        "split": args.split,
        "num_examples": len(dataset),
        "num_claims": len(labels),
        "positive_rate": float(np.mean(labels)),
        "feature_shape": list(current_features[0].shape),
        "dtype_on_disk": "float16",
        "elapsed_seconds": time.time() - start_time,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
