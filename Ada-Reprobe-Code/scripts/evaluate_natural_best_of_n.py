#!/usr/bin/env python3
"""Generate numeric-reasoning candidates and evaluate verifier-guided Best-of-N."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from peft import PeftModel
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from luh import AutoUncertaintyHead


NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[\d,]+(?:\.\d+)?)?")

ASDIV_SCALAR_FILTER_RULE = (
    "strict_scalar_numeric_v1: keep answers with exactly one NUMBER_RE match; "
    "exclude ':' and ';' structured answers and '-(...)' parenthesized negatives"
)

DATASET_SPECS = {
    "gsm8k": {
        "repo": "gsm8k",
        "config": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
    },
    "svamp": {
        "repo": "ChilleD/SVAMP",
        "config": None,
        "split": "test",
        "revision": "5e0bf1e5e7c0e9c4bc39180d224f41f3f801b7ef",
    },
    "multiarith": {
        "repo": "ChilleD/MultiArith",
        "config": None,
        "split": "test",
        "revision": "144d44c3fb87c0b9097ac9593c789e716a282e3e",
    },
    "asdiv": {
        "repo": "EleutherAI/asdiv",
        "config": None,
        "split": "validation",
        "revision": "8f95807222d87b4c688c3c22a6ba2801e1fa03e2",
    },
}

DATASET_CONFIG_FIELDS = {
    "dataset",
    "dataset_repo",
    "dataset_config",
    "dataset_split",
    "dataset_revision",
    "dataset_filter_rule",
    "dataset_total_rows",
    "dataset_eligible_rows",
    "dataset_excluded_rows",
    "dataset_eligible_sha256",
    "dataset_sample_sha256",
}


def parse_number(text: str):
    """Extract the final numeric answer from common GSM8K/Qwen formats."""
    candidates = []
    if "####" in text:
        candidates = NUMBER_RE.findall(text.rsplit("####", 1)[-1])
        if candidates:
            return parse_numeric_candidate(candidates[-1])
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    for value in boxed:
        candidates.extend(NUMBER_RE.findall(value))
    if candidates:
        return parse_numeric_candidate(candidates[-1])
    final_phrases = re.findall(
        r"(?:final answer|answer is|therefore)[^\d+\-]*([-+]?\d[\d,]*(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if final_phrases:
        return parse_numeric_candidate(final_phrases[-1])
    candidates = NUMBER_RE.findall(text)
    if not candidates:
        return None
    return parse_numeric_candidate(candidates[-1])


def parse_numeric_candidate(candidate: str):
    value = candidate.replace(",", "")
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            return float(numerator) / float(denominator)
        return float(value)
    except (ValueError, ZeroDivisionError):
        return None


def is_strict_scalar_numeric_answer(answer: str):
    """Whether an ASDiv reference is one scalar supported by our evaluator."""
    answer = str(answer).strip()
    compact = re.sub(r"\s+", "", answer)
    return (
        len(NUMBER_RE.findall(answer)) == 1
        and ":" not in answer
        and ";" not in answer
        and "-(" not in compact
    )


def benchmark_rows_sha256(rows):
    """Fingerprint normalized source rows in their deterministic order."""
    payload = json.dumps(
        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_benchmark_row(dataset_name, row, source_index):
    """Map pinned benchmark schemas to the evaluator's numeric QA contract."""
    if dataset_name == "gsm8k":
        question = row["question"]
        answer = row["answer"]
    elif dataset_name == "svamp":
        question = f"{row['Body'].strip()} {row['Question'].strip()}"
        answer = row["Answer"]
    elif dataset_name == "multiarith":
        question = row["question"]
        answer = row["final_ans"]
    elif dataset_name == "asdiv":
        question = f"{row['body'].strip()} {row['question'].strip()}"
        answer = row["answer"]
    else:  # Kept explicit so a new dataset cannot silently use the wrong fields.
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    reference = parse_number(str(answer))
    if reference is None:
        raise ValueError(
            f"Dataset {dataset_name} row {source_index} has no numeric reference: {answer!r}"
        )
    return {
        "question_id": int(source_index),
        "question": str(question).strip(),
        "answer": str(answer),
        "reference_answer": reference,
    }


def load_benchmark_rows(dataset_name, num_questions, seed, return_metadata=False):
    """Load, schema-filter, and deterministically sample a pinned benchmark."""
    spec = DATASET_SPECS[dataset_name]
    dataset = load_dataset(
        spec["repo"],
        spec["config"],
        split=spec["split"],
        revision=spec["revision"],
    )
    dataset = dataset.add_column("_source_index", list(range(len(dataset))))
    total_rows = len(dataset)
    if dataset_name == "asdiv":
        eligible_indices = [
            index
            for index, answer in enumerate(dataset["answer"])
            if is_strict_scalar_numeric_answer(answer)
        ]
        dataset = dataset.select(eligible_indices)
        filter_rule = ASDIV_SCALAR_FILTER_RULE
    else:
        filter_rule = "none_v1: all rows in the pinned split"
    eligible_rows = [
        normalize_benchmark_row(dataset_name, row, row["_source_index"])
        for row in dataset
    ]
    if num_questions > len(eligible_rows):
        raise ValueError(
            f"Requested {num_questions} questions from {dataset_name}, which has "
            f"{len(eligible_rows)} evaluator-compatible rows"
        )
    dataset = dataset.shuffle(seed=seed).select(range(num_questions))
    sampled_rows = [
        normalize_benchmark_row(dataset_name, row, row["_source_index"])
        for row in dataset
    ]
    metadata = {
        "dataset_filter_rule": filter_rule,
        "dataset_total_rows": total_rows,
        "dataset_eligible_rows": len(eligible_rows),
        "dataset_excluded_rows": total_rows - len(eligible_rows),
        "dataset_eligible_sha256": benchmark_rows_sha256(eligible_rows),
        "dataset_sample_sha256": benchmark_rows_sha256(sampled_rows),
    }
    if return_metadata:
        return sampled_rows, metadata
    return sampled_rows


def run_configs_compatible(previous, current):
    """Accept exact configs plus legacy GSM8K configs predating dataset metadata."""
    if previous == current:
        return True
    if current.get("dataset") != "gsm8k":
        return False
    legacy_current = {
        key: value for key, value in current.items() if key not in DATASET_CONFIG_FIELDS
    }
    return previous == legacy_current


def numerically_equal(left, right):
    if left is None or right is None:
        return False
    return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-6)


def safe_binary_ranking(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return (
        float(average_precision_score(labels, scores)),
        float(roc_auc_score(labels, scores)),
    )


def bootstrap_mean_and_difference(selected, random_choice, seed=2026, repetitions=2000):
    selected = np.asarray(selected, dtype=float)
    random_choice = np.asarray(random_choice, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(selected), size=(repetitions, len(selected)))
    selected_bootstrap = selected[indices].mean(axis=1)
    difference_bootstrap = (selected - random_choice)[indices].mean(axis=1)
    return {
        "selected_ci95": np.quantile(selected_bootstrap, [0.025, 0.975]).tolist(),
        "selected_minus_random": float((selected - random_choice).mean()),
        "selected_minus_random_ci95": np.quantile(
            difference_bootstrap, [0.025, 0.975]
        ).tolist(),
    }


def generation_mask(generated_ids, stop_ids):
    mask = torch.ones_like(generated_ids, dtype=torch.long)
    for row_index, row in enumerate(generated_ids.tolist()):
        for token_index, token in enumerate(row):
            if token in stop_ids:
                mask[row_index, token_index + 1 :] = 0
                break
    return mask


def generation_batch_seed(seed, batch_index):
    """Derive a stable independent RNG seed for one original dataset batch."""
    return (int(seed) * 1_000_003 + int(batch_index) * 97_409) % (2**63 - 1)


def generation_batches(dataset_rows, completed_ids, batch_size, resume_safe=True):
    """Build batches without changing later RNG units after a safe resume."""
    if not resume_safe:
        pending = [row for row in dataset_rows if row["question_id"] not in completed_ids]
        return [
            (batch_start // batch_size, pending[batch_start : batch_start + batch_size])
            for batch_start in range(0, len(pending), batch_size)
        ]
    batches = []
    for batch_start in range(0, len(dataset_rows), batch_size):
        batch = dataset_rows[batch_start : batch_start + batch_size]
        if all(row["question_id"] in completed_ids for row in batch):
            continue
        batches.append((batch_start // batch_size, batch))
    return batches


def score_sequences(model, heads, sequences, attention_mask, response_mask, chunk_size):
    backbone = (
        model.get_base_model().model
        if hasattr(model, "get_base_model")
        else model.model
    )
    scores = {name: [] for name in heads}
    for start in range(0, len(sequences), chunk_size):
        end = min(start + chunk_size, len(sequences))
        input_ids = sequences[start:end]
        batch_attention = attention_mask[start:end]
        batch_response = response_mask[start:end]
        claims = []
        for row in batch_response:
            claim = row[1:].to(torch.int64).unsqueeze(0)
            if not claim.any():
                raise ValueError("Generated an empty response claim")
            claims.append(claim)
        with torch.inference_mode():
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=batch_attention,
                output_hidden_states=True,
                return_dict=True,
            )
            head_inputs = {
                "input_ids": input_ids,
                "attention_mask": batch_attention,
                "claims": claims,
            }
            for name, head in heads.items():
                values = head(head_inputs, outputs).reshape(-1).float().cpu().numpy()
                scores[name].extend(float(value) for value in values)
        del outputs
    return scores


def summarize(records, head_names):
    by_question = defaultdict(list)
    for row in records:
        by_question[int(row["question_id"])].append(row)
    groups = [sorted(rows, key=lambda row: row["sample_id"]) for rows in by_question.values()]
    random_choice_by_question = np.asarray(
        [np.mean([row["correct"] for row in rows]) for rows in groups], dtype=float
    )
    summary = {
        "num_questions": len(groups),
        "num_samples": len(records),
        "samples_per_question": len(groups[0]) if groups else 0,
        "sample_accuracy": float(random_choice_by_question.mean()),
        "first_sample_accuracy": float(np.mean([rows[0]["correct"] for rows in groups])),
        "pass_at_n": float(np.mean([any(row["correct"] for row in rows) for rows in groups])),
    }

    majority_results = []
    for rows in groups:
        parsed = [row["predicted_answer"] for row in rows if row["predicted_answer"] is not None]
        if not parsed:
            majority_results.append(False)
            continue
        rounded = [round(float(value), 8) for value in parsed]
        majority = Counter(rounded).most_common(1)[0][0]
        majority_results.append(numerically_equal(majority, rows[0]["reference_answer"]))
    summary["majority_vote_accuracy"] = float(np.mean(majority_results))

    incorrect = np.asarray([not row["correct"] for row in records], dtype=np.int8)
    for name in head_names:
        raw_scores = np.asarray([row["verifier_logits"][name] for row in records], dtype=float)
        selected = [min(rows, key=lambda row: row["verifier_logits"][name]) for rows in groups]
        selected_correct = np.asarray([row["correct"] for row in selected], dtype=float)
        summary[f"{name}_selected_accuracy"] = float(selected_correct.mean())
        ap, auc = safe_binary_ranking(incorrect, expit(raw_scores))
        summary[f"{name}_incorrect_ap"] = ap
        summary[f"{name}_incorrect_roc_auc"] = auc
        summary[f"{name}_bootstrap"] = bootstrap_mean_and_difference(
            selected_correct, random_choice_by_question
        )

    bridge_names = [name for name in head_names if name.startswith("bridge")]
    if len(bridge_names) > 1:
        selected = []
        ensemble_scores = []
        for rows in groups:
            selected.append(
                min(
                    rows,
                    key=lambda row: np.mean(
                        [row["verifier_logits"][name] for name in bridge_names]
                    ),
                )
            )
        for row in records:
            ensemble_scores.append(
                np.mean([row["verifier_logits"][name] for name in bridge_names])
            )
        selected_correct = np.asarray([row["correct"] for row in selected], dtype=float)
        summary["bridge_ensemble_selected_accuracy"] = float(selected_correct.mean())
        ap, auc = safe_binary_ranking(incorrect, expit(ensemble_scores))
        summary["bridge_ensemble_incorrect_ap"] = ap
        summary["bridge_ensemble_incorrect_roc_auc"] = auc
        summary["bridge_ensemble_bootstrap"] = bootstrap_mean_and_difference(
            selected_correct, random_choice_by_question
        )
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument(
        "--adapter",
        default=None,
        help="Optional PEFT adapter; omit to evaluate the frozen base model.",
    )
    parser.add_argument(
        "--head", action="append", required=True, help="Verifier checkpoint as NAME=PATH"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_SPECS),
        default="gsm8k",
        help="Revision-pinned numeric reasoning benchmark.",
    )
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--legacy-global-seeding",
        action="store_true",
        help=(
            "Reproduce pre-2026-09-05 runs whose global RNG depends on earlier "
            "batch lengths. The default fixed-batch protocol is resume-safe."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "samples.jsonl"
    existing = []
    completed_ids = set()
    if args.resume and records_path.exists():
        existing = [json.loads(line) for line in records_path.read_text().splitlines() if line]
        counts = Counter(row["question_id"] for row in existing)
        completed_ids = {idx for idx, count in counts.items() if count == args.num_samples}
        existing = [row for row in existing if row["question_id"] in completed_ids]

    head_paths = {}
    for item in args.head:
        if "=" not in item:
            raise ValueError(f"head must be NAME=PATH, got {item}")
        name, path = item.split("=", 1)
        head_paths[name] = path
    dataset_spec = DATASET_SPECS[args.dataset]
    dataset_rows, dataset_metadata = load_benchmark_rows(
        args.dataset, args.num_questions, args.seed, return_metadata=True
    )
    run_config = {
        "base_model": str(Path(args.base_model).resolve()),
        "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
        "heads": {name: str(Path(path).resolve()) for name, path in head_paths.items()},
        "num_questions": args.num_questions,
        "num_samples": args.num_samples,
        "question_batch_size": args.question_batch_size,
        "score_batch_size": args.score_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": args.seed,
        "enable_thinking": args.enable_thinking,
        "dataset": args.dataset,
        "dataset_repo": dataset_spec["repo"],
        "dataset_config": dataset_spec["config"],
        "dataset_split": dataset_spec["split"],
        "dataset_revision": dataset_spec["revision"],
        **dataset_metadata,
        "seeding_protocol": (
            "legacy_global_v1"
            if args.legacy_global_seeding
            else "fixed_original_batch_v2"
        ),
    }
    run_config_path = args.output_dir / "run_config.json"
    if existing and args.resume:
        if not run_config_path.exists() and not args.legacy_global_seeding:
            raise ValueError(
                "Existing records predate resume-safe run_config metadata. "
                "Use --legacy-global-seeding to resume them or choose a new output directory."
            )
        if run_config_path.exists():
            previous_config = json.loads(run_config_path.read_text())
            if not run_configs_compatible(previous_config, run_config):
                raise ValueError(
                    "Resume configuration differs from run_config.json; refusing to mix protocols"
                )
    run_config_path.write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        low_cpu_mem_usage=True,
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=False)
    model.eval()

    heads = {
        name: AutoUncertaintyHead.from_pretrained(path, model).to(
            device=model.device, dtype=torch.bfloat16
        ).eval()
        for name, path in head_paths.items()
    }

    batches = generation_batches(
        dataset_rows,
        completed_ids,
        args.question_batch_size,
        resume_safe=not args.legacy_global_seeding,
    )

    eos_ids = model.generation_config.eos_token_id
    stop_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    stop_ids.update(tokenizer.all_special_ids)
    all_records = list(existing)

    for batch_index, batch in batches:
        batch_ids = {int(row["question_id"]) for row in batch}
        if not args.legacy_global_seeding:
            # A write interrupted partway through a batch can leave one or more
            # complete questions. Regenerate the entire original batch with its
            # independent seed so the sampling schedule is resume-invariant.
            all_records = [
                row for row in all_records if int(row["question_id"]) not in batch_ids
            ]
            batch_rng_seed = generation_batch_seed(args.seed, batch_index)
            random.seed(batch_rng_seed)
            np.random.seed(batch_rng_seed % (2**32 - 1))
            torch.manual_seed(batch_rng_seed)
            torch.cuda.manual_seed_all(batch_rng_seed)
        prompts = []
        for row in batch:
            messages = [{
                "role": "user",
                "content": (
                    row["question"]
                    + "\nSolve the problem carefully. End with a line in the exact form: "
                    + "Final answer: <number>"
                ),
            }]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking,
                )
            )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                do_sample=True,
                num_return_sequences=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                use_cache=True,
            )
        prompt_width = inputs.input_ids.shape[1]
        generated_ids = sequences[:, prompt_width:]
        generated_attention = generation_mask(generated_ids, stop_ids)
        prompt_attention = inputs.attention_mask.repeat_interleave(args.num_samples, dim=0)
        full_attention = torch.cat([prompt_attention, generated_attention], dim=1)
        response_mask = torch.cat(
            [torch.zeros_like(prompt_attention), generated_attention], dim=1
        )
        for special_id in tokenizer.all_special_ids:
            response_mask[sequences == special_id] = 0

        verifier_scores = score_sequences(
            model,
            heads,
            sequences,
            full_attention,
            response_mask,
            args.score_batch_size,
        )
        for local_question, row in enumerate(batch):
            reference = row["reference_answer"]
            for sample_id in range(args.num_samples):
                flat_index = local_question * args.num_samples + sample_id
                valid_ids = generated_ids[flat_index][generated_attention[flat_index].bool()]
                response = tokenizer.decode(valid_ids, skip_special_tokens=True)
                predicted = parse_number(response)
                record = {
                    "question_id": int(row["question_id"]),
                    "dataset": args.dataset,
                    "sample_id": sample_id,
                    "question": row["question"],
                    "reference_answer": reference,
                    "predicted_answer": predicted,
                    "correct": numerically_equal(predicted, reference),
                    "response": response,
                    "verifier_logits": {
                        name: verifier_scores[name][flat_index] for name in heads
                    },
                }
                all_records.append(record)
        with records_path.open("w", encoding="utf-8") as handle:
            for record in all_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        progress = summarize(all_records, list(heads))
        (args.output_dir / "summary.partial.json").write_text(
            json.dumps(progress, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({"completed_questions": len({r['question_id'] for r in all_records}), **progress}))

    summary = summarize(all_records, list(heads))
    summary.update(
        {
            "seed": args.seed,
            "base_model": str(Path(args.base_model).resolve()),
            "adapter": str(Path(args.adapter).resolve()) if args.adapter else None,
            "heads": {name: str(Path(path).resolve()) for name, path in head_paths.items()},
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "enable_thinking": args.enable_thinking,
            "num_questions_requested": args.num_questions,
            "num_samples_requested": args.num_samples,
            "question_batch_size": args.question_batch_size,
            "score_batch_size": args.score_batch_size,
            "dataset": args.dataset,
            "dataset_repo": dataset_spec["repo"],
            "dataset_config": dataset_spec["config"],
            "dataset_split": dataset_spec["split"],
            "dataset_revision": dataset_spec["revision"],
            **dataset_metadata,
            "question_ids_in_shuffle_order": [
                int(row["question_id"]) for row in dataset_rows
            ],
            "seeding_protocol": (
                "legacy_global_v1"
                if args.legacy_global_seeding
                else "fixed_original_batch_v2"
            ),
            "resume_safe_seeding": not args.legacy_global_seeding,
        }
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
