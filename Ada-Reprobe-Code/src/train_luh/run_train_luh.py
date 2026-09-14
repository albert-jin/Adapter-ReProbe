import hydra
from pathlib import Path
import os
import json
import sys
import warnings
from scipy.special import expit
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    average_precision_score,
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
import numpy as np
import random
from itertools import chain
from collections import defaultdict

import torch
import torch.nn.init as init

from datasets import DatasetDict, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers import logging as transformers_logging

from causal_lm_with_uncertainty_layer import CausalLMWithUncertaintyLayer
from causal_lm_with_uncertainty_layer_claim import CausalLMWithUncertaintyLayerClaim
from luh.utils import load_any_dataset
from luh import AutoUncertaintyHead
from luh.alignment import locate_reply_start

import logging
from transformers import modeling_utils
import os
import pandas as pd
from datasets import Dataset

if not hasattr(modeling_utils, "ALL_PARALLEL_STYLES") or modeling_utils.ALL_PARALLEL_STYLES is None:
    modeling_utils.ALL_PARALLEL_STYLES = ["tp", "none", "colwise", 'rowwise']
transformers_logging.set_verbosity_info()
transformers_logging.enable_default_handler()

log = logging.getLogger()
hf_logger = logging.getLogger("transformers")


def resolve_torch_dtype(value):
    """Resolve Hydra strings such as ``torch.bfloat16`` into torch dtypes."""
    if value is None or isinstance(value, torch.dtype):
        return value
    name = str(value)
    if name.startswith("torch."):
        name = name[len("torch."):]
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {value}")
    return dtype


def load_tokenizer(config):
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.pretrained_model_name_or_path,
        model_max_length=2400,
        padding_side="left",
        cache_dir=getattr(config, 'hf_cache', None),
        token=getattr(config, 'hf_token', None),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(config):
    torch_dtype = resolve_torch_dtype(config.model.torch_dtype)

    log.info(f"Loading model {config.model.pretrained_model_name_or_path}...")

    # For DeepSpeed, load model on CPU initially to avoid OOM
    # DeepSpeed will handle the GPU distribution
    if hasattr(config, 'deepspeed_config') and config.deepspeed_config:
        device_map = "cpu"
        log.info("Using CPU device map for initial loading due to DeepSpeed configuration")
    else:
        device_map = config.model.device_map

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model.pretrained_model_name_or_path,
        torch_dtype=torch_dtype,  # Use the configured dtype
        trust_remote_code=True,
        device_map=device_map,  # Use CPU for DeepSpeed, config device_map otherwise
        low_cpu_mem_usage=True,  # Reduce CPU memory usage during loading
        cache_dir=getattr(config, 'hf_cache', None),
        token=getattr(config, 'hf_token', None),
    )

    adapter_path = getattr(config.model, "adapter_path", None)
    if adapter_path:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("model.adapter_path requires the peft package") from exc
        log.info("Loading PEFT adapter %s...", adapter_path)
        base_model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
            cache_dir=getattr(config, 'hf_cache', None),
            token=getattr(config, 'hf_token', None),
        )
        log.info("Loaded PEFT adapter %s.", adapter_path)

    # ReProbe freezes the target language model. Freeze before deciding whether
    # checkpointing is useful; the upstream order enabled checkpointing and then
    # froze every parameter, causing warnings and unnecessary recomputation.
    for param in base_model.parameters():
        param.requires_grad = False
    # Do not enable gradient checkpointing when the base model is frozen (no parameters require gradients).
    # Enabling it in that case triggers the warning:
    # "None of the inputs have requires_grad=True" and wastes memory.
    if any(p.requires_grad for p in base_model.parameters()):
        if hasattr(base_model, "gradient_checkpointing_enable"):
            base_model.gradient_checkpointing_enable()
        base_model.config.use_cache = False  # Disable KV cache when using checkpointing
    else:
        # Keep cache enabled for faster inference since we're not using gradient checkpointing
        base_model.config.use_cache = True

    if config.ue_layer.path:
        uq_head = AutoUncertaintyHead.from_pretrained(config.ue_layer.path, base_model)

    else:
        uq_head = AutoUncertaintyHead.from_config(config.ue_layer.head_cfg, base_model)

        def reinitialize_weights(module):
            if hasattr(module, "weight") and module.weight is not None and (
                    not (hasattr(module, "name") and "positional_encoding" in module.name)):
                if module.weight.ndim >= 2:
                    init.xavier_uniform_(module.weight)
                else:
                    init.uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                init.zeros_(module.bias)

        uq_head.apply(reinitialize_weights)

    # Transformers >=4.57 no longer falls back from SDPA when callers request
    # attention tensors.  Attention-based ReProbe checkpoints therefore need
    # the eager implementation selected explicitly on the already-built model.
    if uq_head.output_attentions:
        if hasattr(base_model, "set_attn_implementation"):
            base_model.set_attn_implementation("eager")
        else:
            base_model.config._attn_implementation = "eager"
        log.info("Enabled eager attention for attention-based uncertainty features.")

    adaptation_cfg = getattr(config.ue_layer, "adaptation", None)
    if adaptation_cfg and hasattr(uq_head, "configure_adaptation"):
        bridge_rank = int(getattr(adaptation_cfg, "bridge_rank", 0))
        use_calibration = bool(getattr(adaptation_cfg, "calibration", False))
        if bridge_rank > 0 or use_calibration:
            uq_head.configure_adaptation(
                bridge_rank=bridge_rank,
                bridge_alpha=float(getattr(adaptation_cfg, "bridge_alpha", 1.0)),
                bridge_dropout=float(getattr(adaptation_cfg, "bridge_dropout", 0.0)),
                calibration=use_calibration,
            )

    if uq_head.model_type == "claim":
        model = CausalLMWithUncertaintyLayerClaim(
            base_model,
            ue_head=uq_head,
            ue_pos_weight=config.ue_layer.pos_weight,
            output_attention=uq_head.output_attentions,
        )
    elif uq_head.model_type == "token":
        model = CausalLMWithUncertaintyLayer(
            base_model,
            ue_head=uq_head,
            ue_pos_weight=config.ue_layer.pos_weight,
            output_attention=uq_head.output_attentions,
        )

    # Ensure uncertainty head uses the same dtype as base model for mixed precision compatibility
    if hasattr(base_model, 'dtype'):
        model.ue_head = model.ue_head.to(dtype=base_model.dtype)
    elif torch_dtype:
        model.ue_head = model.ue_head.to(dtype=torch_dtype)

    # Temperature and bias are only two scalars and are particularly prone to
    # quantized updates in bf16. Keep them in fp32 while the larger modules
    # follow the frozen base model dtype.
    if getattr(model.ue_head, "calibration_log_temperature", None) is not None:
        model.ue_head.calibration_log_temperature.data = (
            model.ue_head.calibration_log_temperature.data.float()
        )
        model.ue_head.calibration_bias.data = model.ue_head.calibration_bias.data.float()

    trainable_scope = str(getattr(config.ue_layer, "trainable_scope", "all"))
    for name, param in model.named_parameters():
        if "ue_head" not in name:
            param.requires_grad = False
        elif trainable_scope == "all":
            param.requires_grad = True
        elif trainable_scope == "bridge":
            param.requires_grad = "representation_bridge" in name
        elif trainable_scope == "calibration":
            param.requires_grad = "calibration_" in name
        elif trainable_scope == "calibration_bias":
            param.requires_grad = "calibration_bias" in name
        elif trainable_scope == "calibration_temperature":
            param.requires_grad = "calibration_log_temperature" in name
        elif trainable_scope in {"bridge_calibration", "calibration_bridge"}:
            param.requires_grad = "representation_bridge" in name or "calibration_" in name
        elif trainable_scope == "none":
            param.requires_grad = False
        else:
            raise ValueError(f"Unknown ue_layer.trainable_scope: {trainable_scope}")

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable scope: %s | parameters: %d", trainable_scope, trainable_params)
    log.info("Trainable tensors: %s", trainable_names)

    return model


def expected_calibration_error(labels, probabilities, n_bins=10):
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges[1:-1], right=True), 0, n_bins - 1)
    ece = 0.0
    for bin_id in range(n_bins):
        in_bin = bin_ids == bin_id
        if not np.any(in_bin):
            continue
        confidence = probabilities[in_bin].mean()
        accuracy = labels[in_bin].mean()
        ece += in_bin.mean() * abs(confidence - accuracy)
    return float(ece)


def binary_metrics(labels, logits):
    labels = np.asarray(labels, dtype=int)
    logits = np.asarray(logits, dtype=float)
    probabilities = expit(logits)
    predictions = (probabilities > 0.5).astype(int)
    precision_curve, recall_curve, thresholds = precision_recall_curve(labels, probabilities)
    curve_f1 = 2 * precision_curve * recall_curve / np.clip(precision_curve + recall_curve, 1e-12, None)
    best_index = int(np.nanargmax(curve_f1))
    best_threshold = float(thresholds[best_index]) if best_index < len(thresholds) else 1.0
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": auc(recall_curve, precision_curve),
        "average_precision": average_precision_score(labels, probabilities),
        "best_f1": float(curve_f1[best_index]),
        "best_f1_threshold": best_threshold,
        "positive_rate": float(labels.mean()),
        "brier": brier_score_loss(labels, probabilities),
        "ece_10": expected_calibration_error(labels, probabilities, n_bins=10),
    }


def load_data(config, tokenizer):
    log.info(f"Loading dataset {config.dataset.path}...")
    dataset = load_any_dataset(config.dataset.path, config)

    if config.dataset.label_key_claim != "verified":
        dataset = dataset.remove_columns(["verified"])
        dataset = dataset.rename_column(config.dataset.label_key_claim, "verified")

    if config.dataset.label_key_token != "uncertainty_labels":
        dataset = dataset.remove_columns(["uncertainty_labels"])
        dataset = dataset.rename_column(config.dataset.label_key_token, "uncertainty_labels")

    if type(dataset) is not DatasetDict:
        dataset = DatasetDict({"train": dataset})

    if config.dataset.num_instances:
        dataset["train"] = dataset["train"].select(range(config.dataset.num_instances))

    tokenized_data = dataset["train"]

    if "test" not in dataset:
        log.info("Performing train-test split...")
        tokenized_data = tokenized_data.train_test_split(
            test_size=config.dataset.test_size,
            seed=42,
        )

    else:
        val_dataset_name = config.dataset.validation if hasattr(config.dataset, "validation") else "eval"
        tokenized_data = DatasetDict({"train": tokenized_data, "test": dataset[val_dataset_name]})

    # Add subset functionality for debugging
    if hasattr(config.dataset, 'subset') and config.dataset.subset is not None:
        subset_size = config.dataset.subset
        log.info(f"Using subset of {subset_size} samples for debugging...")

        # Suppress gradient checkpointing warnings for small subsets
        # These warnings are expected when batches have no trainable parameters
        warnings.filterwarnings("ignore", message="None of the inputs have requires_grad=True")

        # Apply subset to training data
        if len(tokenized_data["train"]) > subset_size:
            tokenized_data["train"] = tokenized_data["train"].select(range(subset_size))
            log.info(f"Training dataset reduced to {len(tokenized_data['train'])} samples")

        # Apply subset to test data (use smaller subset for faster evaluation)
        test_subset_size = min(subset_size // 4, len(tokenized_data["test"]))  # Use 1/4 of subset size for test
        if len(tokenized_data["test"]) > test_subset_size and test_subset_size > 0:
            tokenized_data["test"] = tokenized_data["test"].select(range(test_subset_size))
            log.info(f"Test dataset reduced to {len(tokenized_data['test'])} samples")

    train_subset = getattr(config.dataset, "train_subset", None)
    if train_subset is not None and int(train_subset) > 0:
        train_subset = min(int(train_subset), len(tokenized_data["train"]))
        subset_seed = int(getattr(config, "seed", 42))
        tokenized_data["train"] = tokenized_data["train"].shuffle(
            seed=subset_seed
        ).select(range(train_subset))
        log.info("Training dataset limited to %d adaptation examples", train_subset)

    eval_subset = getattr(config.dataset, "eval_subset", None)
    if eval_subset is not None and int(eval_subset) > 0:
        eval_subset = min(int(eval_subset), len(tokenized_data["test"]))
        tokenized_data["test"] = tokenized_data["test"].select(range(eval_subset))
        log.info("Evaluation dataset limited to %d examples", eval_subset)

    def prompt_tokens(inst):
        # The released dataset was produced with a prompt that is not identical
        # to every prompt template currently shipped in this repository.  The
        # old implementation re-tokenized the current template, which produced
        # impossible negative reply lengths and empty/misaligned claim masks.
        # Locate the already-saved reply in the already-saved input IDs instead.
        reply_start = locate_reply_start(inst["input_ids"], inst["reply"], tokenizer)
        return {
            "prompt_tokens": inst["input_ids"][:reply_start],
            "reply_start": reply_start,
        }

    tokenized_data = tokenized_data.map(prompt_tokens)
    log.info(f"Final length of the training dataset: {len(tokenized_data['train'])}")
    log.info(f"Final length of the testing dataset: {len(tokenized_data['test'])}")

    return tokenized_data


def load_additional_test_datasets(config, tokenizer):
    if not getattr(config, "additional_test_datasets", {}):
        return {}

    additional_datasets = {}
    def prompt_tokens(inst):
        reply_start = locate_reply_start(inst["input_ids"], inst["reply"], tokenizer)
        return {
            "prompt_tokens": inst["input_ids"][:reply_start],
            "reply_start": reply_start,
        }

    for name, path in config.additional_test_datasets.items():
        log.info(f"Loading additional test dataset '{name}' from: {path}")
        dataset = load_any_dataset(path, config)
        if "test" in dataset:
            dataset_test = dataset["test"]
        else:
            log.info("Performing train-test split...")
            dataset_test = dataset['train'].train_test_split(
                test_size=config.dataset.test_size,
                seed=42,
            )['test']
        dataset_test = dataset_test.map(prompt_tokens)
        additional_datasets[name] = dataset_test
        log.info(f"Loaded additional dataset '{name}' with {len(dataset_test)} examples.")

    return additional_datasets


# def _add_attention_mask(e):
#     if "attention_mask" not in e.keys() or e["attention_mask"] is None:
#         e["attention_mask"] = [1 for _ in e["input_ids"]]
#     return e


class DataCollatorForLanguageModelingWithUncertainty(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, *args, **kwargs):
        self._tokenizer = tokenizer
        super().__init__(tokenizer, *args, **kwargs)

    def torch_call(self, examples):
        # examples = [_add_attention_mask(e) for e in examples]
        # ex = [{"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]} for e in examples]
        # examples = [_add_attention_mask(e) for e in examples]
        # ex = [
        #     {k: v for k, v in e.items() if k not in [
        #         "uncertainty_labels", "reply"
        #      ]} for e in examples
        # ]
        # batch = super().torch_call(ex)

        batch_size = len(examples)

        # Do padding of input_ids
        batch = super().torch_call([{'input_ids': e['input_ids']} for e in examples])

        # Construct context lengths
        context_lengths = []
        for i in range(batch_size):
            reply_len = len(examples[i]['input_ids']) - len(examples[i]['prompt_tokens'])
            context_lengths.append(batch["input_ids"][i].shape[0] - reply_len)

        batch["context_lengths"] = torch.tensor(context_lengths)

        # Do padding of labels
        all_padded_labels = []
        for idx in range(len(examples)):
            uncertainty_labels = examples[idx]["uncertainty_labels"]
            difference = len(batch["input_ids"][0]) - len(uncertainty_labels)

            if self.tokenizer.padding_side == "right":
                # Llama 3.2
                raise Exception("Internal: detected right padding side, but set 'left' before")
                padded_labels = uncertainty_labels + [-100] * difference
            elif self.tokenizer.padding_side == "left":
                # Mistral, Gemma 2
                padded_labels = [-100] * difference + uncertainty_labels
            else:
                raise ValueError(f"Unknown padding side: {self.tokenizer.padding_side}")

            all_padded_labels.append(padded_labels)

        # print("Before:=========", batch["uncertainty_labels"])
        # for i in range(len(batch)):
        #     batch["uncertainty_labels"][i] = torch.tensor(all_padded_labels[i])
        batch["uncertainty_labels"] = torch.tensor(all_padded_labels)

        # context_lengths = []
        # for i in range(len(batch["input_ids"])):
        #     reply = examples[i]["reply"]
        #     input_ids = batch["input_ids"][i]
        #     ctx = 0
        #     pref_len = len(self._tokenizer.decode(input_ids).split(reply)[0])
        #     while ctx < len(input_ids) and len(self._tokenizer.decode(input_ids[:ctx + 1])) <= pref_len:
        #         ctx += 1
        #     context_lengths.append(ctx)

        # batch["context_lengths"] = torch.tensor(context_lengths)
        # print("After:=========",  batch["uncertainty_labels"])
        return batch


class DataCollatorForLanguageModelingWithUncertaintyClaim(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, claim_num_upper_bound=-1, *args, **kwargs):
        self._tokenizer = tokenizer
        self.claim_num_upper_bound = claim_num_upper_bound
        super().__init__(tokenizer, *args, **kwargs)

    def _adjust_claim_positions(self, context_length, input_ids, claim_obj):
        claim_token_positions = claim_obj['aligned_token_ids']
        mapping = []
        for idx, token_id in enumerate(input_ids[context_length:]):
            if token_id not in self.tokenizer.all_special_ids:
                mapping.append(idx)

        try:
            adjusted_positions = [mapping[i] for i in claim_token_positions if len(mapping) > i]
        except Exception as e:
            print('Mapping:', len(mapping))
            print('Claim token positions:', claim_token_positions)
            raise e
        return (context_length + torch.tensor(adjusted_positions)).int()

    def torch_call(self, examples):
        batch_size = len(examples)

        # Do padding
        batch = super().torch_call([{'input_ids': e['input_ids']} for e in examples])

        # Construct context lengths
        context_lengths = []
        for i in range(batch_size):
            reply_len = len(examples[i]['input_ids']) - len(examples[i]['prompt_tokens'])
            context_lengths.append(batch["input_ids"][i].shape[0] - reply_len)

        batch["context_lengths"] = torch.tensor(context_lengths)

        log.info(f'CONTEXT LENGTHS: {context_lengths}')

        # Construct claim tensors
        all_claim_tensors = []
        all_verified = []
        for i in range(len(batch["input_ids"])):
            instance_claims = []
            full_len = batch["input_ids"].shape[1]
            if self.claim_num_upper_bound > 0:
                rand_idx = random.sample(range(len(examples[i]["claims"])), min(self.claim_num_upper_bound, len(examples[i]["claims"])))
                rand_idx.sort()
                claims = [examples[i]["claims"][j] for j in rand_idx]
                verified = [examples[i]["verified"][j] for j in rand_idx]
            else:
                claims = examples[i]["claims"]
                verified = examples[i]["verified"]
            all_verified.append(verified)
            for claim in claims:
                mask = torch.zeros(batch["input_ids"][i].shape, dtype=int)
                claim_token_positions = self._adjust_claim_positions(batch["context_lengths"][i], batch["input_ids"][i],
                                                                     claim)
                if any(e >= mask.shape[0] for e in claim_token_positions):
                    raise IndexError("Aligned claim token is outside the padded input")

                mask[claim_token_positions] = 1
                instance_claims.append(mask[1:])  # ignoring <s>

            all_claim_tensors.append(
                torch.stack(instance_claims) if len(instance_claims) > 0 else torch.zeros(0, full_len - 1, dtype=int))

        batch["claims"] = all_claim_tensors

        # Construct labels
        all_labels = []
        for i in range(len(all_verified)):
            uncertainty_labels = all_verified[i]
            all_labels.append([e if not np.isnan(e) else -100 for e in uncertainty_labels])

        batch["verified"] = all_labels

        dict_batch = dict(batch)
        return dict_batch

def compute_claim_level_metrics(tokenized_data, logits):
    """Reference implementation before optimization."""
    claim_level_targets = list(chain(*tokenized_data["verified"]))

    num_instances = logits.shape[0]
    claim_level_preds = []
    for i in range(num_instances):
        prompt_tokens = tokenized_data[i]["prompt_tokens"]
        generated_tokens = tokenized_data[i]["input_ids"][len(prompt_tokens):]
        context_length = logits[i].shape[0] - len(generated_tokens)
        for claim in tokenized_data["claims"][i]:
            claim_preds = [
                logits[i, context_length + token - 1] for token in claim["aligned_token_ids"]
            ]
            ue_score = np.mean(claim_preds)
            claim_level_preds.append(ue_score)

    assert len(claim_level_targets) == len(claim_level_preds)

    mask = (~np.isnan(claim_level_targets)) & (claim_level_targets != -100)
    claim_level_targets = np.array(claim_level_targets)[mask]
    claim_level_preds = np.array(claim_level_preds)[mask]
    precs, recs, _ = precision_recall_curve(claim_level_targets, claim_level_preds)
    pr_auc = auc(recs, precs)
    return {"claim_level_pr_auc": pr_auc}


def compute_metrics(tokenized_data, eval_pred):
    logits, labels = eval_pred
    labels = labels[1]
    labels = labels[:, 1:]  # Shifting labels
    logits = logits[1] if type(logits) is tuple else logits

    mask = (labels != -100).reshape(-1)
    labels = labels.reshape(-1)[mask]
    probas = expit(logits.reshape(-1)[mask])
    predictions = (probas > 0.5).astype(int)

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    roc_auc = roc_auc_score(labels, probas)
    precs, recs, _ = precision_recall_curve(labels, probas)
    pr_auc = auc(recs, precs)

    neg_probas = 1.0 - probas
    neg_labels = 1.0 - labels
    neg_predictions = (neg_probas > 0.5).astype(int).reshape(-1)
    neg_f1 = f1_score(neg_labels, neg_predictions)
    neg_roc_auc = roc_auc_score(neg_labels, neg_probas)
    neg_precs, neg_recs, _ = precision_recall_curve(neg_labels, neg_probas)
    neg_pr_auc = auc(neg_recs, neg_precs)

    claim_level_metrics = compute_claim_level_metrics(tokenized_data, logits)

    final_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "neg_f1": neg_f1,
        "neg_roc_auc": neg_roc_auc,
        "neg_pr_auc": neg_pr_auc,
    }

    final_metrics.update(claim_level_metrics)
    return final_metrics


def compute_metrics_claims(tokenized_data, eval_pred):
    logits, labels = eval_pred

    labels = np.asarray([e if not np.isnan(e) else -100 for e in list(chain(*tokenized_data["verified"]))])

    mask = (labels != -100).reshape(-1)
    labels = labels.reshape(-1)[mask]

    logits = logits.reshape(-1)
    logits = logits[logits != -100]
    logits = logits[mask]
    return binary_metrics(labels, logits)


def save_claim_predictions(tokenized_data, predictions, output_path):
    """Persist compact claim logits without saving the language-model logits."""
    if isinstance(predictions, tuple):
        if len(predictions) != 1:
            raise ValueError(
                f"Expected only uncertainty predictions after ignoring LM logits, got {len(predictions)}"
            )
        predictions = predictions[0]

    example_indices = []
    claim_indices = []
    labels = []
    logits = []
    for example_index, row in enumerate(tokenized_data):
        row_logits = np.asarray(predictions[example_index]).reshape(-1)
        row_labels = np.asarray(row["verified"], dtype=float)
        if len(row_logits) < len(row_labels):
            raise ValueError(
                f"Example {example_index} has {len(row_logits)} predictions for {len(row_labels)} claims"
            )
        row_logits = row_logits[: len(row_labels)]
        for claim_index, (label, logit_value) in enumerate(zip(row_labels, row_logits)):
            if np.isnan(label) or label == -100:
                continue
            if logit_value == -100:
                raise ValueError(
                    f"Example {example_index} claim {claim_index} unexpectedly has a padding logit"
                )
            example_indices.append(example_index)
            claim_indices.append(claim_index)
            labels.append(int(label))
            logits.append(float(logit_value))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        example_indices=np.asarray(example_indices, dtype=np.int64),
        claim_indices=np.asarray(claim_indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int8),
        logits=np.asarray(logits, dtype=np.float32),
    )
    log.info("Saved %d compact claim predictions to %s", len(logits), output_path)


class LoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            log.info(logs)


class UHeadOnlySaveCallback(TrainerCallback):
    """Custom callback to save only uncertainty heads instead of full model checkpoints."""
    
    def on_save(self, args, state, control, **kwargs):
        """Override default save behavior to save only uncertainty head."""
        if state.is_local_process_zero:
            # Get the model and trainer from kwargs
            model = kwargs.get('model')
            trainer = kwargs.get('trainer')

            if model is not None and hasattr(model, 'ue_head'):
                # Prepare checkpoint directory path
                checkpoint_folder = f"uhead-{state.global_step}"
                output_dir = Path(args.output_dir) / checkpoint_folder
                uhead_path = output_dir

                # Check if DeepSpeed is being used
                if trainer and hasattr(trainer, 'deepspeed') and trainer.deepspeed:
                    # Import deepspeed if available
                    try:
                        import deepspeed
                        # Gather parameters for saving when using DeepSpeed
                        with torch.no_grad(), deepspeed.zero.GatheredParameters(model.ue_head.parameters(), modifier_rank=0):
                            if trainer.is_world_process_zero():
                                output_dir.mkdir(exist_ok=True, parents=True)
                                model.ue_head.save(uhead_path)
                                log.info(f"Saved uncertainty head checkpoint (DeepSpeed) to: {uhead_path}")
                    except ImportError:
                        log.warning("DeepSpeed not available, falling back to regular save")
                        output_dir.mkdir(exist_ok=True, parents=True)
                        model.ue_head.save(uhead_path)
                        log.info(f"Saved uncertainty head checkpoint to: {uhead_path}")
                else:
                    # Regular saving without DeepSpeed
                    output_dir.mkdir(exist_ok=True, parents=True)
                    model.ue_head.save(uhead_path)
                    log.info(f"Saved uncertainty head checkpoint to: {uhead_path}")

        # Allow default model saving to continue
        control.should_save = True
        return control


from collections import defaultdict
import numpy as np
from datasets import Dataset
from transformers import Trainer


class TrainerCustom(Trainer):
    def __init__(
            self,
            *args,
            eval_dataset: Dataset,
            additional_eval_datasets: dict[str, Dataset] | None = None,
            compute_metrics_fn=compute_metrics_claims,
            **kwargs,
    ):
        super().__init__(*args, **kwargs, eval_dataset=eval_dataset)
        self.eval_dataset = eval_dataset
        self.additional_eval_datasets = additional_eval_datasets or {}
        self.compute_metrics_fn = compute_metrics_fn

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Emits:
          - main eval:                   eval_<metric>
          - per-additional-dataset:      eval_<ds_name>_<metric>
          - means (all sets):            eval_mean_<metric>
        """
        callbacks_to_remove = []
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, EarlyStoppingCallback):
                callbacks_to_remove.append(callback)

        # Temporarily remove problematic callbacks
        for callback in callbacks_to_remove:
            self.callback_handler.callbacks.remove(callback)

        try:
            results: dict[str, float] = {}
            avg_metrics = {
                "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                "average_precision", "best_f1", "positive_rate", "brier", "ece_10",
            }
            buckets = defaultdict(list)  # metric -> [values from main eval + all additional evals]

            # ---- Main eval set ----
            self.compute_metrics = lambda eval_pred: self.compute_metrics_fn(self.eval_dataset, eval_pred)
            base_eval_ds = self.eval_dataset if eval_dataset is None else eval_dataset
            log.info(f"Running base eval on dataset of size {len(base_eval_ds)}")
            base_metrics = super().evaluate(
                eval_dataset=base_eval_ds,
                ignore_keys=["logits"] if ignore_keys is None else ignore_keys,
                metric_key_prefix="eval",
            )
            # base_metrics keys like "eval_f1", "eval_pr_auc", plus loss/runtime/etc.
            for k, v in base_metrics.items():
                if k.startswith("eval_"):
                    metric = k[len("eval_"):]  # "f1", "roc_auc", ...
                    val = float(v)
                    # nice hierarchy + table-friendly alias
                    results[f"eval_{metric}"] = val
                    if metric in avg_metrics:
                        buckets[metric].append(val)
                else:
                    results[k] = v
            log.info("Finished base eval")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # ---- Additional datasets ----
            for name, ds in self.additional_eval_datasets.items():
                log.info(f"Running additional eval '{name}' on dataset of size {len(ds)}")
                self.compute_metrics = (lambda eval_pred, d=ds: self.compute_metrics_fn(d, eval_pred))
                m = super().evaluate(
                    eval_dataset=ds,
                    ignore_keys=["logits"] if ignore_keys is None else ignore_keys,
                    metric_key_prefix=f"eval_{name}",
                )
                for k, v in m.items():
                    if not k.startswith(f"eval_{name}_"):
                        # preserve any unexpected extras under a namespaced key
                        results[f"{name}_{k}"] = v
                        continue
                    metric = k[len(f"eval_{name}_"):]
                    val = float(v)
                    results[f"eval_{name}_{metric}"] = val
                    if metric in avg_metrics:
                        buckets[metric].append(val)
                log.info(f"Finished additional eval '{name}'")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            mean_results = {}
            for metric, vals in buckets.items():
                if not vals:
                    continue
                mean_val = float(np.mean(vals))
                mean_results[f"eval_mean_{metric}"] = mean_val
                results[f"eval_mean_{metric}"] = mean_val
            
            # log mean results, others are logged in the callback in super().evaluate
            log.info(f"Mean results: {mean_results}")
            self.log(mean_results)

        finally:
            # Restore the callbacks we temporarily removed
            for callback in callbacks_to_remove:
                self.callback_handler.callbacks.append(callback)

        # Now trigger callbacks with the final computed metrics
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, results)

        self._memory_tracker.stop_and_update_metrics(results)

        return results


def wandb_save_directory(directory_path):
    import wandb

    for file_name in os.listdir(directory_path):
        full_path = os.path.join(directory_path, file_name)
        if os.path.isfile(full_path):  # Make sure it's a file, not a directory
            wandb.save(full_path)


hydra_cfg_path = os.environ.get("HYDRA_CONFIG", None)
hydra_cfg_dir = str(Path(hydra_cfg_path).parent) if hydra_cfg_path is not None else None
hydra_cfg_name = str(Path(hydra_cfg_path).name) if hydra_cfg_path is not None else None


@hydra.main(
    version_base=None,
    config_path=hydra_cfg_dir,
    config_name=hydra_cfg_name,
)
def main(config):
    output_dir = HydraConfig.get().runtime.output_dir
    log.info(f"Output directory: {output_dir}")

    # setup huggingface logger
    hf_logger.handlers = []
    for h in log.handlers:
        hf_logger.addHandler(h)

    if config.report_to == "wandb":
        import wandb

        wandb_cfg = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        config_path_hydra = [
            path["path"]
            for path in HydraConfig.get().runtime.config_sources
            if path["schema"] == "file"
        ][0]
        wandb_cfg["HYDRA_CONFIG"] = (
                Path(config_path_hydra) / HydraConfig.get().job.config_name
        )
        os.environ["WANDB_DIR"] = str(Path(output_dir))
        project = os.environ["WANDB_PROJECT"]
        wandb.init(project=project, dir=output_dir, config=wandb_cfg)
        wandb_save_directory(Path(output_dir) / ".hydra")

    hf_logger.info("Init transformers logger.")

    random_seed = int(getattr(config, "seed", 42))
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    set_seed(random_seed)
    np.random.seed(random_seed)

    if os.environ.get("CUDNN_DETERMINISTIC", "0") == "1":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model = None
    f_model_init = None
    if config.do_hyperopt:

        def model_init(trial):
            if trial is None:
                return load_model(config)

            log.info(repr(trial))
            model_params = OmegaConf.to_container(config, resolve=True)
            model_params["ue_layer"]["n_layers"] = trial["n_layers"]
            model_params["ue_layer"]["n_heads"] = trial["n_heads"]
            model_params["ue_layer"]["pos_weight"] = trial["pos_weight"]
            omega_model_params = OmegaConf.create(model_params)

            return load_model(omega_model_params)

        f_model_init = model_init

    else:
        log.info("Loading model...")
        model = load_model(config)
        log.info("Done.")
        log.info(repr(model))

    log.info("Loading tokenizer...")
    tokenizer = load_tokenizer(config)
    log.info("Done.")

    log.info("Loading dataset...")

    tokenized_data = load_data(config, tokenizer)
    additional_test_datasets = load_additional_test_datasets(config, tokenizer)
    log.info("Done.")
    log.info(repr(tokenized_data))

    train_args = TrainingArguments(
        num_train_epochs=config.training_arguments.num_train_epochs,
        per_device_train_batch_size=config.training_arguments.per_device_train_batch_size,
        per_device_eval_batch_size=config.training_arguments.per_device_train_batch_size,
        # TODO: add parameter for eval batch size
        gradient_accumulation_steps=config.training_arguments.gradient_accumulation_steps,
        eval_accumulation_steps=4,
        learning_rate=config.training_arguments.learning_rate,
        weight_decay=config.training_arguments.weight_decay,
        max_grad_norm=config.training_arguments.max_grad_norm,
        warmup_ratio=config.training_arguments.warmup_ratio,
        lr_scheduler_type="linear",
        # fp16=True,  # Had to comment for Qwen2.5-Math-1.5B, othervise it output nan logits and attentions
        # fp16_full_eval=False,
        load_best_model_at_end=True if config.do_save_checkpoints else False,
        metric_for_best_model="eval_mean_pr_auc",
        greater_is_better=True,
        # Calibration-only sweeps do not need to replay the full validation set
        # after every epoch.  Honour the Hydra option instead of silently
        # hard-coding epoch evaluation; the final `do_eval` pass below is still
        # run and persisted for every experiment.
        eval_strategy=str(getattr(config.training_arguments, "eval_strategy", "epoch")),
        logging_strategy=str(getattr(config.training_arguments, "logging_strategy", "epoch")),
        save_strategy="epoch" if config.do_save_checkpoints else "no",
        output_dir=Path(output_dir) / "outputs",
        logging_dir=Path(output_dir) / "transformers_logs",
        report_to=config.report_to if config.report_to else None,
        seed=random_seed,
        data_seed=random_seed,
        include_num_input_tokens_seen=True,
        gradient_checkpointing=False,
        dataloader_num_workers=1,
        remove_unused_columns=False,
        save_total_limit=getattr(config.training_arguments, 'save_total_limit', 3),  # Add missing save_total_limit
        # label_names=["verified"]
    )

    if model.ue_head.model_type == "claim":
        def dataset_filter(inst):
            return len(inst['claims']) > 0

        # tokenized_data = {
        #     split: ds.filter(dataset_filter)
        #     for split, ds in tokenized_data.items()
        # }
        tokenized_data = tokenized_data.filter(dataset_filter)
        data_collator = DataCollatorForLanguageModelingWithUncertaintyClaim(tokenizer, claim_num_upper_bound=config.claim_num_upper_bound if config.claim_num_upper_bound is not None else -1, mlm=False)
    elif model.ue_head.model_type == "token":
        data_collator = DataCollatorForLanguageModelingWithUncertainty(tokenizer, mlm=False)

    callbacks = [LoggerCallback()]
    if config.do_save_checkpoints:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        callbacks.append(UHeadOnlySaveCallback())

    if model.ue_head.model_type == "claim":
        f_eval = compute_metrics_claims
    elif model.ue_head.model_type == "token":
        f_eval = compute_metrics

    trainer = TrainerCustom(
        model=model,
        model_init=f_model_init,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        args=train_args,
        data_collator=data_collator,
        callbacks=callbacks,
        additional_eval_datasets=additional_test_datasets,
        compute_metrics_fn=f_eval,
    )

    if config.do_hyperopt:
        # This option is only for hperparameter optimization using optuna
        # For optimization with wandb, use the wandb sweep feature

        def compute_objective(metrics):
            return metrics["eval_mean_pr_auc"]

        def hp_space(trial):
            return {
                "training_arguments": {
                    "learning_rate": trial.suggest_categorical(
                        "learning_rate", [1e-5, 5e-5, 1e-4]
                    ),
                    "weight_decay": trial.suggest_categorical(
                        "weight_decay", [0.0, 0.01, 0.1, 0.5]
                    ),
                    "warmup_ratio": trial.suggest_categorical(
                        "warmup_ratio", [0.0, 0.1]
                    ),
                    "num_train_epochs": trial.suggest_categorical(
                        "num_train_epochs", [5, 7, 10, 15]
                    ),
                },
                "ue_layer": {
                    "n_layers": trial.suggest_categorical("n_layers", [1, 2]),
                    "n_heads": trial.suggest_categorical("n_heads", [16, 32, 64]),
                    "pos_weight": trial.suggest_categorical(
                        "pos_weight", [4.0, 6.0, 12.0]
                    ),
                },
            }

        best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=30,
            compute_objective=compute_objective,
        )

        log.info(f"Best metric: {repr(best_trial.objective)}")
        log.info(f"Best hyperparameters: {repr(str(best_trial.hyperparameters))}")
        with open(Path(output_dir) / "best_hyperparameters.json", "w") as f:
            json.dump(best_trial.hyperparameters, f)

    else:
        if config.do_train:
            if not any(p.requires_grad for p in trainer.model.parameters()):
                raise ValueError("do_train=true but no parameters are trainable")
            trainer.model.orig_base_model.config.use_cache = False

            try:
                train_result = trainer.train(ignore_keys_for_eval=["logits"])
            except KeyboardInterrupt:
                log.info("Training interrupted.")
                train_result = None

            if train_result is not None:
                with open(Path(output_dir) / "train_metrics.json", "w") as f:
                    json.dump(train_result.metrics, f, indent=2, sort_keys=True)

            log.info("Done with training.")

            if config.do_save_final_model:
                log.info("Saving model...")
                save_path = Path(output_dir) / "model"
                trainer.model.ue_head.save(save_path)
                # # trainer.save_model(Path(output_dir) / "training_dir" / "final_model")
                # torch.save(
                #     trainer.model.ue_head.state_dict(),
                #     Path(output_dir) / "ue_layer.pth",
                # )
                log.info(f"Saved to: {save_path}")
                if getattr(config, 'save_dir', None) is not None:
                    trainer.model.ue_head.save(Path(config.save_dir))
                    log.info(f"Saved to: {config.save_dir}.")
                if getattr(config, 'hf_save_path', None) is not None:
                    trainer.model.ue_head.push_to_hub(config.hf_save_path)
                    log.info(f"Saved to HF: {config.hf_save_path}.")

        if config.do_eval:
            log.info("Evaluating...")
            eval_metrics = trainer.evaluate(ignore_keys=["logits"])
            log.info(eval_metrics)
            with open(Path(output_dir) / "eval_metrics.json", "w") as f:
                json.dump(eval_metrics, f, indent=2, sort_keys=True)
            log.info("Done with evaluation.")

        if config.do_predict:
            log.info("Predicting...")
            prediction_split = str(getattr(config.dataset, "prediction_split", "test"))
            if prediction_split not in tokenized_data:
                raise ValueError(
                    f"Prediction split {prediction_split!r} is not available; "
                    f"choose one of {list(tokenized_data)}"
                )
            # The base LM logits are [batch, sequence, vocabulary] and are not
            # needed for process-verifier analysis. Ignoring them avoids a very
            # large, redundant prediction artifact.
            predictions = trainer.predict(
                tokenized_data[prediction_split], ignore_keys=["logits"]
            )
            log.info("Done with prediction.")
            prediction_filename = (
                "claim_predictions.npz"
                if prediction_split == "test"
                else f"claim_predictions_{prediction_split}.npz"
            )
            save_claim_predictions(
                tokenized_data[prediction_split],
                predictions.predictions,
                Path(output_dir) / prediction_filename,
            )


if __name__ == "__main__":
    main()
