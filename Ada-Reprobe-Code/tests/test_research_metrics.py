import math

import numpy as np
import torch

from auto_res.scripts.evaluate_natural_best_of_n import (
    generation_batch_seed,
    generation_batches,
    generation_mask,
    is_strict_scalar_numeric_answer,
    normalize_benchmark_row,
    numerically_equal,
    parse_number,
    run_configs_compatible,
)
from auto_res.scripts.analyze_natural_bestofn import verifier_consensus_correct
from auto_res.scripts.analyze_consensus_ambiguity import (
    ambiguity_unit,
    cluster_bootstrap as ambiguity_cluster_bootstrap,
    summarize as summarize_ambiguity,
)
from auto_res.scripts.ablate_calibration_parameters import (
    bernoulli_nll,
    calibrated_probabilities,
)
from auto_res.scripts.evaluate_selective_pruning import aurc, oracle_aurc


def test_gsm8k_number_parser_handles_common_answer_formats():
    assert parse_number("work\n#### 1,234") == 1234.0
    assert parse_number(r"therefore \boxed{3/4}") == 0.75
    assert parse_number("Final answer: -2.5") == -2.5
    assert parse_number("no numeric result") is None
    assert numerically_equal(0.75, 3 / 4)
    assert not numerically_equal(None, 0.0)


def test_generation_mask_stops_after_first_terminal_token():
    values = torch.tensor([[1, 2, 9, 9], [1, 3, 4, 5]])
    assert generation_mask(values, {2, 9}).tolist() == [
        [1, 1, 0, 0],
        [1, 1, 1, 1],
    ]


def test_fixed_generation_batches_preserve_original_resume_boundaries():
    rows = [{"question_id": index} for index in range(8)]
    batches = generation_batches(rows, {0}, batch_size=4, resume_safe=True)
    assert [index for index, _ in batches] == [0, 1]
    assert [[row["question_id"] for row in batch] for _, batch in batches] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert generation_batches(rows, {0, 1, 2, 3}, 4, resume_safe=True) == [
        (1, rows[4:])
    ]


def test_generation_batch_seeds_are_stable_and_distinct():
    assert generation_batch_seed(2032, 7) == generation_batch_seed(2032, 7)
    assert generation_batch_seed(2032, 7) != generation_batch_seed(2032, 8)


def test_numeric_benchmark_schemas_are_normalized_without_answer_leakage():
    cases = [
        (
            "gsm8k",
            {"question": "What is 2+3?", "answer": "work #### 5"},
            "What is 2+3?",
            5.0,
        ),
        (
            "svamp",
            {"Body": "There are two bags.", "Question": "How many?", "Answer": "2"},
            "There are two bags. How many?",
            2.0,
        ),
        (
            "multiarith",
            {"question": "Half of six?", "final_ans": "3"},
            "Half of six?",
            3.0,
        ),
        (
            "asdiv",
            {"body": "Seven red apples.", "question": "How many?", "answer": "7 (apples)"},
            "Seven red apples. How many?",
            7.0,
        ),
    ]
    for name, raw, expected_question, expected_answer in cases:
        normalized = normalize_benchmark_row(name, raw, 17)
        assert normalized["question_id"] == 17
        assert normalized["question"] == expected_question
        assert normalized["reference_answer"] == expected_answer
        assert str(raw) not in normalized["question"]


def test_asdiv_filter_keeps_only_supported_scalar_numeric_answers():
    assert is_strict_scalar_numeric_answer("9 (apples)")
    assert is_strict_scalar_numeric_answer("-1/3 (mile)")
    assert not is_strict_scalar_numeric_answer("Online")
    assert not is_strict_scalar_numeric_answer("3:30 p.m.")
    assert not is_strict_scalar_numeric_answer("5; 15; 20")
    assert not is_strict_scalar_numeric_answer("-(1/3)")


def test_legacy_gsm_config_is_compatible_but_cross_dataset_is_not():
    legacy = {"seed": 2030, "num_questions": 100}
    gsm = {
        **legacy,
        "dataset": "gsm8k",
        "dataset_repo": "gsm8k",
        "dataset_config": "main",
        "dataset_split": "test",
        "dataset_revision": "pinned",
    }
    svamp = {**gsm, "dataset": "svamp", "dataset_repo": "ChilleD/SVAMP"}
    assert run_configs_compatible(legacy, gsm)
    assert not run_configs_compatible(legacy, svamp)


def test_temperature_bias_calibration_improves_shifted_synthetic_logits():
    train_logits = np.asarray([-5.0, -3.0, -1.0, 1.0, 3.0, 5.0])
    train_labels = np.asarray([0, 0, 0, 1, 1, 1])
    shifted = train_logits + 4.0
    probabilities, parameters = calibrated_probabilities(
        "temperature_bias", shifted, train_labels, shifted
    )
    clipped = np.clip(probabilities, 1e-8, 1 - 1e-8)
    fitted_nll = -np.mean(
        train_labels * np.log(clipped) + (1 - train_labels) * np.log(1 - clipped)
    )
    assert fitted_nll < bernoulli_nll(train_labels, shifted)
    assert parameters["bias"] < 0


def test_constant_prevalence_is_finite_for_one_class_support():
    probabilities, parameters = calibrated_probabilities(
        "constant_prevalence",
        np.asarray([-2.0, -1.0]),
        np.asarray([0, 0]),
        np.asarray([-10.0, 10.0, 0.0]),
    )
    assert np.all(np.isfinite(probabilities))
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert parameters["constant_probability"] == 1 / 6


def test_oracle_selective_risk_is_no_worse_than_reversed_ranking():
    labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
    oracle = oracle_aurc(labels)
    reversed_ranking = aurc(labels, 1 - labels)
    assert math.isclose(oracle, aurc(labels, labels))
    assert oracle < reversed_ranking


def test_rank_weighted_consensus_can_override_a_single_low_risk_outlier():
    rows = [
        {"predicted_answer": 1.0, "reference_answer": 2.0},
        {"predicted_answer": 2.0, "reference_answer": 2.0},
        {"predicted_answer": 2.0, "reference_answer": 2.0},
        {"predicted_answer": 2.0, "reference_answer": 2.0},
    ]
    risks = [0.1, 0.2, 0.3, 0.4]
    assert verifier_consensus_correct(rows, risks, "rank_weighted")


def test_plurality_tie_is_broken_by_lowest_verifier_risk():
    rows = [
        {"predicted_answer": 1.0, "reference_answer": 2.0},
        {"predicted_answer": 1.0, "reference_answer": 2.0},
        {"predicted_answer": 2.0, "reference_answer": 2.0},
        {"predicted_answer": 2.0, "reference_answer": 2.0},
    ]
    risks = [0.3, 0.4, 0.1, 0.2]
    assert verifier_consensus_correct(rows, risks, "plurality_tiebreak")


def test_ambiguity_audit_records_a_beneficial_tiebreak_intervention():
    rows = [
        {"predicted_answer": 1.0, "reference_answer": 2.0, "verifier_logits": {"source": 0.3}},
        {"predicted_answer": 1.0, "reference_answer": 2.0, "verifier_logits": {"source": 0.4}},
        {"predicted_answer": 2.0, "reference_answer": 2.0, "verifier_logits": {"source": 0.1}},
        {"predicted_answer": 2.0, "reference_answer": 2.0, "verifier_logits": {"source": 0.2}},
    ]
    result = ambiguity_unit(rows)
    assert result == {
        "ambiguous": True,
        "intervened": True,
        "majority_correct": False,
        "tiebreak_correct": True,
    }


def test_ambiguity_audit_never_intervenes_for_unique_plurality():
    rows = [
        {"predicted_answer": 1.0, "reference_answer": 1.0, "verifier_logits": {"source": 0.9}},
        {"predicted_answer": 1.0, "reference_answer": 1.0, "verifier_logits": {"source": 0.8}},
        {"predicted_answer": 2.0, "reference_answer": 1.0, "verifier_logits": {"source": 0.1}},
    ]
    result = ambiguity_unit(rows)
    assert not result["ambiguous"]
    assert not result["intervened"]
    assert result["majority_correct"]
    assert result["tiebreak_correct"]


def test_ambiguity_summary_handles_no_tied_pluralities():
    units = [
        {
            "question_id": index,
            "ambiguous": False,
            "intervened": False,
            "majority_correct": True,
            "tiebreak_correct": True,
        }
        for index in range(4)
    ]
    summary = summarize_ambiguity(units)
    intervals = ambiguity_cluster_bootstrap(units, repetitions=20, seed=7)
    assert summary["num_ambiguous"] == 0
    assert summary["ambiguous_majority_accuracy"] is None
    assert intervals["ambiguous_delta_cluster_ci95"] is None
