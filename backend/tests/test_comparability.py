"""Comparability engine tests (PRD 18).

Covers all four defined statuses, the reason codes behind them, model-aware
config handling, and the extensibility hooks.
"""

from __future__ import annotations

import pytest

from app.models.contracts import (
    ComparabilityReasonCode,
    ComparabilityStatus,
    EvidenceStatus,
    ExperimentStatus,
    MetricDirection,
)
from app.services.comparability import (
    ComparabilityEngine,
    ComparabilitySettings,
    ModelFamilyProfile,
    get_profile,
    register_model_family,
)
from tests.conftest import ACCURACY, make_experiment


def codes(assessment) -> set[ComparabilityReasonCode]:
    return {reason.code for reason in assessment.reasons}


# ---------------------------------------------------------------------------
# The four defined statuses (FR-4)
# ---------------------------------------------------------------------------


def test_identical_runs_are_comparable(comparability_engine, experiments):
    left, right = experiments[0], experiments[1]
    # Make the config equal so the only difference is the measured value.
    right = right.model_copy(update={"configuration": left.configuration})

    result = comparability_engine.assess(left, right, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.COMPARABLE
    assert result.score == 1.0
    assert result.confidence == "high"
    assert codes(result) == {ComparabilityReasonCode.IDENTICAL}
    assert "dataset_version" in result.matched_dimensions
    assert result.explanation


def test_different_dataset_version_is_not_comparable(comparability_engine, experiments):
    v1 = next(e for e in experiments if e.id == "exp_baseline")
    v2 = next(e for e in experiments if e.id == "exp_on_v2")

    result = comparability_engine.assess(v1, v2, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.NOT_COMPARABLE
    assert ComparabilityReasonCode.DATASET_VERSION_MISMATCH in codes(result)
    assert result.score == 0.0
    assert "Not comparable" in result.explanation


def test_different_evaluation_split_is_not_comparable(comparability_engine, experiments):
    validation = next(e for e in experiments if e.id == "exp_baseline")
    test = next(e for e in experiments if e.id == "exp_test_split")

    result = comparability_engine.assess(validation, test, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.NOT_COMPARABLE
    assert ComparabilityReasonCode.EVAL_SPLIT_MISMATCH in codes(result)


def test_different_dataset_identity_is_not_comparable(comparability_engine, experiments):
    churn = next(e for e in experiments if e.id == "exp_baseline")
    fraud = churn.model_copy(
        update={"id": "exp_fraud", "dataset_id": "ds_fraud", "dataset_name": "fraud"}
    )

    result = comparability_engine.assess(churn, fraud, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.NOT_COMPARABLE
    assert ComparabilityReasonCode.DATASET_MISMATCH in codes(result)


def test_missing_material_field_is_insufficient_information(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    no_version = baseline.model_copy(
        update={"id": "exp_noversion", "dataset_version": None, "dataset_version_id": None}
    )

    result = comparability_engine.assess(baseline, no_version, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.INSUFFICIENT_INFORMATION
    assert result.missing_dimensions
    assert result.confidence == "low"
    assert "Cannot decide" in result.explanation


def test_label_and_node_id_are_not_interchangeable(comparability_engine, experiments):
    """A label must never be compared against a graph node id (PRD 4)."""
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    labelled = baseline.model_copy(
        update={
            "id": "exp_labelled",
            # Same version, but recorded only as a label while the baseline uses
            # a node id. "v1" and "dsv_churn_v1" are not compared as strings.
            "dataset_version_id": None,
        }
    )

    result = comparability_engine.assess(baseline, labelled, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.INSUFFICIENT_INFORMATION
    assert ComparabilityReasonCode.DATASET_VERSION_MISMATCH not in codes(result)
    assert ComparabilityReasonCode.DATASET_VERSION_MISSING in codes(result)


def test_node_id_alone_is_enough_to_match(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    id_only = baseline.model_copy(update={"id": "exp_idonly", "dataset_version": None})

    result = comparability_engine.assess(baseline, id_only, metric_name=ACCURACY)

    assert ComparabilityReasonCode.DATASET_VERSION_MISMATCH not in codes(result)
    assert ComparabilityReasonCode.DATASET_VERSION_MISSING not in codes(result)


def test_missing_metric_value_is_insufficient_information(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    unmeasured = baseline.model_copy(
        update={
            "id": "exp_unmeasured",
            "metrics": [baseline.metrics[0].model_copy(update={"value": None})],
        }
    )

    result = comparability_engine.assess(baseline, unmeasured, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.INSUFFICIENT_INFORMATION
    assert ComparabilityReasonCode.METRIC_MISSING in codes(result)


def test_no_shared_metric_is_not_comparable(comparability_engine, experiments):
    gbm = next(e for e in experiments if e.id == "exp_baseline")
    transformer = next(e for e in experiments if e.id == "exp_transformer")

    result = comparability_engine.assess(gbm, transformer)

    assert result.status is ComparabilityStatus.NOT_COMPARABLE
    assert ComparabilityReasonCode.METRIC_MISMATCH in codes(result)


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def test_different_seed_is_a_caveat_not_a_blocker(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    other_seed = next(e for e in experiments if e.id == "exp_seed7")

    result = comparability_engine.assess(baseline, other_seed, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS
    assert ComparabilityReasonCode.RANDOM_SEED_MISMATCH in codes(result)
    assert result.is_usable_as_baseline is True


def test_missing_seed_is_a_caveat_not_insufficient_information(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    no_seed = baseline.model_copy(update={"id": "exp_noseed", "random_seed": None})

    result = comparability_engine.assess(baseline, no_seed, metric_name=ACCURACY)

    assert ComparabilityReasonCode.SEED_MISSING in codes(result)
    # A missing seed lowers confidence in the comparison but does not block it.
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS


def test_single_config_change_reads_as_controlled_ablation(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    deeper = next(e for e in experiments if e.id == "exp_depth6")

    result = comparability_engine.assess(baseline, deeper, metric_name=ACCURACY)

    assert ComparabilityReasonCode.MODEL_CONFIG_MISMATCH in codes(result)
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS
    reason = next(r for r in result.reasons if r.code is ComparabilityReasonCode.MODEL_CONFIG_MISMATCH)
    assert reason.left_value == {"max_depth": 4}
    assert reason.right_value == {"max_depth": 6}


def test_multiple_config_changes_are_reported_as_confounded(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    changed = baseline.model_copy(
        update={"id": "exp_multi", "configuration": {"max_depth": 6, "learning_rate": 0.3}}
    )

    result = comparability_engine.assess(baseline, changed, metric_name=ACCURACY)

    reason = next(r for r in result.reasons if r.code is ComparabilityReasonCode.MODEL_CONFIG_MISMATCH)
    assert set(reason.left_value) == {"max_depth", "learning_rate"}
    assert "confounded" in reason.message


def test_preprocessing_difference_is_a_caveat(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    rescaled = baseline.model_copy(
        update={"id": "exp_rescaled", "preprocessing": {"scaling": "robust"}}
    )

    result = comparability_engine.assess(baseline, rescaled, metric_name=ACCURACY)

    assert ComparabilityReasonCode.PREPROCESSING_MISMATCH in codes(result)
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS


def test_training_budget_difference_respects_tolerance(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")

    close = baseline.model_copy(update={"id": "exp_close", "training_steps": 1100})
    result = comparability_engine.assess(baseline, close, metric_name=ACCURACY)
    assert ComparabilityReasonCode.TRAINING_BUDGET_MISMATCH not in codes(result)

    far = baseline.model_copy(update={"id": "exp_far", "training_steps": 5000})
    result_far = comparability_engine.assess(baseline, far, metric_name=ACCURACY)
    assert ComparabilityReasonCode.TRAINING_BUDGET_MISMATCH in codes(result_far)


def test_budget_tolerance_is_configurable(experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    stretch = baseline.model_copy(update={"id": "exp_stretch", "training_steps": 1500})

    strict = ComparabilityEngine(
        ComparabilitySettings(training_budget_relative_tolerance=0.05)
    )
    result = strict.assess(baseline, stretch, metric_name=ACCURACY)
    assert ComparabilityReasonCode.TRAINING_BUDGET_MISMATCH in codes(result)


def test_declared_training_ceiling_counts_as_the_budget(comparability_engine, experiments):
    """A run with no realised step count still declares its planned budget."""
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    planned = baseline.model_copy(
        update={"id": "exp_planned", "training_steps": None, "max_training_steps": 50000}
    )

    result = comparability_engine.assess(baseline, planned, metric_name=ACCURACY)

    assert ComparabilityReasonCode.TRAINING_BUDGET_MISMATCH in codes(result)
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS


def test_missing_dataset_identity_uses_its_own_reason_code(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    anonymous = baseline.model_copy(
        update={"id": "exp_anon", "dataset_id": None, "dataset_name": None}
    )

    result = comparability_engine.assess(baseline, anonymous, metric_name=ACCURACY)

    assert ComparabilityReasonCode.DATASET_MISSING in codes(result)
    # Must not be reported under the dataset-version code.
    assert "dataset_identity" in result.missing_dimensions


def test_null_metric_value_is_missing_not_a_disagreement(comparability_engine, experiments):
    """Same metric name, no value: unrecorded, not a different measurement."""
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    blank = baseline.model_copy(
        update={
            "id": "exp_blank",
            "metrics": [baseline.metrics[0].model_copy(update={"value": None})],
        }
    )

    result = comparability_engine.assess(baseline, blank)

    assert ComparabilityReasonCode.METRIC_MISSING in codes(result)
    assert ComparabilityReasonCode.METRIC_MISMATCH not in codes(result)
    assert result.status is ComparabilityStatus.INSUFFICIENT_INFORMATION


def test_run_with_no_metrics_at_all_is_missing_not_matched(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    silent = baseline.model_copy(update={"id": "exp_silent", "metrics": []})

    result = comparability_engine.assess(baseline, silent)

    assert ComparabilityReasonCode.METRIC_MISSING in codes(result)
    assert "metric_availability" not in result.matched_dimensions


def test_reason_values_do_not_alias_the_stored_record(comparability_engine, experiments):
    """An assessment consumer must not be able to corrupt the experiment."""
    left = next(e for e in experiments if e.id == "exp_baseline")
    right = left.model_copy(
        update={"id": "exp_rescaled", "preprocessing": {"scaling": "robust"}}
    )

    result = comparability_engine.assess(left, right, metric_name=ACCURACY)
    reason = next(
        r for r in result.reasons if r.code is ComparabilityReasonCode.PREPROCESSING_MISMATCH
    )

    assert reason.right_value is not right.preprocessing
    reason.right_value["scaling"] = "MUTATED"
    assert right.preprocessing == {"scaling": "robust"}


def test_comparability_score_is_labelled_as_a_heuristic(comparability_engine, experiments):
    left = next(e for e in experiments if e.id == "exp_baseline")
    result = comparability_engine.assess(left, left, metric_name=ACCURACY)

    assert result.score_kind == "severity_penalty_heuristic"


def test_protocol_difference_is_a_caveat(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    # model_copy is shallow, so the metric object must be rebuilt explicitly or
    # this would mutate the baseline's metric instead.
    other = baseline.model_copy(
        update={
            "id": "exp_proto",
            "metrics": [
                baseline.metrics[0].model_copy(update={"evaluation_protocol": "holdout"})
            ],
        }
    )

    result = comparability_engine.assess(baseline, other, metric_name=ACCURACY)

    assert ComparabilityReasonCode.EVAL_PROTOCOL_MISMATCH in codes(result)
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS


def test_disputed_dataset_qualifies_without_blocking(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    disputed = baseline.model_copy(
        update={"id": "exp_disputed", "dataset_quality_status": EvidenceStatus.DISPUTED}
    )

    result = comparability_engine.assess(baseline, disputed, metric_name=ACCURACY)

    assert ComparabilityReasonCode.DISPUTED_EVIDENCE in codes(result)
    # Structurally valid comparison, but the data is not trusted.
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS


# ---------------------------------------------------------------------------
# Intentional changes and metric direction
# ---------------------------------------------------------------------------


def test_intended_change_does_not_become_a_confounder(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    deeper = next(e for e in experiments if e.id == "exp_depth6")

    result = comparability_engine.assess(
        baseline, deeper, metric_name=ACCURACY, ignore_config_fields=["max_depth"]
    )

    assert ComparabilityReasonCode.MODEL_CONFIG_MISMATCH not in codes(result)
    assert result.status is ComparabilityStatus.COMPARABLE


def test_intended_dataset_change_is_reported_as_info(comparability_engine, experiments):
    v1 = next(e for e in experiments if e.id == "exp_baseline")
    v2 = next(e for e in experiments if e.id == "exp_on_v2")

    result = comparability_engine.assess(
        v1, v2, metric_name=ACCURACY, ignore_dimensions=["dataset_version"]
    )

    assert result.status is not ComparabilityStatus.NOT_COMPARABLE
    reason = next(r for r in result.reasons if r.code is ComparabilityReasonCode.DATASET_VERSION_MISMATCH)
    assert "intentional independent variable" in reason.message


def test_opposite_metric_direction_is_not_comparable(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    flipped = baseline.model_copy(
        update={"id": "exp_flipped", "metrics": [baseline.metrics[0].model_copy(
            update={"direction": MetricDirection.MINIMIZE})]}
    )

    result = comparability_engine.assess(baseline, flipped, metric_name=ACCURACY)

    assert result.status is ComparabilityStatus.NOT_COMPARABLE
    assert ComparabilityReasonCode.METRIC_DIRECTION_MISMATCH in codes(result)


# ---------------------------------------------------------------------------
# Model awareness and extensibility (PRD 3: general purpose)
# ---------------------------------------------------------------------------


def test_profiles_are_model_aware():
    transformer = get_profile("transformer")
    assert "n_layers" in transformer.config_fields
    assert "max_depth" not in transformer.config_fields

    gbm = get_profile("gradient_boosting")
    assert "max_depth" in gbm.config_fields


def test_unknown_family_falls_back_to_shared_keys(comparability_engine):
    known = make_experiment("exp_known", value=0.9, model_family="gradient_boosting",
                            configuration={"max_depth": 4, "odd_knob": 1})
    alien = make_experiment("exp_alien", value=0.9, model_family="quantum_regressor",
                            configuration={"max_depth": 4, "odd_knob": 1})

    result = comparability_engine.assess(known, alien, metric_name=ACCURACY)

    # Family differs, but nothing in the shared configuration differs.
    assert ComparabilityReasonCode.MODEL_FAMILY_MISMATCH in codes(result)
    assert ComparabilityReasonCode.MODEL_CONFIG_MISMATCH not in codes(result)


def test_register_model_family_is_extensible(comparability_engine):
    register_model_family(
        ModelFamilyProfile(
            family="quantum_regressor",
            config_fields=("qubit_count", "shots"),
            sweep_ladders={"qubit_count": (8, 16, 32)},
            default_metric=ACCURACY,
        )
    )
    try:
        left = make_experiment("exp_q1", value=0.9, model_family="quantum_regressor",
                               configuration={"qubit_count": 8})
        right = make_experiment("exp_q2", value=0.9, model_family="quantum_regressor",
                                configuration={"qubit_count": 32})

        result = comparability_engine.assess(left, right, metric_name=ACCURACY)

        assert ComparabilityReasonCode.MODEL_CONFIG_MISMATCH in codes(result)
        assert get_profile("quantum_regressor").sweep_ladders["qubit_count"] == (8, 16, 32)
    finally:
        # Keep global registry clean for other tests.
        from app.services import comparability as comparability_module

        comparability_module._PROFILES.pop("quantum_regressor", None)


def test_register_model_family_refuses_silent_overwrite():
    with pytest.raises(ValueError, match="already registered"):
        register_model_family(ModelFamilyProfile(family="transformer"))


# ---------------------------------------------------------------------------
# Batch helpers
# ---------------------------------------------------------------------------


def test_assess_all_orders_by_score(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    candidates = [e for e in experiments if e.project_id == baseline.project_id]

    ranked = comparability_engine.assess_all(baseline, candidates, metric_name=ACCURACY)

    scores = [assessment.score for _, assessment in ranked]
    assert scores == sorted(scores, reverse=True)


def test_comparable_subset_excludes_blocked_pairs(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    candidates = [e for e in experiments if e.project_id == baseline.project_id]

    usable = comparability_engine.comparable_subset(baseline, candidates, metric_name=ACCURACY)
    ids = {experiment.id for experiment, _ in usable}

    assert "exp_on_v2" not in ids, "a different dataset version must not be usable evidence"
    assert "exp_test_split" not in ids
    assert "exp_baseline" in ids


def test_failed_run_is_still_evidence(comparability_engine, experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    failed = baseline.model_copy(
        update={"id": "exp_failed", "status": ExperimentStatus.FAILED, "metrics": []}
    )

    result = comparability_engine.assess(baseline, failed, metric_name=ACCURACY)

    # A failed run produced no measurement, so there is nothing to compare.
    assert result.status is ComparabilityStatus.INSUFFICIENT_INFORMATION