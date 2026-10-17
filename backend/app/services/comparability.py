"""Experiment comparability engine (FR-4).

Area: comparability rules.

Decides whether two experiments may be meaningfully compared, using a defined
status plus machine-readable reason codes, and explains the reasoning (FR-8).

Design constraints taken from the PRD:
  * General purpose. Nothing here is specific to transformers or attention
    pruning; model families are described by a pluggable profile table and an
    unknown family degrades gracefully instead of failing.
  * Rules must be extensible and model-aware. New families register through
    :func:`register_model_family`; new dimensions are added to the ``checks``
    tuple in :meth:`ComparabilityEngine.assess` and, if they carry a different
    missing-value policy, to ``MATERIAL_DIMENSIONS``.
  * Never infer causality from similarity, and never invent missing data: an
    absent field yields ``INSUFFICIENT_INFORMATION``, not an assumption.

Dimension policy
----------------
``blocker``     forces ``NOT_COMPARABLE`` (different measurement target).
``major``/``minor``  produce ``COMPARABLE_WITH_CAVEATS``.
``material missing`` forces ``INSUFFICIENT_INFORMATION``.

Precedence when several apply: NOT_COMPARABLE > INSUFFICIENT_INFORMATION >
COMPARABLE_WITH_CAVEATS > COMPARABLE. A known blocker outranks missing
information, because "these differ for a known reason" is more useful than
"we cannot tell".
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from app.models.contracts import (
    ComparabilityAssessment,
    ComparabilityReason,
    ComparabilityReasonCode,
    ComparabilityStatus,
    ExperimentRecord,
    MetricDirection,
    MetricValue,
    Severity,
)

from app.services.evidence import UNTRUSTED_QUALITY_STATUSES

# Severity weights used to derive the ordering score. Documented in
# services/ranking_rules.md. Not probabilities.
SEVERITY_PENALTY: dict[Severity, float] = {
    Severity.BLOCKER: 1.0,
    Severity.MAJOR: 0.15,
    Severity.MINOR: 0.05,
    Severity.INFO: 0.0,
}

# Dimensions whose absence prevents a verdict.
MATERIAL_DIMENSIONS: frozenset[str] = frozenset(
    {
        "dataset_identity",
        "dataset_version",
        "evaluation_split",
        "evaluation_protocol",
        "metric_availability",
    }
)

# Human labels used in explanations and in the UI.
DIMENSION_LABELS: dict[str, str] = {
    "dataset_identity": "dataset",
    "dataset_version": "dataset version",
    "evaluation_split": "evaluation split",
    "evaluation_protocol": "evaluation protocol",
    "metric_availability": "shared metric",
    "metric_direction": "metric direction",
    "model_family": "model family",
    "preprocessing": "preprocessing",
    "training_budget": "training budget",
    "random_seed": "random seed",
    "model_config": "model configuration",
    "evidence_quality": "dataset quality status",
}


@dataclass(frozen=True)
class ModelFamilyProfile:
    """Which configuration fields matter for one model family (FR-4).

    ``config_fields`` are the hyperparameters whose change can alter results.
    ``sweep_ladders`` are the discrete values the recommendation engine suggests
    for each field when a gap is found.
    """

    family: str
    config_fields: tuple[str, ...] = ()
    sweep_ladders: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    default_metric: str | None = None
    default_metric_direction: MetricDirection = MetricDirection.MAXIMIZE
    default_unit_cost_gpu_hours: float | None = None


DEFAULT_PROFILES: tuple[ModelFamilyProfile, ...] = (
    ModelFamilyProfile(
        family="gradient_boosting",
        config_fields=("max_depth", "learning_rate", "n_estimators", "subsample", "colsample_bytree"),
        sweep_ladders={
            "max_depth": (3, 4, 6, 8, 12),
            "learning_rate": (0.01, 0.03, 0.1, 0.3),
            "n_estimators": (100, 300, 600, 1200),
            "subsample": (0.6, 0.8, 1.0),
        },
        default_metric="validation_accuracy",
        default_unit_cost_gpu_hours=0.8,
    ),
    ModelFamilyProfile(
        family="random_forest",
        config_fields=("n_estimators", "max_depth", "min_samples_leaf", "bootstrap"),
        sweep_ladders={
            "max_depth": (None, 8, 16, 32),
            "n_estimators": (100, 300, 600),
            "min_samples_leaf": (1, 2, 5),
        },
        default_metric="validation_accuracy",
        default_unit_cost_gpu_hours=1.2,
    ),
    ModelFamilyProfile(
        family="transformer",
        config_fields=("n_layers", "d_model", "n_heads", "ffn_dim", "dropout", "activation"),
        sweep_ladders={
            "n_layers": (2, 4, 6, 8, 12),
            "d_model": (128, 256, 512, 768),
            "dropout": (0.0, 0.1, 0.2, 0.3),
        },
        default_metric="validation_loss",
        default_metric_direction=MetricDirection.MINIMIZE,
        default_unit_cost_gpu_hours=4.0,
    ),
    ModelFamilyProfile(
        family="cnn",
        config_fields=("conv_kernel_size", "n_conv_layers", "channels", "pooling", "dropout"),
        sweep_ladders={
            "n_conv_layers": (2, 3, 4, 6),
            "conv_kernel_size": (3, 5, 7),
            "dropout": (0.0, 0.1, 0.25, 0.5),
        },
        default_metric="validation_accuracy",
        default_unit_cost_gpu_hours=2.0,
    ),
    ModelFamilyProfile(
        family="linear",
        config_fields=("regularization", "alpha"),
        sweep_ladders={"alpha": (0.0, 0.001, 0.01, 0.1, 1.0)},
        default_metric="validation_accuracy",
        default_unit_cost_gpu_hours=0.1,
    ),
)

_PROFILES: dict[str, ModelFamilyProfile] = {p.family: p for p in DEFAULT_PROFILES}
UNKNOWN_PROFILE = ModelFamilyProfile(family="unknown")


def register_model_family(profile: ModelFamilyProfile, *, replace_existing: bool = False) -> None:
    """Register or extend a model family profile (FR-4 extensibility).

    New families may be added freely. Adding a family changes only which
    configuration fields count as material, never the status vocabulary.
    """
    key = profile.family.strip().lower()
    if key in _PROFILES and not replace_existing:
        raise ValueError(
            f"model family {key!r} already registered; pass replace_existing=True to override"
        )
    _PROFILES[key] = replace(profile, family=key)


def get_profile(model_family: str | None) -> ModelFamilyProfile:
    """Profile for a family; a permissive default for unregistered families.

    An unknown family gets no material config fields, so the engine compares
    only shared keys and reports the rest as INFO rather than guessing which
    hyperparameters matter (PRD 3: must not be hardcoded per family).
    """
    if not model_family:
        return UNKNOWN_PROFILE
    return _PROFILES.get(model_family.strip().lower(), UNKNOWN_PROFILE)


def _identity_key(
    record: ExperimentRecord, *, id_field: str, label_field: str
) -> tuple[str, str] | None:
    """Comparable identity for a record's dataset or dataset version.

    ``dataset_id`` / ``dataset_version_id`` are graph node identifiers while
    ``dataset_name`` / ``dataset_version`` are human labels. The two live in
    different namespaces, so the source is tagged and a label is never compared
    against an id. When two runs offer different kinds of key, equality cannot be
    established and the caller treats the dimension as missing rather than
    guessing that ``"v1"`` means ``"dsv_churn_v1"``.
    """
    node_id = getattr(record, id_field, None)
    label = getattr(record, label_field, None)
    if node_id:
        return ("id", str(node_id).strip().lower())
    if label:
        return ("label", str(label).strip().lower())
    return None


def _identity_mismatch_reason(
    dimension: str,
    *,
    id_field: str,
    label_field: str,
    mismatch_code: ComparabilityReasonCode,
    missing_code: ComparabilityReasonCode,
    left: ExperimentRecord,
    right: ExperimentRecord,
) -> tuple[str, ComparabilityReason | None]:
    """Build the finding for a dimension keyed by :func:`_identity_key`."""
    left_key = _identity_key(left, id_field=id_field, label_field=label_field)
    right_key = _identity_key(right, id_field=id_field, label_field=label_field)
    label = DIMENSION_LABELS[dimension]

    if left_key is None or right_key is None:
        return dimension, ComparabilityReason(
            code=missing_code,
            dimension=dimension,
            severity=Severity.MAJOR,
            message=(
                f"The {label} is not recorded for {_label(left, right)}; comparability "
                "cannot be established without it."
            ),
            left_value=left_key[1] if left_key else None,
            right_value=right_key[1] if right_key else None,
        )

    if left_key[0] != right_key[0]:
        # One run is identified by node id, the other only by label: not proven
        # equal, but not proven different either.
        return dimension, ComparabilityReason(
            code=missing_code,
            dimension=dimension,
            severity=Severity.MAJOR,
            message=(
                f"The two runs identify the {label} differently (one by node id, one by "
                "label), so they cannot be confirmed as the same."
            ),
            left_value=left_key[1],
            right_value=right_key[1],
        )

    if left_key[1] != right_key[1]:
        return dimension, ComparabilityReason(
            code=mismatch_code,
            dimension=dimension,
            severity=Severity.BLOCKER,
            message=(
                f"Different {label}s ({left_key[1]!r} vs {right_key[1]!r}). Results are "
                "measured against different data and are not comparable."
            ),
            left_value=left_key[1],
            right_value=right_key[1],
        )

    return dimension, None


def canonical(value: Any) -> str:
    """Stable string form for arbitrary nested configuration values."""
    try:
        return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _snapshot(value: Any) -> Any:
    """Detach a structured value before storing it on a reason.

    ``left_value`` / ``right_value`` end up in API responses. Handing them the
    caller's own dict would let any consumer that edits an assessment corrupt the
    stored experiment record for every later request in the process.
    """
    if isinstance(value, (dict, list, set)):
        return copy.deepcopy(value)
    return value


def _merge_config(record: ExperimentRecord) -> dict[str, Any]:
    """Model config plus hyperparameters, as one comparable mapping.

    Hyperparameters win on collision: they are the more specific record of what
    the run actually used.
    """
    merged = dict(record.configuration or {})
    merged.update(record.hyperparameters or {})
    return merged


@dataclass
class ComparabilitySettings:
    """Thresholds. Every one is explicit so the evaluation can sweep them."""

    training_budget_relative_tolerance: float = 0.2
    """Fractional step difference that stops counting as the same budget."""

    seed_difference_is_material: bool = False
    """Different seeds are a caveat by default, never a blocker."""

    material_dimensions: frozenset[str] = MATERIAL_DIMENSIONS
    """Dimensions whose absence blocks a verdict rather than qualifying one."""

    profile: ModelFamilyProfile | None = None


#: Reason codes that mean "this field was not recorded".
MISSING_REASON_CODES: frozenset[ComparabilityReasonCode] = frozenset(
    {
        ComparabilityReasonCode.DATASET_MISSING,
        ComparabilityReasonCode.METRIC_MISSING,
        ComparabilityReasonCode.DATASET_VERSION_MISSING,
        ComparabilityReasonCode.EVAL_SPLIT_MISSING,
        ComparabilityReasonCode.EVAL_PROTOCOL_MISSING,
        ComparabilityReasonCode.SEED_MISSING,
        ComparabilityReasonCode.CONFIG_MISSING,
    }
)


class ComparabilityEngine:
    """Rule engine for FR-4. Stateless apart from its settings."""

    def __init__(self, settings: ComparabilitySettings | None = None) -> None:
        self.settings = settings or ComparabilitySettings()

    # -- public API --------------------------------------------------------

    def assess(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        *,
        metric_name: str | None = None,
        ignore_dimensions: Sequence[str] = (),
        ignore_config_fields: Sequence[str] = (),
        settings: ComparabilitySettings | None = None,
    ) -> ComparabilityAssessment:
        """Compare two experiments and explain the verdict.

        ``metric_name`` narrows the check to one metric when the researcher
        cares about a specific one. ``ignore_dimensions`` and
        ``ignore_config_fields`` mark variables the caller intends to change
        (an independent variable), so they are not counted as confounders.
        """
        active = settings or self.settings
        ignored = {d.strip() for d in ignore_dimensions if d and d.strip()}
        ignored_fields = {f.strip() for f in ignore_config_fields if f and f.strip()}

        reasons: list[ComparabilityReason] = []
        matched: list[str] = []
        differing: list[str] = []
        missing: list[str] = []

        checks = (
            self._check_dataset_identity,
            self._check_dataset_version,
            self._check_evaluation_split,
            self._check_evaluation_protocol,
            self._check_metric_availability,
            self._check_metric_direction,
            self._check_model_family,
            self._check_preprocessing,
            self._check_training_budget,
            self._check_random_seed,
            self._check_model_config,
            self._check_evidence_quality,
        )

        for check in checks:
            outcome = check(left, right, metric_name, active, ignored_fields)
            if outcome is None:
                continue
            dimension, reason = outcome
            if dimension in ignored:
                if reason is not None:
                    reasons.append(
                        ComparabilityReason(
                            code=reason.code,
                            dimension=dimension,
                            severity=Severity.INFO,
                            message=(
                                f"{reason.message} Treated as an intentional independent "
                                "variable and excluded from the comparability verdict."
                            ),
                            left_value=reason.left_value,
                            right_value=reason.right_value,
                        )
                    )
                continue
            if reason is None:
                matched.append(dimension)
                continue

            reasons.append(reason)
            if reason.code in MISSING_REASON_CODES:
                # Material gaps become the verdict; non-material ones (a missing
                # seed, missing config) stay advisory.
                if dimension in active.material_dimensions:
                    missing.append(dimension)
            elif reason.severity is Severity.BLOCKER:
                differing.append(dimension)
            elif reason.severity in (Severity.MAJOR, Severity.MINOR):
                differing.append(dimension)

        if not reasons:
            reasons.append(
                ComparabilityReason(
                    code=ComparabilityReasonCode.IDENTICAL,
                    dimension="all_checked_dimensions",
                    severity=Severity.INFO,
                    message=(
                        "Every checked dimension agrees, so the metric difference is "
                        "attributable to the recorded differences."
                    ),
                )
            )

        status = self._resolve_status(reasons, missing)
        score = self._score(reasons)
        confidence = self._confidence(status, missing)

        return ComparabilityAssessment(
            status=status,
            reasons=reasons,
            matched_dimensions=matched,
            differing_dimensions=differing,
            missing_dimensions=missing,
            score=score,
            confidence=confidence,
            left_experiment_id=left.id,
            right_experiment_id=right.id,
            explanation=self.explain(status, reasons, missing),
        )

    def assess_all(
        self,
        baseline: ExperimentRecord,
        candidates: Sequence[ExperimentRecord],
        *,
        metric_name: str | None = None,
        ignore_dimensions: Sequence[str] = (),
        ignore_config_fields: Sequence[str] = (),
    ) -> list[tuple[ExperimentRecord, ComparabilityAssessment]]:
        """Assess ``candidates`` against one baseline, best score first."""
        assessed = [
            (
                candidate,
                self.assess(
                    baseline,
                    candidate,
                    metric_name=metric_name,
                    ignore_dimensions=ignore_dimensions,
                    ignore_config_fields=ignore_config_fields,
                ),
            )
            for candidate in candidates
        ]
        assessed.sort(key=lambda pair: pair[1].score, reverse=True)
        return assessed

    def comparable_subset(
        self,
        baseline: ExperimentRecord,
        candidates: Sequence[ExperimentRecord],
        *,
        metric_name: str | None = None,
        allow_caveats: bool = True,
    ) -> list[tuple[ExperimentRecord, ComparabilityAssessment]]:
        """Candidates usable as evidence for or against ``baseline``."""
        usable = []
        for candidate, assessment in self.assess_all(
            baseline, candidates, metric_name=metric_name
        ):
            if assessment.is_usable_as_baseline:
                if assessment.status is ComparabilityStatus.COMPARABLE or allow_caveats:
                    usable.append((candidate, assessment))
        return usable

    # -- dimension checks --------------------------------------------------

    def _check_dataset_identity(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        dimension, reason = _identity_mismatch_reason(
            "dataset_identity",
            id_field="dataset_id",
            label_field="dataset_name",
            mismatch_code=ComparabilityReasonCode.DATASET_MISMATCH,
            missing_code=ComparabilityReasonCode.DATASET_MISSING,
            left=left,
            right=right,
        )
        return dimension, reason

    def _check_dataset_version(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        dimension, reason = _identity_mismatch_reason(
            "dataset_version",
            id_field="dataset_version_id",
            label_field="dataset_version",
            mismatch_code=ComparabilityReasonCode.DATASET_VERSION_MISMATCH,
            missing_code=ComparabilityReasonCode.DATASET_VERSION_MISSING,
            left=left,
            right=right,
        )
        return dimension, reason

    def _check_evaluation_split(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        left_split = _metric_split(left, metric_name)
        right_split = _metric_split(right, metric_name)
        if not left_split or not right_split:
            return "evaluation_split", ComparabilityReason(
                code=ComparabilityReasonCode.EVAL_SPLIT_MISSING,
                dimension="evaluation_split",
                severity=Severity.MAJOR,
                message=(
                    "Evaluation split is not recorded for "
                    f"{_label(left, right)}; test and validation numbers are not interchangeable."
                ),
                left_value=left_split,
                right_value=right_split,
            )
        if left_split.lower() != right_split.lower():
            return "evaluation_split", ComparabilityReason(
                code=ComparabilityReasonCode.EVAL_SPLIT_MISMATCH,
                dimension="evaluation_split",
                severity=Severity.BLOCKER,
                message=(
                    f"Different evaluation splits ({left_split!r} vs {right_split!r}). "
                    "The runs were scored on different data."
                ),
                left_value=left_split,
                right_value=right_split,
            )
        return "evaluation_split", None

    def _check_evaluation_protocol(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        left_protocol = _metric_protocol(left, metric_name)
        right_protocol = _metric_protocol(right, metric_name)
        if not left_protocol or not right_protocol:
            return "evaluation_protocol", ComparabilityReason(
                code=ComparabilityReasonCode.EVAL_PROTOCOL_MISSING,
                dimension="evaluation_protocol",
                severity=Severity.MAJOR,
                message=(
                    "Evaluation protocol is not recorded for "
                    f"{_label(left, right)}; identical splits can still use different protocols."
                ),
                left_value=left_protocol,
                right_value=right_protocol,
            )
        if canonical(left_protocol) != canonical(right_protocol):
            return "evaluation_protocol", ComparabilityReason(
                code=ComparabilityReasonCode.EVAL_PROTOCOL_MISMATCH,
                dimension="evaluation_protocol",
                severity=Severity.MAJOR,
                message=(
                    f"Evaluation protocols differ ({left_protocol!r} vs {right_protocol!r}). "
                    "Compare with caution and state the protocol in any conclusion."
                ),
                left_value=left_protocol,
                right_value=right_protocol,
            )
        return "evaluation_protocol", None

    def _check_metric_availability(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        if metric_name:
            left_metric = left.metric(metric_name)
            right_metric = right.metric(metric_name)
            absent = [
                exp.id
                for exp, metric in ((left, left_metric), (right, right_metric))
                if metric is None or metric.value is None
            ]
            if absent:
                return "metric_availability", ComparabilityReason(
                    code=ComparabilityReasonCode.METRIC_MISSING,
                    dimension="metric_availability",
                    severity=Severity.MAJOR,
                    message=(
                        f"Metric {metric_name!r} has no recorded value for "
                        f"{', '.join(absent)}; no comparison is possible."
                    ),
                    left_value=left_metric.value if left_metric else None,
                    right_value=right_metric.value if right_metric else None,
                )
            return "metric_availability", None

        # No specific metric requested: the two runs must at least measure the
        # same thing, with a value on both sides.
        if not left.metrics and not right.metrics:
            return "metric_availability", ComparabilityReason(
                code=ComparabilityReasonCode.METRIC_MISSING,
                dimension="metric_availability",
                severity=Severity.MAJOR,
                message=(
                    f"Neither {_label(left, right)} records any metric, so they cannot be "
                    "compared on a measurement."
                ),
            )
        if not left.metrics or not right.metrics:
            silent = left.id if not left.metrics else right.id
            return "metric_availability", ComparabilityReason(
                code=ComparabilityReasonCode.METRIC_MISSING,
                dimension="metric_availability",
                severity=Severity.MAJOR,
                message=(
                    f"{silent} records no metric, so what it measured is unknown."
                ),
                left_value=left.metric_names() or None,
                right_value=right.metric_names() or None,
            )

        # Name overlap is checked before value overlap: two runs that both name
        # "validation_accuracy" measured the same thing even if one value is
        # blank, which is a missing measurement rather than a disagreement.
        left_names = {m.name.strip().lower() for m in left.metrics}
        right_names = {m.name.strip().lower() for m in right.metrics}
        shared_names = left_names & right_names

        if not shared_names:
            return "metric_availability", ComparabilityReason(
                code=ComparabilityReasonCode.METRIC_MISMATCH,
                dimension="metric_availability",
                severity=Severity.BLOCKER,
                message=(
                    f"No metric is shared by both runs ({', '.join(sorted(left_names))} "
                    f"vs {', '.join(sorted(right_names))}); they answer different questions."
                ),
            )

        unmeasured = [
            name for name in sorted(shared_names)
            if _metric_value(left, name) is None or _metric_value(right, name) is None
        ]
        if unmeasured:
            return "metric_availability", ComparabilityReason(
                code=ComparabilityReasonCode.METRIC_MISSING,
                dimension="metric_availability",
                severity=Severity.MAJOR,
                message=(
                    f"{', '.join(unmeasured)} is named by both runs but has no value on at "
                    "least one side."
                ),
                left_value={n: _metric_value(left, n) for n in sorted(shared_names)},
                right_value={n: _metric_value(right, n) for n in sorted(shared_names)},
            )
        return "metric_availability", None

    def _check_metric_direction(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        left_metrics = _metrics_of_interest(left, metric_name)
        right_metrics = _metrics_of_interest(right, metric_name)
        for name in sorted(set(left_metrics) & set(right_metrics)):
            left_direction = left_metrics[name].direction
            right_direction = right_metrics[name].direction
            if left_direction is not right_direction:
                return "metric_direction", ComparabilityReason(
                    code=ComparabilityReasonCode.METRIC_DIRECTION_MISMATCH,
                    dimension="metric_direction",
                    severity=Severity.BLOCKER,
                    message=(
                        f"Metric {name!r} is optimised in opposite directions "
                        f"({left_direction.value} vs {right_direction.value}); "
                        "higher is better for one run and not the other."
                    ),
                    left_value=left_direction.value,
                    right_value=right_direction.value,
                )
        return "metric_direction", None

    def _check_model_family(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        left_family = (left.model_family or "").strip().lower()
        right_family = (right.model_family or "").strip().lower()
        if not left_family or not right_family or left_family == right_family:
            return "model_family", None
        return "model_family", ComparabilityReason(
            code=ComparabilityReasonCode.MODEL_FAMILY_MISMATCH,
            dimension="model_family",
            severity=Severity.MAJOR,
            message=(
                f"Different model families ({left_family!r} vs {right_family!r}). "
                "The gap reflects architecture as well as configuration."
            ),
            left_value=left_family,
            right_value=right_family,
        )

    def _check_preprocessing(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        left_pp = left.preprocessing or {}
        right_pp = right.preprocessing or {}
        if not left_pp and not right_pp:
            return "preprocessing", None
        if canonical(left_pp) != canonical(right_pp):
            return "preprocessing", ComparabilityReason(
                code=ComparabilityReasonCode.PREPROCESSING_MISMATCH,
                dimension="preprocessing",
                severity=Severity.MAJOR,
                message=(
                    "Preprocessing differs between the runs; feature distributions "
                    "are not the same input distribution."
                ),
                left_value=_snapshot(left_pp),
                right_value=_snapshot(right_pp),
            )
        return "preprocessing", None

    def _check_training_budget(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        left_steps = left.training_steps
        right_steps = right.training_steps
        # A declared training ceiling counts as the budget when no realised step
        # count was recorded.
        if left_steps is None:
            left_steps = left.max_training_steps
        if right_steps is None:
            right_steps = right.max_training_steps
        if left_steps is None or right_steps is None:
            return "training_budget", None
        if left_steps == 0 or right_steps == 0:
            if left_steps != right_steps:
                return "training_budget", ComparabilityReason(
                    code=ComparabilityReasonCode.TRAINING_BUDGET_MISMATCH,
                    dimension="training_budget",
                    severity=Severity.MAJOR,
                    message=(
                        f"Training budgets differ ({left_steps} vs {right_steps} steps); "
                        "the runs had different amounts of optimisation."
                    ),
                    left_value=left_steps,
                    right_value=right_steps,
                )
            return "training_budget", None
        difference = abs(left_steps - right_steps) / max(left_steps, right_steps)
        if difference > settings.training_budget_relative_tolerance:
            return "training_budget", ComparabilityReason(
                code=ComparabilityReasonCode.TRAINING_BUDGET_MISMATCH,
                dimension="training_budget",
                severity=Severity.MAJOR,
                message=(
                    f"Training budgets differ by {difference:.0%} "
                    f"({left_steps} vs {right_steps} steps, tolerance "
                    f"{settings.training_budget_relative_tolerance:.0%}); "
                    "a longer run is not automatically a better one."
                ),
                left_value=left_steps,
                right_value=right_steps,
            )
        return "training_budget", None

    def _check_random_seed(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        if left.random_seed is None or right.random_seed is None:
            return "random_seed", ComparabilityReason(
                code=ComparabilityReasonCode.SEED_MISSING,
                dimension="random_seed",
                severity=Severity.MINOR,
                message=(
                    "Random seed is not recorded for "
                    f"{_label(left, right)}; run-to-run variance is unknown, so a small "
                    "difference between the runs may be noise."
                ),
                left_value=left.random_seed,
                right_value=right.random_seed,
            )
        if left.random_seed != right.random_seed:
            return "random_seed", ComparabilityReason(
                code=ComparabilityReasonCode.RANDOM_SEED_MISMATCH,
                dimension="random_seed",
                severity=Severity.MINOR,
                message=(
                    f"Different random seeds ({left.random_seed} vs {right.random_seed}). "
                    "This is acceptable and even useful, but part of the gap is seed variance."
                ),
                left_value=left.random_seed,
                right_value=right.random_seed,
            )
        return "random_seed", None

    def _check_model_config(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        profile = settings.profile or get_profile(left.model_family)
        left_config = _merge_config(left)
        right_config = _merge_config(right)
        if not left_config or not right_config:
            return "model_config", ComparabilityReason(
                code=ComparabilityReasonCode.CONFIG_MISSING,
                dimension="model_config",
                severity=Severity.MINOR,
                message=(
                    "Model configuration is not recorded for "
                    f"{_label(left, right)}; configuration differences cannot be ruled out."
                ),
                left_value=_snapshot(left_config),
                right_value=_snapshot(right_config),
            )

        fields = [f for f in profile.config_fields if f not in ignored_fields]
        # For unregistered families, fall back to the keys both runs recorded.
        fields = fields or sorted(set(left_config) & set(right_config))

        differing = [
            field_name
            for field_name in fields
            if canonical(left_config.get(field_name)) != canonical(right_config.get(field_name))
        ]
        if not differing:
            return "model_config", None

        # A single differing field is the expected shape of a controlled
        # ablation, so it reads as a caveat rather than an unexplained gap.
        severity = Severity.MINOR if len(differing) == 1 else Severity.MAJOR
        return "model_config", ComparabilityReason(
            code=ComparabilityReasonCode.MODEL_CONFIG_MISMATCH,
            dimension="model_config",
            severity=severity,
            message=(
                f"Configuration differs on {', '.join(sorted(differing))}. "
                + (
                    "This looks like a controlled single-variable change."
                    if severity is Severity.MINOR
                    else "Several parameters changed at once, so the effect is confounded."
                )
            ),
            left_value={k: _snapshot(left_config.get(k)) for k in sorted(differing)},
            right_value={k: _snapshot(right_config.get(k)) for k in sorted(differing)},
        )

    def _check_evidence_quality(
        self,
        left: ExperimentRecord,
        right: ExperimentRecord,
        metric_name: str | None,
        settings: ComparabilitySettings,
        ignored_fields: set[str],
    ) -> tuple[str, ComparabilityReason | None]:
        disputed = [
            exp.id
            for exp in (left, right)
            if exp.dataset_quality_status in UNTRUSTED_QUALITY_STATUSES
        ]
        if not disputed:
            return "evidence_quality", None
        grades = sorted(
            {
                exp.dataset_quality_status.value
                for exp in (left, right)
                if exp.dataset_quality_status is not None
            }
        )
        return "evidence_quality", ComparabilityReason(
            code=ComparabilityReasonCode.DISPUTED_EVIDENCE,
            dimension="evidence_quality",
            severity=Severity.MINOR,
            message=(
                f"{', '.join(disputed)} run(s) sit on a dataset version that is not trusted "
                f"({', '.join(grades)}). The comparison is structurally valid but its data "
                "may be wrong or awaiting review."
            ),
            left_value=_snapshot(left.dataset_quality_status.value if left.dataset_quality_status else None),
            right_value=_snapshot(right.dataset_quality_status.value if right.dataset_quality_status else None),
        )

    # -- aggregation -------------------------------------------------------

    @staticmethod
    def _resolve_status(
        reasons: Sequence[ComparabilityReason], missing: Sequence[str]
    ) -> ComparabilityStatus:
        # A known blocker outranks missing information: "these differ for a known
        # reason" is more actionable than "we cannot tell".
        if any(r.severity is Severity.BLOCKER for r in reasons):
            return ComparabilityStatus.NOT_COMPARABLE
        if missing:
            return ComparabilityStatus.INSUFFICIENT_INFORMATION
        if any(r.severity in (Severity.MAJOR, Severity.MINOR) for r in reasons):
            return ComparabilityStatus.COMPARABLE_WITH_CAVEATS
        return ComparabilityStatus.COMPARABLE

    @staticmethod
    def _score(reasons: Sequence[ComparabilityReason]) -> float:
        """Ordering aid in ``[0, 1]``. Not a probability."""
        penalty = sum(SEVERITY_PENALTY.get(r.severity, 0.0) for r in reasons)
        return round(max(0.0, 1.0 - penalty), 4)

    @staticmethod
    def _confidence(
        status: ComparabilityStatus, missing: Sequence[str]
    ) -> str:
        if status is ComparabilityStatus.INSUFFICIENT_INFORMATION:
            return "low"
        if missing:
            return "low"
        if status is ComparabilityStatus.COMPARABLE:
            return "high"
        return "medium"

    def explain(
        self,
        status: ComparabilityStatus,
        reasons: Sequence[ComparabilityReason],
        missing: Sequence[str],
    ) -> str:
        """One deterministic paragraph a researcher can quote in a write-up."""
        if status is ComparabilityStatus.COMPARABLE:
            return (
                "The runs share dataset, dataset version, evaluation split, protocol, "
                "metric and relevant configuration, so the metric difference is "
                "attributable to the recorded differences."
            )
        if status is ComparabilityStatus.NOT_COMPARABLE:
            blockers = [r for r in reasons if r.severity is Severity.BLOCKER]
            head = blockers[0].message if blockers else "A required comparison field differs."
            return f"Not comparable. {head}"
        if status is ComparabilityStatus.INSUFFICIENT_INFORMATION:
            labels = ", ".join(DIMENSION_LABELS.get(d, d) for d in missing) or "required fields"
            return (
                f"Cannot decide: {labels} not recorded for at least one run. "
                "Re-import the missing field rather than assuming it was the same."
            )
        caveats = [r for r in reasons if r.severity in (Severity.MAJOR, Severity.MINOR)]
        if not caveats:
            return "Comparable on every dimension that was checked."
        detail = " ".join(r.message for r in caveats)
        return f"Comparable with caveats. {detail}"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _label(left: ExperimentRecord, right: ExperimentRecord) -> str:
    return f"{left.id} and {right.id}"


def _metrics_of_interest(
    record: ExperimentRecord, metric_name: str | None
) -> dict[str, MetricValue]:
    if metric_name:
        metric = record.metric(metric_name)
        return {metric_name: metric} if metric else {}
    return {m.name: m for m in record.metrics}


def _metric_split(record: ExperimentRecord, metric_name: str | None) -> str | None:
    metrics = _metrics_of_interest(record, metric_name)
    for metric in metrics.values():
        if metric.evaluation_split:
            return metric.evaluation_split
    return None


def _metric_protocol(record: ExperimentRecord, metric_name: str | None) -> Any | None:
    metrics = _metrics_of_interest(record, metric_name)
    for metric in metrics.values():
        if metric.evaluation_protocol is not None:
            return metric.evaluation_protocol
    return None


def _metric_value(record: ExperimentRecord, name: str) -> float | None:
    metric = record.metric(name)
    return metric.value if metric is not None else None