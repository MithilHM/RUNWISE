"""Recommendation engine: gaps, candidates, budget, ranking (FR-5).

Area: the recommendation engine.

Turns experiment history into a defensible next step. The pipeline follows PRD
section 11 in order, and each stage is separately testable:

  1. parse the objective            -> :class:`ObjectiveContext`
  2-3. retrieve and characterise    -> trusted experiment set
  4. assess comparability           -> :mod:`app.services.comparability`
  5. exclude or qualify corrected evidence -> :func:`partition_by_trust`
  6. identify gaps                  -> :func:`identify_gaps`
  7. generate candidates            -> :func:`generate_candidates`
  8. enforce the compute budget     -> :func:`apply_budget`
  9. rank with an explainable heuristic -> :func:`score_candidate`
  10. return structured output with supporting evidence

Trust rules that shape the code:
  * Every returned identifier was retrieved from the graph in this call.
  * A recommendation is INFERRED, never OBSERVED. It is a proposal.
  * The score is a weighted heuristic, exposed with its components so a
    researcher can disagree with a term instead of with an opaque number.
    It is never presented as a probability (PRD 5).
  * An unknown compute cost is reported as unknown. It is never defaulted into
    the ranking as though it were measured.
"""

from __future__ import annotations

import statistics
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from app.models.contracts import (
    Candidate,
    CandidateKind,
    ComparabilityAssessment,
    CORRECTION_STATUSES_BLOCKING_TRUST,
    DatasetVersionSummary,
    Decision,
    ExcludedEvidence,
    EvidenceStatus,
    ExperimentRecord,
    ExperimentStatus,
    MetricDirection,
    Recommendation,
    RecommendationDecision,
    RecommendationRequest,
    RecommendationSet,
    ScoreComponent,
    ValidationCriterion,
)
from app.ports.graph import GraphService
from app.services.comparability import (
    ComparabilityEngine,
    ModelFamilyProfile,
    get_profile,
)
from app.services.evidence import partition_by_trust

# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingWeights:
    """The six factors named in PRD 5, with the weights this build uses.

    Sum is 1.0 so a weighted sum stays in ``[0, 1]``. Changing a weight is a
    product decision; the evaluation harness should sweep them rather than assume
    these are optimal.
    """

    relevance: float = 0.22
    evidence_quality: float = 0.20
    information_gain: float = 0.22
    cost_efficiency: float = 0.14
    uncertainty_reduction: float = 0.10
    feasibility: float = 0.12

    def __post_init__(self) -> None:
        total = sum(self.as_dict().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"RankingWeights must sum to 1.0, got {total}")

    def as_dict(self) -> dict[str, float]:
        return {
            "relevance": self.relevance,
            "evidence_quality": self.evidence_quality,
            "information_gain": self.information_gain,
            "cost_efficiency": self.cost_efficiency,
            "uncertainty_reduction": self.uncertainty_reduction,
            "feasibility": self.feasibility,
        }


@dataclass
class RankingConfig:
    weights: RankingWeights = field(default_factory=RankingWeights)
    max_sweep_fields: int = 3
    """Cap on how many hyperparameters generate candidates in one request."""
    max_untested_values_per_field: int = 1
    min_cluster_for_strong_evidence: int = 3
    underexplored_field_threshold: int = 2
    """A field seen at most this many times is treated as underexplored."""
    seed_replication_target: int = 3


DEFAULT_CONTROLLED_DIMENSIONS: tuple[str, ...] = (
    "dataset_version",
    "evaluation_split",
    "evaluation_protocol",
    "preprocessing",
    "random_seed",
    "training_budget",
)


# ---------------------------------------------------------------------------
# Objective parsing
# ---------------------------------------------------------------------------

_TOKEN_SPLIT = str.maketrans({",": " ", ";": " ", "/": " ", "\n": " "})


def _has_metric_value(record: ExperimentRecord, metric_name: str) -> bool:
    """True only when the metric exists *and* carries a value.

    A CSV that imported the column but left it blank produces a metric with a
    null value. That is an unrecorded measurement, which is what the comparability
    engine treats as missing, so gap detection must agree.
    """
    metric = record.metric(metric_name)
    return metric is not None and metric.value is not None


def _measured(pool: Sequence[ExperimentRecord], metric_name: str | None) -> list[ExperimentRecord]:
    if not metric_name:
        return list(pool)
    return [e for e in pool if _has_metric_value(e, metric_name)]


def _tokens(text: str) -> set[str]:
    return {t for t in text.lower().translate(_TOKEN_SPLIT).split() if len(t) > 2}


@dataclass
class ObjectiveContext:
    """What the objective is actually asking for."""

    objective: str
    metric_name: str | None
    metric_direction: MetricDirection
    model_family: str | None
    dataset_id: str | None
    dataset_version_id: str | None
    tokens: set[str]
    resolution_notes: list[str] = field(default_factory=list)


def parse_objective(
    objective: str,
    experiments: Sequence[ExperimentRecord],
    dataset_versions: Sequence[DatasetVersionSummary],
    *,
    request: RecommendationRequest | None = None,
) -> ObjectiveContext:
    """Extract the target metric, family and dataset from an objective.

    Deterministic by design: an optional LLM may rephrase this later, but it may
    not choose the target, because a wrong target makes every downstream number
    meaningless.
    """
    notes: list[str] = []
    tokens = _tokens(objective)
    lowered = objective.lower()

    metric_counts: dict[str, int] = {}
    for experiment in experiments:
        for name in experiment.metric_names():
            metric_counts[name] = metric_counts.get(name, 0) + 1

    metric_name = None
    if request and request.target_metric:
        metric_name = request.target_metric
    else:
        # A metric named in the question wins over the most common one.
        for name in sorted(metric_counts, key=lambda n: (-metric_counts[n], n)):
            if name.lower() in lowered:
                metric_name = name
                notes.append(f"Target metric {name!r} was named in the objective.")
                break

    families: dict[str, int] = {}
    for experiment in experiments:
        family = (experiment.model_family or "").strip().lower()
        if family:
            families[family] = families.get(family, 0) + 1

    model_family = None
    if request and request.model_family:
        model_family = request.model_family.strip().lower()
    else:
        for family in sorted(families):
            if family in lowered or family.replace("_", " ") in lowered:
                model_family = family
                notes.append(f"Model family {family!r} was named in the objective.")
                break

    if model_family is None and families:
        model_family = max(sorted(families), key=lambda f: families[f])
        notes.append(
            f"No family named in the objective; using the most used family "
            f"({model_family!r}, {families[model_family]} runs)."
        )

    direction = MetricDirection.MAXIMIZE
    if model_family:
        profile = get_profile(model_family)
        if profile.default_metric_direction is MetricDirection.MINIMIZE:
            direction = MetricDirection.MINIMIZE

    if metric_name is None and model_family:
        profile = get_profile(model_family)
        metric_name = profile.default_metric
        if metric_name:
            notes.append(
                f"No metric named; defaulting to {metric_name!r} for family "
                f"{model_family!r}."
            )

    if metric_name is None:
        if metric_counts:
            metric_name = max(sorted(metric_counts), key=lambda n: metric_counts[n])
            notes.append(
                f"No default metric for this family; using the most recorded metric "
                f"({metric_name!r})."
            )
        else:
            notes.append("No metric could be resolved from the objective or history.")

    dataset_id = request.dataset_id if request else None
    dataset_version_id = None
    if dataset_versions:
        newest = max(dataset_versions, key=lambda v: (_natural_key(v.version), v.id))
        dataset_version_id = newest.id

    if direction is MetricDirection.MINIMIZE and "loss" not in (metric_name or ""):
        notes.append(
            "Optimisation direction is minimize for this family; check the metric "
            "definition before ranking runs."
        )

    return ObjectiveContext(
        objective=objective,
        metric_name=metric_name,
        metric_direction=direction,
        model_family=model_family,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        tokens=tokens,
        resolution_notes=notes,
    )


# ---------------------------------------------------------------------------
# Gaps
# ---------------------------------------------------------------------------


@dataclass
class Gap:
    """Something previous experiments have not established."""

    kind: CandidateKind
    field_name: str | None
    description: str
    priority: float
    observed_values: list[Any] = field(default_factory=list)
    untested_values: list[Any] = field(default_factory=list)
    detail: dict[str, Any] = field(default_factory=dict)


def identify_gaps(
    trusted: Sequence[ExperimentRecord],
    context: ObjectiveContext,
    dataset_versions: Sequence[DatasetVersionSummary],
    *,
    config: RankingConfig | None = None,
) -> list[Gap]:
    """Find untested levers (PRD 11, step 6)."""
    cfg = config or RankingConfig()
    if not trusted:
        return []

    family = context.model_family
    family_runs = [e for e in trusted if (e.model_family or "").lower() == family] if family else list(trusted)
    profile = get_profile(family) if family else ModelFamilyProfile(family="unknown")
    gaps: list[Gap] = []

    if context.metric_name:
        measured_runs = [
            e for e in family_runs if _has_metric_value(e, context.metric_name)
        ]
        unmeasured = [e for e in family_runs if e not in measured_runs]
        if family_runs and unmeasured:
            gaps.append(
                Gap(
                    kind=CandidateKind.EVALUATE_MISSING_METRIC,
                    field_name=context.metric_name,
                    description=(
                        f"{len(unmeasured)} of {len(family_runs)} runs did not record a value "
                        f"for {context.metric_name!r}, so their results cannot be compared."
                    ),
                    priority=0.55,
                    detail={"runs_missing": [e.id for e in unmeasured]},
                )
            )

    seeds = {e.random_seed for e in family_runs if e.random_seed is not None}
    if family_runs and len(seeds) < cfg.seed_replication_target:
        gaps.append(
            Gap(
                kind=CandidateKind.REPLICATE_SEED,
                field_name="random_seed",
                description=(
                    f"Only {len(seeds)} distinct seed(s) across {len(family_runs)} runs; "
                    "run-to-run variance is unknown, so any improvement may be noise."
                ),
                priority=0.9,
                observed_values=sorted(seeds),
                untested_values=list(range(cfg.seed_replication_target + 1)),
            )
        )

    for field_name, ladder in sorted(profile.sweep_ladders.items()):
        observed = _observed_config_values(family_runs, field_name)
        if not observed:
            continue
        untested = [_normalise_value(v) for v in ladder if _normalise_value(v) not in observed]
        if not untested:
            continue
        underexplored = len(observed) <= cfg.underexplored_field_threshold
        gaps.append(
            Gap(
                kind=CandidateKind.SWEEP_PARAMETER,
                field_name=field_name,
                description=(
                    f"{field_name} has {len(observed)} observed value(s) "
                    f"({', '.join(str(v) for v in sorted(observed, key=str))}) and "
                    f"{len(untested)} untested value(s) in the family's ladder."
                ),
                priority=0.8 if underexplored else 0.6,
                observed_values=sorted(observed, key=str),
                untested_values=untested,
                detail={"underexplored": underexplored, "observed_count": len(observed)},
            )
        )

    if family_runs:
        used_versions = {e.dataset_version_id for e in family_runs if e.dataset_version_id}
        for version in dataset_versions:
            if version.id not in used_versions:
                gaps.append(
                    Gap(
                        kind=CandidateKind.TRY_DATASET_VERSION,
                        field_name="dataset_version",
                        description=(
                            f"Dataset version {version.version!r} has never been used with "
                            f"{family!r}; its effect on the metric is unmeasured."
                        ),
                        priority=0.7 if version.id == context.dataset_version_id else 0.55,
                        detail={"dataset_version_id": version.id, "dataset_id": version.dataset_id},
                    )
                )

    other_families = {
        (e.model_family or "").strip().lower()
        for e in trusted
        if e.model_family and (e.model_family or "").strip().lower() != family
    }
    if family and other_families:
        gaps.append(
            Gap(
                kind=CandidateKind.NEW_MODEL_FAMILY,
                field_name="model_family",
                description=(
                    f"Family {family!r} has not been compared against "
                    f"{', '.join(sorted(other_families))} on the same data."
                ),
                priority=0.5,
                detail={"alternatives": sorted(other_families)},
            )
        )

    gaps.sort(key=lambda g: g.priority, reverse=True)
    return gaps


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


@dataclass
class CandidateContext:
    """Shared state for candidate construction."""

    baseline: ExperimentRecord | None
    cluster: list[ExperimentRecord]
    comparabilities: dict[str, ComparabilityAssessment]
    profile: ModelFamilyProfile
    context: ObjectiveContext
    config: RankingConfig
    excluded: list[ExcludedEvidence]


def generate_candidates(
    gaps: Sequence[Gap],
    ctx: CandidateContext,
) -> list[Candidate]:
    """Turn gaps into concrete, exportable proposals (PRD 11, step 7)."""
    candidates: list[Candidate] = []
    sweep_fields_used = 0

    for gap in gaps:
        if gap.kind is CandidateKind.SWEEP_PARAMETER:
            if sweep_fields_used >= ctx.config.max_sweep_fields:
                continue
            sweep_fields_used += 1
            candidates.extend(_sweep_candidates(gap, ctx))
        elif gap.kind is CandidateKind.REPLICATE_SEED:
            candidates.append(_seed_candidate(gap, ctx))
        elif gap.kind is CandidateKind.TRY_DATASET_VERSION:
            candidates.append(_dataset_version_candidate(gap, ctx))
        elif gap.kind is CandidateKind.NEW_MODEL_FAMILY:
            candidates.extend(_new_family_candidates(gap, ctx))
        elif gap.kind is CandidateKind.EVALUATE_MISSING_METRIC:
            candidates.append(_missing_metric_candidate(gap, ctx))

    return [c for c in candidates if c is not None]


def _controlled_variables(
    independent: Sequence[str], ctx: CandidateContext, *, dataset_version_id: str | None = None
) -> list[str]:
    """What stays fixed so the change is interpretable (FR-5)."""
    independent_set = {i.strip() for i in independent}
    controlled = []
    for dimension in DEFAULT_CONTROLLED_DIMENSIONS:
        if dimension in independent_set:
            continue
        if dimension == "dataset_version" and dataset_version_id:
            # The candidate deliberately moves the dataset, so it is controlled by
            # being held at one specific version.
            continue
        controlled.append(dimension)
    return controlled


def _base_configuration(ctx: CandidateContext) -> dict[str, Any]:
    if ctx.baseline is None:
        return {}
    merged = dict(ctx.baseline.configuration or {})
    merged.update(ctx.baseline.hyperparameters or {})
    return merged


def _metrics_for(ctx: CandidateContext) -> list[str]:
    if ctx.context.metric_name:
        return [ctx.context.metric_name]
    return []


def _estimate_cost(ctx: CandidateContext) -> tuple[float | None, str | None, list[str]]:
    """Median runtime of comparable runs, else the family default, else unknown.

    PRD 5: mark the uncertainty or ask for input rather than inventing a number.
    """
    uncertainties: list[str] = []
    costs = [
        e.compute_cost_gpu_hours
        for e in ctx.cluster
        if e.compute_cost_gpu_hours is not None and e.compute_cost_gpu_hours > 0
    ]
    if costs:
        median = round(float(statistics.median(costs)), 4)
        return median, f"median runtime of {len(costs)} comparable run(s)", uncertainties

    if ctx.profile.default_unit_cost_gpu_hours is not None:
        uncertainties.append(
            "No comparable run recorded a cost; the estimate uses the default for family "
            f"{ctx.profile.family!r} and not a measurement."
        )
        return ctx.profile.default_unit_cost_gpu_hours, "model_family_profile_default", uncertainties

    uncertainties.append(
        "Compute estimate unavailable: no comparable run recorded a cost and the family has "
        "no default. Supply an estimate before planning this run."
    )
    return None, None, uncertainties


def _validation_criteria(ctx: CandidateContext, gap: Gap) -> list[ValidationCriterion]:
    """Deterministic stopping rules, flagged as product defaults (FR-5)."""
    metric = ctx.context.metric_name
    criteria: list[ValidationCriterion] = []

    if gap.kind is CandidateKind.REPLICATE_SEED:
        criteria.append(
            ValidationCriterion(
                kind="replication",
                description=(
                    f"Repeat the configuration under at least {ctx.config.seed_replication_target} "
                    "distinct seeds and report the spread, not only the best run."
                ),
                threshold=float(ctx.config.seed_replication_target),
                metric_name=metric,
            )
        )
        criteria.append(
            ValidationCriterion(
                kind="stopping",
                metric_name=metric,
                description=(
                    "Stop once the standard deviation across seeds is smaller than the "
                    "difference you are trying to detect; otherwise the result is unresolvable."
                ),
            )
        )
        return criteria

    criteria.append(
        ValidationCriterion(
            kind="baseline_parity",
            metric_name=metric,
            description=(
                "Re-run the baseline under identical conditions in the same session so the "
                "comparison is not confounded by drift between runs."
            ),
        )
    )
    if gap.kind is CandidateKind.SWEEP_PARAMETER:
        criteria.append(
            ValidationCriterion(
                kind="single_variable",
                description=(
                    f"Change only {gap.field_name}. If any other factor moves, the result "
                    "cannot be attributed to this parameter."
                ),
            )
        )
    criteria.append(
        ValidationCriterion(
            kind="replication",
            metric_name=metric,
            description=(
                "Confirm any improvement on at least three seeds before treating it as real."
            ),
        )
    )
    criteria.append(
        ValidationCriterion(
            kind="stopping",
            metric_name=metric,
            description=(
                "Stop when the change stops producing a difference larger than seed variance, "
                "or after the planned values are exhausted."
            ),
        )
    )
    return criteria


def _sweep_candidates(gap: Gap, ctx: CandidateContext) -> list[Candidate]:
    if gap.field_name is None:
        return []
    if ctx.baseline is None:
        return []

    configuration = _base_configuration(ctx)
    previous = configuration.get(gap.field_name)
    supporting = [e.id for e in ctx.cluster][:10]

    out: list[Candidate] = []
    for value in gap.untested_values[: ctx.config.max_untested_values_per_field]:
        proposed = dict(configuration)
        proposed[gap.field_name] = value
        cost, cost_source, cost_uncertainties = _estimate_cost(ctx)
        uncertainties = list(cost_uncertainties)
        if len(ctx.cluster) < 2:
            uncertainties.append(
                "Only one comparable baseline is available, so no variance estimate exists."
            )
        out.append(
            Candidate(
                kind=CandidateKind.SWEEP_PARAMETER,
                hypothesis=(
                    f"Changing {gap.field_name} from {previous!r} to {value!r}, holding the "
                    "dataset version, evaluation protocol and all other parameters fixed, "
                    f"will move {ctx.context.metric_name or 'the target metric'}."
                ),
                configuration=proposed,
                independent_variables=[gap.field_name],
                controlled_variables=_controlled_variables([gap.field_name], ctx),
                metrics=_metrics_for(ctx),
                baseline_experiment_ids=[ctx.baseline.id],
                supporting_experiment_ids=supporting,
                estimated_gpu_hours=cost,
                estimate_source=cost_source,
                uncertainties=uncertainties,
                validation_criteria=_validation_criteria(ctx, gap),
                model_family=ctx.baseline.model_family,
                dataset_version_id=ctx.baseline.dataset_version_id,
                independent_variable_values={gap.field_name: value},
            )
        )
    return out


def _seed_candidate(gap: Gap, ctx: CandidateContext) -> Candidate | None:
    if ctx.baseline is None:
        return None
    configuration = _base_configuration(ctx)
    existing = [s for s in gap.observed_values if s is not None]
    next_seed = (max(existing) + 1) if existing else 0
    proposed = dict(configuration)
    proposed["random_seed"] = next_seed

    cost, cost_source, cost_uncertainties = _estimate_cost(ctx)
    metric = ctx.context.metric_name
    return Candidate(
        kind=CandidateKind.REPLICATE_SEED,
        hypothesis=(
            f"Re-running the configuration of {ctx.baseline.id} under additional seeds "
            f"will show that any gap to the best run is larger than seed variance for "
            f"{metric or 'the target metric'}."
        ),
        configuration=proposed,
        independent_variables=["random_seed"],
        controlled_variables=_controlled_variables(["random_seed"], ctx),
        metrics=_metrics_for(ctx),
        baseline_experiment_ids=[ctx.baseline.id],
        supporting_experiment_ids=[e.id for e in ctx.cluster][:10],
        estimated_gpu_hours=cost,
        estimate_source=cost_source,
        uncertainties=cost_uncertainties,
        validation_criteria=_validation_criteria(ctx, gap),
        model_family=ctx.baseline.model_family,
        dataset_version_id=ctx.baseline.dataset_version_id,
        independent_variable_values={"random_seed": next_seed},
    )


def _dataset_version_candidate(gap: Gap, ctx: CandidateContext) -> Candidate | None:
    if ctx.baseline is None:
        return None
    version_id = gap.detail.get("dataset_version_id")
    if not version_id:
        return None

    proposed = _base_configuration(ctx)
    cost, cost_source, cost_uncertainties = _estimate_cost(ctx)
    uncertainties = list(cost_uncertainties)
    uncertainties.append(
        "This measures the dataset version as a confounded change; it does not isolate which "
        "records differ."
    )
    return Candidate(
        kind=CandidateKind.TRY_DATASET_VERSION,
        hypothesis=(
            f"Training on dataset version {version_id!r} with an unchanged configuration "
            f"will change {ctx.context.metric_name or 'the target metric'} by more than the "
            "seed variance of the existing runs."
        ),
        configuration=proposed,
        independent_variables=["dataset_version"],
        controlled_variables=_controlled_variables(
            ["dataset_version"], ctx, dataset_version_id=version_id
        ),
        metrics=_metrics_for(ctx),
        baseline_experiment_ids=[ctx.baseline.id],
        supporting_experiment_ids=[e.id for e in ctx.cluster][:10],
        estimated_gpu_hours=cost,
        estimate_source=cost_source,
        uncertainties=uncertainties,
        validation_criteria=_validation_criteria(ctx, gap),
        model_family=ctx.baseline.model_family,
        dataset_version_id=version_id,
        independent_variable_values={"dataset_version": version_id},
    )


def _new_family_candidates(gap: Gap, ctx: CandidateContext) -> list[Candidate]:
    if ctx.baseline is None:
        return []
    alternatives: Sequence[str] = gap.detail.get("alternatives") or []
    baseline_family = ((ctx.baseline.model_family or "").strip().lower() if ctx.baseline else "")
    metric = ctx.context.metric_name
    out: list[Candidate] = []
    for alternative in alternatives[:1]:
        # Never propose the baseline's own family as "another family".
        if alternative.strip().lower() == baseline_family:
            continue
        cost = get_profile(alternative).default_unit_cost_gpu_hours
        cost_source = "model_family_profile_default" if cost is not None else None
        uncertainties = [
            "Architecture change is a multi-variable change; attribute the outcome to the "
            "family only as a comparison, not as a causal claim."
        ]
        if cost is None:
            uncertainties.append(
                f"No cost default is registered for family {alternative!r}; supply an estimate."
            )
        out.append(
            Candidate(
                kind=CandidateKind.NEW_MODEL_FAMILY,
                hypothesis=(
                    f"An {alternative!r} model on the same data and protocol will reach "
                    f"{metric or 'a comparable score'} that differs from "
                    f"{ctx.baseline.model_family!r} by more than seed variance."
                ),
                # The proposed value must appear in the exported configuration, not
                # only in the hypothesis sentence, or FR-9 has nothing to edit.
                configuration={"model_family": alternative},
                independent_variables=["model_family"],
                controlled_variables=_controlled_variables(["model_family"], ctx),
                metrics=_metrics_for(ctx),
                baseline_experiment_ids=[ctx.baseline.id],
                supporting_experiment_ids=[ctx.baseline.id],
                estimated_gpu_hours=cost,
                estimate_source=cost_source,
                uncertainties=uncertainties,
                validation_criteria=_validation_criteria(ctx, gap),
                model_family=alternative,
                dataset_version_id=ctx.baseline.dataset_version_id,
                independent_variable_values={"model_family": alternative},
            )
        )
    return out


def _missing_metric_candidate(gap: Gap, ctx: CandidateContext) -> Candidate | None:
    if ctx.baseline is None:
        return None
    cost, cost_source, cost_uncertainties = _estimate_cost(ctx)
    return Candidate(
        kind=CandidateKind.EVALUATE_MISSING_METRIC,
        hypothesis=(
            f"Re-evaluating existing checkpoints on {gap.field_name!r} will make the "
            "unmeasured runs comparable with the rest of the history."
        ),
        configuration=_base_configuration(ctx),
        independent_variables=[f"evaluation:{gap.field_name}"],
        controlled_variables=_controlled_variables(["evaluation"], ctx),
        metrics=[gap.field_name] if gap.field_name else [],
        baseline_experiment_ids=[ctx.baseline.id],
        supporting_experiment_ids=[e.id for e in ctx.cluster][:10],
        estimated_gpu_hours=cost,
        estimate_source=cost_source,
        uncertainties=list(cost_uncertainties),
        validation_criteria=[
            ValidationCriterion(
                kind="completeness",
                description=(
                    "Record the metric on every run in the cluster; a metric present on some "
                    "runs only does not make them comparable."
                ),
            )
        ],
        model_family=ctx.baseline.model_family,
        dataset_version_id=ctx.baseline.dataset_version_id,
    )


def _natural_key(value: Any) -> tuple:
    """Sort ``v9`` before ``v10``.

    Pure lexicographic ordering picks ``v9`` as the newest version, which would
    send a proposal backwards in time.
    """
    text = str(value)
    parts: list[tuple[int, Any]] = []
    buffer = ""
    is_digit: bool | None = None
    for char in text:
        char_is_digit = char.isdigit()
        if is_digit is None or char_is_digit == is_digit:
            buffer += char
        else:
            parts.append((1, int(buffer)) if is_digit else (0, buffer))
            buffer = char
        is_digit = char_is_digit
    if buffer:
        parts.append((1, int(buffer)) if is_digit else (0, buffer))
    return tuple(parts)


def _observed_config_values(experiments: Sequence[ExperimentRecord], field_name: str) -> set[Any]:
    values: set[Any] = set()
    for experiment in experiments:
        merged = dict(experiment.configuration or {})
        merged.update(experiment.hyperparameters or {})
        if field_name in merged and merged[field_name] is not None:
            values.add(_normalise_value(merged[field_name]))
    return values


def _normalise_value(value: Any) -> Any:
    """Compare numbers numerically so 4 and 4.0 are the same setting."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def score_candidate(
    candidate: Candidate,
    ctx: CandidateContext,
    *,
    budget: float | None,
    cost_bounds: tuple[float | None, float | None],
) -> tuple[float, list[ScoreComponent]]:
    """Weighted heuristic over the six PRD factors, with every term exposed."""
    weights = ctx.config.weights.as_dict()

    relevance = _relevance(candidate, ctx)
    evidence_quality = _evidence_quality(candidate, ctx)
    information_gain = _information_gain(candidate, ctx)
    cost_efficiency = _cost_efficiency(candidate, budget, cost_bounds)
    uncertainty_reduction = _uncertainty_reduction(candidate, ctx)
    feasibility = _feasibility(candidate, budget)

    values = {
        "relevance": relevance[0],
        "evidence_quality": evidence_quality[0],
        "information_gain": information_gain[0],
        "cost_efficiency": cost_efficiency[0],
        "uncertainty_reduction": uncertainty_reduction[0],
        "feasibility": feasibility[0],
    }
    rationales = {
        "relevance": relevance[1],
        "evidence_quality": evidence_quality[1],
        "information_gain": information_gain[1],
        "cost_efficiency": cost_efficiency[1],
        "uncertainty_reduction": uncertainty_reduction[1],
        "feasibility": feasibility[1],
    }

    components = [
        ScoreComponent(
            name=name,
            weight=weights[name],
            value=round(values[name], 4),
            weighted_value=round(weights[name] * values[name], 4),
            rationale=rationales[name],
        )
        for name in weights
    ]
    total = round(sum(c.weighted_value for c in components), 4)
    return total, components


def _relevance(candidate: Candidate, ctx: CandidateContext) -> tuple[float, str]:
    parts: list[float] = []
    notes: list[str] = []

    if candidate.model_family and ctx.context.model_family:
        same = candidate.model_family.lower() == ctx.context.model_family.lower()
        parts.append(1.0 if same else 0.5)
        notes.append(
            f"family {candidate.model_family!r} {'matches' if same else 'differs from'} the objective"
        )

    if candidate.dataset_version_id and ctx.baseline and ctx.baseline.dataset_version_id:
        same_version = candidate.dataset_version_id == ctx.baseline.dataset_version_id
        parts.append(1.0 if same_version else 0.6)
        notes.append(
            f"dataset version {'held fixed' if same_version else 'changed'} relative to the baseline"
        )

    hypothesis_tokens = _tokens(candidate.hypothesis) | _tokens(str(candidate.configuration))
    if ctx.context.tokens and hypothesis_tokens:
        overlap = len(ctx.context.tokens & hypothesis_tokens) / len(ctx.context.tokens)
        parts.append(min(1.0, overlap))
        notes.append(f"terms from the objective reused: {overlap:.0%}")

    value = sum(parts) / len(parts) if parts else 0.5
    return value, "; ".join(notes) or "no explicit objective constraints to match against"


def _evidence_quality(candidate: Candidate, ctx: CandidateContext) -> tuple[float, str]:
    cluster_size = len(ctx.cluster)
    cfg = ctx.config
    size_score = min(1.0, cluster_size / max(1, cfg.min_cluster_for_strong_evidence))

    baseline_score = 1.0
    if ctx.baseline is not None:
        assessment = ctx.comparabilities.get(ctx.baseline.id)
        if assessment is not None:
            baseline_score = assessment.score

    penalty = 0.0
    if ctx.excluded:
        penalty = 0.15
    value = round(max(0.0, (0.5 * size_score + 0.5 * baseline_score) - penalty), 4)

    notes = [
        f"{cluster_size} comparable run(s) available",
        f"baseline comparability score {baseline_score:.2f}",
    ]
    if ctx.excluded:
        notes.append(
            f"{len(ctx.excluded)} record(s) withheld by an open correction, so the "
            "evidence base is smaller than the raw history"
        )
    return value, "; ".join(notes)


def _information_gain(candidate: Candidate, ctx: CandidateContext) -> tuple[float, str]:
    if candidate.kind is CandidateKind.REPLICATE_SEED:
        return 0.95, (
            "no variance estimate exists for this configuration, so replicating is the only "
            "way to know whether an observed difference is real"
        )
    if candidate.kind is CandidateKind.SWEEP_PARAMETER:
        field_name = candidate.independent_variables[0] if candidate.independent_variables else ""
        observed = _observed_config_values(ctx.cluster, field_name)
        total_seen = len(observed) + len(candidate.independent_variable_values)
        if not total_seen:
            return 0.5, f"{field_name} has no recorded values to build on"
        gain = min(1.0, 1.0 / total_seen + 0.35)
        return round(gain, 4), (
            f"{field_name} has {len(observed)} recorded value(s); one more value adds more "
            "information than another sweep of an already-explored parameter"
        )
    if candidate.kind is CandidateKind.TRY_DATASET_VERSION:
        return 0.7, "an untested dataset version is an entirely unmeasured direction"
    if candidate.kind is CandidateKind.NEW_MODEL_FAMILY:
        return 0.6, "a new family adds an independent comparison point"
    return 0.5, "fills a measurement gap rather than opening a new direction"


def _cost_efficiency(
    candidate: Candidate,
    budget: float | None,
    cost_bounds: tuple[float | None, float | None],
) -> tuple[float, str]:
    cost = candidate.estimated_gpu_hours
    if cost is None:
        return 0.25, "cost unknown, so efficiency cannot be assessed and is scored conservatively"
    low, high = cost_bounds
    if low is None or high is None or high <= low:
        return 0.5, f"estimated {cost} GPU-hours, but the candidate pool has no cost spread"
    span = high - low
    value = max(0.0, min(1.0, 1.0 - (cost - low) / span))
    return round(value, 4), (
        f"estimated {cost} GPU-hours against a pool range of {low}-{high}"
    )


def _uncertainty_reduction(candidate: Candidate, ctx: CandidateContext) -> tuple[float, str]:
    if candidate.uncertainties:
        return 0.5, f"resolves {len(candidate.uncertainties)} recorded uncertainty/uncertainties"
    return 0.3, "adds no specific uncertainty reduction"


def _feasibility(candidate: Candidate, budget: float | None) -> tuple[float, str]:
    cost = candidate.estimated_gpu_hours
    if cost is None:
        return 0.4, "no cost estimate, so feasibility is unknown"
    if budget is None:
        return 0.7, f"estimated {cost} GPU-hours with no budget cap supplied"
    if cost > budget:
        return 0.15, f"estimated {cost} GPU-hours exceeds the budget of {budget}"
    return 1.0, f"estimated {cost} GPU-hours fits inside the budget of {budget}"


# ---------------------------------------------------------------------------
# Budget
# ---------------------------------------------------------------------------


@dataclass
class RankedCandidate:
    """A candidate with its score already computed.

    Kept beside the candidate rather than stashed on it: the contract models stay
    clean and nothing depends on pydantic permitting private attribute writes.
    """

    candidate: Candidate
    score: float
    components: list[ScoreComponent]

    @property
    def estimated_gpu_hours(self) -> float | None:
        return self.candidate.estimated_gpu_hours


@dataclass
class BudgetOutcome:
    selected: list[RankedCandidate]
    unselected: list[RankedCandidate]
    total_gpu_hours: float
    exhausted: bool
    """True only when the *budget* was what blocked a candidate.

    Candidates dropped for an unknown cost or because ``limit`` was reached do not
    make the budget the binding constraint, and saying otherwise would
    misattribute the shortfall.
    """

    blocked_by_budget: list[str] = field(default_factory=list)
    blocked_by_unknown_cost: list[str] = field(default_factory=list)
    blocked_by_limit: list[str] = field(default_factory=list)


def apply_budget(
    ranked: Sequence[RankedCandidate],
    *,
    budget: float | None,
    limit: int,
) -> BudgetOutcome:
    """Greedy selection under a hard GPU-hour ceiling (PRD 11, step 8).

    Ranked by score, not by score-per-hour: a cheap but weakly informative
    experiment should not displace an expensive decisive one. Cost still enters
    through the cost_efficiency term, where the trade-off is visible.
    """
    ordered = sorted(ranked, key=lambda r: r.score, reverse=True)

    def total(chosen: Sequence[RankedCandidate]) -> float:
        return round(
            sum(r.estimated_gpu_hours for r in chosen if r.estimated_gpu_hours is not None), 4
        )

    if budget is None:
        chosen = ordered[:limit]
        rest = ordered[limit:]
        return BudgetOutcome(
            selected=chosen,
            unselected=list(rest),
            total_gpu_hours=total(chosen),
            exhausted=False,
            blocked_by_limit=[r.candidate.hypothesis for r in rest],
        )

    selected: list[RankedCandidate] = []
    unselected: list[RankedCandidate] = []
    by_budget: list[str] = []
    by_unknown_cost: list[str] = []
    by_limit: list[str] = []
    spent = 0.0

    for ranked_candidate in ordered:
        cost = ranked_candidate.estimated_gpu_hours
        # An unknown cost is treated as unfunded rather than assumed free, so a
        # plan never silently exceeds a real ceiling.
        if cost is None:
            unselected.append(ranked_candidate)
            by_unknown_cost.append(ranked_candidate.candidate.hypothesis)
            continue
        if len(selected) >= limit:
            unselected.append(ranked_candidate)
            by_limit.append(ranked_candidate.candidate.hypothesis)
            continue
        if spent + cost <= budget + 1e-9:
            selected.append(ranked_candidate)
            spent += cost
        else:
            unselected.append(ranked_candidate)
            by_budget.append(ranked_candidate.candidate.hypothesis)

    return BudgetOutcome(
        selected=selected,
        unselected=unselected,
        total_gpu_hours=round(spent, 4),
        exhausted=bool(by_budget),
        blocked_by_budget=by_budget,
        blocked_by_unknown_cost=by_unknown_cost,
        blocked_by_limit=by_limit,
    )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class RecommendationEngine:
    """FR-5 end to end. Depends only on the :class:`GraphService` port."""

    def __init__(
        self,
        graph: GraphService,
        *,
        comparability: ComparabilityEngine | None = None,
        config: RankingConfig | None = None,
        id_factory: Any = None,
    ) -> None:
        self.graph = graph
        self.comparability = comparability or ComparabilityEngine()
        self.config = config or RankingConfig()
        self._id_factory = id_factory or (lambda: f"rec_{uuid.uuid4().hex[:12]}")

    # -- main entry point --------------------------------------------------

    def generate(
        self,
        project_id: str,
        request: RecommendationRequest,
        *,
        experiments: Sequence[ExperimentRecord] | None = None,
        dataset_versions: Sequence[DatasetVersionSummary] | None = None,
        persist: bool = True,
    ) -> RecommendationSet:
        """Produce a budget-compliant set of proposals.

        ``experiments`` and ``dataset_versions`` may be supplied by a caller that
        already retrieved them, so a question that needs both a narrative and a
        recommendation reads the graph once.

        ``persist`` writes the proposals as ``Recommendation`` nodes so they can
        be approved later (FR-9). It defaults to true because the copilot's
        approve/reject actions address a stored record; a read-only caller that
        never approves can turn it off to avoid unbounded growth.
        """
        if experiments is None:
            experiments = self.graph.list_experiments(
                project_id,
                model_family=request.model_family,
                dataset_id=request.dataset_id,
            )
        if dataset_versions is None:
            dataset_versions = self.graph.list_dataset_versions(project_id)

        considered = [e.id for e in experiments]
        if not experiments:
            return RecommendationSet(
                compute_budget_gpu_hours=request.compute_budget_gpu_hours,
                uncertainties=[
                    "This project has no experiment records, so no recommendation can be "
                    "grounded. Import a history first."
                ],
            )

        context = parse_objective(request.objective, experiments, dataset_versions, request=request)

        # Step 5: correct the evidence base before reasoning from it.
        trusted, excluded = self.partition_evidence(project_id, experiments, request)

        cluster, comparabilities = self.select_baseline_cluster(trusted, context)
        baseline = self.best_run(cluster, context)

        gaps = identify_gaps(trusted, context, dataset_versions, config=self.config)
        profile = get_profile(context.model_family)
        candidate_ctx = CandidateContext(
            baseline=baseline,
            cluster=cluster,
            comparabilities=comparabilities,
            profile=profile,
            context=context,
            config=self.config,
            excluded=excluded,
        )

        candidates = generate_candidates(gaps, candidate_ctx)
        ranked = self._score_all(candidates, candidate_ctx, request.compute_budget_gpu_hours)
        outcome = apply_budget(
            ranked,
            budget=request.compute_budget_gpu_hours,
            limit=request.max_recommendations,
        )

        recommendations = [
            self._to_recommendation(item, project_id, context) for item in outcome.selected
        ]
        for recommendation in recommendations:
            if persist:
                self.graph.save_recommendation(recommendation)

        uncertainties = list(context.resolution_notes)
        if not trusted:
            if excluded:
                uncertainties.append(
                    "Every retrieved record is under an open correction, so no recommendation "
                    "rests on trusted evidence."
                )
            else:
                uncertainties.append(
                    "No run in this project can serve as a baseline, so no proposal can be "
                    "grounded."
                )
        elif not cluster:
            # Trusted records exist but none completed or measured the metric, so
            # there is no reference point to improve on.
            uncertainties.append(
                "No completed, measured run remains in the evidence base, so no proposal "
                "can be grounded."
            )
        if not outcome.selected:
            # An empty set must always say why, including the case where there was
            # simply nothing left to try.
            uncertainties.extend(
                _explain_empty_selection(outcome, request.compute_budget_gpu_hours)
            )
        elif outcome.blocked_by_budget:
            uncertainties.append(
                f"The budget of {request.compute_budget_gpu_hours} GPU-hours was the binding "
                f"constraint; {len(outcome.blocked_by_budget)} higher-ranked candidate(s) did "
                "not fit."
            )
        if outcome.blocked_by_unknown_cost:
            uncertainties.append(
                f"{len(outcome.blocked_by_unknown_cost)} candidate(s) were not proposed because "
                "no compute estimate could be derived and an unbudgeted run cannot be planned."
            )
        if outcome.blocked_by_limit:
            uncertainties.append(
                f"Only {request.max_recommendations} proposal(s) were returned; "
                f"{len(outcome.blocked_by_limit)} further candidate(s) were left out by the "
                "requested result limit."
            )

        return RecommendationSet(
            recommendations=recommendations,
            compute_budget_gpu_hours=request.compute_budget_gpu_hours,
            estimated_gpu_hours=outcome.total_gpu_hours,
            budget_exhausted=outcome.exhausted,
            unselected_candidates=[item.candidate for item in outcome.unselected],
            considered_experiment_ids=considered,
            excluded_evidence=excluded,
            uncertainties=uncertainties,
        )

    # -- stages ------------------------------------------------------------

    def partition_evidence(
        self,
        project_id: str,
        experiments: Sequence[ExperimentRecord],
        request: RecommendationRequest,
    ) -> tuple[list[ExperimentRecord], list[ExcludedEvidence]]:
        corrections = self.graph.list_corrections(project_id)
        disputed_versions: set[str] = set()
        disputed_experiments: set[str] = set()

        for correction in corrections:
            if correction.status not in CORRECTION_STATUSES_BLOCKING_TRUST:
                continue
            impact = self.graph.get_correction_impact(correction.id)
            if impact is None:
                disputed_versions.add(correction.source_id)
                continue
            disputed_experiments.update(impact.directly_affected_experiment_ids)
            disputed_experiments.update(impact.indirectly_affected_experiment_ids)

        trusted, excluded, _ = partition_by_trust(
            experiments,
            disputed_experiment_ids=disputed_experiments,
            corrected_dataset_version_ids=disputed_versions,
        )
        if request.include_disputed_evidence:
            # Opt-in: disputed records come back, but still labelled.
            return list(experiments), excluded
        return trusted, excluded

    def select_baseline_cluster(
        self,
        trusted: Sequence[ExperimentRecord],
        context: ObjectiveContext,
    ) -> tuple[list[ExperimentRecord], dict[str, ComparabilityAssessment]]:
        """Narrow to runs that can legitimately be compared (FR-4 in the loop).

        The anchor is chosen inside the dominant *measurement cohort*, not across
        the whole history. Picking the global best run would anchor on whatever
        dataset version happened to score highest, which is precisely the
        comparison trap the product exists to prevent: a number from a different
        data version is not a stronger baseline, it is a different measurement.
        """
        if not trusted:
            return [], {}

        candidates = self.graph.find_comparable_candidates(
            project_id=trusted[0].project_id,
            model_family=context.model_family,
            dataset_id=context.dataset_id,
            metric=context.metric_name,
        )
        known = {e.id for e in trusted}
        pool = [e for e in candidates if e.id in known] or list(trusted)
        if not pool:
            return list(trusted), {}

        # A run that did not finish produced no result to anchor a proposal on. Its
        # number, if any, must not become the reference point. If nothing
        # completed, there is no baseline at all rather than a failed one.
        finished = [e for e in pool if e.status is ExperimentStatus.COMPLETED]
        if not finished:
            return [], {}
        pool = finished

        if context.metric_name:
            # Only runs that actually recorded a value can anchor anything.
            pool = _measured(pool, context.metric_name) or pool

        cohort = self._dominant_cohort(pool, context)
        anchor = self._anchor_run(cohort, context)
        if anchor is None:
            return list(cohort), {}

        comparabilities: dict[str, ComparabilityAssessment] = {}
        cohort_ids = {e.id for e in cohort}
        cluster: list[ExperimentRecord] = []

        # Every retrieved run is assessed, not just the cohort, so the answer can
        # say *why* a run was excluded rather than omitting it silently.
        for experiment in pool:
            assessment = self.comparability.assess(
                anchor, experiment, metric_name=context.metric_name
            )
            comparabilities[experiment.id] = assessment
            if experiment.id in cohort_ids and assessment.is_usable_as_baseline:
                cluster.append(experiment)

        if not cluster:
            # Nothing is comparable; fall back to the anchor alone and let the
            # caveats surface rather than pretending the history is comparable.
            cluster = [anchor]
        return cluster, comparabilities

    @staticmethod
    def _dominant_cohort(
        pool: Sequence[ExperimentRecord], context: ObjectiveContext
    ) -> list[ExperimentRecord]:
        """The largest group of runs sharing dataset, version, split and protocol.

        Ties break on the highest total recorded metric, then on identifier, so the
        choice is deterministic and still explainable.
        """
        groups: dict[tuple[str, ...], list[ExperimentRecord]] = {}
        for experiment in pool:
            metric = (
                experiment.metric(context.metric_name)
                if context.metric_name
                else (experiment.metrics[0] if experiment.metrics else None)
            )
            key = (
                (experiment.dataset_id or "").strip().lower(),
                (experiment.dataset_version_id or "").strip().lower(),
                (metric.evaluation_split if metric else None) or "",
                str(metric.evaluation_protocol) if metric and metric.evaluation_protocol else "",
                # The family is part of what makes a cohort comparable; including it
                # stops a sweep from mixing parameters across architectures.
                (context.model_family or "").strip().lower(),
            )
            groups.setdefault(key, []).append(experiment)

        if not groups:
            return list(pool)

        def group_rank(item: tuple[tuple[str, ...], list[ExperimentRecord]]) -> tuple:
            key, members = item
            values = [
                m.value
                for e in members
                if context.metric_name
                for m in [e.metric(context.metric_name)]
                if m is not None and m.value is not None
            ]
            total = sum(values) if values else 0.0
            return (len(members), total, key[1], key[0])

        best_key = max(groups.items(), key=group_rank)[0]
        return groups[best_key]

    def _anchor_run(
        self, pool: Sequence[ExperimentRecord], context: ObjectiveContext
    ) -> ExperimentRecord | None:
        best = self.best_run(pool, context)
        if best is not None:
            return best
        return pool[0] if pool else None

    @staticmethod
    def best_run(
        pool: Sequence[ExperimentRecord], context: ObjectiveContext
    ) -> ExperimentRecord | None:
        """Best run by the target metric, honouring the metric direction."""
        if not pool:
            return None
        if not context.metric_name:
            return max(pool, key=lambda e: e.id)

        scored: list[tuple[float, ExperimentRecord]] = []
        for experiment in pool:
            metric = experiment.metric(context.metric_name)
            if metric is None or metric.value is None:
                continue
            scored.append((metric.value, experiment))
        if not scored:
            return None

        direction = _direction_for(scored[0][1], context.metric_name, context)
        if direction is MetricDirection.MINIMIZE:
            return min(scored, key=lambda pair: pair[0])[1]
        return max(scored, key=lambda pair: pair[0])[1]

    def _score_all(
        self,
        candidates: Sequence[Candidate],
        ctx: CandidateContext,
        budget: float | None,
    ) -> list[RankedCandidate]:
        known_costs = [c.estimated_gpu_hours for c in candidates if c.estimated_gpu_hours is not None]
        bounds = (min(known_costs), max(known_costs)) if known_costs else (None, None)

        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            score, components = score_candidate(
                candidate, ctx, budget=budget, cost_bounds=bounds
            )
            ranked.append(
                RankedCandidate(candidate=candidate, score=score, components=components)
            )
        return ranked

    def _to_recommendation(
        self, ranked: RankedCandidate, project_id: str, context: ObjectiveContext
    ) -> Recommendation:
        candidate = ranked.candidate
        components = ranked.components
        return Recommendation(
            id=self._id_factory(),
            project_id=project_id,
            objective=context.objective,
            hypothesis=candidate.hypothesis,
            candidate_kind=candidate.kind,
            baseline_experiment_ids=candidate.baseline_experiment_ids,
            configuration=candidate.configuration,
            independent_variables=candidate.independent_variables,
            controlled_variables=candidate.controlled_variables,
            metrics=candidate.metrics,
            estimated_gpu_hours=candidate.estimated_gpu_hours,
            estimate_source=candidate.estimate_source,
            supporting_experiment_ids=candidate.supporting_experiment_ids,
            independent_variable_values=dict(candidate.independent_variable_values),
            uncertainties=candidate.uncertainties,
            validation_criteria=candidate.validation_criteria,
            rationale=_rationale(candidate, context, components),
            score=ranked.score,
            score_kind="weighted_heuristic",
            score_components=components,
            evidence_status=EvidenceStatus.INFERRED,
            excluded_evidence_ids=[],
        )

    # -- decisions (FR-9) --------------------------------------------------

    def decide(
        self,
        recommendation_id: str,
        request: Decision,
        *,
        decided_by: str,
        note: str | None = None,
        edited_configuration: dict[str, Any] | None = None,
    ) -> Recommendation:
        """Record a human decision. Recommendations are advisory, never applied."""
        if self.graph.get_recommendation(recommendation_id) is None:
            raise KeyError(recommendation_id)

        # The status transition itself belongs to the graph implementation, which
        # owns the Recommendation node (PRD 8). This area only records the intent.
        decision = RecommendationDecision(
            id=f"dec_{uuid.uuid4().hex[:12]}",
            recommendation_id=recommendation_id,
            decision=request,
            decided_by=decided_by,
            note=note,
            edited_configuration=edited_configuration,
        )
        return self.graph.record_recommendation_decision(decision)


def _explain_empty_selection(outcome: BudgetOutcome, budget: float | None) -> list[str]:
    """State why nothing was proposed.

    An empty ``recommendations`` list with no explanation reads as a broken
    feature. The cause is almost always the budget, an unusable cohort, or a
    missing cost estimate, and the caller needs to know which.
    """
    if not outcome.unselected:
        return [
            "No gap was found in the current history: every lever in this family's "
            "sweep ladder has already been tried, every dataset version has been used, "
            "and no measurement is missing. There is nothing this engine can usefully "
            "propose from the recorded evidence."
        ]
    if outcome.blocked_by_unknown_cost and len(outcome.blocked_by_unknown_cost) == len(
        outcome.unselected
    ):
        return [
            "Candidates were found but none could be budgeted: no comparable run recorded "
            "a compute cost and no family default applies. Supply an estimate to proceed."
        ]
    if budget is not None:
        cheapest = min(
            (
                r.estimated_gpu_hours
                for r in outcome.unselected
                if r.estimated_gpu_hours is not None
            ),
            default=None,
        )
        detail = (
            f" The cheapest unselected candidate needs {cheapest} GPU-hours."
            if cheapest is not None
            else ""
        )
        return [
            f"No proposal fits inside the budget of {budget} GPU-hours.{detail} Raise the "
            "budget or record a cheaper comparable run."
        ]
    return [
        "Candidates were generated but none were selected; see unselected_candidates for "
        "the detail."
    ]


def _direction_for(
    experiment: ExperimentRecord, metric_name: str, context: ObjectiveContext
) -> MetricDirection:
    metric = experiment.metric(metric_name)
    if metric is not None:
        return metric.direction
    return context.metric_direction


def _rationale(
    candidate: Candidate, context: ObjectiveContext, components: Sequence[ScoreComponent]
) -> str:
    """One paragraph a researcher can challenge term by term (FR-8)."""
    baseline = (
        f" based on {', '.join(candidate.baseline_experiment_ids)}"
        if candidate.baseline_experiment_ids
        else " without a comparable baseline"
    )
    cost = (
        f" at an estimated {candidate.estimated_gpu_hours} GPU-hours"
        if candidate.estimated_gpu_hours is not None
        else " with no compute estimate available"
    )
    drivers = ", ".join(
        f"{c.name} {c.value:.2f}" for c in sorted(components, key=lambda c: -c.weighted_value)[:3]
    )
    return (
        f"Proposed {candidate.kind.value.replace('_', ' ')}{baseline}{cost}, closing a gap in "
        f"{context.metric_name or 'the target metric'}. Score {sum(c.weighted_value for c in components):.2f} "
        f"is driven mainly by {drivers}. This is a proposal, not a result."
    )