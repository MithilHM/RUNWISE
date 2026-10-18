"""Research copilot: natural-language investigation (FR-3).

Endpoint:

    POST /api/v1/projects/{project_id}/query

This module also holds :class:`ResearchQueryService`, the orchestration service
for PRD 18. It is where retrieval, comparability, correction state and
recommendation come together into one answer.

Order of operations, and why:

1. Retrieve once. The narrative and the recommendation read the same snapshot, so
   the two halves of an answer cannot disagree about what exists.
2. Divide the evidence into trusted and withheld *before* reasoning. A disputed
   record may be discussed, but never counted as support.
3. Assess comparability, so "best run" means "best among runs that may be
   compared" rather than "highest number".
4. Compose the deterministic answer from what was actually found.
5. Optionally hand the prose to an LLM. Its text is accepted only if every
   identifier it names is in the retrieved set; otherwise the deterministic text
   stands.
6. Verify every citation. An unknown identifier fails the response.

An LLM is never asked what happened, only how to phrase what the engine found.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Path

from app.api.deps import ComparabilityDep, EngineDep, GraphDep, WriterDep
from app.models.contracts import (
    AnswerMode,
    ComparabilityStatus,
    ClaimStatus,
    CorrectionImpact,
    CORRECTION_STATUSES_BLOCKING_TRUST,
    EvidenceStatus,
    ExperimentRecord,
    MetricDirection,
    QueryRequest,
    Recommendation,
    RecommendationRequest,
    ResearchAnswer,
)
from app.services.evidence import (
    EvidenceLedger,
    GroundingViolation,
    partition_by_trust,
)
from app.services.recommendations import (
    ObjectiveContext,
    RecommendationEngine,
    parse_objective,
)

# Mounted by ``app/main.py`` under ``settings.API_V1_STR``; see the note in
# ``app/api/recommendations.py``.
router = APIRouter(tags=["copilot"])


class ResearchQueryService:
    """Orchestrates one natural-language research question."""

    def __init__(
        self,
        graph: Any,
        *,
        engine: RecommendationEngine,
        comparability: Any,
        writer: Any,
    ) -> None:
        self.graph = graph
        self.engine = engine
        self.comparability = comparability
        self.writer = writer

    # -- entry point -------------------------------------------------------

    def answer(self, project_id: str, request: QueryRequest) -> ResearchAnswer:
        experiments = self.graph.list_experiments(project_id)
        dataset_versions = self.graph.list_dataset_versions(project_id)

        # Everything the backend returned is citable, including records we then
        # withhold. The ledger models what exists; trust is a separate question.
        ledger = EvidenceLedger()
        ledger.register(experiments)

        if not experiments:
            return self._answer_without_data(project_id, request)

        trusted, excluded, statuses = self._partition_evidence(project_id, experiments)

        recommendation_request = RecommendationRequest(
            objective=request.question,
            compute_budget_gpu_hours=request.compute_budget_gpu_hours,
            max_recommendations=request.max_recommendations,
        )
        context = parse_objective(
            request.question, experiments, dataset_versions, request=recommendation_request
        )

        result = self.engine.generate(
            project_id,
            recommendation_request,
            experiments=experiments,
            dataset_versions=dataset_versions,
        )

        # The engine partitions again internally; reuse its view so the narrative
        # and the recommendation agree on what was withheld.
        excluded = result.excluded_evidence or excluded

        cluster, comparabilities = self.engine.select_baseline_cluster(trusted, context)
        anchor = self.engine.best_run(cluster, context) if cluster else None
        corrections = self._correction_impacts(project_id)

        claims, missing, uncertainties, insufficient = self._build_claims(
            ledger=ledger,
            context=context,
            cluster=cluster,
            anchor=anchor,
            comparabilities=comparabilities,
            excluded=excluded,
            recommendations=result.recommendations,
            statuses=statuses,
        )

        # Every non-COMPARABLE verdict is surfaced, including runs outside the cohort,
        # so the client can explain an exclusion instead of silently dropping it
        # (FR-8).
        comparisons = sorted(
            (
                assessment
                for assessment in comparabilities.values()
                if assessment.status is not ComparabilityStatus.COMPARABLE
            ),
            key=lambda a: (a.status.value, a.right_experiment_id or ""),
        )

        # This list is named "comparable", so it must not carry a run the same
        # answer says cannot be compared to anything. A single-run cluster is the
        # fallback case, not a cohort, so it reports nothing.
        comparable_ids = [e.id for e in cluster] if len(cluster) > 1 else []

        deterministic = self._compose(
            question=request.question,
            context=context,
            experiments=experiments,
            cluster=cluster,
            anchor=anchor,
            comparabilities=comparabilities,
            excluded=excluded,
            recommendations=result.recommendations,
            budget=request.compute_budget_gpu_hours,
            missing=missing,
            uncertainties=uncertainties,
            insufficient=insufficient,
        )

        facts = self._facts(
            question=request.question,
            context=context,
            experiments=experiments,
            cluster=cluster,
            anchor=anchor,
            comparabilities=comparabilities,
            excluded=excluded,
            recommendations=result.recommendations,
        )

        explanation = self.writer.write(
            question=request.question,
            facts=facts,
            deterministic_text=deterministic,
            allowed_ids=sorted(ledger.known_ids),
        )

        cited: set[str] = set()
        for claim in claims:
            cited.update(ref.experiment_id for ref in claim.support if ref.experiment_id)
        for recommendation in result.recommendations:
            cited.update(recommendation.baseline_experiment_ids)
            cited.update(recommendation.supporting_experiment_ids)

        try:
            grounding = ledger.verify(cited)
        except GroundingViolation as exc:
            # Defensive: an ungrounded answer must not reach a researcher.
            grounding = ledger.report(cited)
            deterministic = (
                f"{deterministic}\n\n"
                f"[withheld] {len(exc.unknown_citations)} citation(s) could not be matched to "
                "retrieved records, so this answer is reported without them."
            )
            explanation_text = deterministic
            explanation_mode = AnswerMode.DETERMINISTIC
            llm_used = False
        else:
            explanation_text = explanation.text
            explanation_mode = explanation.mode
            llm_used = explanation.llm_used

        return ResearchAnswer(
            question=request.question,
            answer=explanation_text,
            mode=explanation_mode,
            llm_used=llm_used,
            grounding=grounding,
            claims=claims,
            supporting_experiments=list(cluster[:10]),
            comparisons=comparisons,
            recommendations=result.recommendations,
            corrections_considered=corrections,
            excluded_evidence=excluded,
            comparable_experiment_ids=comparable_ids,
            uncertainties=list(dict.fromkeys(uncertainties + result.uncertainties)),
            missing_information=missing,
            insufficient_evidence=insufficient,
        )

    # -- evidence handling -------------------------------------------------

    def _partition_evidence(
        self, project_id: str, experiments: Sequence[ExperimentRecord]
    ) -> tuple[list[ExperimentRecord], list[Any], dict[str, EvidenceStatus]]:
        disputed_experiments: set[str] = set()
        disputed_versions: set[str] = set()

        for correction in self.graph.list_corrections(project_id):
            if correction.status not in CORRECTION_STATUSES_BLOCKING_TRUST:
                continue
            impact = self.graph.get_correction_impact(correction.id)
            if impact is None:
                disputed_versions.add(correction.source_id)
                continue
            disputed_experiments.update(impact.directly_affected_experiment_ids)
            disputed_experiments.update(impact.indirectly_affected_experiment_ids)

        return partition_by_trust(
            experiments,
            disputed_experiment_ids=disputed_experiments,
            corrected_dataset_version_ids=disputed_versions,
        )

    def _correction_impacts(self, project_id: str) -> list[CorrectionImpact]:
        impacts: list[CorrectionImpact] = []
        for correction in self.graph.list_corrections(project_id):
            impact = self.graph.get_correction_impact(correction.id)
            if impact is not None:
                impacts.append(impact)
        return impacts

    # -- claims ------------------------------------------------------------

    def _build_claims(
        self,
        *,
        ledger: EvidenceLedger,
        context: ObjectiveContext,
        cluster: Sequence[ExperimentRecord],
        anchor: ExperimentRecord | None,
        comparabilities: dict[str, Any],
        excluded: Sequence[Any],
        recommendations: Sequence[Recommendation],
        statuses: dict[str, EvidenceStatus],
    ) -> tuple[list[Any], list[str], list[str], bool]:
        claims: list[Any] = []
        missing: list[str] = []
        uncertainties: list[str] = []
        insufficient = False

        if anchor is not None and context.metric_name:
            metric = anchor.metric(context.metric_name)
            value = metric.value if metric else None
            if value is None:
                missing.append(
                    f"The best-scoring run {anchor.id} has no recorded value for "
                    f"{context.metric_name!r}."
                )
                insufficient = True
            else:
                assessment = comparabilities.get(anchor.id)
                status_note = (
                    f" (comparability of that run: {assessment.status.value})"
                    if assessment
                    else ""
                )
                # Grade the claim with the record's real epistemic status rather
                # than assuming every retrieved run is an observation.
                anchor_status = statuses.get(anchor.id, EvidenceStatus.OBSERVED)
                claim_text = _headline_claim(
                    metric_name=context.metric_name,
                    value=value,
                    anchor=anchor,
                    assessment=comparabilities.get(anchor.id),
                    cluster_size=len(cluster),
                )

                claims.append(
                    ledger.claim(
                        "claim_best_run",
                        claim_text,
                        [(anchor.id, context.metric_name)],
                        status=ClaimStatus.SUPPORTED,
                        evidence_status=anchor_status,
                    )
                )
        elif not cluster and excluded:
            # Every record was withheld. Saying "no run records this metric" would
            # be false: the records exist, they are simply not trusted.
            missing.append(
                f"No trusted run remains. {len(excluded)} record(s) that do record "
                f"{context.metric_name or 'the metric'} are withheld under an open correction, "
                "so no ranking is possible from trusted evidence."
            )
            insufficient = True
        elif context.metric_name:
            missing.append(
                f"No run in this project records a value for {context.metric_name!r}, so no "
                "ranking is possible."
            )
            insufficient = True

        if not cluster:
            insufficient = True
            missing.append(
                "No two runs in this project are comparable under the current records, so "
                "there is no valid baseline."
            )
        elif len(cluster) == 1:
            uncertainties.append(
                f"Only one comparable run exists ({cluster[0].id}); any comparison against it "
                "is unverified."
            )

        caveat_count = sum(
            1
            for assessment in comparabilities.values()
            if assessment.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS
        )
        if caveat_count:
            uncertainties.append(
                f"{caveat_count} run(s) are comparable only with caveats; the reason is listed "
                "with each comparison."
            )

        insufficient_assessments = [
            eid
            for eid, assessment in comparabilities.items()
            if assessment.status is ComparabilityStatus.INSUFFICIENT_INFORMATION
        ]
        if insufficient_assessments:
            missing.append(
                f"{len(insufficient_assessments)} run(s) lack a field needed for a verdict "
                f"({', '.join(sorted(insufficient_assessments)[:5])})."
            )

        if excluded:
            claims.append(
                _claim_about_withheld(ledger, excluded, statuses)
            )

        if recommendations:
            top = recommendations[0]
            claims.append(
                ledger.claim(
                    "claim_recommendation",
                    top.hypothesis,
                    list(top.supporting_experiment_ids) or list(top.baseline_experiment_ids),
                    status=ClaimStatus.PROPOSED,
                    evidence_status=EvidenceStatus.INFERRED,
                    detail=top.rationale,
                )
            )
            if top.estimated_gpu_hours is None:
                uncertainties.append(
                    "The top proposal has no compute estimate, so it cannot be budgeted."
                )

        if not claims:
            insufficient = True

        return claims, missing, uncertainties, insufficient

    # -- prose -------------------------------------------------------------

    def _compose(
        self,
        *,
        question: str,
        context: ObjectiveContext,
        experiments: Sequence[ExperimentRecord],
        cluster: Sequence[ExperimentRecord],
        anchor: ExperimentRecord | None,
        comparabilities: dict[str, Any],
        excluded: Sequence[Any],
        recommendations: Sequence[Recommendation],
        budget: float | None,
        missing: Sequence[str],
        uncertainties: Sequence[str],
        insufficient: bool,
    ) -> str:
        """Deterministic answer. Every number here came from a retrieved record."""
        families = sorted({(e.model_family or "unknown") for e in experiments})
        versions = sorted({(e.dataset_version or "?") for e in experiments})

        lines: list[str] = [
            f"Evidence base: {len(experiments)} run(s) across {len(families)} model "
            f"family/families ({', '.join(families)}) and {len(versions)} dataset version(s)."
        ]

        if anchor is not None and context.metric_name:
            metric = anchor.metric(context.metric_name)
            value = metric.value if metric else "not recorded"
            word = _superlative(metric)
            anchor_assessment = comparabilities.get(anchor.id)
            if _anchor_is_unverified(anchor_assessment):
                gap = ", ".join(anchor_assessment.missing_dimensions) or "required fields"
                lines.append(
                    f"{word.capitalize()} recorded {context.metric_name}: {value} on "
                    f"{anchor.id} ({anchor.name}). It cannot be compared to anything: "
                    f"{gap} not recorded."
                )
            elif len(cluster) < 2:
                lines.append(
                    f"{word.capitalize()} recorded {context.metric_name}: {value} on "
                    f"{anchor.id} ({anchor.name}). It is the only comparable run here, so "
                    "this is a measurement rather than the result of a comparison."
                )
            else:
                lines.append(
                    f"Best comparable {context.metric_name}: {value} on {anchor.id} "
                    f"({anchor.name})."
                )

        if cluster:
            listed = ", ".join(e.id for e in cluster[:8])
            more = f" and {len(cluster) - 8} more" if len(cluster) > 8 else ""
            lines.append(f"Runs in the comparable set: {listed}{more}.")

        not_comparable = [
            eid
            for eid, assessment in comparabilities.items()
            if assessment.status is ComparabilityStatus.NOT_COMPARABLE
        ]
        if not_comparable:
            lines.append(
                f"Not comparable with the baseline: {', '.join(sorted(not_comparable)[:8])}. "
                "Each exclusion names the field that differs."
            )

        if excluded:
            ids = ", ".join(item.experiment_id for item in excluded[:6])
            lines.append(
                f"Withheld under an open correction: {len(excluded)} run(s) ({ids}). "
                "These numbers are not used as support."
            )

        if recommendations:
            top = recommendations[0]
            cost = (
                f"about {top.estimated_gpu_hours} GPU-hours"
                if top.estimated_gpu_hours is not None
                else "with no compute estimate available"
            )
            budget_note = f" (budget {budget} GPU-hours)" if budget is not None else ""
            lines.append(f"Highest-ranked proposal, {cost}{budget_note}: {top.hypothesis}")
            if top.independent_variables:
                lines.append(
                    f"  independent variable(s): {', '.join(top.independent_variables)}; "
                    f"held fixed: {', '.join(top.controlled_variables) or 'none recorded'}."
                )
        else:
            lines.append(
                "No proposal could be grounded in this project's records."
                + (f" Budget was {budget} GPU-hours." if budget is not None else "")
            )

        if missing:
            lines.append("What is not known: " + " ".join(missing))
        if uncertainties:
            lines.append("Caveats: " + " ".join(uncertainties))
        if insufficient and not missing:
            lines.append(
                "The available records do not answer this question well enough to conclude."
            )

        lines.append(
            "Recommendations are proposals derived from the records above; they are not results."
        )
        return "\n".join(lines)

    def _facts(self, **kwargs: Any) -> dict[str, Any]:
        """The exact payload handed to an LLM, if one is configured."""
        context: ObjectiveContext = kwargs["context"]
        anchor: ExperimentRecord | None = kwargs["anchor"]
        cluster: Sequence[ExperimentRecord] = kwargs["cluster"]

        def brief(experiment: ExperimentRecord) -> dict[str, Any]:
            return {
                "id": experiment.id,
                "name": experiment.name,
                "model_family": experiment.model_family,
                "dataset_version": experiment.dataset_version,
                "status": experiment.status.value
                if hasattr(experiment.status, "value")
                else experiment.status,
                "metrics": {
                    m.name: m.value for m in experiment.metrics if m.value is not None
                },
                "compute_cost_gpu_hours": experiment.compute_cost_gpu_hours,
            }

        return {
            "question": kwargs["question"],
            "target_metric": context.metric_name,
            "metric_direction": context.metric_direction.value,
            "best_run": brief(anchor) if anchor else None,
            "comparable_runs": [brief(e) for e in cluster],
            "not_comparable": kwargs["comparabilities"],
            "withheld_by_correction": [
                {
                    "experiment_id": item.experiment_id,
                    "reason": item.reason,
                    "detail": item.detail,
                }
                for item in kwargs["excluded"]
            ],
            "proposals": [
                {
                    "hypothesis": r.hypothesis,
                    "independent_variables": r.independent_variables,
                    "estimated_gpu_hours": r.estimated_gpu_hours,
                    "supporting_experiment_ids": r.supporting_experiment_ids,
                    "score": r.score,
                }
                for r in kwargs["recommendations"]
            ],
        }

    # -- degenerate cases --------------------------------------------------

    def _answer_without_data(self, project_id: str, request: QueryRequest) -> ResearchAnswer:
        text = (
            f"This project ({project_id}) has no experiment records, so the question "
            f"{request.question!r} cannot be answered from evidence. Import an experiment "
            "history first; nothing is inferred without records."
        )
        return ResearchAnswer(
            question=request.question,
            answer=text,
            mode=AnswerMode.DETERMINISTIC,
            llm_used=False,
            insufficient_evidence=True,
            missing_information=["No experiments exist in this project."],
            uncertainties=["No recommendation can be produced without stored records."],
        )


def _anchor_is_unverified(assessment: Any) -> bool:
    """True when the anchor run's own comparability could not be established.

    ``select_baseline_cluster`` falls back to a single-run cluster when nothing
    else can be compared against the best-scoring run. That fallback keeps the
    response useful, but the run is then a measurement rather than a leader, and
    the answer must not describe it as one.
    """
    if assessment is None:
        return False
    return assessment.status is ComparabilityStatus.INSUFFICIENT_INFORMATION


def _superlative(metric: Any) -> str:
    """``lowest`` or ``highest``, from the metric's own direction.

    Hardcoding "highest" is wrong for every loss-like metric: the anchor for a
    minimize metric is the run with the *smallest* value, so calling it the
    highest contradicts the number printed next to it.
    """
    direction = getattr(metric, "direction", None)
    return "lowest" if direction is MetricDirection.MINIMIZE else "highest"


def _headline_claim(
    *,
    metric_name: str | None,
    value: Any,
    anchor: ExperimentRecord,
    assessment: Any,
    cluster_size: int,
) -> str:
    """One sentence stating the measurement and exactly how far it is supported.

    A comparison needs two runs. With fewer, the honest claim is about the
    measurement alone, not about winning a field.
    """
    metric = anchor.metric(metric_name) if metric_name else None
    word = _superlative(metric)

    if _anchor_is_unverified(assessment):
        gap = ", ".join(assessment.missing_dimensions) or "required fields"
        return (
            f"{word.capitalize()} recorded {metric_name} is {value} on {anchor.id}, "
            f"but its comparability could not be verified ({gap} not recorded), "
            "and no other run in this project is comparable to it."
        )

    if cluster_size < 2:
        return (
            f"{word.capitalize()} recorded {metric_name} is {value} on {anchor.id}. "
            "It is the only run that can be compared here, so this is a measurement "
            "rather than the result of a comparison."
        )

    status_note = (
        f" (comparability of that run: {assessment.status.value})" if assessment else ""
    )
    return (
        f"{word.capitalize()} {metric_name} among comparable runs is "
        f"{value} on {anchor.id}{status_note}."
    )


def _claim_about_withheld(
    ledger: EvidenceLedger, excluded: Sequence[Any], statuses: dict[str, EvidenceStatus]
) -> Any:
    """A claim that only asserts records were withheld, and cites them."""
    ids = [item.experiment_id for item in excluded][:5]
    reasons = "; ".join(
        f"{item.experiment_id}: {item.detail}" for item in excluded[:3]
    )
    # If the withheld records carry a single grade, use it; otherwise the claim is
    # about a mixed set and DISPUTED is the conservative label.
    grades = {statuses.get(item.experiment_id, EvidenceStatus.DISPUTED) for item in excluded}
    evidence_status = grades.pop() if len(grades) == 1 else EvidenceStatus.DISPUTED

    return ledger.claim(
        "claim_excluded_evidence",
        f"{len(excluded)} run(s) are withheld under an open correction and are not used as "
        f"support ({', '.join(ids)}).",
        ids,
        status=ClaimStatus.SUPPORTED,
        evidence_status=evidence_status,
        detail=reasons,
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@router.post(
    "/projects/{project_id}/query",
    response_model=ResearchAnswer,
    summary="Ask a question about this project's experiment history",
)
def query_project(
    graph: GraphDep,
    engine: EngineDep,
    comparability: ComparabilityDep,
    writer: WriterDep,
    body: QueryRequest,
    project_id: Annotated[str, Path(min_length=1)],
) -> ResearchAnswer:
    """FR-3.

    Project scoping is enforced by the graph service, which applies the project id
    inside every query rather than filtering afterwards (PRD 12: no unauthorised
    cross-project access).
    """
    service = ResearchQueryService(
        graph, engine=engine, comparability=comparability, writer=writer
    )
    return service.answer(project_id, body)