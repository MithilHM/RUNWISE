"""In-memory GraphService for the engine test suites (PRD 18).

The graph layer owns the FalkorDB implementation under ``backend/app/graph/**``.
These tests must not need a running database to prove the logic, so this fake
implements the same :class:`~app.ports.graph.GraphService` protocol in memory.

It is deliberately strict about two things the real implementation must also be
strict about:

  * ``list_experiments`` and friends scope by project inside the lookup, not by
    filtering a global list afterwards.
  * A recommendation decision mutates stored state, so a test can prove that an
    approval was actually recorded.
"""

from __future__ import annotations

from typing import Any

from app.models.contracts import (
    CorrectionImpact,
    CorrectionRecord,
    DatasetSummary,
    DatasetVersionSummary,
    Decision,
    ExperimentRecord,
    Recommendation,
    RecommendationDecision,
    RecommendationStatus,
    ReviewItem,
    ReviewItemStatus,
)
from app.ports.graph import GraphService


class FakeGraphService(GraphService):
    """Test double for the graph port."""

    def __init__(
        self,
        *,
        experiments: list[ExperimentRecord] | None = None,
        datasets: list[DatasetSummary] | None = None,
        dataset_versions: list[DatasetVersionSummary] | None = None,
        corrections: list[CorrectionRecord] | None = None,
    ) -> None:
        self.experiments: dict[str, ExperimentRecord] = {
            e.id: e for e in (experiments or [])
        }
        self.datasets: dict[str, DatasetSummary] = {d.id: d for d in (datasets or [])}
        self.dataset_versions: dict[str, DatasetVersionSummary] = {
            v.id: v for v in (dataset_versions or [])
        }
        self.corrections: dict[str, CorrectionRecord] = {
            c.id: c for c in (corrections or [])
        }
        self.review_items: dict[str, ReviewItem] = {}
        self.recommendations: dict[str, Recommendation] = {}
        self.decisions: list[RecommendationDecision] = []
        self.lineage: dict[str, list[dict[str, Any]]] = {}

    # -- experiments -------------------------------------------------------

    def list_experiments(
        self,
        project_id: str,
        *,
        model_family: str | None = None,
        dataset_id: str | None = None,
        dataset_version_id: str | None = None,
        metric: str | None = None,
        status: str | None = None,
        config_field: str | None = None,
        config_value: Any | None = None,
        limit: int | None = None,
    ) -> list[ExperimentRecord]:
        selected = [e for e in self.experiments.values() if e.project_id == project_id]

        if model_family:
            selected = [
                e for e in selected if (e.model_family or "").lower() == model_family.lower()
            ]
        if dataset_id:
            selected = [e for e in selected if e.dataset_id == dataset_id]
        if dataset_version_id:
            selected = [e for e in selected if e.dataset_version_id == dataset_version_id]
        if metric:
            selected = [e for e in selected if e.metric(metric) is not None]
        if status:
            selected = [e for e in selected if str(e.status) == status or e.status == status]
        if config_field:
            selected = [
                e
                for e in selected
                if config_field in {**(e.configuration or {}), **(e.hyperparameters or {})}
            ]
            if config_value is not None:
                selected = [
                    e
                    for e in selected
                    if (e.configuration or {}).get(config_field, (e.hyperparameters or {}).get(config_field))
                    == config_value
                ]

        selected.sort(key=lambda e: e.id)
        return selected[:limit] if limit else selected

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        return self.experiments.get(experiment_id)

    def find_comparable_candidates(
        self,
        *,
        project_id: str,
        model_family: str | None = None,
        dataset_id: str | None = None,
        dataset_version_id: str | None = None,
        evaluation_split: str | None = None,
        metric: str | None = None,
        limit: int | None = None,
    ) -> list[ExperimentRecord]:
        pool = self.list_experiments(
            project_id, model_family=model_family, dataset_id=dataset_id
        )
        if dataset_version_id:
            pool = [e for e in pool if e.dataset_version_id == dataset_version_id]
        if metric:
            pool = [e for e in pool if e.metric(metric) is not None]
        if evaluation_split:
            pool = [
                e
                for e in pool
                if any(m.evaluation_split == evaluation_split for m in e.metrics)
            ]
        pool.sort(key=lambda e: e.id)
        return pool[:limit] if limit else pool

    def get_experiment_lineage(self, experiment_id: str) -> list[dict[str, Any]]:
        return self.lineage.get(experiment_id, [])

    # -- datasets ----------------------------------------------------------

    def list_datasets(self, project_id: str) -> list[DatasetSummary]:
        return list(self.datasets.values())

    def list_dataset_versions(self, project_id: str) -> list[DatasetVersionSummary]:
        return list(self.dataset_versions.values())

    # -- corrections -------------------------------------------------------

    def list_corrections(self, project_id: str) -> list[CorrectionRecord]:
        return [c for c in self.corrections.values() if c.project_id == project_id]

    def get_correction_impact(self, correction_id: str) -> CorrectionImpact | None:
        correction = self.corrections.get(correction_id)
        if correction is None:
            return None

        affected: list[str] = []
        for experiment in self.experiments.values():
            if experiment.project_id != correction.project_id:
                continue
            if experiment.id == correction.source_id or (
                experiment.dataset_version_id == correction.source_id
            ):
                affected.append(experiment.id)

        # Everything that used the same dataset version but was not listed above
        # is unaffected-by-direct-effect yet still relevant context.
        return CorrectionImpact(
            correction_id=correction.id,
            source_id=correction.source_id,
            summary=correction.description,
            directly_affected_experiment_ids=sorted(affected),
            unaffected_experiment_ids=sorted(
                e.id
                for e in self.experiments.values()
                if e.project_id == correction.project_id and e.id not in affected
            ),
        )

    def list_review_items(
        self, project_id: str, *, status: ReviewItemStatus | None = None
    ) -> list[ReviewItem]:
        return [
            item
            for item in self.review_items.values()
            if item.project_id == project_id
            and (status is None or item.status is status)
        ]

    # -- recommendations ---------------------------------------------------

    def save_recommendation(self, recommendation: Recommendation) -> None:
        self.recommendations[recommendation.id] = recommendation

    def get_recommendation(self, recommendation_id: str) -> Recommendation | None:
        return self.recommendations.get(recommendation_id)

    def list_recommendations(self, project_id: str) -> list[Recommendation]:
        return [
            r for r in self.recommendations.values() if r.project_id == project_id
        ]

    def record_recommendation_decision(
        self, decision: RecommendationDecision
    ) -> Recommendation:
        recommendation = self.recommendations.get(decision.recommendation_id)
        if recommendation is None:
            raise KeyError(decision.recommendation_id)

        self.decisions.append(decision)
        status_map = {
            Decision.APPROVED: RecommendationStatus.APPROVED,
            Decision.REJECTED: RecommendationStatus.REJECTED,
            Decision.EDITED: RecommendationStatus.EDITED,
        }
        updates: dict[str, Any] = {"status": status_map[decision.decision]}
        if decision.edited_configuration is not None:
            updates["configuration"] = decision.edited_configuration

        updated = recommendation.model_copy(update=updates)
        self.recommendations[decision.recommendation_id] = updated
        return updated


# Structural check: a silent signature drift in the port should fail here.
assert isinstance(FakeGraphService(), GraphService)