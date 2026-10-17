"""Graph access port: the contract between the analysis engines and the graph layer.

PRD 21, Rule 3 requires that the comparability and recommendation engines must not
depend on the internal implementation details of the graph queries, and that the
two sides communicate through a small graph service interface.

This module is a :class:`typing.Protocol`, so it carries no FalkorDB import and
no query text. The graph layer implements it against ``backend/app/graph/**``; the
analysis engines consume it. Neither side edits the other's files.

Rules for this interface:
  * One method per analysis need, never a generic ``run_cypher`` escape hatch.
    PRD 9 prohibits unrestricted frontend-supplied Cypher, and the same rule
    applies in-process: the set of traversals is finite, auditable and tested.
  * Every method is project-scoped. Cross-project reads must fail, not filter.
  * Read methods return contract models, never raw nodes.
  * If an operation is missing, agree the signature here first, then implement
    it on both sides (PRD 21, Rule 3).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from app.models.contracts import (
    CorrectionImpact,
    CorrectionRecord,
    DatasetSummary,
    DatasetVersionSummary,
    ExperimentRecord,
    Recommendation,
    RecommendationDecision,
    ReviewItem,
    ReviewItemStatus,
)


@runtime_checkable
class GraphService(Protocol):
    """Read/write operations the graph layer provides over FalkorDB."""

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
        """FR-2 experiment history with the documented filters.

        Implementations must apply the project scope to the query itself, not
        filter afterwards.
        """
        ...

    def get_experiment(self, experiment_id: str) -> ExperimentRecord | None:
        """Return one experiment, or ``None`` when it does not exist."""
        ...

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
        """Cheap candidate preselection.

        The graph layer should filter on indexed identity fields only. Full comparability
        reasoning belongs to this area; this is a narrowing step, not a verdict.
        """
        ...

    def get_experiment_lineage(self, experiment_id: str) -> list[dict[str, Any]]:
        """Return lineage edges as ``{"from", "to", "type"}`` triples."""
        ...

    # -- datasets ----------------------------------------------------------

    def list_datasets(self, project_id: str) -> list[DatasetSummary]:
        ...

    def list_dataset_versions(self, project_id: str) -> list[DatasetVersionSummary]:
        """Include versions whose ``quality_status`` is DISPUTED."""
        ...

    # -- corrections and review (FR-6) ------------------------------------

    def list_corrections(self, project_id: str) -> list[CorrectionRecord]:
        ...

    def get_correction_impact(self, correction_id: str) -> CorrectionImpact | None:
        """Bounded traversal with explicit paths, direct vs indirect separated."""
        ...

    def list_review_items(
        self, project_id: str, *, status: ReviewItemStatus | None = None
    ) -> list[ReviewItem]:
        ...

    # -- recommendations (FR-5 persistence) -------------------------------

    def save_recommendation(self, recommendation: Recommendation) -> None:
        ...

    def get_recommendation(self, recommendation_id: str) -> Recommendation | None:
        ...

    def list_recommendations(self, project_id: str) -> list[Recommendation]:
        ...

    def record_recommendation_decision(
        self, decision: RecommendationDecision
    ) -> Recommendation:
        """Apply an approve/reject/edit decision and return the updated record."""
        ...


class GraphServiceUnavailable(RuntimeError):
    """Raised when no graph service is wired into the application.

    The integration owner registers the FalkorDB-backed implementation on
    ``app.state.graph_service`` (see ``app/api/deps.py``).
    """