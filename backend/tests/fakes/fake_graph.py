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

        # A run's validity also rests on what it was derived from: everything
        # descending from an affected run is affected transitively, even when
        # its own dataset version is clean. Mirrors the repository's PARENT_OF
        # traversal (the repository bounds the hop depth; the histories this
        # fake serves stay far below that bound, so the two agree in practice).
        children: dict[str, list[str]] = {}
        for experiment in self.experiments.values():
            if experiment.project_id != correction.project_id:
                continue
            if experiment.parent_experiment_id:
                children.setdefault(experiment.parent_experiment_id, []).append(
                    experiment.id
                )

        descendants: list[str] = []
        touched = set(affected)
        frontier = list(touched)
        while frontier:
            current = frontier.pop()
            for child_id in children.get(current, []):
                if child_id not in touched:
                    touched.add(child_id)
                    descendants.append(child_id)
                    frontier.append(child_id)
        descendants.sort()

        # Everything that used the same dataset version but was not listed above
        # is unaffected-by-direct-effect yet still relevant context.
        return CorrectionImpact(
            correction_id=correction.id,
            source_id=correction.source_id,
            summary=correction.description,
            directly_affected_experiment_ids=sorted(affected),
            indirectly_affected_experiment_ids=descendants,
            unaffected_experiment_ids=sorted(
                e.id
                for e in self.experiments.values()
                if e.project_id == correction.project_id and e.id not in touched
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


# ---------------------------------------------------------------------------
# FakeGraphStore for Member B (MEMBER_B.md Section 3.3)
# ---------------------------------------------------------------------------

from app.models.contracts import (  # noqa: E402
    DatasetProfile,
    Decision as HarnessDecision,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    LogEntry,
    RunState,
    TrialOutcome,
)
from app.ports import GraphStore  # noqa: E402


class FakeGraphStore:
    """In-memory GraphStore for the autonomous ML harness.

    Enforces no Cypher, maintains nodes and edges in dicts, supports
    read_logs(after_seq), write_trial with CITES edges from decision.citations.
    """

    def __init__(self) -> None:
        self.datasets: dict[str, DatasetProfile] = {}
        self.runs: dict[str, RunState] = {}
        self.trials: dict[str, TrialOutcome] = {}
        self.decisions: dict[str, HarnessDecision] = {}
        self.logs: list[LogEntry] = []
        self.nodes: dict[str, GraphNode] = {}
        self.edges: list[GraphEdge] = []

    def write_dataset(self, profile: DatasetProfile) -> None:
        self.datasets[profile.id] = profile
        self.nodes[profile.id] = GraphNode(
            id=profile.id,
            label=profile.name,
            type="Dataset",
            props={
                "n_rows": profile.n_rows,
                "n_cols": profile.n_cols,
                "task": profile.task,
            },
        )

    def write_run(self, run: RunState) -> None:
        self.runs[run.run_id] = run
        self.nodes[run.run_id] = GraphNode(
            id=run.run_id,
            label=f"Run {run.run_id[:8]}",
            type="Run",
            props={
                "mode": run.mode,
                "status": run.status,
                "metric": run.metric_name,
                "budget": run.budget,
            },
        )
        if run.dataset_id in self.nodes:
            self.edges.append(
                GraphEdge(
                    source=run.run_id,
                    target=run.dataset_id,
                    type="EVALUATED_ON",
                    props={},
                )
            )

    def write_trial(self, trial: TrialOutcome, decision: HarnessDecision) -> None:
        self.trials[trial.trial_id] = trial
        self.decisions[decision.decision_id] = decision

        # Trial node
        self.nodes[trial.trial_id] = GraphNode(
            id=trial.trial_id,
            label=f"{trial.model_id} (s{trial.step})",
            type="Trial",
            props={
                "step": trial.step,
                "model_id": trial.model_id,
                "status": trial.status,
                "metric_value": trial.metric_value,
                "params": trial.params,
            },
        )
        # Decision node
        self.nodes[decision.decision_id] = GraphNode(
            id=decision.decision_id,
            label=f"Decision ({decision.actor})",
            type="Decision",
            props={
                "actor": decision.actor,
                "strategy": decision.strategy,
                "why": decision.why,
                "confidence": decision.confidence,
            },
        )
        # Connect Decision -> Trial
        self.edges.append(
            GraphEdge(
                source=decision.decision_id,
                target=trial.trial_id,
                type="PRODUCED",
                props={},
            )
        )
        # Connect Trial -> Run
        if trial.run_id in self.nodes:
            self.edges.append(
                GraphEdge(
                    source=trial.trial_id,
                    target=trial.run_id,
                    type="PART_OF",
                    props={},
                )
            )
        # Create CITES edges from decision.citations
        for cited_id in decision.citations:
            self.edges.append(
                GraphEdge(
                    source=decision.decision_id,
                    target=cited_id,
                    type="CITES",
                    props={},
                )
            )

    def write_log(self, entry: LogEntry) -> None:
        self.logs.append(entry)

    def read_graph(self, run_id: str | None = None, dataset_id: str | None = None) -> GraphSnapshot:
        if not run_id and not dataset_id:
            return GraphSnapshot(
                nodes=list(self.nodes.values())[:500],
                edges=self.edges[:1000],
            )
        matched_node_ids = set()
        if run_id and run_id in self.nodes:
            matched_node_ids.add(run_id)
        if dataset_id and dataset_id in self.nodes:
            matched_node_ids.add(dataset_id)

        for t in self.trials.values():
            if run_id and t.run_id == run_id:
                matched_node_ids.add(t.trial_id)
        for d in self.decisions.values():
            if run_id and d.run_id == run_id:
                matched_node_ids.add(d.decision_id)
                for c in d.citations:
                    matched_node_ids.add(c)

        relevant_nodes = [n for n in self.nodes.values() if n.id in matched_node_ids]
        relevant_edges = [
            e for e in self.edges if e.source in matched_node_ids and e.target in matched_node_ids
        ]
        return GraphSnapshot(nodes=relevant_nodes, edges=relevant_edges)

    def read_logs(self, run_id: str, after_seq: int = 0) -> list[LogEntry]:
        return [
            log
            for log in self.logs
            if log.run_id == run_id and log.seq > after_seq
        ]


assert isinstance(FakeGraphStore(), GraphStore)