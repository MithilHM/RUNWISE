"""Experiments, lineage, comparability and graph endpoints for RUNWISE.

PRD Section 9:
- GET  /api/v1/projects/{project_id}/experiments
- GET  /api/v1/experiments/{experiment_id}
- GET  /api/v1/experiments/{experiment_id}/comparability
- GET  /api/v1/experiments/{experiment_id}/lineage
- GET  /api/v1/experiments/{experiment_id}/related
- GET  /api/v1/projects/{project_id}/graph/summary
- GET  /api/v1/projects/{project_id}/graph

Every read takes an optional ``project_id`` so the caller can bind the lookup to a
project. Without it the query still resolves the record's own ``project_id``; the
scope belongs in the graph query, not in a filter applied afterwards (PRD 12).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query

from app.api.errors import not_found
from app.api.deps import ComparabilityDep
from app.graph.repository import GraphRepository
from app.models.contracts import (
    ComparabilityAssessment,
    ExperimentSummary,
    ExperimentDetail,
    ExperimentLineageResponse,
    ProjectGraphSummary,
    ProjectGraphResponse,
)

router = APIRouter(tags=["experiments"])


def get_repository() -> GraphRepository:
    return GraphRepository()


@router.get(
    "/projects/{project_id}/experiments",
    response_model=List[ExperimentSummary],
    summary="List experiments in a project",
    description=(
        "FR-2 experiment history. Filters run inside the graph query: model family, "
        "dataset id or version, metric, evaluation split, run status, review status, "
        "creation window, and a configuration field/value pair."
    ),
)
def list_experiments(
    project_id: str,
    model_family: Optional[str] = Query(None, description="Filter by model family"),
    dataset_id: Optional[str] = Query(None, description="Filter by dataset node id"),
    dataset_version_id: Optional[str] = Query(None, description="Filter by dataset version node id"),
    metric: Optional[str] = Query(None, description="Keep only runs that recorded this metric"),
    evaluation_split: Optional[str] = Query(None, description="Keep only runs measured on this split"),
    status: Optional[str] = Query(None, description="Filter by run status: completed, failed, ..."),
    quality_status: Optional[str] = Query(
        None, description="Filter by review status: valid, needs_review, disputed"
    ),
    config_field: Optional[str] = Query(None, description="Configuration key to match"),
    config_value: Optional[str] = Query(
        None, description="Value that configuration key must have, as JSON"
    ),
    created_after: Optional[str] = Query(None, description="ISO timestamp lower bound"),
    created_before: Optional[str] = Query(None, description="ISO timestamp upper bound"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repo: GraphRepository = Depends(get_repository),
) -> List[ExperimentSummary]:
    parsed_value: Any = config_value
    if config_value is not None and config_value != "":
        import json

        try:
            parsed_value = json.loads(config_value)
        except ValueError:
            parsed_value = config_value

    records = repo.list_experiments(
        project_id,
        model_family=model_family,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        metric=metric,
        evaluation_split=evaluation_split,
        status=status,
        quality_status=quality_status,
        config_field=config_field,
        config_value=parsed_value,
        created_after=created_after,
        created_before=created_before,
        limit=limit + offset if offset else limit,
    )
    return [ExperimentSummary.model_validate(r.model_dump()) for r in records[offset:]]


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetail,
    summary="Get full experiment details",
    description=(
        "Returns run metadata, configuration, metrics, dataset identity, "
        "parent/child lineage, claims and open review items."
    ),
)
def get_experiment_detail(
    experiment_id: str,
    project_id: Optional[str] = Query(None, description="Bind the read to one project"),
    repo: GraphRepository = Depends(get_repository),
) -> ExperimentDetail:
    exp = repo.get_experiment_detail(experiment_id, project_id=project_id)
    if not exp:
        raise not_found("experiment", experiment_id)
    return exp


@router.get(
    "/experiments/{experiment_id}/comparability",
    response_model=ComparabilityAssessment,
    summary="Assess comparability (FR-4)",
    description=(
        "Compares one run against a peer and returns a defined status plus the "
        "reason codes behind it. The score is a weighted heuristic ordering aid, "
        "not a probability."
    ),
)
def get_experiment_comparability(
    experiment_id: str,
    compare_experiment_id: Optional[str] = Query(
        None, description="Peer to compare against. Defaults to a best-effort peer."
    ),
    metric_name: Optional[str] = Query(None, description="Restrict the assessment to one metric"),
    project_id: Optional[str] = Query(None, description="Bind the read to one project"),
    repo: GraphRepository = Depends(get_repository),
    comparability: ComparabilityDep = None,  # type: ignore[assignment]
) -> ComparabilityAssessment:
    left = repo.get_experiment(experiment_id, project_id=project_id)
    if not left:
        raise not_found("experiment", experiment_id)

    if compare_experiment_id:
        right = repo.get_experiment(compare_experiment_id, project_id=left.project_id)
        if not right:
            raise not_found("experiment", compare_experiment_id)
    else:
        # No peer named: pick one the graph says shares this run's dataset version,
        # and say so in the assessment rather than returning a verdict about nothing.
        candidates = repo.find_comparable_candidates(
            project_id=left.project_id,
            model_family=left.model_family,
            dataset_version_id=left.dataset_version_id,
        )
        right = next((c for c in candidates if c.id != left.id), None)
        if right is None:
            raise not_found("comparable_experiment_for", experiment_id)

    return comparability.assess(left, right, metric_name=metric_name)


@router.get(
    "/experiments/{experiment_id}/lineage",
    response_model=ExperimentLineageResponse,
    summary="Get experiment lineage tree",
    description="Bounded ancestor and descendant traversal through PARENT_OF relationships.",
)
def get_experiment_lineage(
    experiment_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Maximum traversal depth"),
    project_id: Optional[str] = Query(None, description="Bind the read to one project"),
    repo: GraphRepository = Depends(get_repository),
) -> ExperimentLineageResponse:
    lineage = repo.get_experiment_lineage_graph(experiment_id, max_depth=max_depth)
    if not lineage.nodes or (project_id and lineage.nodes[0].id and
                            repo.get_experiment(experiment_id, project_id=project_id) is None):
        raise not_found("experiment", experiment_id)
    return lineage


@router.get(
    "/experiments/{experiment_id}/related",
    response_model=Dict[str, List[str]],
    summary="Find related experiments",
    description=(
        "Runs sharing this one's dataset version, configuration, parent lineage or "
        "explicit comparison. Restricted to the same project."
    ),
)
def get_related_experiments(
    experiment_id: str,
    project_id: Optional[str] = Query(None, description="Bind the read to one project"),
    repo: GraphRepository = Depends(get_repository),
) -> Dict[str, List[str]]:
    if repo.get_experiment(experiment_id, project_id=project_id) is None:
        raise not_found("experiment", experiment_id)
    return repo.find_related_experiments(experiment_id)


@router.get(
    "/projects/{project_id}/graph/summary",
    response_model=ProjectGraphSummary,
    summary="Get project graph entity counts",
)
def get_project_graph_summary(
    project_id: str,
    repo: GraphRepository = Depends(get_repository),
) -> ProjectGraphSummary:
    summary = repo.get_project_graph_summary(project_id)
    if summary is None:
        raise not_found("project", project_id)
    return summary


@router.get(
    "/projects/{project_id}/graph",
    response_model=ProjectGraphResponse,
    summary="Get full graph structure for visualization",
    description=(
        "Real nodes and edges for the lineage and impact screen. ``truncated`` is set "
        "when the node budget cut the traversal so the UI can say the view is partial."
    ),
)
def get_project_graph(
    project_id: str,
    limit: int = Query(500, ge=1, le=2000),
    repo: GraphRepository = Depends(get_repository),
) -> ProjectGraphResponse:
    if repo.get_project(project_id) is None:
        raise not_found("project", project_id)
    return repo.get_project_graph(project_id, limit=limit)
