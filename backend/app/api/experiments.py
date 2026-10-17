"""
Experiments & Lineage API endpoints for RUNWISE.
Conforms to PRD Section 9:
- GET /api/v1/projects/{project_id}/experiments
- GET /api/v1/experiments/{experiment_id}
- GET /api/v1/experiments/{experiment_id}/lineage
- GET /api/v1/experiments/{experiment_id}/related
- GET /api/v1/projects/{project_id}/graph/summary
- GET /api/v1/projects/{project_id}/graph
"""

from typing import List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, Query

from app.graph.repository import GraphRepository
from app.models.contracts import (
    ExperimentSummary,
    ExperimentDetail,
    ExperimentLineageResponse,
    ProjectGraphSummary,
    ProjectGraphResponse
)

router = APIRouter(tags=["experiments"])


def get_repository() -> GraphRepository:
    return GraphRepository()


@router.get(
    "/projects/{project_id}/experiments",
    response_model=List[ExperimentSummary],
    summary="List experiments in a project",
    description="Retrieves experiment history with metrics, filterable by model family, dataset version, and execution/quality status."
)
def list_experiments(
    project_id: str,
    model_family: Optional[str] = Query(None, description="Filter by model family, e.g. transformer, gradient_boosting"),
    dataset_version_id: Optional[str] = Query(None, description="Filter by dataset version ID"),
    status: Optional[str] = Query(None, description="Filter by run status: completed, failed, etc."),
    quality_status: Optional[str] = Query(None, description="Filter by quality status: valid, needs_review, disputed"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    repo: GraphRepository = Depends(get_repository)
) -> List[ExperimentSummary]:
    return repo.list_experiments(
        project_id=project_id,
        model_family=model_family,
        dataset_version_id=dataset_version_id,
        status=status,
        quality_status=quality_status,
        limit=limit,
        offset=offset
    )


@router.get(
    "/experiments/{experiment_id}",
    response_model=ExperimentDetail,
    summary="Get full experiment details",
    description="Returns detailed experiment run metadata, configuration parameters, metrics, parent/child lineage, and quality status."
)
def get_experiment_detail(
    experiment_id: str,
    repo: GraphRepository = Depends(get_repository)
) -> ExperimentDetail:
    exp = repo.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return exp


@router.get(
    "/experiments/{experiment_id}/lineage",
    response_model=ExperimentLineageResponse,
    summary="Get experiment lineage tree",
    description="Traverses parent and child experiment relationships up to max_depth to provide a full directed lineage graph."
)
def get_experiment_lineage(
    experiment_id: str,
    max_depth: int = Query(5, ge=1, le=10, description="Max depth of ancestor/descendant traversal"),
    repo: GraphRepository = Depends(get_repository)
) -> ExperimentLineageResponse:
    lineage = repo.get_experiment_lineage(experiment_id, max_depth=max_depth)
    if not lineage.nodes:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return lineage


@router.get(
    "/experiments/{experiment_id}/related",
    response_model=Dict[str, List[str]],
    summary="Find related experiments",
    description="Identifies experiments sharing the same dataset version, configuration, parent lineage, or comparisons."
)
def get_related_experiments(
    experiment_id: str,
    repo: GraphRepository = Depends(get_repository)
) -> Dict[str, List[str]]:
    exp = repo.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return repo.find_related_experiments(experiment_id)


@router.get(
    "/projects/{project_id}/graph/summary",
    response_model=ProjectGraphSummary,
    summary="Get project graph entity counts",
    description="Returns count summaries for nodes and relationships belonging to the project graph."
)
def get_project_graph_summary(
    project_id: str,
    repo: GraphRepository = Depends(get_repository)
) -> ProjectGraphSummary:
    return repo.get_project_graph_summary(project_id)


@router.get(
    "/projects/{project_id}/graph",
    response_model=ProjectGraphResponse,
    summary="Get full graph structure for visualization",
    description="Returns all nodes and directed relationships for rendering interactive lineage and impact graphs."
)
def get_project_graph(
    project_id: str,
    limit: int = Query(500, ge=1, le=2000),
    repo: GraphRepository = Depends(get_repository)
) -> ProjectGraphResponse:
    return repo.get_project_graph(project_id, limit=limit)
