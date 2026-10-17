"""
Projects API endpoints for RUNWISE.
Conforms to PRD Section 9:
- POST /api/v1/projects
- GET /api/v1/projects
- GET /api/v1/projects/{project_id}
"""

from typing import List
from fastapi import APIRouter, Depends

from app.api.errors import not_found
from app.graph.repository import GraphRepository
from app.models.contracts import ProjectCreate, ProjectResponse

router = APIRouter(tags=["projects"])


def get_repository() -> GraphRepository:
    return GraphRepository()


@router.post(
    "/projects",
    response_model=ProjectResponse,
    summary="Create or update project"
)
def create_project(
    project: ProjectCreate,
    repo: GraphRepository = Depends(get_repository)
) -> ProjectResponse:
    res = repo.create_or_update_project(
        project_id=project.id,
        name=project.name,
        description=project.description
    )
    stored = repo.get_project(project.id) or {}
    return ProjectResponse(
        id=project.id,
        name=res.get("name", project.name),
        description=res.get("description"),
        created_at=str(stored.get("created_at")) if stored.get("created_at") else None,
        experiment_count=stored.get("experiment_count", 0)
    )


@router.get(
    "/projects",
    response_model=List[ProjectResponse],
    summary="List all projects"
)
def list_projects(
    repo: GraphRepository = Depends(get_repository)
) -> List[ProjectResponse]:
    projects = repo.list_projects()
    return [
        ProjectResponse(
            id=p["id"],
            name=p.get("name", p["id"]),
            description=p.get("description"),
            created_at=p.get("created_at"),
            experiment_count=p.get("experiment_count", 0)
        )
        for p in projects
    ]


@router.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    summary="Get project by ID"
)
def get_project(
    project_id: str,
    repo: GraphRepository = Depends(get_repository)
) -> ProjectResponse:
    p = repo.get_project(project_id)
    if not p:
        raise not_found("project", project_id)
    return ProjectResponse(
        id=p["id"],
        name=p.get("name", p["id"]),
        description=p.get("description"),
        created_at=str(p["created_at"]) if p.get("created_at") else None,
        experiment_count=p.get("experiment_count", 0)
    )
