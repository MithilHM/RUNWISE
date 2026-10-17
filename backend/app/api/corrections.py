"""
Corrections & Impact Analysis API endpoints for RUNWISE.
Conforms to PRD Section 9:
- POST /api/v1/projects/{project_id}/corrections
- GET /api/v1/corrections/{correction_id}/impact
- GET /api/v1/projects/{project_id}/review-queue
- POST /api/v1/review-items/{review_item_id}/resolve
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException

from app.services.impact_analysis import ImpactAnalysisService
from app.graph.repository import GraphRepository
from app.models.contracts import (
    CorrectionCreate,
    CorrectionImpactResponse,
    ReviewItemResponse,
    ReviewItemResolve
)

router = APIRouter(tags=["corrections"])


def get_impact_service() -> ImpactAnalysisService:
    return ImpactAnalysisService()


def get_repository() -> GraphRepository:
    return GraphRepository()


@router.post(
    "/projects/{project_id}/corrections",
    response_model=CorrectionImpactResponse,
    summary="Record evidence correction and analyze downstream impact",
    description="Marks a dataset version, experiment, or claim as unreliable/invalidated, traverses graph dependencies, flags affected nodes for review, and returns dependency paths."
)
def record_correction(
    project_id: str,
    correction: CorrectionCreate,
    service: ImpactAnalysisService = Depends(get_impact_service)
) -> CorrectionImpactResponse:
    return service.analyze_and_apply_correction(correction, project_id=project_id)


@router.get(
    "/corrections/{correction_id}/impact",
    response_model=CorrectionImpactResponse,
    summary="Get impact details for a correction",
    description="Retrieves the recorded impact report, affected experiments, and review items associated with a correction."
)
def get_correction_impact(
    correction_id: str,
    service: ImpactAnalysisService = Depends(get_impact_service)
) -> CorrectionImpactResponse:
    impact = service.get_correction_impact(correction_id)
    if not impact:
        raise HTTPException(status_code=404, detail=f"Correction '{correction_id}' not found.")
    return impact


@router.get(
    "/projects/{project_id}/review-queue",
    response_model=List[ReviewItemResponse],
    summary="Get revalidation review queue",
    description="Returns list of review items for experiments impacted by evidence corrections that require researcher assessment."
)
def get_review_queue(
    project_id: str,
    repo: GraphRepository = Depends(get_repository)
) -> List[ReviewItemResponse]:
    return repo.get_review_queue(project_id=project_id)


@router.post(
    "/review-items/{review_item_id}/resolve",
    summary="Resolve a review item",
    description="Marks a review item as resolved or dismissed with optional resolution notes."
)
def resolve_review_item(
    review_item_id: str,
    resolution: ReviewItemResolve,
    repo: GraphRepository = Depends(get_repository)
):
    result = repo.resolve_review_item(
        review_item_id=review_item_id,
        status=resolution.status,
        notes=resolution.notes,
        resolved_by=resolution.resolved_by
    )
    if not result:
        raise HTTPException(status_code=404, detail=f"Review item '{review_item_id}' not found.")
    return result
