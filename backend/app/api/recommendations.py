"""Recommendation API (PRD 9, FR-5, FR-9).

Area: recommendations API. Endpoints:

    POST /api/v1/projects/{project_id}/recommendations
    GET  /api/v1/recommendations/{recommendation_id}
    POST /api/v1/recommendations/{recommendation_id}/decision

Router registration is the integration owner's step in ``app/main.py``:

.. code-block:: python

    app.include_router(recommendations.router)
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Path, status

from app.api.deps import EngineDep, GraphDep, bad_request, not_found
from app.models.contracts import (
    Decision,
    Recommendation,
    RecommendationDecisionRequest,
    RecommendationRequest,
    RecommendationSet,
)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.post(
    "/projects/{project_id}/recommendations",
    response_model=RecommendationSet,
    status_code=status.HTTP_200_OK,
    summary="Propose the next experiment under a compute budget",
)
def create_recommendations(
    engine: EngineDep,
    body: RecommendationRequest,
    project_id: Annotated[str, Path(min_length=1)],
) -> RecommendationSet:
    """FR-5.

    An empty ``recommendations`` list is a valid answer, not an error: a project
    with no records, or one whose every record is disputed, cannot support a
    grounded proposal. The reason is carried in ``uncertainties``.
    """
    return engine.generate(project_id, body)


@router.get(
    "/recommendations/{recommendation_id}",
    response_model=Recommendation,
    summary="Fetch one recommendation",
)
def get_recommendation(
    graph: GraphDep,
    recommendation_id: Annotated[str, Path(min_length=1)],
) -> Recommendation:
    found = graph.get_recommendation(recommendation_id)
    if found is None:
        raise not_found("recommendation", recommendation_id)
    return found


@router.post(
    "/recommendations/{recommendation_id}/decision",
    response_model=Recommendation,
    summary="Approve, reject or edit a proposal (FR-9)",
)
def decide_recommendation(
    engine: EngineDep,
    body: RecommendationDecisionRequest,
    recommendation_id: Annotated[str, Path(min_length=1)],
) -> Recommendation:
    """FR-9: recommendations are advisory.

    No training is triggered here. An edit replaces the proposed configuration so
    the exported JSON matches what the researcher agreed to.
    """
    if body.decision is Decision.EDITED and body.edited_configuration is None:
        # An "edit" with no new configuration is a no-op that would record a
        # decision the researcher did not actually make.
        raise bad_request(
            "missing_edited_configuration",
            "decision 'edited' requires edited_configuration describing the change.",
            field_name="edited_configuration",
        )

    try:
        return engine.decide(
            recommendation_id,
            body.decision,
            decided_by=body.decided_by,
            note=body.note,
            edited_configuration=body.edited_configuration,
        )
    except KeyError as exc:
        raise not_found("recommendation", str(exc)) from exc


@router.get(
    "/projects/{project_id}/recommendations",
    response_model=list[Recommendation],
    summary="List stored recommendations for a project",
)
def list_recommendations(
    graph: GraphDep,
    project_id: Annotated[str, Path(min_length=1)],
) -> Any:
    return graph.list_recommendations(project_id)


# Kept importable so ``Body`` stays available for later router additions
# without editing this file.
__all__ = ["router", "create_recommendations", "get_recommendation",
           "decide_recommendation", "list_recommendations", "Body"]