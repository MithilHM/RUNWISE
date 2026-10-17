"""Shared FastAPI dependencies for the recommendation and copilot routers.

These routers do not own ``app/main.py`` (the integration owner does), so they never
build their own collaborators. Instead they declare dependencies here, and the
integration owner registers one implementation on the application state:

.. code-block:: python

    app.state.graph_service = FalkorDBGraphService(...)   # graph layer
    app.include_router(recommendations.router)            # recommendations
    app.include_router(copilot.router)                   # copilot

That keeps the wiring point in one documented place and lets tests inject an
in-memory implementation without a database.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.api.errors import bad_request, graph_unavailable, not_found
from app.ports.graph import GraphService
from app.services.comparability import ComparabilityEngine
from app.services.llm_adapter import ExplanationWriter, build_llm_client
from app.services.recommendations import RankingConfig, RecommendationEngine

GRAPH_NOT_CONFIGURED = (
    "graph_service is not configured. The integration owner registers the FalkorDB-backed "
    "implementation on app.state.graph_service (see backend/app/api/deps.py)."
)


def get_graph_service(request: Request) -> GraphService:
    """Resolve the graph service, or fail with a structured 503."""
    service = getattr(request.app.state, "graph_service", None)
    if service is None:
        raise graph_unavailable(GRAPH_NOT_CONFIGURED)
    return service


def get_comparability_engine() -> ComparabilityEngine:
    return ComparabilityEngine()


def get_recommendation_engine(
    graph: Annotated[GraphService, Depends(get_graph_service)],
    comparability: Annotated[ComparabilityEngine, Depends(get_comparability_engine)],
) -> RecommendationEngine:
    return RecommendationEngine(graph, comparability=comparability, config=RankingConfig())


def get_explanation_writer(request: Request) -> ExplanationWriter:
    """Build the writer for this request.

    A client supplied on ``app.state`` wins, so tests can inject one; otherwise it
    is constructed from the environment, which yields the deterministic
    no-key path (PRD 15).
    """
    injected = getattr(request.app.state, "explanation_writer", None)
    if injected is not None:
        return injected
    return ExplanationWriter(build_llm_client())


GraphDep = Annotated[GraphService, Depends(get_graph_service)]
ComparabilityDep = Annotated[ComparabilityEngine, Depends(get_comparability_engine)]
EngineDep = Annotated[RecommendationEngine, Depends(get_recommendation_engine)]
WriterDep = Annotated[ExplanationWriter, Depends(get_explanation_writer)]


# Re-exported from ``app.api.errors`` so every router raises the same envelope.
__all__ = [
    "GRAPH_NOT_CONFIGURED",
    "get_graph_service",
    "get_comparability_engine",
    "get_recommendation_engine",
    "get_explanation_writer",
    "GraphDep",
    "ComparabilityDep",
    "EngineDep",
    "WriterDep",
    "not_found",
    "bad_request",
]