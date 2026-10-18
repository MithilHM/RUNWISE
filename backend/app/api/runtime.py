"""FastAPI Router for Autonomous Graph-Guided ML Experiment Runtime.

Endpoints:
- POST /api/runtime/start: Accepts RuntimeObjective, starts background loop, returns run_id.
- GET  /api/runtime/{run_id}/stream: Real-time Server-Sent Events (SSE) stream of RuntimeStepEvents.
- GET  /api/runtime/{run_id}/status: Returns execution tree, completed trials, best score, graph mutations.
- POST /api/runtime/{run_id}/ablation: Runs automated 10-trial comparison (Graph+Laya vs Random Search).
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional
from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, StreamingResponse

from app.api.errors import not_found
from app.models.runtime_contracts import (
    AblationResponse,
    RuntimeObjective,
    RuntimeStatusResponse,
    RuntimeStepEvent,
)
from app.services.runtime_loop import (
    create_runtime_loop,
    get_runtime_loop,
    runtime_registry,
)

router = APIRouter(prefix="/runtime", tags=["runtime"])


@router.post(
    "/start",
    status_code=status.HTTP_201_CREATED,
    summary="Start an autonomous optimization loop",
    description="Initializes and launches the closed-loop autonomous ML experiment loop in the background.",
)
async def start_runtime_loop(objective: RuntimeObjective = Body(...)) -> Dict[str, Any]:
    loop = create_runtime_loop(objective=objective)
    run_id = loop.start()
    return {
        "run_id": run_id,
        "status": "STARTED",
        "dataset_name": objective.dataset_name,
        "target_metric": objective.target_metric,
        "trial_budget": objective.trial_budget,
    }


@router.get(
    "/{run_id}/stream",
    summary="Real-time SSE event stream for an autonomous run",
    description="Yields RuntimeStepEvents via Server-Sent Events as trials progress through graph querying, evaluation, training, and graph mutation.",
)
async def stream_runtime_events(
    run_id: str,
    request: Request,
    poll: bool = Query(False, description="If true, return collected events as JSON instead of SSE stream"),
):
    loop = get_runtime_loop(run_id)
    if not loop:
        raise not_found("runtime_run", run_id)

    if poll:
        return [event.model_dump() for event in loop.events]

    async def event_generator():
        queue = loop.subscribe()
        try:
            while True:
                # Disconnect check
                if await request.is_disconnected():
                    break

                try:
                    event: RuntimeStepEvent = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {event.model_dump_json()}\n\n"
                    if event.stage == "COMPLETED" or loop.status in ("COMPLETED", "FAILED"):
                        break
                except asyncio.TimeoutError:
                    # Heartbeat comment to keep SSE connection alive
                    yield ": keep-alive\n\n"
                    if loop.status in ("COMPLETED", "FAILED") and queue.empty():
                        break
        finally:
            loop.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/{run_id}/status",
    response_model=RuntimeStatusResponse,
    summary="Get current runtime execution status and tree",
    description="Returns the execution tree, all completed trials, current best metric, and graph mutation statistics.",
)
async def get_runtime_status(run_id: str) -> RuntimeStatusResponse:
    loop = get_runtime_loop(run_id)
    if not loop:
        raise not_found("runtime_run", run_id)

    return RuntimeStatusResponse(
        run_id=loop.run_id,
        status=loop.status,
        objective=loop.objective,
        best_score=loop.best_score,
        best_hyperparameters=loop.best_hyperparameters,
        completed_trials=loop.completed_trials,
        execution_tree=loop.execution_tree,
        graph_mutation_stats=loop.graph_mutation_stats,
    )


@router.post(
    "/{run_id}/ablation",
    response_model=AblationResponse,
    summary="Run Graph+Laya vs Random Search ablation benchmark",
    description="Runs an automated comparative evaluation between multi-hop graph guided Laya vs uniform random search on the same budget.",
)
async def run_runtime_ablation(
    run_id: str,
    budget: int = Query(10, ge=2, le=30, description="Trial budget for ablation comparison"),
) -> AblationResponse:
    loop = get_runtime_loop(run_id)
    if not loop:
        raise not_found("runtime_run", run_id)

    res = await loop.run_ablation(budget=budget)
    return res


@router.get(
    "/{run_id}/handoffs",
    summary="Get multi-agent coordination handoffs",
    description="Returns chronological agent handoff transitions recorded in FalkorDB.",
)
async def get_runtime_handoffs(run_id: str) -> Dict[str, Any]:
    loop = get_runtime_loop(run_id)
    if not loop:
        raise not_found("runtime_run", run_id)

    handoffs = loop.graph_service.get_agent_handoffs(run_id)
    return {
        "run_id": run_id,
        "handoffs": handoffs,
        "count": len(handoffs),
    }
