"""Runs router for autonomous ML harness lifecycle.

Defined in MEMBER_B.md Section 4 (B2 — Routers).
Implements run creation, pause/resume, human review decision handling,
explanation why queries, log retrieval, and SSE event streaming.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Literal, Optional
from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.datasets import DATASET_STORE
from app.api.errors import not_found, unprocessable
from app.models.contracts import (
    LogEntry,
    Proposal,
    RunCreate,
    RunState,
    WhyRecord,
)
from app.services.harness import HarnessRun, HarnessService
from tests.fakes.fake_graph import FakeGraphStore
from tests.fakes.fake_trainer import FakeDecisionMaker, FakeTrainer, InMemoryRunLogger

router = APIRouter(prefix="/runs", tags=["runs"])

# Global harness service instance
_fake_graph = FakeGraphStore()
_fake_trainer = FakeTrainer()
_fake_decision = FakeDecisionMaker()
_fake_logger = InMemoryRunLogger()

harness_service = HarnessService(
    trainer=_fake_trainer,
    decision_maker=_fake_decision,
    graph_store=_fake_graph,
    run_logger=_fake_logger,
)


class RunsListResponse(BaseModel):
    runs: list[RunState]


class PendingProposalResponse(BaseModel):
    pending: Optional[Proposal] = None


class DecideRequest(BaseModel):
    step: int
    action: Literal["approve", "reject"]
    hyperparameters: Optional[dict[str, Any]] = None
    note: Optional[str] = None


class LogsResponse(BaseModel):
    entries: list[LogEntry]


@router.get("", response_model=RunsListResponse, summary="List all harness runs")
async def list_runs() -> RunsListResponse:
    all_runs = [h.state for h in harness_service.runs.values()]
    return RunsListResponse(runs=all_runs)


@router.post("", response_model=RunState, status_code=status.HTTP_201_CREATED, summary="Create and start a new harness run")
async def create_run(payload: RunCreate) -> RunState:
    # Validate metric compatibility with dataset
    if payload.dataset_id in DATASET_STORE:
        profile, _ = DATASET_STORE[payload.dataset_id]
        if payload.metric == "pr_auc" and profile.task == "multiclass":
            raise unprocessable(
                code="metric_incompatible",
                message="pr_auc is binary-only and cannot be computed on multiclass datasets",
                field_name="metric",
            )
    elif payload.metric == "pr_auc" and "multiclass" in payload.dataset_id:
        raise unprocessable(
            code="metric_incompatible",
            message="pr_auc is binary-only and cannot be computed on multiclass datasets",
            field_name="metric",
        )

    harness_run = harness_service.create_run(
        dataset_id=payload.dataset_id,
        model_ids=payload.model_ids,
        mode=payload.mode,
        metric=payload.metric,
        budget=payload.budget,
    )
    # Start loop as background task; return immediately
    asyncio.create_task(harness_run.start())
    return harness_run.state


@router.get("/{run_id}", response_model=RunState, summary="Get state of a run")
async def get_run(run_id: str) -> RunState:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)
    return harness_run.state


@router.post("/{run_id}/pause", response_model=RunState, summary="Pause a running harness")
async def pause_run(run_id: str) -> RunState:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)
    return harness_run.pause()


@router.post("/{run_id}/resume", response_model=RunState, summary="Resume a paused harness")
async def resume_run(run_id: str) -> RunState:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)
    return harness_run.resume()


@router.get("/{run_id}/pending", response_model=PendingProposalResponse, summary="Get pending proposal under human review")
async def get_pending_proposal(run_id: str) -> PendingProposalResponse:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)
    return PendingProposalResponse(pending=harness_run.pending_proposal)


@router.post("/{run_id}/decide", response_model=RunState, summary="Submit researcher decision (approve/reject/edit)")
async def decide_proposal(run_id: str, payload: DecideRequest) -> RunState:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)

    try:
        updated_state = await harness_run.decide(
            step=payload.step,
            action=payload.action,
            hyperparameters=payload.hyperparameters,
            note=payload.note,
        )
        return updated_state
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=[{"code": "invalid_decision", "message": str(err), "field": "step"}],
        )


@router.get("/{run_id}/why", response_model=WhyRecord, summary="Get F7 why explanation record for a trial")
async def get_why_record(run_id: str, trial_id: Optional[str] = Query(None)) -> WhyRecord:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)

    target_trial_id = trial_id or harness_run.state.best_trial_id or "trial_001"
    why_record = harness_service.decision_maker.explain(target_trial_id)
    return why_record


@router.get("/{run_id}/logs", response_model=LogsResponse, summary="Fetch sequential run logs")
async def get_logs(run_id: str, after: int = Query(0)) -> LogsResponse:
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)

    entries = harness_service.graph_store.read_logs(run_id=run_id, after_seq=after)
    return LogsResponse(entries=entries)


@router.get("/{run_id}/events", summary="SSE stream of live harness events")
async def stream_events(run_id: str, request: Request):
    harness_run = harness_service.get_run(run_id)
    if not harness_run:
        raise not_found("run", run_id)

    async def event_generator():
        q = harness_service.logger.subscribe(run_id)
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(q.get(), timeout=10.0)
                    yield f"data: {event.model_dump_json()}\n\n"
                    if harness_run.state.status in ("COMPLETED", "FAILED"):
                        break
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    if harness_run.state.status in ("COMPLETED", "FAILED") and q.empty():
                        break
        finally:
            harness_service.logger.unsubscribe(run_id, q)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
