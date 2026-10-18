"""Runs router for the autonomous ML harness lifecycle.

Run creation, pause/resume, human review decisions, the "why" query, log
retrieval, and the SSE event stream.

The harness is wired to the real FalkorDB-backed implementations by default and
to the in-memory fakes only when `RUNWISE_FAKE=1` is set.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal, Optional
from fastapi import APIRouter, Body, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.api.errors import not_found, unprocessable
from app.models.contracts import (
    LogEntry,
    Proposal,
    RunCreate,
    RunState,
    WhyRecord,
)
from app.services import catalog
from app.services.harness import HarnessRun, HarnessService
from app.services.ingestion import IngestionError, get_ingestor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


def _require_dataset(dataset_id: str):
    """Loads the dataset profile, or 404s.

    A run against a dataset the harness has never seen would fail on its first
    trial with "Dataset unavailable", which reads like a data bug rather than a
    wrong id. Rejecting it here names the actual problem.
    """
    try:
        return get_ingestor().get_profile(dataset_id)
    except IngestionError as exc:
        if exc.code in ("unknown_dataset", "dataset_evicted"):
            raise not_found("dataset", dataset_id) from exc
        raise unprocessable(
            code=exc.code, message=exc.message, field_name="dataset_id"
        ) from exc


def _use_fakes() -> bool:
    """Whether to run against in-memory substitutes.

    Off unless `RUNWISE_FAKE=1`. The real FalkorDB store and the real trainer are
    the product; a fake that silently took their place would let the app appear to
    work while writing nothing to the graph.
    """
    return os.getenv("RUNWISE_FAKE", "").strip().lower() in ("1", "true", "yes")


def _build_harness_service() -> HarnessService:
    """Wires the harness to either the real implementations or the fakes."""
    if _use_fakes():
        logger.warning("RUNWISE_FAKE is set: the harness is running on in-memory fakes. "
                       "Nothing is written to FalkorDB and no model is actually trained.")
        from tests.fakes.fake_graph import FakeGraphStore
        from tests.fakes.fake_trainer import (
            FakeDecisionMaker,
            FakeTrainer,
            InMemoryRunLogger,
        )

        return HarnessService(
            trainer=FakeTrainer(),
            decision_maker=FakeDecisionMaker(),
            graph_store=FakeGraphStore(),
            run_logger=InMemoryRunLogger(),
        )

    from app.graph.repository import get_graph_store
    from app.services.executor import get_trainer
    from app.services.laya import get_laya
    from app.services.logs import get_logger_

    return HarnessService(
        trainer=get_trainer(),
        decision_maker=get_laya(),
        graph_store=get_graph_store(),
        run_logger=get_logger_(),
    )


harness_service = _build_harness_service()


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
    # Validate the metric against the *actual* task of the actual dataset.
    #
    # The previous check asked whether the string "multiclass" appeared inside the
    # dataset id, which is never true, so `pr_auc` on a multiclass dataset was
    # accepted and then failed its first trial — burning a budget step to report
    # something the request could have been rejected for.
    profile = _require_dataset(payload.dataset_id)

    if payload.metric not in catalog.available_metrics(profile.task):
        usable = ", ".join(catalog.available_metrics(profile.task))
        raise unprocessable(
            code="metric_incompatible",
            message=(
                f"{payload.metric} cannot be computed on this dataset: it is "
                f"{profile.task} with {profile.n_classes} classes. Use one of: {usable}."
            ),
            field_name="metric",
        )

    unknown = [m for m in payload.model_ids if m not in catalog.MODEL_IDS]
    if unknown:
        raise unprocessable(
            code="unknown_model",
            message=f"Unknown model(s): {', '.join(unknown)}. "
                    f"Available: {', '.join(catalog.MODEL_IDS)}.",
            field_name="model_ids",
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

    target = trial_id or harness_run.state.best_trial_id
    if not target:
        # The previous fallback invented "trial_001". A fabricated id resolves to
        # no decision, so the user got an empty explanation for a trial that never
        # existed. Better to say the run has produced nothing to explain.
        raise not_found("trial for run", run_id)

    try:
        return harness_service.decision_maker.explain(run_id, target)
    except KeyError as exc:
        raise not_found("decision for trial", target) from exc


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
