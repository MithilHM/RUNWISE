"""In-memory FakeTrainer, FakeDecisionMaker, and InMemoryRunLogger for RUNWISE.

Defined in MEMBER_B.md Section 3.3.
Enables offline, unit, and local development with RUNWISE_FAKE=1.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import uuid
from typing import Any

from app.models.contracts import (
    HarnessDecision as Decision,
    EvidenceItem,
    GraphSnapshot,
    LogEntry,
    LogEvent,
    Proposal,
    RunState,
    TrialOutcome,
    WhyRecord,
)

from app.ports import DecisionMaker, RunLogger, Trainer

logger = logging.getLogger(__name__)


class FakeTrainer:
    """In-memory Trainer double returning synthetic but plausible scores."""

    _warned_once: bool = False

    def validate(self, model_id: str, params: dict[str, Any]) -> list[str]:
        errors: list[str] = []
        valid_models = {"xgboost", "lightgbm", "random_forest"}
        if model_id not in valid_models:
            errors.append(f"Unknown model_id '{model_id}', expected one of {sorted(valid_models)}")

        # Check common param constraints
        if "max_depth" in params and params["max_depth"] is not None:
            if isinstance(params["max_depth"], (int, float)) and params["max_depth"] <= 0:
                errors.append("max_depth must be > 0")
        if "learning_rate" in params:
            if isinstance(params["learning_rate"], (int, float)) and params["learning_rate"] <= 0:
                errors.append("learning_rate must be > 0")
        if "n_estimators" in params:
            if isinstance(params["n_estimators"], (int, float)) and params["n_estimators"] <= 0:
                errors.append("n_estimators must be > 0")

        return errors

    def train(
        self,
        dataset_id: str,
        model_id: str,
        params: dict[str, Any],
        metric: str,
        seed: int = 42,
    ) -> TrialOutcome:
        if not FakeTrainer._warned_once:
            logger.warning("FakeTrainer is active: returning synthetic evaluation metrics")
            FakeTrainer._warned_once = True

        validation_errors = self.validate(model_id, params)
        now_iso = datetime.now(timezone.utc).isoformat()
        trial_id = f"trial_{uuid.uuid4().hex[:8]}"

        if validation_errors:
            return TrialOutcome(
                trial_id=trial_id,
                run_id="",
                step=0,
                model_id=model_id,
                params=params,
                status="FAILED",
                metric_name=metric,
                metric_value=None,
                metrics={"source": None},
                failure_reason="; ".join(validation_errors),
                duration_sec=0.01,
                n_features=10,
                seed=seed,
                created_at=now_iso,
            )

        # Generate a deterministic pseudo-random score based on params hash
        key_str = f"{dataset_id}:{model_id}:{sorted(params.items())}:{seed}"
        hash_val = int(hashlib.md5(key_str.encode("utf-8")).hexdigest()[:8], 16)
        synthetic_score = round(0.72 + (hash_val % 2400) / 10000.0, 4)

        return TrialOutcome(
            trial_id=trial_id,
            run_id="",
            step=0,
            model_id=model_id,
            params=params,
            status="SUCCESS",
            metric_name=metric,
            metric_value=synthetic_score,
            metrics={"source": None, metric: synthetic_score},
            failure_reason=None,
            duration_sec=0.45,
            n_features=14,
            seed=seed,
            created_at=now_iso,
        )


class FakeDecisionMaker:
    """Mock Laya decision maker for offline harness operation."""

    def __init__(self) -> None:
        self.decisions_by_trial: dict[str, WhyRecord] = {}

    def propose(self, run: RunState, graph: GraphSnapshot) -> Proposal:
        model_id = run.model_ids[run.step % len(run.model_ids)] if run.model_ids else "xgboost"
        depth = 3 + (run.step % 6)
        lr = round(0.01 * (1 + (run.step % 10)), 3)
        n_est = 100 + (run.step * 25)
        params = {"max_depth": depth, "learning_rate": lr, "n_estimators": n_est}

        # Select citations from existing graph nodes if any
        citations: list[str] = [n.id for n in graph.nodes if n.type in ("Trial", "Setting", "Dataset")][:3]

        return Proposal(
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            step=run.step,
            model_id=model_id,
            params=params,
            why=f"Exploring depth={depth} and lr={lr} to test gradient flow on {model_id}.",
            strategy="exploit_basin" if run.step > 2 else "broad_explore",
            hypothesis=f"Increasing depth to {depth} improves boundary resolution without severe overfitting.",
            confidence=round(0.75 + (run.step % 20) / 100.0, 2),
            citations=citations,
        )

    def explain(self, trial_id: str) -> WhyRecord:
        if trial_id in self.decisions_by_trial:
            return self.decisions_by_trial[trial_id]

        decision = Decision(
            decision_id=f"dec_{uuid.uuid4().hex[:8]}",
            run_id="run_default",
            step=1,
            trial_id=trial_id,
            actor="laya",
            strategy="exploit_basin",
            why="Selected based on highest F1 in neighbouring parameter regime.",
            hypothesis="Controlled learning rate avoids oscillatory convergence.",
            confidence=0.88,
            citations=["dataset_001", "param_lr"],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        return WhyRecord(
            decision=decision,
            config={"max_depth": 6, "learning_rate": 0.05, "n_estimators": 200},
            evidence=[
                EvidenceItem(kind="Model", id="xgboost", label="XGBoost", detail="Gradient boosting base"),
                EvidenceItem(kind="Param", id="param_lr", label="learning_rate", detail="learning_rate = 0.05"),
                EvidenceItem(kind="Dataset", id="dataset_001", label="Dataset", detail="Stratified 80/20"),
            ],
            highlight=["dataset_001", "param_lr"],
        )


class InMemoryRunLogger:
    """In-memory RunLogger protocol implementation with live SSE queues."""

    def __init__(self) -> None:
        self.logs: list[LogEntry] = []
        self._seq: int = 0
        self._subscribers: dict[str, list[asyncio.Queue[LogEvent]]] = {}

    def append(
        self,
        run_id: str,
        level: str,
        stage: str,
        message: str,
        step: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> LogEntry:
        self._seq += 1
        entry = LogEntry(
            seq=self._seq,
            run_id=run_id,
            step=step,
            level=level,
            stage=stage,
            message=message,
            meta=meta or {},
            ts=datetime.now(timezone.utc).isoformat(),
        )
        self.logs.append(entry)

        # Broadcast to subscribers
        if run_id in self._subscribers:
            event = LogEvent(type="log", payload=entry.model_dump())
            dead = []
            for q in self._subscribers[run_id]:
                try:
                    q.put_nowait(event)
                except Exception:
                    dead.append(q)
            for d in dead:
                self.unsubscribe(run_id, d)

        return entry

    def history(self, run_id: str, after_seq: int = 0) -> list[LogEntry]:
        """Entries after `after_seq`, in order.

        Scoped to the run rather than returning everything: the real logger reads
        back from the graph filtered by `run_id`, and a fake that returned another
        run's lines would let a polling client double-count.
        """
        entries = [
            entry for entry in self.logs
            if entry.run_id == run_id and entry.seq > after_seq
        ]
        entries.sort(key=lambda e: e.seq)
        return entries

    def publish(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        """Broadcasts a non-log notification. Not recorded as a line, matching the
        real logger so a subscriber sees the same two kinds of message."""
        self.broadcast_event(run_id, event_type, payload)

    def broadcast_event(self, run_id: str, event_type: str, payload: dict[str, Any]) -> None:
        if run_id in self._subscribers:
            event = LogEvent(type=event_type, payload=payload)  # type: ignore
            dead = []
            for q in self._subscribers[run_id]:
                try:
                    q.put_nowait(event)
                except Exception:
                    dead.append(q)
            for d in dead:
                self.unsubscribe(run_id, d)

    def subscribe(self, run_id: str) -> asyncio.Queue[LogEvent]:
        q: asyncio.Queue[LogEvent] = asyncio.Queue()
        if run_id not in self._subscribers:
            self._subscribers[run_id] = []
        self._subscribers[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue[LogEvent]) -> None:
        if run_id in self._subscribers and q in self._subscribers[run_id]:
            self._subscribers[run_id].remove(q)


assert isinstance(FakeTrainer(), Trainer)
assert isinstance(FakeDecisionMaker(), DecisionMaker)
assert isinstance(InMemoryRunLogger(), RunLogger)
