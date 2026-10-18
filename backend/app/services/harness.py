"""Autonomous ML Harness Loop for RUNWISE.

Defined in MEMBER_B.md Section 4 (B1 — The harness loop).
Manages state machine transitions, human-in-the-loop interventions,
and graph mutations across trials.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Literal, Optional
import uuid

from app.models.contracts import (
    HarnessDecision as Decision,
    GraphSnapshot,
    Proposal,
    RunState,
    TrialOutcome,
)

from app.ports import DecisionMaker, GraphStore, RunLogger, Trainer

logger = logging.getLogger(__name__)


class HarnessRun:
    """Represents a single execution of the autonomous/human harness state machine."""

    def __init__(
        self,
        run_state: RunState,
        trainer: Trainer,
        decision_maker: DecisionMaker,
        graph_store: GraphStore,
        run_logger: RunLogger,
    ) -> None:
        self.state = run_state
        self.trainer = trainer
        self.decision_maker = decision_maker
        self.graph_store = graph_store
        self.logger = run_logger

        self.pending_proposal: Optional[Proposal] = None
        self._pause_requested: bool = False
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()
        self._task: Optional[asyncio.Task] = None
        self._step_lock: asyncio.Lock = asyncio.Lock()
        self._decided_steps: set[int] = set()
        self._rejection_notes: list[str] = []

    @property
    def run_id(self) -> str:
        return self.state.run_id

    def set_status(self, status: str, stage: str, message: str) -> None:
        self.state.status = status  # type: ignore
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.logger.append(
            run_id=self.run_id,
            level="INFO",
            stage=stage,
            message=message,
            step=self.state.step,
            meta={"status": status},
        )
        self.graph_store.write_run(self.state)

    def pause(self) -> RunState:
        """Pause mid-trial: finishes currently executing trial, then parks in PAUSED."""
        self._pause_requested = True
        self._resume_event.clear()
        if self.state.status not in ("COMPLETED", "FAILED"):
            self.set_status("PAUSED", "execute", "Pause requested: harness parked.")
        return self.state

    def resume(self) -> RunState:
        """Resume paused run: returns back to PROPOSING and unparks."""
        self._pause_requested = False
        self._resume_event.set()
        if self.state.status == "PAUSED":
            self.set_status("PROPOSING", "propose", "Resumed execution from pause.")
        return self.state

    async def decide(
        self,
        step: int,
        action: Literal["approve", "reject"],
        hyperparameters: Optional[dict[str, Any]] = None,
        note: Optional[str] = None,
    ) -> RunState:
        """Human review decision for the proposal at the given step."""
        async with self._step_lock:
            if step in self._decided_steps:
                raise ValueError(f"Step {step} has already been decided.")
            if self.state.step != step or self.pending_proposal is None:
                raise ValueError(f"No pending proposal for step {step} (current step: {self.state.step})")

            self._decided_steps.add(step)
            proposal = self.pending_proposal
            self.pending_proposal = None

            if action == "reject":
                if note:
                    self._rejection_notes.append(note)
                self.logger.append(
                    run_id=self.run_id,
                    level="INFO",
                    stage="decide",
                    message=f"Proposal for step {step} rejected by researcher: {note or 'no note'}",
                    step=step,
                )
                self.state.step += 1
                self.set_status("PROPOSING", "propose", f"Step {step} rejected, advancing to step {self.state.step}")
                return self.state

            # Action == 'approve'
            actor: Literal["laya", "researcher"] = "researcher" if hyperparameters else "laya"
            chosen_params = hyperparameters if hyperparameters is not None else proposal.params

            self.logger.append(
                run_id=self.run_id,
                level="INFO",
                stage="decide",
                message=f"Proposal approved ({actor}): {chosen_params}",
                step=step,
            )

            # Execute the approved trial
            await self._execute_and_mutate(proposal, chosen_params, actor)
            return self.state

    async def start(self) -> None:
        """Kicks off harness execution in background."""
        self.set_status("PROPOSING", "ingest", f"Initialized run with budget={self.state.budget}")
        self._task = asyncio.create_task(self._run_loop())

    async def _run_loop(self) -> None:
        try:
            while self.state.step < self.state.budget:
                await self._resume_event.wait()
                if self._pause_requested:
                    self.set_status("PAUSED", "execute", "Run paused before proposing.")
                    await self._resume_event.wait()

                # Step >= budget check
                if self.state.step >= self.state.budget:
                    break

                self.set_status("PROPOSING", "propose", f"Proposing candidate for step {self.state.step + 1}")
                graph_snapshot = self.graph_store.read_graph(run_id=self.run_id, dataset_id=self.state.dataset_id)
                proposal = self.decision_maker.propose(self.state, graph_snapshot)
                self.pending_proposal = proposal

                if self.state.mode == "human":
                    self.set_status(
                        "AWAITING_REVIEW",
                        "propose",
                        f"Proposal generated for {proposal.model_id} (step {self.state.step}); awaiting human review.",
                    )
                    # Loop stops here for human mode; further progress driven by decide()
                    return

                # Autonomous mode: execute directly
                await self._execute_and_mutate(proposal, proposal.params, "laya")

            self.set_status("COMPLETED", "done", f"Run completed budget of {self.state.budget} trials.")
        except Exception as exc:
            logger.exception("Harness run %s failed with exception", self.run_id)
            self.set_status("FAILED", "done", f"Error during execution: {str(exc)}")

    async def _execute_and_mutate(
        self,
        proposal: Proposal,
        params: dict[str, Any],
        actor: Literal["laya", "researcher"],
    ) -> None:
        step = self.state.step
        self.set_status("EXECUTING", "execute", f"Training {proposal.model_id} with params: {params}")

        # Validate params first
        validation_errors = self.trainer.validate(proposal.model_id, params)
        if validation_errors:
            trial_id = f"trial_{uuid.uuid4().hex[:8]}"
            failure_reason = "; ".join(validation_errors)
            trial = TrialOutcome(
                trial_id=trial_id,
                run_id=self.run_id,
                step=step,
                model_id=proposal.model_id,
                params=params,
                status="FAILED",
                metric_name=self.state.metric_name,
                metric_value=None,
                metrics={},
                failure_reason=failure_reason,
                duration_sec=0.01,
                n_features=10,
                seed=42,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
        else:
            trial = self.trainer.train(
                dataset_id=self.state.dataset_id,
                model_id=proposal.model_id,
                params=params,
                metric=self.state.metric_name,
                seed=42 + step,
            )
            trial.run_id = self.run_id
            trial.step = step

        # Update best metric
        if trial.status == "SUCCESS" and trial.metric_value is not None:
            if self.state.best_metric is None or trial.metric_value > self.state.best_metric:
                self.state.best_metric = trial.metric_value
                self.state.best_trial_id = trial.trial_id
                self.state.best_trial = trial

        # Mutate Graph
        self.set_status("MUTATING", "mutate", f"Recording trial {trial.trial_id} and decision into graph")
        decision = Decision(
            decision_id=f"dec_{uuid.uuid4().hex[:8]}",
            run_id=self.run_id,
            step=step,
            trial_id=trial.trial_id,
            actor=actor,
            strategy=proposal.strategy,
            why=proposal.why,
            hypothesis=proposal.hypothesis,
            confidence=proposal.confidence,
            citations=proposal.citations,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        self.graph_store.write_trial(trial, decision)
        self.state.step += 1

        # Check if pause requested right after completing trial
        if self._pause_requested:
            self.set_status("PAUSED", "execute", "Paused after completing trial.")
        elif self.state.step >= self.state.budget:
            self.set_status("COMPLETED", "done", f"Run completed all {self.state.budget} trials.")
        elif self.state.mode == "autonomous":
            # Continue next step
            asyncio.create_task(self._run_loop())


class HarnessService:
    """Registry and orchestrator for active harness runs."""

    def __init__(
        self,
        trainer: Trainer,
        decision_maker: DecisionMaker,
        graph_store: GraphStore,
        run_logger: RunLogger,
    ) -> None:
        self.trainer = trainer
        self.decision_maker = decision_maker
        self.graph_store = graph_store
        self.logger = run_logger
        self.runs: dict[str, HarnessRun] = {}

    def create_run(
        self,
        dataset_id: str,
        model_ids: list[str],
        mode: Literal["autonomous", "human"],
        metric: str,
        budget: int,
    ) -> HarnessRun:
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        state = RunState(
            run_id=run_id,
            dataset_id=dataset_id,
            mode=mode,
            status="IDLE",
            step=0,
            budget=budget,
            metric_name=metric,
            model_ids=model_ids,
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        harness_run = HarnessRun(
            run_state=state,
            trainer=self.trainer,
            decision_maker=self.decision_maker,
            graph_store=self.graph_store,
            run_logger=self.logger,
        )
        self.runs[run_id] = harness_run
        self.graph_store.write_run(state)
        return harness_run

    def get_run(self, run_id: str) -> Optional[HarnessRun]:
        return self.runs.get(run_id)
