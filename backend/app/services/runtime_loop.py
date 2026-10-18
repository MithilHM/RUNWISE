"""Autonomous Closed-Loop Runtime State Machine for RUNWISE.

Manages the autonomous optimization cycle:
1. QUERYING_GRAPH: FalkorDB multi-hop regime queries
2. GENERATING_CANDIDATES: 3-candidate pool (exploit, explore, regime-boundary)
3. LAYA_EVALUATING: Laya LLM / deterministic evaluator selecting candidate with hypothesis
4. TRAINING_WORKER: Sandboxed tabular model execution with early stopping
5. MUTATING_GRAPH: Atomic FalkorDB mutation recording trial, hypothesis, and failure edges
6. COMPLETED: Emitting real-time SSE stream events and ablation benchmarks
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.graph.runtime_queries import RuntimeGraphService, runtime_graph_service
from app.models.runtime_contracts import (
    AblationResponse,
    CandidateConfig,
    LayaDecision,
    RuntimeObjective,
    RuntimeStepEvent,
    TrialResult,
    is_better,
    metric_is_minimized,
)
from app.services.executor import SandboxedMLWorker, sandboxed_worker
from app.services.laya_agent import LayaAgent

logger = logging.getLogger(__name__)


class RuntimeLoop:
    """State machine driving closed-loop autonomous ML optimization."""

    def __init__(
        self,
        objective: RuntimeObjective,
        run_id: Optional[str] = None,
        graph_service: Optional[RuntimeGraphService] = None,
        worker: Optional[SandboxedMLWorker] = None,
        laya_agent: Optional[LayaAgent] = None,
    ):
        self.run_id = run_id or f"run_{uuid.uuid4().hex[:8]}"
        self.objective = objective
        self.graph_service = graph_service or runtime_graph_service
        self.worker = worker or sandboxed_worker
        self.laya = laya_agent or LayaAgent()

        self.status: str = "INITIALIZING"
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None

        self.current_step: int = 0
        self.best_score: Optional[float] = None
        self.best_hyperparameters: Optional[Dict[str, Any]] = None
        self.best_trial_id: Optional[str] = None

        self.completed_trials: List[TrialResult] = []
        self.execution_tree: List[Dict[str, Any]] = []
        self.graph_mutation_stats: Dict[str, int] = {"nodes_added": 0, "edges_added": 0}

        self.events: List[RuntimeStepEvent] = []
        self._subscribers: List[asyncio.Queue] = []
        self._background_task: Optional[asyncio.Task] = None

    def subscribe(self) -> asyncio.Queue:
        """Subscribes an event queue to receive historical and live RuntimeStepEvents."""
        queue: asyncio.Queue = asyncio.Queue()
        # Backfill existing events
        for event in self.events:
            queue.put_nowait(event)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Removes a subscriber queue."""
        if queue in self._subscribers:
            self._subscribers.remove(queue)

    def _broadcast_event(self, event: RuntimeStepEvent) -> None:
        """Appends event and broadcasts it to all active subscriber queues."""
        self.events.append(event)
        dead_queues = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except Exception:
                dead_queues.append(q)
        for dq in dead_queues:
            self.unsubscribe(dq)

    def start(self) -> str:
        """Kicks off the autonomous loop as an asynchronous background task."""
        self.status = "RUNNING"
        self._background_task = asyncio.create_task(self._run_loop())
        return self.run_id

    async def _run_loop(self) -> None:
        """Executes the closed-loop optimization cycle up to trial_budget or timeout."""
        start_time = time.perf_counter()
        logger.info("Starting autonomous loop %s (budget=%d)", self.run_id, self.objective.trial_budget)

        try:
            for step in range(1, self.objective.trial_budget + 1):
                self.current_step = step

                # Check timeout
                if time.perf_counter() - start_time > self.objective.timeout_seconds:
                    logger.warning("Runtime loop %s reached timeout %.1fs", self.run_id, self.objective.timeout_seconds)
                    break

                # ---------------------------------------------------------
                # 1. QUERYING_GRAPH
                # ---------------------------------------------------------
                similar_regimes = self.graph_service.query_similar_regimes(self.objective.dataset_name)
                failed_subgraphs = self.graph_service.query_failed_subgraphs(self.objective.model_family)

                self._broadcast_event(
                    RuntimeStepEvent(
                        step_number=step,
                        stage="QUERYING_GRAPH",
                        current_best_metric=self.best_score,
                        message=f"Queried FalkorDB: {len(similar_regimes)} similar regime records and {len(failed_subgraphs)} failed subgraphs",
                    )
                )

                # ---------------------------------------------------------
                # 2. GENERATING_CANDIDATES
                # ---------------------------------------------------------
                candidates = self.laya.generator.generate_candidate_pool(
                    objective=self.objective,
                    similar_regimes=similar_regimes,
                    failed_subgraphs=failed_subgraphs,
                    completed_trials=self.completed_trials,
                    step_number=step,
                )

                self._broadcast_event(
                    RuntimeStepEvent(
                        step_number=step,
                        stage="GENERATING_CANDIDATES",
                        current_best_metric=self.best_score,
                        message=f"Generated pool of {len(candidates)} typed candidates (exploit, explore, boundary)",
                    )
                )

                # ---------------------------------------------------------
                # 3. LAYA_EVALUATING (Handoff: CandidatePlanner -> LayaEvaluator)
                # ---------------------------------------------------------
                try:
                    self.graph_service.record_agent_handoff(
                        from_agent="CandidatePlanner",
                        to_agent="LayaEvaluator",
                        run_id=self.run_id,
                        step=step,
                        state="CANDIDATE_EVALUATION",
                        metadata={"candidate_count": len(candidates)},
                    )
                except Exception:
                    pass

                decision = self.laya.evaluate_candidates(
                    objective=self.objective,
                    candidates=candidates,
                    similar_regimes=similar_regimes,
                    failed_subgraphs=failed_subgraphs,
                    current_best_score=self.best_score,
                    step_number=step,
                )

                chosen_cand = next(
                    (c for c in candidates if c.candidate_id == decision.selected_candidate_id),
                    candidates[0],
                )

                self._broadcast_event(
                    RuntimeStepEvent(
                        step_number=step,
                        stage="LAYA_EVALUATING",
                        current_best_metric=self.best_score,
                        laya_decision=decision,
                        message=f"Laya evaluated candidates and selected '{chosen_cand.candidate_id}' ({decision.decision_type})",
                    )
                )

                # ---------------------------------------------------------
                # 4. TRAINING_WORKER (Handoff: LayaEvaluator -> SandboxedWorker)
                # ---------------------------------------------------------
                try:
                    self.graph_service.record_agent_handoff(
                        from_agent="LayaEvaluator",
                        to_agent="SandboxedWorker",
                        run_id=self.run_id,
                        step=step,
                        state="EXECUTING_TRIAL",
                        metadata={"candidate_id": chosen_cand.candidate_id, "strategy": chosen_cand.strategy},
                    )
                except Exception:
                    pass

                trial_res = await self.worker.execute_trial(
                    candidate=chosen_cand,
                    target_metric=self.objective.target_metric,
                    model_family=self.objective.model_family,
                    timeout_seconds=30.0,
                )
                self.completed_trials.append(trial_res)

                # Update best score. Compared in the metric's own direction, so a
                # loss target keeps the lowest loss instead of the highest.
                score = trial_res.metrics.get(self.objective.target_metric)
                minimize = metric_is_minimized(self.objective.target_metric)
                if score is not None and trial_res.status == "SUCCESS":
                    if is_better(score, self.best_score, minimize):
                        self.best_score = score
                        self.best_hyperparameters = trial_res.hyperparameters
                        self.best_trial_id = trial_res.trial_id

                self._broadcast_event(
                    RuntimeStepEvent(
                        step_number=step,
                        stage="TRAINING_WORKER",
                        current_best_metric=self.best_score,
                        trial_result=trial_res,
                        message=f"Worker executed trial '{trial_res.trial_id}': status={trial_res.status}, score={score}",
                    )
                )

                # ---------------------------------------------------------
                # 5. MUTATING_GRAPH (Handoff: SandboxedWorker -> GraphAuditor)
                # ---------------------------------------------------------
                try:
                    self.graph_service.record_agent_handoff(
                        from_agent="SandboxedWorker",
                        to_agent="GraphAuditor",
                        run_id=self.run_id,
                        step=step,
                        state="GRAPH_MUTATION",
                        metadata={"trial_id": trial_res.trial_id, "status": trial_res.status},
                    )
                except Exception:
                    pass
                delta = self.graph_service.mutate_runtime_trial(
                    trial_result=trial_res,
                    dataset_name=self.objective.dataset_name,
                    parent_hypothesis_id=f"hyp_{chosen_cand.candidate_id}",
                    hypothesis_text=decision.hypothesis_text,
                    model_family=self.objective.model_family,
                )
                self.graph_mutation_stats["nodes_added"] += delta.get("nodes_added", 0)
                self.graph_mutation_stats["edges_added"] += delta.get("edges_added", 0)

                # Append execution tree node
                self.execution_tree.append({
                    "step": step,
                    "trial_id": trial_res.trial_id,
                    "candidate_id": chosen_cand.candidate_id,
                    "decision_type": decision.decision_type,
                    "hypothesis": decision.hypothesis_text,
                    "score": score,
                    "status": trial_res.status,
                    "hyperparameters": trial_res.hyperparameters,
                    "graph_delta": delta,
                })

                self._broadcast_event(
                    RuntimeStepEvent(
                        step_number=step,
                        stage="MUTATING_GRAPH",
                        current_best_metric=self.best_score,
                        trial_result=trial_res,
                        laya_decision=decision,
                        graph_delta=delta,
                        message=f"Mutated FalkorDB: added {delta.get('nodes_added')} nodes and {delta.get('edges_added')} edges",
                    )
                )

                # Yield control briefly to ensure non-blocking IO and clean SSE flushing
                await asyncio.sleep(0.01)

            # Completion
            self.status = "COMPLETED"
            self.finished_at = datetime.now(timezone.utc).isoformat()
            self._broadcast_event(
                RuntimeStepEvent(
                    step_number=self.current_step,
                    stage="COMPLETED",
                    current_best_metric=self.best_score,
                    message=f"Autonomous optimization loop {self.run_id} completed successfully with best {self.objective.target_metric}={self.best_score}",
                )
            )

        except Exception as exc:
            self.status = "FAILED"
            self.finished_at = datetime.now(timezone.utc).isoformat()
            logger.error("Error in autonomous loop %s: %s", self.run_id, exc, exc_info=True)
            self._broadcast_event(
                RuntimeStepEvent(
                    step_number=self.current_step,
                    stage="COMPLETED",
                    current_best_metric=self.best_score,
                    message=f"Autonomous optimization failed: {str(exc)}",
                )
            )

    async def run_ablation(self, budget: int = 10) -> AblationResponse:
        """Executes an automated comparison between Graph+Laya vs Random Search with the same budget."""
        rng = random.Random(42)
        space = self.objective.search_space
        minimize = metric_is_minimized(self.objective.target_metric)

        def scored(trial: TrialResult) -> Optional[float]:
            """The target metric of a trial, or None when the trial never produced one.

            A failed or pruned trial has no score. Coercing that to 0.0 would plot a
            failure as a real measurement and, on a maximised metric, would register
            it as the worst result so far rather than as no result at all.
            """
            value = trial.metrics.get(self.objective.target_metric)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        # 1. Curve for Graph + Laya. Only trials that actually scored are plotted.
        laya_curve: List[Dict[str, Any]] = []
        laya_best: Optional[float] = None
        unscored = 0
        for t in self.completed_trials[:budget]:
            s = scored(t)
            if s is None:
                unscored += 1
                continue
            if is_better(s, laya_best, minimize):
                laya_best = s
            laya_curve.append({
                "trial": len(laya_curve) + 1,
                "score": round(s, 4),
                "best_score": round(laya_best, 4),
            })

        # If loop has fewer than budget trials, simulate remaining with Laya generator
        while len(laya_curve) < budget:
            i = len(laya_curve)
            cands = self.laya.generator.generate_candidate_pool(
                self.objective, [], [], self.completed_trials, step_number=i + 1
            )
            dec = self.laya.evaluate_candidates(self.objective, cands, [], [], laya_best, step_number=i + 1)
            chosen = next((c for c in cands if c.candidate_id == dec.selected_candidate_id), cands[0])
            t_res = await self.worker.execute_trial(chosen, self.objective.target_metric)
            s = scored(t_res)
            if s is None:
                unscored += 1
                continue
            if is_better(s, laya_best, minimize):
                laya_best = s
            laya_curve.append({
                "trial": len(laya_curve) + 1,
                "score": round(s, 4),
                "best_score": round(laya_best, 4),
            })

        # 2. Curve for Random Search Baseline
        random_curve: List[Dict[str, Any]] = []
        random_best: Optional[float] = None
        for i in range(budget):
            rand_hparams = {
                "learning_rate": float(rng.choice(space.get("learning_rate", [0.01, 0.05, 0.1, 0.2]))),
                "max_depth": int(rng.choice(space.get("max_depth", [3, 4, 6, 8]))),
                "n_estimators": int(rng.choice(space.get("n_estimators", [50, 100, 150]))),
                "colsample_bytree": float(rng.choice(space.get("colsample_bytree", [0.6, 0.8, 1.0]))),
                "scale_pos_weight": float(rng.choice(space.get("scale_pos_weight", [1.0, 5.0, 10.0, 20.0]))),
            }
            cand = CandidateConfig(
                candidate_id=f"rand_cand_{i+1}",
                hyperparameters=rand_hparams,
                rationale="Uninformed uniform random sampling",
            )
            t_res = await self.worker.execute_trial(cand, self.objective.target_metric)
            s = scored(t_res)
            if s is None:
                unscored += 1
                continue
            if is_better(s, random_best, minimize):
                random_best = s
            random_curve.append({
                "trial": len(random_curve) + 1,
                "score": round(s, 4),
                "best_score": round(random_best, 4),
            })

        # An arm that produced no scored trial has no best. The response contract
        # requires a float, so report 0.0 and say so plainly in the summary rather
        # than letting the sentinel read as a measured result. The empty curve is
        # the real signal, and it is what the UI keys off.
        laya_val = laya_best if laya_best is not None else 0.0
        random_val = random_best if random_best is not None else 0.0

        if laya_best is None or random_best is None:
            imp = 0.0
            missing = [name for name, val in (("Graph+Laya", laya_best), ("Random Search", random_best)) if val is None]
            summary = (
                f"No relative improvement is reported: {', '.join(missing)} produced no scored trial "
                f"within {budget} trials on imbalanced regime '{self.objective.dataset_name}'. "
                f"{unscored} trial(s) across both arms failed or were pruned and are excluded from the curves."
            )
        else:
            # Signed so that positive always means the leading arm did better, which
            # for a minimised metric means a smaller number.
            delta = (random_best - laya_best) if minimize else (laya_best - random_best)
            imp = round((delta / max(abs(random_best), 1e-4)) * 100.0, 2)
            comparison = "lower is better" if minimize else "higher is better"
            shortfall = (
                f" {unscored} trial(s) produced no score (failed or pruned) and are excluded from the curves."
                if unscored
                else ""
            )
            summary = (
                f"FalkorDB Multi-hop Graph + Laya achieved {self.objective.target_metric}={round(laya_best, 4)} "
                f"vs Random Search {round(random_best, 4)} ({'+' if imp >= 0 else ''}{imp}% relative improvement, "
                f"{comparison}) within {budget} trials on imbalanced regime '{self.objective.dataset_name}'."
                f"{shortfall}"
            )

        return AblationResponse(
            run_id=self.run_id,
            budget=budget,
            laya_curve=laya_curve,
            random_curve=random_curve,
            laya_best=round(laya_val, 4),
            random_best=round(random_val, 4),
            improvement_percentage=imp,
            summary=summary,
        )


# Global registry of active and completed loops
runtime_registry: Dict[str, RuntimeLoop] = {}


def create_runtime_loop(objective: RuntimeObjective) -> RuntimeLoop:
    """Creates, registers, and returns a new RuntimeLoop."""
    loop = RuntimeLoop(objective=objective)
    runtime_registry[loop.run_id] = loop
    return loop


def get_runtime_loop(run_id: str) -> Optional[RuntimeLoop]:
    """Retrieves an active or completed RuntimeLoop by run_id."""
    return runtime_registry.get(run_id)
