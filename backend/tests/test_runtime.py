"""Comprehensive unit and integration test suite for RUNWISE Autonomous ML Experiment Runtime.

Tests:
1. Data Contracts validation and serialization
2. Sandboxed ML Worker execution, metrics calculation, and failure handling
3. CandidatePoolGenerator triad synthesis (exploit, explore, regime-boundary)
4. Laya Decision Evaluator (deterministic graph policy & LLM structured paths)
5. FalkorDB multi-hop queries and atomic mutations (EVALUATED_ON, FAILED_UNDER, DERIVED_FROM)
6. RuntimeLoop closed-loop state machine execution and event streaming
7. Ablation comparison benchmark (Laya vs Random Search)
8. FastAPI runtime endpoints (/api/runtime/start, /stream, /status, /ablation)
"""

from __future__ import annotations

import asyncio
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.runtime_contracts import (
    AblationResponse,
    CandidateConfig,
    LayaDecision,
    RuntimeObjective,
    RuntimeStatusResponse,
    RuntimeStepEvent,
    TrialResult,
)
from app.services.executor import SandboxedMLWorker
from app.services.laya_agent import CandidatePoolGenerator, LayaAgent
from app.services.llm_adapter import LLMResult
from app.graph.runtime_queries import RuntimeGraphService
from app.services.runtime_loop import RuntimeLoop, create_runtime_loop, get_runtime_loop


# ---------------------------------------------------------------------------
# 1. Data Contracts
# ---------------------------------------------------------------------------


def test_runtime_objective_defaults():
    obj = RuntimeObjective()
    assert obj.dataset_name == "credit_card_fraud"
    assert obj.target_metric == "val_pr_auc"
    assert obj.model_family == "lightgbm"
    assert obj.trial_budget == 10
    assert "learning_rate" in obj.search_space
    assert "scale_pos_weight" in obj.search_space


def test_runtime_contracts_serialization():
    cand = CandidateConfig(
        candidate_id="cand_test_01",
        hyperparameters={"learning_rate": 0.05, "max_depth": 4},
        rationale="Exploiting top performer",
        source_regime_edges=["exp_hist_01"],
    )
    json_data = cand.model_dump_json()
    assert "cand_test_01" in json_data

    dec = LayaDecision(
        selected_candidate_id="cand_test_01",
        decision_type="exploit_promising_regime",
        hypothesis_text="Hypothesis grounded in graph",
        cited_graph_paths=["exp_hist_01"],
        confidence=0.92,
    )
    assert dec.confidence == 0.92

    trial = TrialResult(
        trial_id="trial_01",
        candidate_id="cand_test_01",
        hyperparameters={"learning_rate": 0.05},
        metrics={"val_pr_auc": 0.88, "train_time_sec": 0.5},
        status="SUCCESS",
        memory_mb=12.5,
    )
    assert trial.status == "SUCCESS"
    assert trial.metrics["val_pr_auc"] == 0.88

    event = RuntimeStepEvent(
        step_number=1,
        stage="TRAINING_WORKER",
        current_best_metric=0.88,
        trial_result=trial,
        laya_decision=dec,
        graph_delta={"nodes_added": 2, "edges_added": 2},
    )
    assert event.stage == "TRAINING_WORKER"
    assert event.graph_delta["nodes_added"] == 2


# ---------------------------------------------------------------------------
# 2. Sandboxed ML Worker
# ---------------------------------------------------------------------------


def test_sandboxed_worker_successful_trial():
    worker = SandboxedMLWorker()
    cand = CandidateConfig(
        candidate_id="cand_valid_01",
        hyperparameters={
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 80,
            "colsample_bytree": 0.8,
            "scale_pos_weight": 10.0,
        },
        rationale="Valid test candidate",
    )
    result = worker.run_trial(cand, target_metric="val_pr_auc", model_family="lightgbm")
    assert result.status == "SUCCESS"
    assert result.failure_reason is None
    assert result.metrics["val_pr_auc"] is not None
    assert 0.0 <= result.metrics["val_pr_auc"] <= 1.0
    assert result.metrics["val_f1"] is not None
    assert result.metrics["val_loss"] is not None
    assert result.metrics["train_time_sec"] > 0.0
    assert result.memory_mb >= 0.0


def test_sandboxed_worker_catches_invalid_hyperparameters():
    worker = SandboxedMLWorker()

    # Invalid negative learning rate
    cand_bad_lr = CandidateConfig(
        candidate_id="cand_bad_lr",
        hyperparameters={"learning_rate": -0.05, "max_depth": 4},
        rationale="Negative learning rate",
    )
    res_bad_lr = worker.run_trial(cand_bad_lr)
    assert res_bad_lr.status == "FAILED"
    assert "Invalid learning rate" in (res_bad_lr.failure_reason or "")

    # Invalid zero max_depth
    cand_bad_depth = CandidateConfig(
        candidate_id="cand_bad_depth",
        hyperparameters={"learning_rate": 0.05, "max_depth": 0},
        rationale="Zero depth",
    )
    res_bad_depth = worker.run_trial(cand_bad_depth)
    assert res_bad_depth.status == "FAILED"
    assert "Invalid max_depth" in (res_bad_depth.failure_reason or "")


@pytest.mark.asyncio
async def test_sandboxed_worker_async_execution():
    worker = SandboxedMLWorker()
    cand = CandidateConfig(
        candidate_id="cand_async",
        hyperparameters={"learning_rate": 0.1, "max_depth": 3, "n_estimators": 50},
        rationale="Async test",
    )
    res = await worker.execute_trial(cand)
    assert res.status == "SUCCESS"
    assert res.candidate_id == "cand_async"


# ---------------------------------------------------------------------------
# 3. CandidatePoolGenerator & Laya Decision Evaluator
# ---------------------------------------------------------------------------


def test_candidate_pool_generation():
    generator = CandidatePoolGenerator(random_seed=42)
    obj = RuntimeObjective()
    sim_regimes = [
        {
            "exp_id": "exp_hist_01",
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 100,
            "scale_pos_weight": 10.0,
            "val_pr_auc": 0.85,
        }
    ]
    failed_subgraphs = [{"exp_id": "exp_fail_01", "failure_reason": "Invalid depth"}]

    cands = generator.generate_candidate_pool(
        objective=obj,
        similar_regimes=sim_regimes,
        failed_subgraphs=failed_subgraphs,
        completed_trials=[],
        step_number=1,
    )
    assert len(cands) == 3
    cand_ids = [c.candidate_id for c in cands]
    assert any("exploit" in cid for cid in cand_ids)
    assert any("explore" in cid for cid in cand_ids)
    assert any("boundary" in cid for cid in cand_ids)


def test_laya_agent_deterministic_mode():
    agent = LayaAgent(random_seed=42)
    obj = RuntimeObjective()
    cands = agent.generator.generate_candidate_pool(obj, [], [], [], step_number=1)

    decision = agent.evaluate_candidates(
        objective=obj,
        candidates=cands,
        similar_regimes=[],
        failed_subgraphs=[],
        current_best_score=None,
        step_number=1,
    )
    assert isinstance(decision, LayaDecision)
    assert decision.selected_candidate_id in [c.candidate_id for c in cands]
    assert decision.confidence > 0.0
    assert len(decision.hypothesis_text) > 10


def test_laya_agent_llm_mode_with_mock():
    class MockLLMClient:
        @property
        def is_available(self):
            return True

        def complete(self, prompt, system=None, max_tokens=400):
            return LLMResult(
                text=json.dumps({
                    "selected_candidate_id": "cand_exploit_s1",
                    "decision_type": "exploit_promising_regime",
                    "hypothesis_text": "Graph evidence indicates depth 4 is optimal on imbalanced fraud.",
                    "cited_graph_paths": ["exp_hist_01"],
                    "confidence": 0.95,
                }),
                model="test-gpt",
                used=True,
            )

    agent = LayaAgent(llm_client=MockLLMClient())
    obj = RuntimeObjective()
    cand = CandidateConfig(
        candidate_id="cand_exploit_s1",
        hyperparameters={"learning_rate": 0.05, "max_depth": 4},
        rationale="Exploit candidate",
    )
    decision = agent.evaluate_candidates(obj, [cand], [], [], current_best_score=0.82, step_number=1)
    assert decision.selected_candidate_id == "cand_exploit_s1"
    assert decision.decision_type == "exploit_promising_regime"
    assert decision.confidence == 0.95
    assert "exp_hist_01" in decision.cited_graph_paths


# ---------------------------------------------------------------------------
# 4. FalkorDB Multi-Hop Queries and Mutations
# ---------------------------------------------------------------------------


def test_runtime_graph_service_queries_and_mutations():
    service = RuntimeGraphService()
    regimes = service.query_similar_regimes("credit_card_fraud")
    assert len(regimes) > 0
    assert "val_pr_auc" in regimes[0] or "learning_rate" in regimes[0]

    failed = service.query_failed_subgraphs("lightgbm")
    assert len(failed) > 0
    assert "failure_reason" in failed[0]

    # Test mutation of successful trial
    trial_success = TrialResult(
        trial_id="trial_mut_success_01",
        candidate_id="cand_s1",
        hyperparameters={"learning_rate": 0.05, "max_depth": 4, "scale_pos_weight": 10.0},
        metrics={"val_pr_auc": 0.885, "val_loss": 0.04},
        status="SUCCESS",
        memory_mb=14.2,
    )
    delta_s = service.mutate_runtime_trial(
        trial_result=trial_success,
        dataset_name="credit_card_fraud",
        parent_hypothesis_id="hyp_cand_s1",
        hypothesis_text="Testing depth 4",
    )
    assert delta_s["nodes_added"] >= 1
    assert delta_s["edges_added"] >= 1

    # Test mutation of failed trial (creates FAILED_UNDER)
    trial_fail = TrialResult(
        trial_id="trial_mut_fail_01",
        candidate_id="cand_bad",
        hyperparameters={"learning_rate": -0.01},
        metrics={},
        status="FAILED",
        failure_reason="Invalid parameter bounds",
        memory_mb=0.1,
    )
    delta_f = service.mutate_runtime_trial(
        trial_result=trial_fail,
        dataset_name="credit_card_fraud",
    )
    assert delta_f["nodes_added"] >= 1
    assert delta_f["edges_added"] >= 1


# ---------------------------------------------------------------------------
# 5. Closed-Loop RuntimeLoop Execution and Events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runtime_loop_full_lifecycle():
    obj = RuntimeObjective(
        dataset_name="credit_card_fraud",
        target_metric="val_pr_auc",
        trial_budget=3,
        timeout_seconds=60.0,
    )
    loop = RuntimeLoop(objective=obj)
    queue = loop.subscribe()

    loop.start()
    assert loop.status == "RUNNING"

    # Wait for completion task
    await loop._background_task

    assert loop.status == "COMPLETED"
    assert len(loop.completed_trials) == 3
    assert loop.best_score is not None
    assert loop.best_score > 0.0
    assert len(loop.execution_tree) == 3
    assert loop.graph_mutation_stats["nodes_added"] >= 3

    # Check received events from subscriber queue
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())

    assert len(events) >= 3
    stages = [e.stage for e in events]
    assert "QUERYING_GRAPH" in stages
    assert "GENERATING_CANDIDATES" in stages
    assert "LAYA_EVALUATING" in stages
    assert "TRAINING_WORKER" in stages
    assert "MUTATING_GRAPH" in stages
    assert "COMPLETED" in stages


@pytest.mark.asyncio
async def test_runtime_ablation_benchmark():
    obj = RuntimeObjective(trial_budget=2)
    loop = RuntimeLoop(objective=obj)
    await loop._run_loop()

    ablation = await loop.run_ablation(budget=3)
    assert isinstance(ablation, AblationResponse)
    assert len(ablation.laya_curve) == 3
    assert len(ablation.random_curve) == 3
    assert ablation.laya_best > 0.0
    assert ablation.random_best > 0.0
    assert "improvement_percentage" in ablation.model_dump()
    assert len(ablation.summary) > 20


# ---------------------------------------------------------------------------
# 6. FastAPI Runtime Endpoints
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_runtime_endpoints():
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. POST /api/runtime/start
        start_payload = {
            "dataset_name": "credit_card_fraud",
            "target_metric": "val_pr_auc",
            "trial_budget": 2,
            "timeout_seconds": 30.0,
        }
        res = await client.post("/api/runtime/start", json=start_payload)
        assert res.status_code == 201
        data = res.json()
        assert "run_id" in data
        assert data["status"] == "STARTED"
        run_id = data["run_id"]

        # Let the loop execute trials
        loop = get_runtime_loop(run_id)
        assert loop is not None
        if loop._background_task:
            await loop._background_task

        # 2. GET /api/runtime/{run_id}/status
        res_status = await client.get(f"/api/runtime/{run_id}/status")
        assert res_status.status_code == 200
        status_data = res_status.json()
        assert status_data["run_id"] == run_id
        assert status_data["status"] == "COMPLETED"
        assert len(status_data["completed_trials"]) == 2
        assert status_data["best_score"] is not None

        # 3. GET /api/runtime/{run_id}/stream with poll=true
        res_poll = await client.get(f"/api/runtime/{run_id}/stream?poll=true")
        assert res_poll.status_code == 200
        poll_events = res_poll.json()
        assert len(poll_events) > 0
        assert any(e["stage"] == "COMPLETED" for e in poll_events)

        # 4. POST /api/runtime/{run_id}/ablation
        res_ablation = await client.post(f"/api/runtime/{run_id}/ablation?budget=2")
        assert res_ablation.status_code == 200
        ablation_data = res_ablation.json()
        assert ablation_data["run_id"] == run_id
        assert "laya_curve" in ablation_data
        assert "random_curve" in ablation_data

        # 5. Non-existent run_id returns structured 404
        res_404 = await client.get("/api/runtime/non_existent_run/status")
        assert res_404.status_code == 404
        err = res_404.json()
        assert "detail" in err
        assert err["detail"][0]["code"] == "runtime_run_not_found"
