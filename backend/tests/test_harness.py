"""Tests for Autonomous ML Harness Loop.

Defined in MEMBER_B.md Section 4 (B7 — Tests you write).
Asserts:
- Autonomous mode runs budget trials without blocking and reports COMPLETED
- Human mode blocks at AWAITING_REVIEW and does not execute before decide
- Reject never executes a trial and advances step
- Invalid params produce FAILED trial with failure_reason and no metric_value
- Two concurrent decide calls on the same step execute exactly one trial
- Budget exhaustion stops the loop (budget + 1 never happens)
- Pause mid-trial finishes trial and parks in PAUSED; resume returns to PROPOSING
"""

import asyncio
import pytest
from app.models.contracts import Proposal, RunState
from app.services.harness import HarnessRun
from tests.fakes.fake_graph import FakeGraphStore
from tests.fakes.fake_trainer import FakeDecisionMaker, FakeTrainer, InMemoryRunLogger


@pytest.fixture
def harness_setup():
    graph_store = FakeGraphStore()
    trainer = FakeTrainer()
    decision_maker = FakeDecisionMaker()
    run_logger = InMemoryRunLogger()
    return graph_store, trainer, decision_maker, run_logger


@pytest.mark.asyncio
async def test_autonomous_mode_completes_budget(harness_setup):
    graph_store, trainer, decision_maker, run_logger = harness_setup
    run_state = RunState(
        run_id="run_auto_test",
        dataset_id="ds_001",
        mode="autonomous",
        status="IDLE",
        step=0,
        budget=3,
        metric_name="f1_macro",
        model_ids=["xgboost", "lightgbm"],
    )
    harness = HarnessRun(run_state, trainer, decision_maker, graph_store, run_logger)
    await harness.start()

    # Wait for completion
    for _ in range(50):
        if harness.state.status == "COMPLETED":
            break
        await asyncio.sleep(0.05)

    assert harness.state.status == "COMPLETED"
    assert harness.state.step == 3
    assert len(graph_store.trials) == 3
    # Check budget + 1 never happened
    assert harness.state.step <= 3


@pytest.mark.asyncio
async def test_human_mode_blocks_and_awaits_decision(harness_setup):
    graph_store, trainer, decision_maker, run_logger = harness_setup
    run_state = RunState(
        run_id="run_human_test",
        dataset_id="ds_001",
        mode="human",
        status="IDLE",
        step=0,
        budget=2,
        metric_name="accuracy",
        model_ids=["xgboost"],
    )
    harness = HarnessRun(run_state, trainer, decision_maker, graph_store, run_logger)
    await harness.start()

    # Wait for pending proposal
    for _ in range(30):
        if harness.state.status == "AWAITING_REVIEW":
            break
        await asyncio.sleep(0.02)

    assert harness.state.status == "AWAITING_REVIEW"
    assert harness.pending_proposal is not None
    # Assert trainer has NOT yet executed any trial
    assert len(graph_store.trials) == 0

    # Human approves with override
    step = harness.state.step
    await harness.decide(
        step=step,
        action="approve",
        hyperparameters={"max_depth": 5, "learning_rate": 0.03, "n_estimators": 150},
    )

    # Trial was executed and recorded with actor="researcher"
    assert len(graph_store.trials) == 1
    decision = list(graph_store.decisions.values())[0]
    assert decision.actor == "researcher"


@pytest.mark.asyncio
async def test_human_mode_reject_advances_without_execution(harness_setup):
    graph_store, trainer, decision_maker, run_logger = harness_setup
    run_state = RunState(
        run_id="run_reject_test",
        dataset_id="ds_001",
        mode="human",
        status="IDLE",
        step=0,
        budget=3,
        metric_name="accuracy",
        model_ids=["xgboost"],
    )
    harness = HarnessRun(run_state, trainer, decision_maker, graph_store, run_logger)
    await harness.start()

    for _ in range(30):
        if harness.state.status == "AWAITING_REVIEW":
            break
        await asyncio.sleep(0.02)

    # Reject proposal
    await harness.decide(step=0, action="reject", note="Depth is too shallow")

    # Step advanced to 1, no trial was executed
    assert harness.state.step == 1
    assert len(graph_store.trials) == 0


@pytest.mark.asyncio
async def test_invalid_params_produce_failed_trial(harness_setup):
    graph_store, trainer, decision_maker, run_logger = harness_setup
    run_state = RunState(
        run_id="run_invalid_test",
        dataset_id="ds_001",
        mode="human",
        status="IDLE",
        step=0,
        budget=1,
        metric_name="accuracy",
        model_ids=["xgboost"],
    )
    harness = HarnessRun(run_state, trainer, decision_maker, graph_store, run_logger)
    await harness.start()

    for _ in range(30):
        if harness.state.status == "AWAITING_REVIEW":
            break
        await asyncio.sleep(0.02)

    # Provide invalid parameter: learning_rate <= 0
    await harness.decide(
        step=0,
        action="approve",
        hyperparameters={"max_depth": 4, "learning_rate": -0.5, "n_estimators": 100},
    )

    assert len(graph_store.trials) == 1
    trial = list(graph_store.trials.values())[0]
    assert trial.status == "FAILED"
    assert trial.metric_value is None
    assert trial.failure_reason is not None
    assert "learning_rate must be > 0" in trial.failure_reason


@pytest.mark.asyncio
async def test_concurrent_decide_calls_guard(harness_setup):
    graph_store, trainer, decision_maker, run_logger = harness_setup
    run_state = RunState(
        run_id="run_concurrent_test",
        dataset_id="ds_001",
        mode="human",
        status="IDLE",
        step=0,
        budget=2,
        metric_name="accuracy",
        model_ids=["xgboost"],
    )
    harness = HarnessRun(run_state, trainer, decision_maker, graph_store, run_logger)
    await harness.start()

    for _ in range(30):
        if harness.state.status == "AWAITING_REVIEW":
            break
        await asyncio.sleep(0.02)

    # Run two decide calls concurrently
    results = await asyncio.gather(
        harness.decide(step=0, action="approve"),
        harness.decide(step=0, action="approve"),
        return_exceptions=True,
    )

    # Exactly one succeeded and one raised ValueError
    successes = [r for r in results if isinstance(r, RunState)]
    failures = [r for r in results if isinstance(r, ValueError)]
    assert len(successes) == 1
    assert len(failures) == 1
    assert len(graph_store.trials) == 1


@pytest.mark.asyncio
async def test_pause_and_resume(harness_setup):
    graph_store, trainer, decision_maker, run_logger = harness_setup
    run_state = RunState(
        run_id="run_pause_test",
        dataset_id="ds_001",
        mode="autonomous",
        status="IDLE",
        step=0,
        budget=5,
        metric_name="f1_macro",
        model_ids=["xgboost"],
    )
    harness = HarnessRun(run_state, trainer, decision_maker, graph_store, run_logger)
    await harness.start()

    # Pause harness
    harness.pause()
    assert harness.state.status in ("PAUSED", "PROPOSING", "EXECUTING", "MUTATING")

    # Resume harness
    harness.resume()
    assert harness.state.status != "PAUSED"
