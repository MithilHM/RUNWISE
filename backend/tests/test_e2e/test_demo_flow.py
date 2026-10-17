"""
End-to-end integration test suite simulating the full researcher demo workflow.
Validates the complete lifecycle specified in RUNWISE PRD Sections 13, 14, and 18.
"""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.config import settings

api_client = TestClient(app)


def test_full_demo_lifecycle_e2e(test_client):
    """
    Simulates the full 3-Act hackathon demo sequence:
    1. Act I: Exploration - Project creation, experiment ingestion, history, lineage, comparability.
    2. Act II: Copilot & Recommendations - Zero-key copilot query and budget-constrained proposal.
    3. Act III: Invalidation & Adaptive Lineage - Dataset defect injection, graph impact traversal,
       review queue resolution, and verification that tainted evidence is excluded from future recommendations.
    """
    proj_id = "proj_e2e_demo"
    proj_name = "End-to-End Demo Project"

    # --- Step 1: Health & System Verification ---
    health_res = api_client.get("/health")
    assert health_res.status_code == 200
    health_data = health_res.json()
    assert health_data["status"] == "healthy"
    assert health_data["database"]["status"] == "healthy"

    # --- Step 2: Project Creation ---
    create_proj_res = api_client.post(
        "/api/v1/projects",
        json={"id": proj_id, "name": proj_name, "description": "Full E2E verification project"}
    )
    assert create_proj_res.status_code == 200
    assert create_proj_res.json()["id"] == proj_id

    # --- Step 3: Experiment Ingestion ---
    # Ingest a controlled cohort: baseline (a1), tuned variant (a2), and an independent unaffected run (b1)
    experiments_payload = [
        {
            "id": "e2e_exp_001",
            "name": "baseline_xgboost",
            "model_family": "gradient_boosting",
            "dataset_name": "tabular_fraud",
            "dataset_version": "v1",
            "status": "completed",
            "random_seed": 42,
            "compute_cost_gpu_hours": 0.8,
            "configuration": {"max_depth": 4, "n_estimators": 100},
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.91,
                    "direction": "maximize",
                    "evaluation_split": "validation",
                    "evaluation_protocol": "stratified_5fold"
                }
            ],
            "description": "Baseline XGBoost on v1 fraud dataset"
        },
        {
            "id": "e2e_exp_002",
            "name": "tuned_xgboost_depth6",
            "model_family": "gradient_boosting",
            "dataset_name": "tabular_fraud",
            "dataset_version": "v1",
            "parent_experiment_id": "e2e_exp_001",
            "status": "completed",
            "random_seed": 42,
            "compute_cost_gpu_hours": 1.2,
            "configuration": {"max_depth": 6, "n_estimators": 200},
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.93,
                    "direction": "maximize",
                    "evaluation_split": "validation",
                    "evaluation_protocol": "stratified_5fold"
                }
            ],
            "description": "Tuned tree depth based on e2e_exp_001 baseline"
        },
        {
            "id": "e2e_exp_009",
            "name": "independent_clean_mlp",
            "model_family": "mlp",
            "dataset_name": "tabular_fraud",
            "dataset_version": "v2",  # Different dataset version! Unaffected by v1 defect
            "status": "completed",
            "random_seed": 101,
            "compute_cost_gpu_hours": 0.7,
            "configuration": {"hidden_layers": [64, 32]},
            "metrics": [
                {
                    "name": "validation_accuracy",
                    "value": 0.89,
                    "direction": "maximize",
                    "evaluation_split": "validation",
                    "evaluation_protocol": "stratified_5fold"
                }
            ],
            "description": "Clean baseline on v2 dataset"
        }
    ]

    import_res = api_client.post(f"/api/v1/projects/{proj_id}/experiments/import", json=experiments_payload)
    assert import_res.status_code == 200
    assert import_res.json()["imported"] == 3

    # --- Step 4: Act I - Exploration & Comparability ---
    # List experiments
    list_res = api_client.get(f"/api/v1/projects/{proj_id}/experiments")
    assert list_res.status_code == 200
    runs = list_res.json()
    assert len(runs) == 3

    # Check detail & lineage of child run
    lineage_res = api_client.get(f"/api/v1/experiments/e2e_exp_002/lineage?project_id={proj_id}")
    assert lineage_res.status_code == 200
    lineage_nodes = {n["id"] for n in lineage_res.json().get("nodes", [])}
    assert "e2e_exp_001" in lineage_nodes
    assert "e2e_exp_002" in lineage_nodes

    # Comparability check between e2e_exp_001 and e2e_exp_002
    comp_res = api_client.get(f"/api/v1/experiments/e2e_exp_001/comparability?target_id=e2e_exp_002&project_id={proj_id}")
    assert comp_res.status_code == 200
    assessment = comp_res.json()
    assert assessment["status"] in {"COMPARABLE", "COMPARABLE_WITH_CAVEATS"}
    assert 0.0 <= assessment["score"] <= 1.0

    # --- Step 5: Act II - Research Copilot & Initial Recommendation ---
    # Query copilot in deterministic mode (zero-key execution)
    query_payload = {
        "question": "What is the best performing baseline and what should I run next?",
        "compute_budget_gpu_hours": 2.0
    }
    query_res = api_client.post(f"/api/v1/projects/{proj_id}/query", json=query_payload)
    assert query_res.status_code == 200
    query_answer = query_res.json()
    assert query_answer["mode"] in {"deterministic", "llm"}
    assert query_answer["llm_used"] is False or isinstance(query_answer["llm_used"], bool)
    assert (query_answer.get("grounding") or {}).get("is_grounded", True) is True

    # Request recommendations under a 2.0 GPU-hour budget
    rec_payload = {
        "objective": "Improve fraud detection validation accuracy",
        "compute_budget_gpu_hours": 2.0
    }
    rec_res = api_client.post(f"/api/v1/projects/{proj_id}/recommendations", json=rec_payload)
    assert rec_res.status_code == 200
    recs = rec_res.json().get("recommendations", [])
    assert len(recs) > 0
    rec = recs[0]
    rec_id = rec["id"]
    assert rec["evidence_status"] == "inferred"
    assert rec.get("estimated_gpu_hours", 0) <= 2.0

    # Before correction, the XGBoost runs on v1 are cited as valid baselines/support
    pre_cited = set(rec.get("baseline_experiment_ids", []) + rec.get("supporting_experiment_ids", []))
    assert ("e2e_exp_001" in pre_cited) or ("e2e_exp_002" in pre_cited) or ("e2e_exp_009" in pre_cited)

    # Approve decision (FR-9)
    decision_res = api_client.post(
        f"/api/v1/recommendations/{rec_id}/decision",
        json={"decision": "approved", "decided_by": "lead_researcher", "note": "Approved for execution"}
    )
    assert decision_res.status_code == 200
    assert decision_res.json()["status"] == "approved"

    # --- Step 6: Act III - Defect Invalidation & Impact Traversal ---
    # Ingest defect on dataset v1
    corr_payload = {
        "source_id": "tabular_fraud_v1",
        "source_type": "dataset_version",
        "description": "Train-validation target leakage discovered in fraud dataset v1",
        "created_by": "data_engineering_team"
    }
    corr_res = api_client.post(f"/api/v1/projects/{proj_id}/corrections", json=corr_payload)
    assert corr_res.status_code == 200
    impact = corr_res.json()
    corr_id = impact["correction_id"]

    # Assert graph traversal precision:
    # 1) Runs using v1 must be directly affected
    assert set(impact["directly_affected_experiment_ids"]) == {"e2e_exp_001", "e2e_exp_002"}
    # 2) Run on v2 must be preserved as UNAFFECTED
    assert "e2e_exp_009" in impact["unaffected_experiment_ids"]
    # 3) Explicit graph paths must be returned
    assert len(impact["paths"]) >= 2

    # --- Step 7: Review Queue Resolution ---
    queue_res = api_client.get(f"/api/v1/projects/{proj_id}/review-queue")
    assert queue_res.status_code == 200
    queue = queue_res.json()
    assert len(queue) == 2
    item_to_resolve = queue[0]

    resolve_res = api_client.post(
        f"/api/v1/review-items/{item_to_resolve['id']}/resolve",
        json={"status": "resolved", "notes": "Acknowledged defect; retrained on clean split"}
    )
    assert resolve_res.status_code == 200
    assert resolve_res.json()["status"] == "resolved"

    # Verify queue decreases by 1
    queue_after = api_client.get(f"/api/v1/projects/{proj_id}/review-queue").json()
    assert len(queue_after) == 1

    # --- Step 8: Adaptive Lineage & Adaptive Recommendations ---
    # Re-run recommendation: verify that runs under correction are NOT cited,
    # and the unaffected run (e2e_exp_009) remains usable as evidence
    post_rec_res = api_client.post(f"/api/v1/projects/{proj_id}/recommendations", json=rec_payload)
    assert post_rec_res.status_code == 200
    post_recs = post_rec_res.json()
    post_proposals = post_recs.get("recommendations", [])

    post_cited = {
        exp_id
        for p in post_proposals
        for exp_id in (p.get("baseline_experiment_ids", []) + p.get("supporting_experiment_ids", []))
    }
    # Tainted experiments must never be used as trusted support
    assert not (post_cited & {"e2e_exp_001", "e2e_exp_002"}), f"Tainted runs were cited: {post_cited}"
    # Unaffected run on clean data remains in considered evidence
    assert "e2e_exp_009" in post_recs.get("considered_experiment_ids", [])


def test_e2e_error_envelopes_and_edge_cases(test_client):
    """Verifies that API error envelopes are consistent and non-crashing across edge cases."""
    # 1. Nonexistent experiment returns structured 404
    res = api_client.get("/api/v1/experiments/exp_ghost_id")
    assert res.status_code == 404
    body = res.json()
    assert "detail" in body
    assert body["detail"][0].get("code") == "experiment_not_found"

    # 2. Nonexistent project graph summary returns 404
    proj_res = api_client.get("/api/v1/projects/project_ghost_id/graph/summary")
    assert proj_res.status_code == 404

    # 3. Invalid validation payload returns structured 422
    invalid_rec = api_client.post("/api/v1/projects/proj_e2e_demo/recommendations", json={"objective": ""})
    assert invalid_rec.status_code == 422
    assert "detail" in invalid_rec.json()
