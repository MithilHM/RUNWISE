"""Integration tests for Member B HTTP APIs.

Defined in MEMBER_B.md Section 4 (B7 — Tests you write).
Tests:
- Every failure matches {"detail": [{"code", "message", "field"}]}
- GET /health returns {status, app, database}
- GET /api/v1/catalog/models returns 3 models and parameter spaces
- POST /api/v1/datasets uploads and profiles CSV
- POST /api/v1/runs with pr_auc on multiclass returns 422
- Unknown run id returns 404 with code: "run_not_found"
- GET /api/v1/runs/{id}/why returns WhyRecord where highlight matches citations
- GET /api/v1/graph returns GraphSnapshot
"""

import io
import time

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "app" in data
    assert "database" in data


def test_catalog_models_endpoint():
    response = client.get("/api/v1/catalog/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "metrics" in data
    assert len(data["models"]) == 3
    model_ids = [m["id"] for m in data["models"]]
    assert "xgboost" in model_ids
    assert "lightgbm" in model_ids
    assert "random_forest" in model_ids

    # Verify ParamSpec structure
    for model in data["models"]:
        assert len(model["params"]) > 0
        for p in model["params"]:
            assert "name" in p
            assert "kind" in p


def test_dataset_upload_and_confirm_target():
    csv_content = (
        "feature_1,feature_2,target\n"
        "1.0,0.5,0\n"
        "2.0,0.4,1\n"
        "1.5,0.6,0\n"
        "3.0,0.2,1\n"
        "2.5,0.7,0\n"
    ).encode("utf-8")

    response = client.post(
        "/api/v1/datasets",
        files={"file": ("test.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert response.status_code == 201
    profile = response.json()
    assert profile["id"].startswith("ds_")
    assert profile["n_rows"] == 5
    assert profile["n_cols"] == 3
    assert profile["task"] == "binary"
    assert profile["target_inferred"] is True

    # Confirm target
    confirm_res = client.post(
        f"/api/v1/datasets/{profile['id']}/confirm-target",
        json={"target": "target"},
    )
    assert confirm_res.status_code == 200
    updated = confirm_res.json()
    assert updated["target_inferred"] is False


def test_unknown_run_returns_404_canonical_envelope():
    response = client.get("/api/v1/runs/run_non_existent")
    assert response.status_code == 404
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)
    assert len(body["detail"]) > 0
    err = body["detail"][0]
    assert "code" in err
    assert "message" in err
    assert err["code"] == "run_not_found"


def test_pr_auc_multiclass_returns_422():
    # A real multiclass dataset: every class needs enough members for a split,
    # so this is 60 rows rather than the four the original fixture used.
    csv_content = (
        "f1,f2,class_target\n"
        + "".join(
            f"{i * 0.37 % 3:.2f},{(i * 0.11) % 2:.2f},{'ABC'[i % 3]}\n"
            for i in range(1, 61)
        )
    ).encode("utf-8")

    upload_res = client.post(
        "/api/v1/datasets",
        files={"file": ("multiclass.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert upload_res.status_code == 201, upload_res.text
    assert upload_res.json()["task"] == "multiclass", upload_res.json()
    ds_id = upload_res.json()["id"]

    # Try creating run with pr_auc on multiclass
    run_res = client.post(
        "/api/v1/runs",
        json={
            "dataset_id": ds_id,
            "model_ids": ["xgboost"],
            "mode": "autonomous",
            "metric": "pr_auc",
            "budget": 3,
        },
    )
    assert run_res.status_code == 422
    err_body = run_res.json()
    assert "detail" in err_body
    assert err_body["detail"][0]["code"] == "metric_incompatible"
    assert "pr_auc" in err_body["detail"][0]["message"]
    assert "multiclass" in err_body["detail"][0]["message"]


def test_run_lifecycle_and_why_explanation():
    # A real upload, not an injected fixture: the router resolves the dataset
    # through the ingestor, so a made-up id is now (correctly) a 404.
    csv_content = (
        "f1,f2,target\n"
        + "".join(
            f"{i * 0.37 % 3:.2f},{(i * 0.11) % 2:.2f},{int(i % 2)}\n" for i in range(1, 61)
        )
    ).encode("utf-8")
    upload_res = client.post(
        "/api/v1/datasets",
        files={"file": ("lifecycle.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert upload_res.status_code == 201, upload_res.text
    dataset_id = upload_res.json()["id"]

    # 1. Create run
    run_res = client.post(
        "/api/v1/runs",
        json={
            "dataset_id": dataset_id,
            "model_ids": ["xgboost", "lightgbm"],
            "mode": "autonomous",
            "metric": "accuracy",
            "budget": 2,
        },
    )
    assert run_res.status_code == 201, run_res.text
    run_data = run_res.json()
    run_id = run_data["run_id"]

    # 2. Wait for the run to produce at least one trial to explain.
    deadline = time.time() + 60
    while time.time() < deadline:
        state = client.get(f"/api/v1/runs/{run_id}").json()
        if state.get("best_trial_id"):
            break
        if state.get("status") in ("COMPLETED", "FAILED"):
            break
        time.sleep(0.5)
    assert state.get("best_trial_id"), f"run produced nothing to explain: {state}"

    # 3. Query why record
    why_res = client.get(f"/api/v1/runs/{run_id}/why")
    assert why_res.status_code == 200, why_res.text
    why_record = why_res.json()
    assert "decision" in why_record
    assert "evidence" in why_record
    assert "highlight" in why_record
    # highlight is exactly decision.citations
    assert why_record["highlight"] == why_record["decision"]["citations"]

    # 3. Query logs
    logs_res = client.get(f"/api/v1/runs/{run_id}/logs?after=0")
    assert logs_res.status_code == 200
    assert "entries" in logs_res.json()


def test_graph_snapshot_endpoint():
    response = client.get("/api/v1/graph")
    assert response.status_code == 200
    graph = response.json()
    assert "nodes" in graph
    assert "edges" in graph
