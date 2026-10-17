"""
End-to-end API integration tests for Person A endpoints.
Verifies HTTP contract compatibility with PRD Section 9.
"""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_api_health_check(test_client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database"]["status"] == "healthy"


def test_api_project_lifecycle(test_client):
    """Test project creation, listing, and fetching."""
    res = client.post("/api/v1/projects", json={
        "id": "proj_api_test",
        "name": "API Test Project",
        "description": "Integration testing project"
    })
    assert res.status_code == 200
    assert res.json()["id"] == "proj_api_test"

    list_res = client.get("/api/v1/projects")
    assert list_res.status_code == 200
    ids = [p["id"] for p in list_res.json()]
    assert "proj_api_test" in ids

    get_res = client.get("/api/v1/projects/proj_api_test")
    assert get_res.status_code == 200
    assert get_res.json()["name"] == "API Test Project"


def test_api_experiment_import_and_queries(test_client):
    """Test importing experiments via JSON and CSV, querying history and lineage."""
    proj_id = "proj_api_exp"
    client.post("/api/v1/projects", json={"id": proj_id, "name": "Exp Test Project"})

    # Import JSON
    import_payload = [
        {
            "id": "api_exp_01",
            "name": "API Baseline",
            "model_family": "gradient_boosting",
            "dataset_name": "credit",
            "dataset_version": "v1",
            "metrics": [{"name": "accuracy", "value": 0.91}]
        },
        {
            "id": "api_exp_02",
            "name": "API Tuned",
            "model_family": "gradient_boosting",
            "dataset_name": "credit",
            "dataset_version": "v1",
            "parent_experiment_id": "api_exp_01",
            "metrics": [{"name": "accuracy", "value": 0.93}]
        }
    ]
    imp_res = client.post(f"/api/v1/projects/{proj_id}/experiments/import", json=import_payload)
    assert imp_res.status_code == 200
    assert imp_res.json()["imported"] == 2

    # Query experiments
    list_res = client.get(f"/api/v1/projects/{proj_id}/experiments")
    assert list_res.status_code == 200
    assert len(list_res.json()) == 2

    # Query detail
    detail_res = client.get("/api/v1/experiments/api_exp_01")
    assert detail_res.status_code == 200
    assert detail_res.json()["id"] == "api_exp_01"
    assert len(detail_res.json()["metrics"]) == 1

    # Query lineage
    lin_res = client.get("/api/v1/experiments/api_exp_02/lineage")
    assert lin_res.status_code == 200
    node_ids = {n["id"] for n in lin_res.json()["nodes"]}
    assert "api_exp_01" in node_ids
    assert "api_exp_02" in node_ids

    # Query graph summary
    summary_res = client.get(f"/api/v1/projects/{proj_id}/graph/summary")
    assert summary_res.status_code == 200
    assert summary_res.json()["node_counts"]["Experiment"] == 2

    # Query visualization graph
    graph_res = client.get(f"/api/v1/projects/{proj_id}/graph")
    assert graph_res.status_code == 200
    assert len(graph_res.json()["nodes"]) > 0


def test_api_correction_and_review_workflow(test_client):
    """Test recording correction, getting impact, reviewing queue and resolving items."""
    proj_id = "proj_api_corr"
    client.post("/api/v1/projects", json={"id": proj_id, "name": "Corr Test Project"})

    # Import an experiment
    client.post(f"/api/v1/projects/{proj_id}/experiments/import", json=[{
        "id": "corr_exp_01",
        "name": "Corr Experiment",
        "model_family": "cnn",
        "dataset_name": "cifar",
        "dataset_version": "v1"
    }])

    # Post correction
    corr_payload = {
        "source_id": "cifar_v1",
        "source_type": "dataset_version",
        "description": "Label corruption discovered in test split",
        "created_by": "data_curator"
    }
    corr_res = client.post(f"/api/v1/projects/{proj_id}/corrections", json=corr_payload)
    assert corr_res.status_code == 200
    impact = corr_res.json()
    assert "corr_exp_01" in impact["directly_affected_experiment_ids"]
    assert len(impact["paths"]) > 0
    corr_id = impact["correction_id"]

    # Get impact by ID
    impact_get = client.get(f"/api/v1/corrections/{corr_id}/impact")
    assert impact_get.status_code == 200

    # Get review queue
    queue_res = client.get(f"/api/v1/projects/{proj_id}/review-queue")
    assert queue_res.status_code == 200
    assert len(queue_res.json()) >= 1
    review_item_id = queue_res.json()[0]["id"]

    # Resolve review item
    resolve_res = client.post(f"/api/v1/review-items/{review_item_id}/resolve", json={
        "status": "RESOLVED",
        "notes": "Verified robust under clean test set."
    })
    assert resolve_res.status_code == 200
    assert resolve_res.json()["status"] == "RESOLVED"
