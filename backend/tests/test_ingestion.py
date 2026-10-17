"""
Tests for experiment ingestion pipeline (CSV, JSON, validation, duplicate handling).
"""

import io
import json
from app.services.ingestion import compute_canonical_config


def test_canonical_config_hashing():
    """Verify compute_canonical_config is deterministic regardless of key insertion order."""
    cfg1 = {"lr": 0.001, "batch_size": 32, "nested": {"b": 2, "a": 1}}
    cfg2 = {"nested": {"a": 1, "b": 2}, "batch_size": 32, "lr": 0.001}

    id1, json1, hash1 = compute_canonical_config(cfg1)
    id2, json2, hash2 = compute_canonical_config(cfg2)

    assert id1 == id2
    assert hash1 == hash2
    assert json1 == json2


def test_ingest_json_valid_batch(test_repo, test_ingestion_service):
    """Test importing a valid batch of experiments from JSON."""
    payload = [
        {
            "id": "json_exp_1",
            "name": "JSON Exp 1",
            "model_family": "gradient_boosting",
            "dataset_name": "credit_data",
            "dataset_version": "v1",
            "metrics": [{"name": "accuracy", "value": 0.92}],
            "configuration": {"max_depth": 5}
        },
        {
            "id": "json_exp_2",
            "name": "JSON Exp 2",
            "model_family": "gradient_boosting",
            "dataset_name": "credit_data",
            "dataset_version": "v1",
            "metrics": [{"name": "accuracy", "value": 0.94}],
            "configuration": {"max_depth": 7}
        }
    ]

    summary = test_ingestion_service.ingest_json(payload, project_id="proj_json")
    assert summary.imported == 2
    assert summary.failed == 0
    assert len(summary.errors) == 0

    exp1 = test_repo.get_experiment("json_exp_1")
    assert exp1 is not None
    assert exp1.name == "JSON Exp 1"
    assert exp1.configuration == {"max_depth": 5}


def test_ingest_csv_content(test_repo, test_ingestion_service):
    """Test importing experiments from CSV content with aliases."""
    csv_text = """run_id,exp_name,architecture,dataset,ds_version,val_accuracy,val_loss,max_depth
csv_01,CSV Model 1,transformer,squad,v2,0.88,0.22,12
csv_02,CSV Model 2,transformer,squad,v2,0.91,0.18,12
"""
    summary = test_ingestion_service.ingest_csv(csv_text, project_id="proj_csv")
    assert summary.imported == 2
    assert summary.failed == 0

    exp = test_repo.get_experiment("csv_01")
    assert exp is not None
    assert exp.model_family == "transformer"
    assert exp.dataset_name == "squad"
    metric_names = [m.name for m in exp.metrics]
    assert "val_accuracy" in metric_names or "validation_accuracy" in metric_names


def test_ingest_partial_failure_and_validation(test_repo, test_ingestion_service):
    """Test batch with an invalid record (missing id) handles partial failure."""
    records = [
        {
            "id": "valid_01",
            "name": "Valid Run",
            "model_family": "cnn",
            "dataset_name": "mnist",
            "metrics": [{"name": "accuracy", "value": 0.99}]
        },
        {
            # Missing 'id'
            "name": "Invalid Run Without ID",
            "model_family": "cnn",
            "dataset_name": "mnist"
        },
        {
            "id": "valid_02",
            "name": "Second Valid Run",
            "model_family": "cnn",
            "dataset_name": "mnist",
            "metrics": [{"name": "accuracy", "value": 0.98}]
        }
    ]

    summary = test_ingestion_service.ingest_records(records, project_id="proj_partial")
    assert summary.imported == 2
    assert summary.failed == 1
    assert len(summary.errors) == 1
    assert summary.errors[0].row == 2
    assert "Missing required experiment identifier" in summary.errors[0].reason

    # Valid records were still persisted
    assert test_repo.get_experiment("valid_01") is not None
    assert test_repo.get_experiment("valid_02") is not None


def test_duplicate_ingestion_idempotency(test_repo, test_ingestion_service):
    """Test re-importing the same experiment updates properties without duplicating nodes."""
    record = [{
        "id": "idempotent_01",
        "name": "Initial Run",
        "model_family": "linear",
        "dataset_name": "boston",
        "metrics": [{"name": "mse", "value": 15.2}]
    }]

    # Ingest 1st time
    s1 = test_ingestion_service.ingest_records(record, project_id="proj_idem")
    assert s1.imported == 1

    # Ingest 2nd time with updated metric
    record[0]["metrics"] = [{"name": "mse", "value": 14.8}]
    s2 = test_ingestion_service.ingest_records(record, project_id="proj_idem")
    assert s2.imported == 1

    exp = test_repo.get_experiment("idempotent_01")
    assert exp is not None
    assert len(exp.metrics) == 1
    assert exp.metrics[0].value == 14.8

    # Ensure only 1 experiment node exists in project
    all_exps = test_repo.list_experiments(project_id="proj_idem")
    assert len(all_exps) == 1
