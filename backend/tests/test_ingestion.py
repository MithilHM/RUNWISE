"""
Tests for experiment ingestion pipeline (CSV, JSON, validation, duplicate handling).
"""

import io
import json
from types import SimpleNamespace

from app.services.ingestion import IngestionService, compute_canonical_config


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


# ---------------------------------------------------------------------------
# FR-1 #8 "never silently invent missing values" and FR-1 #4 "report malformed
# records and validation errors".
#
# These read node properties straight off the graph, because the read helpers
# answer with contract models that cannot express an absent value.
# ---------------------------------------------------------------------------


def _nodes(test_repo, project_id, label):
    graph = test_repo.get_project_graph(project_id)
    return {node.id: node.properties for node in graph.nodes if node.label == label}


class _StubRepository:
    """In-memory stand-in for the graph repository.

    It implements only what the importer calls, and records how often the
    Experiment node is written, so the idempotency rules (FR-1 #5) can be
    proved without depending on the graph read path.
    """

    def __init__(self):
        self.projects = {}
        self.datasets = {}
        self.dataset_versions = {}
        self.configurations = {}
        self.experiments = {}
        self.metrics = {}
        self.experiment_writes = 0

    # -- writes -----------------------------------------------------------
    def get_project(self, project_id):
        return self.projects.get(project_id)

    def create_or_update_project(self, project_id, name, description=None):
        self.projects.setdefault(project_id, {"id": project_id, "name": name})
        return self.projects[project_id]

    def create_or_update_dataset(self, dataset_id, name, description=None):
        self.datasets.setdefault(dataset_id, {"id": dataset_id, "name": name})
        return self.datasets[dataset_id]

    def create_or_update_dataset_version(self, version_id, dataset_id, version, quality_status="valid"):
        self.dataset_versions.setdefault(
            version_id,
            {"id": version_id, "dataset_id": dataset_id, "version": version},
        )
        return self.dataset_versions[version_id]

    def create_or_update_configuration(self, config_id, canonical_json, config_hash):
        self.configurations.setdefault(config_id, canonical_json)
        return self.configurations[config_id]

    def create_or_update_experiment(self, exp_data):
        self.experiment_writes += 1
        exp_id = exp_data["id"]
        existing = self.experiments.get(exp_id)
        if existing is None:
            self.experiments[exp_id] = dict(exp_data)
        else:
            existing.update(exp_data)
            if exp_data.get("parent_experiment_id"):
                existing["parent_experiment_id"] = exp_data["parent_experiment_id"]
        return self.experiments[exp_id]

    def create_or_update_metric(
        self, metric_id, experiment_id, name, value, direction="maximize",
        evaluation_split="validation", evaluation_protocol=None,
    ):
        self.metrics[metric_id] = {
            "id": metric_id,
            "experiment_id": experiment_id,
            "name": name,
            "value": value,
            "direction": direction,
            "evaluation_split": evaluation_split,
            "evaluation_protocol": evaluation_protocol,
        }
        return self.metrics[metric_id]

    # -- reads ------------------------------------------------------------
    def get_experiment(self, experiment_id):
        stored = self.experiments.get(experiment_id)
        if stored is None:
            return None
        return SimpleNamespace(
            id=experiment_id,
            project_id=stored.get("project_id"),
            name=stored.get("name"),
            model_family=stored.get("model_family"),
            status=stored.get("status"),
            description=stored.get("description"),
            notes=stored.get("notes"),
            source_reference=stored.get("source_reference"),
            dataset_version_id=stored.get("dataset_version_id"),
            duration_seconds=stored.get("duration_seconds"),
            compute_cost_gpu_hours=stored.get("compute_cost_gpu_hours"),
            random_seed=stored.get("random_seed"),
            created_at=stored.get("created_at"),
            parent_experiment_ids=(
                [stored["parent_experiment_id"]] if stored.get("parent_experiment_id") else []
            ),
            configuration=json.loads(self.configurations[stored["config_id"]])
            if stored.get("config_id") in self.configurations
            else None,
            metrics=[SimpleNamespace(**m) for m in self.metrics.values()
                     if m["experiment_id"] == experiment_id],
        )


_FULL_RECORD = {
    "id": "stub_01",
    "name": "Stub Run",
    "model_family": "linear",
    "dataset_name": "housing",
    "dataset_version": "v3",
    "status": "completed",
    "random_seed": 7,
    "duration_seconds": 12.5,
    "compute_cost_gpu_hours": 0.4,
    "created_at": "2024-05-05T00:00:00Z",
    "source_reference": "runs/stub_01.json",
    "configuration": {"lr": 0.01, "batch": 32},
    "metrics": [
        {
            "name": "validation_accuracy",
            "value": 0.8,
            "direction": "maximize",
            "evaluation_split": "validation",
        }
    ],
}


def test_reimport_of_an_identical_record_touches_nothing():
    """FR-1 #5: an unchanged re-import is skipped and writes nothing."""
    stub = _StubRepository()
    service = IngestionService(repository=stub)

    first = service.ingest_records([dict(_FULL_RECORD)], project_id="stub_proj")
    assert (first.imported, first.skipped, first.failed) == (1, 0, 0)
    assert stub.experiment_writes == 1

    second = service.ingest_records([dict(_FULL_RECORD)], project_id="stub_proj")
    assert (second.imported, second.skipped, second.failed) == (0, 1, 0)
    assert any("identical" in e.reason for e in second.errors)
    assert stub.experiment_writes == 1, "an unchanged re-import wrote to the graph"
    assert len(stub.experiments) == 1
    assert len(stub.metrics) == 1


def test_a_changed_record_is_still_applied():
    """Skipping is only for records that would change nothing."""
    stub = _StubRepository()
    service = IngestionService(repository=stub)

    service.ingest_records([dict(_FULL_RECORD)], project_id="stub_proj")

    config_changed = service.ingest_records(
        [{**_FULL_RECORD, "configuration": {"lr": 0.02, "batch": 32}}], project_id="stub_proj"
    )
    assert (config_changed.imported, config_changed.skipped) == (1, 0)
    assert stub.experiment_writes == 2

    metric_changed = service.ingest_records(
        [
            {
                **_FULL_RECORD,
                "configuration": {"lr": 0.02, "batch": 32},
                "metrics": [
                    {
                        "name": "validation_accuracy",
                        "value": 0.81,
                        "direction": "maximize",
                        "evaluation_split": "validation",
                    }
                ],
            }
        ],
        project_id="stub_proj",
    )
    assert (metric_changed.imported, metric_changed.skipped) == (1, 0)


def test_unrecorded_values_never_become_writes():
    """A record with nothing to say does not fabricate a stored value."""
    stub = _StubRepository()
    service = IngestionService(repository=stub)

    service.ingest_records(
        [{"id": "stub_sparse", "name": "Sparse", "configuration": {"lr": 0.1}}],
        project_id="stub_proj",
    )
    stored = stub.experiments["stub_sparse"]
    assert stored["model_family"] is None
    assert stored["status"] is None
    assert stored["dataset_version_id"] is None
    assert stub.metrics == {}


def test_blank_metric_is_unrecorded_never_zero(test_repo, test_ingestion_service):
    """A blank measurement is unrecorded: reported, omitted, never 0.0."""
    proj = "proj_blank_metric"

    summary = test_ingestion_service.ingest_records(
        [
            {
                "id": "blank_metric_01",
                "name": "Blank Metric Run",
                "model_family": "linear",
                "metrics": [
                    {"name": "accuracy", "value": None, "direction": "maximize"},
                    {"name": "rmse", "value": 1.25},
                ],
            }
        ],
        project_id=proj,
    )

    assert summary.imported == 1
    assert summary.failed == 0
    assert any(e.row == 1 and "accuracy" in e.reason for e in summary.errors)

    # No Metric node claims a value for the blank, and none is 0.0.
    metrics = _nodes(test_repo, proj, "Metric")
    assert list(metrics) == ["metric_blank_metric_01_rmse"]
    assert metrics["metric_blank_metric_01_rmse"]["value"] == 1.25
    assert all(p.get("value") not in (0, 0.0, None) for p in metrics.values())

    # The same holds for a blank cell in a CSV column.
    csv_summary = test_ingestion_service.ingest_csv(
        "run_id,model_family,val_accuracy,rmse\nblank_csv_01,linear,,\n",
        project_id=proj,
    )
    assert csv_summary.imported == 1
    assert {e.row for e in csv_summary.errors} == {1}
    assert _nodes(test_repo, proj, "Metric") == metrics


def test_unknown_metric_direction_is_reported_not_coerced(test_repo, test_ingestion_service):
    """An unrecognised direction is reported; the record still imports."""
    proj = "proj_bad_direction"

    summary = test_ingestion_service.ingest_records(
        [
            {
                "id": "bad_direction_01",
                "name": "Bad Direction Run",
                "model_family": "cnn",
                "metrics": [
                    {"name": "validation_accuracy", "value": 0.91, "direction": "sideways"},
                    {"name": "validation_loss", "value": 0.12, "direction": "minimize"},
                ],
            }
        ],
        project_id=proj,
    )

    assert summary.imported == 1
    assert summary.failed == 0
    assert any(
        e.row == 1 and "validation_accuracy" in e.reason and "direction" in e.reason
        for e in summary.errors
    )

    metrics = _nodes(test_repo, proj, "Metric")
    assert [p["name"] for p in metrics.values()] == ["validation_loss"]
    assert metrics["metric_bad_direction_01_validation_loss"]["direction"] == "minimize"


def test_absent_optional_fields_are_stored_as_absent(test_repo, test_ingestion_service):
    """model_family, status and dataset version stay absent, not defaulted."""
    proj = "proj_absent_fields"

    summary = test_ingestion_service.ingest_records(
        [
            {
                "id": "absent_01",
                "name": "Sparse Record",
                "configuration": {"lr": 0.1},
            }
        ],
        project_id=proj,
    )

    assert summary.imported == 1
    assert summary.failed == 0

    props = _nodes(test_repo, proj, "Experiment")["absent_01"]
    for fabricated in ("model_family", "status"):
        assert fabricated not in props, f"{fabricated} was invented as {props.get(fabricated)!r}"

    # No dataset at all means no invented dataset node or version node.
    assert _nodes(test_repo, proj, "Dataset") == {}
    assert _nodes(test_repo, proj, "DatasetVersion") == {}

    # The record itself reports the absence rather than a fabricated value.
    record, _ = test_ingestion_service.parse_record(
        {"id": "absent_01", "name": "Sparse Record"}, row_num=1, default_project_id=proj
    )
    assert record.model_family is None
    assert record.status is None
    assert record.dataset_version is None
    assert record.dataset_version_id is None


def test_created_at_survives_from_the_source(test_repo, test_ingestion_service):
    """A source timestamp is kept (FR-1 #6) instead of the import time."""
    proj = "proj_created_at"

    summary = test_ingestion_service.ingest_csv(
        "run_id,model_family,created_at,timestamp\n"
        "stamped_01,linear,2024-03-01T10:00:00Z,\n"
        "stamped_02,linear,,2024-04-02T11:30:00Z\n",
        project_id=proj,
    )
    assert summary.imported == 2

    experiments = _nodes(test_repo, proj, "Experiment")
    assert str(experiments["stamped_01"]["created_at"]) == "2024-03-01T10:00:00Z"
    assert str(experiments["stamped_02"]["created_at"]) == "2024-04-02T11:30:00Z"


def test_duplicate_id_in_one_batch_counts_as_skipped(test_repo, test_ingestion_service):
    """Two rows with one id: the later one is skipped, not imported twice."""
    proj = "proj_in_batch_dup"

    summary = test_ingestion_service.ingest_records(
        [
            {"id": "dup_01", "name": "First", "model_family": "linear", "configuration": {"lr": 0.1}},
            {"id": "dup_01", "name": "Second", "model_family": "linear", "configuration": {"lr": 0.2}},
        ],
        project_id=proj,
    )

    assert summary.imported == 1
    assert summary.skipped == 1
    assert summary.failed == 0
    assert any(e.row == 2 and "Duplicate experiment id" in e.reason for e in summary.errors)

    experiments = _nodes(test_repo, proj, "Experiment")
    assert set(experiments) == {"dup_01"}
    assert experiments["dup_01"]["name"] == "First"


def test_reimport_of_an_identical_record_is_skipped(test_repo, test_ingestion_service):
    """Re-importing the same record is idempotent: skipped, no second node."""
    proj = "proj_reimport"
    record = {
        "id": "reimport_01",
        "name": "Repeat Run",
        "model_family": "linear",
        "dataset_name": "housing",
        "dataset_version": "v3",
        "status": "completed",
        "random_seed": 7,
        "duration_seconds": 12.5,
        "compute_cost_gpu_hours": 0.4,
        "created_at": "2024-05-05T00:00:00Z",
        "source_reference": "runs/reimport_01.json",
        "configuration": {"lr": 0.01, "batch": 32},
        "metrics": [
            {
                "name": "validation_accuracy",
                "value": 0.8,
                "direction": "maximize",
                "evaluation_split": "validation",
            }
        ],
    }

    first = test_ingestion_service.ingest_records([dict(record)], project_id=proj)
    assert (first.imported, first.skipped, first.failed) == (1, 0, 0)

    second = test_ingestion_service.ingest_records([dict(record)], project_id=proj)
    assert (second.imported, second.skipped, second.failed) == (0, 1, 0)
    assert any("identical" in e.reason for e in second.errors)

    assert set(_nodes(test_repo, proj, "Experiment")) == {"reimport_01"}
    assert len(_nodes(test_repo, proj, "Metric")) == 1

    # A record that really changed is still applied.
    changed = test_ingestion_service.ingest_records(
        [{**record, "configuration": {"lr": 0.02, "batch": 32}}], project_id=proj
    )
    assert (changed.imported, changed.skipped, changed.failed) == (1, 0, 0)


def test_row_conflicting_with_the_project_is_rejected(test_repo, test_ingestion_service):
    """A row may not write into a project other than the one in the URL."""
    proj = "proj_boundary"

    summary = test_ingestion_service.ingest_records(
        [
            {"id": "in_scope_01", "name": "Mine", "model_family": "linear"},
            {
                "id": "out_of_scope_01",
                "project_id": "some_other_project",
                "name": "Theirs",
                "model_family": "linear",
            },
        ],
        project_id=proj,
    )

    assert summary.imported == 1
    assert summary.failed == 1
    assert any(
        e.row == 2 and "some_other_project" in e.reason and "another project" in e.reason
        for e in summary.errors
    )

    assert set(_nodes(test_repo, proj, "Experiment")) == {"in_scope_01"}
    assert _nodes(test_repo, "some_other_project", "Experiment") == {}


def test_malformed_values_are_reported_with_row_numbers(test_repo, test_ingestion_service):
    """Unparsable values are reported; the rest of the record still imports."""
    proj = "proj_malformed"

    summary = test_ingestion_service.ingest_records(
        [
            {
                "id": "ok_01",
                "name": "Fine",
                "model_family": "linear",
                "metrics": [{"name": "rmse", "value": 1.5}],
            },
            {
                "id": "bad_02",
                "name": "Broken",
                "model_family": "linear",
                "duration_seconds": "not-a-number",
                "metrics": [{"name": "rmse", "value": "oops"}, {"name": "accuracy", "value": 0.5}],
            },
            {"id": "ok_03", "name": "Fine Again", "model_family": "linear", "random_seed": "seed?"},
        ],
        project_id=proj,
    )

    assert summary.imported == 3
    assert summary.failed == 0

    row_two = [e.reason for e in summary.errors if e.row == 2]
    assert any("rmse" in reason for reason in row_two)
    assert any("duration_seconds" in reason for reason in row_two)
    assert any("random_seed" in e.reason for e in summary.errors if e.row == 3)

    experiments = _nodes(test_repo, proj, "Experiment")
    assert experiments["bad_02"].get("duration_seconds") is None
    assert experiments["ok_03"].get("random_seed") is None

    metrics = [p for p in _nodes(test_repo, proj, "Metric").values()]
    assert sorted(p["name"] for p in metrics if p["experiment_id"] == "bad_02") == ["accuracy"]


def test_dataset_ids_and_labels_stay_in_separate_namespaces(test_repo, test_ingestion_service):
    """`dataset_id` is a node id; `dataset_name`/`version` are labels."""
    proj = "proj_dataset_ns"

    summary = test_ingestion_service.ingest_csv(
        "run_id,model_family,dataset_id,dataset_name,version,dataset_version_id\n"
        "ns_01,linear,ds_churn,Customer Churn,v3,dsv_churn_v3\n"
        "ns_02,linear,ds_housing,,v1,\n",
        project_id=proj,
    )
    assert summary.imported == 2

    datasets = _nodes(test_repo, proj, "Dataset")
    assert datasets["ds_churn"]["name"] == "Customer Churn"
    assert "Customer Churn" not in datasets

    versions = _nodes(test_repo, proj, "DatasetVersion")
    assert versions["dsv_churn_v3"]["version"] == "v3"
    assert versions["dsv_churn_v3"]["dataset_id"] == "ds_churn"

    # A row with an id but no label keeps the id and invents no label.
    assert "ds_housing" in datasets
    assert "ds_housing_v1" in versions

    experiments = _nodes(test_repo, proj, "Experiment")
    assert experiments["ns_01"]["dataset_version_id"] == "dsv_churn_v3"
    assert experiments["ns_02"]["dataset_version_id"] == "ds_housing_v1"
