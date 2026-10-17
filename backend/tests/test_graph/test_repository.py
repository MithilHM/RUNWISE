"""
Tests for GraphRepository operations, lineage traversal, and query services.
"""

from app.models.contracts import MetricModel


def test_project_crud(test_repo):
    """Test creating and retrieving projects."""
    p = test_repo.create_or_update_project("p_test", "Test Project", "Description for test")
    assert p["id"] == "p_test"
    assert p["name"] == "Test Project"

    fetched = test_repo.get_project("p_test")
    assert fetched is not None
    assert fetched["id"] == "p_test"
    assert fetched["experiment_count"] == 0


def test_experiment_crud_and_relationships(test_repo):
    """Test creating experiments, configurations, datasets, and relationships."""
    test_repo.create_or_update_project("p1", "Project 1")
    test_repo.create_or_update_dataset("d1", "Dataset 1")
    test_repo.create_or_update_dataset_version("d1_v1", "d1", "v1")
    test_repo.create_or_update_configuration("cfg_1", '{"lr": 0.01}', "hash123")

    exp_data = {
        "id": "exp_01",
        "project_id": "p1",
        "name": "Exp 01",
        "model_family": "transformer",
        "status": "completed",
        "config_id": "cfg_1",
        "dataset_version_id": "d1_v1",
        "duration_seconds": 150.0,
        "compute_cost_gpu_hours": 0.5
    }
    created = test_repo.create_or_update_experiment(exp_data)
    assert created["id"] == "exp_01"

    # Add metrics
    test_repo.create_or_update_metric(
        metric_id="m1",
        experiment_id="exp_01",
        name="accuracy",
        value=0.95,
        direction="maximize"
    )

    # Fetch full detail
    detail = test_repo.get_experiment("exp_01")
    assert detail is not None
    assert detail.id == "exp_01"
    assert detail.model_family == "transformer"
    assert detail.dataset_version == "v1"
    assert len(detail.metrics) == 1
    assert detail.metrics[0].name == "accuracy"
    assert detail.metrics[0].value == 0.95
    assert detail.configuration == {"lr": 0.01}


def test_lineage_traversal(test_repo):
    """Test parent-child experiment lineage traversal."""
    test_repo.create_or_update_project("p_lin", "Lineage Project")

    # Create lineage chain: E1 -> E2 -> E3
    test_repo.create_or_update_experiment({"id": "E1", "project_id": "p_lin", "name": "Exp 1"})
    test_repo.create_or_update_experiment({"id": "E2", "project_id": "p_lin", "name": "Exp 2", "parent_experiment_id": "E1"})
    test_repo.create_or_update_experiment({"id": "E3", "project_id": "p_lin", "name": "Exp 3", "parent_experiment_id": "E2"})

    # Check lineage centered at E2
    lineage_e2 = test_repo.get_experiment_lineage("E2")
    node_ids = {n.id for n in lineage_e2.nodes}
    assert "E1" in node_ids
    assert "E2" in node_ids
    assert "E3" in node_ids
    assert len(lineage_e2.edges) == 2


def test_related_experiments(test_repo):
    """Test finding related experiments sharing datasets or configs."""
    test_repo.create_or_update_project("p_rel", "Related Project")
    test_repo.create_or_update_dataset("d_rel", "Data Rel")
    test_repo.create_or_update_dataset_version("d_rel_v1", "d_rel", "v1")

    test_repo.create_or_update_experiment({"id": "rel_1", "project_id": "p_rel", "dataset_version_id": "d_rel_v1"})
    test_repo.create_or_update_experiment({"id": "rel_2", "project_id": "p_rel", "dataset_version_id": "d_rel_v1"})
    test_repo.create_or_update_experiment({"id": "rel_3", "project_id": "p_rel"})

    related = test_repo.find_related_experiments("rel_1")
    assert "rel_2" in related["same_dataset_version"]
    assert "rel_3" not in related["same_dataset_version"]


def test_nonexistent_experiment(test_repo):
    """Test querying non-existent experiment returns None without exception."""
    res = test_repo.get_experiment("does_not_exist_xyz")
    assert res is None
