"""
Tests for Graph schema initialization and idempotency.
"""

from app.graph.schema import initialize_schema, get_existing_indices, INDICES_TO_CREATE


def test_schema_initialization(test_client):
    """Verify initialize_schema sets up indices on a fresh graph."""
    graph = test_client.get_graph()
    res = initialize_schema(graph)
    assert res["status"] == "ready"
    assert len(res["failed"]) == 0

    indices = get_existing_indices(graph)
    # Check key indices exist
    assert "Project.id" in indices
    assert "Experiment.id" in indices
    assert "DatasetVersion.id" in indices


def test_schema_initialization_idempotency(test_client):
    """Verify running schema initialization repeatedly is idempotent and safe."""
    graph = test_client.get_graph()
    # First run
    res1 = initialize_schema(graph)
    assert len(res1["failed"]) == 0

    # Second run should skip already indexed properties without error
    res2 = initialize_schema(graph)
    assert res2["status"] == "ready"
    assert len(res2["failed"]) == 0
    assert len(res2["skipped"]) >= len(res1["created"])
