"""
Tests for FalkorDB client connection, health checks, parameterization, and error handling.
"""

import pytest
from app.graph.client import (
    FalkorDBClient,
    FalkorDBConnectionError,
    FalkorDBQueryError
)


def test_client_configuration():
    """Verify FalkorDBClient instantiates with configurable settings."""
    client = FalkorDBClient(host="127.0.0.1", port=6379, graph_name="custom_graph")
    assert client.host == "127.0.0.1"
    assert client.port == 6379
    assert client.default_graph_name == "custom_graph"


def test_client_health_check(test_client):
    """Verify health check returns healthy status on active instance."""
    health = test_client.check_health()
    assert health["status"] == "healthy"
    assert "latency_ms" in health
    assert health["graph"] == test_client.default_graph_name


def test_client_unhealthy_on_invalid_port():
    """Verify client reports unhealthy and captures error on unreachable endpoint."""
    bad_client = FalkorDBClient(host="127.0.0.1", port=59999, graph_name="bad_graph")
    health = bad_client.check_health()
    assert health["status"] == "unhealthy"
    assert "error" in health


def test_client_parameterized_query(test_client):
    """Verify parameterized Cypher queries execute and return values safely."""
    cypher = "RETURN $val AS res, $num AS number"
    params = {"val": "hello_falkor", "num": 42}
    rows = test_client.execute_query(cypher, params)
    assert len(rows) == 1
    assert rows[0][0] == "hello_falkor"
    assert rows[0][1] == 42


def test_client_query_syntax_error(test_client):
    """Verify invalid Cypher query raises FalkorDBQueryError."""
    with pytest.raises(FalkorDBQueryError):
        test_client.execute_query("MALFORMED CYPHER STATEMENT")
