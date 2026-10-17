"""Live FalkorDB test fixtures for Person A graph, ingestion, and impact test suites."""

from __future__ import annotations

import pytest

from app.core.config import settings
from app.graph.client import FalkorDBClient, graph_client
from app.graph.schema import initialize_schema
from app.graph.repository import GraphRepository
from app.services.ingestion import IngestionService
from app.services.impact_analysis import ImpactAnalysisService

TEST_GRAPH_NAME = "runwise_test_graph"


@pytest.fixture(scope="session")
def falkordb_available():
    """Checks if a live FalkorDB instance is reachable."""
    client = FalkorDBClient(graph_name=TEST_GRAPH_NAME)
    health = client.check_health()
    return health.get("status") == "healthy"


@pytest.fixture
def test_client(falkordb_available):
    """Provides a FalkorDBClient pointed to an isolated test graph."""
    if not falkordb_available:
        pytest.skip("Live FalkorDB instance is not reachable on localhost:6379")

    orig_graph = settings.FALKORDB_GRAPH
    settings.FALKORDB_GRAPH = TEST_GRAPH_NAME
    graph_client.default_graph_name = TEST_GRAPH_NAME
    graph_client.delete_graph(TEST_GRAPH_NAME)

    client = FalkorDBClient(graph_name=TEST_GRAPH_NAME)
    client.delete_graph(TEST_GRAPH_NAME)
    graph = client.get_graph(TEST_GRAPH_NAME)
    initialize_schema(graph)

    yield client

    client.delete_graph(TEST_GRAPH_NAME)
    client.close()
    graph_client.delete_graph(TEST_GRAPH_NAME)
    graph_client.close()
    settings.FALKORDB_GRAPH = orig_graph
    graph_client.default_graph_name = orig_graph


@pytest.fixture
def test_repo(test_client):
    """Provides a GraphRepository instance bound to the test graph."""
    return GraphRepository(client=test_client)


@pytest.fixture
def test_ingestion_service(test_repo):
    """Provides an IngestionService bound to the test repository."""
    return IngestionService(repository=test_repo)


@pytest.fixture
def test_impact_service(test_repo):
    """Provides an ImpactAnalysisService bound to the test repository."""
    return ImpactAnalysisService(repository=test_repo)
