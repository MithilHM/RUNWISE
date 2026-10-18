"""
Tests for Model Context Protocol (MCP) server and endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mcp.server import mcp_server, MCP_TOOLS


@pytest.fixture
def client():
    return TestClient(app)


def test_mcp_tools_list_spec():
    """Verify MCP tools specification conforms to schema."""
    tools = MCP_TOOLS
    assert len(tools) >= 5
    names = [t["name"] for t in tools]
    assert "get_experiment_lineage" in names
    assert "check_experiment_comparability" in names
    assert "trace_invalidation_impact" in names
    assert "query_regime_evidence" in names
    assert "recommend_next_experiment" in names
    assert "get_memory_summary" in names


def test_mcp_dispatch_initialize():
    """Test standard MCP initialize handshake."""
    res = mcp_server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert "tools" in res["result"]["capabilities"]
    assert res["result"]["serverInfo"]["name"] == "runwise-graph-agent-mcp"


def test_mcp_dispatch_tools_list():
    """Test MCP tools/list."""
    res = mcp_server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    assert res["id"] == 2
    assert "tools" in res["result"]
    assert len(res["result"]["tools"]) >= 5


def test_mcp_dispatch_tools_call_memory_summary():
    """Test calling get_memory_summary via MCP."""
    res = mcp_server.dispatch({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_memory_summary",
            "arguments": {"project_id": "project_demo"},
        },
    })
    assert res["id"] == 3
    assert not res["result"]["isError"]
    content = res["result"]["content"][0]["text"]
    assert "episodic" in content
    assert "semantic" in content
    assert "procedural" in content


def test_mcp_dispatch_unknown_method():
    """Test handling of unknown MCP method."""
    res = mcp_server.dispatch({"jsonrpc": "2.0", "id": 99, "method": "unknown/method", "params": {}})
    assert res["id"] == 99
    assert "error" in res
    assert res["error"]["code"] == -32601


def test_mcp_http_endpoints(client):
    """Test MCP HTTP endpoints: /api/v1/mcp/tools and /api/v1/mcp/rpc."""
    tools_res = client.get("/api/v1/mcp/tools")
    assert tools_res.status_code == 200
    assert "tools" in tools_res.json()

    rpc_res = client.post("/api/v1/mcp/rpc", json={
        "jsonrpc": "2.0",
        "id": "http_test_1",
        "method": "tools/list",
        "params": {},
    })
    assert rpc_res.status_code == 200
    assert rpc_res.json()["id"] == "http_test_1"
    assert "tools" in rpc_res.json()["result"]
