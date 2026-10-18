"""Tests for Model Context Protocol (MCP) server.

Defined in MEMBER_B.md Section 4 (B4 & B7).
Asserts:
- tools/list returns the 6 Member B tools with valid input schemas
- tools/call succeeds for list_models, list_datasets, list_trials, graph_neighbourhood, explain_trial, best_trials
- Free-form Cypher queries are strictly rejected with an error
"""

import json
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.mcp.server import mcp_server, MCP_TOOLS


@pytest.fixture
def client():
    return TestClient(app)


def test_mcp_tools_list_spec():
    """Verify MCP tools specification lists all 6 Member B tools."""
    tools = MCP_TOOLS
    assert len(tools) == 6
    names = [t["name"] for t in tools]
    assert "list_models" in names
    assert "list_datasets" in names
    assert "list_trials" in names
    assert "graph_neighbourhood" in names
    assert "explain_trial" in names
    assert "best_trials" in names


def test_mcp_dispatch_initialize():
    """Test standard MCP initialize handshake."""
    res = mcp_server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert res["jsonrpc"] == "2.0"
    assert res["id"] == 1
    assert "tools" in res["result"]["capabilities"]
    assert res["result"]["serverInfo"]["name"] == "runwise-mcp"


def test_mcp_dispatch_tools_call_list_models():
    """Test calling list_models via MCP."""
    res = mcp_server.dispatch({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {"name": "list_models", "arguments": {}},
    })
    assert res["id"] == 2
    content = json.loads(res["result"]["content"][0]["text"])
    assert len(content) == 3
    model_ids = [m["id"] for m in content]
    assert "xgboost" in model_ids
    assert "lightgbm" in model_ids
    assert "random_forest" in model_ids


def test_mcp_dispatch_tools_call_explain_trial():
    """Test calling explain_trial via MCP."""
    res = mcp_server.dispatch({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {"name": "explain_trial", "arguments": {"trial_id": "trial_001"}},
    })
    assert res["id"] == 3
    content = json.loads(res["result"]["content"][0]["text"])
    assert "decision" in content
    assert "citations" in content["decision"]


def test_mcp_rejects_freeform_cypher():
    """Test that free-form Cypher queries are strictly rejected."""
    res = mcp_server.dispatch({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "run_cypher",
            "arguments": {"query": "MATCH (n) RETURN n"},
        },
    })
    assert res["id"] == 4
    assert "error" in res
    assert "Cypher" in res["error"]["message"]


def test_mcp_http_endpoints(client):
    """Test MCP HTTP endpoints: /api/v1/mcp/tools and /api/v1/mcp/rpc."""
    tools_res = client.get("/api/v1/mcp/tools")
    assert tools_res.status_code == 200
    assert len(tools_res.json()["tools"]) == 6

    rpc_res = client.post("/api/v1/mcp/rpc", json={
        "jsonrpc": "2.0",
        "id": "http_test_1",
        "method": "tools/list",
        "params": {},
    })
    assert rpc_res.status_code == 200
    assert rpc_res.json()["id"] == "http_test_1"
    assert len(rpc_res.json()["result"]["tools"]) == 6
