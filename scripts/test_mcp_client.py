"""
Test client for RUNWISE Model Context Protocol (MCP) Server.
Demonstrates tool discovery and agent tool execution.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.mcp.server import mcp_server


def test_mcp():
    print("Testing MCP initialize...")
    init_res = mcp_server.dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert "result" in init_res, f"Init failed: {init_res}"
    print(f"[OK] Initialized: {init_res['result']['serverInfo']['name']}")

    print("\nTesting MCP tools/list...")
    tools_res = mcp_server.dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    tools = tools_res["result"]["tools"]
    print(f"[OK] Found {len(tools)} graph reasoning tools:")
    for t in tools:
        print(f"  - {t['name']}: {t['description'][:60]}...")

    print("\nTesting MCP tool call 'get_memory_summary'...")
    call_res = mcp_server.dispatch({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_memory_summary",
            "arguments": {"project_id": "project_demo"},
        },
    })
    assert not call_res["result"]["isError"], f"Tool call error: {call_res}"
    print("[OK] Tool 'get_memory_summary' executed successfully!")

    print("\nTesting MCP tool call 'query_regime_evidence'...")
    regime_res = mcp_server.dispatch({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "query_regime_evidence",
            "arguments": {"dataset_name": "credit_card_fraud"},
        },
    })
    assert not regime_res["result"]["isError"], f"Tool call error: {regime_res}"
    print("[OK] Tool 'query_regime_evidence' executed successfully!")

    print("\nAll MCP tests passed with 100% success!")


if __name__ == "__main__":
    test_mcp()
