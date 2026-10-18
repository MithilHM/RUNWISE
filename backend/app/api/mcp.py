"""FastAPI router exposing MCP (Model Context Protocol) endpoints.

Conforms strictly to Track 01 (Agents That Act on Connected Data - Tool selection through MCP).
"""

from __future__ import annotations

from typing import Any, Dict
from fastapi import APIRouter, Body

from app.mcp.server import mcp_server, MCP_TOOLS

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get(
    "/tools",
    summary="List available Model Context Protocol (MCP) graph tools (Track 01)",
    description="Returns the schemas and descriptions of all FalkorDB graph reasoning tools available to agents.",
)
def list_tools() -> Dict[str, Any]:
    return {"tools": MCP_TOOLS}


@router.post(
    "/rpc",
    summary="Execute Model Context Protocol (MCP) JSON-RPC request (Track 01)",
    description="Accepts standard MCP JSON-RPC 2.0 messages (initialize, tools/list, tools/call) and returns results.",
)
def handle_mcp_rpc(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
    return mcp_server.dispatch(request)
