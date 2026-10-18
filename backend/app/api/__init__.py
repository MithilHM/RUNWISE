"""API routers for RUNWISE.

Four routers, one per surface the harness exposes, plus the MCP endpoint. The
copilot, recommendations, corrections, projects and experiment routers were
removed when the product was reduced to its seven features.
"""

from app.api.catalog import router as catalog_router
from app.api.datasets import router as datasets_router
from app.api.graph import router as graph_router
from app.api.mcp import router as mcp_router
from app.api.runs import router as runs_router

__all__ = [
    "catalog_router",
    "datasets_router",
    "graph_router",
    "mcp_router",
    "runs_router",
]