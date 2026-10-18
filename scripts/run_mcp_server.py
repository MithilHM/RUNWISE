"""
Standalone runner for RUNWISE Model Context Protocol (MCP) Server.
Allows external MCP clients (Claude Desktop, Cursor, Antigravity, or subagents)
to connect via standard input/output (stdio).

Usage:
    python scripts/run_mcp_server.py
"""

import asyncio
import os
import sys

# Ensure backend is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_DIR = os.path.join(BASE_DIR, "backend")
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.mcp.server import mcp_server


def main():
    asyncio.run(mcp_server.run_stdio())


if __name__ == "__main__":
    main()
