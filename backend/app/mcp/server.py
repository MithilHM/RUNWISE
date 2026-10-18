"""Model Context Protocol (MCP) Server for RUNWISE.

Defined in MEMBER_B.md Section 4 (B4 — MCP tool surface).
Provides six JSON-RPC 2.0 tools for autonomous agents to inspect and act on connected data:
1. list_models: the 3 models with their full parameter spaces
2. list_datasets: ingested datasets and their profiles
3. list_trials: trials for a run, filterable by model / status
4. graph_neighbourhood: N-hop traversal around a node (act on connected data)
5. explain_trial: the F7 why record, including citations
6. best_trials: leaderboard for a run, correct direction per metric

No tool accepts free-form Cypher. MCP is read-only.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from app.api.runs import harness_service
from app.services import catalog
from app.services.ingestion import get_ingestor

logger = logging.getLogger(__name__)

MCP_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_models",
        "description": "List the 3 supported models (XGBoost, LightGBM, Random Forest) with their full parameter spaces.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_datasets",
        "description": "List all ingested datasets and their profiling summaries.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "list_trials",
        "description": "List trials for a specific run, optionally filtered by model_id or status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Identifier of the run"},
                "model_id": {"type": "string", "description": "Optional model filter (e.g. 'xgboost')"},
                "status": {"type": "string", "description": "Optional status filter ('SUCCESS' | 'FAILED')"},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "graph_neighbourhood",
        "description": "Multi-hop graph traversal around a specific node (N-hop neighbourhood).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "node_id": {"type": "string", "description": "Root node ID to inspect"},
                "hops": {"type": "integer", "description": "Traversal depth (1 to 3)", "default": 1},
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "explain_trial",
        "description": "Retrieve the F7 explainability record (prose justification, citations, and evidence chips) for a trial.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Identifier of the run the trial belongs to"},
                "trial_id": {"type": "string", "description": "Identifier of the trial to explain"},
            },
            # `run_id` is required rather than optional. A trial id alone is
            # ambiguous across runs, and resolving one without it could return a
            # decision belonging to a different run.
            "required": ["run_id", "trial_id"],
        },
    },
    {
        "name": "best_trials",
        "description": "Retrieve the ranked trial leaderboard for a run in the correct direction per metric.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string", "description": "Identifier of the run"},
                "metric": {"type": "string", "description": "Target metric to rank by (defaults to run metric)"},
            },
            "required": ["run_id"],
        },
    },
]


class MCPServer:
    """JSON-RPC 2.0 handler for the 6 Member B MCP tools."""

    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        msg_id = request.get("id", 1)
        method = request.get("method", "")

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "runwise-mcp", "version": "1.0.0"},
                    "capabilities": {"tools": {}},
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": MCP_TOOLS},
            }

        if method == "tools/call":
            params = request.get("params", {})
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})

            # Explicit rule: Free-form Cypher is rejected
            if "cypher" in tool_name.lower() or "cypher" in arguments or "query" in arguments:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {
                        "code": -32600,
                        "message": "Free-form Cypher queries are strictly prohibited in RUNWISE MCP. Use designated domain tools.",
                    },
                }

            try:
                res = self.execute_tool(tool_name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32000, "message": str(exc)},
                }

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method '{method}' not found"},
        }

    def execute_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name == "list_models":
            return [m.model_dump() for m in catalog.list_models()]

        if tool_name == "list_datasets":
            return [p.model_dump() for p in get_ingestor().list_datasets()]

        if tool_name == "list_trials":
            run_id = args.get("run_id")
            model_id = args.get("model_id")
            status = args.get("status")

            h_run = harness_service.get_run(run_id)
            if not h_run:
                # Check graph store
                snapshot = harness_service.graph_store.read_graph(run_id=run_id)
                trials = [n.props for n in snapshot.nodes if n.type == "Trial"]
            else:
                snapshot = harness_service.graph_store.read_graph(run_id=run_id)
                trials = [n.props for n in snapshot.nodes if n.type == "Trial"]

            if model_id:
                trials = [t for t in trials if t.get("model_id") == model_id]
            if status:
                trials = [t for t in trials if t.get("status") == status]
            return trials

        if tool_name == "graph_neighbourhood":
            node_id = args.get("node_id", "")
            hops = min(max(args.get("hops", 1), 1), 3)

            snapshot = harness_service.graph_store.read_graph()
            visited_nodes = {node_id}
            current_level = {node_id}

            for _ in range(hops):
                next_level = set()
                for edge in snapshot.edges:
                    if edge.source in current_level:
                        next_level.add(edge.target)
                    elif edge.target in current_level:
                        next_level.add(edge.source)
                visited_nodes.update(next_level)
                current_level = next_level

            nodes = [n.model_dump() for n in snapshot.nodes if n.id in visited_nodes]
            edges = [
                e.model_dump()
                for e in snapshot.edges
                if e.source in visited_nodes and e.target in visited_nodes
            ]
            return {"nodes": nodes, "edges": edges, "hops": hops}

        if tool_name == "explain_trial":
            run_id = args.get("run_id", "")
            trial_id = args.get("trial_id", "")
            why_record = harness_service.decision_maker.explain(run_id, trial_id)
            return why_record.model_dump()

        if tool_name == "best_trials":
            run_id = args.get("run_id", "")
            metric = args.get("metric", "f1_macro")
            snapshot = harness_service.graph_store.read_graph(run_id=run_id)
            trials = [n.props for n in snapshot.nodes if n.type == "Trial" and n.props.get("metric_value") is not None]

            # Rank higher is better for accuracy, f1_macro, roc_auc, pr_auc; lower for loss
            reverse = "loss" not in metric.lower()
            trials.sort(key=lambda x: x.get("metric_value", 0.0), reverse=reverse)
            return trials

        raise ValueError(f"Unknown tool '{tool_name}'")


mcp_server = MCPServer()
