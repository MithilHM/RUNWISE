"""
Model Context Protocol (MCP) Server for RUNWISE.

Exposes FalkorDB graph algorithms and queries as standard MCP tools:
- get_experiment_lineage: Multi-hop derivation and dataset version lineage
- check_experiment_comparability: Rule-based comparability and caveat detection
- trace_invalidation_impact: Multi-hop defect invalidation propagation
- query_regime_evidence: Multi-hop SIMILAR_REGIME and FAILED_UNDER subgraph analysis
- recommend_next_experiment: Evidence-backed hypothesis recommendation under compute budget
- get_memory_summary: 3-tier agent memory taxonomy (Episodic, Semantic, Procedural)
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional

from app.graph.client import graph_client
from app.graph.repository import GraphRepository
from app.graph.runtime_queries import runtime_graph_service
from app.services.comparability import ComparabilityEngine
from app.services.evidence import partition_by_trust
from app.services.memory_service import memory_service
from app.services.recommendations import RecommendationEngine, parse_objective

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool Specifications (MCP Schema)
# ---------------------------------------------------------------------------
MCP_TOOLS = [
    {
        "name": "get_experiment_lineage",
        "description": "Inspect multi-hop derivation lineage (PARENT_OF) and dataset version usage of an experiment in FalkorDB.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment_id": {
                    "type": "string",
                    "description": "Identifier of the experiment to inspect (e.g., 'exp_001')",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID (defaults to 'project_demo')",
                    "default": "project_demo",
                },
            },
            "required": ["experiment_id"],
        },
    },
    {
        "name": "check_experiment_comparability",
        "description": "Evaluate whether two experiments in FalkorDB are strictly comparable or confounded under controlled experimental conditions.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "experiment_a_id": {
                    "type": "string",
                    "description": "First experiment ID",
                },
                "experiment_b_id": {
                    "type": "string",
                    "description": "Second experiment ID to compare against",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID (defaults to 'project_demo')",
                    "default": "project_demo",
                },
            },
            "required": ["experiment_a_id", "experiment_b_id"],
        },
    },
    {
        "name": "trace_invalidation_impact",
        "description": "Trace multi-hop downstream invalidation through USES_DATASET_VERSION and PARENT_OF derivation paths when a dataset defect or leakage is reported.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_version_id": {
                    "type": "string",
                    "description": "Defective dataset version ID (e.g. 'tabular_fraud_v1')",
                },
                "reason": {
                    "type": "string",
                    "description": "Root cause explanation for the correction",
                    "default": "Data leakage between train and test splits",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID (defaults to 'project_demo')",
                    "default": "project_demo",
                },
            },
            "required": ["dataset_version_id"],
        },
    },
    {
        "name": "query_regime_evidence",
        "description": "Execute multi-hop Cypher queries across SIMILAR_REGIME and FAILED_UNDER subgraphs in FalkorDB to find relevant priors and avoid known failure regimes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dataset_name": {
                    "type": "string",
                    "description": "Dataset name to find similar regimes for",
                    "default": "credit_card_fraud",
                },
            },
            "required": [],
        },
    },
    {
        "name": "recommend_next_experiment",
        "description": "Generate an evidence-backed hypothesis and controlled experiment recommendation bounded by compute budget (GPU-hours).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "objective": {
                    "type": "string",
                    "description": "Research goal (e.g., 'Optimize PR-AUC for fraud detection under class imbalance')",
                },
                "compute_budget_gpu_hours": {
                    "type": "number",
                    "description": "Maximum compute budget in GPU hours",
                    "default": 2.0,
                },
                "project_id": {
                    "type": "string",
                    "description": "Project ID (defaults to 'project_demo')",
                    "default": "project_demo",
                },
            },
            "required": ["objective"],
        },
    },
    {
        "name": "get_memory_summary",
        "description": "Query RUNWISE's 3-tier FalkorDB agent memory (Episodic experiment logs, Semantic regime similarity, and Procedural ranking rules).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project ID (defaults to 'project_demo')",
                    "default": "project_demo",
                },
            },
            "required": [],
        },
    },
]


class MCPServer:
    """Model Context Protocol (MCP) server exposing FalkorDB agent tools."""

    def __init__(self, repo: Optional[GraphRepository] = None):
        self.repo = repo or GraphRepository(graph_client)
        self.engine = RecommendationEngine(self.repo)
        self.comparability = ComparabilityEngine(self.repo)

    def dispatch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a standard JSON-RPC 2.0 MCP request synchronously or asynchronously."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "runwise-graph-agent-mcp",
                        "version": "1.0.0",
                    },
                },
            }

        elif method in ("notifications/initialized", "initialized"):
            return {}

        elif method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        elif method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS},
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})
            try:
                res_content = self.call_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res_content, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as e:
                logger.exception("Error executing MCP tool '%s': %s", tool_name, e)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution failed: {str(e)}"}],
                        "isError": True,
                    },
                }

        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32601,
                    "message": f"Method '{method}' not found",
                },
            }

    def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """Executes the requested graph tool."""
        project_id = args.get("project_id", "project_demo")

        if name == "get_experiment_lineage":
            exp_id = args["experiment_id"]
            lineage = self.repo.get_experiment_lineage(exp_id, project_id=project_id)
            if not lineage:
                return {"found": False, "message": f"Experiment '{exp_id}' not found in project '{project_id}'"}
            return {
                "found": True,
                "experiment_id": exp_id,
                "nodes": [n.model_dump() for n in lineage.nodes],
                "edges": [e.model_dump() for e in lineage.edges],
            }

        elif name == "check_experiment_comparability":
            exp_a = args["experiment_a_id"]
            exp_b = args["experiment_b_id"]
            assessment = self.comparability.assess(exp_a, exp_b, project_id=project_id)
            return {
                "experiment_a": exp_a,
                "experiment_b": exp_b,
                "status": assessment.status.value,
                "confidence": assessment.confidence,
                "caveats": assessment.caveats,
                "summary": assessment.summary,
            }

        elif name == "trace_invalidation_impact":
            ds_ver = args["dataset_version_id"]
            reason = args.get("reason", "Defect discovered in dataset")
            # Dry run traversal or query existing impact
            impact = self.repo.get_correction_impact(f"corr_{ds_ver}")
            if not impact:
                # Direct traversal check
                exp_runs = self.repo.list_experiments(project_id, dataset_version_id=ds_ver)
                return {
                    "dataset_version_id": ds_ver,
                    "reason": reason,
                    "directly_affected_runs": [e.id for e in exp_runs],
                    "status": "simulated_traversal",
                }
            return {
                "correction_id": impact.correction_id,
                "directly_affected_experiments": impact.directly_affected_experiments,
                "transitively_affected_experiments": impact.transitively_affected_experiments,
                "total_affected": len(impact.directly_affected_experiments) + len(impact.transitively_affected_experiments),
            }

        elif name == "query_regime_evidence":
            ds_name = args.get("dataset_name", "credit_card_fraud")
            similar = runtime_graph_service.query_similar_regimes(ds_name)
            failed = runtime_graph_service.query_failed_subgraphs(ds_name)
            return {
                "dataset_name": ds_name,
                "similar_regimes_count": len(similar),
                "similar_regimes": similar,
                "failed_subgraphs_count": len(failed),
                "failed_subgraphs": failed,
            }

        elif name == "recommend_next_experiment":
            obj_text = args["objective"]
            budget = float(args.get("compute_budget_gpu_hours", 2.0))
            recs = self.engine.recommend(
                project_id=project_id,
                objective=obj_text,
                compute_budget_gpu_hours=budget,
            )
            return {
                "objective": obj_text,
                "budget_gpu_hours": budget,
                "recommendation_count": len(recs),
                "recommendations": [r.model_dump() for r in recs],
            }

        elif name == "get_memory_summary":
            return memory_service.get_memory_breakdown(project_id)

        else:
            raise ValueError(f"Unknown tool name: {name}")

    async def run_stdio(self) -> None:
        """Runs the MCP server over standard input / output (stdio transport)."""
        logger.info("RUNWISE MCP Server started in stdio transport mode.")
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        while True:
            line = await reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            try:
                req = json.loads(line_str)
                resp = self.dispatch(req)
                if resp:
                    sys.stdout.write(json.dumps(resp) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


mcp_server = MCPServer()
