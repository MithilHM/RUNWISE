"""Graph snapshot API router.

Defined in MEMBER_B.md Section 4 (B2 — Routers).
Exposes GET /api/v1/graph for force-directed dependency graph rendering.
"""

from __future__ import annotations

import os
from typing import Optional
from fastapi import APIRouter, Query

from app.api.errors import graph_unreachable
from app.api.runs import harness_service
from app.models.contracts import GraphSnapshot

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("", response_model=GraphSnapshot, summary="Get dependency graph snapshot")
async def get_graph(
    run_id: Optional[str] = Query(None, description="Optional filter by run_id"),
    dataset_id: Optional[str] = Query(None, description="Optional filter by dataset_id"),
) -> GraphSnapshot:
    # If production mode and FalkorDB requested but not reachable, raise 503
    # Check if RUNWISE_FAKE=1 is set or if fake graph store is active
    is_fake = os.environ.get("RUNWISE_FAKE", "1") == "1"

    if not is_fake:
        # In real production mode, verify FalkorDB reachability
        try:
            from app.graph.client import graph_client
            if not graph_client.is_connected():
                raise graph_unreachable("FalkorDB is unreachable. Please verify the container is running.")
        except Exception as exc:
            raise graph_unreachable(f"FalkorDB is unreachable: {exc}")

    snapshot = harness_service.graph_store.read_graph(run_id=run_id, dataset_id=dataset_id)
    return snapshot

