"""FalkorDB schema for the RUNWISE graph.

Ten node labels and fourteen relationship types. The shape is chosen so that the
whole story of a run is reachable by traversal rather than by filtering:

    (:Dataset)-[:HAS_COLUMN]->(:Column)
    (:Run)-[:ON_DATASET]->(:Dataset)
    (:Run)-[:TARGETS_MODEL]->(:Model)
    (:Run)-[:HAS_TRIAL]->(:Trial)-[:USES_MODEL]->(:Model)
                                |
                                +-[:USES_SETTING]->(:Setting)-[:FOR_PARAM]->(:Param)
                                +                                  |
                                +                                  +->[:OF_MODEL]->(:Model)
                                +[:DERIVED_FROM]->(:Trial)
    (:Decision)-[:CHOSE]->(:Trial)
    (:Decision)-[:CITES]->(:Model|:Param|:Setting|:Trial)   <- the F7 highlight mechanism
    (:Decision)-[:DURING]->(:Run)
    (:Agent)-[:DECIDED]->(:Decision)
    (:LogEntry)-[:IN_RUN]->(:Run)

Two decisions worth stating:

- **Metrics are properties of `:Trial`, not nodes.** A `Metric` node per trial
  quadruples the graph and turns the visualisation into soup. The values live on
  the trial and the frontend shades them.
- **`:Setting` is a real node, deduplicated by value.** Ten trials that share
  `max_depth=6` create one `Setting` and ten edges to it. That is what makes
  "which values of this parameter have been tried, and what did each score?"
  a traversal instead of a scan.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

__all__ = ["NODE_LABELS", "RELATIONSHIPS", "INDEXES", "initialize_schema"]


NODE_LABELS: List[str] = [
    "Dataset",
    "Column",
    "Run",
    "Model",
    "Param",
    "Setting",
    "Trial",
    "Decision",
    "LogEntry",
    "Agent",
]

RELATIONSHIPS: List[str] = [
    "HAS_COLUMN",
    "ON_DATASET",
    "TARGETS_MODEL",
    "HAS_TRIAL",
    "USES_MODEL",
    "USES_SETTING",
    "DERIVED_FROM",
    "FOR_PARAM",
    "OF_MODEL",
    "HAS_PARAM",
    "CHOSE",
    "CITES",
    "DURING",
    "DECIDED",
    "IN_RUN",
]

# Only the properties the read queries actually filter or join on get an index.
# FalkorDB indexes every property of a label when given one, so adding indices
# for display-only fields costs writes and buys nothing.
INDEXES: List[tuple[str, str]] = [
    ("Dataset", "id"),
    ("Column", "dataset_id"),
    ("Run", "id"),
    ("Run", "dataset_id"),
    ("Model", "id"),
    ("Param", "id"),
    ("Param", "model_id"),
    ("Setting", "id"),
    ("Setting", "param_id"),
    ("Setting", "model_id"),
    ("Trial", "id"),
    ("Trial", "run_id"),
    ("Trial", "model_id"),
    ("Decision", "id"),
    ("Decision", "run_id"),
    ("LogEntry", "run_id"),
    ("LogEntry", "seq"),
    ("Agent", "name"),
    ("Agent", "id"),
]


def initialize_schema(graph: Any) -> Dict[str, Any]:
    """Creates the indices. Idempotent.

    FalkorDB raises on a duplicate index, and that is the normal case here since
    this runs on every startup. Index creation that already exists is therefore
    treated as success rather than as a failure.
    """
    created: List[str] = []
    skipped: List[str] = []
    failed: List[Dict[str, str]] = []

    for label, prop in INDEXES:
        statement = f"CREATE INDEX ON :{label}({prop})"
        try:
            graph.query(statement)
            created.append(f":{label}({prop})")
        except Exception as exc:  # already exists is the expected case
            message = str(exc).lower()
            if "exist" in message or "index" in message:
                skipped.append(f":{label}({prop})")
            else:
                failed.append({"index": f":{label}({prop})", "error": str(exc)})

    summary = {
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "node_labels": len(NODE_LABELS),
        "relationships": len(RELATIONSHIPS),
    }
    if failed:
        logger.warning("Schema init had %d index failure(s): %s", len(failed), failed)
    else:
        logger.info(
            "FalkorDB schema ready: %d new index(es), %d already present",
            len(created), len(skipped),
        )
    return summary