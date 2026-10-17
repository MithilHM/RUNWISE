"""
FalkorDB Graph Schema Definitions and Initialization for RUNWISE.
Conforms strictly to RUNWISE PRD Section 8.

Defines:
- Node labels and property schemas
- Relationship types and connection semantics
- Idempotent index initialization
"""

import logging
from typing import Dict, List, Set, Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Node Labels (PRD Section 8)
# ---------------------------------------------------------------------------
class NodeLabel:
    PROJECT = "Project"
    EXPERIMENT = "Experiment"
    CONFIGURATION = "Configuration"
    DATASET = "Dataset"
    DATASET_VERSION = "DatasetVersion"
    METRIC = "Metric"
    HYPOTHESIS = "Hypothesis"
    CLAIM = "Claim"
    EVIDENCE = "Evidence"
    CORRECTION = "Correction"
    RECOMMENDATION = "Recommendation"
    REVIEW_ITEM = "ReviewItem"
    RESEARCHER = "Researcher"


# ---------------------------------------------------------------------------
# Relationship Types (PRD Section 8)
# ---------------------------------------------------------------------------
class RelType:
    CONTAINS = "CONTAINS"                       # (Project)-[:CONTAINS]->(Experiment)
    USES_CONFIG = "USES_CONFIG"                 # (Experiment)-[:USES_CONFIG]->(Configuration)
    USES_DATASET_VERSION = "USES_DATASET_VERSION" # (Experiment)-[:USES_DATASET_VERSION]->(DatasetVersion)
    VERSION_OF = "VERSION_OF"                   # (DatasetVersion)-[:VERSION_OF]->(Dataset)
    PRODUCES_METRIC = "PRODUCES_METRIC"         # (Experiment)-[:PRODUCES_METRIC]->(Metric)
    TESTS = "TESTS"                             # (Experiment)-[:TESTS]->(Hypothesis)
    BASED_ON = "BASED_ON"                       # (Experiment)-[:BASED_ON]->(Claim)
    SUPPORTED_BY = "SUPPORTED_BY"               # (Claim)-[:SUPPORTED_BY]->(Evidence), (Recommendation)-[:SUPPORTED_BY]->(Experiment)
    CONTRADICTS = "CONTRADICTS"                 # (Claim)-[:CONTRADICTS]->(Claim)
    SUPERSEDES = "SUPERSEDES"                   # (Claim)-[:SUPERSEDES]->(Claim)
    PARENT_OF = "PARENT_OF"                     # (Experiment)-[:PARENT_OF]->(Experiment)
    COMPARED_WITH = "COMPARED_WITH"             # (Experiment)-[:COMPARED_WITH]->(Experiment)
    CHALLENGES = "CHALLENGES"                   # (Correction)-[:CHALLENGES]->(Evidence)
    AFFECTS = "AFFECTS"                         # (Correction)-[:AFFECTS]->(DatasetVersion | Experiment)
    PROPOSES = "PROPOSES"                       # (Recommendation)-[:PROPOSES]->(Experiment)
    CONCERNS = "CONCERNS"                       # (ReviewItem)-[:CONCERNS]->(Experiment)
    REVIEWED_BY = "REVIEWED_BY"                 # (Experiment)-[:REVIEWED_BY]->(Researcher)


# Node properties reference documentation
NODE_SCHEMAS: Dict[str, List[str]] = {
    NodeLabel.PROJECT: ["id", "name", "description", "created_at"],
    NodeLabel.EXPERIMENT: [
        "id", "project_id", "name", "model_family", "status", "created_at",
        "duration_seconds", "compute_cost_gpu_hours", "quality_status",
        "random_seed", "source_reference", "description", "notes"
    ],
    NodeLabel.CONFIGURATION: ["id", "canonical_json", "config_hash", "created_at"],
    NodeLabel.DATASET: ["id", "name", "description"],
    NodeLabel.DATASET_VERSION: ["id", "dataset_id", "version", "quality_status", "created_at"],
    NodeLabel.METRIC: [
        "id", "experiment_id", "name", "value", "direction",
        "evaluation_split", "evaluation_protocol"
    ],
    NodeLabel.HYPOTHESIS: ["id", "text", "status", "created_at"],
    NodeLabel.CLAIM: ["id", "text", "status", "confidence", "created_at", "reviewed_at"],
    NodeLabel.EVIDENCE: ["id", "source_type", "source_reference", "description", "created_at"],
    NodeLabel.CORRECTION: ["id", "description", "status", "created_at", "created_by"],
    NodeLabel.RECOMMENDATION: ["id", "objective", "status", "estimated_gpu_hours", "created_at"],
    NodeLabel.REVIEW_ITEM: ["id", "reason", "status", "created_at"],
    NodeLabel.RESEARCHER: ["id", "name"]
}

# Key indices to maintain for fast lookups and unique reference guarantees
INDICES_TO_CREATE = [
    (NodeLabel.PROJECT, "id"),
    (NodeLabel.EXPERIMENT, "id"),
    (NodeLabel.EXPERIMENT, "project_id"),
    (NodeLabel.CONFIGURATION, "id"),
    (NodeLabel.CONFIGURATION, "config_hash"),
    (NodeLabel.DATASET, "id"),
    (NodeLabel.DATASET_VERSION, "id"),
    (NodeLabel.DATASET_VERSION, "dataset_id"),
    (NodeLabel.METRIC, "id"),
    (NodeLabel.METRIC, "experiment_id"),
    (NodeLabel.HYPOTHESIS, "id"),
    (NodeLabel.CLAIM, "id"),
    (NodeLabel.EVIDENCE, "id"),
    (NodeLabel.CORRECTION, "id"),
    (NodeLabel.RECOMMENDATION, "id"),
    (NodeLabel.REVIEW_ITEM, "id"),
    (NodeLabel.RESEARCHER, "id"),
]


def get_existing_indices(graph) -> Set[str]:
    """Retrieve existing index definitions from FalkorDB."""
    existing = set()
    try:
        res = graph.query("CALL db.indexes()")
        for row in res.result_set:
            if row and len(row) >= 2:
                label = row[0]
                attrs = row[1] if isinstance(row[1], list) else [row[1]]
                for attr in attrs:
                    existing.add(f"{label}.{attr}")
    except Exception as e:
        logger.debug(f"Could not inspect db.indexes(): {e}")
    return existing


def initialize_schema(graph) -> Dict[str, Any]:
    """
    Initializes indexes on the FalkorDB graph idempotently.
    Safe to execute repeatedly.
    """
    existing_indices = get_existing_indices(graph)
    created = []
    skipped = []
    failed = []

    for label, prop in INDICES_TO_CREATE:
        key = f"{label}.{prop}"
        if key in existing_indices:
            skipped.append(key)
            continue

        cypher = f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
        try:
            graph.query(cypher)
            created.append(key)
            logger.info(f"Created index on {key}")
        except Exception as e:
            err_str = str(e).lower()
            if "already indexed" in err_str:
                skipped.append(key)
            else:
                logger.warning(f"Failed to create index on {key}: {e}")
                failed.append({"index": key, "error": str(e)})

    return {
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "status": "ready"
    }
