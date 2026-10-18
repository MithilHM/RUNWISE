"""
FalkorDB Graph Schema Definitions and Initialization for RUNWISE.
Conforms strictly to RUNWISE PRD Section 8.

Defines:
- Node labels and property schemas
- Relationship types and connection semantics
- Idempotent index initialization

FalkorDB version notes (verified against the running container, FalkorDB module
version 42201 / FalkorDB 4.22.x):

* ``CALL db.indexes()`` returns nine columns, one row per index *object*
  (a label can own several index objects, and one index object can cover several
  properties)::

      [0] label        str        e.g. 'Experiment'
      [1] properties   list[str]  e.g. ['id', 'project_id']   <- indexed properties
      [2] types        dict       e.g. {'id': ['RANGE'], ...} <- index algorithm per property
      [3] options      dict
      [4] language     str        'english'
      [5] stopwords    list
      [6] entitytype   str        'NODE' | 'RELATIONSHIP'
      [7] status       str        'OPERATIONAL' | 'POPULATING'
      [8] info         dict

  The parser below is driven by the *result header* rather than by hardcoded
  positions, so a column re-order in a future release degrades gracefully instead
  of silently reading the wrong column.

* Uniqueness constraints are **not** available on this build. Both Cypher forms
  (``CREATE CONSTRAINT FOR (n:L) REQUIRE n.p IS UNIQUE`` and the Neo4j-style
  ``CREATE CONSTRAINT ON (n:L) ASSERT n.p IS UNIQUE``) are rejected with a parse
  error / "use the GRAPH.CONSTRAINT command instead", and that module command
  rejects every arity tried (1-5 arguments). ``initialize_schema`` therefore
  probes once and, when unsupported, records that fact in its result and relies on
  ``MERGE (n:Label {id: $id})`` for duplicate prevention. Duplicate node ids are
  NOT rejected by the database: MERGE discipline in the repository layer is the
  only thing keeping identities unique.
"""

import logging
from typing import Dict, List, Optional, Set, Any

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
    CHAT_SESSION = "ChatSession"
    CHAT_MESSAGE = "ChatMessage"
    AGENT = "Agent"


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
    HAS_RECOMMENDATION = "HAS_RECOMMENDATION"   # (Project)-[:HAS_RECOMMENDATION]->(Recommendation)
    CONCERNS = "CONCERNS"                       # (ReviewItem)-[:CONCERNS]->(Experiment)
    RAISED_BY = "RAISED_BY"                     # (ReviewItem)-[:RAISED_BY]->(Correction)
    REVIEWED_BY = "REVIEWED_BY"                 # (Experiment)-[:REVIEWED_BY]->(Researcher)
    SIMILAR_REGIME = "SIMILAR_REGIME"           # (Dataset)-[:SIMILAR_REGIME]->(Dataset)
    EVALUATED_ON = "EVALUATED_ON"               # (Experiment)-[:EVALUATED_ON]->(Dataset)
    FAILED_UNDER = "FAILED_UNDER"               # (Experiment)-[:FAILED_UNDER]->(FailureReason)
    DERIVED_FROM = "DERIVED_FROM"               # (Experiment)-[:DERIVED_FROM]->(Hypothesis)
    CONTAINS_SESSION = "CONTAINS_SESSION"       # (Project)-[:CONTAINS_SESSION]->(ChatSession)
    HAS_MESSAGE = "HAS_MESSAGE"                 # (ChatSession)-[:HAS_MESSAGE]->(ChatMessage)
    CITES_EXPERIMENT = "CITES_EXPERIMENT"       # (ChatMessage)-[:CITES_EXPERIMENT]->(Experiment)
    HANDED_OFF = "HANDED_OFF"                   # (Agent)-[:HANDED_OFF]->(Agent)


# ---------------------------------------------------------------------------
# Controlled vocabularies
#
# The shared contract enum (app.models.contracts.ReviewItemStatus) is the single
# source of truth and is lowercase. Every write into the graph uses these values;
# nothing else may write an uppercase legacy value such as "NEEDS_REVIEW".
# ---------------------------------------------------------------------------
class ReviewItemStatus:
    """Lowercase review-queue vocabulary, mirroring contracts.ReviewItemStatus."""

    OPEN = "open"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"

    ALL = (OPEN, RESOLVED, DISMISSED)


class QualityStatus:
    """Lowercase experiment/dataset-version trust vocabulary (FR-6)."""

    VALID = "valid"
    NEEDS_REVIEW = "needs_review"
    DISPUTED = "disputed"
    SUPERSEDED = "superseded"
    UNRELIABLE = "unreliable"
    LEAKAGE_DETECTED = "leakage_detected"


# ---------------------------------------------------------------------------
# Node property reference documentation.
#
# This is the documented schema surface (PRD Section 8 plus the fields the
# services, contracts and graph ports actually persist). A property that is
# written by a service must appear here.
# ---------------------------------------------------------------------------
NODE_SCHEMAS: Dict[str, List[str]] = {
    NodeLabel.PROJECT: ["id", "name", "description", "created_at"],
    NodeLabel.EXPERIMENT: [
        "id", "project_id", "name", "model_family", "status", "created_at",
        "duration_seconds", "compute_cost_gpu_hours", "quality_status",
        "random_seed", "source_reference", "description", "notes",
        "dataset_version_id"
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
    NodeLabel.CORRECTION: [
        "id", "description", "status", "created_at", "created_by",
        "project_id", "source_id", "source_type", "evidence_reference",
        "impact_summary"
    ],
    NodeLabel.RECOMMENDATION: [
        "id", "objective", "status", "estimated_gpu_hours", "created_at",
        # Full recommendation payload plus the FR-9 decision recorded on the node.
        "canonical_json", "decided_by", "decision_note", "decided_at"
    ],
    NodeLabel.REVIEW_ITEM: [
        "id", "reason", "status", "created_at",
        # Resolution fields (docs/GRAPH_SCHEMA.md) and correction linkage.
        "resolved_at", "resolved_by", "resolution_notes",
        "project_id", "correction_id", "experiment_id"
    ],
    NodeLabel.RESEARCHER: ["id", "name"]
}

# Relationship documentation block. Mirrors docs/GRAPH_SCHEMA.md section 3.
RELATIONSHIP_DOCS: List[str] = [
    "(Project)-[:CONTAINS]->(Experiment)",
    "(Project)-[:HAS_RECOMMENDATION]->(Recommendation)",
    "(Experiment)-[:USES_CONFIG]->(Configuration)",
    "(Experiment)-[:USES_DATASET_VERSION]->(DatasetVersion)",
    "(DatasetVersion)-[:VERSION_OF]->(Dataset)",
    "(Experiment)-[:PRODUCES_METRIC]->(Metric)",
    "(Experiment)-[:TESTS]->(Hypothesis)",
    "(Experiment)-[:BASED_ON]->(Claim)",
    "(Claim)-[:SUPPORTED_BY]->(Evidence)",
    "(Claim)-[:CONTRADICTS]->(Claim)",
    "(Claim)-[:SUPERSEDES]->(Claim)",
    "(Experiment)-[:PARENT_OF]->(Experiment)",
    "(Experiment)-[:COMPARED_WITH]->(Experiment)",
    "(Correction)-[:CHALLENGES]->(Evidence)",
    "(Correction)-[:AFFECTS]->(DatasetVersion | Experiment)",
    "(Recommendation)-[:SUPPORTED_BY]->(Experiment)",
    "(Recommendation)-[:PROPOSES]->(Experiment)",
    "(ReviewItem)-[:CONCERNS]->(Experiment)",
    "(ReviewItem)-[:RAISED_BY]->(Correction)",
    "(Experiment)-[:REVIEWED_BY]->(Researcher)",
]

# Key indices to maintain for fast lookups and unique reference guarantees.
# Every "id" property is the identity of its label; see the uniqueness-constraint
# note in the module docstring for how duplicate ids are prevented.
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
    (NodeLabel.CHAT_SESSION, "id"),
    (NodeLabel.CHAT_SESSION, "project_id"),
    (NodeLabel.CHAT_MESSAGE, "id"),
    (NodeLabel.CHAT_MESSAGE, "session_id"),
    (NodeLabel.AGENT, "name"),
]

# Identity properties, i.e. the ones a uniqueness constraint would cover.
IDENTITY_INDICES = [(label, prop) for label, prop in INDICES_TO_CREATE if prop == "id"]

# Two syntaxes seen across FalkorDB releases. Probed at runtime, never assumed.
UNIQUE_CONSTRAINT_CYPHER = "CREATE CONSTRAINT FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE"
UNIQUE_CONSTRAINT_LEGACY_CYPHER = "CREATE CONSTRAINT ON (n:{label}) ASSERT n.{prop} IS UNIQUE"

# Cached per process: None = not probed yet, True/False = answer for this build.
_UNIQUE_CONSTRAINTS_SUPPORTED: Optional[bool] = None


def _header_columns(res: Any) -> Dict[str, int]:
    """
    Maps lowercased column name -> column index from a FalkorDB result header.

    The python client exposes the header as a list of ``[type, name]`` pairs, but
    older/newer clients have also handed back plain name strings, so both shapes
    are accepted. An unknown layout yields an empty mapping and the caller falls
    back to positional parsing.
    """
    header = getattr(res, "header", None) or []
    columns: Dict[str, int] = {}
    for idx, entry in enumerate(header):
        name = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 else entry
        if isinstance(name, str) and name:
            columns[name.strip().lower()] = idx
    return columns


def _as_property_list(value: Any) -> List[str]:
    """Coerces an index properties cell into a list of property names."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(k) for k in value.keys()]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if isinstance(v, str)]
    return []


def get_existing_indices(graph) -> Set[str]:
    """
    Retrieve existing index definitions from FalkorDB as ``Label.property`` keys.

    Parses the real ``CALL db.indexes()`` layout (see module docstring): the
    property list lives in the ``properties`` column, located by header name when
    the header is available and positionally otherwise.
    """
    existing: Set[str] = set()
    try:
        res = graph.query("CALL db.indexes()")
        columns = _header_columns(res)
        label_idx = columns.get("label")
        props_idx = columns.get("properties", 1 if not columns else None)
        entity_idx = columns.get("entitytype")

        if label_idx is None or props_idx is None:
            logger.debug(
                "Unrecognised db.indexes() header; falling back to positional parsing."
            )

        for row in res.result_set:
            if not row:
                continue
            try:
                label = row[label_idx if label_idx is not None else 0]
                if not label:
                    continue
                if entity_idx is not None and len(row) > entity_idx:
                    # Relationship indexes are not node property indexes.
                    if str(row[entity_idx]).upper() != "NODE":
                        continue
                properties = _as_property_list(row[props_idx])
                if not properties and len(row) > 2 and isinstance(row[2], (dict, list)):
                    # Layout variants where the properties are carried by a
                    # sibling column: recover rather than report a false miss.
                    properties = _as_property_list(row[2])
                for prop in properties:
                    existing.add(f"{label}.{prop}")
            except (IndexError, TypeError) as exc:
                logger.debug(f"Skipping unparsable db.indexes() row {row!r}: {exc}")
    except Exception as e:
        logger.debug(f"Could not inspect db.indexes(): {e}")
    return existing


def _probe_uniqueness_support(graph) -> bool:
    """
    Probes once per process whether this FalkorDB build accepts uniqueness
    constraints. On the pinned 4.22.x build it does not, and duplicate ids are
    prevented only by MERGE in the repository layer.
    """
    global _UNIQUE_CONSTRAINTS_SUPPORTED
    if _UNIQUE_CONSTRAINTS_SUPPORTED is not None:
        return _UNIQUE_CONSTRAINTS_SUPPORTED

    for template in (UNIQUE_CONSTRAINT_CYPHER, UNIQUE_CONSTRAINT_LEGACY_CYPHER):
        label, prop = IDENTITY_INDICES[0]
        try:
            graph.query(template.format(label=label, prop=prop))
            logger.info("FalkorDB accepts uniqueness constraints; applying them to node identity.")
            _UNIQUE_CONSTRAINTS_SUPPORTED = True
            return True
        except Exception:
            continue

    logger.info(
        "FalkorDB build does not support CREATE CONSTRAINT ... IS UNIQUE; "
        "duplicate prevention rests on MERGE on the id property."
    )
    _UNIQUE_CONSTRAINTS_SUPPORTED = False
    return False


def _apply_uniqueness_constraints(graph, created: List[str], failed: List[Dict[str, Any]]) -> Optional[bool]:
    """
    Applies uniqueness constraints to node identity properties when supported.

    Returns True when constraints are in place, False when the build rejects
    them, and None when the probe itself failed for an unrelated reason.
    A rejection is not a schema failure: MERGE keeps identities unique, so the
    caller's ``failed`` list is left untouched on a plain "unsupported" answer.
    """
    try:
        supported = _probe_uniqueness_support(graph)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not probe uniqueness constraint support: {e}")
        return None

    if not supported:
        return False

    for label, prop in IDENTITY_INDICES:
        key = f"UNIQUE({label}.{prop})"
        for template in (UNIQUE_CONSTRAINT_CYPHER, UNIQUE_CONSTRAINT_LEGACY_CYPHER):
            try:
                graph.query(template.format(label=label, prop=prop))
                created.append(key)
                break
            except Exception as e:
                err = str(e).lower()
                if "already" in err or "exist" in err:
                    created.append(key)
                    break
        else:
            failed.append({"constraint": key, "error": "uniqueness constraint rejected"})
    return True


def initialize_schema(graph) -> Dict[str, Any]:
    """
    Initializes indexes on the FalkorDB graph idempotently.
    Safe to execute repeatedly: a second run reports every index as ``skipped``
    and creates nothing.
    """
    existing_indices = get_existing_indices(graph)
    created: List[str] = []
    skipped: List[str] = []
    failed: List[Dict[str, Any]] = []

    for label, prop in INDICES_TO_CREATE:
        key = f"{label}.{prop}"
        if key in existing_indices:
            skipped.append(key)
            continue

        cypher = f"CREATE INDEX FOR (n:{label}) ON (n.{prop})"
        try:
            graph.query(cypher)
            created.append(key)
            existing_indices.add(key)
            logger.info(f"Created index on {key}")
        except Exception as e:
            err_str = str(e).lower()
            if "already indexed" in err_str:
                # Lost a race with a concurrent initializer, or the index exists
                # under a layout the parser could not read.
                skipped.append(key)
            else:
                logger.warning(f"Failed to create index on {key}: {e}")
                failed.append({"index": key, "error": str(e)})

    unique_supported = _apply_uniqueness_constraints(graph, created, failed)

    if failed:
        for item in failed:
            logger.error(
                "Schema initialization failure: {item}".format(
                    item=item.get("index") or item.get("constraint")
                ),
                extra={"detail": item.get("error")},
            )
        status = "degraded"
    else:
        status = "ready"

    return {
        "created": created,
        "skipped": skipped,
        "failed": failed,
        "status": status,
        "unique_constraints": {
            "supported": bool(unique_supported),
            "detail": (
                "uniqueness constraints applied to node identity properties"
                if unique_supported
                else "FalkorDB build rejects CREATE CONSTRAINT ... IS UNIQUE; "
                     "duplicate prevention rests on MERGE on the id property"
            ),
        },
    }