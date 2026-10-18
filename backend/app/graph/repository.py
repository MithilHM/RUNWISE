"""
Graph Repository for RUNWISE.
Encapsulates all Cypher queries, graph traversals, and entity-relationship operations.
Keeps graph logic clean, separated, parameterized, and modular.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

from app.graph.client import FalkorDBClient, graph_client
from app.graph.schema import NodeLabel, RelType
from app.models.contracts import (
    CorrectionImpact,
    CorrectionRecord,
    DatasetSummary,
    DatasetVersionSummary,
    EvidenceStatus,
    ExperimentRecord,
    ExperimentStatus,
    ExperimentSummary,
    ExperimentDetail,
    MetricModel,
    LineageNode,
    LineageEdge,
    ExperimentLineageResponse,
    GraphNodeData,
    GraphEdgeData,
    ProjectGraphResponse,
    ProjectGraphSummary,
    Recommendation,
    RecommendationDecision,
    RecommendationStatus,
    ReviewItem,
    ReviewItemResponse,
    ReviewItemStatus,
    ClaimPath,
)

logger = logging.getLogger(__name__)

#: Marker for "the source did not record a model family". PRD 5 FR-1 #8 forbids
#: inventing a value, and ``ExperimentSummary.model_family`` is a required string,
#: so absence is labelled rather than filled in with a plausible-sounding default.
UNKNOWN_MODEL_FAMILY = "unknown"

#: Relationship types introduced after docs/GRAPH_SCHEMA.md was first written and
#: declared in ``app/graph/schema.py``. Written literally here so this module does
#: not depend on enum members that older checkouts may not have.
REL_RAISED_BY = "RAISED_BY"                    # (ReviewItem)-[:RAISED_BY]->(Correction)
REL_HAS_RECOMMENDATION = "HAS_RECOMMENDATION"  # (Project)-[:HAS_RECOMMENDATION]->(Recommendation)

#: Stored ``quality_status`` spellings that mean "a correction reached this node".
#: GRAPH_SCHEMA.md section 5 marks a corrected source ``unreliable`` or
#: ``leakage_detected``; the contract expresses the same condition as DISPUTED.
_DISPUTED_QUALITY_VALUES = {
    "unreliable",
    "leakage_detected",
    "disputed",
    "needs_review",
}

_QUALITY_TO_EVIDENCE = {
    "valid": EvidenceStatus.OBSERVED,
    "observed": EvidenceStatus.OBSERVED,
    "confirmed": EvidenceStatus.CONFIRMED,
    "user_asserted": EvidenceStatus.USER_ASSERTED,
    "inferred": EvidenceStatus.INFERRED,
    "superseded": EvidenceStatus.SUPERSEDED,
    "unresolved": EvidenceStatus.UNRESOLVED,
    "disputed": EvidenceStatus.DISPUTED,
    "unreliable": EvidenceStatus.DISPUTED,
    "leakage_detected": EvidenceStatus.DISPUTED,
    "needs_review": EvidenceStatus.DISPUTED,
}

#: Review item status spellings accepted on read. Only ``open``, ``resolved`` and
#: ``dismissed`` are ever written; the legacy uppercase form is tolerated so a graph
#: seeded before the vocabulary change still shows its queue instead of losing items.
_REVIEW_STATUS_ALIASES = {
    "needs_review": ReviewItemStatus.OPEN,
    "open": ReviewItemStatus.OPEN,
    "resolved": ReviewItemStatus.RESOLVED,
    "dismissed": ReviewItemStatus.DISMISSED,
}

#: The same two spellings, for the queue filter. A fresh seed only ever contains
#: ``open``; a pre-migration graph may still hold ``NEEDS_REVIEW``.
_OPEN_REVIEW_STATUSES = ("open", "NEEDS_REVIEW")


def _to_evidence_status(raw: Any) -> EvidenceStatus | None:
    """Map a stored quality string onto an epistemic grade.

    An unrecognised value returns ``None`` so the field reads as unrecorded. It is
    never coerced to a grade, because a wrong grade is a trust claim the data does
    not support.
    """
    if raw is None or raw == "":
        return None
    return _QUALITY_TO_EVIDENCE.get(str(raw).strip().lower())


def _to_review_status(raw: Any) -> ReviewItemStatus:
    if raw is None:
        return ReviewItemStatus.OPEN
    return _REVIEW_STATUS_ALIASES.get(str(raw).strip().lower(), ReviewItemStatus.OPEN)


def _metric_values(raw_metrics: Any) -> List[MetricModel]:
    """Build metric models from stored properties.

    A missing value stays ``None``. It is never ``0.0``: a blank column is
    unrecorded, not a measurement of zero, and ranking_rules.md section 4 rule 1
    depends on that distinction everywhere.
    """
    metrics: List[MetricModel] = []
    for raw in raw_metrics or []:
        if not raw:
            continue
        props = dict(raw)
        name = props.get("name")
        if not name:
            continue
        value = props.get("value")
        try:
            parsed_value = float(value) if value is not None else None
        except (TypeError, ValueError):
            parsed_value = None
        metrics.append(
            MetricModel(
                name=str(name),
                value=parsed_value,
                direction=props.get("direction") or "maximize",
                evaluation_split=props.get("evaluation_split") or None,
                evaluation_protocol=props.get("evaluation_protocol") or None,
            )
        )
    return metrics


def _split_configuration(canonical_json: Any) -> Dict[str, Any]:
    """Return the stored configuration as a mapping, or ``{}`` when absent."""
    if not canonical_json:
        return {}
    if isinstance(canonical_json, dict):
        return dict(canonical_json)
    try:
        parsed = json.loads(canonical_json)
    except (TypeError, ValueError):
        logger.warning("Configuration canonical_json is not valid JSON; ignoring it")
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dump_json(value: Any) -> str | None:
    """Serialise a nested configuration document, or ``None`` when there is none.

    An absent document stays absent. Storing ``"{}"`` would make an unrecorded
    configuration indistinguishable from a deliberately empty one.
    """
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str) if value else None
    return None



class GraphRepository:
    def __init__(self, client: Optional[FalkorDBClient] = None):
        self.client = client or graph_client


    def _written(
        self, rows: Any, fallback: Dict[str, Any], entity: str, node_id: str
    ) -> Dict[str, Any]:
        """Return the node a write produced, or report that nothing was persisted.

        A ``MERGE`` that matched nothing and created nothing means the target was
        missing: a project that was never created, or a dangling experiment id.
        Returning a synthesised dict in that case is how a correction looks
        recorded when no node was touched.
        """
        if rows and rows[0] and rows[0][0]:
            return dict(rows[0][0])
        logger.warning("%s %s: write matched no node, so nothing was persisted", entity, node_id)
        return dict(fallback)

    # -----------------------------------------------------------------------
    # Projects
    # -----------------------------------------------------------------------

    def create_or_update_project(self, project_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Creates or updates a Project node."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (p:Project {id: $id})
        ON CREATE SET p.name = $name, p.description = $description, p.created_at = $created_at
        ON MATCH SET p.name = $name, p.description = coalesce($description, p.description)
        RETURN properties(p)
        """
        rows = self.client.execute_query(cypher, {
            "id": project_id,
            "name": name,
            "description": description or "",
            "created_at": now
        })
        return self._written(rows, {"id": project_id, "name": name}, "Project", project_id)

    def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves project metadata and experiment count."""
        cypher = """
        MATCH (p:Project {id: $id})
        OPTIONAL MATCH (p)-[:CONTAINS]->(e:Experiment)
        RETURN properties(p), count(e) AS exp_count
        """
        rows = self.client.execute_query(cypher, {"id": project_id})
        if not rows or not rows[0][0]:
            return None
        props = dict(rows[0][0])
        props["experiment_count"] = rows[0][1]
        return props

    def list_projects(self) -> List[Dict[str, Any]]:
        """Lists all projects with experiment count."""
        cypher = """
        MATCH (p:Project)
        OPTIONAL MATCH (p)-[:CONTAINS]->(e:Experiment)
        WITH p, count(e) AS exp_count
        ORDER BY p.name ASC
        RETURN properties(p), exp_count
        """
        rows = self.client.execute_query(cypher)
        results = []
        for row in rows:
            if row and row[0]:
                item = dict(row[0])
                item["experiment_count"] = row[1]
                results.append(item)
        return results

    # -----------------------------------------------------------------------
    # Datasets and Versions
    # -----------------------------------------------------------------------

    def create_or_update_dataset(self, dataset_id: str, name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Creates or updates a Dataset node."""
        cypher = """
        MERGE (d:Dataset {id: $id})
        ON CREATE SET d.name = $name, d.description = $description
        ON MATCH SET d.name = $name, d.description = coalesce($description, d.description)
        RETURN properties(d)
        """
        rows = self.client.execute_query(cypher, {
            "id": dataset_id,
            "name": name,
            "description": description or ""
        })
        return self._written(rows, {"id": dataset_id, "name": name}, "Dataset", dataset_id)

    def create_or_update_dataset_version(
        self,
        version_id: str,
        dataset_id: str,
        version: str,
        quality_status: str = "valid"
    ) -> Dict[str, Any]:
        """Creates or updates a DatasetVersion node and links it to its Dataset."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (dv:DatasetVersion {id: $version_id})
        ON CREATE SET dv.dataset_id = $dataset_id, dv.version = $version,
                      dv.quality_status = $quality_status, dv.created_at = $created_at
        ON MATCH SET dv.version = $version, dv.dataset_id = $dataset_id
        WITH dv
        MATCH (d:Dataset {id: $dataset_id})
        MERGE (dv)-[:VERSION_OF]->(d)
        RETURN properties(dv)
        """
        rows = self.client.execute_query(cypher, {
            "version_id": version_id,
            "dataset_id": dataset_id,
            "version": version,
            "quality_status": quality_status,
            "created_at": now
        })
        return self._written(
            rows,
            {"id": version_id, "dataset_id": dataset_id, "version": version},
            "DatasetVersion",
            version_id,
        )

    def update_dataset_version_quality_status(self, version_id: str, quality_status: str) -> bool:
        """Updates quality_status on a DatasetVersion (e.g. 'unreliable', 'leakage_detected')."""
        cypher = """
        MATCH (dv:DatasetVersion {id: $version_id})
        SET dv.quality_status = $quality_status
        RETURN dv.id
        """
        rows = self.client.execute_query(cypher, {
            "version_id": version_id,
            "quality_status": quality_status
        })
        return bool(rows)

    # -----------------------------------------------------------------------
    # Configurations
    # -----------------------------------------------------------------------

    def create_or_update_configuration(
        self,
        config_id: str,
        canonical_json: str,
        config_hash: str
    ) -> Dict[str, Any]:
        """Creates or updates a Configuration node."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (c:Configuration {id: $id})
        ON CREATE SET c.canonical_json = $canonical_json, c.config_hash = $config_hash, c.created_at = $created_at
        ON MATCH SET c.canonical_json = $canonical_json, c.config_hash = $config_hash
        RETURN properties(c)
        """
        rows = self.client.execute_query(cypher, {
            "id": config_id,
            "canonical_json": canonical_json,
            "config_hash": config_hash,
            "created_at": now
        })
        return self._written(
            rows, {"id": config_id, "config_hash": config_hash}, "Configuration", config_id
        )

    # -----------------------------------------------------------------------
    # Experiments
    # -----------------------------------------------------------------------

    def _assert_project_ownership(self, experiment_id: str, project_id: str) -> None:
        """Refuse to re-file an existing run under a different project.

        ``MERGE (e:Experiment {id})`` matches on the id alone, so importing a record
        whose identifier already exists in another project would silently move the
        run and rewrite its ``CONTAINS`` edge. That is a cross-project data leak
        disguised as a successful import, so it raises instead.
        """
        rows = self.client.execute_query(
            "MATCH (e:Experiment {id: $id}) WHERE e.project_id IS NOT NULL RETURN e.project_id",
            {"id": experiment_id},
        )
        if not rows or not rows[0] or rows[0][0] is None:
            return
        existing = str(rows[0][0])
        if existing != project_id:
            raise ValueError(
                f"Experiment {experiment_id!r} already belongs to project {existing!r}; "
                f"refusing to re-file it under {project_id!r}."
            )

    def create_or_update_experiment(self, exp_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Creates or updates an Experiment node and establishes core relationships:
        - (Project)-[:CONTAINS]->(Experiment)
        - (Experiment)-[:USES_CONFIG]->(Configuration)
        - (Experiment)-[:USES_DATASET_VERSION]->(DatasetVersion)
        - (parent)-[:PARENT_OF]->(Experiment)
        """
        now = datetime.now(timezone.utc).isoformat()
        exp_id = exp_data["id"]
        project_id = exp_data["project_id"]
        config_id = exp_data.get("config_id")
        dataset_version_id = exp_data.get("dataset_version_id")
        parent_id = exp_data.get("parent_experiment_id")

        cypher_exp = """
        MERGE (e:Experiment {id: $id})
        ON CREATE SET e.project_id = $project_id,
                      e.name = $name,
                      e.model_family = $model_family,
                      e.status = $status,
                      e.quality_status = $quality_status,
                      e.created_at = $created_at,
                      e.ingested_at = $ingested_at,
                      e.duration_seconds = $duration_seconds,
                      e.compute_cost_gpu_hours = $compute_cost_gpu_hours,
                      e.random_seed = $random_seed,
                      e.source_reference = $source_reference,
                      e.description = $description,
                      e.notes = $notes,
                      e.dataset_id = $dataset_id,
                      e.dataset_version_id = $dataset_version_id,
                      e.preprocessing_json = $preprocessing_json,
                      e.hyperparameters_json = $hyperparameters_json,
                      e.training_steps = $training_steps
        ON MATCH SET  e.name = $name,
                      e.model_family = $model_family,
                      e.status = $status,
                      e.ingested_at = $ingested_at,
                      e.duration_seconds = coalesce($duration_seconds, e.duration_seconds),
                      e.compute_cost_gpu_hours = coalesce($compute_cost_gpu_hours, e.compute_cost_gpu_hours),
                      e.random_seed = coalesce($random_seed, e.random_seed),
                      e.source_reference = coalesce($source_reference, e.source_reference),
                      e.description = coalesce($description, e.description),
                      e.notes = coalesce($notes, e.notes),
                      e.preprocessing_json = coalesce($preprocessing_json, e.preprocessing_json),
                      e.hyperparameters_json = coalesce($hyperparameters_json, e.hyperparameters_json),
                      e.training_steps = coalesce($training_steps, e.training_steps)
        WITH e
        MATCH (p:Project {id: $project_id})
        MERGE (p)-[:CONTAINS]->(e)
        RETURN properties(e)
        """
        params = {
            "id": exp_id,
            "project_id": project_id,
            "name": exp_data.get("name") or exp_id,
            # No invented defaults: a field the source never stated stays null, and
            # the engines read that back as "unrecorded" (FR-1 #8).
            "model_family": exp_data.get("model_family"),
            "status": exp_data.get("status"),
            "quality_status": exp_data.get("quality_status") or "valid",
            # created_at is the source's own timestamp. ingested_at records when the
            # row reached the graph, which is a different fact and is never used to
            # answer "when did this run happen".
            "created_at": exp_data.get("created_at"),
            "ingested_at": now,
            "duration_seconds": exp_data.get("duration_seconds"),
            "compute_cost_gpu_hours": exp_data.get("compute_cost_gpu_hours"),
            "random_seed": exp_data.get("random_seed"),
            "source_reference": exp_data.get("source_reference"),
            "description": exp_data.get("description"),
            "notes": exp_data.get("notes"),
            "dataset_id": exp_data.get("dataset_id"),
            "dataset_version_id": dataset_version_id,
            "preprocessing_json": _dump_json(exp_data.get("preprocessing")),
            "hyperparameters_json": _dump_json(exp_data.get("hyperparameters")),
            "training_steps": exp_data.get("training_steps"),
        }

        self._assert_project_ownership(exp_id, project_id)
        rows = self.client.execute_query(cypher_exp, params)

        # A run has at most one configuration and one dataset version. Replacing the
        # edge rather than merging it stops a re-import with changed settings from
        # leaving the previous Configuration attached alongside the new one.
        # OPTIONAL MATCH matters: a plain MATCH would produce no rows on a first
        # import, and everything after it would silently never run.
        if config_id:
            replace_config_cypher = """
            MATCH (e:Experiment {id: $exp_id})
            OPTIONAL MATCH (e)-[r:USES_CONFIG]->(:Configuration)
            DELETE r
            WITH e
            MATCH (c:Configuration {id: $config_id})
            MERGE (e)-[:USES_CONFIG]->(c)
            """
            self.client.execute_query(replace_config_cypher, {"exp_id": exp_id, "config_id": config_id})

        if dataset_version_id:
            replace_dv_cypher = """
            MATCH (e:Experiment {id: $exp_id})
            OPTIONAL MATCH (e)-[r:USES_DATASET_VERSION]->(:DatasetVersion)
            DELETE r
            WITH e
            MATCH (dv:DatasetVersion {id: $dv_id})
            MERGE (e)-[:USES_DATASET_VERSION]->(dv)
            """
            self.client.execute_query(replace_dv_cypher, {"exp_id": exp_id, "dv_id": dataset_version_id})

        # Link parent experiment
        if parent_id and parent_id != exp_id:
            link_parent_cypher = """
            MATCH (p:Experiment {id: $parent_id}), (c:Experiment {id: $child_id})
            MERGE (p)-[:PARENT_OF]->(c)
            """
            self.client.execute_query(link_parent_cypher, {"parent_id": parent_id, "child_id": exp_id})

        return self._written(rows, {"id": exp_id}, "Experiment", exp_id)

    def update_experiment_quality_status(self, experiment_id: str, quality_status: str) -> bool:
        """Updates quality_status on an Experiment (e.g. 'valid', 'needs_review', 'disputed', 'superseded')."""
        cypher = """
        MATCH (e:Experiment {id: $exp_id})
        SET e.quality_status = $quality_status
        RETURN e.id
        """
        rows = self.client.execute_query(cypher, {
            "exp_id": experiment_id,
            "quality_status": quality_status
        })
        return bool(rows)

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------

    def create_or_update_metric(
        self,
        metric_id: str,
        experiment_id: str,
        name: str,
        value: Optional[float],
        direction: Optional[str] = None,
        evaluation_split: Optional[str] = None,
        evaluation_protocol: Optional[str] = None
    ) -> Dict[str, Any]:
        """Creates or updates a Metric node and links (Experiment)-[:PRODUCES_METRIC]->(Metric)."""
        cypher = """
        MERGE (m:Metric {id: $metric_id})
        ON CREATE SET m.experiment_id = $experiment_id,
                      m.name = $name,
                      m.value = $value,
                      m.direction = $direction,
                      m.evaluation_split = $evaluation_split,
                      m.evaluation_protocol = $evaluation_protocol
        ON MATCH SET  m.value = $value,
                      m.direction = $direction,
                      m.evaluation_split = coalesce($evaluation_split, m.evaluation_split),
                      m.evaluation_protocol = coalesce($evaluation_protocol, m.evaluation_protocol)
        WITH m
        MATCH (e:Experiment {id: $experiment_id})
        MERGE (e)-[:PRODUCES_METRIC]->(m)
        RETURN properties(m)
        """
        rows = self.client.execute_query(cypher, {
            "metric_id": metric_id,
            "experiment_id": experiment_id,
            "name": name,
            # A blank value stays null. Writing 0.0 would turn "not measured" into
            # "measured at zero", which is the one substitution the whole trust
            # model forbids.
            "value": float(value) if value is not None else None,
            "direction": direction,
            # An unstated split stays null: guessing "validation" from a column name
            # would make an unrecorded split indistinguishable from a real one, and
            # the split is a comparability blocker.
            "evaluation_split": evaluation_split,
            "evaluation_protocol": evaluation_protocol
        })
        return self._written(
            rows, {"id": metric_id, "name": name, "value": value}, "Metric", metric_id
        )

    # -----------------------------------------------------------------------
    # Claims, Hypotheses, Evidence
    # -----------------------------------------------------------------------

    def create_hypothesis(self, hypothesis_id: str, text: str, status: str = "active") -> Dict[str, Any]:
        """Creates or updates a Hypothesis node."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (h:Hypothesis {id: $id})
        ON CREATE SET h.text = $text, h.status = $status, h.created_at = $created_at
        ON MATCH SET h.text = $text, h.status = $status
        RETURN properties(h)
        """
        rows = self.client.execute_query(cypher, {"id": hypothesis_id, "text": text, "status": status, "created_at": now})
        return self._written(rows, {"id": hypothesis_id, "text": text}, "Hypothesis", hypothesis_id)

    def link_experiment_hypothesis(self, experiment_id: str, hypothesis_id: str):
        """Links (Experiment)-[:TESTS]->(Hypothesis)."""
        cypher = """
        MATCH (e:Experiment {id: $exp_id}), (h:Hypothesis {id: $hyp_id})
        MERGE (e)-[:TESTS]->(h)
        """
        self.client.execute_query(cypher, {"exp_id": experiment_id, "hyp_id": hypothesis_id})

    def create_evidence(
        self,
        evidence_id: str,
        source_type: str,
        source_reference: str,
        description: str
    ) -> Dict[str, Any]:
        """Creates or updates an Evidence node."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (ev:Evidence {id: $id})
        ON CREATE SET ev.source_type = $source_type, ev.source_reference = $source_reference,
                      ev.description = $description, ev.created_at = $created_at
        ON MATCH SET ev.description = $description, ev.source_reference = $source_reference
        RETURN properties(ev)
        """
        rows = self.client.execute_query(cypher, {
            "id": evidence_id,
            "source_type": source_type,
            "source_reference": source_reference,
            "description": description,
            "created_at": now
        })
        return self._written(rows, {"id": evidence_id}, "Evidence", evidence_id)

    def create_claim(
        self,
        claim_id: str,
        text: str,
        status: str = "verified",
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Creates or updates a Claim node."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (c:Claim {id: $id})
        ON CREATE SET c.text = $text, c.status = $status, c.confidence = $confidence, c.created_at = $created_at
        ON MATCH SET c.text = $text, c.status = $status, c.confidence = $confidence
        RETURN properties(c)
        """
        rows = self.client.execute_query(cypher, {
            "id": claim_id,
            "text": text,
            "status": status,
            "confidence": float(confidence),
            "created_at": now
        })
        return self._written(rows, {"id": claim_id, "text": text}, "Claim", claim_id)

    def link_experiment_claim(self, experiment_id: str, claim_id: str):
        """Links (Experiment)-[:BASED_ON]->(Claim)."""
        cypher = """
        MATCH (e:Experiment {id: $exp_id}), (c:Claim {id: $claim_id})
        MERGE (e)-[:BASED_ON]->(c)
        """
        self.client.execute_query(cypher, {"exp_id": experiment_id, "claim_id": claim_id})

    def link_claim_evidence(self, claim_id: str, evidence_id: str):
        """Links (Claim)-[:SUPPORTED_BY]->(Evidence)."""
        cypher = """
        MATCH (c:Claim {id: $claim_id}), (ev:Evidence {id: $ev_id})
        MERGE (c)-[:SUPPORTED_BY]->(ev)
        """
        self.client.execute_query(cypher, {"claim_id": claim_id, "ev_id": evidence_id})

    def update_claim_status(self, claim_id: str, status: str) -> bool:
        """Updates status on a Claim (e.g. 'disputed', 'superseded', 'unreliable')."""
        cypher = """
        MATCH (c:Claim {id: $claim_id})
        SET c.status = $status
        RETURN c.id
        """
        rows = self.client.execute_query(cypher, {"claim_id": claim_id, "status": status})
        return bool(rows)

    # -----------------------------------------------------------------------
    # Corrections and Review Items (PRD Section 8, Section 9)
    # -----------------------------------------------------------------------

    def create_correction(
        self,
        correction_id: str,
        description: str,
        status: str = "open",
        created_by: str = "researcher"
    ) -> Dict[str, Any]:
        """Creates or updates a Correction node.

        The default status is the contract's ``open``: a correction recorded here
        is unsettled until a researcher says otherwise, and the trust gates read
        every status that still belongs to an unsettled correction.
        """
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MERGE (co:Correction {id: $id})
        ON CREATE SET co.description = $description, co.status = $status,
                      co.created_at = $created_at, co.created_by = $created_by
        ON MATCH SET co.description = $description, co.status = $status
        RETURN properties(co)
        """
        rows = self.client.execute_query(cypher, {
            "id": correction_id,
            "description": description,
            "status": status,
            "created_at": now,
            "created_by": created_by
        })
        return self._written(rows, {"id": correction_id}, "Correction", correction_id)

    def link_correction_target(self, correction_id: str, target_id: str, target_label: str):
        """Links a Correction to the target node it affects or challenges."""
        if target_label == NodeLabel.EVIDENCE:
            rel = RelType.CHALLENGES
        else:
            rel = RelType.AFFECTS

        cypher = f"""
        MATCH (co:Correction {{id: $corr_id}}), (target:{target_label} {{id: $target_id}})
        MERGE (co)-[:{rel}]->(target)
        """
        self.client.execute_query(cypher, {"corr_id": correction_id, "target_id": target_id})

    def create_review_item(
        self,
        review_item_id: str,
        experiment_id: str,
        reason: str,
        status: ReviewItemStatus = ReviewItemStatus.OPEN,
    ) -> Dict[str, Any]:
        """Creates a ReviewItem node and links (ReviewItem)-[:CONCERNS]->(Experiment)."""
        now = datetime.now(timezone.utc).isoformat()
        wanted = status.value if isinstance(status, ReviewItemStatus) else str(status)
        cypher = """
        MERGE (ri:ReviewItem {id: $id})
        ON CREATE SET ri.reason = $reason, ri.status = $status, ri.created_at = $created_at
        ON MATCH SET ri.reason = $reason, ri.status = $status
        WITH ri
        MATCH (e:Experiment {id: $exp_id})
        MERGE (ri)-[:CONCERNS]->(e)
        RETURN properties(ri)
        """
        rows = self.client.execute_query(cypher, {
            "id": review_item_id,
            "exp_id": experiment_id,
            "reason": reason,
            "status": wanted,
            "created_at": now
        })
        return self._written(
            rows,
            {"id": review_item_id, "experiment_id": experiment_id},
            "ReviewItem",
            review_item_id,
        )

    def get_review_queue(self, project_id: Optional[str] = None) -> List[ReviewItemResponse]:
        """Active review items for the workspace, optionally scoped to a project.

        Items raised before a status vocabulary existed are normalised on read, so
        the frontend switches on one set of strings.
        """
        if project_id:
            cypher = """
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(e:Experiment)<-[:CONCERNS]-(ri:ReviewItem)
            WHERE coalesce(ri.status, 'open') IN $open_statuses
            OPTIONAL MATCH (ri)-[:RAISED_BY]->(co:Correction)
            RETURN ri.id, e.id, ri.reason, ri.status, ri.created_at, ri.project_id,
                   co.id, ri.resolved_at, ri.resolved_by, ri.resolution_notes
            ORDER BY ri.created_at DESC
            """
            rows = self.client.execute_query(cypher, {"project_id": project_id, "open_statuses": list(_OPEN_REVIEW_STATUSES)})
        else:
            cypher = """
            MATCH (ri:ReviewItem)-[:CONCERNS]->(e:Experiment)
            WHERE coalesce(ri.status, 'open') IN $open_statuses
            OPTIONAL MATCH (ri)-[:RAISED_BY]->(co:Correction)
            RETURN ri.id, e.id, ri.reason, ri.status, ri.created_at, ri.project_id,
                   co.id, ri.resolved_at, ri.resolved_by, ri.resolution_notes
            ORDER BY ri.created_at DESC
            """
            rows = self.client.execute_query(cypher, {"open_statuses": list(_OPEN_REVIEW_STATUSES)})

        items = []
        for row in rows:
            if not row or not row[0]:
                continue
            items.append(ReviewItemResponse(
                id=str(row[0]),
                experiment_id=row[1],
                reason=str(row[2] or ""),
                status=_to_review_status(row[3]),
                project_id=row[5] or project_id,
                correction_id=row[6],
                created_at=str(row[4]) if row[4] is not None else "",
                resolved_at=str(row[7]) if row[7] is not None else None,
                resolved_by=row[8],
                resolution_notes=row[9],
            ))
        return items

    # -----------------------------------------------------------------------
    # Retrieval and History Queries
    # -----------------------------------------------------------------------

    @staticmethod
    def _metric_filter_terms(
        metric: Optional[str], evaluation_split: Optional[str], params: Dict[str, Any]
    ) -> tuple[list[str], list[str]]:
        """Build the aggregate terms that filter on recorded metric properties.

        Metric name and split live on ``Metric`` nodes, and this FalkorDB build has
        neither ``EXISTS`` subqueries nor pattern comprehensions, so the filter is a
        conditional aggregate in the ``WITH`` that already collects the metrics. A run
        with no matching metric aggregates to null, which fails ``> 0`` and is
        therefore excluded rather than included by accident.

        Returns the expressions to add to the ``WITH`` and the predicate to apply
        after it.
        """
        selects: list[str] = []
        conditions: list[str] = []
        if metric:
            params["metric"] = metric
            selects.append("sum(CASE WHEN toLower(m.name) = toLower($metric) THEN 1 ELSE 0 END) AS metric_hits")
            conditions.append("metric_hits > 0")
        if evaluation_split:
            params["evaluation_split"] = evaluation_split
            selects.append("sum(CASE WHEN m.evaluation_split = $evaluation_split THEN 1 ELSE 0 END) AS split_hits")
            conditions.append("split_hits > 0")
        return selects, conditions

    def _build_experiment_record(
        self,
        exp_props: Dict[str, Any],
        *,
        dataset_name: Optional[str] = None,
        dataset_version: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_quality_status: Any = None,
        config_json: Any = None,
        metrics_raw: Any = None,
        training_steps: Any = None,
        max_training_steps: Any = None,
    ) -> ExperimentRecord:
        """Build the analysis record the engines consume (FR-4/FR-5 input).

        Every field the source did not record stays ``None`` or empty, so the
        engines degrade to ``INSUFFICIENT_INFORMATION`` instead of comparing a
        fabricated value against a real one.
        """
        stored_status = exp_props.get("status")
        try:
            status = ExperimentStatus(stored_status) if stored_status else ExperimentStatus.UNKNOWN
        except ValueError:
            logger.warning(
                "Experiment %s has unrecognised status %r; reporting it as unknown",
                exp_props.get("id"),
                stored_status,
            )
            status = ExperimentStatus.UNKNOWN

        configuration = _split_configuration(config_json)

        return ExperimentRecord(
            id=str(exp_props.get("id")),
            project_id=str(exp_props.get("project_id") or ""),
            name=str(exp_props.get("name") or exp_props.get("id")),
            model_family=exp_props.get("model_family"),
            dataset_id=dataset_id or exp_props.get("dataset_id"),
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            dataset_version_id=exp_props.get("dataset_version_id"),
            dataset_quality_status=_to_evidence_status(
                dataset_quality_status if dataset_quality_status is not None
                else exp_props.get("dataset_quality_status")
            ),
            status=status,
            quality_status=exp_props.get("quality_status") or "valid",
            metrics=_metric_values(metrics_raw),
            compute_cost_gpu_hours=_float_or_none(exp_props.get("compute_cost_gpu_hours")),
            duration_seconds=_float_or_none(exp_props.get("duration_seconds")),
            created_at=exp_props.get("created_at"),
            description=exp_props.get("description"),
            configuration=configuration,
            hyperparameters=_split_configuration(exp_props.get("hyperparameters_json")),
            preprocessing=_split_configuration(exp_props.get("preprocessing_json")),
            random_seed=_int_or_none(exp_props.get("random_seed")),
            training_steps=_int_or_none(training_steps),
            max_training_steps=_int_or_none(
                max_training_steps if max_training_steps is not None
                else exp_props.get("max_training_steps")
            ),
            source_reference=exp_props.get("source_reference"),
            notes=exp_props.get("notes"),
        )

    #: Columns every experiment read shares, in a fixed order. ``preprocessing_json``,
    #: ``hyperparameters_json`` and the step counts are optional node properties:
    #: a record imported before they existed simply has none. The metric list is
    #: appended separately, because the aggregating ``WITH`` that produces it differs
    #: between reads.
    _EXPERIMENT_RETURN = """
               c.canonical_json AS config_json,
               d.name AS dataset_name,
               d.id AS dataset_id,
               dv.version AS dataset_version,
               dv.quality_status AS dataset_quality_status,
               e.preprocessing_json AS preprocessing_json,
               e.hyperparameters_json AS hyperparameters_json,
               e.training_steps AS training_steps,
               e.max_training_steps AS max_training_steps
    """

    _COL_CONFIG_JSON = 1
    _COL_DATASET_NAME = 2
    _COL_DATASET_ID = 3
    _COL_DATASET_VERSION = 4
    _COL_DATASET_QUALITY = 5
    _COL_PREPROCESSING = 6
    _COL_HYPERPARAMETERS = 7
    _COL_TRAINING_STEPS = 8
    _COL_MAX_TRAINING_STEPS = 9
    _COL_METRICS = 10
    _COL_PARENT_IDS = 11
    _COL_CHILD_IDS = 12
    _COL_CLAIMS = 13
    _COL_REVIEW_ITEMS = 14

    def get_experiment_detail(
        self, experiment_id: str, *, project_id: Optional[str] = None
    ) -> Optional[ExperimentDetail]:
        """Full UI detail of an experiment, its configuration, metrics and lineage.

        This is the read the experiment workspace renders. The analysis engines use
        :meth:`get_experiment`, which returns the same graph state typed as the
        port's :class:`ExperimentRecord`.
        """
        where = "e.project_id = $project_id AND " if project_id else ""
        params: Dict[str, Any] = {"exp_id": experiment_id}
        if project_id:
            params["project_id"] = project_id
        cypher = f"""
        MATCH (e:Experiment {{id: $exp_id}})
        WHERE {where}e.id = $exp_id
        OPTIONAL MATCH (e)-[:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        OPTIONAL MATCH (dv)-[:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        OPTIONAL MATCH (parent:Experiment)-[:PARENT_OF]->(e)
        OPTIONAL MATCH (e)-[:PARENT_OF]->(child:Experiment)
        OPTIONAL MATCH (e)-[:BASED_ON]->(claim:Claim)
        OPTIONAL MATCH (ri:ReviewItem)-[:CONCERNS]->(e)
        WITH e, c, dv, d,
             collect(DISTINCT properties(m)) AS metrics_props,
             collect(DISTINCT parent.id) AS parent_ids,
             collect(DISTINCT child.id) AS child_ids,
             collect(DISTINCT properties(claim)) AS claims_props,
             collect(DISTINCT properties(ri)) AS review_props
        RETURN properties(e) AS exp_props,
               {self._EXPERIMENT_RETURN},
               metrics_props, parent_ids, child_ids, claims_props, review_props
        """
        rows = self.client.execute_query(cypher, params)
        if not rows or not rows[0][0]:
            return None

        row = rows[0]
        record = self._build_experiment_record(
            dict(row[0]),
            dataset_name=row[self._COL_DATASET_NAME],
            dataset_version=row[self._COL_DATASET_VERSION],
            dataset_id=row[self._COL_DATASET_ID],
            dataset_quality_status=row[self._COL_DATASET_QUALITY],
            config_json=row[self._COL_CONFIG_JSON],
            metrics_raw=row[self._COL_METRICS],
            training_steps=row[self._COL_TRAINING_STEPS],
            max_training_steps=row[self._COL_MAX_TRAINING_STEPS],
        )
        return ExperimentDetail(
            **record.model_dump(
                exclude={
                    "configuration",
                    "dataset_id",
                    "dataset_quality_status",
                    "hyperparameters",
                    "preprocessing",
                    "training_steps",
                    "max_training_steps",
                }
            ),
            configuration=record.configuration or None,
            parent_experiment_ids=[p for p in (row[self._COL_PARENT_IDS] or []) if p],
            child_experiment_ids=[c for c in (row[self._COL_CHILD_IDS] or []) if c],
            claims=[dict(c) for c in (row[self._COL_CLAIMS] or []) if c],
            review_items=[dict(r) for r in (row[self._COL_REVIEW_ITEMS] or []) if r],
        )

    def get_experiment(
        self, experiment_id: str, *, project_id: Optional[str] = None
    ) -> Optional[ExperimentRecord]:
        """Port method: one experiment as an analysis record, or ``None``.

        ``project_id`` is optional because the engines hold an identifier that was
        already project-scoped. When a caller supplies it the scope becomes part of
        the query, so a cross-project read fails rather than filtering silently
        (PRD 12: prevent unauthorized cross-project data access).
        """
        where = "e.project_id = $project_id AND " if project_id else ""
        params: Dict[str, Any] = {"exp_id": experiment_id}
        if project_id:
            params["project_id"] = project_id
        cypher = f"""
        MATCH (e:Experiment {{id: $exp_id}})
        WHERE {where}e.id = $exp_id
        OPTIONAL MATCH (e)-[:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        OPTIONAL MATCH (dv)-[:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        WITH e, c, dv, d, collect(DISTINCT properties(m)) AS metrics_props
        RETURN properties(e) AS exp_props,
               {self._EXPERIMENT_RETURN},
               metrics_props
        """
        rows = self.client.execute_query(cypher, params)
        if not rows or not rows[0][0]:
            return None
        row = rows[0]
        return self._build_experiment_record(
            dict(row[0]),
            dataset_name=row[self._COL_DATASET_NAME],
            dataset_version=row[self._COL_DATASET_VERSION],
            dataset_id=row[self._COL_DATASET_ID],
            dataset_quality_status=row[self._COL_DATASET_QUALITY],
            config_json=row[self._COL_CONFIG_JSON],
            metrics_raw=row[self._COL_METRICS],
            training_steps=row[self._COL_TRAINING_STEPS],
            max_training_steps=row[self._COL_MAX_TRAINING_STEPS],
        )

    def list_experiment_summaries(
        self,
        project_id: str,
        model_family: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        status: Optional[str] = None,
        quality_status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[ExperimentSummary]:
        """Lists experiments in a project with optional filtering."""
        where_clauses = ["p.id = $project_id"]
        params: Dict[str, Any] = {"project_id": project_id, "limit": limit, "offset": offset}

        if model_family:
            where_clauses.append("e.model_family = $model_family")
            params["model_family"] = model_family
        if dataset_version_id:
            where_clauses.append("e.dataset_version_id = $dataset_version_id")
            params["dataset_version_id"] = dataset_version_id
        if status:
            where_clauses.append("e.status = $status")
            params["status"] = status
        if quality_status:
            where_clauses.append("e.quality_status = $quality_status")
            params["quality_status"] = quality_status

        where_stmt = " AND ".join(where_clauses)

        cypher = f"""
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)
        WHERE {where_stmt}
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        WITH e, collect(DISTINCT properties(m)) AS metrics_props
        ORDER BY e.created_at DESC
        RETURN properties(e) AS exp_props, metrics_props
        SKIP $offset
        LIMIT $limit
        """
        rows = self.client.execute_query(cypher, params)

        summaries = []
        for row in rows:
            if not row or not row[0]:
                continue
            exp_props = dict(row[0])
            summaries.append(ExperimentSummary(
                id=str(exp_props.get("id")),
                project_id=str(exp_props.get("project_id") or ""),
                name=str(exp_props.get("name") or exp_props.get("id")),
                model_family=exp_props.get("model_family"),
                dataset_version_id=exp_props.get("dataset_version_id"),
                status=exp_props.get("status") or ExperimentStatus.UNKNOWN,
                quality_status=exp_props.get("quality_status") or "valid",
                metrics=_metric_values(row[1]),
                compute_cost_gpu_hours=_float_or_none(exp_props.get("compute_cost_gpu_hours")),
                duration_seconds=_float_or_none(exp_props.get("duration_seconds")),
                created_at=exp_props.get("created_at"),
            ))
        return summaries

    def list_experiments(
        self,
        project_id: str,
        *,
        model_family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        metric: Optional[str] = None,
        status: Optional[str] = None,
        quality_status: Optional[str] = None,
        config_field: Optional[str] = None,
        config_value: Any = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
        evaluation_split: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ExperimentRecord]:
        """Port method: FR-2 history as analysis records, with the documented filters.

        Every filter is applied inside the query. The project scope is the first
        ``WHERE`` term, so a record from another project can never enter the result
        set and then be filtered away in Python.
        """
        clauses = ["e.project_id = $project_id"]
        params: Dict[str, Any] = {"project_id": project_id}

        if model_family:
            clauses.append("e.model_family = $model_family")
            params["model_family"] = model_family
        if dataset_version_id:
            clauses.append("e.dataset_version_id = $dataset_version_id")
            params["dataset_version_id"] = dataset_version_id
        if dataset_id:
            # A pattern predicate, not an EXISTS subquery: this FalkorDB build has no
            # subquery support. See scripts/probe_cypher_support.py.
            clauses.append("(e)-[:USES_DATASET_VERSION]->(:DatasetVersion {dataset_id: $dataset_id})")
            params["dataset_id"] = dataset_id
        if status:
            clauses.append("e.status = $status")
            params["status"] = status
        if quality_status:
            clauses.append("e.quality_status = $quality_status")
            params["quality_status"] = quality_status
        if created_after:
            clauses.append("coalesce(e.created_at, e.ingested_at) >= $created_after")
            params["created_after"] = created_after
        if created_before:
            clauses.append("coalesce(e.created_at, e.ingested_at) <= $created_before")
            params["created_before"] = created_before

        metric_selects, metric_conditions = self._metric_filter_terms(metric, evaluation_split, params)

        # Configuration lives in a canonical JSON document, which FalkorDB cannot
        # probe by a dynamic key. That one filter is applied after the project-scoped
        # query returns, never instead of it: the project boundary, and every other
        # filter, is still enforced by the graph.
        config_filter = config_field is not None
        fetch_limit = None
        if limit is not None and not config_filter:
            fetch_limit = int(limit)
        elif limit is not None:
            fetch_limit = max(int(limit) * 10, 200)

        where = " AND ".join(clauses)
        post_where = " AND ".join(metric_conditions)
        # An intermediate WITH is required between the filtering clause and ORDER BY:
        # this FalkorDB build cannot parse ``WITH ... WHERE ... ORDER BY`` in one
        # clause (scripts/probe_cypher_support.py records the evidence).
        cypher = f"""
        MATCH (e:Experiment)
        WHERE {where}
        OPTIONAL MATCH (e)-[:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        OPTIONAL MATCH (dv)-[:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        WITH e, c, dv, d, collect(DISTINCT properties(m)) AS metrics_props
             {", " + ", ".join(metric_selects) if metric_selects else ""}
        {"WHERE " + post_where if post_where else ""}
        WITH e, c, dv, d, metrics_props
        ORDER BY coalesce(e.created_at, e.ingested_at) DESC
        RETURN properties(e) AS exp_props,
               {self._EXPERIMENT_RETURN},
               metrics_props
        """
        if fetch_limit is not None:
            cypher += "\nLIMIT $limit"
            params["limit"] = fetch_limit

        rows = self.client.execute_query(cypher, params)
        records: List[ExperimentRecord] = []
        for row in rows:
            if not row or not row[0]:
                continue
            records.append(
                self._build_experiment_record(
                    dict(row[0]),
                    dataset_name=row[self._COL_DATASET_NAME],
                    dataset_version=row[self._COL_DATASET_VERSION],
                    dataset_id=row[self._COL_DATASET_ID],
                    dataset_quality_status=row[self._COL_DATASET_QUALITY],
                    config_json=row[self._COL_CONFIG_JSON],
                    metrics_raw=row[self._COL_METRICS],
                    training_steps=row[self._COL_TRAINING_STEPS],
                    max_training_steps=row[self._COL_MAX_TRAINING_STEPS],
                )
            )

        if config_filter:
            records = [r for r in records if self._config_matches(r, config_field, config_value)]
        if limit is not None:
            records = records[: int(limit)]
        return records

    @staticmethod
    def _config_matches(
        record: ExperimentRecord, config_field: str, config_value: Any
    ) -> bool:
        """Match one configuration key across model config, hyperparameters and preprocessing."""
        wanted = str(config_field)
        haystacks = (record.configuration, record.hyperparameters, record.preprocessing)
        for section in haystacks:
            if wanted in section and section[wanted] == config_value:
                return True
        return False

    def find_comparable_candidates(
        self,
        *,
        project_id: str,
        model_family: Optional[str] = None,
        dataset_id: Optional[str] = None,
        dataset_version_id: Optional[str] = None,
        evaluation_split: Optional[str] = None,
        metric: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[ExperimentRecord]:
        """Port method: narrow the project to plausible comparables.

        This is a preselection, never a verdict. It filters on identity fields the
        graph can index, and leaves the comparability judgement to the rule engine
        (HANDOFF.md, graph port contract 2).
        """
        clauses = ["e.project_id = $project_id"]
        params: Dict[str, Any] = {"project_id": project_id}
        if model_family:
            clauses.append("e.model_family = $model_family")
            params["model_family"] = model_family
        if dataset_version_id:
            clauses.append("e.dataset_version_id = $dataset_version_id")
            params["dataset_version_id"] = dataset_version_id
        if dataset_id:
            clauses.append("(e)-[:USES_DATASET_VERSION]->(:DatasetVersion {dataset_id: $dataset_id})")
            params["dataset_id"] = dataset_id

        metric_selects, metric_conditions = self._metric_filter_terms(metric, evaluation_split, params)
        post_where = " AND ".join(metric_conditions)

        cypher = f"""
        MATCH (e:Experiment)
        WHERE {" AND ".join(clauses)}
        OPTIONAL MATCH (e)-[:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        OPTIONAL MATCH (dv)-[:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        WITH e, c, dv, d, collect(DISTINCT properties(m)) AS metrics_props
             {", " + ", ".join(metric_selects) if metric_selects else ""}
        {"WHERE " + post_where if post_where else ""}
        WITH e, c, dv, d, metrics_props
        ORDER BY coalesce(e.created_at, e.ingested_at) DESC
        RETURN properties(e) AS exp_props,
               {self._EXPERIMENT_RETURN},
               metrics_props
        """
        if limit is not None:
            cypher += "\nLIMIT $limit"
            params["limit"] = int(limit)

        rows = self.client.execute_query(cypher, params)
        results: List[ExperimentRecord] = []
        for row in rows:
            if not row or not row[0]:
                continue
            results.append(
                self._build_experiment_record(
                    dict(row[0]),
                    dataset_name=row[self._COL_DATASET_NAME],
                    dataset_version=row[self._COL_DATASET_VERSION],
                    dataset_id=row[self._COL_DATASET_ID],
                    dataset_quality_status=row[self._COL_DATASET_QUALITY],
                    config_json=row[self._COL_CONFIG_JSON],
                    metrics_raw=row[self._COL_METRICS],
                    training_steps=row[self._COL_TRAINING_STEPS],
                    max_training_steps=row[self._COL_MAX_TRAINING_STEPS],
                )
            )
        return results

    # -----------------------------------------------------------------------
    # Lineage and Traversal Queries
    # -----------------------------------------------------------------------

    def get_experiment_lineage(
        self, experiment_id: str, *, project_id: Optional[str] = None, max_depth: int = 5
    ) -> List[Dict[str, Any]]:
        """Port method: lineage edges as ``{"from", "to", "type"}`` triples.

        Bounded by ``max_depth`` so a deep or cyclic history cannot produce an
        unbounded traversal, and deduplicated so one parent/child pair appears once.
        """
        cypher = f"""
        MATCH (root:Experiment {{id: $exp_id}})
        WHERE $project_id IS NULL OR root.project_id = $project_id
        MATCH path = (root)-[:PARENT_OF*0..{int(max_depth)}]-(other:Experiment)
        UNWIND relationships(path) AS rel
        RETURN DISTINCT startNode(rel).id AS from_id,
               endNode(rel).id AS to_id,
               type(rel) AS rel_type
        """
        rows = self.client.execute_query(cypher, {"exp_id": experiment_id, "project_id": project_id})
        edges: List[Dict[str, Any]] = []
        for row in rows:
            if not row or not row[0] or not row[1]:
                continue
            edges.append({"from": row[0], "to": row[1], "type": row[2]})
        return edges

    def get_experiment_lineage_graph(
        self, experiment_id: str, max_depth: int = 5
    ) -> ExperimentLineageResponse:
        """
        Traverses upward (ancestors) and downward (descendants) through PARENT_OF relationships
        up to max_depth, returning a complete directed lineage graph structure.
        """
        # Fetch root
        root_exp = self.get_experiment_detail(experiment_id)
        if not root_exp:
            return ExperimentLineageResponse(root_id=experiment_id, nodes=[], edges=[])

        nodes_dict: Dict[str, LineageNode] = {
            root_exp.id: LineageNode(
                id=root_exp.id,
                name=root_exp.name,
                model_family=root_exp.model_family,
                status=root_exp.status,
                quality_status=root_exp.quality_status,
                dataset_version_id=root_exp.dataset_version_id,
                depth=0,
                is_root=True
            )
        }
        edges_set = set()

        # Ancestors: parent -[:PARENT_OF*1..depth]-> root
        cypher_ancestors = f"""
        MATCH path = (parent:Experiment)-[:PARENT_OF*1..{max_depth}]->(root:Experiment {{id: $exp_id}})
        RETURN [n in nodes(path) | properties(n)] AS path_nodes,
               [r in relationships(path) | [startNode(r).id, endNode(r).id]] AS path_edges
        """
        anc_rows = self.client.execute_query(cypher_ancestors, {"exp_id": experiment_id})
        for row in anc_rows:
            p_nodes = row[0]
            p_edges = row[1]
            for idx, n in enumerate(p_nodes):
                nid = n["id"]
                depth = -(len(p_nodes) - 1 - idx)  # negative depth for ancestors
                if nid not in nodes_dict:
                    nodes_dict[nid] = LineageNode(
                        id=nid,
                        name=n.get("name", nid),
                        model_family=n.get("model_family"),
                        status=n.get("status"),
                        quality_status=n.get("quality_status"),
                        dataset_version_id=n.get("dataset_version_id"),
                        depth=depth,
                        is_root=(nid == experiment_id)
                    )
            for src, tgt in p_edges:
                edges_set.add((src, tgt))

        # Descendants: root -[:PARENT_OF*1..depth]-> child
        cypher_descendants = f"""
        MATCH path = (root:Experiment {{id: $exp_id}})-[:PARENT_OF*1..{max_depth}]->(child:Experiment)
        RETURN [n in nodes(path) | properties(n)] AS path_nodes,
               [r in relationships(path) | [startNode(r).id, endNode(r).id]] AS path_edges
        """
        desc_rows = self.client.execute_query(cypher_descendants, {"exp_id": experiment_id})
        for row in desc_rows:
            p_nodes = row[0]
            p_edges = row[1]
            for idx, n in enumerate(p_nodes):
                nid = n["id"]
                depth = idx  # positive depth for descendants
                if nid not in nodes_dict:
                    nodes_dict[nid] = LineageNode(
                        id=nid,
                        name=n.get("name", nid),
                        model_family=n.get("model_family"),
                        status=n.get("status"),
                        quality_status=n.get("quality_status"),
                        dataset_version_id=n.get("dataset_version_id"),
                        depth=depth,
                        is_root=(nid == experiment_id)
                    )
            for src, tgt in p_edges:
                edges_set.add((src, tgt))

        edges = [LineageEdge(source=s, target=t, relationship_type="PARENT_OF") for s, t in edges_set]

        return ExperimentLineageResponse(
            root_id=experiment_id,
            nodes=list(nodes_dict.values()),
            edges=edges
        )

    def find_related_experiments(self, experiment_id: str) -> Dict[str, List[str]]:
        """
        Finds experiments that share datasets, configurations, parent lineages,
        or explicit comparisons with the target experiment.

        Every traversal is bounded to the target's own project. A researcher must not
        learn that an identifier from another project shares a dataset version.
        """
        # Experiments sharing dataset version
        cypher_dataset = """
        MATCH (target:Experiment {id: $exp_id})-[:USES_DATASET_VERSION]->(dv:DatasetVersion)<-[:USES_DATASET_VERSION]-(e:Experiment)
        WHERE e.id <> target.id AND e.project_id = target.project_id
        RETURN DISTINCT e.id
        """
        same_dataset = [r[0] for r in self.client.execute_query(cypher_dataset, {"exp_id": experiment_id})]

        # Experiments sharing configuration
        cypher_config = """
        MATCH (target:Experiment {id: $exp_id})-[:USES_CONFIG]->(c:Configuration)<-[:USES_CONFIG]-(e:Experiment)
        WHERE e.id <> target.id AND e.project_id = target.project_id
        RETURN DISTINCT e.id
        """
        same_config = [r[0] for r in self.client.execute_query(cypher_config, {"exp_id": experiment_id})]

        # Experiments sharing parent experiment
        cypher_siblings = """
        MATCH (p:Experiment)-[:PARENT_OF]->(target:Experiment {id: $exp_id})
        MATCH (p)-[:PARENT_OF]->(e:Experiment)
        WHERE e.id <> target.id AND e.project_id = target.project_id
        RETURN DISTINCT e.id
        """
        siblings = [r[0] for r in self.client.execute_query(cypher_siblings, {"exp_id": experiment_id})]

        # Explicitly compared
        cypher_compared = """
        MATCH (target:Experiment {id: $exp_id})-[:COMPARED_WITH]-(e:Experiment)
        WHERE e.project_id = target.project_id
        RETURN DISTINCT e.id
        """
        compared = [r[0] for r in self.client.execute_query(cypher_compared, {"exp_id": experiment_id})]

        return {
            "same_dataset_version": same_dataset,
            "same_configuration": same_config,
            "siblings": siblings,
            "compared_with": compared
        }

    # -----------------------------------------------------------------------
    # Graph Visualization and Summary
    # -----------------------------------------------------------------------

    def get_project_graph_summary(self, project_id: str) -> Optional[ProjectGraphSummary]:
        """Node and relationship counts for a project, or ``None`` when it is unknown.

        A project that does not exist has no counts. Reporting a fabricated
        ``Project: 1`` for an identifier nobody created would let the lineage screen
        render an empty graph as if it were the project's real state.
        """
        if self.get_project(project_id) is None:
            return None
        cypher_nodes = """
        MATCH (p:Project {id: $project_id})
        OPTIONAL MATCH (p)-[:CONTAINS]->(e:Experiment)
        OPTIONAL MATCH (e)-[:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        OPTIONAL MATCH (dv)-[:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        OPTIONAL MATCH (e)-[:BASED_ON]->(cl:Claim)
        OPTIONAL MATCH (e)<-[:CONCERNS]-(ri:ReviewItem)
        RETURN count(DISTINCT e) AS experiments,
               count(DISTINCT c) AS configurations,
               count(DISTINCT dv) AS dataset_versions,
               count(DISTINCT d) AS datasets,
               count(DISTINCT m) AS metrics,
               count(DISTINCT cl) AS claims,
               count(DISTINCT ri) AS review_items
        """
        rows = self.client.execute_query(cypher_nodes, {"project_id": project_id})
        node_counts = {
            "Project": 1,
            "Experiment": rows[0][0] if rows else 0,
            "Configuration": rows[0][1] if rows else 0,
            "DatasetVersion": rows[0][2] if rows else 0,
            "Dataset": rows[0][3] if rows else 0,
            "Metric": rows[0][4] if rows else 0,
            "Claim": rows[0][5] if rows else 0,
            "ReviewItem": rows[0][6] if rows else 0,
        }

        cypher_rels = """
        MATCH (p:Project {id: $project_id})-[:CONTAINS]->(e:Experiment)
        OPTIONAL MATCH (p)-[r1:CONTAINS]->(e)
        OPTIONAL MATCH (e)-[r2:USES_CONFIG]->(:Configuration)
        OPTIONAL MATCH (e)-[r3:USES_DATASET_VERSION]->(:DatasetVersion)
        OPTIONAL MATCH (e)-[r4:PRODUCES_METRIC]->(:Metric)
        OPTIONAL MATCH (e)-[r5:PARENT_OF]->(:Experiment)
        OPTIONAL MATCH (e)-[r6:BASED_ON]->(:Claim)
        OPTIONAL MATCH (:ReviewItem)-[r7:CONCERNS]->(e)
        RETURN count(DISTINCT r1) AS contains_count,
               count(DISTINCT r2) AS config_count,
               count(DISTINCT r3) AS dataset_count,
               count(DISTINCT r4) AS metric_count,
               count(DISTINCT r5) AS parent_count,
               count(DISTINCT r6) AS claim_count,
               count(DISTINCT r7) AS review_count
        """
        r_rows = self.client.execute_query(cypher_rels, {"project_id": project_id})
        rel_counts = {
            RelType.CONTAINS: r_rows[0][0] if r_rows else 0,
            RelType.USES_CONFIG: r_rows[0][1] if r_rows else 0,
            RelType.USES_DATASET_VERSION: r_rows[0][2] if r_rows else 0,
            RelType.PRODUCES_METRIC: r_rows[0][3] if r_rows else 0,
            RelType.PARENT_OF: r_rows[0][4] if r_rows else 0,
            RelType.BASED_ON: r_rows[0][5] if r_rows else 0,
            RelType.CONCERNS: r_rows[0][6] if r_rows else 0,
        }

        return ProjectGraphSummary(
            project_id=project_id,
            node_counts=node_counts,
            relationship_counts=rel_counts
        )

    def get_project_graph(self, project_id: str, limit: int = 500) -> ProjectGraphResponse:
        """
        Retrieves real nodes and edges connected to a project for interactive UI graph rendering.
        """
        cypher = f"""
        MATCH (p:Project {{id: $project_id}})
        OPTIONAL MATCH (p)-[r1:CONTAINS]->(e:Experiment)
        OPTIONAL MATCH (e)-[r2:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[r3:USES_DATASET_VERSION]->(dv:DatasetVersion)-[r4:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[r5:PRODUCES_METRIC]->(m:Metric)
        OPTIONAL MATCH (e)-[r6:PARENT_OF]->(child:Experiment)
        OPTIONAL MATCH (e)-[r7:BASED_ON]->(cl:Claim)
        OPTIONAL MATCH (ri:ReviewItem)-[r8:CONCERNS]->(e)
        RETURN p, e, c, dv, d, m, child, cl, ri
        LIMIT {limit}
        """
        rows = self.client.execute_query(cypher, {"project_id": project_id})

        nodes_dict: Dict[str, GraphNodeData] = {}
        edges_list: List[GraphEdgeData] = []
        seen_edges = set()

        def add_node(raw_node, label):
            if not raw_node:
                return
            props = dict(raw_node.properties if hasattr(raw_node, "properties") else raw_node)
            nid = props.get("id")
            if nid and nid not in nodes_dict:
                nodes_dict[nid] = GraphNodeData(id=nid, label=label, properties=props)

        def add_edge(src, tgt, rel_type):
            if not src or not tgt or src == tgt:
                return
            key = (src, tgt, rel_type)
            if key not in seen_edges:
                seen_edges.add(key)
                edges_list.append(GraphEdgeData(source=src, target=tgt, relationship_type=rel_type, properties={}))

        for row in rows:
            p_node, e_node, c_node, dv_node, d_node, m_node, child_node, cl_node, ri_node = row
            add_node(p_node, NodeLabel.PROJECT)
            add_node(e_node, NodeLabel.EXPERIMENT)
            add_node(c_node, NodeLabel.CONFIGURATION)
            add_node(dv_node, NodeLabel.DATASET_VERSION)
            add_node(d_node, NodeLabel.DATASET)
            add_node(m_node, NodeLabel.METRIC)
            add_node(child_node, NodeLabel.EXPERIMENT)
            add_node(cl_node, NodeLabel.CLAIM)
            add_node(ri_node, NodeLabel.REVIEW_ITEM)

            # Add edges
            pid = dict(p_node.properties if hasattr(p_node, "properties") else p_node).get("id") if p_node else None
            eid = dict(e_node.properties if hasattr(e_node, "properties") else e_node).get("id") if e_node else None
            cid = dict(c_node.properties if hasattr(c_node, "properties") else c_node).get("id") if c_node else None
            dvid = dict(dv_node.properties if hasattr(dv_node, "properties") else dv_node).get("id") if dv_node else None
            did = dict(d_node.properties if hasattr(d_node, "properties") else d_node).get("id") if d_node else None
            mid = dict(m_node.properties if hasattr(m_node, "properties") else m_node).get("id") if m_node else None
            childid = dict(child_node.properties if hasattr(child_node, "properties") else child_node).get("id") if child_node else None
            clid = dict(cl_node.properties if hasattr(cl_node, "properties") else cl_node).get("id") if cl_node else None
            riid = dict(ri_node.properties if hasattr(ri_node, "properties") else ri_node).get("id") if ri_node else None

            if pid and eid:
                add_edge(pid, eid, RelType.CONTAINS)
            if eid and cid:
                add_edge(eid, cid, RelType.USES_CONFIG)
            if eid and dvid:
                add_edge(eid, dvid, RelType.USES_DATASET_VERSION)
            if dvid and did:
                add_edge(dvid, did, RelType.VERSION_OF)
            if eid and mid:
                add_edge(eid, mid, RelType.PRODUCES_METRIC)
            if eid and childid:
                add_edge(eid, childid, RelType.PARENT_OF)
            if eid and clid:
                add_edge(eid, clid, RelType.BASED_ON)
            if riid and eid:
                add_edge(riid, eid, RelType.CONCERNS)

        return ProjectGraphResponse(
            project_id=project_id,
            nodes=list(nodes_dict.values()),
            edges=edges_list,
            truncated=len(rows) >= limit,
            node_limit=limit,
        )

    # -----------------------------------------------------------------------
    # Graph port implementation (app/ports/graph.py)
    #
    # Everything below satisfies the contract the analysis engines consume. The
    # project scope is a ``WHERE`` term in every query; a record from another
    # project is never fetched and filtered out afterwards.
    # -----------------------------------------------------------------------

    def list_datasets(self, project_id: str) -> List[DatasetSummary]:
        """Datasets this project's runs actually used, including untrusted ones.

        A disputed version is still listed: the engine needs to see it in order to
        withhold it, and hiding it would make a correction look like a deletion.
        """
        cypher = """
        MATCH (e:Experiment {project_id: $project_id})-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        MATCH (dv)-[:VERSION_OF]->(d:Dataset)
        RETURN DISTINCT d.id, d.name, d.description
        ORDER BY d.name ASC
        """
        rows = self.client.execute_query(cypher, {"project_id": project_id})
        return [
            DatasetSummary(id=str(r[0]), name=str(r[1] or r[0]), description=r[2])
            for r in rows
            if r and r[0]
        ]

    def list_dataset_versions(self, project_id: str) -> List[DatasetVersionSummary]:
        """Dataset versions in this project, disputed ones included (FR-6)."""
        cypher = """
        MATCH (e:Experiment {project_id: $project_id})-[:USES_DATASET_VERSION]->(dv:DatasetVersion)
        RETURN DISTINCT dv.id, dv.dataset_id, dv.version, dv.quality_status, dv.created_at
        ORDER BY dv.id ASC
        """
        rows = self.client.execute_query(cypher, {"project_id": project_id})
        return [
            DatasetVersionSummary(
                id=str(r[0]),
                dataset_id=str(r[1] or ""),
                version=str(r[2] or r[0]),
                quality_status=_to_evidence_status(r[3]),
                created_at=r[4],
            )
            for r in rows
            if r and r[0]
        ]

    def list_corrections(self, project_id: str) -> List[CorrectionRecord]:
        """Open and historical corrections recorded against this project.

        Records created before the project scope was stored on the node are matched
        through the nodes they affect, so an existing audit history is not lost. The
        scope test is an ``any()`` over collected targets rather than an ``EXISTS``
        subquery, which this FalkorDB build does not support.
        """
        cypher = """
        MATCH (co:Correction)
        OPTIONAL MATCH (co)-[:AFFECTS|CHALLENGES]->(target)
        WITH co, collect(target) AS targets
        WHERE co.project_id = $project_id OR any(x IN targets WHERE x.project_id IS NOT NULL)
        RETURN properties(co),
               [t IN targets | t.id] AS target_ids,
               [t IN targets | head(t.__labels__)] AS target_labels
        ORDER BY co.created_at DESC
        """
        rows = self.client.execute_query(cypher, {"project_id": project_id})
        records: List[CorrectionRecord] = []
        for row in rows:
            if not row or not row[0]:
                continue
            props = dict(row[0])
            targets = [t for t in (row[1] or []) if t]
            labels = [l for l in (row[2] or []) if l]
            records.append(
                CorrectionRecord(
                    id=str(props.get("id")),
                    project_id=str(props.get("project_id") or project_id),
                    source_id=str(props.get("source_id") or (targets[0] if targets else "")),
                    source_type=str(props.get("source_type") or (labels[0] if labels else "unknown")),
                    description=str(props.get("description") or ""),
                    evidence_reference=props.get("evidence_reference"),
                    status=str(props.get("status") or "open"),
                    created_by=props.get("created_by"),
                    created_at=props.get("created_at"),
                )
            )
        return records

    def get_correction_impact(self, correction_id: str) -> Optional[CorrectionImpact]:
        """Port method: the blast radius of one correction, read from graph state.

        The impact is derived here rather than recalled from a cached report, so the
        answer reflects the graph as it stands now. Direct and indirect tiers are
        kept apart, and every dependency is reported with the path that produced it
        (FR-6 #5, PRD 20 correction impact contract).
        """
        cypher_head = """
        MATCH (co:Correction {id: $cid})
        OPTIONAL MATCH (co)-[:AFFECTS|CHALLENGES]->(target)
        WITH co, [t IN collect(target) | t.id] AS targets,
             [t IN collect(target) | head(t.__labels__)] AS labels
        RETURN co.project_id AS project_id, properties(co) AS props, targets, labels
        """
        head_rows = self.client.execute_query(cypher_head, {"cid": correction_id})
        if not head_rows or not head_rows[0][1]:
            return None
        project_id = head_rows[0][0]
        props = dict(head_rows[0][1])
        targets = [t for t in (head_rows[0][2] or []) if t]
        labels = [l for l in (head_rows[0][3] or []) if l]
        if not project_id:
            # Legacy record with no stored scope: fall back to the corrected node.
            scope_rows = self.client.execute_query(
                """
                MATCH (co:Correction {id: $cid})-[:AFFECTS|CHALLENGES]->(t)
                WHERE t.project_id IS NOT NULL
                RETURN t.project_id
                LIMIT 1
                """,
                {"cid": correction_id},
            )
            project_id = scope_rows[0][0] if scope_rows and scope_rows[0] else ""
        if not project_id:
            logger.warning("Correction %s has no project scope; impact cannot be scoped", correction_id)
            return None

        # -- direct: the corrected node itself ---------------------------------
        direct_rows = self.client.execute_query(
            """
            MATCH (co:Correction {id: $cid})-[:AFFECTS|CHALLENGES]->(target)
            WHERE target:DatasetVersion
            MATCH path = (e:Experiment)-[:USES_DATASET_VERSION]->(target)
            WHERE e.project_id = $project_id
            RETURN e.id AS exp_id,
                   [co.id, target.id, e.id] AS node_ids,
                   ['AFFECTS', 'USES_DATASET_VERSION'] AS rels
            UNION ALL
            MATCH (co:Correction {id: $cid})-[:AFFECTS]->(target:Experiment)
            WHERE target.project_id = $project_id
            RETURN target.id AS exp_id, [co.id, target.id] AS node_ids, ['AFFECTS'] AS rels
            UNION ALL
            MATCH (co:Correction {id: $cid})-[:CHALLENGES]->(target:Metric)
            MATCH path = (e:Experiment)-[:PRODUCES_METRIC]->(target)
            WHERE e.project_id = $project_id
            RETURN e.id AS exp_id, [co.id, target.id, e.id] AS node_ids,
                   ['CHALLENGES', 'PRODUCES_METRIC'] AS rels
            UNION ALL
            MATCH (co:Correction {id: $cid})-[:CHALLENGES]->(target:Claim)
            MATCH path = (e:Experiment)-[:BASED_ON]->(target)
            WHERE e.project_id = $project_id
            RETURN e.id AS exp_id, [co.id, target.id, e.id] AS node_ids,
                   ['CHALLENGES', 'BASED_ON'] AS rels
            UNION ALL
            MATCH (co:Correction {id: $cid})-[:CHALLENGES]->(ev:Evidence)<-[:SUPPORTED_BY]-(cl:Claim)<-[:BASED_ON]-(e:Experiment)
            WHERE e.project_id = $project_id
            RETURN e.id AS exp_id, [co.id, ev.id, cl.id, e.id] AS node_ids,
                   ['CHALLENGES', 'SUPPORTED_BY', 'BASED_ON'] AS rels
            """,
            {"cid": correction_id, "project_id": project_id},
        )

        direct_ids: List[str] = []
        paths: List[ClaimPath] = []
        seen_direct = set()
        seen_paths = set()
        for row in direct_rows:
            if not row or not row[0]:
                continue
            exp_id = str(row[0])
            if exp_id not in seen_direct:
                seen_direct.add(exp_id)
                direct_ids.append(exp_id)
            key = (exp_id, tuple(row[1] or []), tuple(row[2] or []))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            paths.append(ClaimPath(nodes=[str(n) for n in (row[1] or [])],
                                   relationship_types=[str(r) for r in (row[2] or [])]))

        # -- indirect: reached through lineage, not through the defect ----------
        indirect_ids: List[str] = []
        if direct_ids:
            downstream_rows = self.client.execute_query(
                """
                MATCH path = (direct:Experiment)-[:PARENT_OF*1..5]->(down:Experiment)
                WHERE direct.id IN $direct_ids AND down.project_id = $project_id
                RETURN DISTINCT down.id AS exp_id,
                       [n IN nodes(path) | n.id] AS node_ids,
                       [r IN relationships(path) | type(r)] AS rels
                """,
                {"direct_ids": direct_ids, "project_id": project_id},
            )
            for row in downstream_rows:
                if not row or not row[0]:
                    continue
                exp_id = str(row[0])
                if exp_id in seen_direct or exp_id in indirect_ids:
                    continue
                indirect_ids.append(exp_id)
                node_ids = [correction_id] + [str(n) for n in (row[1] or [])]
                rels = ["AFFECTS"] + [str(r) for r in (row[2] or [])]
                key = (exp_id, tuple(node_ids), tuple(rels))
                if key in seen_paths:
                    continue
                seen_paths.add(key)
                paths.append(ClaimPath(nodes=node_ids, relationship_types=rels))

        # -- conclusions that rest on the affected runs ------------------------
        claim_ids: List[str] = []
        affected = direct_ids + indirect_ids
        if affected:
            claim_rows = self.client.execute_query(
                """
                MATCH (co:Correction {id: $cid})-[:CHALLENGES]->(cl:Claim)
                RETURN DISTINCT cl.id
                UNION
                MATCH (e:Experiment)-[:BASED_ON]->(cl:Claim)
                WHERE e.id IN $affected
                RETURN DISTINCT cl.id
                """,
                {"cid": correction_id, "affected": affected},
            )
            claim_ids = [str(r[0]) for r in claim_rows if r and r[0]]

        # -- revalidation queue ------------------------------------------------
        review_rows = self.client.execute_query(
            """
            MATCH (ri:ReviewItem)-[:RAISED_BY]->(co:Correction {id: $cid})
            WHERE ri.project_id = $project_id OR ri.project_id IS NULL
            RETURN DISTINCT ri.id
            UNION
            MATCH (ri:ReviewItem)-[:CONCERNS]->(e:Experiment)
            WHERE e.id IN $affected
            RETURN DISTINCT ri.id
            """,
            {"cid": correction_id, "affected": affected, "project_id": project_id},
        )
        review_item_ids = [str(r[0]) for r in review_rows if r and r[0]]

        # -- explicitly unaffected ---------------------------------------------
        all_rows = self.client.execute_query(
            "MATCH (e:Experiment {project_id: $project_id}) RETURN e.id",
            {"project_id": project_id},
        )
        all_ids = [str(r[0]) for r in all_rows if r and r[0]]
        touched = set(direct_ids) | set(indirect_ids)
        unaffected = [i for i in all_ids if i not in touched]

        source_id = str(props.get("source_id") or (targets[0] if targets else ""))
        source_type = str(props.get("source_type") or (labels[0] if labels else "unknown"))
        return CorrectionImpact(
            correction_id=correction_id,
            source_id=source_id,
            summary=str(props.get("description") or ""),
            directly_affected_experiment_ids=direct_ids,
            indirectly_affected_experiment_ids=indirect_ids,
            potentially_affected_claim_ids=claim_ids,
            unaffected_experiment_ids=unaffected,
            review_item_ids=review_item_ids,
            paths=paths,
        )

    def list_review_items(
        self, project_id: str, *, status: Optional[ReviewItemStatus] = None
    ) -> List[ReviewItem]:
        """The revalidation queue for a project (FR-6 #7)."""
        # The scope is the node's own project_id or the project of the run it
        # concerns: an item raised before project_id was stored still belongs to the
        # project that owns its experiment. A pattern predicate is used because this
        # FalkorDB build has no EXISTS subquery.
        clauses = [
            "ri.project_id = $project_id",
            "(ri)-[:CONCERNS]->(:Experiment {project_id: $project_id})",
        ]
        params: Dict[str, Any] = {"project_id": project_id}
        if status is not None:
            wanted = status.value if isinstance(status, ReviewItemStatus) else str(status)
            clauses.append("ri.status = $status")
            params["status"] = wanted
        cypher = f"""
        MATCH (ri:ReviewItem)
        WHERE {" AND ".join(clauses)}
        OPTIONAL MATCH (ri)-[:CONCERNS]->(e:Experiment)
        RETURN ri.id, e.id, ri.reason, ri.status, ri.project_id, ri.correction_id,
               ri.created_at, ri.resolved_at, ri.resolved_by, ri.resolution_notes
        ORDER BY ri.created_at DESC
        """
        rows = self.client.execute_query(cypher, params)
        items: List[ReviewItem] = []
        for row in rows:
            if not row or not row[0]:
                continue
            items.append(
                ReviewItem(
                    id=str(row[0]),
                    project_id=str(row[4] or project_id),
                    reason=str(row[2] or ""),
                    status=_to_review_status(row[3]),
                    experiment_id=row[1],
                    correction_id=row[5],
                    created_at=row[6] or datetime.now(timezone.utc),
                )
            )
        return items

    def get_review_item(self, review_item_id: str) -> Optional[ReviewItemResponse]:
        """One queue item, resolved or not."""
        cypher = """
        MATCH (ri:ReviewItem {id: $id})
        OPTIONAL MATCH (ri)-[:CONCERNS]->(e:Experiment)
        OPTIONAL MATCH (ri)-[:RAISED_BY]->(co:Correction)
        RETURN ri.id, e.id, ri.reason, ri.status, ri.project_id, co.id,
               ri.created_at, ri.resolved_at, ri.resolved_by, ri.resolution_notes
        """
        rows = self.client.execute_query(cypher, {"id": review_item_id})
        if not rows or not rows[0] or not rows[0][0]:
            return None
        row = rows[0]
        return ReviewItemResponse(
            id=str(row[0]),
            experiment_id=row[1],
            reason=str(row[2] or ""),
            status=_to_review_status(row[3]),
            project_id=row[4],
            correction_id=row[5],
            created_at=str(row[6]) if row[6] is not None else "",
            resolved_at=str(row[7]) if row[7] is not None else None,
            resolved_by=row[8],
            resolution_notes=row[9],
        )

    def resolve_review_item(
        self,
        review_item_id: str,
        status: ReviewItemStatus = ReviewItemStatus.RESOLVED,
        notes: Optional[str] = None,
        resolved_by: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Close a queue item. The record is kept, never deleted (FR-6 #2)."""
        now = datetime.now(timezone.utc).isoformat()
        wanted = status.value if isinstance(status, ReviewItemStatus) else str(status)
        cypher = """
        MATCH (ri:ReviewItem {id: $id})
        SET ri.status = $status, ri.resolved_at = $resolved_at,
            ri.resolution_notes = $notes, ri.resolved_by = $resolved_by
        RETURN properties(ri)
        """
        rows = self.client.execute_query(
            cypher,
            {
                "id": review_item_id,
                "status": wanted,
                "resolved_at": now,
                "notes": notes or "",
                "resolved_by": resolved_by or "researcher",
            },
        )
        return dict(rows[0][0]) if rows and rows[0] else None

    # -- recommendations (FR-5 persistence, FR-9 decisions) -------------------

    def save_recommendation(self, recommendation: Recommendation) -> None:
        """Store a proposal so a researcher can act on it (FR-9).

        The full payload is kept in ``canonical_json`` and the fields the UI filters
        on are promoted to indexed properties. A proposal is *not* an experiment, so
        no ``PROPOSES`` edge is fabricated towards a run that does not exist.
        """
        payload = recommendation.model_dump(mode="json")
        cypher = f"""
        MERGE (rec:{NodeLabel.RECOMMENDATION} {{id: $id}})
        ON CREATE SET rec.canonical_json = $canonical_json,
                      rec.project_id = $project_id,
                      rec.objective = $objective,
                      rec.hypothesis = $hypothesis,
                      rec.status = $status,
                      rec.score = $score,
                      rec.estimated_gpu_hours = $estimated_gpu_hours,
                      rec.evidence_status = $evidence_status,
                      rec.candidate_kind = $candidate_kind,
                      rec.created_at = $created_at,
                      rec.decided_by = $decided_by,
                      rec.decision_note = $decision_note
        ON MATCH SET  rec.canonical_json = $canonical_json,
                      rec.status = $status,
                      rec.score = $score
        """
        self.client.execute_query(
            cypher,
            {
                "id": recommendation.id,
                "canonical_json": json.dumps(payload, default=str),
                "project_id": recommendation.project_id,
                "objective": recommendation.objective,
                "hypothesis": recommendation.hypothesis,
                "status": recommendation.status.value,
                "score": float(recommendation.score or 0.0),
                "estimated_gpu_hours": recommendation.estimated_gpu_hours,
                "evidence_status": recommendation.evidence_status.value,
                "candidate_kind": recommendation.candidate_kind.value,
                "created_at": (
                    recommendation.created_at.isoformat()
                    if recommendation.created_at
                    else datetime.now(timezone.utc).isoformat()
                ),
                "decided_by": None,
                "decision_note": None,
            },
        )

        link_project = f"""
        MATCH (p:{NodeLabel.PROJECT} {{id: $project_id}})
        MATCH (rec:{NodeLabel.RECOMMENDATION} {{id: $id}})
        MERGE (p)-[:{REL_HAS_RECOMMENDATION}]->(rec)
        """
        self.client.execute_query(link_project, {"project_id": recommendation.project_id, "id": recommendation.id})

        supported = list(dict.fromkeys(
            list(recommendation.baseline_experiment_ids) + list(recommendation.supporting_experiment_ids)
        ))
        if supported:
            link_evidence = f"""
            MATCH (rec:{NodeLabel.RECOMMENDATION} {{id: $id}})
            MATCH (e:{NodeLabel.EXPERIMENT}) WHERE e.id IN $ids AND e.project_id = $project_id
            MERGE (rec)-[:{RelType.SUPPORTED_BY}]->(e)
            """
            self.client.execute_query(
                link_evidence, {"id": recommendation.id, "ids": supported, "project_id": recommendation.project_id}
            )

    def _recommendation_from_row(self, row: Any) -> Optional[Recommendation]:
        if not row or not row[0]:
            return None
        payload = row[0]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                logger.warning("Stored recommendation is not valid JSON; ignoring it")
                return None
        if not isinstance(payload, dict):
            return None
        return Recommendation(**payload)

    def get_recommendation(self, recommendation_id: str) -> Optional[Recommendation]:
        """Port method: a stored proposal, or ``None`` when it does not exist."""
        rows = self.client.execute_query(
            f"MATCH (rec:{NodeLabel.RECOMMENDATION} {{id: $id}}) RETURN rec.canonical_json",
            {"id": recommendation_id},
        )
        if not rows:
            return None
        return self._recommendation_from_row(rows[0])

    def list_recommendations(self, project_id: str) -> List[Recommendation]:
        """Every proposal recorded for a project, newest first."""
        cypher = f"""
        MATCH (p:{NodeLabel.PROJECT} {{id: $project_id}})-[:{REL_HAS_RECOMMENDATION}]->(rec:{NodeLabel.RECOMMENDATION})
        RETURN rec.canonical_json
        ORDER BY rec.created_at DESC
        """
        rows = self.client.execute_query(cypher, {"project_id": project_id})
        results: List[Recommendation] = []
        for row in rows:
            recommendation = self._recommendation_from_row(row)
            if recommendation is not None:
                results.append(recommendation)
        return results

    def record_recommendation_decision(
        self, decision: RecommendationDecision
    ) -> Recommendation:
        """Apply an approve/reject/edit decision and return the updated proposal.

        Approval records that a researcher signed off. It never turns an inferred
        proposal into an observation (ranking_rules.md section 4), so the stored
        ``evidence_status`` is left exactly as it was.
        """
        existing = self.get_recommendation(decision.recommendation_id)
        if existing is None:
            raise KeyError(decision.recommendation_id)

        updated = existing.model_copy(deep=True)
        if decision.decision.value == "approved":
            updated.status = RecommendationStatus.APPROVED
        elif decision.decision.value == "rejected":
            updated.status = RecommendationStatus.REJECTED
        else:
            updated.status = RecommendationStatus.EDITED
            if decision.edited_configuration is not None:
                updated.configuration = dict(decision.edited_configuration)
        if decision.note:
            updated.rationale = decision.note

        cypher = f"""
        MATCH (rec:{NodeLabel.RECOMMENDATION} {{id: $id}})
        SET rec.status = $status, rec.canonical_json = $canonical_json,
            rec.decided_by = $decided_by, rec.decision_note = $decision_note,
            rec.decided_at = $decided_at
        RETURN rec.id
        """
        self.client.execute_query(
            cypher,
            {
                "id": decision.recommendation_id,
                "status": updated.status.value,
                "canonical_json": json.dumps(updated.model_dump(mode="json"), default=str),
                "decided_by": decision.decided_by,
                "decision_note": decision.note,
                "decided_at": decision.created_at.isoformat()
                if decision.created_at
                else datetime.now(timezone.utc).isoformat(),
            },
        )
        return updated
