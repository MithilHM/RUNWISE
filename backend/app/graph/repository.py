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
    ReviewItemResponse
)

logger = logging.getLogger(__name__)


class GraphRepository:
    def __init__(self, client: Optional[FalkorDBClient] = None):
        self.client = client or graph_client

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
        return dict(rows[0][0]) if rows and rows[0] else {"id": project_id, "name": name}

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
        return dict(rows[0][0]) if rows else {"id": dataset_id, "name": name}

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
        return dict(rows[0][0]) if rows else {"id": version_id, "dataset_id": dataset_id, "version": version}

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
        return dict(rows[0][0]) if rows else {"id": config_id, "config_hash": config_hash}

    # -----------------------------------------------------------------------
    # Experiments
    # -----------------------------------------------------------------------

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
                      e.duration_seconds = $duration_seconds,
                      e.compute_cost_gpu_hours = $compute_cost_gpu_hours,
                      e.random_seed = $random_seed,
                      e.source_reference = $source_reference,
                      e.description = $description,
                      e.notes = $notes,
                      e.dataset_version_id = $dataset_version_id
        ON MATCH SET  e.name = $name,
                      e.model_family = $model_family,
                      e.status = $status,
                      e.duration_seconds = coalesce($duration_seconds, e.duration_seconds),
                      e.compute_cost_gpu_hours = coalesce($compute_cost_gpu_hours, e.compute_cost_gpu_hours),
                      e.random_seed = coalesce($random_seed, e.random_seed),
                      e.source_reference = coalesce($source_reference, e.source_reference),
                      e.description = coalesce($description, e.description),
                      e.notes = coalesce($notes, e.notes),
                      e.dataset_version_id = coalesce($dataset_version_id, e.dataset_version_id)
        WITH e
        MATCH (p:Project {id: $project_id})
        MERGE (p)-[:CONTAINS]->(e)
        RETURN properties(e)
        """
        params = {
            "id": exp_id,
            "project_id": project_id,
            "name": exp_data.get("name", exp_id),
            "model_family": exp_data.get("model_family", "generic"),
            "status": exp_data.get("status", "completed"),
            "quality_status": exp_data.get("quality_status", "valid"),
            "created_at": exp_data.get("created_at") or now,
            "duration_seconds": exp_data.get("duration_seconds"),
            "compute_cost_gpu_hours": exp_data.get("compute_cost_gpu_hours"),
            "random_seed": exp_data.get("random_seed"),
            "source_reference": exp_data.get("source_reference"),
            "description": exp_data.get("description"),
            "notes": exp_data.get("notes"),
            "dataset_version_id": dataset_version_id
        }
        rows = self.client.execute_query(cypher_exp, params)

        # Link configuration
        if config_id:
            link_config_cypher = """
            MATCH (e:Experiment {id: $exp_id}), (c:Configuration {id: $config_id})
            MERGE (e)-[:USES_CONFIG]->(c)
            """
            self.client.execute_query(link_config_cypher, {"exp_id": exp_id, "config_id": config_id})

        # Link dataset version
        if dataset_version_id:
            link_dv_cypher = """
            MATCH (e:Experiment {id: $exp_id}), (dv:DatasetVersion {id: $dv_id})
            MERGE (e)-[:USES_DATASET_VERSION]->(dv)
            """
            self.client.execute_query(link_dv_cypher, {"exp_id": exp_id, "dv_id": dataset_version_id})

        # Link parent experiment
        if parent_id and parent_id != exp_id:
            link_parent_cypher = """
            MATCH (p:Experiment {id: $parent_id}), (c:Experiment {id: $child_id})
            MERGE (p)-[:PARENT_OF]->(c)
            """
            self.client.execute_query(link_parent_cypher, {"parent_id": parent_id, "child_id": exp_id})

        return dict(rows[0][0]) if rows else {"id": exp_id}

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
        value: float,
        direction: str = "maximize",
        evaluation_split: Optional[str] = "validation",
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
            "value": float(value),
            "direction": direction,
            "evaluation_split": evaluation_split or "validation",
            "evaluation_protocol": evaluation_protocol or ""
        })
        return dict(rows[0][0]) if rows else {"id": metric_id, "name": name, "value": value}

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
        return dict(rows[0][0]) if rows else {"id": hypothesis_id, "text": text}

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
        return dict(rows[0][0]) if rows else {"id": evidence_id}

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
        return dict(rows[0][0]) if rows else {"id": claim_id, "text": text}

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
        status: str = "active",
        created_by: str = "researcher"
    ) -> Dict[str, Any]:
        """Creates or updates a Correction node."""
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
        return dict(rows[0][0]) if rows else {"id": correction_id}

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
        status: str = "NEEDS_REVIEW"
    ) -> Dict[str, Any]:
        """Creates a ReviewItem node and links (ReviewItem)-[:CONCERNS]->(Experiment)."""
        now = datetime.now(timezone.utc).isoformat()
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
            "status": status,
            "created_at": now
        })
        return dict(rows[0][0]) if rows else {"id": review_item_id, "experiment_id": experiment_id}

    def get_review_queue(self, project_id: Optional[str] = None) -> List[ReviewItemResponse]:
        """Retrieves active review items, optionally filtered by project."""
        if project_id:
            cypher = """
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(e:Experiment)<-[:CONCERNS]-(ri:ReviewItem)
            WHERE ri.status = 'NEEDS_REVIEW'
            RETURN ri.id, e.id, ri.reason, ri.status, ri.created_at
            ORDER BY ri.created_at DESC
            """
            rows = self.client.execute_query(cypher, {"project_id": project_id})
        else:
            cypher = """
            MATCH (ri:ReviewItem)-[:CONCERNS]->(e:Experiment)
            WHERE ri.status = 'NEEDS_REVIEW'
            RETURN ri.id, e.id, ri.reason, ri.status, ri.created_at
            ORDER BY ri.created_at DESC
            """
            rows = self.client.execute_query(cypher)

        items = []
        for row in rows:
            items.append(ReviewItemResponse(
                id=row[0],
                experiment_id=row[1],
                reason=row[2],
                status=row[3],
                created_at=str(row[4])
            ))
        return items

    def resolve_review_item(
        self,
        review_item_id: str,
        status: str = "RESOLVED",
        notes: Optional[str] = None,
        resolved_by: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Resolves a review item and marks its resolution."""
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MATCH (ri:ReviewItem {id: $id})
        SET ri.status = $status, ri.resolved_at = $resolved_at,
            ri.resolution_notes = $notes, ri.resolved_by = $resolved_by
        RETURN properties(ri)
        """
        rows = self.client.execute_query(cypher, {
            "id": review_item_id,
            "status": status,
            "resolved_at": now,
            "notes": notes or "",
            "resolved_by": resolved_by or "researcher"
        })
        return dict(rows[0][0]) if rows else None

    # -----------------------------------------------------------------------
    # Retrieval and History Queries
    # -----------------------------------------------------------------------

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentDetail]:
        """Retrieves full detail of an experiment, its configuration, metrics, dataset, and relationships."""
        cypher = """
        MATCH (e:Experiment {id: $exp_id})
        OPTIONAL MATCH (e)-[:USES_CONFIG]->(c:Configuration)
        OPTIONAL MATCH (e)-[:USES_DATASET_VERSION]->(dv:DatasetVersion)-[:VERSION_OF]->(d:Dataset)
        OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
        OPTIONAL MATCH (parent:Experiment)-[:PARENT_OF]->(e)
        OPTIONAL MATCH (e)-[:PARENT_OF]->(child:Experiment)
        OPTIONAL MATCH (e)-[:BASED_ON]->(claim:Claim)
        OPTIONAL MATCH (ri:ReviewItem)-[:CONCERNS]->(e)
        RETURN properties(e) AS exp_props,
               c.canonical_json AS config_json,
               d.name AS dataset_name,
               dv.version AS dataset_version,
               collect(DISTINCT properties(m)) AS metrics_props,
               collect(DISTINCT parent.id) AS parent_ids,
               collect(DISTINCT child.id) AS child_ids,
               collect(DISTINCT properties(claim)) AS claims_props,
               collect(DISTINCT properties(ri)) AS review_props
        """
        rows = self.client.execute_query(cypher, {"exp_id": experiment_id})
        if not rows or not rows[0][0]:
            return None

        exp_props = dict(rows[0][0])
        config_json = rows[0][1]
        dataset_name = rows[0][2]
        dataset_version = rows[0][3]
        metrics_raw = rows[0][4] or []
        parent_ids = [p for p in (rows[0][5] or []) if p]
        child_ids = [c for c in (rows[0][6] or []) if c]
        claims_raw = rows[0][7] or []
        review_raw = rows[0][8] or []

        parsed_config = None
        if config_json:
            try:
                parsed_config = json.loads(config_json)
            except Exception:
                parsed_config = {"raw": config_json}

        metrics = []
        for m in metrics_raw:
            if m:
                m_dict = dict(m)
                metrics.append(MetricModel(
                    name=m_dict.get("name", ""),
                    value=float(m_dict.get("value", 0.0)),
                    direction=m_dict.get("direction", "maximize"),
                    evaluation_split=m_dict.get("evaluation_split"),
                    evaluation_protocol=m_dict.get("evaluation_protocol")
                ))

        claims = [dict(c) for c in claims_raw if c]
        review_items = [dict(r) for r in review_raw if r]

        return ExperimentDetail(
            id=exp_props["id"],
            project_id=exp_props["project_id"],
            name=exp_props.get("name", exp_props["id"]),
            model_family=exp_props.get("model_family", "generic"),
            dataset_version_id=exp_props.get("dataset_version_id"),
            status=exp_props.get("status", "completed"),
            quality_status=exp_props.get("quality_status", "valid"),
            metrics=metrics,
            compute_cost_gpu_hours=exp_props.get("compute_cost_gpu_hours"),
            created_at=exp_props.get("created_at"),
            description=exp_props.get("description"),
            configuration=parsed_config,
            dataset_name=dataset_name,
            dataset_version=dataset_version,
            duration_seconds=exp_props.get("duration_seconds"),
            random_seed=exp_props.get("random_seed"),
            source_reference=exp_props.get("source_reference"),
            notes=exp_props.get("notes"),
            parent_experiment_ids=parent_ids,
            child_experiment_ids=child_ids,
            claims=claims,
            review_items=review_items
        )

    def list_experiments(
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
            metrics_raw = row[1] or []
            metrics = []
            for m in metrics_raw:
                if m:
                    m_dict = dict(m)
                    metrics.append(MetricModel(
                        name=m_dict.get("name", ""),
                        value=float(m_dict.get("value", 0.0)),
                        direction=m_dict.get("direction", "maximize"),
                        evaluation_split=m_dict.get("evaluation_split"),
                        evaluation_protocol=m_dict.get("evaluation_protocol")
                    ))
            summaries.append(ExperimentSummary(
                id=exp_props["id"],
                project_id=exp_props["project_id"],
                name=exp_props.get("name", exp_props["id"]),
                model_family=exp_props.get("model_family", "generic"),
                dataset_version_id=exp_props.get("dataset_version_id"),
                status=exp_props.get("status", "completed"),
                quality_status=exp_props.get("quality_status", "valid"),
                metrics=metrics,
                compute_cost_gpu_hours=exp_props.get("compute_cost_gpu_hours"),
                created_at=exp_props.get("created_at")
            ))
        return summaries

    # -----------------------------------------------------------------------
    # Lineage and Traversal Queries
    # -----------------------------------------------------------------------

    def get_experiment_lineage(self, experiment_id: str, max_depth: int = 5) -> ExperimentLineageResponse:
        """
        Traverses upward (ancestors) and downward (descendants) through PARENT_OF relationships
        up to max_depth, returning a complete directed lineage graph structure.
        """
        # Fetch root
        root_exp = self.get_experiment(experiment_id)
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
        """
        # Experiments sharing dataset version
        cypher_dataset = """
        MATCH (target:Experiment {id: $exp_id})-[:USES_DATASET_VERSION]->(dv:DatasetVersion)<-[:USES_DATASET_VERSION]-(e:Experiment)
        WHERE e.id <> target.id
        RETURN DISTINCT e.id
        """
        same_dataset = [r[0] for r in self.client.execute_query(cypher_dataset, {"exp_id": experiment_id})]

        # Experiments sharing configuration
        cypher_config = """
        MATCH (target:Experiment {id: $exp_id})-[:USES_CONFIG]->(c:Configuration)<-[:USES_CONFIG]-(e:Experiment)
        WHERE e.id <> target.id
        RETURN DISTINCT e.id
        """
        same_config = [r[0] for r in self.client.execute_query(cypher_config, {"exp_id": experiment_id})]

        # Experiments sharing parent experiment
        cypher_siblings = """
        MATCH (p:Experiment)-[:PARENT_OF]->(target:Experiment {id: $exp_id})
        MATCH (p)-[:PARENT_OF]->(e:Experiment)
        WHERE e.id <> target.id
        RETURN DISTINCT e.id
        """
        siblings = [r[0] for r in self.client.execute_query(cypher_siblings, {"exp_id": experiment_id})]

        # Explicitly compared
        cypher_compared = """
        MATCH (target:Experiment {id: $exp_id})-[:COMPARED_WITH]-(e:Experiment)
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

    def get_project_graph_summary(self, project_id: str) -> ProjectGraphSummary:
        """Returns node and relationship counts for a specific project."""
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
            edges=edges_list
        )
