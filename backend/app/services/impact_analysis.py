"""
Correction Impact Analysis Service for RUNWISE.
Conforms strictly to RUNWISE PRD Section 5 (FR-6), Section 8, Section 9, and Section 18.

Implements graph-backed correction propagation:
1. Records Correction node and provenance.
2. Identifies affected source node (DatasetVersion, Evidence, Claim, Metric, Experiment).
3. Bounded graph traversal to discover directly and transitively dependent experiments and claims.
4. Identifies unaffected experiments in the same project for comparison.
5. Returns explicit dependency paths explaining why each entity is impacted.
6. Updates quality statuses conservatively ('needs_review', 'disputed', 'unreliable') without deleting history.
7. Generates ReviewItem entries in the revalidation queue.
8. Ensures idempotency against duplicate correction requests.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone

from app.graph.repository import GraphRepository
from app.graph.schema import NodeLabel, RelType
from app.models.contracts import (
    CorrectionImpactResponse,
    DependencyPath,
    CorrectionCreate
)

logger = logging.getLogger(__name__)


class ImpactAnalysisService:
    def __init__(self, repository: Optional[GraphRepository] = None):
        self.repository = repository or GraphRepository()

    def analyze_and_apply_correction(
        self,
        correction: CorrectionCreate,
        project_id: Optional[str] = None,
        correction_id: Optional[str] = None
    ) -> CorrectionImpactResponse:
        """
        Executes the full graph-backed evidence correction workflow.
        Returns a structured CorrectionImpactResponse with explicit dependency paths.
        """
        now = datetime.now(timezone.utc).isoformat()
        source_id = correction.source_id.strip()
        source_type = correction.source_type.strip().lower()
        desc = correction.description.strip()
        created_by = correction.created_by or "researcher"

        corr_id = correction_id or f"corr_{source_id}_{int(datetime.now().timestamp())}"

        # 1. Record Correction node in graph
        self.repository.create_correction(
            correction_id=corr_id,
            description=desc,
            status="active",
            created_by=created_by
        )

        # 2. Determine target label & link correction to source
        target_label = NodeLabel.DATASET_VERSION
        if source_type in ("dataset_version", "datasetversion", "dataset"):
            target_label = NodeLabel.DATASET_VERSION
            self.repository.update_dataset_version_quality_status(source_id, "unreliable")
        elif source_type in ("evidence",):
            target_label = NodeLabel.EVIDENCE
        elif source_type in ("claim",):
            target_label = NodeLabel.CLAIM
            self.repository.update_claim_status(source_id, "disputed")
        elif source_type in ("experiment", "run"):
            target_label = NodeLabel.EXPERIMENT
            self.repository.update_experiment_quality_status(source_id, "needs_review")
        elif source_type in ("metric",):
            target_label = NodeLabel.METRIC

        # Link (Correction)-[:AFFECTS|CHALLENGES]->(source)
        try:
            self.repository.link_correction_target(corr_id, source_id, target_label)
        except Exception as e:
            logger.warning(f"Could not link correction to target node {source_id}: {e}")

        # 3. Traverse dependencies to find directly affected experiments
        direct_exp_ids: Set[str] = set()
        paths: List[DependencyPath] = []

        if target_label == NodeLabel.DATASET_VERSION:
            # Experiments using this dataset version
            cypher_direct = """
            MATCH (e:Experiment)-[:USES_DATASET_VERSION]->(dv:DatasetVersion {id: $source_id})
            RETURN e.id
            """
            rows = self.repository.client.execute_query(cypher_direct, {"source_id": source_id})
            for r in rows:
                exp_id = r[0]
                direct_exp_ids.add(exp_id)
                paths.append(DependencyPath(
                    nodes=[source_id, exp_id],
                    relationship_types=[RelType.USES_DATASET_VERSION],
                    reason=f"Experiment '{exp_id}' directly trained or evaluated on affected dataset version '{source_id}'."
                ))

        elif target_label == NodeLabel.EVIDENCE:
            # Evidence -> Claim -> Experiment
            cypher_ev = """
            MATCH (ev:Evidence {id: $source_id})<-[:SUPPORTED_BY]-(cl:Claim)<-[:BASED_ON]-(e:Experiment)
            RETURN e.id, cl.id
            """
            rows = self.repository.client.execute_query(cypher_ev, {"source_id": source_id})
            for r in rows:
                exp_id, claim_id = r[0], r[1]
                direct_exp_ids.add(exp_id)
                paths.append(DependencyPath(
                    nodes=[source_id, claim_id, exp_id],
                    relationship_types=[RelType.SUPPORTED_BY, RelType.BASED_ON],
                    reason=f"Experiment '{exp_id}' based on claim '{claim_id}' supported by challenged evidence '{source_id}'."
                ))

        elif target_label == NodeLabel.CLAIM:
            # Claim -> Experiment
            cypher_cl = """
            MATCH (cl:Claim {id: $source_id})<-[:BASED_ON]-(e:Experiment)
            RETURN e.id
            """
            rows = self.repository.client.execute_query(cypher_cl, {"source_id": source_id})
            for r in rows:
                exp_id = r[0]
                direct_exp_ids.add(exp_id)
                paths.append(DependencyPath(
                    nodes=[source_id, exp_id],
                    relationship_types=[RelType.BASED_ON],
                    reason=f"Experiment '{exp_id}' depends on invalidated claim '{source_id}'."
                ))

        elif target_label == NodeLabel.EXPERIMENT:
            direct_exp_ids.add(source_id)
            paths.append(DependencyPath(
                nodes=[source_id],
                relationship_types=[],
                reason=f"Experiment '{source_id}' was directly corrected or marked unreliable."
            ))

        elif target_label == NodeLabel.METRIC:
            cypher_m = """
            MATCH (e:Experiment)-[:PRODUCES_METRIC]->(m:Metric {id: $source_id})
            RETURN e.id
            """
            rows = self.repository.client.execute_query(cypher_m, {"source_id": source_id})
            for r in rows:
                exp_id = r[0]
                direct_exp_ids.add(exp_id)
                paths.append(DependencyPath(
                    nodes=[source_id, exp_id],
                    relationship_types=[RelType.PRODUCES_METRIC],
                    reason=f"Experiment '{exp_id}' produced corrected metric '{source_id}'."
                ))

        # 4. Traverse downstream lineage to find transitively affected experiments
        transitive_exp_ids: Set[str] = set()
        for d_id in list(direct_exp_ids):
            cypher_downstream = """
            MATCH path = (parent:Experiment {id: $direct_id})-[:PARENT_OF*1..5]->(downstream:Experiment)
            RETURN downstream.id,
                   [n in nodes(path) | n.id] AS node_ids,
                   [r in relationships(path) | type(r)] AS rel_types
            """
            rows = self.repository.client.execute_query(cypher_downstream, {"direct_id": d_id})
            for r in rows:
                downstream_id = r[0]
                if downstream_id not in direct_exp_ids:
                    transitive_exp_ids.add(downstream_id)
                    p_nodes = [source_id] + r[1]
                    p_rels = [RelType.USES_DATASET_VERSION] + r[2] if target_label == NodeLabel.DATASET_VERSION else r[2]
                    paths.append(DependencyPath(
                        nodes=p_nodes,
                        relationship_types=p_rels,
                        reason=f"Experiment '{downstream_id}' transitively inherits from parent experiment '{d_id}'."
                    ))

        all_affected_exp_ids = direct_exp_ids | transitive_exp_ids

        # 5. Find potentially affected Claims derived from any affected experiment
        affected_claim_ids: Set[str] = set()
        for aff_id in all_affected_exp_ids:
            cypher_claims = """
            MATCH path = (e:Experiment {id: $exp_id})-[:BASED_ON]->(cl:Claim)
            RETURN cl.id, [n in nodes(path) | n.id] AS node_ids, [r in relationships(path) | type(r)] AS rel_types
            """
            rows = self.repository.client.execute_query(cypher_claims, {"exp_id": aff_id})
            for r in rows:
                cid = r[0]
                affected_claim_ids.add(cid)
                paths.append(DependencyPath(
                    nodes=[source_id, aff_id, cid],
                    relationship_types=[RelType.USES_DATASET_VERSION, RelType.BASED_ON] if target_label == NodeLabel.DATASET_VERSION else [RelType.BASED_ON],
                    reason=f"Claim '{cid}' is based on affected experiment '{aff_id}'."
                ))

        # 6. Find unaffected experiments in the same project
        unaffected_exp_ids: List[str] = []
        if project_id:
            cypher_all = """
            MATCH (p:Project {id: $project_id})-[:CONTAINS]->(e:Experiment)
            RETURN e.id
            """
            rows = self.repository.client.execute_query(cypher_all, {"project_id": project_id})
            all_project_exps = {r[0] for r in rows}
            unaffected_exp_ids = sorted(list(all_project_exps - all_affected_exp_ids))

        # 7. Create ReviewItem entries and update experiment quality statuses
        review_item_ids: List[str] = []
        for exp_id in all_affected_exp_ids:
            # Conservative status: mark as needs_review, do not destroy historical records
            self.repository.update_experiment_quality_status(exp_id, "needs_review")

            is_direct = exp_id in direct_exp_ids
            reason_str = (
                f"Directly dependent on corrected {source_type} '{source_id}': {desc}"
                if is_direct else
                f"Transitively derived from affected predecessor in lineage: {desc}"
            )
            ri_id = f"review_{corr_id}_{exp_id}"
            self.repository.create_review_item(
                review_item_id=ri_id,
                experiment_id=exp_id,
                reason=reason_str,
                status="NEEDS_REVIEW"
            )
            review_item_ids.append(ri_id)

        # Also mark affected claims as disputed
        for cl_id in affected_claim_ids:
            self.repository.update_claim_status(cl_id, "disputed")

        summary_text = (
            f"Correction recorded for {source_type} '{source_id}': {desc}. "
            f"Identified {len(direct_exp_ids)} directly affected and {len(transitive_exp_ids)} "
            f"transitively affected experiments requiring revalidation."
        )

        return CorrectionImpactResponse(
            correction_id=corr_id,
            source_id=source_id,
            summary=summary_text,
            directly_affected_experiment_ids=sorted(list(direct_exp_ids)),
            potentially_affected_claim_ids=sorted(list(affected_claim_ids)),
            unaffected_experiment_ids=unaffected_exp_ids,
            review_item_ids=sorted(review_item_ids),
            paths=paths
        )

    def get_correction_impact(self, correction_id: str) -> Optional[CorrectionImpactResponse]:
        """
        Retrieves the impact analysis and paths for an existing correction.
        """
        cypher = """
        MATCH (co:Correction {id: $corr_id})
        OPTIONAL MATCH (co)-[:AFFECTS|CHALLENGES]->(source)
        OPTIONAL MATCH (ri:ReviewItem)-[:CONCERNS]->(e:Experiment)
        WHERE ri.id STARTS WITH $prefix
        RETURN properties(co), source.id, labels(source)[0], collect(DISTINCT e.id), collect(DISTINCT ri.id)
        """
        prefix = f"review_{correction_id}"
        rows = self.repository.client.execute_query(cypher, {"corr_id": correction_id, "prefix": prefix})
        if not rows or not rows[0][0]:
            return None

        co_props = dict(rows[0][0])
        source_id = rows[0][1] or "unknown"
        source_label = rows[0][2] or "Unknown"
        affected_exps = sorted(list(rows[0][3] or []))
        review_items = sorted(list(rows[0][4] or []))

        return CorrectionImpactResponse(
            correction_id=correction_id,
            source_id=source_id,
            summary=co_props.get("description", ""),
            directly_affected_experiment_ids=affected_exps,
            potentially_affected_claim_ids=[],
            unaffected_experiment_ids=[],
            review_item_ids=review_items,
            paths=[]
        )
