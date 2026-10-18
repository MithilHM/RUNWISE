"""
Correction Impact Analysis Service for RUNWISE.
Conforms strictly to RUNWISE PRD Section 5 (FR-6), Section 8, Section 9, and Section 18.

Implements graph-backed correction propagation:
1. Records Correction node and provenance.
2. Identifies affected source node (DatasetVersion, Evidence, Claim, Metric, Experiment).
3. Bounded, *project scoped* graph traversal to discover directly and transitively
   dependent experiments and claims. Every traversal carries the project scope in
   Cypher, so a correction raised in one project can never mark experiments in
   another project (PRD 8: corrections are project scoped).
4. Identifies unaffected experiments in the same project for comparison.
5. Returns explicit dependency paths explaining why each entity is impacted,
   de-duplicated so one entity is never explained twice (PRD 8 line 341:
   "Prevent duplicate results").
6. Updates quality statuses conservatively ('needs_review', 'disputed',
   'unreliable') without deleting history.
7. Generates ReviewItem entries in the revalidation queue, linked to the
   correction through a real ``RAISED_BY`` relationship.
8. Ensures idempotency against duplicate correction requests.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
from datetime import datetime, timezone

from app.graph.repository import GraphRepository
from app.graph.schema import NodeLabel, RelType, ReviewItemStatus
from app.models.contracts import (
    CorrectionImpactResponse,
    DependencyPath,
    CorrectionCreate
)

logger = logging.getLogger(__name__)

# Bounded lineage traversal depth (PRD 8: bounded graph traversal).
MAX_LINEAGE_DEPTH = 5

# The only accepted correction source types, and the node label each one targets.
# Aliases are normalised to the canonical name that is stored on the node.
SOURCE_TYPE_ALIASES: Dict[str, str] = {
    "dataset_version": "dataset_version",
    "datasetversion": "dataset_version",
    "dataset-version": "dataset_version",
    "dataset": "dataset_version",
    "evidence": "evidence",
    "claim": "claim",
    "experiment": "experiment",
    "run": "experiment",
    "metric": "metric",
}

SUPPORTED_SOURCE_TYPES: Tuple[str, ...] = ("dataset_version", "evidence", "claim", "experiment", "metric")

# Canonical source_type -> target node label.
SOURCE_TYPE_TARGETS: Dict[str, str] = {
    "dataset_version": NodeLabel.DATASET_VERSION,
    "evidence": NodeLabel.EVIDENCE,
    "claim": NodeLabel.CLAIM,
    "experiment": NodeLabel.EXPERIMENT,
    "metric": NodeLabel.METRIC,
}

# Cypher used to derive the owning project(s) of a source node when the caller
# did not state one. Every pattern starts from a Project so the scope is a graph
# fact, not an assumption.
SCOPE_QUERIES: Dict[str, str] = {
    NodeLabel.DATASET_VERSION: """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)
              -[:USES_DATASET_VERSION]->(dv:DatasetVersion {id: $source_id})
        RETURN DISTINCT p.id
    """,
    NodeLabel.EVIDENCE: """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)-[:BASED_ON]->(c:Claim)
              -[:SUPPORTED_BY]->(ev:Evidence {id: $source_id})
        RETURN DISTINCT p.id
    """,
    NodeLabel.CLAIM: """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)-[:BASED_ON]->(c:Claim {id: $source_id})
        RETURN DISTINCT p.id
    """,
    NodeLabel.METRIC: """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)-[:PRODUCES_METRIC]->(m:Metric {id: $source_id})
        RETURN DISTINCT p.id
    """,
    NodeLabel.EXPERIMENT: """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment {id: $source_id})
        RETURN DISTINCT p.id
    """,
}

# Direct-impact traversal per target label. Each one is anchored on a Project so
# the project predicate lives in the query, not in Python filtering.
DIRECT_TRAVERSALS: Dict[str, Tuple[str, str]] = {
    # label -> (cypher, reason template)
    NodeLabel.DATASET_VERSION: (
        """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)
              -[:USES_DATASET_VERSION]->(dv:DatasetVersion {id: $source_id})
        WHERE p.id IN $project_ids
        RETURN DISTINCT e.id
        """,
        "Experiment '{exp_id}' directly trained or evaluated on affected dataset version '{source_id}'.",
    ),
    NodeLabel.EVIDENCE: (
        """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)-[:BASED_ON]->(c:Claim)
              -[:SUPPORTED_BY]->(ev:Evidence {id: $source_id})
        WHERE p.id IN $project_ids
        RETURN DISTINCT e.id, c.id
        """,
        "Experiment '{exp_id}' based on claim '{claim_id}' supported by challenged evidence '{source_id}'.",
    ),
    NodeLabel.CLAIM: (
        """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)
              -[:BASED_ON]->(c:Claim {id: $source_id})
        WHERE p.id IN $project_ids
        RETURN DISTINCT e.id
        """,
        "Experiment '{exp_id}' depends on invalidated claim '{source_id}'.",
    ),
    NodeLabel.EXPERIMENT: (
        """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment {id: $source_id})
        WHERE p.id IN $project_ids
        RETURN DISTINCT e.id
        """,
        "Experiment '{exp_id}' was directly corrected or marked unreliable.",
    ),
    NodeLabel.METRIC: (
        """
        MATCH (p:Project)-[:CONTAINS]->(e:Experiment)
              -[:PRODUCES_METRIC]->(m:Metric {id: $source_id})
        WHERE p.id IN $project_ids
        RETURN DISTINCT e.id
        """,
        "Experiment '{exp_id}' produced corrected metric '{source_id}'.",
    ),
}

# Relationship that links a source node to the experiment that read/wrote it.
# An Experiment source has no such relationship: the corrected run *is* the
# experiment, so its dependency path is the node itself and the lineage prefix
# for downstream paths is empty.
SOURCE_TO_EXPERIMENT_REL: Dict[str, str] = {
    NodeLabel.DATASET_VERSION: RelType.USES_DATASET_VERSION,
    NodeLabel.CLAIM: RelType.BASED_ON,
    NodeLabel.METRIC: RelType.PRODUCES_METRIC,
}

DOWNSTREAM_CYPHER = """
MATCH path = (parent:Experiment {{id: $direct_id}})-[:PARENT_OF*1..{depth}]->(downstream:Experiment)
WHERE downstream.project_id IN $project_ids
RETURN DISTINCT downstream.id,
       [n in nodes(path) | n.id] AS node_ids,
       [r in relationships(path) | type(r)] AS rel_types
"""

CLAIMS_OF_AFFECTED_CYPHER = """
MATCH (p:Project)-[:CONTAINS]->(e:Experiment {id: $exp_id})-[:BASED_ON]->(c:Claim)
WHERE p.id IN $project_ids
RETURN DISTINCT c.id
"""

PROJECT_EXPERIMENTS_CYPHER = """
MATCH (p:Project)-[:CONTAINS]->(e:Experiment)
WHERE p.id IN $project_ids
RETURN DISTINCT e.id
"""


def normalise_source_type(raw_source_type: Optional[str]) -> str:
    """
    Validates a correction source type and returns its canonical spelling.

    An unrecognised type is rejected instead of being coerced to
    ``dataset_version``: silently treating an unknown id as a dataset version
    would mark an unrelated node ``unreliable``.
    """
    key = (raw_source_type or "").strip().lower().replace(" ", "_")
    canonical = SOURCE_TYPE_ALIASES.get(key)
    if canonical is None:
        raise ValueError(
            f"Unsupported correction source_type {raw_source_type!r}. "
            f"Supported source types: {', '.join(SUPPORTED_SOURCE_TYPES)}."
        )
    return canonical


class _PathCollector:
    """
    Collects dependency paths without duplicates.

    Two guards, both required by PRD 8 ("prevent duplicate results"):

    * an identical ``(nodes, relationship_types)`` path is kept once, even when
      several traversals rediscover it;
    * an optional *group* keeps exactly one path per explained entity, so the
      same downstream experiment reachable from two parents produces a single
      (shortest) explanation instead of one per parent.
    """

    def __init__(self) -> None:
        self._paths: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], DependencyPath] = {}
        self._groups: Set[Tuple[str, str]] = set()
        self.duplicate_count = 0

    def add(
        self,
        nodes: Sequence[str],
        relationship_types: Sequence[str],
        reason: str,
        group: Optional[Tuple[str, str]] = None,
    ) -> bool:
        key = (tuple(nodes), tuple(relationship_types))
        if key in self._paths:
            self.duplicate_count += 1
            return False
        if group is not None:
            group_key = (group[0], str(group[1]))
            if group_key in self._groups:
                self.duplicate_count += 1
                return False
            self._groups.add(group_key)
        self._paths[key] = DependencyPath(
            nodes=list(nodes),
            relationship_types=list(relationship_types),
            reason=reason,
        )
        return True

    def paths(self) -> List[DependencyPath]:
        return list(self._paths.values())


class ImpactAnalysisService:
    def __init__(self, repository: Optional[GraphRepository] = None):
        self.repository = repository or GraphRepository()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        source_id = (correction.source_id or "").strip()
        if not source_id:
            raise ValueError("A correction requires a non-empty 'source_id'.")

        source_type = normalise_source_type(correction.source_type)
        target_label = SOURCE_TYPE_TARGETS[source_type]
        desc = (correction.description or "").strip()
        created_by = correction.created_by or "researcher"

        corr_id = correction_id or self._new_correction_id(source_id)
        warnings: List[str] = []

        # 1. Record the Correction node and its provenance.
        self.repository.create_correction(
            correction_id=corr_id,
            description=desc,
            status="open",
            created_by=created_by
        )
        self._annotate_correction(
            corr_id,
            project_id=project_id,
            source_id=source_id,
            source_type=source_type,
            evidence_reference=desc,
        )

        # 2. Mark the corrected node itself, conservatively.
        warnings.extend(self._mark_source_reliability(source_id, target_label))

        # 3. Link (Correction)-[:AFFECTS|CHALLENGES]->(source).
        target_exists = self._target_exists(source_id, target_label)
        if not target_exists:
            warnings.append(
                f"Corrected {source_type} '{source_id}' does not exist in the graph; "
                "no downstream impact was recorded."
            )
        try:
            self.repository.link_correction_target(corr_id, source_id, target_label)
        except Exception as e:
            warnings.append(f"Could not link correction to target node '{source_id}': {e}")
            logger.warning(f"Could not link correction to target node {source_id}: {e}")

        # 4. Resolve the project scope. A stated project wins; otherwise the
        #    scope is derived from the graph, never assumed to be global.
        project_ids = self._resolve_scope(project_id, source_id, target_label)
        if not project_ids:
            warnings.append(
                f"No project scope could be resolved for {source_type} '{source_id}'; "
                "the correction is recorded but impacts nothing."
            )
            logger.warning(
                "Correction %s: no project scope resolved for %s '%s'.",
                corr_id, source_type, source_id,
            )

        # 5. Traverse the (scoped) graph.
        analysis = self._analyze(
            source_id=source_id,
            target_label=target_label,
            project_ids=project_ids,
        )

        direct_exp_ids: Set[str] = set(analysis["direct_experiment_ids"])
        transitive_exp_ids: Set[str] = set(analysis["indirect_experiment_ids"])
        affected_claim_ids: Set[str] = set(analysis["potentially_affected_claim_ids"])
        paths: List[DependencyPath] = analysis["paths"]
        all_affected_exp_ids = direct_exp_ids | transitive_exp_ids

        # 6. Unaffected experiments in the same project scope.
        unaffected_exp_ids: List[str] = []
        if project_ids:
            rows = self.repository.client.execute_query(
                PROJECT_EXPERIMENTS_CYPHER, {"project_ids": sorted(project_ids)}
            )
            scope_exps = {r[0] for r in rows if r and r[0]}
            unaffected_exp_ids = sorted(scope_exps - all_affected_exp_ids)

        # 7. ReviewItem entries + conservative experiment status updates.
        review_item_ids: List[str] = []
        for exp_id in sorted(all_affected_exp_ids):
            updated = self.repository.update_experiment_quality_status(exp_id, "needs_review")
            if not updated:
                warnings.append(f"Experiment '{exp_id}' could not be marked needs_review (node missing).")
                logger.warning(
                    "Correction %s: experiment '%s' was selected as impacted but could not be updated.",
                    corr_id, exp_id,
                )
                continue

            is_direct = exp_id in direct_exp_ids
            reason_str = (
                f"Directly dependent on corrected {source_type} '{source_id}': {desc}"
                if is_direct else
                f"Transitively derived from affected predecessor in lineage: {desc}"
            )
            ri_id = f"review_{corr_id}_{exp_id}"
            self._create_review_item(ri_id, exp_id, corr_id, project_ids, reason_str)
            review_item_ids.append(ri_id)

        # 8. Dependent claims are disputed, never deleted.
        for cl_id in sorted(affected_claim_ids):
            if not self.repository.update_claim_status(cl_id, "disputed"):
                warnings.append(f"Claim '{cl_id}' could not be marked disputed (node missing).")
                logger.warning(
                    "Correction %s: claim '%s' selected as impacted but could not be updated.",
                    corr_id, cl_id,
                )

        summary_text = (
            f"Correction recorded for {source_type} '{source_id}': {desc}. "
            f"Identified {len(direct_exp_ids)} directly affected and {len(transitive_exp_ids)} "
            f"transitively affected experiments in scope "
            f"{sorted(project_ids) if project_ids else '[]'} requiring revalidation."
        )

        snapshot = {
            "summary": summary_text,
            "project_ids": sorted(project_ids),
            "directly_affected_experiment_ids": sorted(direct_exp_ids),
            "indirectly_affected_experiment_ids": sorted(transitive_exp_ids),
            "potentially_affected_claim_ids": sorted(affected_claim_ids),
            "unaffected_experiment_ids": unaffected_exp_ids,
            "review_item_ids": sorted(review_item_ids),
            "paths": [
                {
                    "nodes": p.nodes,
                    "relationship_types": p.relationship_types,
                    "reason": p.reason,
                }
                for p in paths
            ],
            "warnings": warnings,
            "generated_at": now,
        }
        self._store_impact_snapshot(corr_id, snapshot)

        if warnings:
            logger.warning("Correction %s recorded with warnings: %s", corr_id, warnings)

        return CorrectionImpactResponse(
            correction_id=corr_id,
            source_id=source_id,
            source_type=source_type,
            summary=summary_text,
            directly_affected_experiment_ids=sorted(direct_exp_ids),
            indirectly_affected_experiment_ids=sorted(transitive_exp_ids),
            potentially_affected_claim_ids=sorted(affected_claim_ids),
            unaffected_experiment_ids=unaffected_exp_ids,
            review_item_ids=sorted(review_item_ids),
            paths=paths,
            generated_at=now,
        )

    def get_correction_impact(self, correction_id: str) -> Optional[CorrectionImpactResponse]:
        """
        Retrieves the recorded impact analysis and paths for an existing correction.

        Returns ``None`` for an unknown correction id instead of raising. The
        recorded snapshot is returned when present; corrections written before the
        snapshot existed are re-analysed read-only from the graph.
        """
        cypher = """
        MATCH (co:Correction {id: $corr_id})
        OPTIONAL MATCH (co)-[:AFFECTS|CHALLENGES]->(source)
        RETURN properties(co) AS co_props, source.id, labels(source)[0]
        """
        rows = self.repository.client.execute_query(cypher, {"corr_id": correction_id})
        if not rows or not rows[0] or not rows[0][0]:
            logger.info("No correction found for id '%s'.", correction_id)
            return None

        co_props = dict(rows[0][0])
        source_id = rows[0][1] or co_props.get("source_id") or "unknown"
        source_label = rows[0][2] or "Unknown"

        snapshot_raw = co_props.get("impact_snapshot")
        snapshot: Optional[Dict[str, Any]] = None
        if snapshot_raw:
            try:
                snapshot = json.loads(snapshot_raw) if isinstance(snapshot_raw, str) else dict(snapshot_raw)
            except (TypeError, ValueError) as e:
                logger.warning(f"Correction '{correction_id}' has an unreadable impact snapshot: {e}")
                snapshot = None

        if snapshot is None:
            snapshot = self._rebuild_snapshot(co_props, source_id, source_label)

        # Review items are found through the real relationship, never by an
        # id-prefix heuristic on ReviewItem.id.
        ri_rows = self.repository.client.execute_query(
            """
            MATCH (co:Correction {id: $corr_id})<-[:RAISED_BY]-(ri:ReviewItem)
            OPTIONAL MATCH (ri)-[:CONCERNS]->(e:Experiment)
            RETURN collect(DISTINCT ri.id), collect(DISTINCT e.id)
            """,
            {"corr_id": correction_id},
        )
        if ri_rows and ri_rows[0]:
            review_items = sorted(x for x in (ri_rows[0][0] or []) if x)
            concerned = sorted(x for x in (ri_rows[0][1] or []) if x)
        else:
            review_items = []
            concerned = []

        paths = [
            DependencyPath(
                nodes=list(p.get("nodes", [])),
                relationship_types=list(p.get("relationship_types", [])),
                reason=p.get("reason"),
            )
            for p in snapshot.get("paths", [])
        ]

        direct = sorted(snapshot.get("directly_affected_experiment_ids", []))
        indirect = sorted(snapshot.get("indirectly_affected_experiment_ids", []))
        for exp_id in concerned:
            # Backfill safety net: an impacted run without a recorded tier is
            # reported as indirect rather than silently dropped.
            if exp_id not in direct and exp_id not in indirect:
                indirect.append(exp_id)
        indirect.sort()

        return CorrectionImpactResponse(
            correction_id=correction_id,
            source_id=source_id,
            source_type=co_props.get("source_type"),
            summary=snapshot.get("summary") or co_props.get("description", ""),
            directly_affected_experiment_ids=direct,
            indirectly_affected_experiment_ids=indirect,
            potentially_affected_claim_ids=sorted(snapshot.get("potentially_affected_claim_ids", [])),
            unaffected_experiment_ids=sorted(snapshot.get("unaffected_experiment_ids", [])),
            review_item_ids=review_items,
            paths=paths,
            generated_at=snapshot.get("generated_at") or co_props.get("created_at"),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _new_correction_id(source_id: str) -> str:
        """
        Builds a collision-free correction id.

        A whole-second timestamp alone is not unique: two corrections raised in
        the same second MERGE onto the same node and silently overwrite each
        other. The uuid suffix makes the id unique while the readable
        ``corr_<source_id>_<timestamp>_<suffix>`` shape is preserved.
        """
        return f"corr_{source_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    def _annotate_correction(
        self,
        correction_id: str,
        project_id: Optional[str],
        source_id: str,
        source_type: str,
        evidence_reference: str,
    ) -> None:
        """Stores provenance on the Correction node (documented in NODE_SCHEMAS)."""
        cypher = """
        MATCH (co:Correction {id: $corr_id})
        SET co.source_id = $source_id,
            co.source_type = $source_type,
            co.project_id = $project_id,
            co.evidence_reference = $evidence_reference
        """
        self.repository.client.execute_query(cypher, {
            "corr_id": correction_id,
            "source_id": source_id,
            "source_type": source_type,
            "project_id": project_id or "",
            "evidence_reference": evidence_reference or "",
        })

    def _mark_source_reliability(self, source_id: str, target_label: str) -> List[str]:
        """
        Marks the corrected node conservatively and reports a missing target.

        The repository update helpers return ``False`` when the node is absent;
        that answer used to be discarded, which made a correction against a
        non-existent node look successful.
        """
        warnings: List[str] = []
        if target_label == NodeLabel.DATASET_VERSION:
            updated = self.repository.update_dataset_version_quality_status(
                source_id, "unreliable"
            )
            if not updated:
                warnings.append(
                    f"DatasetVersion '{source_id}' not found; quality_status was not set to 'unreliable'."
                )
        elif target_label == NodeLabel.CLAIM:
            if not self.repository.update_claim_status(source_id, "disputed"):
                warnings.append(f"Claim '{source_id}' not found; status was not set to 'disputed'.")
        elif target_label == NodeLabel.EXPERIMENT:
            if not self.repository.update_experiment_quality_status(source_id, "needs_review"):
                warnings.append(
                    f"Experiment '{source_id}' not found; quality_status was not set to 'needs_review'."
                )
        else:
            # Evidence and Metric carry no trust status in the current schema.
            logger.info(
                "Correction target %s '%s' has no quality_status property; "
                "only its dependents are updated.",
                target_label, source_id,
            )
        for w in warnings:
            logger.warning(w)
        return warnings

    def _target_exists(self, source_id: str, target_label: str) -> bool:
        rows = self.repository.client.execute_query(
            f"MATCH (n:{target_label} {{id: $source_id}}) RETURN count(n)",
            {"source_id": source_id},
        )
        return bool(rows and rows[0] and rows[0][0])

    def _resolve_scope(
        self,
        project_id: Optional[str],
        source_id: str,
        target_label: str,
    ) -> List[str]:
        """Resolves the project ids this correction is allowed to affect."""
        if project_id:
            return [project_id]

        cypher = SCOPE_QUERIES.get(target_label)
        if not cypher:
            return []

        try:
            rows = self.repository.client.execute_query(cypher, {"source_id": source_id})
            derived = sorted({r[0] for r in rows if r and r[0]})
        except Exception as e:
            logger.warning(f"Could not derive project scope for '{source_id}': {e}")
            return []

        if not derived and target_label == NodeLabel.EXPERIMENT:
            # Fall back to the denormalised project_id on the node itself.
            rows = self.repository.client.execute_query(
                "MATCH (e:Experiment {id: $source_id}) RETURN DISTINCT e.project_id",
                {"source_id": source_id},
            )
            derived = sorted({r[0] for r in rows if r and r[0]})

        if derived:
            logger.info(
                "No project_id supplied; derived correction scope %s from %s '%s'.",
                derived, target_label, source_id,
            )
        return derived

    def _analyze(
        self,
        source_id: str,
        target_label: str,
        project_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Runs the scoped impact traversal.

        Returns the two impact tiers separately, the dependent claims and the
        de-duplicated dependency paths.
        """
        collector = _PathCollector()
        direct_exp_ids: Set[str] = set()
        transitive_exp_ids: Set[str] = set()
        affected_claim_ids: Set[str] = set()

        params = {"source_id": source_id, "project_ids": project_ids}
        traversal = DIRECT_TRAVERSALS.get(target_label)
        if not traversal:
            return {
                "direct_experiment_ids": direct_exp_ids,
                "indirect_experiment_ids": transitive_exp_ids,
                "potentially_affected_claim_ids": affected_claim_ids,
                "paths": [],
            }

        cypher, reason_template = traversal
        link_rel = SOURCE_TO_EXPERIMENT_REL.get(target_label)
        rows = self.repository.client.execute_query(cypher, params)
        for r in rows:
            if not r or not r[0]:
                continue
            exp_id = r[0]
            direct_exp_ids.add(exp_id)
            claim_id = r[1] if len(r) > 1 and r[1] else None
            fmt: Dict[str, Any] = {"exp_id": exp_id, "source_id": source_id}
            if claim_id:
                fmt["claim_id"] = claim_id
            if target_label == NodeLabel.EVIDENCE:
                nodes = [source_id, claim_id, exp_id]
                rels = [RelType.SUPPORTED_BY, RelType.BASED_ON]
            elif target_label == NodeLabel.EXPERIMENT:
                # The corrected run is the experiment: no edge to traverse.
                nodes = [exp_id]
                rels = []
            else:
                nodes = [source_id, exp_id]
                rels = [link_rel or RelType.AFFECTS]
            collector.add(nodes, rels, reason_template.format(**fmt), group=("direct", exp_id))

        # Downstream lineage: one explanation per downstream experiment, keeping
        # the shortest path so two parents of one child yield a single path.
        downstream_paths: Dict[str, Tuple[List[str], List[str]]] = {}
        for direct_id in sorted(direct_exp_ids):
            rows = self.repository.client.execute_query(DOWNSTREAM_CYPHER.format(depth=MAX_LINEAGE_DEPTH), {
                "direct_id": direct_id,
                "project_ids": project_ids,
            })
            for r in rows:
                if not r or not r[0]:
                    continue
                downstream_id = r[0]
                if downstream_id in direct_exp_ids:
                    continue
                node_ids = [str(n) for n in (r[1] or [])]
                rel_types = [str(t) for t in (r[2] or [])]
                candidate = (node_ids, rel_types)
                current = downstream_paths.get(downstream_id)
                if current is None or (len(candidate[0]), candidate[0]) < (len(current[0]), current[0]):
                    downstream_paths[downstream_id] = candidate

        for downstream_id in sorted(downstream_paths):
            node_ids, rel_types = downstream_paths[downstream_id]
            transitive_exp_ids.add(downstream_id)
            prefix_rels = [link_rel] if link_rel else []
            collector.add(
                [source_id] + node_ids,
                prefix_rels + rel_types,
                f"Experiment '{downstream_id}' transitively inherits from parent experiment "
                f"'{node_ids[0] if node_ids else downstream_id}'.",
                group=("downstream", downstream_id),
            )

        # Claims derived from any impacted experiment, one path per claim. The
        # claim is appended to that experiment's own dependency path so the
        # reported node/relationship chain stays a real walk in the graph.
        all_affected = direct_exp_ids | transitive_exp_ids
        for exp_id in sorted(all_affected):
            if exp_id in direct_exp_ids:
                prefix_nodes = [source_id, exp_id]
                prefix_rels = [link_rel] if link_rel else []
            else:
                lineage_nodes, lineage_rels = downstream_paths[exp_id]
                prefix_nodes = [source_id] + lineage_nodes
                prefix_rels = ([link_rel] if link_rel else []) + lineage_rels

            rows = self.repository.client.execute_query(CLAIMS_OF_AFFECTED_CYPHER, {
                "exp_id": exp_id,
                "project_ids": project_ids,
            })
            for r in rows:
                if not r or not r[0]:
                    continue
                claim_id = r[0]
                if claim_id == source_id:
                    # The corrected claim itself is the subject of the
                    # correction, not a downstream consequence of it.
                    continue
                if claim_id in affected_claim_ids:
                    continue
                affected_claim_ids.add(claim_id)
                collector.add(
                    prefix_nodes + [claim_id],
                    prefix_rels + [RelType.BASED_ON],
                    f"Claim '{claim_id}' is based on affected experiment '{exp_id}'.",
                    group=("claim", claim_id),
                )

        if collector.duplicate_count:
            logger.info(
                "Suppressed %d duplicate impact path(s) for source '%s'.",
                collector.duplicate_count, source_id,
            )

        return {
            "direct_experiment_ids": direct_exp_ids,
            "indirect_experiment_ids": transitive_exp_ids,
            "potentially_affected_claim_ids": affected_claim_ids,
            "paths": collector.paths(),
        }

    def _create_review_item(
        self,
        review_item_id: str,
        experiment_id: str,
        correction_id: str,
        project_ids: List[str],
        reason: str,
    ) -> None:
        """
        Creates a ReviewItem in the revalidation queue and links it to the
        correction that raised it.

        ``RAISED_BY`` is a real relationship, so the correction's review items
        can be found by traversal instead of by an id-prefix guess.
        """
        self.repository.create_review_item(
            review_item_id=review_item_id,
            experiment_id=experiment_id,
            reason=reason,
            status=ReviewItemStatus.OPEN,
        )
        cypher = """
        MATCH (ri:ReviewItem {id: $ri_id}), (co:Correction {id: $corr_id})
        SET ri.project_id = $project_id,
            ri.correction_id = $corr_id,
            ri.experiment_id = $exp_id,
            ri.status = $status
        MERGE (ri)-[:RAISED_BY]->(co)
        """
        self.repository.client.execute_query(cypher, {
            "ri_id": review_item_id,
            "corr_id": correction_id,
            "exp_id": experiment_id,
            "project_id": project_ids[0] if project_ids else "",
            "status": ReviewItemStatus.OPEN,
        })

    def _store_impact_snapshot(self, correction_id: str, snapshot: Dict[str, Any]) -> None:
        """Persists the recorded analysis so it can be replayed by id."""
        cypher = """
        MATCH (co:Correction {id: $corr_id})
        SET co.impact_summary = $impact_summary,
            co.impact_snapshot = $impact_snapshot
        """
        self.repository.client.execute_query(cypher, {
            "corr_id": correction_id,
            "impact_summary": snapshot["summary"],
            "impact_snapshot": json.dumps(snapshot, default=str),
        })

    def _rebuild_snapshot(
        self,
        co_props: Dict[str, Any],
        source_id: str,
        source_label: str,
    ) -> Dict[str, Any]:
        """Re-analyses a legacy correction (written before snapshots existed)."""
        project_id = co_props.get("project_id") or None
        source_type = co_props.get("source_type") or ""
        try:
            target_label = SOURCE_TYPE_TARGETS.get(normalise_source_type(source_type))
        except ValueError:
            target_label = None
        if target_label is None:
            target_label = source_label if source_label in SOURCE_TYPE_TARGETS.values() else NodeLabel.EXPERIMENT
            source_type = next(
                (k for k, v in SOURCE_TYPE_TARGETS.items() if v == target_label), source_type or "experiment"
            )

        project_ids = self._resolve_scope(project_id, source_id, target_label)
        analysis = self._analyze(source_id, target_label, project_ids)

        direct = sorted(analysis["direct_experiment_ids"])
        indirect = sorted(analysis["indirect_experiment_ids"])
        return {
            "summary": co_props.get("description", ""),
            "project_ids": sorted(project_ids),
            "directly_affected_experiment_ids": direct,
            "indirectly_affected_experiment_ids": indirect,
            "potentially_affected_claim_ids": sorted(analysis["potentially_affected_claim_ids"]),
            "unaffected_experiment_ids": [],
            "review_item_ids": [],
            "paths": [
                {"nodes": p.nodes, "relationship_types": p.relationship_types, "reason": p.reason}
                for p in analysis["paths"]
            ],
            "warnings": ["Re-analysed from the graph; no recorded snapshot on this correction."],
            "generated_at": co_props.get("created_at"),
        }