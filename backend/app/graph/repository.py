"""FalkorDB implementation of the `GraphStore` port.

Everything the product knows lives here: the dataset and its columns, the three
models and their parameter spaces, one `Setting` node per concrete hyperparameter
value ever used, every trial, every decision, and every log line.

Two things this module is careful about:

- **`Setting` deduplication.** Its id is derived from the model, the parameter and
  the value, so the same value in a later run is the same node. That is what lets
  a traversal answer "which values of `max_depth` has XGBoost tried?".
- **`CITES` edges are real.** The ids in `Decision.citations` become
  `(Decision)-[:CITES]->(target)` relationships, not a JSON array. "Why this
  trial, highlight the node" only works if the highlight is a graph edge you can
  follow in both directions.

Every write uses `MERGE` on an explicit `id`, so replaying a run does not
duplicate the graph.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, Iterable, List, Optional

from app.graph.client import graph_client
from app.graph.schema import initialize_schema
from app.models.contracts import (
    Decision,
    DatasetProfile,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    LogEntry,
    RunState,
    TrialOutcome,
    WhyRecord,
    EvidenceItem,
    is_better,
    is_number,
)
from app.services import catalog

logger = logging.getLogger(__name__)

__all__ = ["FalkorDBGraphStore", "get_graph_store", "setting_id", "param_id", "decision_id"]

# Labels a decision is allowed to cite. The writer and the reader must agree on
# this list exactly: when they disagreed, `WhyRecord.highlight` could name a node
# that the reader resolved but the writer had refused to draw a `CITES` edge to,
# leaving a highlight with no relationship behind it.
CITABLE_LABELS = ("Model", "Param", "Setting", "Trial", "Dataset", "Column", "Run", "Agent")
_LABEL_FILTER = " OR ".join(f"t:{label}" for label in CITABLE_LABELS)


# ---------------------------------------------------------------------------
# Deterministic ids. Two runs that make the same choice address the same node.
# ---------------------------------------------------------------------------


def _digest(*parts: Any) -> str:
    blob = "|".join(json.dumps(part, sort_keys=True, default=str) for part in parts)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def param_id(model_id: str, name: str) -> str:
    return f"param_{model_id}_{name}"


def setting_id(model_id: str, name: str, value: Any) -> str:
    """One node per (model, parameter, value). The point of F3."""
    return f"setting_{model_id}_{name}_{_digest(value)}"


def decision_id(run_id: str, step: int, actor: str) -> str:
    return f"decision_{run_id}_s{step:03d}_{actor}"


# ---------------------------------------------------------------------------


class FalkorDBGraphStore:
    """Implements `GraphStore` against FalkorDB."""

    def __init__(self, graph_name: Optional[str] = None) -> None:
        self.graph_name = graph_name

    @property
    def _graph(self):
        return graph_client.get_graph(self.graph_name)

    def _run(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[List[Any]]:
        return graph_client.execute_query(cypher, params or {}, self.graph_name)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        initialize_schema(self._graph)

    def is_healthy(self) -> bool:
        return graph_client.check_health().get("status") == "healthy"

    def reset(self) -> bool:
        """Drops every node and edge. For tests and for a clean demo."""
        return graph_client.delete_graph(self.graph_name)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def write_dataset(self, profile: DatasetProfile) -> int:
        """Persists the dataset and its surviving columns.

        Every run uses one seeded stratified split, so one split is stored per
        dataset rather than per run. Re-ingesting the same file overwrites the
        dataset node; the trials that referenced it keep their history.
        """
        self._run(
            """
            MERGE (d:Dataset {id: $id})
            SET d.name = $name,
                d.n_rows = $n_rows,
                d.n_cols = $n_cols,
                d.task = $task,
                d.n_classes = $n_classes,
                d.class_balance = $class_balance,
                d.split_hash = $split_hash,
                d.target = $target,
                d.target_inferred = $target_inferred,
                d.dropped_columns = $dropped,
                d.created_at = $created_at
            """,
            {
                "id": profile.id,
                "name": profile.name,
                "n_rows": profile.n_rows,
                "n_cols": profile.n_cols,
                "task": profile.task,
                "n_classes": profile.n_classes,
                "class_balance": json.dumps(profile.class_balance),
                "split_hash": profile.split_hash,
                "target": profile.target,
                "target_inferred": profile.target_inferred,
                "dropped": json.dumps([d.model_dump() for d in profile.dropped_columns]),
                "created_at": profile.created_at,
            },
        )

        for column in profile.columns:
            self._run(
                """
                MATCH (d:Dataset {id: $dataset_id})
                MERGE (c:Column {id: $id})
                SET c.name = $name,
                    c.dtype = $dtype,
                    c.role = $role,
                    c.n_missing = $n_missing,
                    c.n_unique = $n_unique
                MERGE (d)-[:HAS_COLUMN]->(c)
                """,
                {
                    "dataset_id": profile.id,
                    "id": f"column_{profile.id}_{column.name}",
                    "name": column.name,
                    "dtype": column.dtype,
                    "role": column.role,
                    "n_missing": column.n_missing,
                    "n_unique": column.n_unique,
                },
            )
        return len(profile.columns)

    def write_models(self, models: Iterable[Any]) -> int:
        """Persists the fixed catalog and its `Param` nodes."""
        count = 0
        for model in models:
            self._run(
                """
                MERGE (m:Model {id: $id})
                SET m.name = $name,
                    m.library = $library,
                    m.description = $description
                """,
                {
                    "id": model.id,
                    "name": model.name,
                    "library": model.library,
                    "description": model.description,
                },
            )
            for param in model.params:
                self._run(
                    """
                    MATCH (m:Model {id: $model_id})
                    MERGE (p:Param {id: $id})
                    SET p.model_id = $model_id,
                        p.name = $name,
                        p.kind = $kind,
                        p.low = $low,
                        p.high = $high,
                        p.step = $step,
                        p.log_scale = $log_scale,
                        p.choices = $choices,
                        p.default_value = $default_value
                    MERGE (m)-[:HAS_PARAM]->(p)
                    """,
                    {
                        "model_id": model.id,
                        "id": param_id(model.id, param.name),
                        "name": param.name,
                        "kind": param.kind,
                        "low": param.low,
                        "high": param.high,
                        "step": param.step,
                        "log_scale": param.log_scale,
                        "choices": json.dumps(param.choices) if param.choices is not None else "",
                        "default_value": json.dumps(param.default),
                    },
                )
                count += 1
        return count

    def write_run(self, run: RunState) -> None:
        self._run(
            """
            MERGE (r:Run {id: $id})
            SET r.dataset_id = $dataset_id,
                r.mode = $mode,
                r.status = $status,
                r.step = $step,
                r.budget = $budget,
                r.metric_name = $metric_name,
                r.best_metric = $best_metric,
                r.best_trial_id = $best_trial_id,
                r.model_ids = $model_ids,
                r.created_at = $created_at,
                r.updated_at = $updated_at
            WITH r
            MATCH (d:Dataset {id: $dataset_id})
            MERGE (r)-[:ON_DATASET]->(d)
            """,
            {
                "id": run.run_id,
                "dataset_id": run.dataset_id,
                "mode": run.mode,
                "status": run.status,
                "step": run.step,
                "budget": run.budget,
                "metric_name": run.metric_name,
                "best_metric": run.best_metric,
                "best_trial_id": run.best_trial_id,
                "model_ids": json.dumps(run.model_ids),
                "created_at": run.created_at,
                "updated_at": run.updated_at,
            },
        )
        for model_id in run.model_ids:
            self._run(
                """
                MATCH (r:Run {id: $run_id})
                MATCH (m:Model {id: $model_id})
                MERGE (r)-[:TARGETS_MODEL]->(m)
                """,
                {"run_id": run.run_id, "model_id": model_id},
            )

    def write_trial(self, trial: TrialOutcome, decision: Decision) -> Dict[str, int]:
        """Records the trial, its settings, the decision and the citations.

        Settings and their edges are written first so the trial's `USES_SETTING`
        relationships always resolve to a node that exists.
        """
        settings_written = self._write_settings(trial)

        self._run(
            """
            MERGE (t:Trial {id: $id})
            SET t.run_id = $run_id,
                t.step = $step,
                t.model_id = $model_id,
                t.dataset_id = $dataset_id,
                t.status = $status,
                t.metric_name = $metric_name,
                t.metric_value = $metric_value,
                t.metrics = $metrics,
                t.params = $params,
                t.failure_reason = $failure_reason,
                t.duration_sec = $duration_sec,
                t.n_features = $n_features,
                t.seed = $seed,
                t.created_at = $created_at
            """,
            {
                "id": trial.trial_id,
                "run_id": trial.run_id,
                "step": trial.step,
                "model_id": trial.model_id,
                "dataset_id": self._dataset_of_run(trial.run_id),
                "status": trial.status,
                "metric_name": trial.metric_name,
                "metric_value": trial.metric_value,
                "metrics": json.dumps(trial.metrics),
                "params": json.dumps(trial.params, default=str),
                "failure_reason": trial.failure_reason or "",
                "duration_sec": trial.duration_sec,
                "n_features": trial.n_features,
                "seed": trial.seed,
                "created_at": trial.created_at,
            },
        )

        edges = 0
        if trial.run_id:
            self._run(
                """
                MATCH (r:Run {id: $run_id})
                MATCH (t:Trial {id: $trial_id})
                MERGE (r)-[:HAS_TRIAL]->(t)
                """,
                {"run_id": trial.run_id, "trial_id": trial.trial_id},
            )
            edges += 1
        else:
            # The Trial node exists but is attached to nothing, so it is absent
            # from `read_graph`, `list_trials(run_id=...)` and `best_trials`.
            # Silent, so it is logged loudly.
            logger.warning(
                "Trial %s has no run_id; it will not appear under any run.",
                trial.trial_id,
            )

        self._run(
            """
            MATCH (t:Trial {id: $trial_id})
            MATCH (m:Model {id: $model_id})
            MERGE (t)-[:USES_MODEL]->(m)
            """,
            {"trial_id": trial.trial_id, "model_id": trial.model_id},
        )

        for name, value in trial.params.items():
            sid = setting_id(trial.model_id, name, value)
            self._run(
                """
                MATCH (t:Trial {id: $trial_id})
                MATCH (s:Setting {id: $setting_id})
                MERGE (t)-[:USES_SETTING]->(s)
                """,
                {"trial_id": trial.trial_id, "setting_id": sid},
            )

        self._write_decision(decision, trial)
        return {"settings_written": settings_written, "nodes_added": 2, "edges_added": edges}

    def _write_settings(self, trial: TrialOutcome) -> int:
        """One deduplicated node per parameter value.

        Skipped entirely for failed trials. A configuration the harness rejected
        was never tried, and writing a `Setting` for it would put a value the
        model never saw into the answer to "which values has XGBoost tried?".

        `:Model` and `:Param` are `MATCH`ed, never merged. Declaring a node in an
        inline endpoint position of a `MERGE` re-creates it when it already
        exists from an earlier statement, which silently duplicated every model
        and parameter on each trial and broke the deduplication this feature
        exists to provide.
        """
        if trial.status != "SUCCESS":
            return 0

        written = 0
        for name, value in trial.params.items():
            sid = setting_id(trial.model_id, name, value)
            spec = catalog.get_model(trial.model_id).param(name)
            if spec is None:
                # Never mint a `Param` node the catalog does not declare.
                logger.warning(
                    "Trial %s used parameter '%s', which %s does not declare; "
                    "not writing a Setting for it",
                    trial.trial_id, name, trial.model_id,
                )
                continue
            self._run(
                """
                MERGE (s:Setting {id: $id})
                SET s.model_id = $model_id,
                    s.param_id = $param_id,
                    s.name = $name,
                    s.value = $value,
                    s.value_num = $value_num
                WITH s
                MATCH (m:Model {id: $model_id})
                MERGE (m)-[:OF_MODEL]->(s)
                WITH s
                MATCH (p:Param {id: $param_id})
                MERGE (s)-[:FOR_PARAM]->(p)
                """,
                {
                    "id": sid,
                    "model_id": trial.model_id,
                    "param_id": param_id(trial.model_id, name),
                    "name": name,
                    "value": json.dumps(value, default=str),
                    "value_num": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                },
            )
            written += 1
        return written

    def _ensure_agent(self, name: str) -> None:
        """Creates an `Agent` node if absent, in its own statement.

        Kept separate from the relationship write for the same reason as
        `_write_settings`: a node declared inline in a `MERGE` chain can be
        duplicated rather than matched.

        It carries an `id` as well as a `name` because every other label is
        addressed by `id`. An agent keyed only on `name` is invisible to
        `read_graph`, whose scope is built from `id` columns — so `DECIDED` edges
        would silently never appear in the graph the UI reads.
        """
        self._run(
            "MERGE (a:Agent {name: $name}) SET a.id = $id, a.actor = $name",
            {"name": name, "id": f"agent_{name}"},
        )

    def _write_decision(self, decision: Decision, trial: TrialOutcome) -> None:
        """Persists the decision and materialises its citations as real edges."""
        self._ensure_agent(decision.actor)
        self._run(
            """
            MERGE (d:Decision {id: $id})
            SET d.run_id = $run_id,
                d.step = $step,
                d.trial_id = $trial_id,
                d.actor = $actor,
                d.strategy = $strategy,
                d.why = $why,
                d.hypothesis = $hypothesis,
                d.confidence = $confidence,
                d.citations = $citations,
                d.created_at = $created_at
            """,
            {
                "id": decision.decision_id,
                "run_id": decision.run_id,
                "step": decision.step,
                "trial_id": decision.trial_id or trial.trial_id,
                "actor": decision.actor,
                "strategy": decision.strategy,
                "why": decision.why,
                "hypothesis": decision.hypothesis,
                "confidence": decision.confidence,
                "citations": json.dumps(decision.citations),
                "created_at": decision.created_at,
            },
        )
        self._run(
            """
            MATCH (d:Decision {id: $decision_id})
            MATCH (a:Agent {name: $agent_name})
            MERGE (a)-[:DECIDED]->(d)
            """,
            {"decision_id": decision.decision_id, "agent_name": decision.actor},
        )

        if trial.trial_id:
            self._run(
                """
                MATCH (d:Decision {id: $decision_id})
                MATCH (t:Trial {id: $trial_id})
                MERGE (d)-[:CHOSE]->(t)
                """,
                {"decision_id": decision.decision_id, "trial_id": trial.trial_id},
            )
        if decision.run_id:
            self._run(
                """
                MATCH (d:Decision {id: $decision_id})
                MATCH (r:Run {id: $run_id})
                MERGE (d)-[:DURING]->(r)
                """,
                {"decision_id": decision.decision_id, "run_id": decision.run_id},
            )

        for citation in decision.citations:
            self._run(
                f"""
                MATCH (d:Decision {{id: $decision_id}})
                MATCH (t)
                WHERE ({_LABEL_FILTER}) AND t.id = $citation
                MERGE (d)-[r:CITES]->(t)
                SET r.kind = labels(t)[0]
                """,
                {"decision_id": decision.decision_id, "citation": citation},
            )

    def write_log(self, entry: LogEntry) -> None:
        self._run(
            """
            MERGE (l:LogEntry {id: $id})
            SET l.run_id = $run_id,
                l.seq = $seq,
                l.step = $step,
                l.level = $level,
                l.stage = $stage,
                l.message = $message,
                l.meta = $meta,
                l.ts = $ts
            WITH l
            MATCH (r:Run {id: $run_id})
            MERGE (l)-[:IN_RUN]->(r)
            """,
            {
                "id": f"log_{entry.run_id}_{entry.seq:09d}",
                "run_id": entry.run_id,
                "seq": entry.seq,
                "step": entry.step,
                "level": entry.level,
                "stage": entry.stage,
                "message": entry.message,
                "meta": json.dumps(entry.meta, default=str),
                "ts": entry.ts,
            },
        )

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_logs(self, run_id: str, after_seq: int = 0) -> List[LogEntry]:
        rows = self._run(
            """
            MATCH (l:LogEntry {run_id: $run_id})
            WHERE l.seq > $after
            RETURN l.seq, l.step, l.level, l.stage, l.message, l.meta, l.ts
            ORDER BY l.seq ASC
            """,
            {"run_id": run_id, "after": after_seq},
        )
        return [self._log_from_row(run_id, row) for row in rows]

    @staticmethod
    def _log_from_row(run_id: str, row: List[Any]) -> LogEntry:
        seq, step, level, stage, message, meta, ts = row
        try:
            parsed = json.loads(meta) if meta else {}
        except json.JSONDecodeError:
            parsed = {"raw": meta}
        return LogEntry(
            seq=int(seq or 0),
            run_id=run_id,
            step=int(step or 0),
            level=level or "INFO",
            stage=stage or "execute",
            message=message or "",
            meta=parsed,
            ts=ts or "",
        )

    def read_graph(
        self, run_id: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> GraphSnapshot:
        """Builds the node and edge set for a run, or for a whole dataset.

        Scoped by a set of ids gathered first, then two flat queries. Deliberately
        not one giant OPTIONAL MATCH: that produces a cartesian product of every
        log entry against every trial, which is both slow and wrong.
        """
        node_ids = self._scoped_ids(run_id=run_id, dataset_id=dataset_id)
        if not node_ids:
            return GraphSnapshot()

        node_rows = self._run(
            """
            MATCH (n)
            WHERE n.id IN $ids
            RETURN n.id, labels(n), properties(n)
            """,
            {"ids": node_ids},
        )
        edge_rows = self._run(
            """
            MATCH (a)-[r]->(b)
            WHERE a.id IN $ids AND b.id IN $ids
            RETURN a.id, type(r), b.id, properties(r)
            """,
            {"ids": node_ids},
        )

        nodes = [
            GraphNode(id=str(row[0]), label=self._label_for(row[1], row[2]), type=(row[1] or ["Node"])[0], props=self._clean(row[2]))
            for row in node_rows
        ]
        edges = [
            GraphEdge(source=str(row[0]), target=str(row[2]), type=str(row[1]), props=self._clean(row[3]))
            for row in edge_rows
        ]
        return GraphSnapshot(nodes=nodes, edges=edges)

    def _scoped_ids(
        self, run_id: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> List[str]:
        """Every node id in scope for a run, a dataset, or the whole graph.

        Deliberately *not* one query with a chain of `OPTIONAL MATCH`es. Each
        optional match multiplies the rows of the ones before it: three trials
        against nine settings and fourteen log entries is already a five-figure
        intermediate, and FalkorDB started silently truncating it — a run would
        report two of its three trials at random. Several small independent
        queries, unioned here, cost a handful of sub-millisecond round trips and
        return the same set every time.
        """
        found: set[str] = set()
        actors: set[str] = set()

        def collect(cypher: str, params: Dict[str, Any]) -> None:
            for row in self._run(cypher, params):
                for cell in row:
                    if cell:
                        found.add(str(cell))

        def collect_actors(cypher: str, params: Dict[str, Any]) -> None:
            """Actor names, mapped to their node ids.

            An `Agent` node is keyed `agent_<actor>`, so the actor string on the
            decision is not itself a node id and cannot be collected directly.
            """
            for row in self._run(cypher, params):
                if row and row[0]:
                    actors.add(str(row[0]))

        if run_id:
            collect(
                "MATCH (r:Run {id: $run_id}) RETURN r.id, r.dataset_id",
                {"run_id": run_id},
            )
            collect(
                "MATCH (r:Run {id: $run_id})-[:HAS_TRIAL]->(t:Trial) RETURN t.id, t.model_id",
                {"run_id": run_id},
            )
            collect(
                "MATCH (t:Trial)-[:USES_SETTING]->(s:Setting) "
                "WHERE t.run_id = $run_id RETURN s.id, s.param_id, s.model_id",
                {"run_id": run_id},
            )
            collect(
                "MATCH (d:Decision)-[:DURING]->(r:Run {id: $run_id}) RETURN d.id",
                {"run_id": run_id},
            )
            collect_actors(
                "MATCH (d:Decision)-[:DURING]->(r:Run {id: $run_id}) RETURN d.actor",
                {"run_id": run_id},
            )
            collect(
                "MATCH (d:Decision)-[:CITES]->(c) WHERE d.run_id = $run_id RETURN c.id",
                {"run_id": run_id},
            )
            collect(
                "MATCH (l:LogEntry)-[:IN_RUN]->(r:Run {id: $run_id}) RETURN l.id",
                {"run_id": run_id},
            )
            collect(
                "MATCH (r:Run {id: $run_id})-[:ON_DATASET]->(d:Dataset) RETURN d.id",
                {"run_id": run_id},
            )
            collect(
                "MATCH (t:Trial) WHERE t.run_id = $run_id "
                "OPTIONAL MATCH (t)-[:DERIVED_FROM]->(p:Trial) RETURN p.id",
                {"run_id": run_id},
            )

        elif dataset_id:
            found.add(dataset_id)
            collect(
                "MATCH (d:Dataset {id: $dataset_id})-[:HAS_COLUMN]->(c:Column) RETURN c.id",
                {"dataset_id": dataset_id},
            )
            collect(
                "MATCH (r:Run)-[:ON_DATASET]->(d:Dataset {id: $dataset_id}) RETURN r.id",
                {"dataset_id": dataset_id},
            )
            collect(
                "MATCH (r:Run)-[:ON_DATASET]->(d:Dataset {id: $dataset_id}) "
                "MATCH (r)-[:HAS_TRIAL]->(t:Trial) RETURN t.id, t.model_id",
                {"dataset_id": dataset_id},
            )
            collect(
                "MATCH (r:Run)-[:ON_DATASET]->(d:Dataset {id: $dataset_id}) "
                "MATCH (r)-[:HAS_TRIAL]->(t:Trial)-[:USES_SETTING]->(s:Setting) "
                "RETURN s.id, s.param_id, s.model_id",
                {"dataset_id": dataset_id},
            )
            collect(
                "MATCH (r:Run)-[:ON_DATASET]->(d:Dataset {id: $dataset_id}) "
                "MATCH (dec:Decision)-[:DURING]->(r) RETURN dec.id",
                {"dataset_id": dataset_id},
            )
            collect_actors(
                "MATCH (r:Run)-[:ON_DATASET]->(d:Dataset {id: $dataset_id}) "
                "MATCH (dec:Decision)-[:DURING]->(r) RETURN dec.actor",
                {"dataset_id": dataset_id},
            )
            collect(
                "MATCH (l:LogEntry)-[:IN_RUN]->(r:Run)-[:ON_DATASET]->"
                "(d:Dataset {id: $dataset_id}) RETURN l.id",
                {"dataset_id": dataset_id},
            )
        else:
            collect("MATCH (n) RETURN n.id", {})

        found.update(f"agent_{actor}" for actor in actors)
        return sorted(found)

    @staticmethod
    def _label_for(labels: List[str], props: Dict[str, Any]) -> str:
        """Human label for a node, preferring its own name property."""
        for key in ("name", "label", "target", "message", "strategy"):
            value = props.get(key)
            if value:
                return str(value)
        if "Setting" in labels:
            value = props.get("value")
            return f"{props.get('name')} = {value}"
        if "Run" in labels:
            return f"{props.get('id', 'run')} (step {props.get('step', 0)})"
        if "Trial" in labels:
            metric = props.get("metric_value")
            status = props.get("status", "")
            return f"{props.get('id', 'trial')} {status} {metric if metric is not None else ''}".strip()
        return str(props.get("id", ""))

    @staticmethod
    def _clean(props: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Strips JSON-encoded columns so the frontend gets real values."""
        cleaned: Dict[str, Any] = {}
        for key, value in (props or {}).items():
            if key in ("class_balance", "dropped_columns", "model_ids", "choices",
                       "default_value", "value", "metrics", "params", "citations", "meta"):
                try:
                    cleaned[key] = json.loads(value) if isinstance(value, str) else value
                except (json.JSONDecodeError, TypeError):
                    cleaned[key] = value
            else:
                cleaned[key] = value
        return cleaned

    # ------------------------------------------------------------------
    # F7: why, and what it points at
    # ------------------------------------------------------------------

    def explain_trial(self, run_id: str, trial_id: str) -> WhyRecord:
        """The decision behind a trial plus the graph elements it cited."""
        rows = self._run(
            """
            MATCH (d:Decision)-[:CHOSE]->(t:Trial {id: $trial_id})
            WHERE $run_id = '' OR d.run_id = $run_id
            RETURN d.id, d.run_id, d.step, d.actor, d.strategy, d.why, d.hypothesis,
                   d.confidence, d.citations, d.created_at
            ORDER BY d.created_at DESC
            LIMIT 1
            """,
            {"trial_id": trial_id, "run_id": run_id or ""},
        )
        if not rows:
            raise KeyError(f"no decision found for trial '{trial_id}'")
        row = rows[0]
        citations = json.loads(row[8]) if row[8] else []

        decision = Decision(
            decision_id=str(row[0]),
            run_id=str(row[1]),
            step=int(row[2] or 0),
            trial_id=trial_id,
            actor=row[3] or "laya",
            strategy=row[4] or "",
            why=row[5] or "",
            hypothesis=row[6] or "",
            confidence=float(row[7] or 0.5),
            citations=citations,
            created_at=row[9] or "",
        )

        trial = self.get_trial(trial_id)
        evidence = self.resolve_citations(citations)
        resolved_ids = {item.id for item in evidence}
        dangling = [cid for cid in citations if cid not in resolved_ids]
        if dangling:
            # A citation to a node that is not there would highlight nothing. The
            # decision record keeps the full list; the highlight only names nodes
            # the graph can actually show.
            logger.warning(
                "Decision %s cites %d node(s) that do not resolve: %s",
                decision.decision_id, len(dangling), dangling[:5],
            )
        return WhyRecord(
            decision=decision,
            trial=trial,
            config=dict(trial.params),
            evidence=evidence,
            highlight=[cid for cid in citations if cid in resolved_ids],
        )

    def resolve_citations(self, citations: List[str]) -> List[EvidenceItem]:
        """Turns citation ids into labelled evidence items.

        A citation that no longer resolves is dropped rather than rendered as an
        empty chip. The `highlight` list the UI receives still names it, so a
        dangling citation shows up as an unhighlighted request instead of a
        confident blank.
        """
        if not citations:
            return []
        rows = self._run(
            f"""
            MATCH (n)
            WHERE ({_LABEL_FILTER.replace('t:', 'n:')}) AND n.id IN $ids
            RETURN n.id, labels(n), properties(n)
            """,
            {"ids": list(citations)},
        )
        by_id = {str(row[0]): row for row in rows}
        items: List[EvidenceItem] = []
        for citation in citations:
            row = by_id.get(citation)
            if row is None:
                continue
            labels, props = row[1] or [], row[2] or {}
            kind = labels[0] if labels else "Trial"
            items.append(
                EvidenceItem(
                    kind=kind,
                    id=citation,
                    label=self._label_for(labels, props),
                    detail=self._detail_for(kind, props),
                )
            )
        return items

    @staticmethod
    def _detail_for(kind: str, props: Dict[str, Any]) -> str:
        if kind == "Setting":
            return f"{props.get('name')} = {props.get('value')} ({props.get('model_id')})"
        if kind == "Param":
            lo, hi = props.get("low"), props.get("high")
            if props.get("choices"):
                return f"{props.get('name')}: choices {props.get('choices')}"
            return f"{props.get('name')}: [{lo}, {hi}]"
        if kind == "Trial":
            metric = props.get("metric_value")
            return (
                f"step {props.get('step')} scored "
                f"{props.get('metric_name')}={metric}" if metric is not None
                else f"step {props.get('step')} ({props.get('status')})"
            )
        if kind == "Model":
            return str(props.get("library", ""))
        if kind == "Dataset":
            return f"{props.get('n_rows')} rows, {props.get('task')}"
        return ""

    def get_trial(self, trial_id: str) -> TrialOutcome:
        rows = self._run(
            """
            MATCH (t:Trial {id: $id})
            RETURN t.id, t.run_id, t.step, t.model_id, t.status, t.metric_name,
                   t.metric_value, t.metrics, t.params, t.failure_reason,
                   t.duration_sec, t.n_features, t.seed, t.created_at
            """,
            {"id": trial_id},
        )
        if not rows:
            raise KeyError(f"no trial with id '{trial_id}'")
        row = rows[0]
        metrics = json.loads(row[7]) if row[7] else {}
        # A NaN can only exist here if it was written by an older build or by a
        # direct Cypher write. It is coerced on the way out so it cannot become a
        # permanent best-so-far through this path either.
        if row[6] is not None and not is_number(row[6]):
            metrics = {k: (None if v is not None and not is_number(v) else v) for k, v in metrics.items()}
            value = None
        else:
            value = row[6]
        return TrialOutcome(
            trial_id=str(row[0]),
            run_id=str(row[1] or ""),
            step=int(row[2] or 0),
            model_id=str(row[3]),
            status=row[4] or "FAILED",
            metric_name=row[5] or "",
            metric_value=value,
            metrics=metrics,
            params=json.loads(row[8]) if row[8] else {},
            failure_reason=row[9] or None,
            duration_sec=float(row[10] or 0.0),
            n_features=int(row[11] or 0),
            seed=int(row[12] or 42),
            created_at=row[13] or "",
        )

    # ------------------------------------------------------------------
    # Query surfaces used by the MCP tools
    # ------------------------------------------------------------------

    def list_models(self) -> List[Dict[str, Any]]:
        rows = self._run("MATCH (m:Model) RETURN m.id, m.name, m.library ORDER BY m.name")
        return [{"id": r[0], "name": r[1], "library": r[2]} for r in rows]

    def list_datasets(self, limit: int = 20) -> List[Dict[str, Any]]:
        rows = self._run(
            """
            MATCH (d:Dataset)
            RETURN d.id, d.name, d.n_rows, d.task, d.target, d.created_at
            ORDER BY d.created_at DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        return [
            {"id": r[0], "name": r[1], "n_rows": r[2], "task": r[3],
             "target": r[4], "created_at": r[5]}
            for r in rows
        ]

    def list_trials(
        self,
        run_id: Optional[str] = None,
        model_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses, params = ["MATCH (t:Trial)"], {"limit": limit}
        filters = []
        if run_id:
            filters.append("t.run_id = $run_id")
            params["run_id"] = run_id
        if model_id:
            filters.append("t.model_id = $model_id")
            params["model_id"] = model_id
        if status:
            filters.append("t.status = $status")
            params["status"] = status
        if filters:
            clauses.append("WHERE " + " AND ".join(filters))
        clauses.append("RETURN t.id, t.run_id, t.model_id, t.status, t.metric_name, "
                       "t.metric_value, t.step ORDER BY t.step ASC LIMIT $limit")
        return [
            {
                "id": r[0], "run_id": r[1], "model_id": r[2], "status": r[3],
                "metric_name": r[4], "metric_value": r[5], "step": r[6],
            }
            for r in self._run(" ".join(clauses), params)
        ]

    def best_trials(self, run_id: str) -> List[Dict[str, Any]]:
        """Best trial per model for a run.

        Reduced in Python rather than with `max()`. Cypher puts every non-
        aggregated field in the implicit `GROUP BY`, so `RETURN t.model_id, t.id,
        max(t.metric_value)` groups by the trial id: `max()` is a no-op over one
        row and the query returns every trial rather than the best per model. A
        caller taking the first row then ranks an arbitrary trial.
        """
        rows = self._run(
            """
            MATCH (r:Run {id: $run_id})-[:HAS_TRIAL]->(t:Trial)
            WHERE t.status = 'SUCCESS' AND t.metric_value IS NOT NULL
            RETURN t.model_id, t.id, t.metric_value, t.metric_name, t.params
            ORDER BY t.model_id ASC
            """,
            {"run_id": run_id},
        )
        best: Dict[str, Dict[str, Any]] = {}
        for model_id, trial_id, value, metric_name, params in rows:
            if not is_number(value):
                continue
            current = best.get(model_id)
            if current is None or is_better(float(value), float(current["metric_value"]), metric_name):
                best[model_id] = {
                    "model_id": model_id,
                    "trial_id": trial_id,
                    "metric_value": float(value),
                    "metric_name": metric_name,
                    "params": json.loads(params) if params else {},
                }
        return [best[key] for key in sorted(best)]

    def graph_neighbourhood(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        """N-hop traversal around a node. The multi-hop tool.

        Depth is capped at 3: past that the traversal returns most of the graph and
        stops being an answer to a question.
        """
        depth = max(1, min(int(depth), 3))
        rows = self._run(
            f"""
            MATCH (start) WHERE start.id = $node_id
            MATCH path = (start)-[*1..{depth}]-(other)
            RETURN start.id, [n IN nodes(path) | n.id] AS ids,
                   [r IN relationships(path) | type(r)] AS rels
            LIMIT 200
            """,
            {"node_id": node_id},
        )
        if not rows:
            raise KeyError(f"no node with id '{node_id}'")
        ids = {str(rows[0][0])}
        edges: List[Dict[str, str]] = []
        for row in rows:
            chain = [str(i) for i in row[1]]
            ids.update(chain)
            for index, rel in enumerate(row[2] or []):
                edges.append({"source": chain[index], "type": str(rel), "target": chain[index + 1]})
        node_rows = self._run(
            """
            MATCH (n) WHERE n.id IN $ids
            RETURN n.id, labels(n), properties(n)
            """,
            {"ids": sorted(ids)},
        )
        return {
            "root": node_id,
            "nodes": [
                {"id": str(r[0]), "type": (r[1] or ["Node"])[0],
                 "label": self._label_for(r[1] or [], r[2] or {})}
                for r in node_rows
            ],
            "edges": edges,
        }

    def _dataset_of_run(self, run_id: str) -> str:
        if not run_id:
            return ""
        rows = self._run(
            "MATCH (r:Run {id: $run_id})-[:ON_DATASET]->(d:Dataset) RETURN d.id LIMIT 1",
            {"run_id": run_id},
        )
        return str(rows[0][0]) if rows else ""


_store: Optional[FalkorDBGraphStore] = None


def get_graph_store() -> FalkorDBGraphStore:
    global _store
    if _store is None:
        _store = FalkorDBGraphStore()
    return _store