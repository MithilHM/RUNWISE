"""In-memory `GraphStore` for development and tests.

Purpose: let the whole harness run with no FalkorDB container, so the console,
the tests and the demo work before — or instead of — a live database.

**It is selected by `RUNWISE_FAKE=1` and never by default.** The real store is
the product; this is a substitute, and a substitute that silently took over in
production would make the tool claim a graph it does not have.

It implements the same `GraphStore` protocol as `FalkorDBGraphStore`, including
the deduplication that F3 depends on: one `Setting` node per
(model, parameter, value), regardless of how many trials used it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.models.contracts import (
    DatasetProfile,
    Decision,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    LogEntry,
    RunState,
    TrialOutcome,
    WhyRecord,
    EvidenceItem,
)
from app.graph.repository import CITABLE_LABELS, param_id, setting_id

__all__ = ["FakeGraphStore", "reset_fake_graph"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, default=str)


class FakeGraphStore:
    """A `GraphStore` held entirely in memory."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: set[tuple[str, str, str]] = set()
        self.runs: Dict[str, RunState] = {}
        self._log_seq: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        return None

    def _put(self, node_id: str, label: str, props: Dict[str, Any]) -> str:
        node = self.nodes.setdefault(node_id, {"id": node_id, "type": label, "props": {}})
        node["type"] = label
        node["props"].update(props)
        return node_id

    def _link(self, source: str, rel: str, target: str) -> None:
        if source in self.nodes and target in self.nodes:
            self.edges.add((source, rel, target))

    def write_dataset(self, profile: DatasetProfile) -> None:
        self._put(profile.id, "Dataset", {
            "name": profile.name, "n_rows": profile.n_rows, "n_cols": profile.n_cols,
            "task": profile.task, "n_classes": profile.n_classes,
            "class_balance": _json(profile.class_balance), "split_hash": profile.split_hash,
            "target": profile.target, "target_inferred": profile.target_inferred,
            "dropped_columns": _json([d.model_dump() for d in profile.dropped_columns]),
            "created_at": profile.created_at,
        })
        for column in profile.columns:
            cid = f"column_{profile.id}_{column.name}"
            self._put(cid, "Column", column.model_dump())
            self._link(profile.id, "HAS_COLUMN", cid)

    def write_models(self, models: List[Any]) -> int:
        count = 0
        for model in models:
            self._put(model.id, "Model", {
                "name": model.name, "library": model.library, "description": model.description,
            })
            for param in model.params:
                pid = param_id(model.id, param.name)
                self._put(pid, "Param", {
                    "model_id": model.id, "name": param.name, "kind": param.kind,
                    "low": param.low, "high": param.high, "step": param.step,
                    "log_scale": param.log_scale, "choices": _json(param.choices),
                    "default_value": _json(param.default),
                })
                self._link(model.id, "HAS_PARAM", pid)
                count += 1
        return count

    def write_run(self, run: RunState) -> None:
        self.runs[run.run_id] = run.model_copy(deep=True)
        self._put(run.run_id, "Run", run.model_dump())
        self._link(run.run_id, "ON_DATASET", run.dataset_id)
        for model_id in run.model_ids:
            self._link(run.run_id, "TARGETS_MODEL", model_id)

    def write_trial(self, trial: TrialOutcome, decision) -> Dict[str, int]:
        # Mirrors the real store: a failed trial wrote no settings, because a
        # configuration the harness rejected was never tried.
        if trial.status == "SUCCESS":
            for name, value in trial.params.items():
                sid = setting_id(trial.model_id, name, value)
                self._put(sid, "Setting", {
                    "model_id": trial.model_id, "param_id": param_id(trial.model_id, name),
                    "name": name, "value": _json(value),
                    "value_num": value if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
                })
                self._link(trial.model_id, "OF_MODEL", sid)
                self._link(sid, "FOR_PARAM", param_id(trial.model_id, name))

        self._put(trial.trial_id, "Trial", {
            "run_id": trial.run_id, "step": trial.step, "model_id": trial.model_id,
            "status": trial.status, "metric_name": trial.metric_name,
            "metric_value": trial.metric_value, "metrics": _json(trial.metrics),
            "params": _json(trial.params), "failure_reason": trial.failure_reason or "",
            "duration_sec": trial.duration_sec, "n_features": trial.n_features,
            "seed": trial.seed, "created_at": trial.created_at or _now(),
        })
        self._link(trial.run_id, "HAS_TRIAL", trial.trial_id)
        self._link(trial.trial_id, "USES_MODEL", trial.model_id)
        for name, value in trial.params.items():
            self._link(trial.trial_id, "USES_SETTING", setting_id(trial.model_id, name, value))

        did = decision.decision_id
        self._put(did, "Decision", decision.model_dump())
        agent = f"agent_{decision.actor}"
        self._put(agent, "Agent", {"name": decision.actor})
        self._link(agent, "DECIDED", did)
        self._link(did, "CHOSE", trial.trial_id or decision.trial_id)
        self._link(did, "DURING", decision.run_id)
        for citation in decision.citations:
            self._link(did, "CITES", citation)
        return {"settings_written": len(trial.params), "nodes_added": 2, "edges_added": 1}

    def write_log(self, entry: LogEntry) -> None:
        lid = f"log_{entry.run_id}_{entry.seq:09d}"
        self._put(lid, "LogEntry", entry.model_dump())
        self._link(lid, "IN_RUN", entry.run_id)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def read_graph(
        self, run_id: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> GraphSnapshot:
        wanted: set[str] = set()
        for source, rel, target in self.edges:
            if source in wanted or target in wanted:
                wanted.add(source)
                wanted.add(target)
        # Seed from the requested scope, then close over connectivity.
        if run_id:
            wanted.add(run_id)
            wanted.add(dataset_id or "")
        elif dataset_id:
            wanted.add(dataset_id)

        changed = True
        while changed:
            changed = False
            for source, _rel, target in self.edges:
                if source in wanted or target in wanted:
                    before = len(wanted)
                    wanted.add(source)
                    wanted.add(target)
                    changed = changed or len(wanted) != before

        nodes = [
            GraphNode(id=nid, label=str(n["props"].get("name") or n["props"].get("id", nid)),
                      type=n["type"], props=dict(n["props"]))
            for nid, n in self.nodes.items() if nid in wanted
        ]
        edges = [
            GraphEdge(source=s, target=t, type=r, props={})
            for s, r, t in self.edges if s in wanted and t in wanted
        ]
        return GraphSnapshot(nodes=nodes, edges=edges)

    def read_logs(self, run_id: str, after_seq: int = 0) -> List[LogEntry]:
        entries = [
            LogEntry(**node["props"])
            for node in self.nodes.values()
            if node["type"] == "LogEntry" and node["props"].get("run_id") == run_id
            and node["props"].get("seq", 0) > after_seq
        ]
        entries.sort(key=lambda e: e.seq)
        return entries

    def explain_trial(self, run_id: str, trial_id: str) -> WhyRecord:
        for node in self.nodes.values():
            if node["type"] != "Decision":
                continue
            props = node["props"]
            if props.get("trial_id") != trial_id or (run_id and props.get("run_id") != run_id):
                continue
            citations = props.get("citations") or []
            if isinstance(citations, str):
                citations = json.loads(citations)
            evidence = []
            for citation in citations:
                target = self.nodes.get(citation)
                if not target or target["type"] not in CITABLE_LABELS:
                    continue
                evidence.append(EvidenceItem(
                    kind=target["type"], id=citation,
                    label=str(target["props"].get("name") or citation),
                    detail=str(target["props"].get("value") or ""),
                ))
            trial = self._trial_from(trial_id)
            return WhyRecord(decision=props, trial=trial, config=dict(trial.params),
                             evidence=evidence, highlight=[e.id for e in evidence])
        raise KeyError(f"no decision found for trial '{trial_id}'")

    def _trial_from(self, trial_id: str) -> TrialOutcome:
        props = self.nodes[trial_id]["props"]
        metrics = props.get("metrics") or {}
        if isinstance(metrics, str):
            metrics = json.loads(metrics)
        params = props.get("params") or {}
        if isinstance(params, str):
            params = json.loads(params)
        # The node's own id *is* the trial id; `write_trial` never stores a
        # separate `trial_id` property, so it has to be supplied from the key.
        return TrialOutcome(
            **{**props, "trial_id": trial_id, "metrics": metrics, "params": params,
               "failure_reason": props.get("failure_reason") or None},
        )

    def list_trials(self, run_id=None, model_id=None, status=None, limit=100) -> List[Dict[str, Any]]:
        out = []
        for node in self.nodes.values():
            if node["type"] != "Trial":
                continue
            p = node["props"]
            if run_id and p.get("run_id") != run_id:
                continue
            if model_id and p.get("model_id") != model_id:
                continue
            if status and p.get("status") != status:
                continue
            out.append({"id": node["id"], "run_id": p.get("run_id"),
                        "model_id": p.get("model_id"), "status": p.get("status"),
                        "metric_name": p.get("metric_name"),
                        "metric_value": p.get("metric_value"), "step": p.get("step")})
        out.sort(key=lambda t: t.get("step") or 0)
        return out[:limit]

    @property
    def trials(self) -> Dict[str, TrialOutcome]:
        """Every trial recorded, keyed by trial id.

        A dict rather than a list so callers can both count it and look a trial
        up by id without a comprehension.
        """
        return {
            node["id"]: self._trial_from(node["id"])
            for node in self.nodes.values()
            if node["type"] == "Trial"
        }

    @property
    def decisions(self) -> Dict[str, Decision]:
        """Every decision recorded, as contracts, keyed by decision id."""
        out: Dict[str, Decision] = {}
        for node in self.nodes.values():
            if node["type"] != "Decision":
                continue
            props = dict(node["props"])
            citations = props.get("citations") or []
            if isinstance(citations, str):
                citations = json.loads(citations)
            props["citations"] = citations
            props["decision_id"] = node["id"]
            out[node["id"]] = Decision(**props)
        return out

    def list_models(self) -> List[Dict[str, Any]]:
        return [{"id": n["id"], "name": n["props"].get("name"),
                 "library": n["props"].get("library")}
                for n in self.nodes.values() if n["type"] == "Model"]

    def list_datasets(self, limit: int = 20) -> List[Dict[str, Any]]:
        out = [
            {"id": n["id"], "name": n["props"].get("name"), "n_rows": n["props"].get("n_rows"),
             "task": n["props"].get("task"), "target": n["props"].get("target"),
             "created_at": n["props"].get("created_at")}
            for n in self.nodes.values() if n["type"] == "Dataset"
        ]
        return out[:limit]

    def best_trials(self, run_id: str) -> List[Dict[str, Any]]:
        best: Dict[str, Dict[str, Any]] = {}
        for trial in self.list_trials(run_id=run_id, status="SUCCESS"):
            value = trial.get("metric_value")
            if value is None:
                continue
            current = best.get(trial["model_id"])
            if current is None or value > current["metric_value"]:
                best[trial["model_id"]] = {
                    "model_id": trial["model_id"], "trial_id": trial["id"],
                    "metric_value": value, "metric_name": trial["metric_name"], "params": {},
                }
        return [best[k] for k in sorted(best)]

    def graph_neighbourhood(self, node_id: str, depth: int = 1) -> Dict[str, Any]:
        frontier = {node_id}
        seen = {node_id}
        edges: List[Dict[str, str]] = []
        for _ in range(max(1, min(depth, 3))):
            nxt = set()
            for source, rel, target in self.edges:
                if source in frontier or target in frontier:
                    if (source, rel, target) not in {(e["source"], e["type"], e["target"]) for e in edges}:
                        edges.append({"source": source, "type": rel, "target": target})
                    nxt.update({source, target})
            nxt -= seen
            seen |= nxt
            frontier = nxt
        return {
            "root": node_id,
            "nodes": [{"id": nid, "type": self.nodes[nid]["type"],
                       "label": str(self.nodes[nid]["props"].get("name") or nid)}
                      for nid in sorted(seen) if nid in self.nodes],
            "edges": edges,
        }


def reset_fake_graph() -> None:
    """Clears the shared instance. Used between tests."""
    get_fake_graph().__init__()


_fake: Optional[FakeGraphStore] = None


def get_fake_graph() -> FakeGraphStore:
    global _fake
    if _fake is None:
        _fake = FakeGraphStore()
    return _fake