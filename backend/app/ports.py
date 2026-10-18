"""Ports: the interfaces the harness is written against.

Everything above this line (the loop, the HTTP layer, the MCP tools) depends on
these Protocols. Everything below it (FalkorDB, pandas, scikit-learn, an LLM) is
an implementation detail. That is what lets the harness run end to end against
in-memory fakes before the real adapters exist.

Nothing in this module imports a concrete implementation.
"""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional, Protocol, runtime_checkable

import pandas as pd

from app.models.contracts import (
    Decision,
    DatasetProfile,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    LogEntry,
    LogEvent,
    Proposal,
    RunState,
    TrialOutcome,
    WhyRecord,
)


@runtime_checkable
class DataStore(Protocol):
    """Read access to ingested data. Write access happens through ingestion."""

    def get_profile(self, dataset_id: str) -> DatasetProfile: ...

    def get_frame(self, dataset_id: str) -> tuple[pd.DataFrame, pd.Series]:
        """Returns the feature frame and the target series, already split.

        The split is stratified on the target and seeded, so the same dataset id
        always yields the same rows. Two trials that are not comparable are worse
        than no trials, so this guarantee matters.
        """
        ...


@runtime_checkable
class Trainer(Protocol):
    """Trains one model on one configuration and measures it."""

    def validate(self, model_id: str, params: Dict[str, Any]) -> List[str]:
        """Returns a list of human-readable problems. Empty means usable.

        Called before training. A configuration that fails validation must never
        produce a metric: a number from an invalid configuration is a fabricated
        measurement, and it would be recorded in the graph as if it were real.
        """
        ...

    def train(
        self,
        dataset_id: str,
        model_id: str,
        params: Dict[str, Any],
        metric: str,
        seed: int,
    ) -> TrialOutcome: ...


@runtime_checkable
class GraphStore(Protocol):
    """Durable persistence for every graph node and relationship the product has."""

    def ensure_schema(self) -> None: ...

    def write_dataset(self, profile: DatasetProfile) -> None: ...

    def write_models(self, models: List[Any]) -> None:
        """Persists the fixed catalog and its `Param` nodes."""
        ...

    def write_run(self, run: RunState) -> None: ...

    def write_trial(self, trial: TrialOutcome, decision: Decision) -> None:
        """Atomically records the trial, its settings, the decision and the
        `(Decision)-[:CITES]->` edges the decision was grounded in."""
        ...

    def write_log(self, entry: LogEntry) -> None: ...

    def read_graph(
        self, run_id: Optional[str] = None, dataset_id: Optional[str] = None
    ) -> GraphSnapshot: ...

    def read_logs(self, run_id: str, after_seq: int = 0) -> List[LogEntry]: ...


@runtime_checkable
class DecisionMaker(Protocol):
    """Laya: turns what the graph already knows into the next configuration."""

    def propose(self, run: RunState, graph: GraphSnapshot) -> Proposal: ...

    def explain(self, run_id: str, trial_id: str) -> WhyRecord: ...


@runtime_checkable
class RunLogger(Protocol):
    """Three sinks, one call: memory, the graph, and every live subscriber."""

    def append(
        self,
        run_id: str,
        level: str,
        stage: str,
        message: str,
        step: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LogEntry: ...

    def publish(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        """Broadcasts a non-log notification without persisting a line."""
        ...

    def history(self, run_id: str, after_seq: int = 0) -> List[LogEntry]: ...

    def subscribe(self, run_id: str) -> "asyncio.Queue[LogEvent]": ...

    def unsubscribe(self, run_id: str, queue: "asyncio.Queue[LogEvent]") -> None: ...


__all__ = [
    "DataStore",
    "Trainer",
    "GraphStore",
    "DecisionMaker",
    "RunLogger",
    "Decision",
    "DatasetProfile",
    "GraphEdge",
    "GraphNode",
    "GraphSnapshot",
    "LogEntry",
    "LogEvent",
    "Proposal",
    "RunState",
    "TrialOutcome",
    "WhyRecord",
    "AsyncIterator",
]