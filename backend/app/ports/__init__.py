"""Application Ports and Protocols for RUNWISE.

Defined in MEMBER_B.md Section 3.2 (Day 1 Frozen Interface).
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from app.models.contracts import (
    DatasetProfile,
    HarnessDecision as Decision,
    GraphSnapshot,
    LogEntry,
    LogEvent,
    Proposal,
    RunState,
    TrialOutcome,
    WhyRecord,
)

from app.ports.graph import GraphService


@runtime_checkable
class DataStore(Protocol):
    def get_frame(self, dataset_id: str) -> tuple[pd.DataFrame, pd.Series]: ...
    def get_profile(self, dataset_id: str) -> DatasetProfile: ...


@runtime_checkable
class Trainer(Protocol):
    def train(
        self, dataset_id: str, model_id: str, params: dict[str, Any], metric: str, seed: int
    ) -> TrialOutcome: ...
    def validate(self, model_id: str, params: dict[str, Any]) -> list[str]: ...


@runtime_checkable
class DecisionMaker(Protocol):
    def propose(self, run: RunState, graph: GraphSnapshot) -> Proposal: ...
    def explain(self, trial_id: str) -> WhyRecord: ...


@runtime_checkable
class GraphStore(Protocol):
    def write_dataset(self, profile: DatasetProfile) -> None: ...
    def write_run(self, run: RunState) -> None: ...
    def write_trial(self, trial: TrialOutcome, decision: Decision) -> None: ...
    def write_log(self, entry: LogEntry) -> None: ...
    def read_graph(self, run_id: str | None, dataset_id: str | None) -> GraphSnapshot: ...
    def read_logs(self, run_id: str, after_seq: int) -> list[LogEntry]: ...


@runtime_checkable
class RunLogger(Protocol):
    def append(
        self,
        run_id: str,
        level: str,
        stage: str,
        message: str,
        step: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> LogEntry: ...
    def subscribe(self, run_id: str) -> asyncio.Queue[LogEvent]: ...
    def unsubscribe(self, run_id: str, q: asyncio.Queue[LogEvent]) -> None: ...


__all__ = [
    "DataStore",
    "Trainer",
    "DecisionMaker",
    "GraphStore",
    "RunLogger",
    "GraphService",
]
