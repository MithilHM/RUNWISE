"""Pydantic contracts for the RUNWISE harness.

This module is the single source of truth for every shape that crosses a boundary
between the graph layer, the harness loop, the HTTP layer and the frontend.

Field-level note for the frontend owner: `frontend/src/api/types.ts` is a
transcription of this file. Nothing else defines these shapes.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Metrics the harness can optimise. All four are maximised; there is no
# minimised metric in the catalog today, and `MINIMIZED_METRICS` records that
# explicitly rather than leaving callers to assume it.
MetricName = Literal["accuracy", "f1_macro", "roc_auc", "pr_auc"]
MINIMIZED_METRICS: frozenset[str] = frozenset()

# `pr_auc` is defined here as binary-only. Substituting another metric under a
# requested name would return a real-looking number that means something else, so
# a multiclass request is rejected instead.
BINARY_ONLY_METRICS: frozenset[str] = frozenset({"pr_auc"})

ParamKind = Literal["int", "float", "categorical", "bool", "int_or_null"]


def metric_is_minimized(metric_name: str) -> bool:
    """Whether a lower value of this metric is the better result."""
    return metric_name in MINIMIZED_METRICS


def is_number(value: Any) -> bool:
    """A usable measurement: a finite real number.

    NaN is excluded deliberately. Every `NaN < x` and `NaN > x` comparison is
    False, which makes NaN the silent winner of the first comparison it appears
    in and then an unbeatable incumbent forever after. A NaN in this graph would
    become a permanent "best result" that no real trial can displace.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def is_better(candidate: float, incumbent: Optional[float], metric_name: str) -> bool:
    """Direction-aware comparison used everywhere a trial promotes a new best."""
    if not is_number(candidate):
        return False
    if incumbent is None or not is_number(incumbent):
        return is_number(candidate)
    return candidate < incumbent if metric_is_minimized(metric_name) else candidate > incumbent


class ColumnProfile(BaseModel):
    """One column of the uploaded dataset."""

    model_config = ConfigDict(extra="ignore")

    name: str
    dtype: str
    role: Literal["feature", "target"]
    n_missing: int
    n_unique: int


class DroppedColumn(BaseModel):
    """A column removed before training, and the reason it was removed."""

    model_config = ConfigDict(extra="ignore")

    name: str
    reason: Literal["constant", "all_null", "high_cardinality_id", "leakage"]


class DatasetProfile(BaseModel):
    """Everything the harness knows about one static CSV."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    n_rows: int
    n_cols: int
    task: Literal["binary", "multiclass"]
    n_classes: int
    class_balance: Dict[str, float] = Field(default_factory=dict)
    split_hash: str = ""
    created_at: str = ""
    target_inferred: bool = True
    columns: List[ColumnProfile] = Field(default_factory=list)
    dropped_columns: List[DroppedColumn] = Field(default_factory=list)

    @property
    def target(self) -> str:
        """Name of the confirmed or inferred target column."""
        for column in self.columns:
            if column.role == "target":
                return column.name
        return ""


class ParamSpec(BaseModel):
    """Declarative description of one tunable hyperparameter of one model.

    The graph stores these as `Param` nodes, the frontend renders controls from
    them, and the executor clamps to them. One definition, three consumers.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    kind: ParamKind
    low: Optional[float] = None
    high: Optional[float] = None
    step: Optional[float] = None
    log_scale: bool = False
    choices: Optional[List[Any]] = None
    default: Any = None

    def in_domain(self, value: Any) -> bool:
        """Whether a value is usable for this parameter *without* coercion.

        This is the strict test: unlike `clamp`, it never moves a value to make it
        legal. Used on anything that arrived from a client, where a wrong value is
        a mistake to report rather than a slider overshoot to absorb.
        """
        if self.kind in ("int", "float", "int_or_null"):
            if value is None:
                return self.kind == "int_or_null"
            # bool is an int subclass. `True` is not a depth of 1 and not a
            # subsample of 1.0, so it is rejected here as well as in `clamp`.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return False
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                return False
            if self.low is not None and value < self.low:
                return False
            if self.high is not None and value > self.high:
                return False
            if self.step:
                base = float(self.low) if self.low is not None else 0.0
                offset = (float(value) - base) / float(self.step)
                if abs(offset - round(offset)) > 1e-9:
                    return False
            if self.kind in ("int", "int_or_null") and not isinstance(value, int):
                return False
            return True

        if self.kind == "bool":
            return isinstance(value, bool)

        if self.choices is None:
            return True
        # `1 in [True]` is True in Python, so membership alone would accept the
        # wrong type for a numeric choice. Compare types as well.
        return any(value is choice or (
            isinstance(value, bool) != isinstance(choice, bool) and value == choice
        ) for choice in self.choices)

    def clamp(self, value: Any) -> Any:
        """Clamps one value into this parameter's domain.

        Out-of-range input is clamped rather than rejected: a researcher dragging a
        slider past its end means "the most extreme thing available", not "an
        error". A value that is not in the domain at all is a different case and
        raises, because silently substituting would hide a real mistake.
        """
        if self.kind in ("int", "float", "int_or_null"):
            if self.kind == "int_or_null" and value is None:
                return None
            if value is None:
                raise ValueError(f"{self.name} may not be null")
            # bool is an int subclass; True is not a depth of 1.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{self.name} must be numeric, got {value!r}")
            low = -float("inf") if self.low is None else float(self.low)
            high = float("inf") if self.high is None else float(self.high)
            clamped = min(max(float(value), low), high)
            if self.step:
                # Snap to the step grid measured from `low`, not from zero: a
                # grid anchored at 0 would push subsample (low 0.4, step 0.05)
                # down to 0.35, which is outside its own domain.
                base = float(self.low) if self.low is not None else 0.0
                snapped = base + round((clamped - base) / float(self.step)) * float(self.step)
                clamped = min(max(snapped, low), high)
            if self.kind in ("int", "int_or_null"):
                return int(round(clamped))
            return round(clamped, 6)

        if self.kind == "bool":
            if isinstance(value, bool):
                return value
            if isinstance(value, str) and value.lower() in ("true", "false"):
                return value.lower() == "true"
            raise ValueError(f"{self.name} must be a boolean, got {value!r}")

        # categorical
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"{self.name} must be one of {self.choices}, got {value!r}")
        return value


class ModelSpec(BaseModel):
    """One of the three models the harness can train."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    library: str
    description: str
    params: List[ParamSpec] = Field(default_factory=list)

    def param(self, name: str) -> Optional[ParamSpec]:
        """The spec for one parameter, or None when this model has no such knob."""
        for spec in self.params:
            if spec.name == name:
                return spec
        return None


class RunCreate(BaseModel):
    """Request body for starting a run."""

    model_config = ConfigDict(extra="ignore")

    dataset_id: str
    model_ids: List[str] = Field(min_length=1, max_length=3)
    mode: Literal["autonomous", "human"] = "autonomous"
    metric: MetricName = "accuracy"
    budget: int = Field(default=12, ge=1, le=200)


class RunState(BaseModel):
    """Lifecycle state of one harness run."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    dataset_id: str
    mode: Literal["autonomous", "human"]
    status: str = "IDLE"
    step: int = 0
    budget: int = 0
    metric_name: str = "accuracy"
    best_metric: Optional[float] = None
    best_trial_id: Optional[str] = None
    model_ids: List[str] = Field(default_factory=list)
    pending_proposal: Optional["Proposal"] = None
    created_at: str = ""
    updated_at: str = ""


class Proposal(BaseModel):
    """The next configuration the harness intends to run.

    In `human` mode this is what the researcher approves, edits or rejects. In
    `autonomous` mode it is still produced and still logged, so the graph shows
    the same decision record either way.
    """

    model_config = ConfigDict(extra="ignore")

    proposal_id: str
    step: int
    model_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    strategy: str = ""
    why: str = ""
    hypothesis: str = ""
    confidence: float = 0.5
    citations: List[str] = Field(default_factory=list)
    rejected: List[Dict[str, Any]] = Field(default_factory=list)


class TrialOutcome(BaseModel):
    """The result of training one model on one configuration."""

    model_config = ConfigDict(extra="ignore")

    trial_id: str
    run_id: str = ""
    step: int = 0
    model_id: str
    params: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["SUCCESS", "FAILED"] = "SUCCESS"
    metric_name: str = "accuracy"
    metric_value: Optional[float] = None
    metrics: Dict[str, Optional[float]] = Field(default_factory=dict)
    failure_reason: Optional[str] = None
    duration_sec: float = 0.0
    n_features: int = 0
    seed: int = 42
    created_at: str = ""

    @model_validator(mode="after")
    def _check_status_consistency(self) -> "TrialOutcome":
        """`status` must agree with the measurement it carries.

        Enforced here rather than left to callers, because the two ways this goes
        wrong are both invisible later: a SUCCESS with no number gets skipped by
        every ranking query, and a FAILED with a number gets plotted as a result.
        Either one puts a fabrication into the graph with nothing downstream able
        to catch it.
        """
        if self.status == "SUCCESS":
            if not is_number(self.metric_value):
                raise ValueError(
                    "a SUCCESS trial must carry a finite metric_value; "
                    f"got {self.metric_value!r}"
                )
        else:
            if self.metric_value is not None:
                raise ValueError(
                    f"a {self.status} trial must not carry a metric_value; "
                    f"got {self.metric_value!r}"
                )
            if not self.failure_reason:
                raise ValueError("a failed trial must carry a failure_reason")
        return self


class Decision(BaseModel):
    """Who chose this configuration, and on what grounds.

    `actor` is the load-bearing field: it distinguishes a choice Laya made from
    one a researcher made by overriding it, so the graph can show which is which.
    """

    model_config = ConfigDict(extra="ignore")

    decision_id: str
    run_id: str
    step: int
    trial_id: str = ""
    actor: Literal["laya", "researcher"] = "laya"
    strategy: str = ""
    why: str = ""
    hypothesis: str = ""
    confidence: float = 0.5
    citations: List[str] = Field(default_factory=list)
    created_at: str = ""


class EvidenceItem(BaseModel):
    """One graph element a decision was grounded in, flattened for the UI."""

    model_config = ConfigDict(extra="ignore")

    kind: Literal["Model", "Param", "Setting", "Trial", "Dataset", "Run"]
    id: str
    label: str
    detail: str = ""


class WhyRecord(BaseModel):
    """The answer to "why this trial?", with the nodes to light up.

    `highlight` is exactly the set of node ids the decision cited. The frontend
    dims everything outside it.
    """

    model_config = ConfigDict(extra="ignore")

    decision: Decision
    trial: TrialOutcome
    config: Dict[str, Any] = Field(default_factory=dict)
    evidence: List[EvidenceItem] = Field(default_factory=list)
    highlight: List[str] = Field(default_factory=list)


class LogEntry(BaseModel):
    """One persisted log line."""

    model_config = ConfigDict(extra="ignore")

    seq: int = 0
    run_id: str = ""
    step: int = 0
    level: Literal["INFO", "WARN", "ERROR"] = "INFO"
    stage: Literal["ingest", "propose", "decide", "execute", "mutate", "done"] = "execute"
    message: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)
    ts: str = ""


class LogEvent(BaseModel):
    """What rides the SSE stream. `log` carries a full LogEntry; the rest are
    lightweight notifications so the UI can refresh derived views."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["log", "decision", "trial", "graph_delta", "status"]
    payload: Dict[str, Any] = Field(default_factory=dict)


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    type: str
    props: Dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    target: str
    type: str
    props: Dict[str, Any] = Field(default_factory=dict)


class GraphSnapshot(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)


RunState.model_rebuild()