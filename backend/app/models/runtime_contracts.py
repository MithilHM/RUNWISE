"""Autonomous Graph-Guided ML Experiment Runtime Data Contracts.

Defines Pydantic models for:
- RuntimeObjective: Task configuration, search space, and budgets
- CandidateConfig: Hyperparameter candidate proposals
- LayaDecision: LLM typed decision evaluator choices & hypotheses
- TrialResult: Sandboxed execution metrics and status
- RuntimeStepEvent: Closed-loop streaming state events
- Additional response contracts for runtime status and ablation benchmarks
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


# Metrics whose best value is the smallest. Anything absent from this set is
# maximised. Both the runtime loop and the ablation benchmark compare in this
# direction, so a loss target keeps the lowest loss rather than the highest.
MINIMIZED_METRICS = frozenset({"val_loss", "val_log_loss"})


def metric_is_minimized(metric_name: str | None) -> bool:
    """Whether a lower value of this metric is the better result."""
    return metric_name in MINIMIZED_METRICS


def is_better(candidate: float, incumbent: float | None, minimize: bool) -> bool:
    """Direction-aware comparison used everywhere a run promotes a new best."""
    if incumbent is None:
        return True
    return candidate < incumbent if minimize else candidate > incumbent


class RuntimeObjective(BaseModel):
    """Configuration contract for starting an autonomous runtime optimization loop."""

    model_config = ConfigDict(extra="ignore")

    dataset_name: str = Field(
        default="credit_card_fraud",
        description="Dataset identifier (e.g. 'credit_card_fraud' or 'bank_churn')",
    )
    target_metric: str = Field(
        default="val_pr_auc",
        description="Metric to optimize: 'val_pr_auc', 'val_f1', or 'val_log_loss'",
    )
    model_family: str = Field(
        default="lightgbm",
        description="Model family to run ('lightgbm' or 'xgboost')",
    )
    search_space: Dict[str, List[Any]] = Field(
        default_factory=lambda: {
            "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
            "max_depth": [3, 4, 6, 8],
            "n_estimators": [50, 100, 150, 200],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "scale_pos_weight": [1.0, 5.0, 10.0, 20.0, 50.0],
        },
        description="Hyperparameter ranges for trial candidate generation",
    )
    trial_budget: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of trials to run in this optimization loop",
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0.0,
        description="Total wall-clock timeout in seconds for the entire loop",
    )
    project_id: str = Field(
        default="project_demo",
        description="Project scope identifier in the FalkorDB graph",
    )


class CandidateConfig(BaseModel):
    """Proposed hyperparameter candidate generated from graph evidence or heuristics."""

    model_config = ConfigDict(extra="ignore")

    candidate_id: str = Field(description="Unique candidate identifier, e.g. cand_001")
    hyperparameters: Dict[str, Any] = Field(
        description="Concrete hyperparameter dictionary to evaluate"
    )
    rationale: str = Field(
        description="Why this candidate was chosen (e.g. regime boundary, prior high-AUC neighbor)"
    )
    source_regime_edges: List[str] = Field(
        default_factory=list,
        description="Graph node or edge IDs supporting this candidate configuration",
    )


class LayaDecision(BaseModel):
    """Evaluation and selection decision made by the Laya agent."""

    model_config = ConfigDict(extra="ignore")

    selected_candidate_id: str = Field(
        description="The candidate_id chosen by Laya to train in the current step"
    )
    decision_type: Literal[
        "exploit_promising_regime",
        "explore_boundary",
        "avoid_failure_subgraph",
    ] = Field(
        description="Strategic category of the decision"
    )
    hypothesis_text: str = Field(
        description="Machine-learning hypothesis driving the candidate selection"
    )
    cited_graph_paths: List[str] = Field(
        default_factory=list,
        description="List of FalkorDB node/edge IDs cited as evidence for this decision",
    )
    confidence: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Confidence score for this decision (0.0 to 1.0)",
    )


class TrialResult(BaseModel):
    """Result of running a single candidate trial in the sandboxed ML worker."""

    model_config = ConfigDict(extra="ignore")

    trial_id: str = Field(description="Unique identifier for this trial execution")
    candidate_id: str = Field(description="Candidate identifier that was executed")
    hyperparameters: Dict[str, Any] = Field(
        description="Hyperparameters evaluated in this trial"
    )
    metrics: Dict[str, Optional[float]] = Field(
        default_factory=dict,
        description="Validation metrics including target_metric, loss, and train_time_sec",
    )
    status: Literal["SUCCESS", "FAILED", "PRUNED"] = Field(
        default="SUCCESS",
        description="Trial execution outcome status",
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description="Explanation if trial failed or pruned (e.g. OOM, bad param combination)",
    )
    memory_mb: float = Field(
        default=0.0,
        description="Peak RAM memory consumption in megabytes",
    )


class RuntimeStepEvent(BaseModel):
    """Real-time event yielded during each step of the autonomous optimization loop."""

    model_config = ConfigDict(extra="ignore")

    step_number: int = Field(description="1-indexed iteration step number")
    stage: Literal[
        "QUERYING_GRAPH",
        "GENERATING_CANDIDATES",
        "LAYA_EVALUATING",
        "TRAINING_WORKER",
        "MUTATING_GRAPH",
        "COMPLETED",
    ] = Field(description="Current stage in the closed loop")
    current_best_metric: Optional[float] = Field(
        default=None,
        description="Best value observed so far for the target metric",
    )
    trial_result: Optional[TrialResult] = Field(
        default=None,
        description="Result of the trial executed in this step if worker ran",
    )
    laya_decision: Optional[LayaDecision] = Field(
        default=None,
        description="Decision and hypothesis produced by Laya in this step",
    )
    graph_delta: Dict[str, int] = Field(
        default_factory=lambda: {"nodes_added": 0, "edges_added": 0},
        description="Count of nodes and edges mutated in FalkorDB during this step",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of event emission",
    )
    message: Optional[str] = Field(
        default=None,
        description="Human-readable status summary message",
    )


class RuntimeStatusResponse(BaseModel):
    """Snapshot response of a runtime loop execution."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    status: str  # "RUNNING", "COMPLETED", "FAILED"
    objective: RuntimeObjective
    best_score: Optional[float] = None
    best_hyperparameters: Optional[Dict[str, Any]] = None
    completed_trials: List[TrialResult] = Field(default_factory=list)
    execution_tree: List[Dict[str, Any]] = Field(default_factory=list)
    graph_mutation_stats: Dict[str, int] = Field(
        default_factory=lambda: {"nodes_added": 0, "edges_added": 0}
    )


class AblationResponse(BaseModel):
    """Comparative convergence evaluation between Graph+Laya vs Random Search."""

    model_config = ConfigDict(extra="ignore")

    run_id: str
    budget: int
    laya_curve: List[Dict[str, Any]] = Field(
        description="List of {trial: int, best_score: float, score: float}"
    )
    random_curve: List[Dict[str, Any]] = Field(
        description="List of {trial: int, best_score: float, score: float}"
    )
    laya_best: float
    random_best: float
    improvement_percentage: float
    summary: str
