"""Shared fixtures for the engine test suites.

The scenario is deliberately shaped around the PRD's demo requirements: runs that
are genuinely comparable, runs that look comparable but are not, one dataset
version with a known defect, and at least one experiment the defect cannot touch.
"""

from __future__ import annotations

import pytest

pytest_plugins = ["tests.conftest_graph"]

from app.models.contracts import (
    CorrectionRecord,
    DatasetSummary,
    DatasetVersionSummary,
    EvidenceStatus,
    ExperimentRecord,
    ExperimentStatus,
    MetricDirection,
    MetricValue,
)
from app.services.comparability import ComparabilityEngine
from app.services.llm_adapter import DeterministicLLMClient, ExplanationWriter
from app.services.recommendations import RecommendationEngine, RankingConfig
from tests.fakes.fake_graph import FakeGraphService

PROJECT_ID = "project_demo"

DATASET_ID = "ds_churn"
VERSION_V1 = "dsv_churn_v1"
VERSION_V2 = "dsv_churn_v2"

ACCURACY = "validation_accuracy"
LOSS = "validation_loss"
PROTOCOL = "stratified_5fold"


def make_experiment(
    experiment_id: str,
    *,
    value: float | None,
    project_id: str = PROJECT_ID,
    model_family: str = "gradient_boosting",
    dataset_id: str = DATASET_ID,
    dataset_version_id: str = VERSION_V1,
    dataset_version: str = "v1",
    split: str | None = "validation",
    protocol: str | None = PROTOCOL,
    metric_name: str = ACCURACY,
    direction: MetricDirection = MetricDirection.MAXIMIZE,
    seed: int | None = 1,
    steps: int | None = 1000,
    configuration: dict | None = None,
    preprocessing: dict | None = None,
    status: ExperimentStatus = ExperimentStatus.COMPLETED,
    cost: float | None = 0.8,
    source_reference: str | None = None,
    quality_status: EvidenceStatus | None = None,
) -> ExperimentRecord:
    """One experiment with every field the comparability engine reads."""
    metrics: list[MetricValue] = []
    if metric_name:
        metrics.append(
            MetricValue(
                name=metric_name,
                value=value,
                direction=direction,
                evaluation_split=split,
                evaluation_protocol=protocol,
            )
        )

    return ExperimentRecord(
        id=experiment_id,
        project_id=project_id,
        name=experiment_id.replace("_", " "),
        model_family=model_family,
        dataset_id=dataset_id,
        dataset_version_id=dataset_version_id,
        dataset_version=dataset_version,
        dataset_quality_status=quality_status,
        status=status,
        metrics=metrics,
        configuration=configuration if configuration is not None else {"max_depth": 4},
        preprocessing=preprocessing or {"scaling": "standard"},
        random_seed=seed,
        training_steps=steps,
        compute_cost_gpu_hours=cost,
        source_reference=source_reference or f"runs/{experiment_id}.json",
    )


@pytest.fixture
def dataset_versions() -> list[DatasetVersionSummary]:
    return [
        DatasetVersionSummary(
            id=VERSION_V1, dataset_id=DATASET_ID, version="v1",
            quality_status=EvidenceStatus.OBSERVED,
        ),
        DatasetVersionSummary(
            id=VERSION_V2, dataset_id=DATASET_ID, version="v2",
            quality_status=EvidenceStatus.OBSERVED,
        ),
    ]


@pytest.fixture
def datasets() -> list[DatasetSummary]:
    return [DatasetSummary(id=DATASET_ID, name="churn", description="Customer churn")]


@pytest.fixture
def experiments() -> list[ExperimentRecord]:
    """A history with comparable runs, traps and one clean outlier."""
    return [
        # Comparable set on v1: the anchor and two supporting runs.
        make_experiment("exp_baseline", value=0.900, configuration={"max_depth": 4}),
        make_experiment("exp_depth6", value=0.911, configuration={"max_depth": 6}),
        make_experiment("exp_depth8", value=0.895, configuration={"max_depth": 8}, cost=1.1),
        # Same data, different seed: comparable, but part of the gap is variance.
        make_experiment("exp_seed7", value=0.884, seed=7),
        # Looks better but is a different dataset version: must not win.
        make_experiment(
            "exp_on_v2", value=0.934, dataset_version_id=VERSION_V2, dataset_version="v2",
            configuration={"max_depth": 6}, cost=0.9,
        ),
        # Different evaluation split: not comparable.
        make_experiment("exp_test_split", value=0.952, split="test"),
        # Different family and metric: answers a different question.
        make_experiment(
            "exp_transformer", value=0.4, model_family="transformer", metric_name=LOSS,
            direction=MetricDirection.MINIMIZE, configuration={"n_layers": 4}, cost=3.5,
        ),
        # Unrelated project record: proves scoping, must never be retrieved.
        make_experiment("exp_other_project", value=0.999, project_id="project_other"),
    ]


@pytest.fixture
def correction() -> CorrectionRecord:
    """A known data-quality defect on v1 (PRD 14 demo step 4)."""
    return CorrectionRecord(
        id="corr_leakage",
        project_id=PROJECT_ID,
        source_id=VERSION_V1,
        source_type="DatasetVersion",
        description="Train-validation leakage found in the v1 feature table.",
        created_by="researcher_a",
    )


@pytest.fixture
def graph(experiments, datasets, dataset_versions) -> FakeGraphService:
    return FakeGraphService(
        experiments=experiments, datasets=datasets, dataset_versions=dataset_versions
    )


@pytest.fixture
def disputed_graph(graph, correction) -> FakeGraphService:
    graph.corrections[correction.id] = correction
    return graph


@pytest.fixture
def comparability_engine() -> ComparabilityEngine:
    return ComparabilityEngine()


@pytest.fixture
def engine(graph) -> RecommendationEngine:
    """Deterministic ids keep assertions stable across runs."""
    counter = {"n": 0}

    def next_id() -> str:
        counter["n"] += 1
        return f"rec_test_{counter['n']:02d}"

    return RecommendationEngine(graph, config=RankingConfig(), id_factory=next_id)


@pytest.fixture
def deterministic_writer() -> ExplanationWriter:
    """The no-key path (PRD 15): never claims a model was called."""
    return ExplanationWriter(DeterministicLLMClient())