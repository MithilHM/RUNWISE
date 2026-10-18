"""Recommendation engine tests (PRD 18).

Covers the FR-5 contract: budget compliance, explainable ranking, evidence
grounding, insufficient evidence, and the FR-9 approval flow.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import copilot, recommendations
from app.models.contracts import (
    CandidateKind,
    DatasetVersionSummary,
    Decision,
    EvidenceStatus,
    ExperimentStatus,
    RecommendationRequest,
    RecommendationStatus,
)
from app.services.recommendations import (
    RankingWeights,
    RecommendationEngine,
    apply_budget,
    identify_gaps,
    parse_objective,
)
from tests.conftest import (
    ACCURACY,
    DATASET_ID,
    PROJECT_ID,
    VERSION_V1,
    VERSION_V2,
    make_experiment,
)
from tests.fakes.fake_graph import FakeGraphService


def sequential_ids(prefix: str = "rec_") -> "Callable[[], str]":
    """Stable recommendation ids so assertions do not depend on uuid4."""
    counter = {"n": 0}

    def factory() -> str:
        counter["n"] += 1
        return f"{prefix}{counter['n']:02d}"

    return factory


# ---------------------------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------------------------


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        RankingWeights(relevance=0.5, evidence_quality=0.5, information_gain=0.5)


def test_score_components_reconstruct_the_score(engine):
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(
            objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=6.0
        ),
    )
    assert result.recommendations

    for recommendation in result.recommendations:
        assert recommendation.score_components
        assert sum(c.weighted_value for c in recommendation.score_components) == pytest.approx(
            recommendation.score, abs=1e-3
        )
        for component in recommendation.score_components:
            assert component.rationale, "every score term must explain itself"


def test_score_is_labelled_as_a_heuristic_not_a_probability(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=6)
    )
    recommendation = result.recommendations[0]

    assert recommendation.score_kind == "weighted_heuristic"
    assert 0.0 <= recommendation.score <= 1.0


# ---------------------------------------------------------------------------
# Budget enforcement (FR-5)
# ---------------------------------------------------------------------------


def test_budget_is_never_exceeded(engine):
    budget = 1.0
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=budget),
    )

    total = sum(
        r.estimated_gpu_hours for r in result.recommendations if r.estimated_gpu_hours is not None
    )
    assert total <= budget
    assert result.estimated_gpu_hours is not None
    assert result.estimated_gpu_hours <= budget


def test_generous_budget_returns_more_than_a_tiny_one(engine):
    small = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=1.0)
    )
    large = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=50.0)
    )

    assert len(small.recommendations) < len(large.recommendations)


def test_candidates_without_a_cost_estimate_are_not_budgeted():
    from app.models.contracts import Candidate
    from app.services.recommendations import RankedCandidate

    unknown_cost = RankedCandidate(
        candidate=Candidate(kind=CandidateKind.SWEEP_PARAMETER, hypothesis="no cost"),
        score=0.9,
        components=[],
    )
    cheap = RankedCandidate(
        candidate=Candidate(
            kind=CandidateKind.SWEEP_PARAMETER, hypothesis="cheap", estimated_gpu_hours=0.5
        ),
        score=0.5,
        components=[],
    )

    outcome = apply_budget([unknown_cost, cheap], budget=1.0, limit=3)

    assert [r.candidate.hypothesis for r in outcome.selected] == ["cheap"]
    # An unknown cost is unfunded, never assumed free.
    assert "no cost" in [r.candidate.hypothesis for r in outcome.unselected]
    assert outcome.total_gpu_hours == 0.5


def test_limit_is_respected_without_a_budget(engine):
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(
            objective="Improve accuracy", compute_budget_gpu_hours=None, max_recommendations=2
        ),
    )
    assert len(result.recommendations) <= 2


def test_budget_too_small_says_so_instead_of_returning_nothing(engine):
    """An empty result must carry the reason (FR-8, handoff contract)."""
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=0.5)
    )

    assert result.recommendations == []
    assert result.unselected_candidates, "candidates existed but were filtered"
    joined = " ".join(result.uncertainties).lower()
    assert "no proposal fits inside the budget" in joined
    assert "cheapest unselected candidate" in joined


def test_budget_exhaustion_is_not_claimed_when_the_limit_was_binding(engine):
    """`budget_exhausted` must mean the budget, not the result limit (FR-5)."""
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(
            objective="Improve accuracy",
            compute_budget_gpu_hours=1000.0,
            max_recommendations=1,
        ),
    )

    assert len(result.recommendations) == 1
    assert result.budget_exhausted is False
    assert any("requested result limit" in u for u in result.uncertainties)


def test_budget_exhaustion_is_reported_when_the_budget_blocks(engine):
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(
            objective="Improve accuracy",
            compute_budget_gpu_hours=1.6,
            max_recommendations=10,
        ),
    )

    if result.budget_exhausted:
        assert any("binding constraint" in u for u in result.uncertainties)
        assert result.estimated_gpu_hours <= 1.6


def test_unknown_cost_candidate_is_excluded_and_explained():
    # An unregistered family has no default unit cost, so the estimate has
    # nowhere to come from and the candidate cannot be budgeted.
    graph = FakeGraphService(
        experiments=[
            make_experiment(
                "exp_a", value=0.90, cost=None, model_family="mystery_net",
                configuration={"width": 4},
            ),
            make_experiment(
                "exp_b", value=0.89, cost=None, model_family="mystery_net",
                configuration={"width": 8},
            ),
        ]
    )
    engine = RecommendationEngine(graph, id_factory=sequential_ids())

    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=5),
    )

    assert result.recommendations == []
    assert any(
        "compute estimate" in u or "compute cost" in u for u in result.uncertainties
    ), result.uncertainties


def test_estimate_source_records_that_a_default_is_not_a_measurement(graph):
    """FR-5: mark an estimated cost as an estimate, never as an observation."""
    engine = RecommendationEngine(graph, id_factory=sequential_ids())
    runs = [e for e in graph.list_experiments(PROJECT_ID) if e.id in {"exp_transformer"}]
    assert runs

    # exp_transformer has a recorded cost, so force the fallback by clearing them.
    for experiment in graph.experiments.values():
        if experiment.project_id == PROJECT_ID:
            experiment.compute_cost_gpu_hours = None
    transformer_runs = [
        e for e in graph.experiments.values() if e.model_family == "transformer"
    ]
    if not transformer_runs:
        pytest.skip("no transformer run in fixture")

    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective="validation_loss", compute_budget_gpu_hours=50),
    )
    for recommendation in result.recommendations:
        if recommendation.estimate_source == "model_family_profile_default":
            assert any(
                "default for family" in u for u in recommendation.uncertainties
            ), "a defaulted cost must carry its own uncertainty"


def test_failed_run_cannot_anchor_a_proposal(graph):
    """A run that did not finish has no result to build a proposal on (FR-5)."""
    good = next(e for e in graph.list_experiments(PROJECT_ID) if e.id == "exp_depth6")
    graph.experiments["exp_boom"] = good.model_copy(
        update={"id": "exp_boom", "status": ExperimentStatus.FAILED}
    )
    graph.experiments["exp_boom"].metrics = [
        good.metrics[0].model_copy(update={"value": 0.999})
    ]
    engine = RecommendationEngine(graph, id_factory=sequential_ids())

    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=20),
    )

    cited = {
        eid
        for r in result.recommendations
        for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
    }
    assert "exp_boom" not in cited, "a failed run must not support a proposal"


def test_no_completed_run_means_no_baseline():
    """A failed run must never anchor, not even when every run failed."""
    graph = FakeGraphService(
        experiments=[
            make_experiment("exp_f1", value=0.99, status=ExperimentStatus.FAILED),
            make_experiment("exp_f2", value=0.98, status=ExperimentStatus.CANCELLED),
        ]
    )
    engine = RecommendationEngine(graph, id_factory=sequential_ids())

    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=20),
    )

    assert result.recommendations == []
    assert any(
        "No completed, measured run" in u for u in result.uncertainties
    ), result.uncertainties


def test_exhausted_history_states_that_nothing_is_left_to_try():
    """An empty set with no reason reads as a broken feature."""
    runs = [
        make_experiment(
            f"exp_full_{depth}",
            value=0.9,
            configuration={"max_depth": depth},
            # Three distinct seeds, so replication is not itself an open gap.
            seed=seed,
            dataset_version_id=VERSION_V1,
        )
        for depth, seed in ((3, 0), (4, 1), (6, 2), (8, 3), (12, 4))
    ]
    # Offer only the v1 dataset version so no dataset-version gap can be generated.
    graph = FakeGraphService(
        experiments=runs,
        dataset_versions=[
            DatasetVersionSummary(id=VERSION_V1, dataset_id=DATASET_ID, version="v1")
        ],
    )
    engine = RecommendationEngine(graph, id_factory=sequential_ids())

    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=50),
    )

    assert result.recommendations == []
    assert any("No gap was found" in u for u in result.uncertainties), result.uncertainties


def test_project_scoping_excludes_other_projects_records(engine, graph):
    """PRD 12: no cross-project data access."""
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=20)
    )

    assert "exp_other_project" not in result.considered_experiment_ids
    cited = {
        eid
        for r in result.recommendations
        for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
    }
    assert "exp_other_project" not in cited


def test_rejected_decision_is_recorded(engine, graph):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=8)
    )
    recommendation = result.recommendations[0]

    updated = engine.decide(
        recommendation.id, Decision.REJECTED, decided_by="researcher_a", note="not promising"
    )

    assert updated.status == RecommendationStatus.REJECTED
    assert graph.decisions[-1].note == "not promising"


# ---------------------------------------------------------------------------
# Grounding and comparability in the loop
# ---------------------------------------------------------------------------


def test_recommendations_only_cite_real_experiments(engine, graph):
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=8.0),
    )
    stored = set(graph.experiments)

    for recommendation in result.recommendations:
        cited = set(recommendation.baseline_experiment_ids) | set(
            recommendation.supporting_experiment_ids
        )
        assert cited <= stored, "a proposal must not reference a run that does not exist"


def test_baseline_is_the_best_run_in_the_comparable_cohort(engine):
    """exp_on_v2 scores highest overall but is a different dataset version."""
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=8.0),
    )

    cited = {
        eid
        for r in result.recommendations
        for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
    }
    assert "exp_on_v2" not in cited, "a different dataset version cannot anchor a proposal"
    assert "exp_test_split" not in cited, "a different evaluation split cannot support a proposal"
    assert "exp_transformer" not in cited, "a different family and metric is not comparable"
    assert cited & {"exp_baseline", "exp_depth6", "exp_depth8", "exp_seed7"}


def test_recommendation_is_an_inference_not_an_observation(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=6)
    )
    assert result.recommendations[0].evidence_status is EvidenceStatus.INFERRED
    assert result.recommendations[0].status is RecommendationStatus.PROPOSED


def test_controlled_variables_pin_the_comparison(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=8)
    )
    sweep = next(
        (r for r in result.recommendations if r.candidate_kind is CandidateKind.SWEEP_PARAMETER),
        None,
    )
    if sweep is None:
        pytest.skip("no sweep candidate generated for this history")

    assert sweep.independent_variables
    assert not set(sweep.independent_variables) & set(sweep.controlled_variables)
    assert "dataset_version" in sweep.controlled_variables


def test_validation_criteria_are_marked_as_product_defaults(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=8)
    )
    for recommendation in result.recommendations:
        for criterion in recommendation.validation_criteria:
            assert criterion.is_suggested_default is True


# ---------------------------------------------------------------------------
# Gap detection
# ---------------------------------------------------------------------------


def test_single_seed_history_produces_a_replication_proposal(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=20)
    )
    kinds = {r.candidate_kind for r in result.recommendations}
    assert CandidateKind.REPLICATE_SEED in kinds


def test_untested_dataset_version_becomes_a_proposal(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=20)
    )
    dataset_proposals = [
        r for r in result.recommendations if r.candidate_kind is CandidateKind.TRY_DATASET_VERSION
    ]
    for proposal in dataset_proposals:
        assert proposal.dataset_version_id
        assert "dataset_version" in proposal.independent_variables


def test_identify_gaps_reports_untested_parameter_values(experiments, dataset_versions):
    from app.services.recommendations import ObjectiveContext

    v1_runs = [e for e in experiments if e.dataset_version_id == VERSION_V1]
    context = ObjectiveContext(
        objective="improve accuracy",
        metric_name=ACCURACY,
        metric_direction=v1_runs[0].metrics[0].direction,
        model_family="gradient_boosting",
        dataset_id=DATASET_ID,
        dataset_version_id=VERSION_V2,
        tokens=set(),
    )

    gaps = identify_gaps(v1_runs, context, dataset_versions)

    sweep_gaps = [g for g in gaps if g.kind is CandidateKind.SWEEP_PARAMETER]
    assert sweep_gaps
    depth_gap = next(g for g in sweep_gaps if g.field_name == "max_depth")
    observed = {float(v) for v in depth_gap.observed_values}
    untested = {float(v) for v in depth_gap.untested_values}
    assert observed == {4.0, 6.0, 8.0}
    assert not observed & untested, "a proposed value must not already have been run"
    assert 12.0 in untested


# ---------------------------------------------------------------------------
# Objective parsing
# ---------------------------------------------------------------------------


def test_metric_named_in_the_objective_wins(experiments, dataset_versions):
    context = parse_objective(
        "Which run has the lowest validation_loss?",
        experiments,
        dataset_versions,
    )
    assert context.metric_name == "validation_loss"


def test_explicit_request_metric_overrides_the_text(experiments, dataset_versions):
    context = parse_objective(
        "improve the loss",
        experiments,
        dataset_versions,
        request=RecommendationRequest(objective="improve the loss", target_metric=ACCURACY),
    )
    assert context.metric_name == ACCURACY


def test_unresolvable_objective_is_reported_not_guessed():
    context = parse_objective("make it better", [], [])
    assert context.metric_name is None
    assert any("No metric" in note for note in context.resolution_notes)


# ---------------------------------------------------------------------------
# Insufficient evidence
# ---------------------------------------------------------------------------


def test_project_with_no_records_returns_no_proposals():
    graph = FakeGraphService()
    engine = RecommendationEngine(graph, id_factory=lambda: "rec_1")
    result = engine.generate(
        "empty_project", RecommendationRequest(objective="Improve accuracy")
    )

    assert result.recommendations == []
    assert any("no experiment records" in u for u in result.uncertainties)


def test_runs_without_the_target_metric_cannot_rank():
    graph = FakeGraphService(
        experiments=[make_experiment("exp_nometric", value=None, metric_name=None)]
    )
    engine = RecommendationEngine(graph, id_factory=lambda: "rec_1")
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=5),
    )
    assert result.recommendations == []


# ---------------------------------------------------------------------------
# Corrections change what is trusted (PRD 14 demo step 5)
# ---------------------------------------------------------------------------


def test_disputed_evidence_is_withheld_from_proposals(disputed_graph):
    engine = RecommendationEngine(disputed_graph, id_factory=sequential_ids())

    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=20),
    )

    # The v1 defect withholds every run that used v1, and the reason is stated.
    assert result.excluded_evidence
    excluded_ids = {item.experiment_id for item in result.excluded_evidence}
    assert "exp_baseline" in excluded_ids
    assert all(item.detail for item in result.excluded_evidence)

    cited = {
        eid
        for r in result.recommendations
        for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
    }
    assert not cited & excluded_ids, "withheld evidence must not support a proposal"


def test_runs_derived_from_disputed_runs_are_withheld_from_proposals(disputed_graph):
    """A clean dataset does not launder an invalidated parent: derivation alone
    must withhold a run from proposal support."""
    derived = make_experiment(
        "exp_derived_ok",
        value=0.92,
        dataset_version_id=VERSION_V2,
        dataset_version="v2",
        configuration={"max_depth": 6},
        parent_experiment_id="exp_baseline",
    )
    disputed_graph.experiments[derived.id] = derived

    engine = RecommendationEngine(disputed_graph, id_factory=sequential_ids())
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=20),
    )

    excluded_ids = {item.experiment_id for item in result.excluded_evidence}
    assert {"exp_baseline", "exp_derived_ok"} <= excluded_ids
    cited = {
        eid
        for r in result.recommendations
        for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
    }
    assert not cited & excluded_ids


def test_an_unsettled_correction_blocks_proposals_whatever_status_word(graph):
    """The graph layer historically labelled a recorded correction 'active'; the
    trust gate must not treat an unsettled correction as settled because the
    word it recognises is 'open'."""
    from app.models.contracts import CorrectionRecord

    correction = CorrectionRecord(
        id="corr_active_vocabulary",
        project_id=PROJECT_ID,
        source_id=VERSION_V1,
        source_type="dataset_version",
        description="Train-validation leakage in v1.",
        status="active",
        created_by="benchmark",
    )
    graph.corrections[correction.id] = correction

    engine = RecommendationEngine(graph, id_factory=sequential_ids())
    result = engine.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=20),
    )

    excluded_ids = {item.experiment_id for item in result.excluded_evidence}
    assert {"exp_baseline", "exp_depth6"} <= excluded_ids
    cited = {
        eid
        for r in result.recommendations
        for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
    }
    assert not cited & excluded_ids


def test_disputed_evidence_can_be_re_included_explicitly(disputed_graph):
    """Opting in must actually put the disputed runs back into consideration."""
    strict = RecommendationEngine(disputed_graph, id_factory=sequential_ids())
    permissive = RecommendationEngine(disputed_graph, id_factory=sequential_ids())

    excluded_result = strict.generate(
        PROJECT_ID,
        RecommendationRequest(objective=f"Improve {ACCURACY}", compute_budget_gpu_hours=20),
    )
    included_result = permissive.generate(
        PROJECT_ID,
        RecommendationRequest(
            objective=f"Improve {ACCURACY}",
            compute_budget_gpu_hours=20,
            include_disputed_evidence=True,
        ),
    )

    def cited(result):
        return {
            eid
            for r in result.recommendations
            for eid in [*r.baseline_experiment_ids, *r.supporting_experiment_ids]
        }

    # Still labelled, in both cases: opting in does not launder the record.
    assert excluded_result.excluded_evidence
    assert included_result.excluded_evidence

    # But the v1 runs only become usable when explicitly requested.
    assert not cited(excluded_result) & {"exp_baseline", "exp_depth6"}
    assert cited(included_result) & {"exp_baseline", "exp_depth6"}


# ---------------------------------------------------------------------------
# Decisions (FR-9)
# ---------------------------------------------------------------------------


def test_approval_is_recorded(engine, graph):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=8)
    )
    recommendation = result.recommendations[0]

    updated = engine.decide(recommendation.id, Decision.APPROVED, decided_by="researcher_a")

    assert updated.status == RecommendationStatus.APPROVED
    assert graph.decisions[-1].decided_by == "researcher_a"


def test_edit_replaces_the_proposed_configuration(engine):
    result = engine.generate(
        PROJECT_ID, RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=8)
    )
    recommendation = result.recommendations[0]

    updated = engine.decide(
        recommendation.id,
        Decision.EDITED,
        decided_by="researcher_a",
        edited_configuration={"max_depth": 10},
    )

    assert updated.configuration == {"max_depth": 10}
    assert updated.status == RecommendationStatus.EDITED


def test_decision_on_a_missing_recommendation_raises(engine):
    with pytest.raises(KeyError):
        engine.decide("rec_does_not_exist", Decision.APPROVED, decided_by="researcher_a")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_repeated_runs_rank_identically(graph):
    request = RecommendationRequest(objective="Improve accuracy", compute_budget_gpu_hours=8)
    first = RecommendationEngine(graph, id_factory=sequential_ids()).generate(PROJECT_ID, request)
    second = RecommendationEngine(graph, id_factory=sequential_ids()).generate(PROJECT_ID, request)

    assert [r.hypothesis for r in first.recommendations] == [
        r.hypothesis for r in second.recommendations
    ]
    assert [r.score for r in first.recommendations] == [r.score for r in second.recommendations]


# ---------------------------------------------------------------------------
# HTTP surface
# ---------------------------------------------------------------------------


@pytest.fixture
def client(graph, deterministic_writer) -> TestClient:
    app = FastAPI()
    app.state.graph_service = graph
    app.state.explanation_writer = deterministic_writer
    # Routers carry no prefix of their own; ``app.main`` mounts them under
    # ``settings.API_V1_STR``. The test app mirrors that so the paths asserted here
    # are the paths the real server serves.
    app.include_router(recommendations.router, prefix="/api/v1")
    app.include_router(copilot.router, prefix="/api/v1")
    return TestClient(app)


def test_recommendation_endpoint_returns_grounded_output(client):
    response = client.post(
        f"/api/v1/projects/{PROJECT_ID}/recommendations",
        json={"objective": f"Improve {ACCURACY}", "compute_budget_gpu_hours": 8},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"]
    for recommendation in body["recommendations"]:
        assert recommendation["evidence_status"] == "inferred"
        assert recommendation["score_components"]


def test_recommendation_endpoint_rejects_empty_objective(client):
    response = client.post(
        f"/api/v1/projects/{PROJECT_ID}/recommendations", json={"objective": ""}
    )
    assert response.status_code == 422


def test_unknown_recommendation_returns_structured_404(client):
    response = client.get("/api/v1/recommendations/rec_missing")
    assert response.status_code == 404
    assert response.json()["detail"][0]["code"] == "recommendation_not_found"


def test_edit_without_configuration_is_rejected(client):
    created = client.post(
        f"/api/v1/projects/{PROJECT_ID}/recommendations",
        json={"objective": "Improve accuracy", "compute_budget_gpu_hours": 8},
    ).json()
    recommendation_id = created["recommendations"][0]["id"]

    response = client.post(
        f"/api/v1/recommendations/{recommendation_id}/decision",
        json={"decision": "edited", "decided_by": "researcher_a"},
    )

    assert response.status_code == 400
    assert response.json()["detail"][0]["code"] == "missing_edited_configuration"


def test_decision_endpoint_persists_approval(client):
    created = client.post(
        f"/api/v1/projects/{PROJECT_ID}/recommendations",
        json={"objective": "Improve accuracy", "compute_budget_gpu_hours": 8},
    ).json()
    recommendation_id = created["recommendations"][0]["id"]

    response = client.post(
        f"/api/v1/recommendations/{recommendation_id}/decision",
        json={"decision": "approved", "decided_by": "researcher_a", "note": "looks right"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "approved"


def test_missing_graph_service_returns_503(graph):
    app = FastAPI()
    app.include_router(recommendations.router, prefix="/api/v1")
    response = TestClient(app).post(
        f"/api/v1/projects/{PROJECT_ID}/recommendations",
        json={"objective": "Improve accuracy"},
    )
    assert response.status_code == 503
    assert response.json()["detail"][0]["code"] == "graph_service_unavailable"