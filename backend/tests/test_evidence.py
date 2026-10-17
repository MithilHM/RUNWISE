"""Evidence grounding and copilot tests (PRD 18).

The product's central trust claim is that an answer cites only what the backend
retrieved. These tests try to break that claim, plus check the no-LLM demo path
from PRD 15.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import copilot
from app.models.contracts import (
    AnswerMode,
    ClaimStatus,
    ComparabilityReasonCode,
    ComparabilityStatus,
    EvidenceStatus,
    ExperimentRecord,
    MetricValue,
    QueryRequest,
)
from app.services.comparability import ComparabilityEngine
from app.services.evidence import (
    EvidenceLedger,
    GroundingViolation,
    describe_paths,
    evidence_status_for,
    partition_by_trust,
)
from app.services.llm_adapter import (
    DeterministicLLMClient,
    ExplanationWriter,
    LLMResult,
    LLMUnavailable,
    OpenAICompatibleLLMClient,
    build_llm_client,
)
from tests.conftest import (
    ACCURACY,
    PROJECT_ID,
    VERSION_V1,
    VERSION_V2,
    make_experiment,
)
from tests.fakes.fake_graph import FakeGraphService


# ---------------------------------------------------------------------------
# The ledger refuses unbacked citations
# ---------------------------------------------------------------------------


def test_ledger_rejects_an_unknown_citation(experiments):
    ledger = EvidenceLedger()
    ledger.register(experiments)

    with pytest.raises(GroundingViolation) as excinfo:
        ledger.ref("exp_invented")

    assert "exp_invented" in excinfo.value.unknown_citations


def test_report_flags_unknown_citations_without_raising(experiments):
    ledger = EvidenceLedger()
    ledger.register(experiments)

    report = ledger.report(["exp_baseline", "exp_invented", "exp_also_fake"])

    assert report.is_grounded is False
    assert set(report.unknown_citations) == {"exp_invented", "exp_also_fake"}
    assert report.citation_coverage == pytest.approx(1 / 3, abs=1e-4)
    assert report.available_experiment_count == len(experiments)


def test_verify_raises_on_unknown_citation(experiments):
    ledger = EvidenceLedger()
    ledger.register(experiments)

    with pytest.raises(GroundingViolation):
        ledger.verify(["exp_baseline", "exp_invented"])

    assert ledger.verify(["exp_baseline"]).is_grounded is True


def test_claim_cannot_be_built_from_an_unknown_id(experiments):
    ledger = EvidenceLedger()
    ledger.register(experiments)

    with pytest.raises(GroundingViolation):
        ledger.claim("claim_bad", "Something unsupported.", ["exp_invented"])


def test_claim_resolves_metric_citations(experiments):
    ledger = EvidenceLedger()
    ledger.register(experiments)

    claim = ledger.claim(
        "claim_best",
        "exp_depth6 reached 0.911.",
        [("exp_depth6", ACCURACY)],
    )

    assert claim.support[0].experiment_id == "exp_depth6"
    assert claim.support[0].metric_name == ACCURACY
    assert claim.support[0].metric_value == 0.911
    assert claim.support[0].source_reference
    assert claim.evidence_status is EvidenceStatus.OBSERVED


def test_empty_citation_list_has_zero_coverage(experiments):
    ledger = EvidenceLedger()
    ledger.register(experiments)

    report = ledger.report([])

    assert report.citation_coverage == 0.0
    assert report.is_grounded is True


# ---------------------------------------------------------------------------
# Epistemic status
# ---------------------------------------------------------------------------


def test_experiment_under_correction_is_disputed(experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")

    status = evidence_status_for(baseline, disputed_experiment_ids={"exp_baseline"})

    assert status is EvidenceStatus.DISPUTED


def test_disputed_dataset_version_disqualifies_its_runs(experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")

    status = evidence_status_for(
        baseline, corrected_dataset_version_ids={VERSION_V1}
    )

    assert status is EvidenceStatus.DISPUTED


def test_untouched_record_stays_observed(experiments):
    baseline = next(e for e in experiments if e.id == "exp_baseline")

    assert evidence_status_for(baseline) is EvidenceStatus.OBSERVED


def test_partition_states_why_evidence_was_withheld(experiments):
    trusted, excluded, statuses = partition_by_trust(
        experiments, disputed_experiment_ids={"exp_baseline"}
    )

    assert "exp_baseline" not in [e.id for e in trusted]
    assert statuses["exp_baseline"] is EvidenceStatus.DISPUTED
    withdrawn = next(item for item in excluded if item.experiment_id == "exp_baseline")
    assert withdrawn.reason == "experiment_disputed"
    assert "challenges this experiment" in withdrawn.detail


def test_partition_keeps_unaffected_records(experiments):
    """PRD 14 requires at least one experiment the defect cannot touch."""
    _, excluded, _ = partition_by_trust(experiments, corrected_dataset_version_ids={VERSION_V1})

    withdrawn = {item.experiment_id for item in excluded}

    # exp_on_v2 never touched v1, so a v1 defect must not reach it.
    assert "exp_on_v2" not in withdrawn
    # The transformer run did use v1, so it is caught even though the family
    # differs. Dependency follows the data, not the architecture.
    assert "exp_transformer" in withdrawn


@pytest.mark.parametrize(
    "grade", [EvidenceStatus.SUPERSEDED, EvidenceStatus.UNRESOLVED, EvidenceStatus.DISPUTED]
)
def test_untrusted_quality_grades_are_not_promoted_to_observed(grade):
    """FR-7: a superseded or unresolved dataset version is not an observation."""
    record = make_experiment("exp_graded", value=0.9, quality_status=grade)

    status = evidence_status_for(record)

    assert status is grade
    trusted, excluded, statuses = partition_by_trust([record])
    assert trusted == []
    assert statuses["exp_graded"] is grade
    assert excluded[0].status is grade
    assert excluded[0].detail


def test_confirmed_quality_grade_passes_through():
    record = make_experiment("exp_signed", value=0.9, quality_status=EvidenceStatus.CONFIRMED)

    assert evidence_status_for(record) is EvidenceStatus.CONFIRMED
    trusted, excluded, _ = partition_by_trust([record])
    assert [e.id for e in trusted] == ["exp_signed"]
    assert excluded == []


def test_untrusted_quality_grade_qualifies_a_comparison(experiments):
    """A superseded dataset version must not pass as clean comparable data."""

    engine = ComparabilityEngine()
    baseline = next(e for e in experiments if e.id == "exp_baseline")
    superseded = baseline.model_copy(
        update={"id": "exp_sup", "dataset_quality_status": EvidenceStatus.SUPERSEDED}
    )

    result = engine.assess(baseline, superseded, metric_name=ACCURACY)

    found = {r.code for r in result.reasons}
    assert ComparabilityReasonCode.DISPUTED_EVIDENCE in found
    assert result.status is ComparabilityStatus.COMPARABLE_WITH_CAVEATS


# ---------------------------------------------------------------------------
# Graph paths
# ---------------------------------------------------------------------------


def test_paths_are_deduplicated_and_bounded():
    edges = [
        {"from": "dsv_v1", "to": "exp_a", "type": "USES_DATASET_VERSION"},
        {"from": "dsv_v1", "to": "exp_a", "type": "USES_DATASET_VERSION"},
        {"from": "exp_a", "to": "claim_1", "type": "BASED_ON"},
    ]

    paths = describe_paths(edges, source_ids=["dsv_v1"])

    assert len({tuple(p.nodes) for p in paths}) == len(paths)
    assert any(p.nodes == ["dsv_v1", "exp_a", "claim_1"] for p in paths)
    assert all(len(p.nodes) <= 7 for p in paths)


def test_paths_do_not_loop_forever():
    edges = [
        {"from": "a", "to": "b", "type": "DEPENDS_ON"},
        {"from": "b", "to": "a", "type": "DEPENDS_ON"},
    ]

    paths = describe_paths(edges, source_ids=["a"], limit=20)

    assert len(paths) <= 20
    assert all("a" not in p.nodes[1:] for p in paths)


# ---------------------------------------------------------------------------
# LLM adapter
# ---------------------------------------------------------------------------


def test_no_key_yields_the_deterministic_client():
    client = build_llm_client({"RUNWISE_LLM_API_KEY": ""})
    assert isinstance(client, DeterministicLLMClient)
    assert client.is_available is False


def test_blank_key_is_treated_as_absent():
    assert isinstance(build_llm_client({"RUNWISE_LLM_API_KEY": "   "}), DeterministicLLMClient)


def test_key_yields_an_openai_compatible_client():
    client = build_llm_client(
        {
            "RUNWISE_LLM_API_KEY": "secret",
            "RUNWISE_LLM_MODEL": "some-model",
            "RUNWISE_LLM_BASE_URL": "https://example.invalid/v1",
        }
    )
    assert isinstance(client, OpenAICompatibleLLMClient)
    assert client.is_available is True


def test_deterministic_client_refuses_to_pretend():
    with pytest.raises(LLMUnavailable):
        DeterministicLLMClient().complete("explain this")


def test_writer_falls_back_when_no_client_is_configured():
    writer = ExplanationWriter(DeterministicLLMClient())

    result = writer.write(
        question="what happened?",
        facts={"best_run": {"id": "exp_a"}},
        deterministic_text="exp_a is the best run.",
        allowed_ids=["exp_a"],
    )

    assert result.mode is AnswerMode.DETERMINISTIC
    assert result.llm_used is False
    assert result.text == "exp_a is the best run."


class _StubLLM:
    """A client that is available and returns canned prose."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.prompts: list[str] = []

    @property
    def is_available(self) -> bool:
        return True

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 700) -> LLMResult:
        self.prompts.append(prompt)
        return LLMResult(text=self._text, model="stub", used=True)


def test_writer_accepts_grounded_prose():
    stub = _StubLLM("The run exp_a leads the comparable set.")
    writer = ExplanationWriter(stub)

    result = writer.write(
        question="who leads?",
        facts={"best_run": {"id": "exp_a"}},
        deterministic_text="deterministic text",
        allowed_ids=["exp_a", "exp_b"],
    )

    assert result.mode is AnswerMode.LLM
    assert result.llm_used is True
    assert result.text.startswith("The run exp_a")
    assert "exp_a" in stub.prompts[0]


def test_writer_rejects_prose_naming_an_unknown_run():
    stub = _StubLLM("exp_invented achieved the best score.")
    writer = ExplanationWriter(stub)

    result = writer.write(
        question="who leads?",
        facts={"best_run": {"id": "exp_a"}},
        deterministic_text="deterministic text",
        allowed_ids=["exp_a"],
    )

    assert result.mode is AnswerMode.DETERMINISTIC
    assert result.llm_used is False
    assert result.text == "deterministic text"
    assert "fabrication" in (result.fallback_reason or "") or "outside" in (
        result.fallback_reason or ""
    )


def test_writer_survives_a_failing_client():
    class _BrokenLLM:
        @property
        def is_available(self) -> bool:
            return True

        def complete(self, prompt: str, *, system: str | None = None,
                     max_tokens: int = 700) -> LLMResult:
            raise LLMUnavailable("network down")

    result = ExplanationWriter(_BrokenLLM()).write(
        question="q",
        facts={},
        deterministic_text="fallback",
        allowed_ids=[],
    )

    assert result.mode is AnswerMode.DETERMINISTIC
    assert "network down" in (result.fallback_reason or "")


def test_writer_rejects_text_from_a_client_that_made_no_call():
    """PRD 15: never present an incomplete call as a model being consulted."""
    class _SilentLLM:
        @property
        def is_available(self) -> bool:
            return True

        def complete(self, prompt: str, *, system: str | None = None,
                     max_tokens: int = 700) -> LLMResult:
            return LLMResult(text="exp_baseline leads.", model="stub", used=False)

    result = ExplanationWriter(_SilentLLM()).write(
        question="who leads?",
        facts={"best_run": {"id": "exp_baseline"}},
        deterministic_text="deterministic text",
        allowed_ids=["exp_baseline"],
    )

    assert result.mode is AnswerMode.DETERMINISTIC
    assert result.llm_used is False
    assert result.text == "deterministic text"
    assert "no completed call" in (result.fallback_reason or "")


def test_writer_rejects_a_fabricated_metric_value():
    """FR-3 / PRD 12: a figure absent from the evidence is a fabrication."""
    stub = _StubLLM("Validation accuracy reached 0.9999 across the history.")

    result = ExplanationWriter(stub).write(
        question="how good is it?",
        facts={"best_run": {"metrics": {"validation_accuracy": 0.911}}},
        deterministic_text="deterministic text",
        allowed_ids=["exp_a"],
    )

    assert result.mode is AnswerMode.DETERMINISTIC
    assert result.text == "deterministic text"
    assert "figures that do not appear" in (result.fallback_reason or "")


def test_writer_accepts_a_restated_evidence_number():
    """Restating a recorded figure is legitimate and must not be rejected."""
    stub = _StubLLM("The comparable set leads with exp_a at 0.911 validation accuracy.")

    result = ExplanationWriter(stub).write(
        question="how good is it?",
        facts={"best_run": {"id": "exp_a", "metrics": {"validation_accuracy": 0.911}}},
        deterministic_text="deterministic text",
        allowed_ids=["exp_a"],
    )

    assert result.mode is AnswerMode.LLM
    assert result.llm_used is True


def test_identifier_check_follows_the_graphs_naming_convention():
    """The guard must not depend on ids starting with ``exp``."""
    stub = _StubLLM("run-9999 was the best run overall.")

    result = ExplanationWriter(stub).write(
        question="who leads?",
        facts={"best_run": {"id": "run-0042"}},
        deterministic_text="deterministic text",
        allowed_ids=["run-0042"],
    )

    assert result.mode is AnswerMode.DETERMINISTIC
    assert "run-9999" in (result.fallback_reason or "")


def test_identifier_check_accepts_a_known_non_exp_id():
    stub = _StubLLM("The best run is run-0042.")

    result = ExplanationWriter(stub).write(
        question="who leads?",
        facts={"best_run": {"id": "run-0042"}},
        deterministic_text="deterministic text",
        allowed_ids=["run-0042"],
    )

    assert result.mode is AnswerMode.LLM


# ---------------------------------------------------------------------------
# Copilot answers
# ---------------------------------------------------------------------------


def _client(graph, writer=None) -> TestClient:
    app = FastAPI()
    app.state.graph_service = graph
    if writer is not None:
        app.state.explanation_writer = writer
    app.include_router(copilot.router)
    return TestClient(app)


def _ask(client, question: str, budget: float | None = None):
    payload: dict = {"question": question}
    if budget is not None:
        payload["compute_budget_gpu_hours"] = budget
    response = client.post(f"/api/v1/projects/{PROJECT_ID}/query", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def test_answer_without_records_says_so(graph):
    empty = FakeGraphService()
    answer = _ask(_client(empty), "Which run is best?")

    assert answer["insufficient_evidence"] is True
    assert answer["recommendations"] == []
    assert "no experiment records" in answer["answer"]
    assert answer["llm_used"] is False


def test_answer_is_grounded(graph):
    answer = _ask(_client(graph), "Which experiments are comparable?", budget=8.0)

    assert answer["grounding"]["is_grounded"] is True
    assert answer["grounding"]["unknown_citations"] == []
    assert answer["grounding"]["citation_coverage"] == 1.0


def test_answer_only_cites_retrieved_experiments(graph):
    answer = _ask(_client(graph), f"Which run has the best {ACCURACY}?", budget=8.0)
    retrieved = {e.id for e in graph.list_experiments(PROJECT_ID)}
    cited = set(answer["grounding"]["cited_experiment_ids"])

    assert cited <= retrieved


def test_answer_explains_why_a_run_was_excluded(graph):
    """FR-8: a non-comparable run must appear with its reason, not vanish."""
    answer = _ask(_client(graph), "Which runs are comparable?", budget=8.0)

    assessed = {
        comparison["right_experiment_id"] or comparison["left_experiment_id"]
        for comparison in answer["comparisons"]
    }
    assert "exp_on_v2" in assessed
    assert "exp_test_split" in assessed

    v2 = next(c for c in answer["comparisons"] if c["right_experiment_id"] == "exp_on_v2")
    assert v2["status"] == "NOT_COMPARABLE"
    assert any("DATASET_VERSION_MISMATCH" in r["code"] for r in v2["reasons"])
    assert v2["explanation"]


def test_answer_reports_the_real_epistemic_grade_of_its_best_run():
    """A claim carries the record's actual grade, not a default (FR-7)."""
    graph = FakeGraphService(
        experiments=[
            make_experiment("exp_ok", value=0.90, configuration={"max_depth": 4}),
            make_experiment(
                "exp_signed", value=0.95, configuration={"max_depth": 8},
                quality_status=EvidenceStatus.CONFIRMED,
            ),
        ]
    )
    answer = _ask(_client(graph), "Which run has the best validation_accuracy?", budget=8.0)

    best = next(c for c in answer["claims"] if c["id"] == "claim_best_run")
    # exp_signed is both the anchor and the highest score, and it is CONFIRMED.
    assert best["support"][0]["experiment_id"] == "exp_signed"
    assert best["evidence_status"] == "confirmed"


def test_superseded_run_is_withheld_not_reported_as_best():
    graph = FakeGraphService(
        experiments=[
            make_experiment("exp_ok", value=0.90, configuration={"max_depth": 4}),
            make_experiment(
                "exp_soft", value=0.95, configuration={"max_depth": 8},
                quality_status=EvidenceStatus.SUPERSEDED,
            ),
        ]
    )
    answer = _ask(_client(graph), "Which run has the best validation_accuracy?", budget=8.0)

    best = next(c for c in answer["claims"] if c["id"] == "claim_best_run")
    assert best["support"][0]["experiment_id"] == "exp_ok"
    withheld = {item["experiment_id"] for item in answer["excluded_evidence"]}
    assert "exp_soft" in withheld


def test_all_records_withheld_does_not_claim_no_records_exist(disputed_graph):
    """Withheld evidence must not be reported as absent evidence."""
    for experiment in disputed_graph.experiments.values():
        if experiment.dataset_version_id == VERSION_V2:
            experiment.dataset_version_id = VERSION_V1

    answer = _ask(_client(disputed_graph), "Which run has the best validation_accuracy?", budget=20.0)

    assert answer["insufficient_evidence"] is True
    assert "no run in this project records" not in answer["answer"].lower()
    assert any("withheld" in m.lower() for m in answer["missing_information"])


def test_answer_states_the_proposal_is_not_a_result(graph):
    answer = _ask(_client(graph), "What should I try next?", budget=8.0)

    assert answer["recommendations"]
    assert "not results" in answer["answer"]


def test_answer_reports_the_budget(graph):
    answer = _ask(_client(graph), "What should I try next?", budget=2.0)

    assert answer["recommendations"]
    total = sum(
        r["estimated_gpu_hours"]
        for r in answer["recommendations"]
        if r["estimated_gpu_hours"] is not None
    )
    assert total <= 2.0


def test_answer_names_withheld_evidence(disputed_graph):
    answer = _ask(_client(disputed_graph), "What should I try next?", budget=20.0)

    assert answer["excluded_evidence"]
    assert "Withheld" in answer["answer"] or "withheld" in answer["answer"]
    assert answer["corrections_considered"]


def test_answer_excludes_disputed_runs_from_the_comparable_set(disputed_graph):
    answer = _ask(_client(disputed_graph), "Which runs are comparable?", budget=8.0)

    assert "exp_baseline" not in answer["comparable_experiment_ids"]


def test_answer_uses_the_deterministic_path_by_default(graph):
    answer = _ask(_client(graph), "Why did performance change?")

    assert answer["mode"] == "deterministic"
    assert answer["llm_used"] is False


def test_answer_may_use_an_injected_llm(graph):
    stub = _StubLLM("The comparable set is led by exp_depth6.")
    answer = _ask(_client(graph, ExplanationWriter(stub)), "Who leads?", budget=8.0)

    assert answer["mode"] == "llm"
    assert answer["llm_used"] is True
    assert answer["grounding"]["is_grounded"] is True


def test_answer_rejects_llm_prose_that_invents_a_run(graph):
    stub = _StubLLM("exp_invented is clearly the best run here.")
    answer = _ask(_client(graph, ExplanationWriter(stub)), "Who leads?", budget=8.0)

    assert answer["mode"] == "deterministic"
    assert "exp_invented" not in answer["answer"]


def test_empty_question_is_rejected(graph):
    response = _client(graph).post(
        f"/api/v1/projects/{PROJECT_ID}/query", json={"question": ""}
    )
    assert response.status_code == 422


def test_negative_budget_is_rejected(graph):
    response = _client(graph).post(
        f"/api/v1/projects/{PROJECT_ID}/query",
        json={"question": "why?", "compute_budget_gpu_hours": -1},
    )
    assert response.status_code == 422


def test_query_request_tolerates_unknown_fields():
    request = QueryRequest(question="why?", unknown_field="ignored")
    assert request.question == "why?"


def test_claim_status_is_never_confirmed_by_the_product(graph):
    """Nothing the product generates may be promoted to confirmed (FR-7)."""
    answer = _ask(_client(graph), "What should I try next?", budget=8.0)

    for claim in answer["claims"]:
        assert claim["status"] != ClaimStatus.CONFIRMED.value
        assert claim["evidence_status"] != EvidenceStatus.CONFIRMED.value


def test_record_without_a_metric_value_is_not_invented():
    unmeasured = make_experiment(
        "exp_blank", value=None, metric_name=ACCURACY
    )

    assert unmeasured.metric(ACCURACY).value is None
    assert unmeasured.metric("nonexistent_metric") is None
    assert all(isinstance(m, MetricValue) for m in unmeasured.metrics)
    assert isinstance(unmeasured, ExperimentRecord)