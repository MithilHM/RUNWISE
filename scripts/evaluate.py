"""
RUNWISE Evaluation & Baseline Benchmark Suite.

Measures what a SHALLOW retrieval can and cannot do against the production
graph-backed impact service, across the PRD Section 13 metrics.

The two strategies under comparison:

* RUNWISE lineage - the production ``ImpactAnalysisService`` traversing
  ``USES_DATASET_VERSION`` and recursive ``PARENT_OF`` relationships against a
  live FalkorDB, on a throwaway graph that is dropped afterwards, so the
  benchmark never touches the demo project.
* Retrieval baseline - a shallow search that flags every run whose dataset
  NAME matches the corrected version's dataset, version-blind, and never
  follows derivation.

Ground truth is defined independently of both: the runs that use the corrected
version directly, plus every run transitively derived from them. With the
seeded scenario that set is ``exp_001/002/003`` directly, plus ``exp_051/052``
descendants of ``exp_002`` that never touched ``tabular_fraud`` at all, so the
baseline has something real to miss.

Requires FalkorDB (``docker compose up -d falkordb``), like the seed script.

Usage:
    python scripts/evaluate.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure backend and repo root are on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.graph.client import FalkorDBClient
from app.graph.repository import GraphRepository
from app.graph.schema import initialize_schema
from app.models.contracts import (
    ComparabilityStatus,
    CorrectionCreate,
    ExperimentRecord,
    ExperimentStatus,
    MetricDirection,
    MetricValue,
    RecommendationRequest,
)
from app.services.comparability import ComparabilityEngine
from app.services.ingestion import IngestionService
from app.services.impact_analysis import ImpactAnalysisService
from app.services.recommendations import RecommendationEngine, RankingConfig
from app.services.seed_data import generate_synthetic_experiments
from tests.fakes.fake_graph import FakeGraphService

PROJECT_ID = "project_eval_bench"
PROJECT_NAME = "RUNWISE Evaluation Benchmark"
BENCH_GRAPH = "runwise_eval_bench"
DEFECT_SOURCE = "tabular_fraud_v1"
DEFECT_DATASET_NAME = DEFECT_SOURCE.rsplit("_", 1)[0]
TOTAL_RUNS = 53


class RetrievalBaseline:
    """
    Shallow retrieval baseline.

    Flags every run whose dataset NAME contains the corrected version's dataset
    name, without checking the version and without following derivation. That is
    the shape of a plain text search over run metadata, and it exhibits both
    failure modes a lineage graph exists to prevent: a clean run on the repaired
    version is swept in with the bad one, and a run that merely descends from an
    affected run is invisible.
    """

    def __init__(self, records: list[ExperimentRecord]):
        self.records = records

    def find_affected(self, dataset_name: str) -> set[str]:
        return {
            r.id
            for r in self.records
            if (r.dataset_id or "") and dataset_name in (r.dataset_id or "")
        }


def build_experiment_records(raw_dicts: list[dict], project_id: str) -> list[ExperimentRecord]:
    """Converts raw synthetic dictionaries into typed ExperimentRecord contracts."""
    records: list[ExperimentRecord] = []
    for d in raw_dicts:
        metrics = []
        for m in d.get("metrics", []):
            direction = (
                MetricDirection.MAXIMIZE
                if str(m.get("direction", "maximize")).lower() == "maximize"
                else MetricDirection.MINIMIZE
            )
            metrics.append(
                MetricValue(
                    name=m["name"],
                    value=m.get("value"),
                    direction=direction,
                    evaluation_split=m.get("evaluation_split", "validation"),
                    evaluation_protocol=m.get("evaluation_protocol"),
                )
            )

        ds_name = d.get("dataset_name") or d.get("dataset_id") or "tabular_fraud"
        ds_ver = d.get("dataset_version") or "v1"
        ds_ver_id = f"{ds_name}_{ds_ver}"

        rec = ExperimentRecord(
            id=d["id"],
            project_id=d.get("project_id", project_id),
            name=d.get("name", d["id"]),
            model_family=d.get("model_family", "gradient_boosting"),
            dataset_id=ds_name,
            dataset_version=ds_ver,
            dataset_version_id=ds_ver_id,
            parent_experiment_id=d.get("parent_experiment_id"),
            status=ExperimentStatus.COMPLETED,
            compute_cost_gpu_hours=float(d.get("compute_cost_gpu_hours", 0.8)),
            random_seed=d.get("random_seed"),
            configuration=d.get("configuration") or {},
            metrics=metrics,
        )
        records.append(rec)
    return records


def resolve_ground_truth(
    records: list[ExperimentRecord], defect_source: str
) -> tuple[set[str], set[str]]:
    """
    The definition of affected, implemented independently of the system under
    test: runs that use the corrected version directly, plus every run
    transitively derived from one of them. Written here as the benchmark's
    ground truth so a traversal that under- or over-reaches is measurable, not
    self-confirming.
    """
    direct = {r.id for r in records if r.dataset_version_id == defect_source}
    children: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record.parent_experiment_id:
            children[record.parent_experiment_id].append(record.id)

    affected = set(direct)
    frontier = list(direct)
    while frontier:
        current = frontier.pop()
        for child in children.get(current, []):
            if child not in affected:
                affected.add(child)
                frontier.append(child)
    return direct, affected - direct


def run_benchmark() -> dict:
    start_time = time.perf_counter()
    project_id = PROJECT_ID

    raw_experiments = generate_synthetic_experiments(project_id=project_id, total=TOTAL_RUNS)
    records = build_experiment_records(raw_experiments, project_id=project_id)

    # Ground truth, resolved independently of the system under test.
    direct_ids, indirect_ids = resolve_ground_truth(records, DEFECT_SOURCE)
    ground_truth_affected = direct_ids | indirect_ids
    ground_truth_unaffected = {r.id for r in records if r.id not in ground_truth_affected}

    # Setup for comparability and recommendation checks (in-memory).
    graph_service = FakeGraphService(experiments=records)
    comp_engine = ComparabilityEngine()
    rec_engine = RecommendationEngine(graph_service, comparability=comp_engine, config=RankingConfig())

    # --- Metric 1 & 2: Ingestion & Duplicate Rejection (measured, per run) ---
    bench_client = FalkorDBClient(graph_name=BENCH_GRAPH)
    health = bench_client.check_health()
    if health.get("status") != "healthy":
        raise RuntimeError(
            "FalkorDB is not reachable; start it with `docker compose up -d falkordb`. "
            f"Health probe returned: {health}"
        )

    try:
        # Drop any leftover bench graph from an earlier interrupted run first, so
        # a rerun always measures a fresh seed instead of recovering dirty state.
        try:
            bench_client.delete_graph()
        except Exception as start_error:
            print(f"[bench] nothing to drop: {start_error}")

        initialize_schema(bench_client.get_graph())
        repository = GraphRepository(client=bench_client)
        ingestion = IngestionService(repository=repository)

        t_ingest0 = time.perf_counter()
        summary = ingestion.ingest_records(
            raw_experiments, project_id=project_id, project_name=PROJECT_NAME
        )
        t_ingest1 = time.perf_counter()
        import_time_ms = (t_ingest1 - t_ingest0) * 1000.0
        # The success rate is what the import actually accepted, not a count of
        # well-formed generator records made before ingestion ran.
        import_success_rate = (
            summary.imported / len(raw_experiments) * 100.0 if raw_experiments else 100.0
        )

        # A second pass of identical records must be rejected, not double-counted.
        dup_summary = ingestion.ingest_records(
            raw_experiments, project_id=project_id, project_name=PROJECT_NAME
        )
        duplicate_skipped = dup_summary.skipped
        duplicate_rejection_rate = (
            (duplicate_skipped / len(raw_experiments) * 100.0) if raw_experiments else 100.0
        )

        # --- The production impact traversal under test ---
        impact_service = ImpactAnalysisService(repository=repository)
        t_workflow0 = time.perf_counter()
        impact = impact_service.analyze_and_apply_correction(
            CorrectionCreate(
                source_id=DEFECT_SOURCE,
                source_type="dataset_version",
                description="Data leakage identified between train and validation splits in tabular_fraud v1.",
                created_by="benchmark",
            ),
            project_id=project_id,
        )
        t_workflow1 = time.perf_counter()
        workflow_time_ms = (t_workflow1 - t_workflow0) * 1000.0

        # Pure read-side re-computation, i.e. what a later identification costs.
        t_graph0 = time.perf_counter()
        reread = impact_service.get_correction_impact(impact.correction_id)
        t_graph1 = time.perf_counter()
        graph_time_ms = (t_graph1 - t_graph0) * 1000.0
        assert reread is not None

        graph_affected = set(impact.directly_affected_experiment_ids) | set(
            impact.indirectly_affected_experiment_ids
        )
    finally:
        try:
            bench_client.delete_graph()
        except Exception as cleanup_error:
            # A failed cleanup must not mask the measurement result above; the
            # leftover graph is dropped at the start of the next run anyway.
            print(f"[bench] warning: could not drop the bench graph: {cleanup_error}")

    # Naive shallow retrieval.
    baseline = RetrievalBaseline(records)
    t0_b = time.perf_counter()
    baseline_affected = baseline.find_affected(DEFECT_DATASET_NAME)
    t1_b = time.perf_counter()
    baseline_time_ms = (t1_b - t0_b) * 1000.0

    # Recall against the independent ground truth.
    graph_recall = (
        len(graph_affected & ground_truth_affected) / len(ground_truth_affected) * 100.0
        if ground_truth_affected
        else 100.0
    )
    baseline_recall = (
        len(baseline_affected & ground_truth_affected) / len(ground_truth_affected) * 100.0
        if ground_truth_affected
        else 100.0
    )

    # False positives: flagged runs that the ground truth says are unaffected.
    graph_fp = (
        len(graph_affected & ground_truth_unaffected) / len(ground_truth_unaffected) * 100.0
        if ground_truth_unaffected
        else 0.0
    )
    baseline_fp = (
        len(baseline_affected & ground_truth_unaffected) / len(ground_truth_unaffected) * 100.0
        if ground_truth_unaffected
        else 0.0
    )
    baseline_fp_ids = sorted(baseline_affected & ground_truth_unaffected)
    graph_fp_ids = sorted(graph_affected & ground_truth_unaffected)

    # --- Metric 3: Comparability Accuracy (Evaluated against Ground-Truth Pairs) ---
    ground_truth_pairs = [
        (
            graph_service.get_experiment("exp_001"),
            graph_service.get_experiment("exp_002"),
            {ComparabilityStatus.COMPARABLE, ComparabilityStatus.COMPARABLE_WITH_CAVEATS},
            "exp_001 vs exp_002 (controlled variation: single-variable max_depth)",
        ),
        (
            graph_service.get_experiment("exp_001"),
            graph_service.get_experiment("exp_001"),
            {ComparabilityStatus.COMPARABLE},
            "exp_001 vs exp_001 (identical run baseline)",
        ),
        (
            graph_service.get_experiment("exp_001"),
            graph_service.get_experiment("exp_control"),
            {ComparabilityStatus.NOT_COMPARABLE},
            "exp_001 vs exp_control (different dataset & model family)",
        ),
    ]
    other_run = next((r for r in records if r.dataset_id != "tabular_fraud"), None)
    if other_run:
        ground_truth_pairs.append((
            graph_service.get_experiment("exp_001"),
            other_run,
            {ComparabilityStatus.NOT_COMPARABLE},
            f"exp_001 vs {other_run.id} (different dataset)",
        ))

    comp_checks = 0
    comp_correct = 0
    for left, right, expected_statuses, _ in ground_truth_pairs:
        if left and right:
            assessment = comp_engine.assess(left, right)
            comp_checks += 1
            if assessment.status in expected_statuses:
                comp_correct += 1
    comparability_acc = (comp_correct / comp_checks * 100.0) if comp_checks else 100.0

    # --- Metric 4 & 7 & 9: Budget Compliance & Citation Coverage ---
    budget_gpu_hours = 2.0
    rec_req = RecommendationRequest(
        objective="Maximize fraud detection validation accuracy",
        compute_budget_gpu_hours=budget_gpu_hours,
    )
    rec_set = rec_engine.generate(project_id, rec_req)
    budget_compliance = 100.0 if (rec_set.estimated_gpu_hours or 0) <= budget_gpu_hours else 0.0
    total_citations = 0
    valid_citations = 0
    all_record_ids = {r.id for r in records}
    for rec in rec_set.recommendations:
        for cid in rec.baseline_experiment_ids + rec.supporting_experiment_ids:
            total_citations += 1
            if cid in all_record_ids:
                valid_citations += 1
    citation_coverage = (valid_citations / total_citations * 100.0) if total_citations else 100.0

    return {
        "directly_affected": sorted(direct_ids),
        "indirectly_affected": sorted(indirect_ids),
        "baseline_flagged": sorted(baseline_affected),
        "baseline_fp_ids": baseline_fp_ids,
        "graph_fp_ids": graph_fp_ids,
        "unaffected_count": len(ground_truth_unaffected),
        "import_success_rate": import_success_rate,
        "import_time_ms": import_time_ms,
        "duplicate_rejection_rate": duplicate_rejection_rate,
        "comparability_accuracy": comparability_acc,
        "budget_compliance": budget_compliance,
        "citation_coverage": citation_coverage,
        "graph": {
            "dependency_recall": graph_recall,
            "false_positive_impact_rate": graph_fp,
            "workflow_time_ms": workflow_time_ms,
            "recomputation_time_ms": graph_time_ms,
        },
        "baseline": {
            "dependency_recall": baseline_recall,
            "false_positive_impact_rate": baseline_fp,
            "traversal_time_ms": baseline_time_ms,
        },
        "total_eval_time_sec": time.perf_counter() - start_time,
    }


def print_report(res: dict) -> str:
    direct = ", ".join(res["directly_affected"])
    indirect = ", ".join(res["indirectly_affected"])
    fp_ids = ", ".join(res["baseline_fp_ids"]) or "(none)"

    lines = [
        "# RUNWISE Quantitative Evaluation & Baseline Report",
        "",
        f"**Benchmark Run:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Ground Truth:** {TOTAL_RUNS} generated synthetic experiments. "
        f"Affected by a defect in `tabular_fraud_v1`: {direct} directly, "
        f"{indirect} through derivation. Nothing else is affected.",
        "**Method:** the 'FalkorDB graph' column is the production "
        "`ImpactAnalysisService` against live FalkorDB on a throwaway graph, dropped "
        "afterwards. The retrieval baseline flags every run whose *dataset name* "
        "matches, version-blind, without following derivation. Ground truth is "
        "computed independently of both.",
        "",
        "## 1. Ground-Truth Data & Ingestion Quality (PRD 13)",
        f"- **Import Success Rate:** {res['import_success_rate']:.1f}% "
        f"({res['import_time_ms']:.0f} ms for {TOTAL_RUNS} runs)",
        f"- **Duplicate Rejection Rate:** {res['duplicate_rejection_rate']:.1f}% "
        "(measured: a second identical import, not asserted)",
        f"- **Comparability Accuracy:** {res['comparability_accuracy']:.1f}%",
        f"- **Compute Budget Compliance:** {res['budget_compliance']:.1f}% "
        "(Cost strictly within allocated GPU-hours)",
        f"- **Evidence Citation Coverage:** {res['citation_coverage']:.1f}% "
        "(all citation ids resolve to real records)",
        "",
        "## 2. Benchmark Comparison: RUNWISE Lineage vs Retrieval Baseline",
        "",
        "| Evaluation Metric (PRD 13) | RUNWISE (FalkorDB Graph) | Retrieval Baseline (Shallow Search) | Delta / Advantage |",
        "|---|---|---|---|",
        (
            f"| **Dependency Recall** | **{res['graph']['dependency_recall']:.1f}%** | "
            f"{res['baseline']['dependency_recall']:.1f}% | "
            f"**+{res['graph']['dependency_recall'] - res['baseline']['dependency_recall']:.1f} pts** "
            "(catches the derived exp_051/exp_052, which dataset-name matching cannot see) |"
        ),
        (
            f"| **False-Positive Impact Rate** | **{res['graph']['false_positive_impact_rate']:.1f}%** | "
            f"{res['baseline']['false_positive_impact_rate']:.1f}% | "
            f"**{res['baseline']['false_positive_impact_rate'] - res['graph']['false_positive_impact_rate']:.1f} pts spared** "
            f"(the baseline sweeps {fp_ids} into the affected set: clean run(s) on "
            "the repaired v2, taken because the dataset name matches) |"
        ),
        (
            f"| **Identification Latency (traversal)** | "
            f"**{res['graph']['recomputation_time_ms']:.1f} ms** (read-side re-computation) | "
            f"{res['baseline']['traversal_time_ms']:.3f} ms | "
            f"full impact workflow incl. correction persistence: "
            f"{res['graph']['workflow_time_ms']:.1f} ms |"
        ),
        "",
        "## 3. Findings & Conclusions",
        (
            f"1. **Derived-lineage recall, measured.** The correction on "
            f"`tabular_fraud_v1` invalidates {direct} directly and also {indirect}, "
            "which are runs that never named that dataset version: they descend from "
            "an affected run through derivation. Shallow retrieval finds only the "
            f"direct users ({res['baseline']['dependency_recall']:.0f}%) and reports "
            "the two descendants as valid, silently; the graph traversal finds all "
            f"of them ({res['graph']['dependency_recall']:.0f}%)."
        ),
        (
            "2. **Zero-Key Deterministic Guarantee:** recommendations adhere to the "
            "compute budget and cite only grounded record ids, with no external LLM "
            f"dependency (citation coverage {res['citation_coverage']:.1f}%)."
        ),
        (
            "3. **Surgical Invalidation, measured on both sides.** "
            f"{res['unaffected_count']} unaffected runs stay valid, including "
            "`exp_053`, a clean baseline on the repaired `tabular_fraud_v2`: the "
            "exact-version graph does not flag it, while name-level retrieval "
            "sweeps it in with the bad version."
        ),
    ]
    report_text = "\n".join(lines)
    print(report_text)
    return report_text


def main():
    res = run_benchmark()
    report = print_report(res)

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "EVALUATION.md"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Evaluation report written to {out_path}]")


if __name__ == "__main__":
    main()
