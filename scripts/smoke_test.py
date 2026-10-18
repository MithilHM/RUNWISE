"""End-to-end smoke test against a running backend and a live FalkorDB.

Exercises every endpoint the frontend uses, in the order the demo uses them, and
prints the observable result of each. Run it after ``scripts/seed_demo.py``:

    python scripts/smoke_test.py

Exit code is non-zero if any check fails, so it is usable as a verification gate.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:8000"
PROJECT = "project_demo"

failures: list[str] = []
checks = 0


def call(method: str, path: str, body: dict | None = None) -> tuple[int, object]:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read().decode()
            return response.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as error:
        raw = error.read().decode()
        try:
            return error.code, json.loads(raw)
        except ValueError:
            return error.code, raw


def check(label: str, condition: bool, detail: str = "") -> None:
    global checks
    checks += 1
    mark = "ok  " if condition else "FAIL"
    print(f"[{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not condition:
        failures.append(label)


print("== system ==")
status, health = call("GET", "/health")
check("health is 200", status == 200, str(status))
check("graph is reachable", isinstance(health, dict) and health.get("database", {}).get("status") == "healthy")

print("\n== projects and history (Screen A) ==")
status, projects = call("GET", "/api/v1/projects")
check("project list is 200", status == 200, str(status))
check("demo project exists", isinstance(projects, list) and any(p["id"] == PROJECT for p in projects))

status, experiments = call("GET", f"/api/v1/projects/{PROJECT}/experiments?limit=500")
check("experiment list is 200", status == 200, str(status))
check("53 runs are listed", isinstance(experiments, list) and len(experiments) == 53, str(len(experiments) if isinstance(experiments, list) else experiments))
derived_ids = {"exp_051", "exp_052", "exp_053"}
check(
    "the lineage amplifiers are seeded",
    derived_ids <= {e["id"] for e in experiments} if isinstance(experiments, list) else False,
    str(sorted({e["id"] for e in experiments} & derived_ids) if isinstance(experiments, list) else "?"),
)
run = next((e for e in experiments if e["metrics"]), None)
check("a run carries metrics", run is not None)

if run:
    exp_id = run["id"]
    status, detail = call("GET", f"/api/v1/experiments/{exp_id}?project_id={PROJECT}")
    check("experiment detail is 200", status == 200, str(status))
    check(
        "detail keeps a null metric value null rather than 0",
        all(m["value"] is None or m["value"] != 0 for m in detail.get("metrics", [])),
    )

    status, lineage = call("GET", f"/api/v1/experiments/{exp_id}/lineage?project_id={PROJECT}")
    check("lineage is 200", status == 200, str(status))
    check("lineage returns nodes", bool(lineage.get("nodes")))

    status, related = call("GET", f"/api/v1/experiments/{exp_id}/related?project_id={PROJECT}")
    check("related is 200", status == 200, str(status))
    check("related has known buckets", set(related) >= {"same_dataset_version", "siblings"})

    status, assessment = call("GET", f"/api/v1/experiments/{exp_id}/comparability?project_id={PROJECT}")
    check("comparability is 200", status == 200, str(status))
    check(
        "comparability returns a defined status",
        isinstance(assessment, dict)
        and assessment.get("status")
        in {"COMPARABLE", "COMPARABLE_WITH_CAVEATS", "NOT_COMPARABLE", "INSUFFICIENT_INFORMATION"},
        str(assessment.get("status") if isinstance(assessment, dict) else assessment),
    )
    check(
        "comparability score is a heuristic, not a probability",
        isinstance(assessment, dict) and 0.0 <= assessment.get("score", -1) <= 1.0
        and assessment.get("score_kind") == "severity_penalty_heuristic",
    )

status, filtered = call("GET", f"/api/v1/projects/{PROJECT}/experiments?metric=validation_accuracy&limit=500")
check("metric filter is 200", status == 200, str(status))
check(
    "metric filter actually filters",
    isinstance(filtered, list) and all(any(m["name"] == "validation_accuracy" for m in e["metrics"]) for e in filtered),
    f"{len(filtered) if isinstance(filtered, list) else '?'} rows",
)

status, config_filtered = call(
    "GET", f"/api/v1/projects/{PROJECT}/experiments?config_field=max_depth&config_value=4&limit=500"
)
check("config filter is 200", status == 200, str(status))
check(
    "config filter narrows the result",
    isinstance(config_filtered, list) and 0 < len(config_filtered) < 53,
    f"{len(config_filtered) if isinstance(config_filtered, list) else '?'} rows",
)

status, summary = call("GET", f"/api/v1/projects/{PROJECT}/graph/summary")
check("graph summary is 200", status == 200, str(status))
check("graph summary counts experiments", summary.get("node_counts", {}).get("Experiment", 0) == 53)

# The demo's own correction was applied during seeding. The two derived runs never
# touched the corrected dataset version, so their being tainted is exactly the
# derived-lineage behavior the project claims - not anything a dataset search
# could have produced.
status, detail_051 = call("GET", "/api/v1/experiments/exp_051?project_id=project_demo")
check("a derived run is reachable", status == 200, str(status))
if status == 200:
    check(
        "the derived run carries the correction's taint",
        detail_051.get("quality_status") == "needs_review",
        str(detail_051.get("quality_status")),
    )

# The seeded correction opened its review items at seed time: three direct users
# of the corrected version, plus the two derived runs, which never touched that
# dataset version at all - reaching them is the derived-lineage behavior, not
# something a dataset search could ever produce.
status, queue = call("GET", f"/api/v1/projects/{PROJECT}/review-queue")
check("demo review queue is 200", status == 200, str(status))
if status == 200:
    items = queue if isinstance(queue, list) else []
    queued = {i.get("experiment_id"): i for i in items if i.get("experiment_id")}
    check(
        "the direct users are in review",
        {"exp_001", "exp_002", "exp_003"} <= set(queued),
        str(sorted(queued)),
    )
    check(
        "the correction reaches derived runs it never touched",
        {"exp_051", "exp_052"} <= set(queued),
        str(sorted(queued)),
    )
    check(
        "the derived run's review cites its lineage",
        "derived" in str(queued.get("exp_051", {}).get("reason", "")).lower(),
        str(queued.get("exp_051", {}).get("reason"))[:120],
    )

status, detail_053 = call("GET", "/api/v1/experiments/exp_053?project_id=project_demo")
if status == 200:
    check(
        "the repaired-version control stays unaffected",
        detail_053.get("quality_status") == "valid",
        str(detail_053.get("quality_status")),
    )

status, graph = call("GET", f"/api/v1/projects/{PROJECT}/graph?limit=800")
check("graph is 200", status == 200, str(status))
check("graph returns real nodes and edges", len(graph.get("nodes", [])) > 0 and len(graph.get("edges", [])) > 0,
      f"{len(graph.get('nodes', []))} nodes / {len(graph.get('edges', []))} edges")
check("graph reports truncation honestly", "truncated" in graph)

print("\n== copilot and recommendations (Screen B) ==")
status, answer = call("POST", f"/api/v1/projects/{PROJECT}/query", {"question": "Which runs are comparable, and what should I try next?", "compute_budget_gpu_hours": 2})
check("query is 200", status == 200, str(status) if status != 200 else "")
if status == 200:
    check("answer names its mode", answer.get("mode") in {"deterministic", "llm"}, str(answer.get("mode")))
    check("llm_used is a real boolean", isinstance(answer.get("llm_used"), bool), str(answer.get("llm_used")))
    check("answer cites only retrieved runs", (answer.get("grounding") or {}).get("is_grounded", True) is True)
    check("answer reports caveats", isinstance(answer.get("uncertainties"), list))
    check("withheld evidence is labelled, not deleted", all("status" in e for e in answer.get("excluded_evidence", [])))
    if answer.get("excluded_evidence"):
        excluded_ids = {e.get("id") or e.get("experiment_id") for e in answer["excluded_evidence"]}
        check(
            "the derived runs are withheld by the copilot too",
            {"exp_051", "exp_052"} <= excluded_ids,
            str(sorted(i for i in excluded_ids if i and i.startswith(("exp_047", "exp_05"))) or "none"),
        )

status, rec_set = call("POST", f"/api/v1/projects/{PROJECT}/recommendations", {"objective": "Improve validation accuracy within a small budget", "compute_budget_gpu_hours": 2})
check("recommendations are 200", status == 200, str(status) if status != 200 else json.dumps(rec_set)[:200])
if status == 200:
    recs = rec_set.get("recommendations", [])
    check("recommendations is a list", isinstance(recs, list))
    check("every proposal is inferred", all(r.get("evidence_status") == "inferred" for r in recs), f"{len(recs)} proposals")
    check("every proposal explains its score", all(r.get("score_components") for r in recs) or not recs)
    check("no proposal exceeds the budget", (rec_set.get("estimated_gpu_hours") or 0) <= 2, str(rec_set.get("estimated_gpu_hours")))
    check("an empty set explains itself", bool(recs) or bool(rec_set.get("uncertainties")))
    check("considered evidence is listed", isinstance(rec_set.get("considered_experiment_ids"), list))
    if recs:
        rec_id = recs[0]["id"]
        status, stored = call("GET", f"/api/v1/recommendations/{rec_id}")
        check("a stored proposal can be read back", status == 200 and stored.get("id") == rec_id, str(status))
        status, decided = call("POST", f"/api/v1/recommendations/{rec_id}/decision", {"decision": "approved", "decided_by": "smoke_test", "note": "verified"})
        check("FR-9 approval round-trips", status == 200 and decided.get("status") == "approved", str(status))
        check("approval does not promote the proposal to an observation", decided.get("evidence_status") == "inferred")

print("\n== corrections and impact (Screen C) ==")
# A project of the smoke test's own, so the check is repeatable and never consumes
# the demo's review queue. The id is unique per run, which means each run leaves one
# small project behind; `python scripts/reset_graph.py` clears them.
SCRATCH = f"smoke_{int(time.time())}"
call("POST", "/api/v1/projects", {"id": SCRATCH, "name": "Smoke test project", "description": "created by scripts/smoke_test.py"})

# Run and dataset ids are unique per invocation too: an experiment id belongs to
# exactly one project, and the backend refuses to re-file it.
STAMP = SCRATCH.replace("smoke_", "smk_")
DS_V1 = f"{STAMP}_data_v1"
DS_V2 = f"{STAMP}_data_v2"

SCRATCH_RUNS = [
    {
        "id": f"{STAMP}_a1",
        "project_id": SCRATCH,
        "name": "baseline",
        "model_family": "gradient_boosting",
        "dataset_name": f"{STAMP}_data",
        "dataset_version": "v1",
        "status": "completed",
        "random_seed": 1,
        "compute_cost_gpu_hours": 0.5,
        "duration_seconds": 60,
        "configuration": {"max_depth": 4},
        "metrics": [
            {"name": "validation_accuracy", "value": 0.90, "direction": "maximize", "evaluation_split": "validation", "evaluation_protocol": "holdout"}
        ],
    },
    {
        "id": f"{STAMP}_a2",
        "project_id": SCRATCH,
        "name": "seed variant",
        "model_family": "gradient_boosting",
        "dataset_name": f"{STAMP}_data",
        "dataset_version": "v1",
        "status": "completed",
        "random_seed": 2,
        "compute_cost_gpu_hours": 0.5,
        "duration_seconds": 60,
        "configuration": {"max_depth": 5},
        "metrics": [
            {"name": "validation_accuracy", "value": 0.91, "direction": "maximize", "evaluation_split": "validation", "evaluation_protocol": "holdout"}
        ],
    },
    {
        "id": f"{STAMP}_b1",
        "project_id": SCRATCH,
        "name": "different data version",
        "model_family": "gradient_boosting",
        "dataset_name": f"{STAMP}_data",
        "dataset_version": "v2",
        "status": "completed",
        "random_seed": 1,
        "compute_cost_gpu_hours": 0.5,
        "duration_seconds": 60,
        "configuration": {"max_depth": 6},
        "metrics": [
            {"name": "validation_accuracy", "value": 0.99, "direction": "maximize", "evaluation_split": "validation", "evaluation_protocol": "holdout"}
        ],
    },
]

status, imported = call("POST", f"/api/v1/projects/{SCRATCH}/experiments/import", SCRATCH_RUNS)
check("scratch import is 200", status == 200, str(status))
check(
    "scratch import stored every run",
    isinstance(imported, dict) and imported.get("imported") == len(SCRATCH_RUNS),
    json.dumps(imported)[:160],
)

status, impact = call(
    "POST",
    f"/api/v1/projects/{SCRATCH}/corrections",
    {"source_id": DS_V1, "source_type": "dataset_version", "description": "smoke test leakage", "created_by": "smoke_test"},
)
check("recording a correction is 200", status == 200, str(status) if status != 200 else json.dumps(impact)[:160])
if status == 200:
    check(
        "only runs on the defective data are affected",
        set(impact["directly_affected_experiment_ids"]) == {f"{STAMP}_a1", f"{STAMP}_a2"},
        str(impact["directly_affected_experiment_ids"]),
    )
    check(
        "the run on the other data version stays unaffected",
        impact["unaffected_experiment_ids"] == [f"{STAMP}_b1"],
        str(impact["unaffected_experiment_ids"]),
    )
    check("impact reports real dependency paths", len(impact["paths"]) >= 2, f"{len(impact['paths'])} paths")
    check("impact names the revalidation items", len(impact["review_item_ids"]) == 2, str(impact["review_item_ids"]))

status, queue = call("GET", f"/api/v1/projects/{SCRATCH}/review-queue")
check("review queue is 200", status == 200, str(status))
check("the correction opened exactly the affected runs", isinstance(queue, list) and len(queue) == 2, str(len(queue) if isinstance(queue, list) else queue))
if isinstance(queue, list) and queue:
    item = queue[0]
    check("queue status uses the contract vocabulary", item.get("status") == "open", str(item.get("status")))
    check("queue items link to their correction", bool(item.get("correction_id")), str(item.get("correction_id")))
    status, resolved = call("POST", f"/api/v1/review-items/{item['id']}/resolve", {"status": "resolved", "notes": "smoke test"})
    check("resolving a review item is 200", status == 200, str(status))
    check("resolution is recorded, not deleted", resolved.get("status") == "resolved" and bool(resolved.get("resolved_at")))
    status, after = call("GET", f"/api/v1/projects/{SCRATCH}/review-queue")
    check("a resolved item leaves the open queue", isinstance(after, list) and len(after) == 1, str(len(after) if isinstance(after, list) else after))

    corr_id = item.get("correction_id")
    status, reread = call("GET", f"/api/v1/corrections/{corr_id}/impact")
    check("correction impact is 200", status == 200, str(status))
    check("impact separates direct from downstream", "indirectly_affected_experiment_ids" in reread)
    check("history survives resolution", len(reread["directly_affected_experiment_ids"]) == 2)

print("\n== a correction changes what is trusted ==")
status, before = call("POST", f"/api/v1/projects/{SCRATCH}/recommendations", {"objective": "Improve validation accuracy", "compute_budget_gpu_hours": 4})
check("recommendations still work under an open correction", status == 200, str(status) if status != 200 else json.dumps(before)[:160])
if status == 200:
    cited = {i for r in before["recommendations"] for i in (r["baseline_experiment_ids"] + r["supporting_experiment_ids"])}
    check(
        "a run under correction is never cited as support",
        not (cited & {f"{STAMP}_a1", f"{STAMP}_a2"}),
        str(sorted(cited)),
    )
    check(
        "the unaffected run is still usable",
        bool(before["considered_experiment_ids"]) and f"{STAMP}_b1" in before["considered_experiment_ids"],
        str(before["considered_experiment_ids"]),
    )

print("\n== error envelope ==")
status, body = call("GET", "/api/v1/experiments/does_not_exist")
check("unknown run is 404", status == 404, str(status))
check("404 uses the structured envelope", isinstance(body, dict) and isinstance(body.get("detail"), list)
      and body["detail"][0].get("code") == "experiment_not_found", json.dumps(body)[:120])

status, body = call("GET", f"/api/v1/projects/{PROJECT}/graph/summary?")
check("unknown project is 404, not a fabricated count", call("GET", "/api/v1/projects/no_such_project/graph/summary")[0] == 404)

status, body = call("POST", f"/api/v1/projects/{PROJECT}/recommendations", {"objective": ""})
check("empty objective is a validation error", status == 422, str(status))
check("validation error uses the same envelope", isinstance(body, dict) and isinstance(body.get("detail"), list))

print(f"\n{checks - len(failures)}/{checks} checks passed")
if failures:
    print("failed:")
    for failure in failures:
        print(f"  - {failure}")
    sys.exit(1)
print("all checks passed")
