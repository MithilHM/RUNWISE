# Handoff: recommendation and comparability engine

Status: complete and tested. 126 tests, no FalkorDB required.

This handoff covers one work area: the recommendation and comparability engine. It
lists what was built, what was deliberately left alone, and what the other owners need
in order to wire it in.

---

## What this area delivers

| Path | What it does |
|---|---|
| `app/models/contracts.py` | Shared Pydantic request/response models (PRD 20 shapes) |
| `app/ports/graph.py` | `GraphService` Protocol, the interface consumed from the graph layer |
| `app/services/comparability.py` | FR-4 comparability engine, reason codes, model family profiles |
| `app/services/recommendations.py` | FR-5 gap detection, candidate generation, ranking, budget, FR-9 decisions |
| `app/services/evidence.py` | Citation ledger, epistemic status, correction-aware partitioning |
| `app/services/llm_adapter.py` | Pluggable LLM client with deterministic fallback |
| `app/services/ranking_rules.md` | Documented ranking rules (deliverable 5) |
| `app/api/recommendations.py` | 4 endpoints |
| `app/api/copilot.py` | Query endpoint + `ResearchQueryService` |
| `tests/test_comparability.py` | 34 tests |
| `tests/test_recommendations.py` | 42 tests |
| `tests/test_evidence.py` | 50 tests |
| `tests/fakes/fake_graph.py` | In-memory `GraphService` used by the test suite |
| `tests/conftest.py` | Shared scenario fixtures |

Four further small files were added because the suite cannot run without them, none
of them inside another owner's directory:
`conftest.py` (sys.path bootstrap), `app/api/deps.py` (dependency providers),
`app/ports/__init__.py`, `app/api/__init__.py`.

---

## Three things needed to wire this in

### 1. Register the routers in `app/main.py`

```python
from app.api import copilot, recommendations

app.include_router(recommendations.router)
app.include_router(copilot.router)
```

### 2. Wire the graph service

```python
app.state.graph_service = FalkorDBGraphService(...)   # the graph layer's implementation
```

Until that is set, these endpoints return a structured `503`
(`graph_service_unavailable`) rather than failing obscurely.

### 3. Dependencies in `backend/pyproject.toml`

`pyproject.toml` is owned by the integration owner (PRD 19), so it was not created
here. This area needs only
what the stack already implies: `fastapi`, `pydantic>=2`, and `pytest` for tests.
The LLM adapter deliberately uses `urllib` from the standard library, so no new
runtime dependency is introduced.

`httpx` is required for `fastapi.testclient`, which the router tests use.

Environment variables used (consolidate into `.env.example`):

| Variable | Required | Default |
|---|---|---|
| `RUNWISE_LLM_API_KEY` | no | unset -> deterministic mode |
| `RUNWISE_LLM_MODEL` | no | `gpt-4o-mini` |
| `RUNWISE_LLM_BASE_URL` | no | `https://api.openai.com/v1` |

No key means deterministic demo mode, and the response says so
(`mode="deterministic"`, `llm_used=false`).

---

## Handoff contract for the graph layer: implementing the graph port

`app/ports/graph.py` is a `Protocol` with no FalkorDB import and no Cypher. You
implement it; this area consumes it and never imports the graph layer's internals.

Required methods: `list_experiments`, `get_experiment`,
`find_comparable_candidates`, `get_experiment_lineage`, `list_datasets`,
`list_dataset_versions`, `list_corrections`, `get_correction_impact`,
`list_review_items`, `save_recommendation`, `get_recommendation`,
`list_recommendations`, `record_recommendation_decision`.

`tests/fakes/fake_graph.py` is a working reference implementation with the exact
signatures.

Three contract requirements:

1. **Project scope belongs in the query.** Do not fetch everything and filter in
   Python. `tests/conftest.py` includes an `exp_other_project` record specifically
   to catch this.
2. **`find_comparable_candidates` is a preselection, not a verdict.** Filter on
   indexed identity fields only. Comparability is decided by the rule engine.
3. **No generic `run_cypher` escape hatch.** PRD 9 prohibits unrestricted
   frontend-supplied Cypher; the same reasoning applies in-process.

### Fields the graph layer must populate

`ExperimentRecord` carries dataset id/name/version, dataset quality status,
`configuration`, `hyperparameters`, `preprocessing`, `random_seed`,
`training_steps`, `compute_cost_gpu_hours`, `source_reference`, and per-metric
`evaluation_split` and `evaluation_protocol`.

Anything you cannot populate stays `None`, and the engine degrades to
`INSUFFICIENT_INFORMATION` rather than guessing. That is the correct outcome — do
not backfill a default to make the tests pass.

### Two things to be careful about

- `dataset_id` / `dataset_version_id` are **node ids**;
  `dataset_name` / `dataset_version` are **human labels**. The engine tags which namespace a
  key came from and never compare across them. `"v1"` must not be assumed equal to
  `"dsv_churn_v1"`; the engine reports that pair as unconfirmable instead.
- `model_config` is named `configuration` in the contract because pydantic
  reserves `model_config`. The `Configuration` node still holds
  `canonical_json` / `config_hash` per PRD 8.

### Not specified here

`get_project_graph(project_id)` is listed in PRD 21 as an example operation, but its
shape is deliberately left open here: it serves the lineage screen, so the graph layer
and the frontend owner should agree on it. It will be consumed once added to the port.

---

## Handoff contract for the frontend: what the endpoints return

Contract models are in `app/models/contracts.py`, which is the file to translate
into TypeScript types. Every response field added after the PRD 20 examples is
optional or has a default, so the frontend can tolerate their absence.

The three shapes you will render most:

**Recommendation** (`Recommendation`) — `configuration`,
`independent_variables`, `controlled_variables`, `metrics`,
`estimated_gpu_hours`, `estimate_source`, `score`, `score_components[]`
(each with `weight`, `value`, `rationale`), `validation_criteria[]` (each with
`is_suggested_default`), `uncertainties[]`, `evidence_status`.

**ResearchAnswer** — `answer`, `mode` (`deterministic` | `llm`), `llm_used`,
`claims[]`, `comparisons[]` (every non-`COMPARABLE` verdict, including runs
outside the cohort, each with `reasons[]`), `recommendations[]`,
`excluded_evidence[]`, `comparable_experiment_ids[]`, `missing_information[]`,
`insufficient_evidence`, `grounding`.

**ComparabilityAssessment** — `status`, `reasons[]` (each with `code`, `dimension`,
`severity`, `message`, `left_value`, `right_value`), `matched_dimensions`,
`differing_dimensions`, `missing_dimensions`, `score`, `score_kind`, `confidence`,
`explanation`.

### Four UI rules the backend depends on

1. **Render `evidence_status`.** `observed`, `user_asserted`, `inferred`,
   `disputed`, `superseded`, `unresolved` are different epistemic grades. A
   recommendation is always `inferred`. If the UI merges them, the product makes a
   trust claim it cannot support.
2. **Never label `score` a probability.** It is a weighted heuristic. Show the
   components.
3. **Show `uncertainties[]` and `missing_information[]`.** An answer with no
   caveats should be rare; if it looks too clean, something is being hidden.
4. **Render `mode` and `llm_used`.** The demo must not imply a model was consulted
   when it was not (PRD 15).

`recommendations: []` is a valid response with the reason in `uncertainties`, not an
error state. Do not render it as a failure.

---

## Endpoints

```
POST /api/v1/projects/{project_id}/recommendations
GET  /api/v1/recommendations/{recommendation_id}
GET  /api/v1/projects/{project_id}/recommendations
POST /api/v1/recommendations/{recommendation_id}/decision
POST /api/v1/projects/{project_id}/query
```

No endpoint accepts Cypher or a free-form graph expression.

---

## Handoff contract for integration: what the tests prove

`cd backend && python -m pytest tests -q` — 126 tests, ~1s, no database, no
network, no API key.

They are not smoke tests. The ones worth knowing about:

- **Budget is never exceeded**, and a candidate with an unknown cost is not
  funded under a budget.
- **Budget attribution is honest.** `budget_exhausted` is true only when the
  budget blocked something; drops caused by an unknown cost or by
  `max_recommendations` are named separately.
- **An empty result carries its reason** — budget too small, no gap, or no
  compute estimate — rather than looking like a broken feature.
- **Recommendations cite only stored experiments**, and a run from another
  project is asserted absent from `considered_experiment_ids` and from every
  citation.
- **The anchor is the best completed, measured run in the comparable cohort.**
  `exp_on_v2` scores highest overall and is asserted *absent* from every proposal,
  because it is a different dataset version. This is the product's core
  behaviour, not a detail. A `FAILED` run with a better number is also asserted
  absent.
- **A correction changes what is trusted.** With the v1 defect reported, v1 runs
  are withheld, none of them support a proposal, and `exp_on_v2` stays usable.
- **Epistemic grades are never promoted.** `superseded` and `unresolved` records
  are withheld rather than counted as observations, and a best-run claim carries
  the record's real grade.
- **Withheld is not the same as absent.** When every record is withheld, the
  answer says so instead of claiming no run records the metric.
- **Hallucinated identifiers and figures are rejected** — by the citation ledger,
  and by the LLM prose filter, which derives id shapes from the graph's own
  convention rather than assuming `exp_` (UUIDs are matched outright).
- **No LLM key still works**, and `llm_used` is `false`. A client that reports no
  completed call is treated as unavailable.
- **Non-comparable runs appear with their reason**, so the UI can explain an
  exclusion instead of showing a silent gap.

`tests/conftest.py` holds the shared scenario; extend it rather than inventing a second
one. `pytest` needs no configuration: `conftest.py` bootstraps `sys.path`.

---

## Known limits

Stated plainly so nobody is surprised:

1. **Cypher is unvalidated against a live FalkorDB.** No queries were written and
  the graph layer does not exist yet, so the port is verified against the fake
  only. Integration risk lands on Milestone C.
2. **`get_project_graph` is unspecified** (see above).
3. **The ranking weights are unvalidated against real data.** They are documented
  and sweepable; PRD 13 asks for an evaluation against a retrieval-only baseline
  before any gain is claimed. None is claimed.
4. **The LLM prose check is lexical, not semantic.** It rejects unknown identifiers
  and figures absent from the evidence. A qualitative claim built from supplied
  words could still pass, and an id with no separator (`abc123`) cannot be
  pattern-matched. See `ranking_rules.md` section 5.
5. **Authentication and project access control are not implemented.** Endpoints
  trust the project id and rely on the graph service scoping it. PRD 9 and PRD 12
  require enforcement; that needs the integration owner's auth layer.
6. **`POST /query` persists its proposals.** Each copilot answer stores its
  recommendations so the approve/reject actions can address a record. Pass
  `persist=False` to `RecommendationEngine.generate` for a read-only caller that
  never approves, to avoid unbounded growth.
7. **No HTTP surface for FR-4.** Comparability is reachable as a library and
  through `ResearchAnswer.comparisons`, but there is no "assess these two
  experiments" endpoint. None was assigned to this area; adding one needs a
  router from the integration owner.
8. **No persistence for claims.** `KnowledgeFact` is defined for FR-7 but no
   storage is wired. It needs a graph-backed store from the graph layer, or the
   integration owner's choice
   of home for it.