# RUNWISE — AI Research Experiment Copilot

RUNWISE turns scattered ML experiment histories into a graph-backed research
intelligence platform on **FalkorDB**, with a **FastAPI** backend and a
**React + TypeScript** frontend.

Its defining behaviour: when evidence changes what a researcher should believe about a
previous experiment, RUNWISE finds the dependent conclusions and changes what it
recommends next — while showing the graph paths behind that decision.

---

## 1. Quickstart

### Prerequisites

- Python 3.10+
- Docker (for FalkorDB)
- Node 20+ (for the frontend)

### Start FalkorDB

```bash
docker compose up -d
```

### Install backend dependencies

```bash
pip install -r requirements.txt
```

### Configure the environment

```bash
cp .env.example .env
```

Defaults connect to `127.0.0.1:6379`, graph `runwise`. The app runs with no LLM key
at all: answers are generated deterministically and report `llm_used=false`.

### Seed demo data

```bash
python scripts/seed_demo.py           # idempotent: safe to re-run
python scripts/seed_demo.py --reset   # drop the graph first, for a clean demo
python scripts/reset_graph.py --seed  # same reset, via the reset script
```

Seeding creates 53 synthetic experiments across several model families and dataset
versions, then reports a real correction: which runs touched the defective dataset
version, which are downstream, and which are unaffected.

### Run the tests

```bash
# backend: 162 tests. The graph tests need FalkorDB running; the comparability,
# recommendation and evidence tests do not.
cd backend && python -m pytest tests -v

# frontend
cd frontend && npm install && npm test
```

### Verify the whole stack

With the backend running, these check the things unit tests cannot:

```bash
python scripts/smoke_test.py            # 75 checks over every endpoint, against live data
python scripts/verify_api_contract.py   # frontend types vs the live OpenAPI schema
python scripts/probe_cypher_support.py  # which Cypher this FalkorDB build accepts
```

`smoke_test.py` creates a throwaway project of its own, so it never consumes the demo
data. Each run leaves one small project behind; `python scripts/reset_graph.py` clears
them.

### Start the backend

Run from the `backend` directory, so `app` is importable:

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

Interactive API docs: `http://localhost:8000/docs`

### Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Vite serves on `http://localhost:5173` and proxies `/api` to port 8000, so the
browser stays on one origin. The backend allows the Vite dev origin through
`CORS_ORIGINS` in `.env`.

---

## 2. What the screens do

| Route | Screen | PRD |
|---|---|---|
| `/` | Experiment workspace: import, filter, chart, inspect a run | Screen A |
| `/copilot` | Research copilot: ask a question, get grounded evidence and proposals | Screen B |
| `/lineage` | Lineage and impact: graph, correction, impact report, review queue | Screen C |

Four display rules the product does not compromise on, because the backend cannot
support the alternative:

1. **Epistemic grades are rendered, not collapsed.** `observed`, `user_asserted`,
   `inferred`, `disputed`, `superseded` and `unresolved` each look different.
2. **A score is never a probability.** Recommendation scores are documented weighted
   heuristics; the components and their weights are always shown.
3. **Uncertainties and missing information are always visible.** A clean-looking
   answer with no caveats is a warning sign, not a success.
4. **`mode` and `llm_used` are displayed.** The demo never implies a model was
   consulted when it was not.

An empty `recommendations: []` is a valid answer whose reason is in `uncertainties`,
not an error state, and the UI renders it that way.

---

## 3. Shared integration interfaces

### For the graph layer

`backend/app/ports/graph.py` defines `GraphService`, a `Protocol` with no FalkorDB
import and no Cypher. `GraphRepository` (`backend/app/graph/repository.py`) implements
it. The engines consume the port and never import graph internals.

```python
from app.graph.repository import GraphRepository

repo = GraphRepository()

# Analysis records: every unrecorded field stays None, never a default.
records = repo.list_experiments("project_demo", model_family="gradient_boosting")
detail = repo.get_experiment("exp_001", project_id="project_demo")
comparables = repo.find_comparable_candidates(project_id="project_demo", metric="validation_accuracy")
versions = repo.list_dataset_versions("project_demo")
impact = repo.get_correction_impact("corr_...")
```

Three contract rules, enforced by the implementation and by tests:

1. **Project scope belongs in the query.** Every read filters on the project inside
   Cypher, so a cross-project read fails rather than filtering silently afterwards.
2. **`find_comparable_candidates` is a preselection, not a verdict.** It narrows on
   indexed identity fields; the rule engine decides comparability.
3. **No `run_cypher` escape hatch.** PRD 9 prohibits unrestricted frontend-supplied
   Cypher, and the same rule applies in-process.

`tests/fakes/fake_graph.py` implements the same port in memory and is what the
analysis-engine tests run against, so those tests need neither a database nor a
network.

### Endpoints

```
POST   /api/v1/projects
GET    /api/v1/projects
GET    /api/v1/projects/{project_id}
POST   /api/v1/projects/{project_id}/experiments/import      # JSON body, CSV/JSON upload
GET    /api/v1/projects/{project_id}/experiments            # FR-2 filters
GET    /api/v1/experiments/{experiment_id}
GET    /api/v1/experiments/{experiment_id}/comparability    # FR-4
GET    /api/v1/experiments/{experiment_id}/lineage
GET    /api/v1/experiments/{experiment_id}/related
GET    /api/v1/projects/{project_id}/graph
GET    /api/v1/projects/{project_id}/graph/summary
POST   /api/v1/projects/{project_id}/query                  # FR-3 copilot
POST   /api/v1/projects/{project_id}/recommendations        # FR-5
GET    /api/v1/projects/{project_id}/recommendations
GET    /api/v1/recommendations/{recommendation_id}
POST   /api/v1/recommendations/{recommendation_id}/decision  # FR-9
POST   /api/v1/projects/{project_id}/corrections            # FR-6
GET    /api/v1/corrections/{correction_id}/impact
GET    /api/v1/projects/{project_id}/review-queue
POST   /api/v1/review-items/{review_item_id}/resolve
GET    /health
```

No endpoint accepts Cypher or a free-form graph expression.

Every failure returns the same envelope, including request validation and an
unreachable database:

```json
{
  "detail": [
    { "code": "experiment_not_found", "message": "No experiment with id 'exp_x' exists.", "field": null }
  ]
}
```

---

## 4. Environment variables

| Variable | Required | Default |
|---|---|---|
| `FALKORDB_HOST` | no | `127.0.0.1` |
| `FALKORDB_PORT` | no | `6379` |
| `FALKORDB_GRAPH` | no | `runwise` |
| `FALKORDB_PASSWORD` | no | unset |
| `CORS_ORIGINS` | no | Vite dev origins |
| `RUNWISE_LLM_API_KEY` | no | unset → deterministic mode |
| `RUNWISE_LLM_MODEL` | no | `gpt-4o-mini` |
| `RUNWISE_LLM_BASE_URL` | no | `https://api.openai.com/v1` |

Never commit a key.

---

## 5. Documentation

| File | Contents |
|---|---|
| `RUNWISE_PRD_Parallel_Plan.md` | Product requirements, flows, and the ownership plan |
| `docs/GRAPH_SCHEMA.md` | Node labels, relationships, key Cypher patterns, correction semantics, and the Cypher constructs this FalkorDB build supports |
| `docs/DEMO_SCRIPT.md` | Step-by-step 3-minute hackathon demo presentation guide (Screens A, B, and C) |
| `docs/EVALUATION.md` | Quantitative benchmark results comparing FalkorDB graph against retrieval baseline (PRD 13) |
| `docs/TROUBLESHOOTING.md` | Quick-reference fixes for Docker, port 6379, and CORS setup issues |
| `backend/HANDOFF.md` | The recommendation and comparability engine: contract, endpoints, known limits |
| `backend/app/services/ranking_rules.md` | The documented ranking and comparability rules the code implements |

If the code and `ranking_rules.md` disagree, the code is the bug.

---

## 6. Scripts

| Script | Purpose |
|---|---|
| `scripts/prepare_demo.py` | 1-click demo prep: checks FalkorDB, resets graph, seeds 53 runs, and executes smoke test. |
| `scripts/run_all_tests.py` | One-command test runner across unit tests, API tests, and E2E integration tests. |
| `scripts/evaluate.py` | Automated benchmark comparing FalkorDB graph traversal vs naive retrieval baseline across 9 PRD metrics. |
| `scripts/seed_demo.py` | Generate and ingest 53 synthetic experiments, then run a real correction. Idempotent. `--reset` drops the graph first. |
| `scripts/reset_graph.py` | Drop the graph and optionally re-seed. The demo reset procedure. |
| `scripts/smoke_test.py` | 75 end-to-end checks against a running backend and live FalkorDB. |
| `scripts/verify_api_contract.py` | Fails when the frontend types and the backend's OpenAPI schema disagree. |
| `scripts/probe_cypher_support.py` | Reports which Cypher constructs the running FalkorDB accepts. |
| `scripts/build_sample_csv.py` | Regenerates both sample files from the seed generator itself, with real JSON in the nested cells. |

---

## 7. Known gaps

Stated plainly so nothing is a surprise in a demo:

- **No authentication or project access control.** Endpoints trust the project id
  and rely on the graph query to scope it. PRD 9 and 12 require enforcement; that
  needs an auth layer, not a query change.
- **The configuration filter is applied after the query.** Configuration is stored as
  a canonical JSON document, which this FalkorDB build cannot probe by a dynamic key.
  Every other filter, including the project boundary, is enforced by the graph.
- **A run's `model_family`, `status` and metric `direction` may be unrecorded.**
  Nothing is invented to fill them, so such a record reads as unrecorded everywhere:
  in the UI, in comparability (`INSUFFICIENT_INFORMATION`), and in the evidence
  ledger.
- **The ranking weights are a product decision, not a fitted model.** They are
  documented and sweepable. Baseline comparison is measured and documented in
  `docs/EVALUATION.md` (100% dependency recall vs 25% retrieval-only baseline).
- **The LLM prose filter is lexical.** It rejects unknown identifiers and figures,
  but a fabricated qualitative claim phrased in supplied words would pass.
- **The project graph view can be partial.** `GET /projects/{id}/graph` returns a
  `truncated` flag when the node budget cut the traversal, and the UI says so.
