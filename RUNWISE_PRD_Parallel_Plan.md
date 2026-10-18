# RUNWISE — Implementation PRD and Four-Person Parallel Work Plan

**Product:** RUNWISE — AI Research Experiment Copilot  
**Core technology:** FalkorDB, Python, FastAPI, React, TypeScript  
**Delivery target:** Functional, reproducible hackathon MVP  
**Collaboration model:** Four contributors working in the same Git branch, in any order

---

## 1. Product Overview

RUNWISE is an AI-powered research copilot for machine learning researchers and engineers who conduct repeated experiments across models, datasets, preprocessing pipelines, and training configurations.

ML experimentation produces disconnected information: model configurations, experiment metrics, datasets, hypotheses, failed attempts, evaluation results, and researcher notes. Researchers need to determine which experiments are comparable, understand why a result occurred, reproduce successful approaches, and identify conclusions that become unreliable when underlying data or assumptions change.

RUNWISE converts experiment history into a persistent, graph-based research intelligence system. Users can import experiment results, investigate prior runs, ask research questions, receive evidence-backed recommendations for the next experiment, and trace downstream effects when datasets or results become unreliable.

**Core value proposition:** Help ML researchers make better experimental decisions, avoid repeating mistakes, and use limited compute resources more effectively.

RUNWISE is general-purpose across ML model families. It must not be hardcoded for transformers or attention-head experiments.

## 2. Problem Statement

Researchers face five recurring problems:

1. Experiment history is scattered across CSV files, JSON logs, notebooks, configuration files, and informal notes.
2. Experiments may look comparable while differing in datasets, evaluation protocols, random seeds, or training budgets.
3. Failed experiments and useful lessons are easily forgotten or repeated.
4. When a dataset, metric, or assumption is found to be unreliable, researchers cannot easily trace every affected conclusion.
5. Choosing the next experiment requires balancing previous evidence, uncertainty, expected usefulness, and compute cost.

RUNWISE adds a decision-support layer that connects experiment records and uses their relationships to influence future recommendations.

## 3. Goals and Non-Goals

### Goals

- Import and normalize ML experiment histories.
- Support diverse model families and experiment types.
- Build an explicit experiment-lineage graph in FalkorDB.
- Enable natural-language investigation of experiment history.
- Identify comparable and non-comparable experiments.
- Recommend controlled follow-up experiments under a compute budget.
- Trace downstream effects when source evidence becomes unreliable.
- Preserve provenance for claims, recommendations, and decisions.
- Demonstrate measurable differences from a retrieval-only baseline.

### Non-Goals

The MVP will not:

- Replace MLflow, Weights & Biases, or other experiment trackers.
- Train models automatically or implement a GPU job scheduler.
- Execute arbitrary user code.
- Guarantee that an experiment will improve model performance.
- Claim causal conclusions from observational comparisons alone.
- Automatically delete or permanently invalidate historical results.
- Implement every integration with external memory frameworks.
- Require a GPU to run the application.

## 4. Target Users and Core User Stories

### Primary user: ML researcher

Needs to compare previous runs, investigate hypotheses, plan controlled experiments, and preserve research knowledge.

### Secondary users

- **ML engineer:** investigates performance regressions and reproduces model behavior.
- **Research team lead:** reviews research progress, unreliable conclusions, and compute allocation.

### User stories

1. Import experiment logs and preserve previous work.
2. Ask natural-language questions about prior experiments.
3. Identify which runs are comparable and why.
4. Receive a next-experiment recommendation with a hypothesis, controls, metrics, and compute estimate.
5. Inspect the evidence supporting each recommendation.
6. Report a dataset defect or incorrect result and discover potentially affected conclusions.
7. Preserve historical records after corrections.
8. Inspect the graph paths that influenced a recommendation.

## 5. Functional Requirements

### FR-1: Experiment Import

Support CSV and JSON imports. Each experiment may contain:

- Experiment ID, project ID, name, and description
- Model family and architecture
- Model configuration
- Dataset name and version
- Preprocessing configuration
- Hyperparameters
- Training budget, steps, and duration
- Random seed
- Evaluation dataset and protocol
- Metrics and metric direction
- Status and timestamp
- Source file or source record
- Researcher notes
- Parent experiment or comparison references, when available

Support arbitrary JSON objects for model-specific configuration fields. Preserve unknown fields instead of discarding them.

Importer requirements:

1. Validate required fields.
2. Normalize recognized fields into a common schema.
3. Preserve model-specific fields.
4. Report malformed records and validation errors.
5. Prevent accidental duplicate ingestion.
6. Preserve source provenance.
7. Support mapping unfamiliar CSV column names to known fields.
8. Never silently invent missing values.

### FR-2: Experiment History

Provide a searchable experiment table with filters for model family, dataset/version, metric, status, date range, configuration, and review status. Selecting a run opens its configuration, metrics, source provenance, comparable runs, and lineage.

### FR-3: Natural-Language Research Copilot

Support questions such as:

- Which experiment performed best under comparable conditions?
- Why did performance decline after a dataset update?
- Which failed approaches have already been attempted?
- What should I test next with two GPU-hours?
- Which results depend on this dataset version?
- Which conclusions require revalidation?

The copilot must retrieve relevant records and use FalkorDB queries to establish relationships. It must not invent experiment IDs, metrics, evidence, or outcomes. If evidence is insufficient, it must say what is unknown.

### FR-4: Experiment Comparability

Assess comparability using task/dataset identity, dataset version, evaluation split/protocol, metric definition/direction, preprocessing, relevant model/training configuration, training budget, and known confounders.

Return one of:

- `COMPARABLE`
- `COMPARABLE_WITH_CAVEATS`
- `NOT_COMPARABLE`
- `INSUFFICIENT_INFORMATION`

Explain the reason for the assessment. Rules must be extensible and model-aware; not every model family uses the same configuration fields.

### FR-5: Next-Experiment Recommendation

Given a research objective, history, and compute budget, return:

- Objective and testable hypothesis
- Proposed configuration
- Baseline experiment
- Independent and controlled variables
- Evaluation metrics
- Estimated compute budget
- Rationale and supporting experiment IDs
- Uncertainties
- Expected information gain or prioritization rationale
- Validation and stopping criteria

Use historical runtimes or user-provided estimates. If estimates are unavailable, mark uncertainty or request input. The initial ranking can use an explainable heuristic based on relevance, evidence quality, information gain, cost, uncertainty, and feasibility. Do not present heuristic scores as calibrated probabilities.

### FR-6: Evidence Correction and Impact Analysis

Allow a user to report that a dataset, experiment, metric, or assumption may be unreliable.

The system must:

1. Record the correction and supporting evidence.
2. Preserve historical records.
3. Mark relevant sources or claims as disputed, superseded, or requiring review.
4. Traverse graph relationships to identify dependent experiments, comparisons, conclusions, and recommendations.
5. Show relationship paths explaining each impact.
6. Distinguish direct effects from potentially affected downstream conclusions.
7. Create a revalidation queue.
8. Recommend corrective evaluations or experiments.
9. Allow researcher review and resolution.
10. Preserve an audit history.

Do not automatically invalidate every experiment in a project. Impact must depend on actual dependencies and the nature of the defect.

### FR-7: Research Memory

Persist hypotheses, successful and failed approaches, lessons learned, outcomes, dataset assumptions, metric definitions, researcher-confirmed conclusions, reusable procedures, failure modes, and evidence provenance.

Distinguish observed results from inferred explanations and researcher-confirmed conclusions. Do not promote every LLM-generated statement to verified knowledge.

### FR-8: Explainability

Every important recommendation and impact result must be traceable to graph records. Show supporting experiment IDs, metrics/configurations, relationship paths, source records, comparability assessments, missing information, assumptions, uncertainty, and reasons for excluding conflicting evidence.

### FR-9: Researcher Approval

Recommendations are advisory. Users must approve consequential status changes and review resolutions. The MVP does not execute training jobs; it may export proposed configurations as JSON.

## 6. Primary User Flows

### Flow A: Import and investigate

1. Create a project.
2. Upload CSV/JSON.
3. Validate and normalize records.
4. Store experiment entities and relationships in FalkorDB.
5. Display runs and data-quality issues.
6. Open a run or ask a question.
7. Retrieve relevant records and query the graph.
8. Show an answer with supporting evidence.

### Flow B: Recommend an experiment

1. Enter a research objective and compute budget.
2. Retrieve relevant historical runs.
3. Assess comparability.
4. Identify gaps in previous experiments.
5. Generate and rank follow-up candidates.
6. Show hypothesis, controls, metrics, estimated cost, and evidence.
7. Let the user approve, edit, or reject.
8. Import results later if the experiment is run externally.

### Flow C: Correct an unreliable result

1. Select a dataset, result, or claim.
2. Record the issue and evidence.
3. Create a correction event and update graph status.
4. Traverse dependency relationships.
5. Generate an impact report and review queue.
6. Inspect affected graph paths.
7. Approve corrective actions.
8. Import new results and link them to the correction.
9. Record resolution without erasing history.

## 7. Technical Architecture

### Frontend

- React, TypeScript, Vite, Tailwind CSS
- A React-compatible graph visualization library
- A charting library for metrics

### Backend

- Python, FastAPI, Pydantic
- FalkorDB Python client
- Pandas for CSV ingestion
- Pluggable LLM client
- Pytest

### Database principle

FalkorDB is the primary database for experiment relationships, lineage, provenance, corrections, and dependency analysis. Do not introduce PostgreSQL or another graph/vector database unless a demonstrated need arises. For the MVP, represent normalized metadata and relationships in FalkorDB. Vector or lexical retrieval is optional and must not replace graph-based dependency analysis.

### Architecture

```text
React + TypeScript UI
        |
        v
FastAPI Application
        |
        +-- Import and Validation Service
        +-- Experiment Comparability Engine
        +-- Research Query Planner
        +-- Recommendation Engine
        +-- Evidence Correction Service
        +-- Graph Lineage and Impact Analyzer
        |
        v
FalkorDB
        +-- Experiments and Configurations
        +-- Datasets and Versions
        +-- Metrics and Claims
        +-- Evidence and Provenance
        +-- Dependencies and Lineage
        +-- Corrections and Review Events
```

## 8. FalkorDB Graph Schema

### Node labels

- **Project:** `id`, `name`, `description`, `created_at`
- **Experiment:** `id`, `project_id`, `name`, `model_family`, `status`, `created_at`, `duration_seconds`, `compute_cost_gpu_hours`
- **Configuration:** `id`, `canonical_json`, `config_hash`, `created_at`
- **Dataset:** `id`, `name`, `description`
- **DatasetVersion:** `id`, `dataset_id`, `version`, `quality_status`, `created_at`
- **Metric:** `id`, `experiment_id`, `name`, `value`, `direction`, `evaluation_split`, `evaluation_protocol`
- **Hypothesis:** `id`, `text`, `status`, `created_at`
- **Claim:** `id`, `text`, `status`, `confidence`, `created_at`, `reviewed_at`
- **Evidence:** `id`, `source_type`, `source_reference`, `description`, `created_at`
- **Correction:** `id`, `description`, `status`, `created_at`, `created_by`
- **Recommendation:** `id`, `objective`, `status`, `estimated_gpu_hours`, `created_at`
- **ReviewItem:** `id`, `reason`, `status`, `created_at`
- **Researcher:** `id`, `name`

Add fields only when needed; avoid duplicating facts across unrelated nodes.

### Relationship types

- `(Project)-[:CONTAINS]->(Experiment)`
- `(Experiment)-[:USES_CONFIG]->(Configuration)`
- `(Experiment)-[:USES_DATASET_VERSION]->(DatasetVersion)`
- `(DatasetVersion)-[:VERSION_OF]->(Dataset)`
- `(Experiment)-[:PRODUCES_METRIC]->(Metric)`
- `(Experiment)-[:TESTS]->(Hypothesis)`
- `(Experiment)-[:BASED_ON]->(Claim)`
- `(Claim)-[:SUPPORTED_BY]->(Evidence)`
- `(Claim)-[:CONTRADICTS]->(Claim)`
- `(Claim)-[:SUPERSEDES]->(Claim)`
- `(Experiment)-[:PARENT_OF]->(Experiment)`
- `(Experiment)-[:COMPARED_WITH]->(Experiment)`
- `(Correction)-[:CHALLENGES]->(Evidence)`
- `(Correction)-[:AFFECTS]->(DatasetVersion)`
- `(Correction)-[:AFFECTS]->(Experiment)`
- `(Recommendation)-[:SUPPORTED_BY]->(Experiment)`
- `(Recommendation)-[:PROPOSES]->(Experiment)`
- `(ReviewItem)-[:CONCERNS]->(Experiment)`
- `(Experiment)-[:REVIEWED_BY]->(Researcher)`

Use stable IDs and uniqueness constraints supported by the selected FalkorDB version. Validate Cypher syntax against that version.

### Example query: runs using a dataset version

```cypher
MATCH (d:DatasetVersion {id: $dataset_version_id})
      <-[:USES_DATASET_VERSION]-(e:Experiment)
OPTIONAL MATCH (e)-[:PRODUCES_METRIC]->(m:Metric)
RETURN e.id AS experiment_id,
       e.name AS experiment_name,
       e.status AS status,
       collect({
         name: m.name,
         value: m.value,
         split: m.evaluation_split
       }) AS metrics
```

### Impact traversal

Use bounded graph traversal over explicitly defined dependency relationships. Avoid treating mere comparison or similarity links as causal dependencies. Prevent duplicate results and distinguish direct from indirect impact.

## 9. API Specification

Use `/api/v1`.

### Projects

- `POST /api/v1/projects`
- `GET /api/v1/projects`
- `GET /api/v1/projects/{project_id}`

### Experiment ingestion/history

- `POST /api/v1/projects/{project_id}/experiments/import`
- `GET /api/v1/projects/{project_id}/experiments`
- `GET /api/v1/experiments/{experiment_id}`
- `GET /api/v1/experiments/{experiment_id}/comparability`
- `GET /api/v1/experiments/{experiment_id}/lineage`

Example import response:

```json
{
  "imported": 80,
  "skipped": 2,
  "failed": 1,
  "errors": [
    {
      "row": 17,
      "reason": "Missing required experiment identifier"
    }
  ]
}
```

### Copilot/recommendations

- `POST /api/v1/projects/{project_id}/query`
- `POST /api/v1/projects/{project_id}/recommendations`
- `GET /api/v1/recommendations/{recommendation_id}`
- `POST /api/v1/recommendations/{recommendation_id}/decision`

Example query body:

```json
{
  "question": "Which experiments are comparable, and what should I try next?",
  "compute_budget_gpu_hours": 4
}
```

### Corrections/impact

- `POST /api/v1/projects/{project_id}/corrections`
- `GET /api/v1/corrections/{correction_id}/impact`
- `GET /api/v1/projects/{project_id}/review-queue`
- `POST /api/v1/review-items/{review_item_id}/resolve`

### Graph

- `GET /api/v1/projects/{project_id}/graph/summary`

All endpoints must validate input, return structured errors, enforce project access, and prohibit unrestricted frontend-supplied Cypher.

## 10. Frontend Screens

### Screen A: Experiment Workspace

- Project selector
- CSV/JSON import
- Validation summary
- Experiment history table
- Dataset/model/metric/status filters
- Metric comparison charts
- Experiment detail drawer

### Screen B: Research Copilot

- Chat-style query input
- Research objective and compute-budget inputs
- Evidence-backed response
- Supporting experiment cards
- Comparability warnings
- Recommendation card
- Hypothesis, controls, metrics, and estimated cost
- Approve/edit/reject actions

### Screen C: Lineage and Impact Analysis

- Interactive graph
- Dataset/version, experiment, metric, evidence, and correction nodes
- Highlighted dependency paths
- Impact summary
- Revalidation queue
- Review/resolution actions

Clearly distinguish verified facts, user assertions, inferred explanations, disputed evidence, and unresolved items. Visualizations must use real backend data.

## 11. Recommendation Engine

Processing pipeline:

1. Parse objective and constraints.
2. Retrieve relevant experiments.
3. Identify relevant model families, datasets, and metrics.
4. Assess comparability.
5. Exclude or qualify evidence affected by corrections.
6. Identify gaps in prior experiments.
7. Generate candidate follow-up experiments.
8. Enforce the compute budget.
9. Rank candidates with an explainable heuristic.
10. Return structured recommendations and supporting evidence.

Start with deterministic rules and templates. Use an LLM for interpreting the question and explaining structured results, not for inventing lineage or metrics. Advanced Bayesian optimization is out of scope.

## 12. Data Quality and Trust

- Never fabricate experimental metrics.
- Never treat missing data as verified.
- Preserve raw records or durable source references.
- Track dataset versions separately.
- Distinguish observed facts from generated explanations.
- Record source provenance for important claims.
- Preserve correction and review history.
- Require approval for consequential status changes.
- Explain uncertainty in comparisons.
- Do not infer causality from connectivity alone.
- Do not invalidate every record associated with a correction.
- Prevent unauthorized cross-project data access.

## 13. Testing and Evaluation

### Unit tests

Test import validation, normalization, duplicate detection, metric normalization, comparability, budget enforcement, correction transitions, dependency traversal, API validation, and authorization.

### Integration tests

Test FalkorDB ingestion, re-import, lineage creation, experiment queries, corrections, dependency traversal, review resolution, and API evidence responses.

### End-to-end tests

A user must be able to create a project, import sample data, inspect runs, ask a question, receive an evidence-backed recommendation, report a dataset defect, inspect impacted runs, and review corrective actions.

### Quantitative evaluation

Use synthetic datasets with known ground truth. Track:

- Import success rate
- Duplicate-import correctness
- Comparability accuracy
- Recommendation constraint compliance
- Dependency recall
- False-positive impact rate
- Evidence citation coverage
- Time to identify affected experiments
- Compute-budget compliance

Compare with a retrieval-only baseline that does not traverse dependency relationships. Do not claim gains before running the evaluation.

## 14. Mandatory Demo

Create 50–100 synthetic experiments across multiple model families, dataset versions, configurations, and outcomes. Include successful/failed runs, different metrics, comparable/non-comparable runs, conclusions with known dependencies, one known data-quality defect, and at least one unaffected experiment.

Demo sequence:

1. Import the synthetic experiment history.
2. Ask which runs are comparable and inspect supporting records.
3. Ask what experiment to run next under a fixed compute budget.
4. Report a dataset defect and show graph traversal to affected runs and conclusions.
5. Repeat the recommendation query and show that evidence affected by the correction is now excluded or qualified, while unaffected evidence remains usable.

The before/after result must come from actual graph state and queries, not hardcoded responses.

## 15. Deployment and Configuration

Provide Docker Compose for local FalkorDB, `.env.example`, backend and frontend setup instructions, seed-data commands, test commands, and a demo reset procedure.

The application must work in deterministic demo mode without an LLM key. Use query templates and deterministic explanations; never pretend an external model was called.

Never commit API keys or secrets.

## 16. Definition of Done

- [ ] Runs locally from documented instructions.
- [ ] CSV/JSON imports work.
- [ ] Multiple ML model families are supported.
- [ ] FalkorDB stores experiment graph and lineage.
- [ ] Experiment history is visible in the UI.
- [ ] Comparability results explain their reasoning.
- [ ] Questions return evidence-backed answers.
- [ ] Recommendations contain controlled variables and compute constraints.
- [ ] Corrections create auditable impact reports.
- [ ] Benchmark tests affected and unaffected records.
- [ ] Corrections change later graph-backed recommendations.
- [ ] Users can inspect supporting graph paths.
- [ ] Baseline and evaluation procedure are documented.
- [ ] Critical graph operations and business logic have tests.
- [ ] Demo is reproducible from seed data.
- [ ] No core demo action depends on hardcoded final answers.

---

# Part II — Four-Person Parallel Implementation Plan

## 17. Collaboration Objective

Four contributors will implement RUNWISE in a **single shared Git branch**. They may work and push in any order. The plan must minimize merge conflicts and avoid requiring one contributor to finish before others can begin.

The key coordination rule is: **agree on interfaces and shared contracts first; divide ownership by files and directories; integrate through small, backward-compatible commits.**

Do not have all four contributors edit the same app entrypoint, graph schema, API router, or package manifest at the same time.

## 18. Team Ownership

### Person A — Graph Database and Data Ingestion

**Mission:** Make experiment data real and queryable in FalkorDB.

Owns:

```text
backend/app/graph/**
backend/app/services/ingestion.py
backend/app/services/impact_analysis.py
backend/app/api/experiments.py
backend/app/api/imports.py
backend/tests/test_graph/**
backend/tests/test_ingestion.py
```

Responsibilities:

- Implement FalkorDB connection and graph initialization.
- Implement node/relationship creation and schema initialization.
- Implement CSV/JSON ingestion and normalization.
- Implement stable IDs and duplicate handling.
- Implement experiment history and lineage queries.
- Implement correction-related graph traversal and impact data retrieval.
- Write tests for graph persistence, ingestion, lineage, and impact queries.
- Document graph schema and key Cypher queries.

Deliverables:

1. Working graph schema and connection.
2. Import endpoint.
3. Experiment list/detail/lineage endpoints.
4. Correction impact query service.
5. Tests and graph documentation.

Must not own the general FastAPI entrypoint, frontend, or recommendation ranking engine.

### Person B — Recommendation and Comparability Engine

**Mission:** Turn experiment history into defensible recommendations.

Owns:

```text
backend/app/services/comparability.py
backend/app/services/recommendations.py
backend/app/services/evidence.py
backend/app/api/recommendations.py
backend/app/api/copilot.py
backend/tests/test_comparability.py
backend/tests/test_recommendations.py
backend/tests/test_evidence.py
```

Responsibilities:

- Implement comparability rules and reason codes.
- Implement candidate experiment generation.
- Implement compute-budget enforcement.
- Implement explainable candidate ranking.
- Define typed recommendation and evidence response models in the agreed contracts module.
- Implement the research query orchestration service.
- Use graph/data access interfaces agreed with Person A.
- Add a pluggable LLM adapter and deterministic fallback.
- Ensure answers cite actual experiment IDs and evidence returned by the backend.
- Test recommendations, insufficient evidence, and budget constraints.

Deliverables:

1. Comparability engine.
2. Recommendation engine.
3. Query/coplan orchestration.
4. Evidence-grounded response models.
5. Tests and documented ranking rules.

Must not edit Person A's graph implementation directly. Request graph-query additions through the shared interface or a small explicit handoff.

### Person C — Frontend and User Experience

**Mission:** Build the three screens that make RUNWISE understandable and demo-ready.

Owns:

```text
frontend/**
```

Responsibilities:

- Create the React/TypeScript/Vite application.
- Implement project selection and experiment workspace.
- Implement CSV/JSON import UX.
- Implement experiment table, filters, metric charts, and detail drawer.
- Implement research copilot chat and recommendation cards.
- Implement graph lineage and impact review screen.
- Build a typed API client using the agreed API contracts.
- Provide loading, empty, error, and demo-mode states.
- Ensure graph visualizations display backend results, not hardcoded demo graphs.
- Write frontend tests for critical interactions where practical.

Deliverables:

1. Experiment Workspace.
2. Research Copilot.
3. Lineage and Impact Analysis.
4. Typed API client.
5. Responsive, polished hackathon UI.

Must not edit backend routes or the root deployment configuration unless coordinated explicitly.

### Person D — Integration, Demo Data, Testing, and Deployment

**Mission:** Ensure a fresh clone runs and the entire demo works end to end.

Owns:

```text
docker-compose.yml
.env.example
sample_data/**
scripts/**
docs/**
backend/tests/test_api/**
backend/tests/test_e2e/**
```

Responsibilities:

- Set up Docker Compose for FalkorDB.
- Create synthetic experiment datasets with known ground truth.
- Build seed/reset scripts.
- Create demo scenarios covering recommendation and correction.
- Add API and end-to-end tests.
- Implement or coordinate the retrieval-only baseline.
- Create the evaluation script and metric report.
- Write setup, troubleshooting, architecture, and demo instructions.
- Run integration checks after other contributors push.
- Verify the project works without an LLM API key.
- Record the final demo or prepare a reproducible demo script.

Deliverables:

1. Local deployment and configuration.
2. Seed dataset and deterministic scenarios.
3. Automated API/end-to-end test suite.
4. Evaluation and baseline.
5. README and demo instructions.

Must not redesign core graph or recommendation logic independently; report gaps and coordinate targeted fixes with the relevant owner.

## 19. Shared Files and Ownership Rules

The following files are high-conflict. Person D coordinates their initial setup, then all contributors follow the contract.

| Shared item | Owner | Rule |
|---|---|---|
| `backend/app/main.py` | Person D initially | Single owner; contributors add routers through documented registration points |
| `backend/app/models/contracts.py` | Person B initially | Shared request/response models; changes announced before use |
| `backend/app/graph/schema.py` | Person A | Others request changes; no parallel schema edits |
| `backend/app/graph/client.py` | Person A | Single owner for database connection and query conventions |
| `backend/pyproject.toml` | Person D | Dependency additions requested in chat; one person edits at a time |
| `frontend/package.json` | Person C | Person C owns all frontend dependencies |
| `.env.example` | Person D | Contributors request new variables; Person D consolidates |
| `docker-compose.yml` | Person D | Single owner |
| `README.md` | Person D | Contributors provide section text; Person D integrates |

Create the agreed contracts and placeholder modules early so that everyone can work against stable interfaces.

## 20. Required Contracts Before Parallel Coding

Agree on these interfaces before implementing dependent logic. Use Pydantic models on the backend and matching TypeScript types on the frontend.

### Experiment summary

```json
{
  "id": "exp_001",
  "project_id": "project_001",
  "name": "baseline_xgboost",
  "model_family": "gradient_boosting",
  "dataset_version_id": "dataset_v1",
  "status": "completed",
  "metrics": [
    {
      "name": "validation_accuracy",
      "value": 0.91,
      "direction": "maximize",
      "evaluation_split": "validation"
    }
  ],
  "compute_cost_gpu_hours": 0.8
}
```

### Recommendation

```json
{
  "id": "rec_001",
  "objective": "Improve validation performance within budget",
  "hypothesis": "A controlled change to max_depth may improve generalization",
  "baseline_experiment_ids": ["exp_001"],
  "configuration": {
    "max_depth": 6
  },
  "independent_variables": ["max_depth"],
  "controlled_variables": [
    "dataset_version",
    "evaluation_protocol",
    "training_budget"
  ],
  "metrics": ["validation_accuracy"],
  "estimated_gpu_hours": 0.7,
  "supporting_experiment_ids": ["exp_001"],
  "uncertainties": ["Only one comparable baseline is available"],
  "status": "proposed"
}
```

### Correction impact

```json
{
  "correction_id": "corr_001",
  "source_id": "dataset_v1",
  "summary": "Potential train-validation leakage discovered",
  "directly_affected_experiment_ids": ["exp_001"],
  "potentially_affected_claim_ids": ["claim_001"],
  "unaffected_experiment_ids": ["exp_009"],
  "review_item_ids": ["review_001"],
  "paths": [
    {
      "nodes": ["dataset_v1", "exp_001", "claim_001"],
      "relationship_types": [
        "USES_DATASET_VERSION",
        "BASED_ON"
      ]
    }
  ]
}
```

These examples are interface contracts, not mandatory hardcoded values. Add fields only through coordinated, backward-compatible changes. The frontend must tolerate optional fields and empty lists.

## 21. Parallel Work Strategy

### Rule 1: Work in separate ownership areas

Each person should primarily edit their assigned directories. This reduces textual conflicts even when commits arrive in any order.

### Rule 2: Establish interfaces first

Create the following minimal files early:

- Pydantic request/response contracts.
- FalkorDB client interface.
- Router registration mechanism.
- Frontend TypeScript API types.
- Seed-data format.
- Environment variable contract.

Use placeholder implementations when needed so parallel work can proceed.

### Rule 3: Integrate through adapters

Person B must not depend on the internal implementation details of Person A's graph queries. Define a small graph service interface with operations such as:

- `list_experiments(project_id, filters)`
- `get_experiment(experiment_id)`
- `find_comparable_candidates(experiment_id or query_context)`
- `trace_dataset_dependencies(dataset_version_id)`
- `create_correction(correction_payload)`
- `get_project_graph(project_id)`

Person A implements the graph access layer. Person B consumes the interface. Person C consumes the HTTP API. Person D verifies the full stack.

If an operation is missing, agree on the input/output contract before adding it.

### Rule 4: Keep commits small

Use focused commits such as:

- `feat(graph): initialize experiment schema`
- `feat(ingestion): add CSV import validation`
- `feat(recommendations): add budget-aware candidate ranking`
- `feat(ui): add experiment workspace`
- `test(demo): add dataset leakage scenario`

Avoid giant commits that mix unrelated frontend, graph, and deployment changes.

### Rule 5: Do not overwrite another person's changes

Before editing a shared file:

1. Check the current Git status.
2. Pull the latest shared branch.
3. Inspect recent commits.
4. Coordinate with the file owner.
5. Make a small edit and commit promptly.

Never use `git push --force` on the shared branch. Do not use destructive resets or restore commands against another contributor's uncommitted work.

## 22. Same-Branch Collaboration Protocol

All contributors push to one branch, for example `main`. This is possible, but requires discipline because simultaneous pushes can race.

### Before each work session

```bash
git status
git pull --rebase origin main
```

If there are local changes, commit or safely stash them before pulling. Do not overwrite uncommitted work.

### After a focused change

```bash
git add <your-owned-files>
git commit -m "feat(area): describe change"
git pull --rebase origin main
git push origin main
```

Replace `<your-owned-files>` with explicit paths. Avoid `git add .` when other contributors may have generated or modified files in the same working tree.

If the push is rejected because another contributor pushed first:

```bash
git pull --rebase origin main
```

Resolve any conflicts, rerun relevant tests, then push again.

If a rebase conflict occurs in a file owned by another person, do not guess at the intended resolution. Contact that owner, inspect both changes, and preserve both compatible behaviors. Continue the rebase only after the conflict is resolved.

### Shared working directory warning

If all four people literally use the same filesystem checkout, they share one index and working tree. They must not run `git checkout`, `git reset`, rebase, or other worktree-changing operations concurrently. Only one person should perform Git operations at a time, or use a shared lock/coordination mechanism.

If each person has a separate clone but pushes to the same branch, the pull/rebase/push protocol above is appropriate. Separate clones are safer even when all work targets the same branch.

## 23. Order-Independent Milestones

These milestones are defined by integration contracts rather than a strict sequence.

### Milestone A: Interfaces and skeleton

- Agree on the graph schema and API contracts.
- Create placeholder routers and service interfaces.
- Establish project setup and lint/test commands.
- Generate a small valid seed dataset.

**Exit criteria:** All contributors can work independently without guessing response shapes.

### Milestone B: Parallel implementation

- Person A builds graph storage and ingestion.
- Person B builds comparability and recommendation logic against agreed interfaces and fixtures.
- Person C builds screens against typed API contracts and mock responses.
- Person D builds deployment, seed data, tests, and demo documentation.

**Exit criteria:** Each area works independently and has a clear integration point.

### Milestone C: Integration

- Connect real frontend API calls.
- Connect recommendation logic to real graph query results.
- Connect correction input to graph impact analysis.
- Replace mock data in the core demo.
- Resolve integration defects with the file owner responsible for each subsystem.

**Exit criteria:** The main demo works against real FalkorDB data.

### Milestone D: Verification and demo

- Run the application from a clean environment.
- Execute the full test suite.
- Run the baseline comparison.
- Verify that a correction changes subsequent recommendations.
- Confirm that at least one unaffected experiment remains usable.
- Record the final demonstration.

**Exit criteria:** The demo is repeatable and all important results are derived from actual stored data.

## 24. Integration Checklist

### Backend and graph

- [ ] FalkorDB starts through documented instructions.
- [ ] Schema initialization is idempotent.
- [ ] Import endpoints return the agreed response shape.
- [ ] Duplicate imports are handled consistently.
- [ ] Graph queries enforce project boundaries.
- [ ] Impact traversal returns explicit paths and bounded results.

### Recommendation engine

- [ ] Comparability returns a defined status and reasons.
- [ ] Recommendations reference real experiment IDs.
- [ ] Compute-budget constraints are enforced.
- [ ] Uncertain or missing evidence is clearly identified.
- [ ] Corrections influence the evidence used by future recommendations.

### Frontend

- [ ] API types match backend response models.
- [ ] Loading, empty, and error states are handled.
- [ ] Experiment records are fetched from the backend.
- [ ] Graph nodes and paths come from actual graph results.
- [ ] Approval/review actions persist through the API.

### Deployment and demo

- [ ] Fresh setup works from the README.
- [ ] `.env.example` includes every required variable.
- [ ] Demo mode works without an LLM API key.
- [ ] Seed/reset scripts are deterministic.
- [ ] Synthetic ground truth is documented.
- [ ] Evaluation results can be reproduced.

## 25. Definition of Done

The MVP is ready when:

- CSV/JSON imports work across multiple ML model families.
- FalkorDB stores experiments, datasets, evidence, and dependencies.
- The UI displays experiment history and lineage.
- Comparability checks explain their conclusions.
- The copilot produces evidence-backed recommendations under a budget.
- Corrections create an auditable impact report.
- The correction changes future recommendations through graph-backed logic.
- Affected and unaffected records are distinguished in the benchmark.
- Critical paths have tests.
- The application runs locally from a clean checkout.
- The hackathon demo is reproducible without hardcoded final answers.

## 26. Final Engineering Principle

RUNWISE is not a general-purpose agent-memory framework and not another experiment tracker. It is a research decision-support product.

Its defining behavior is:

**When evidence changes what a researcher should believe about a previous experiment, RUNWISE finds the dependent conclusions and changes what it recommends next—while explaining the graph paths behind that decision.**

FalkorDB must provide the relationships and dependency traversals that make this behavior possible. The LLM supports interpretation and explanation; it must not replace deterministic lineage analysis.
