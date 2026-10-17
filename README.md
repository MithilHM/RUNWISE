# RUNWISE — AI Research Experiment Copilot

## Architecture & Graph Foundation (Person A Implementation)

RUNWISE converts scattered ML experiment histories into a persistent, graph-backed research intelligence platform powered by **FalkorDB**, **FastAPI**, and **React**.

---

## 1. Quickstart & Setup

### Prerequisites
- Python 3.10+
- Docker (for FalkorDB)

### Start FalkorDB
```bash
docker compose up -d
# or via docker run:
docker run -d --name runwise-falkordb -p 6379:6379 falkordb/falkordb:latest
```

### Install Dependencies
```bash
pip install -r requirements.txt
# or install directly:
pip install falkordb fastapi uvicorn pydantic pydantic-settings pandas pytest pytest-asyncio httpx python-multipart
```

### Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default settings connect to `127.0.0.1:6379` with default graph `runwise`.

### Seed Demo Data & Demonstrate Impact Propagation
```bash
python scripts/seed_demo.py
```

### Run Automated Tests
```bash
python -m pytest backend/tests -v
```

### Start the FastAPI Dev Server
```bash
uvicorn backend.app.main:app --reload --port 8000
```
Interactive API docs available at: `http://localhost:8000/docs`

---

## 2. Shared Integration Interfaces for Teammates

### For Person B (Recommendation & Comparability Engine)
You can directly import and call the graph services without writing raw Cypher:

```python
from backend.app.graph.repository import GraphRepository

repo = GraphRepository()

# 1. Fetch experiment detail with parsed config parameters & metrics
exp = repo.get_experiment("exp_001")
print(exp.model_family, exp.configuration, exp.metrics)

# 2. Filter experiments by project, status, model family, dataset
experiments = repo.list_experiments(
    project_id="project_demo",
    model_family="gradient_boosting",
    quality_status="valid"  # excludes unvalidated/affected runs!
)

# 3. Find candidate comparable experiments
related = repo.find_related_experiments("exp_001")
# Returns: {"same_dataset_version": [...], "same_configuration": [...], "siblings": [...]}

# 4. Check lineage
lineage = repo.get_experiment_lineage("exp_002", max_depth=5)
```

### For Person C (Frontend & User Experience)
The backend exposes typed endpoints matching the PRD contracts:
- `POST /api/v1/projects/{project_id}/experiments/import`: Upload CSV file or JSON array.
- `GET /api/v1/projects/{project_id}/experiments`: Filterable experiment table data.
- `GET /api/v1/experiments/{id}`: Detailed drawer view with config & metrics.
- `GET /api/v1/experiments/{id}/lineage`: Ancestor & descendant graph tree.
- `GET /api/v1/projects/{project_id}/graph`: Full project graph nodes and edges for visualization.
- `POST /api/v1/projects/{project_id}/corrections`: Trigger correction & impact analysis.
- `GET /api/v1/projects/{project_id}/review-queue`: Active items requiring revalidation.
- `POST /api/v1/review-items/{id}/resolve`: Resolve flagged items.

### For Person D (Integration, Baseline, & E2E)
- Seed script: `python scripts/seed_demo.py`
- Synthetic datasets: `sample_data/synthetic_experiments.json` and `sample_data/synthetic_experiments.csv`
- Test suite: `pytest backend/tests -v`
