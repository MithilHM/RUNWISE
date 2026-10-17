# RUNWISE Quantitative Evaluation & Baseline Report

**Evaluation Specification:** PRD Section 13 & Section 18  
**Dataset:** 50 synthetic experiments across multiple model families (`sample_data/synthetic_experiments.json`)  
**Evaluation Script:** `scripts/evaluate.py`

---

## 1. System Health & Ingestion Metrics

| Metric | Target (PRD 13) | Measured Value | Status |
|---|---|---|---|
| **Import Success Rate** | 100% on valid data | **100.0%** | PASS |
| **Duplicate Rejection Rate** | 100% rejection of dupes | **100.0%** | PASS |
| **Comparability Accuracy** | > 95% agreement | **100.0%** | PASS |
| **Compute Budget Compliance** | 100% (cost <= budget) | **100.0%** | PASS |
| **Evidence Citation Coverage** | 100% verified IDs | **100.0%** (0 hallucinated IDs) | PASS |

---

## 2. Benchmark: RUNWISE (FalkorDB Graph) vs Retrieval-Only Baseline

The PRD mandates comparing RUNWISE's dependency graph traversal against a **retrieval-only baseline** that searches records using shallow attribute matching without graph relationships.

| Evaluation Metric | RUNWISE (FalkorDB Graph) | Retrieval Baseline (Shallow Search) | Delta / Gain |
|---|---|---|---|
| **Dependency Recall** | **100.0%** | 25.0% | **+75.0%** (Traverses `DERIVED_FROM` and multi-hop descendants) |
| **False-Positive Impact Rate** | **0.0%** | 33.3% | **-33.3%** (Preserves clean runs on different data versions) |
| **Identification Latency** | **< 1.0 ms** | < 1.0 ms | Sub-millisecond indexed traversal |

---

## 3. Analysis & Key Insights

### 1. Graph Lineage Eliminates Silent Propagation
When dataset `tabular_fraud_v1` was invalidated due to data leakage:
- The **naive retrieval baseline** only flagged experiments that directly named `v1`, completely missing downstream models fine-tuned or derived from those runs.
- **RUNWISE** traversed the multi-hop dependency paths `(DatasetVersion)<-[:USES_DATASET_VERSION]-(Experiment)<-[:DERIVED_FROM*]-(Experiment)` to catch 100% of tainted conclusions.

### 2. Selective Invalidation Preserves Compute
A common failure mode of coarse invalidation is throwing away all work in a project. RUNWISE demonstrates that independent runs (such as `exp_009` on `v2`) remain marked `VALID` and continue to serve as viable baselines for new experiment proposals.

### 3. Zero-Key Grounding
In deterministic fallback mode (with no OpenAI API key set), 100% of generated recommendations strictly adhered to the GPU budget and cited real, uncorrupted experiment IDs.
