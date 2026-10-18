# RUNWISE Quantitative Evaluation & Baseline Report

**Benchmark Run:** 2026-10-18 19:56:16
**Ground Truth:** 53 generated synthetic experiments. Affected by a defect in `tabular_fraud_v1`: exp_001, exp_002, exp_003 directly, exp_051, exp_052 through derivation. Nothing else is affected.
**Method:** the 'FalkorDB graph' column is the production `ImpactAnalysisService` against live FalkorDB on a throwaway graph, dropped afterwards. The retrieval baseline flags every run whose *dataset name* matches, version-blind, without following derivation. Ground truth is computed independently of both.

## 1. Ground-Truth Data & Ingestion Quality (PRD 13)
- **Import Success Rate:** 100.0% (667 ms for 53 runs)
- **Duplicate Rejection Rate:** 100.0% (measured: a second identical import, not asserted)
- **Comparability Accuracy:** 100.0%
- **Compute Budget Compliance:** 100.0% (Cost strictly within allocated GPU-hours)
- **Evidence Citation Coverage:** 100.0% (all citation ids resolve to real records)

## 2. Benchmark Comparison: RUNWISE Lineage vs Retrieval Baseline

| Evaluation Metric (PRD 13) | RUNWISE (FalkorDB Graph) | Retrieval Baseline (Shallow Search) | Delta / Advantage |
|---|---|---|---|
| **Dependency Recall** | **100.0%** | 60.0% | **+40.0 pts** (catches the derived exp_051/exp_052, which dataset-name matching cannot see) |
| **False-Positive Impact Rate** | **0.0%** | 2.1% | **2.1 pts spared** (the baseline sweeps exp_053 into the affected set: clean run(s) on the repaired v2, taken because the dataset name matches) |
| **Identification Latency (traversal)** | **3.6 ms** (read-side re-computation) | 0.020 ms | full impact workflow incl. correction persistence: 41.9 ms |

## 3. Findings & Conclusions
1. **Derived-lineage recall, measured.** The correction on `tabular_fraud_v1` invalidates exp_001, exp_002, exp_003 directly and also exp_051, exp_052, which are runs that never named that dataset version: they descend from an affected run through derivation. Shallow retrieval finds only the direct users (60%) and reports the two descendants as valid, silently; the graph traversal finds all of them (100%).
2. **Zero-Key Deterministic Guarantee:** recommendations adhere to the compute budget and cite only grounded record ids, with no external LLM dependency (citation coverage 100.0%).
3. **Surgical Invalidation, measured on both sides.** 48 unaffected runs stay valid, including `exp_053`, a clean baseline on the repaired `tabular_fraud_v2`: the exact-version graph does not flag it, while name-level retrieval sweeps it in with the bad version.