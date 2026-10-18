# RUNWISE — 3-Minute Hackathon Demo Script

**Target Time:** 3 minutes  
**Goal:** Prove RUNWISE prevents wasted compute and invalid research conclusions by combining FalkorDB graph memory with an adaptive recommendation copilot.

The seeded demo state already carries the `tabular_fraud_v1` correction, so every step runs against a live graph: Act III replays the same deterministic traversal rather than pretending the defect is new.

---

## Pre-Demo Checklist (30 seconds before recording)
1. In terminal: `python scripts/prepare_demo.py` (Resets and seeds the 53-run demo).
2. Start backend: `cd backend && uvicorn app.main:app --port 8000`.
3. Start frontend: `cd frontend && npm run dev` (Open `http://localhost:5173`).

---

## Act I: The Problem & Experiment Workspace (0:00 – 1:00)
**Narrative:** *"ML research generates tons of disjointed runs. When you compare models, how do you know if an improvement is genuine or confounded by different datasets and configs?"*

1. **Screen A (Workspace):**
   - Show `project_demo` with 53 synthetic runs across multiple model families, including two runs derived from the leaked lineage (`exp_051`, `exp_052`) and a clean control on the repaired data (`exp_053`).
   - Filter by `model_family = gradient_boosting`.
   - Click `exp_002` (tuned XGBoost) and open the detail drawer.
   - **Point to Comparability Badge:** `COMPARABLE_WITH_CAVEATS` against `exp_001` — same data and split, the only configuration change is one variable — and the caveat row already names the open correction.
2. **The teaser (say it out loud):** open `exp_051`: its dataset is `tabular_sales_v3`, nothing to do with the fraud version, yet it is amber `Needs Review`. *"Hold that thought."*

---

## Act II: Zero-Key Research Copilot & Budget Recommendations (1:00 – 2:00)
**Narrative:** *"Researchers need to know what to test next without exceeding compute budgets, and without hallucinated claims."*

1. **Screen B (Research Copilot):**
   - In chat, type:
     ```text
     What is the best performing baseline for fraud detection, and what should I test next under a 2 GPU-hour budget?
     ```
   - **Highlight Key Details (all shown in the answer):**
     - Runs **100% deterministically** with no external LLM key needed (`mode: deterministic`, `llm_used: false`).
     - Trust is stated, not hinted: `Withheld under an open correction: 5 run(s) (exp_052, exp_051, exp_003, exp_002, exp_001). These numbers are not used as support.`
     - The best *measured* number is honest about itself: `Highest recorded validation_accuracy: 0.94 on exp_047... evaluation_protocol not recorded` — a measurement, not yet a comparison.
     - The nearest proposal stays within budget (2.0 GPU-hours) and cites only runs the backend actually returned (`exp_047`).
   - Click **"Approve Proposal"** (shows FR-9 decision approval flow).

---

## Act III: The Magic Moment — Dataset Invalidation & Adaptive Lineage (2:00 – 3:00)
**Narrative:** *"Here is the defining behavior of RUNWISE. What happens when a researcher discovers a defect in a dataset version?"*

1. **Screen C (Lineage & Impact Screen):**
   - Click **"Report Dataset Defect"** and replay the recording:
     - **Source:** `tabular_fraud_v1`
     - **Reason:** `Data leakage discovered between train and test splits`
   - **Show Graph Invalidation Traversal** (the seeded replay walks the identical paths, deterministically):
     - FalkorDB follows multi-hop edges: `(tabular_fraud_v1)<-[:USES_DATASET_VERSION]-(exp_002)<-[:PARENT_OF]-(exp_051)-[:PARENT_OF]->(exp_052)`.
     - `exp_001`, `exp_002` and `exp_003` turn amber as **directly** affected.
     - **The magic moment:** `exp_051` and `exp_052` turn amber too — **runs that never touched the defective dataset version at all**, caught only because validity follows derivation through the graph. No dataset-name search can reach them. (The review queue names them "Transitively derived from affected predecessor in lineage.")
     - **Point to `exp_053` (fresh baseline on the repaired v2):** stays green — RUNWISE does not blindly scrap clean research, recorded with its evaluation protocol so it remains citable.
2. **Back to Screen B (Adaptive Recalibration):**
   - Re-run the recommendation query.
   - **Show the Difference:**
     - All five runs — three direct, two *derived* — are strictly excluded from support.
     - The Copilot rebuilds its proposals on trusted evidence only, and the queue offers the revalidation decision to a human.
   - **Conclusion:** *"RUNWISE prevents researchers from building on flawed evidence, even evidence that was derived from it, saving GPU hours and research integrity."*
