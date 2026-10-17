# RUNWISE — 3-Minute Hackathon Demo Script

**Target Time:** 3 minutes  
**Goal:** Prove RUNWISE prevents wasted compute and invalid research conclusions by combining FalkorDB graph memory with an adaptive recommendation copilot.

---

## Pre-Demo Checklist (30 seconds before recording)
1. In terminal: `python scripts/prepare_demo.py` (Resets and seeds clean 50-run demo).
2. Start backend: `cd backend && uvicorn app.main:app --port 8000`.
3. Start frontend: `cd frontend && npm run dev` (Open `http://localhost:5173`).

---

## Act I: The Problem & Experiment Workspace (0:00 – 1:00)
**Narrative:** *"ML research generates tons of disjointed runs. When you compare models, how do you know if an improvement is genuine or confounded by different datasets and configs?"*

1. **Screen A (Workspace):**
   - Show `project_demo` with 50 synthetic runs across multiple model families (XGBoost, MLP, CNN).
   - Filter by `model_family = gradient_boosting`.
   - Click `exp_002` (tuned XGBoost) and open the detail drawer.
   - **Point to Comparability Badge:** Show that `exp_002` is `COMPARABLE` to `exp_001` (same dataset, same holdout split, only `max_depth` changed from 4 to 6).

---

## Act II: Zero-Key Research Copilot & Budget Recommendations (1:00 – 2:00)
**Narrative:** *"Researchers need to know what to test next without exceeding compute budgets, and without hallucinated claims."*

1. **Screen B (Research Copilot):**
   - In chat, type:
     ```text
     What is the best performing baseline for fraud detection, and what should I test next under a 2 GPU-hour budget?
     ```
   - **Highlight Key Details:**
     - Runs **100% deterministically** with no external LLM key needed (`mode: deterministic`, `llm_used: false`).
     - Recommends tuning `learning_rate` or testing an ensemble within `1.5 GPU hours`.
     - Explicitly cites `exp_001` and `exp_002` as supporting baseline evidence.
   - Click **"Approve Proposal"** (shows FR-9 decision approval flow).

---

## Act III: The Magic Moment — Dataset Invalidation & Adaptive Lineage (2:00 – 3:00)
**Narrative:** *"Here is the defining behavior of RUNWISE. What happens when a researcher discovers a defect in a dataset version?"*

1. **Screen C (Lineage & Impact Screen):**
   - Click **"Report Dataset Defect"**:
     - **Source:** `tabular_fraud_v1`
     - **Reason:** `Data leakage discovered between train and test splits`
   - **Show Graph Invalidation Traversal:**
     - FalkorDB traverses multi-hop edges: `(tabular_fraud_v1)<-[:USES_DATASET_VERSION]-(exp_001)<-[:DERIVED_FROM]-(exp_002)`.
     - Both `exp_001` and `exp_002` turn amber/red as directly affected.
     - **Point to `exp_009` (v2):** Remains green/valid! RUNWISE does not blindly scrap clean research.
2. **Back to Screen B (Adaptive Recalibration):**
   - Re-run the recommendation query.
   - **Show the Difference:**
     - `exp_001` and `exp_002` are **strictly excluded** from citations.
     - The Copilot shifts recommendations to build upon the clean, unaffected baseline `exp_009`.
     - **Conclusion:** *"RUNWISE prevents researchers from building on flawed evidence, saving GPU hours and research integrity."*
