"""Sandboxed ML Worker for autonomous tabular model experiment execution.

Executes LightGBM / Scikit-Learn training trials non-blockingly using a bundled benchmark dataset
with early stopping, measuring PR-AUC, F1, LogLoss, memory consumption, and catching invalid
hyperparameter configurations.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import tracemalloc
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

from app.models.runtime_contracts import CandidateConfig, TrialResult

logger = logging.getLogger(__name__)


class SandboxedMLWorker:
    """Lightweight, non-blocking executor for tabular models on benchmark data regimes."""

    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path or self._resolve_default_dataset_path()
        self._X_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._load_and_split_data()

    def _resolve_default_dataset_path(self) -> str:
        """Finds bundled benchmark credit card fraud or churn dataset."""
        candidates = [
            Path("sample_data/benchmark_credit_card_fraud.csv"),
            Path("../sample_data/benchmark_credit_card_fraud.csv"),
            Path("sample_data/synthetic_experiments.csv"),
            Path("../sample_data/synthetic_experiments.csv"),
        ]
        for p in candidates:
            if p.exists():
                return str(p.resolve())

        # Fallback to generating in-memory or saving if path doesn't exist
        fallback_path = Path("sample_data/benchmark_credit_card_fraud.csv")
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        return str(fallback_path.resolve())

    def _load_and_split_data(self) -> None:
        """Loads and pre-splits benchmark tabular data 80/20 train/validation."""
        try:
            if os.path.exists(self.data_path) and os.path.getsize(self.data_path) > 0:
                df = pd.read_csv(self.data_path)
                if "Class" in df.columns:
                    X = df.drop(columns=["Class"]).values
                    y = df["Class"].values
                elif "target" in df.columns:
                    X = df.drop(columns=["target"]).values
                    y = df["target"].values
                else:
                    # Synthetic tabular features
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        X = df[numeric_cols[:-1]].values
                        y = (df[numeric_cols[-1]] > df[numeric_cols[-1]].median()).astype(int).values
                    else:
                        raise ValueError("No numeric columns available in dataset")
            else:
                X, y = self._generate_synthetic_benchmark()
        except Exception as e:
            logger.warning("Error loading %s (%s). Generating synthetic benchmark in-memory.", self.data_path, e)
            X, y = self._generate_synthetic_benchmark()

        # Stratified 80/20 split
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self._feature_names = [f"V{i+1}" for i in range(X.shape[1])]
        self._X_train = pd.DataFrame(X_tr, columns=self._feature_names)
        self._X_val = pd.DataFrame(X_va, columns=self._feature_names)
        self._y_train = y_tr
        self._y_val = y_va

    def _generate_synthetic_benchmark(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generates a realistic imbalanced fraud benchmark dataset (98:2 negative:positive ratio)."""
        rng = np.random.RandomState(42)
        n = 2000
        n_fraud = 40
        n_normal = n - n_fraud
        normal_feats = rng.randn(n_normal, 10)
        fraud_feats = rng.randn(n_fraud, 10) + 1.8
        X = np.vstack([normal_feats, fraud_feats])
        y = np.array([0] * n_normal + [1] * n_fraud)
        idx = rng.permutation(n)
        return X[idx], y[idx]

    def run_trial(
        self,
        candidate: CandidateConfig,
        target_metric: str = "val_pr_auc",
        model_family: str = "lightgbm",
        timeout_seconds: float = 30.0,
    ) -> TrialResult:
        """Synchronously executes a training trial for a candidate configuration."""
        trial_id = f"trial_{candidate.candidate_id}_{uuid.uuid4().hex[:6]}"
        hparams = dict(candidate.hyperparameters)

        # 1. Validate hyperparameters & catch simulated/real invalid combinations
        lr = hparams.get("learning_rate", 0.1)
        max_depth = hparams.get("max_depth", 4)
        n_est = hparams.get("n_estimators", 100)
        colsample = hparams.get("colsample_bytree", 1.0)
        scale_pos = hparams.get("scale_pos_weight", 1.0)

        if lr is not None and lr <= 0:
            return TrialResult(
                trial_id=trial_id,
                candidate_id=candidate.candidate_id,
                hyperparameters=hparams,
                metrics={target_metric: None, "val_loss": None, "train_time_sec": 0.001},
                status="FAILED",
                failure_reason=f"Invalid learning rate {lr}: must be positive",
                memory_mb=0.1,
            )

        if max_depth is not None and max_depth <= 0:
            return TrialResult(
                trial_id=trial_id,
                candidate_id=candidate.candidate_id,
                hyperparameters=hparams,
                metrics={target_metric: None, "val_loss": None, "train_time_sec": 0.001},
                status="FAILED",
                failure_reason=f"Invalid max_depth {max_depth}: must be >= 1",
                memory_mb=0.1,
            )

        if colsample is not None and (colsample <= 0 or colsample > 1.0):
            return TrialResult(
                trial_id=trial_id,
                candidate_id=candidate.candidate_id,
                hyperparameters=hparams,
                metrics={target_metric: None, "val_loss": None, "train_time_sec": 0.001},
                status="FAILED",
                failure_reason=f"Invalid colsample_bytree {colsample}: must be in (0, 1]",
                memory_mb=0.1,
            )

        # 2. Execute model training with memory & time tracking
        tracemalloc.start()
        start_time = time.perf_counter()

        try:
            model = self._build_model(model_family, hparams)
            model.fit(self._X_train, self._y_train)

            # Predict probabilities for imbalanced evaluation
            y_val_proba = model.predict_proba(self._X_val)[:, 1]
            y_val_pred = (y_val_proba >= 0.5).astype(int)

            # Metrics
            pr_auc = float(average_precision_score(self._y_val, y_val_proba))
            f1 = float(f1_score(self._y_val, y_val_pred, zero_division=0))
            loss = float(log_loss(self._y_val, y_val_proba))
            elapsed = round(time.perf_counter() - start_time, 4)

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            memory_mb = round(peak_mem / (1024 * 1024), 2)

            metrics: Dict[str, Optional[float]] = {
                "val_pr_auc": round(pr_auc, 4),
                "val_f1": round(f1, 4),
                "val_loss": round(loss, 4),
                "train_time_sec": elapsed,
            }
            if target_metric not in metrics:
                metrics[target_metric] = round(pr_auc, 4)

            return TrialResult(
                trial_id=trial_id,
                candidate_id=candidate.candidate_id,
                hyperparameters=hparams,
                metrics=metrics,
                status="SUCCESS",
                failure_reason=None,
                memory_mb=memory_mb,
            )

        except Exception as exc:
            elapsed = round(time.perf_counter() - start_time, 4)
            tracemalloc.stop()
            logger.error("Trial training failed: %s", exc)
            return TrialResult(
                trial_id=trial_id,
                candidate_id=candidate.candidate_id,
                hyperparameters=hparams,
                metrics={target_metric: None, "val_loss": None, "train_time_sec": elapsed},
                status="FAILED",
                failure_reason=f"Training error: {str(exc)}",
                memory_mb=1.0,
            )

    def _build_model(self, model_family: str, hparams: Dict[str, Any]) -> Any:
        """Instantiates LightGBM or Sklearn tabular classifier."""
        lr = float(hparams.get("learning_rate", 0.05))
        max_depth = int(hparams.get("max_depth", 4))
        n_est = int(hparams.get("n_estimators", 100))
        colsample = float(hparams.get("colsample_bytree", 1.0))
        scale_pos = float(hparams.get("scale_pos_weight", 1.0))

        if model_family == "lightgbm":
            try:
                import lightgbm as lgb
                return lgb.LGBMClassifier(
                    learning_rate=lr,
                    max_depth=max_depth,
                    n_estimators=n_est,
                    colsample_bytree=colsample,
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    verbosity=-1,
                    n_jobs=2,
                )
            except Exception as e:
                logger.warning("LightGBM unavailable (%s), falling back to HistGradientBoosting", e)

        # Fallback Scikit-Learn gradient boosting
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            learning_rate=lr,
            max_depth=max_depth,
            max_iter=min(n_est, 150),
            class_weight={0: 1.0, 1: scale_pos} if scale_pos != 1.0 else None,
            random_state=42,
        )

    async def execute_trial(
        self,
        candidate: CandidateConfig,
        target_metric: str = "val_pr_auc",
        model_family: str = "lightgbm",
        timeout_seconds: float = 30.0,
    ) -> TrialResult:
        """Non-blocking async execution offloading CPU-bound training to a worker thread."""
        return await asyncio.to_thread(
            self.run_trial,
            candidate,
            target_metric,
            model_family,
            timeout_seconds,
        )


# Global default instance
sandboxed_worker = SandboxedMLWorker()
