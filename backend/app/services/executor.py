"""Executes one model on one configuration and measures it.

Two rules govern this module, and both exist to keep the graph honest:

1. **A configuration that fails validation never produces a metric.** It produces
   a `FAILED` trial with a reason and `metric_value=None`. A number from an
   invalid configuration is not a measurement, and recording it as one would put
   a fabrication in the graph next to real observations.
2. **A metric that was not computed is absent, not substituted.** If a run asks
   for `pr_auc` on a multiclass dataset, the other three metrics are reported and
   `pr_auc` is simply not there. Renaming `roc_auc` to `pr_auc` would return a
   plausible number that means something else.

Everything is seeded from `settings.RANDOM_SEED`, so re-running a configuration
returns the same numbers.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.core.config import settings
from app.models.contracts import BINARY_ONLY_METRICS, TrialOutcome, is_number
from app.services import catalog
from app.services.ingestion import DatasetIngestor, IngestionError, get_ingestor

logger = logging.getLogger(__name__)

__all__ = ["TrialTrainer", "get_trainer", "METRIC_LABELS"]

METRIC_LABELS: Dict[str, str] = {
    "accuracy": "Accuracy",
    "f1_macro": "F1 (macro)",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(results: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Turns any NaN or infinity into None.

    `roc_auc_score` returns NaN on a one-class fold instead of raising, so the
    metric dict has to be swept afterwards. A NaN in this graph is not a display
    problem: every comparison against it is False, so it would become the best
    result on its first comparison and then be impossible to displace.
    """
    return {
        key: (None if value is None or not is_number(value) else float(value))
        for key, value in results.items()
    }


class TrialTrainer:
    """Implements the `Trainer` port against real scikit-learn estimators."""

    def __init__(self, ingestor: Optional[DatasetIngestor] = None) -> None:
        self.ingestor = ingestor or get_ingestor()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, model_id: str, params: Dict[str, Any]) -> list[str]:
        """Problems with this configuration. Empty means it will train."""
        problems: list[str] = []
        try:
            resolved = catalog.coerce_params(model_id, params)
        except (KeyError, ValueError) as exc:
            return [str(exc)]
        problems.extend(catalog.validate_params(model_id, resolved))
        return problems

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        dataset_id: str,
        model_id: str,
        params: Dict[str, Any],
        metric: str = "accuracy",
        seed: Optional[int] = None,
    ) -> TrialOutcome:
        """Trains one model and returns the measured outcome."""
        seed = settings.RANDOM_SEED if seed is None else seed
        trial_id = f"trial_{uuid.uuid4().hex[:8]}"

        # Strict first: a configuration that arrived out of domain is a failed
        # trial, not something to silently clamp. Clamping belongs to the slider
        # path, where overshoot means "as extreme as available".
        raw_problems = catalog.validate_params(model_id, params or {}, strict=True)
        if raw_problems:
            return self._failed(
                trial_id, model_id, dict(params or {}), metric,
                "Invalid configuration: " + "; ".join(raw_problems), seed,
            )

        try:
            resolved = catalog.coerce_params(model_id, params)
        except (KeyError, ValueError) as exc:
            return self._failed(trial_id, model_id, params, metric, f"Invalid configuration: {exc}", seed)

        problems = catalog.validate_params(model_id, resolved)
        if problems:
            return self._failed(
                trial_id, model_id, resolved, metric, "Invalid configuration: " + "; ".join(problems), seed
            )

        try:
            features, target = self.ingestor.get_frame(dataset_id)
            profile = self.ingestor.get_profile(dataset_id)
        except IngestionError as exc:
            return self._failed(trial_id, model_id, resolved, metric, f"Dataset unavailable: {exc.message}", seed)

        # Reject an impossible metric before doing any work, and do not
        # substitute a different one.
        if metric not in catalog.METRICS:
            return self._failed(
                trial_id, model_id, resolved, metric, f"Unknown metric '{metric}'", seed
            )
        if metric in BINARY_ONLY_METRICS and profile.task != "binary":
            return self._failed(
                trial_id,
                model_id,
                resolved,
                metric,
                f"{metric} is only defined for binary targets; this dataset is "
                f"{profile.task}. Choose {', '.join(catalog.available_metrics(profile.task))}.",
                seed,
            )

        try:
            x_train, x_test, y_train, y_test = self._split(features, target, seed)
        except ValueError as exc:
            # Too few members in the rarest class for a stratified split. This is
            # a property of the data, not of the model, and it must not escape as
            # an exception or kill the run.
            return self._failed(
                trial_id, model_id, resolved, metric,
                f"Cannot split the dataset: {exc}", seed,
            )

        try:
            estimator = catalog.build_estimator(model_id, resolved, seed=seed)
        except (KeyError, ValueError, ImportError) as exc:
            return self._failed(trial_id, model_id, resolved, metric, f"Could not build model: {exc}", seed)

        started = time.perf_counter()

        try:
            estimator.fit(x_train, y_train)
            predictions = estimator.predict(x_test)
            metrics = self._metrics(y_test, predictions, self._probabilities(estimator, x_test))
        except Exception as exc:  # a library that raises mid-fit is a failed trial
            elapsed = round(time.perf_counter() - started, 4)
            logger.warning("Trial %s (%s) failed during training: %s", trial_id, model_id, exc)
            outcome = self._failed(trial_id, model_id, resolved, metric, f"Training error: {exc}", seed)
            outcome.duration_sec = elapsed
            return outcome

        duration = round(time.perf_counter() - started, 4)
        value = metrics.get(metric)

        # `roc_auc_score` returns NaN on a one-class fold rather than raising, and
        # NaN would sail through as a real measurement. If the metric the run
        # asked for is not a finite number, this trial produced no usable result.
        if not is_number(value):
            computed = ", ".join(k for k, v in metrics.items() if is_number(v)) or "none"
            return self._failed(
                trial_id, model_id, resolved, metric,
                f"The model trained, but {metric} could not be measured on this split "
                f"(a single-class validation fold, or an all-tie score). "
                f"Measured instead: {computed}. No substitute is reported.",
                seed,
            )

        logger.info(
            "Trial %s: %s on %s -> %s=%.4f in %.2fs",
            trial_id, catalog.get_model(model_id).name, profile.name, metric, value, duration,
        )
        return TrialOutcome(
            trial_id=trial_id,
            step=0,
            model_id=model_id,
            params=resolved,
            status="SUCCESS",
            metric_name=metric,
            metric_value=value,
            metrics=metrics,
            failure_reason=None,
            duration_sec=duration,
            n_features=int(features.shape[1]),
            seed=seed,
            created_at=_utcnow(),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _split(features, target, seed: int):
        """Stratified train/test split, seeded so it is reproducible."""
        from sklearn.model_selection import train_test_split

        stratify = target if target.nunique() >= 2 else None
        return train_test_split(
            features,
            target,
            test_size=settings.TRAIN_TEST_SPLIT,
            random_state=seed,
            stratify=stratify,
        )

    @staticmethod
    def _probabilities(estimator, x_test) -> Optional[np.ndarray]:
        """Positive-class scores, or None when the model has no such method."""
        if hasattr(estimator, "predict_proba"):
            try:
                proba = np.asarray(estimator.predict_proba(x_test))
                return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba
            except Exception:  # pragma: no cover - library dependent
                return None
        if hasattr(estimator, "decision_function"):
            try:
                scores = np.asarray(estimator.decision_function(x_test))
                return scores if scores.ndim == 1 else scores[:, -1]
            except Exception:  # pragma: no cover - library dependent
                return None
        return None

    @staticmethod
    def _metrics(y_true, y_pred, scores: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        """Computes every metric the target actually supports.

        A metric whose input is missing is reported as None, never as a stand-in.
        """
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            f1_score,
            roc_auc_score,
        )

        classes = sorted(pd_unique(y_true))
        is_binary = len(classes) == 2
        results: Dict[str, Optional[float]] = {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 6),
            "roc_auc": None,
            "pr_auc": None,
        }

        if scores is None:
            return _finite(results)

        try:
            if is_binary:
                results["roc_auc"] = round(float(roc_auc_score(y_true, scores)), 6)
            else:
                results["roc_auc"] = round(
                    float(roc_auc_score(y_true, scores, multi_class="ovr", average="weighted")), 6
                )
        except (ValueError, IndexError):
            # A one-class fold. Reported as absent, which is the truth.
            results["roc_auc"] = None

        if is_binary:
            try:
                results["pr_auc"] = round(float(average_precision_score(y_true, scores)), 6)
            except ValueError:  # pragma: no cover - needs a single positive class
                results["pr_auc"] = None
        # Multiclass PR-AUC is deliberately left absent rather than approximated.

        return _finite(results)

    @staticmethod
    def _failed(
        trial_id: str,
        model_id: str,
        params: Dict[str, Any],
        metric: str,
        reason: str,
        seed: int,
    ) -> TrialOutcome:
        return TrialOutcome(
            trial_id=trial_id,
            model_id=model_id,
            params=params,
            status="FAILED",
            metric_name=metric,
            metric_value=None,
            metrics={},
            failure_reason=reason,
            duration_sec=0.0,
            n_features=0,
            seed=seed,
            created_at=_utcnow(),
        )


def pd_unique(series) -> list:
    """Unique values as a sorted list, JSON-safe for mixed dtypes."""
    import pandas as pd

    return sorted(pd.unique(series.dropna()).tolist(), key=lambda v: (str(type(v)), str(v)))


_trainer: Optional[TrialTrainer] = None
_trainer_ingestor: Optional[DatasetIngestor] = None


def get_trainer(ingestor: Optional[DatasetIngestor] = None) -> TrialTrainer:
    """Process-wide trainer, sharing the ingestor's frame cache.

    Rebuilt when handed a different ingestor rather than quietly keeping the
    first one. Silently ignoring the argument means a caller that ingests into
    its own `DatasetIngestor` then calls `get_trainer(mine)` gets a trainer
    looking at a *different* cache: the dataset is missing, `get_frame` falls
    through to a disk reload, and the failure surfaces much later as a confusing
    error inside `get_frame` rather than as "unknown dataset".
    """
    global _trainer, _trainer_ingestor
    target = ingestor or get_ingestor()
    if _trainer is None or _trainer_ingestor is not target:
        _trainer = TrialTrainer(target)
        _trainer_ingestor = target
    return _trainer