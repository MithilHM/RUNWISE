"""Catalog router for available models and metrics.

Defined in MEMBER_B.md Section 4 (B2 — Routers).
Exposes the three supported models (XGBoost, LightGBM, Random Forest)
with complete ParamSpec spaces.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter
from pydantic import BaseModel

from app.models.contracts import ModelSpec, ParamSpec

router = APIRouter(prefix="/catalog", tags=["catalog"])

SUPPORTED_MODELS: list[ModelSpec] = [
    ModelSpec(
        id="xgboost",
        name="XGBoost",
        library="xgboost",
        description="Extreme Gradient Boosting with regularized tree ensembles.",
        params=[
            ParamSpec(name="n_estimators", kind="int", low=50, high=1000, step=10, log_scale=False, default=200),
            ParamSpec(name="max_depth", kind="int", low=1, high=16, step=1, log_scale=False, default=6),
            ParamSpec(name="learning_rate", kind="float", low=0.001, high=0.5, step=0.005, log_scale=True, default=0.05),
            ParamSpec(name="subsample", kind="float", low=0.5, high=1.0, step=0.05, log_scale=False, default=0.8),
            ParamSpec(name="colsample_bytree", kind="float", low=0.5, high=1.0, step=0.05, log_scale=False, default=0.8),
        ],
    ),
    ModelSpec(
        id="lightgbm",
        name="LightGBM",
        library="lightgbm",
        description="Fast histogram-based gradient boosting trees with leaf-wise growth.",
        params=[
            ParamSpec(name="n_estimators", kind="int", low=50, high=1000, step=10, log_scale=False, default=200),
            ParamSpec(name="num_leaves", kind="int", low=15, high=255, step=5, log_scale=False, default=31),
            ParamSpec(name="learning_rate", kind="float", low=0.001, high=0.5, step=0.005, log_scale=True, default=0.05),
            ParamSpec(name="subsample", kind="float", low=0.5, high=1.0, step=0.05, log_scale=False, default=0.8),
        ],
    ),
    ModelSpec(
        id="random_forest",
        name="Random Forest",
        library="sklearn.ensemble",
        description="Ensemble of bagging decision trees with randomized splits.",
        params=[
            ParamSpec(name="n_estimators", kind="int", low=10, high=500, step=10, log_scale=False, default=100),
            ParamSpec(name="max_depth", kind="int_or_null", low=1, high=32, step=1, log_scale=False, default=None),
            ParamSpec(name="min_samples_split", kind="int", low=2, high=20, step=1, log_scale=False, default=2),
            ParamSpec(name="bootstrap", kind="bool", choices=[True, False], default=True),
        ],
    ),
]

SUPPORTED_METRICS: list[str] = ["accuracy", "f1_macro", "roc_auc", "pr_auc"]


class CatalogResponse(BaseModel):
    models: list[ModelSpec]
    metrics: list[str]


@router.get(
    "/models",
    response_model=CatalogResponse,
    summary="List available models and evaluation metrics",
)
async def list_models() -> CatalogResponse:
    return CatalogResponse(models=SUPPORTED_MODELS, metrics=SUPPORTED_METRICS)
