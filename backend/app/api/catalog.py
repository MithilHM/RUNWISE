"""Catalog router for available models and metrics (F2).

Serves `services.catalog` directly. It previously kept its own hard-coded copy of
the three models, and the two copies had already drifted: XGBoost `max_depth`
topped out at 16 here and 12 there, `learning_rate` started at 0.001 and 0.01,
`n_estimators` stepped by 10 and 50. A slider built from this list would have let
a researcher pick a value the trainer then rejected as out of domain, and the
`Param` nodes in the graph would have disagreed with both.

One definition, three consumers: this endpoint, the `HyperparameterPanel`, and
the graph.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from app.models.contracts import BINARY_ONLY_METRICS, ModelSpec
from app.services import catalog

router = APIRouter(prefix="/catalog", tags=["catalog"])


class MetricHelp(BaseModel):
    name: str
    help: str
    binary_only: bool = False


class CatalogResponse(BaseModel):
    models: list[ModelSpec]
    metrics: list[str]
    metric_help: list[MetricHelp]


@router.get(
    "/models",
    response_model=CatalogResponse,
    summary="List available models, their full parameter spaces, and the metrics",
)
async def list_models() -> CatalogResponse:
    return CatalogResponse(
        models=catalog.list_models(),
        metrics=list(catalog.METRICS),
        metric_help=[
            MetricHelp(name=name, help=help, binary_only=name in BINARY_ONLY_METRICS)
            for name, help in catalog.METRIC_HELP.items()
        ],
    )