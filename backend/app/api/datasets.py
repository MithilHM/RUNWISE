"""Datasets intake, profiling and target confirmation (F1).

A thin HTTP layer over `DatasetIngestor`. It does no profiling of its own:
target inference, column dropping, leakage detection, the seeded stratified
split and the data fingerprint all live in one place, and a second copy here
would silently disagree with the one the trainer uses — which is how a dataset
ends up uploaded successfully and then unavailable at training time.

Every accepted upload is written to the graph, so `GET /graph?dataset_id=...`
has something to show before any run exists.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.api.errors import not_found
from app.graph.repository import get_graph_store
from app.models.contracts import DatasetProfile
from app.services.ingestion import IngestionError, get_ingestor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["datasets"])


def _ingestion_error(exc: IngestionError) -> HTTPException:
    """Maps an intake failure onto the one error envelope.

    Every failure is 4xx: a file the harness cannot use is the caller's problem,
    not a server fault, and a 500 here would tell a researcher to retry something
    that will never succeed.
    """
    return HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[{"code": exc.code, "message": exc.message, "field": "file"}],
    )


def _write_to_graph(profile: DatasetProfile) -> None:
    """Persists the dataset. A graph failure must not fail the upload.

    The researcher already has a working dataset at this point; telling them the
    upload failed because the visualisation store is down would be wrong. The
    failure is logged and the run will surface it later.
    """
    try:
        get_graph_store().write_dataset(profile)
    except Exception as exc:  # noqa: BLE001
        logger.error("Uploaded dataset %s but could not write it to the graph: %s",
                     profile.id, exc)


@router.post(
    "",
    response_model=DatasetProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and profile a static CSV dataset",
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetProfile:
    """Accepts one static CSV and returns its profile."""
    content = await file.read()
    filename = file.filename or "dataset.csv"
    try:
        profile = get_ingestor().ingest(filename, content)
    except IngestionError as exc:
        raise _ingestion_error(exc) from exc

    _write_to_graph(profile)
    logger.info("Accepted %s as %s (%s, %d rows).",
                filename, profile.id, profile.task, profile.n_rows)
    return profile


@router.get(
    "/{dataset_id}",
    response_model=DatasetProfile,
    summary="Profile of an ingested dataset",
)
async def get_dataset(dataset_id: str) -> DatasetProfile:
    try:
        return get_ingestor().get_profile(dataset_id)
    except IngestionError as exc:
        if exc.code == "unknown_dataset":
            raise not_found("dataset", dataset_id) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"code": exc.code, "message": exc.message, "field": None}],
        ) from exc


class ConfirmTargetRequest(BaseModel):
    target: str


@router.post(
    "/{dataset_id}/confirm-target",
    response_model=DatasetProfile,
    summary="Confirm or change the target column",
)
async def confirm_target(dataset_id: str, payload: ConfirmTargetRequest) -> DatasetProfile:
    """Re-profiles against a user-chosen label column."""
    try:
        profile = get_ingestor().confirm_target(dataset_id, payload.target)
    except IngestionError as exc:
        if exc.code == "unknown_dataset":
            raise not_found("dataset", dataset_id) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"code": exc.code, "message": exc.message, "field": "target"}],
        ) from exc

    _write_to_graph(profile)
    return profile


@router.get(
    "/{dataset_id}/frame",
    summary="Preview the feature frame the trainer will actually use",
)
async def preview_frame(dataset_id: str, limit: int = 10) -> dict:
    """The post-cleanup, post-encoding matrix, so the user sees what is trained on.

    Returns columns and dtypes rather than values by default: a pre-parse preview
    of the raw file would let the user confirm a target column that intake then
    drops as leakage, which is the one thing the confirmation step exists to catch.
    """
    from app.services.ingestion import IngestionError as _IngestionError

    try:
        features, target = get_ingestor().get_frame(dataset_id)
    except _IngestionError as exc:
        if exc.code in ("unknown_dataset", "dataset_evicted"):
            raise not_found("dataset", dataset_id) from exc
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=[{"code": exc.code, "message": exc.message, "field": None}],
        ) from exc

    head = features.head(max(0, limit))
    return {
        "dataset_id": dataset_id,
        "n_rows": int(features.shape[0]),
        "n_features": int(features.shape[1]),
        "feature_names": [str(c) for c in features.columns],
        "feature_dtypes": {str(c): str(features[c].dtype) for c in features.columns},
        "target_name": str(target.name),
        "target_classes": sorted({str(v) for v in target.dropna().unique()}),
        "rows": head.to_dict(orient="records"),
    }