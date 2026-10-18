"""Datasets intake, profiling, and target confirmation router.

Defined in MEMBER_B.md Section 4 (B2 — Routers).
Implements F1: Static CSV intake, profiling, stratified split, and target confirmation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import uuid
from typing import Any, Optional
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel
import pandas as pd

from app.api.errors import not_found
from app.models.contracts import ColumnProfile, DatasetProfile, DroppedColumn

router = APIRouter(prefix="/datasets", tags=["datasets"])

# In-memory datasets store for fast lookup / offline mode
DATASET_STORE: dict[str, tuple[DatasetProfile, pd.DataFrame]] = {}


def profile_dataframe(df: pd.DataFrame, filename: str) -> DatasetProfile:
    dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
    n_rows, n_cols = df.shape

    columns: list[ColumnProfile] = []
    dropped: list[DroppedColumn] = []

    # Identify potential target
    target_candidate: Optional[str] = None
    for candidate in ["target", "label", "class", "churn", "fraud"]:
        if candidate in df.columns:
            target_candidate = candidate
            break
    if not target_candidate and len(df.columns) > 0:
        target_candidate = df.columns[-1]

    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        n_uniq = int(df[col].nunique(dropna=True))
        dtype_str = str(df[col].dtype)

        # Check dropped reasons
        if n_miss == n_rows:
            dropped.append(DroppedColumn(name=col, reason="all_null"))
            continue
        if n_uniq <= 1:
            dropped.append(DroppedColumn(name=col, reason="constant"))
            continue
        if n_uniq == n_rows and "object" in dtype_str:
            dropped.append(DroppedColumn(name=col, reason="high_cardinality_id"))
            continue

        role = "target" if col == target_candidate else "feature"
        columns.append(
            ColumnProfile(
                name=col,
                dtype=dtype_str,
                role=role,
                n_missing=n_miss,
                n_unique=n_uniq,
            )
        )

    # Class balance and task on target
    class_balance: dict[str, float] = {}
    task: str = "binary"
    n_classes: int = 2
    if target_candidate and target_candidate in df.columns:
        counts = df[target_candidate].value_counts(normalize=True)
        class_balance = {str(k): round(float(v), 4) for k, v in counts.items()}
        n_classes = len(counts)
        task = "binary" if n_classes <= 2 else "multiclass"

    # Compute split hash
    split_bytes = f"{dataset_id}:{n_rows}:{sorted(df.columns)}".encode("utf-8")
    split_hash = hashlib.sha256(split_bytes).hexdigest()[:16]

    profile = DatasetProfile(
        id=dataset_id,
        name=filename,
        n_rows=n_rows,
        n_cols=n_cols,
        task="binary" if task == "binary" else "multiclass",
        n_classes=n_classes,
        class_balance=class_balance,
        split_hash=split_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        target_inferred=True,
        columns=columns,
        dropped_columns=dropped,
    )
    return profile


@router.post(
    "",
    response_model=DatasetProfile,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and profile a static CSV dataset",
)
async def upload_dataset(file: UploadFile = File(...)) -> DatasetProfile:
    content = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=[{"code": "invalid_csv", "message": f"Could not parse CSV: {exc}", "field": "file"}],
        )

    filename = file.filename or "dataset.csv"
    profile = profile_dataframe(df, filename)
    DATASET_STORE[profile.id] = (profile, df)
    return profile


@router.get(
    "/{dataset_id}",
    response_model=DatasetProfile,
    summary="Get profile for a dataset",
)
async def get_dataset(dataset_id: str) -> DatasetProfile:
    if dataset_id not in DATASET_STORE:
        raise not_found("dataset", dataset_id)
    profile, _ = DATASET_STORE[dataset_id]
    return profile


class ConfirmTargetRequest(BaseModel):
    target: str


@router.post(
    "/{dataset_id}/confirm-target",
    response_model=DatasetProfile,
    summary="Confirm or change the target column for a dataset",
)
async def confirm_target(dataset_id: str, payload: ConfirmTargetRequest) -> DatasetProfile:
    if dataset_id not in DATASET_STORE:
        raise not_found("dataset", dataset_id)

    profile, df = DATASET_STORE[dataset_id]
    target_col = payload.target
    if target_col not in df.columns:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=[{"code": "invalid_target", "message": f"Column '{target_col}' not in dataset", "field": "target"}],
        )

    # Re-assign roles
    for col in profile.columns:
        col.role = "target" if col.name == target_col else "feature"

    counts = df[target_col].value_counts(normalize=True)
    profile.class_balance = {str(k): round(float(v), 4) for k, v in counts.items()}
    profile.n_classes = len(counts)
    profile.task = "binary" if profile.n_classes <= 2 else "multiclass"
    profile.target_inferred = False

    DATASET_STORE[dataset_id] = (profile, df)
    return profile
