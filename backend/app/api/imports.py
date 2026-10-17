"""
Import API endpoints for RUNWISE.
Conforms to PRD Section 9 (FR-1):
- POST /api/v1/projects/{project_id}/experiments/import
Supports JSON payload and multipart CSV file upload.
"""

import json
from typing import Optional, List, Dict, Any, Union
from fastapi import APIRouter, Request, Depends, HTTPException

from app.services.ingestion import IngestionService
from app.models.contracts import ImportSummary

router = APIRouter(tags=["imports"])


def get_ingestion_service() -> IngestionService:
    return IngestionService()


@router.post(
    "/projects/{project_id}/experiments/import",
    response_model=ImportSummary,
    summary="Import experiments from JSON or CSV",
    description="Imports and validates experiment runs, linking them to datasets, configurations, and parent lineage in FalkorDB."
)
async def import_experiments(
    project_id: str,
    request: Request,
    service: IngestionService = Depends(get_ingestion_service)
) -> ImportSummary:
    """
    Ingest experiments for a project. Accepts either:
    1. Direct JSON body (array of experiments or {"experiments": [...]})
    2. Uploaded CSV or JSON file via multipart/form-data
    """
    content_type = request.headers.get("content-type", "").lower()

    if "multipart/form-data" in content_type:
        form = await request.form()
        uploaded_file = form.get("file")
        if uploaded_file and hasattr(uploaded_file, "read"):
            content = await uploaded_file.read()
            filename = (getattr(uploaded_file, "filename", "") or "").lower()
            if filename.endswith(".csv"):
                return service.ingest_csv(content, project_id=project_id)
            elif filename.endswith(".json"):
                return service.ingest_json(content, project_id=project_id)
            else:
                try:
                    summary = service.ingest_csv(content, project_id=project_id)
                    if summary.imported > 0:
                        return summary
                except Exception:
                    pass
                return service.ingest_json(content, project_id=project_id)
        raise HTTPException(status_code=400, detail="Form submission missing 'file' field.")

    # Try parsing body as JSON
    try:
        body = await request.json()
        if body is not None:
            return service.ingest_json(body, project_id=project_id)
    except Exception as e:
        # If raw text / CSV in body
        try:
            raw_text = (await request.body()).decode("utf-8")
            if "," in raw_text and "\n" in raw_text:
                return service.ingest_csv(raw_text, project_id=project_id)
        except Exception:
            pass
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payload format: {str(e)}"
        )

    raise HTTPException(
        status_code=400,
        detail="Must provide either a JSON body or an uploaded CSV/JSON file."
    )
