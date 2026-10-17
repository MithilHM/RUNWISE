"""
Experiment Ingestion Pipeline for RUNWISE.
Conforms strictly to RUNWISE PRD Section 5 (FR-1) and Section 18.

Features:
- Ingestion from JSON payloads and CSV files/strings/dataframes
- Flexible header normalization for familiar and unfamiliar ML logger outputs
- Preservation of arbitrary model-specific configuration parameters
- Deterministic canonical JSON and configuration hashing (SHA256)
- Structured validation with partial failure support
- Idempotent graph persistence (no duplicate nodes or edges)
- Comprehensive ImportSummary with row-level error reporting
"""

import io
import json
import hashlib
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd

from app.graph.repository import GraphRepository
from app.models.contracts import (
    ImportSummary,
    ImportErrorItem,
    MetricModel,
    ExperimentIngestRecord
)

logger = logging.getLogger(__name__)

# Flexible column name aliases for CSV mapping
COLUMN_ALIASES: Dict[str, List[str]] = {
    "id": ["id", "experiment_id", "exp_id", "run_id", "trial_id"],
    "project_id": ["project_id", "project", "proj_id"],
    "name": ["name", "experiment_name", "exp_name", "run_name", "title"],
    "model_family": ["model_family", "model", "architecture", "family", "model_type"],
    "status": ["status", "run_status", "state"],
    "dataset_name": ["dataset_name", "dataset", "data", "dataset_id"],
    "dataset_version": ["dataset_version", "version", "ds_version", "dataset_version_id", "data_version"],
    "duration_seconds": ["duration_seconds", "duration", "runtime_seconds", "runtime", "time_seconds"],
    "compute_cost_gpu_hours": ["compute_cost_gpu_hours", "gpu_hours", "compute_cost", "gpu_time_hours", "cost"],
    "random_seed": ["random_seed", "seed"],
    "parent_experiment_id": ["parent_experiment_id", "parent_id", "parent_exp_id", "parent", "parent_run_id"],
    "source_reference": ["source_reference", "source_file", "source_record", "provenance", "source"],
    "description": ["description", "desc", "objective"],
    "notes": ["notes", "researcher_notes", "comments"]
}

# Known metric naming hints and directions
METRIC_DIRECTIONS: Dict[str, str] = {
    "loss": "minimize",
    "val_loss": "minimize",
    "validation_loss": "minimize",
    "test_loss": "minimize",
    "train_loss": "minimize",
    "rmse": "minimize",
    "mae": "minimize",
    "error_rate": "minimize",
    "accuracy": "maximize",
    "val_accuracy": "maximize",
    "validation_accuracy": "maximize",
    "test_accuracy": "maximize",
    "f1": "maximize",
    "f1_score": "maximize",
    "val_f1": "maximize",
    "auc": "maximize",
    "roc_auc": "maximize",
    "precision": "maximize",
    "recall": "maximize",
    "bleu": "maximize",
    "rouge": "maximize"
}


def compute_canonical_config(config_dict: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Computes a canonical JSON representation and SHA256 hash for a configuration dictionary.
    Returns: (config_id, canonical_json, config_hash)
    """
    sorted_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(sorted_json.encode("utf-8")).hexdigest()
    config_id = f"cfg_{config_hash[:12]}"
    return config_id, sorted_json, config_hash


class IngestionService:
    def __init__(self, repository: Optional[GraphRepository] = None):
        self.repository = repository or GraphRepository()

    def _resolve_aliases(self, row_dict: Dict[str, Any], default_project_id: str) -> Dict[str, Any]:
        """Maps varying input keys to normalized field names using aliases."""
        normalized: Dict[str, Any] = {}
        row_keys_lower = {str(k).strip().lower(): k for k in row_dict.keys()}

        for canonical_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in row_keys_lower:
                    orig_key = row_keys_lower[alias]
                    val = row_dict[orig_key]
                    if pd.notna(val) and val != "":
                        normalized[canonical_name] = val
                    break

        if "project_id" not in normalized or not normalized["project_id"]:
            normalized["project_id"] = default_project_id

        return normalized

    def parse_record(
        self,
        raw_record: Dict[str, Any],
        row_num: Optional[int],
        default_project_id: str
    ) -> Tuple[Optional[ExperimentIngestRecord], Optional[ImportErrorItem]]:
        """Validates and normalizes an experiment record."""
        # Normalize keys
        fields = self._resolve_aliases(raw_record, default_project_id)

        exp_id = fields.get("id")
        if not exp_id:
            return None, ImportErrorItem(row=row_num, reason="Missing required experiment identifier ('id' or 'experiment_id')")

        exp_id = str(exp_id).strip()
        project_id = str(fields.get("project_id", default_project_id)).strip()
        name = str(fields.get("name", exp_id)).strip()
        model_family = str(fields.get("model_family", "generic")).strip()

        # Parse dataset information
        dataset_name = fields.get("dataset_name")
        dataset_version = fields.get("dataset_version")
        dataset_version_id = None
        if dataset_name:
            d_name = str(dataset_name).strip()
            v_tag = str(dataset_version).strip() if dataset_version else "v1"
            dataset_version_id = f"{d_name}_{v_tag}".replace(" ", "_").lower()
        elif dataset_version:
            dataset_version_id = str(dataset_version).strip()
            dataset_name = dataset_version_id.split("_")[0]

        # Extract metrics
        metrics: List[MetricModel] = []

        # Case 1: Explicit 'metrics' field (list or dict)
        raw_metrics = raw_record.get("metrics")
        if isinstance(raw_metrics, list):
            for m in raw_metrics:
                if isinstance(m, dict) and "name" in m and "value" in m:
                    try:
                        val = float(m["value"])
                        dir_hint = m.get("direction") or METRIC_DIRECTIONS.get(m["name"].lower(), "maximize")
                        metrics.append(MetricModel(
                            name=str(m["name"]).strip(),
                            value=val,
                            direction=dir_hint,
                            evaluation_split=m.get("evaluation_split", "validation"),
                            evaluation_protocol=m.get("evaluation_protocol")
                        ))
                    except (ValueError, TypeError):
                        pass
        elif isinstance(raw_metrics, dict):
            for m_name, m_val in raw_metrics.items():
                try:
                    val = float(m_val)
                    dir_hint = METRIC_DIRECTIONS.get(m_name.lower(), "maximize")
                    metrics.append(MetricModel(
                        name=str(m_name).strip(),
                        value=val,
                        direction=dir_hint,
                        evaluation_split="validation"
                    ))
                except (ValueError, TypeError):
                    pass

        # Case 2: In-line columns for metrics (e.g. in CSV tables)
        # Skip top-level and alias-handled fields
        reserved_keys = {
            "metrics", "configuration", "hyperparameters", "parameters",
            "id", "project_id", "name", "model_family", "status",
            "dataset_name", "dataset_version", "duration_seconds",
            "compute_cost_gpu_hours", "random_seed", "parent_experiment_id",
            "source_reference", "description", "notes"
        }
        for aliases in COLUMN_ALIASES.values():
            reserved_keys.update(aliases)

        config_params: Dict[str, Any] = {}
        for k, v in raw_record.items():
            k_clean = str(k).strip()
            k_lower = k_clean.lower()

            if v is None or v == "":
                continue
            if not isinstance(v, (list, dict)):
                try:
                    if pd.isna(v):
                        continue
                except Exception:
                    pass

            if k_lower in reserved_keys:
                continue

            # Check if this column is a metric
            is_metric_col = False
            for m_prefix in METRIC_DIRECTIONS:
                if m_prefix in k_lower or k_lower in ("accuracy", "acc", "val_acc", "loss", "f1", "val_loss"):
                    try:
                        val = float(v)
                        # Avoid duplicating if already present
                        if not any(existing.name == k_clean for existing in metrics):
                            metrics.append(MetricModel(
                                name=k_clean,
                                value=val,
                                direction=METRIC_DIRECTIONS.get(k_lower, "maximize"),
                                evaluation_split="validation" if "val" in k_lower else "test" if "test" in k_lower else "train"
                            ))
                        is_metric_col = True
                        break
                    except (ValueError, TypeError):
                        pass

            if not is_metric_col:
                # If not recognized as a metric or reserved key, preserve as config/extra parameter
                config_params[k_clean] = v

        # Configuration & Hyperparameters
        if isinstance(raw_record.get("configuration"), dict):
            config_params.update(raw_record["configuration"])
        if isinstance(raw_record.get("hyperparameters"), dict):
            config_params.update(raw_record["hyperparameters"])
        if isinstance(raw_record.get("parameters"), dict):
            config_params.update(raw_record["parameters"])

        # Numbers conversion
        duration_s = None
        if fields.get("duration_seconds") is not None:
            try:
                duration_s = float(fields["duration_seconds"])
            except (ValueError, TypeError):
                pass

        gpu_cost = None
        if fields.get("compute_cost_gpu_hours") is not None:
            try:
                gpu_cost = float(fields["compute_cost_gpu_hours"])
            except (ValueError, TypeError):
                pass

        seed = None
        if fields.get("random_seed") is not None:
            try:
                seed = int(fields["random_seed"])
            except (ValueError, TypeError):
                pass

        record = ExperimentIngestRecord(
            id=exp_id,
            project_id=project_id,
            name=name,
            model_family=model_family,
            status=str(fields.get("status", "completed")),
            description=fields.get("description"),
            dataset_name=str(dataset_name) if dataset_name else None,
            dataset_version=str(dataset_version) if dataset_version else "v1",
            dataset_version_id=dataset_version_id,
            configuration=config_params,
            metrics=metrics,
            duration_seconds=duration_s,
            compute_cost_gpu_hours=gpu_cost,
            random_seed=seed,
            source_reference=fields.get("source_reference"),
            notes=fields.get("notes"),
            parent_experiment_id=fields.get("parent_experiment_id"),
            created_at=fields.get("created_at")
        )
        return record, None

    def ingest_records(
        self,
        records: List[Dict[str, Any]],
        project_id: str,
        project_name: Optional[str] = None
    ) -> ImportSummary:
        """
        Ingests a list of raw record dictionaries into the FalkorDB graph.
        Ensures idempotency, partial failure handling, and returns an ImportSummary.
        """
        summary = ImportSummary()

        # Ensure project node exists
        self.repository.create_or_update_project(
            project_id=project_id,
            name=project_name or project_id,
            description="Project initialized by RUNWISE experiment ingestion"
        )

        valid_records: List[ExperimentIngestRecord] = []

        # Phase 1: Parse and validate all records
        for idx, raw_record in enumerate(records, start=1):
            if not isinstance(raw_record, dict):
                summary.failed += 1
                summary.errors.append(ImportErrorItem(row=idx, reason="Record must be a JSON/dict object"))
                continue

            parsed, error = self.parse_record(raw_record, row_num=idx, default_project_id=project_id)
            if error:
                summary.failed += 1
                summary.errors.append(error)
            elif parsed:
                valid_records.append(parsed)

        # Phase 2: Persist valid records to FalkorDB
        for rec in valid_records:
            try:
                # 1. Dataset & DatasetVersion
                if rec.dataset_name and rec.dataset_version_id:
                    d_id = rec.dataset_name.replace(" ", "_").lower()
                    self.repository.create_or_update_dataset(
                        dataset_id=d_id,
                        name=rec.dataset_name,
                        description=f"Dataset for {rec.dataset_name}"
                    )
                    self.repository.create_or_update_dataset_version(
                        version_id=rec.dataset_version_id,
                        dataset_id=d_id,
                        version=rec.dataset_version or "v1",
                        quality_status="valid"
                    )

                # 2. Configuration
                config_id, canonical_json, config_hash = compute_canonical_config(rec.configuration)
                self.repository.create_or_update_configuration(
                    config_id=config_id,
                    canonical_json=canonical_json,
                    config_hash=config_hash
                )

                # 3. Experiment
                exp_dict = {
                    "id": rec.id,
                    "project_id": rec.project_id,
                    "name": rec.name,
                    "model_family": rec.model_family,
                    "status": rec.status,
                    "quality_status": "valid",
                    "description": rec.description,
                    "config_id": config_id,
                    "dataset_version_id": rec.dataset_version_id,
                    "parent_experiment_id": rec.parent_experiment_id,
                    "duration_seconds": rec.duration_seconds,
                    "compute_cost_gpu_hours": rec.compute_cost_gpu_hours,
                    "random_seed": rec.random_seed,
                    "source_reference": rec.source_reference,
                    "notes": rec.notes,
                    "created_at": rec.created_at
                }
                self.repository.create_or_update_experiment(exp_dict)

                # 4. Metrics
                for metric in rec.metrics:
                    m_id = f"metric_{rec.id}_{metric.name}".replace(" ", "_").lower()
                    self.repository.create_or_update_metric(
                        metric_id=m_id,
                        experiment_id=rec.id,
                        name=metric.name,
                        value=metric.value,
                        direction=metric.direction,
                        evaluation_split=metric.evaluation_split,
                        evaluation_protocol=metric.evaluation_protocol
                    )

                summary.imported += 1

            except Exception as e:
                logger.error(f"Failed to persist experiment {rec.id}: {e}")
                summary.failed += 1
                summary.errors.append(ImportErrorItem(
                    experiment_id=rec.id,
                    reason=f"Database persistence failure: {str(e)}"
                ))

        return summary

    def ingest_csv(
        self,
        csv_content: Union[str, bytes, io.StringIO, io.BytesIO],
        project_id: str,
        project_name: Optional[str] = None
    ) -> ImportSummary:
        """Parses CSV content into records and ingests them."""
        try:
            if isinstance(csv_content, (str, bytes)):
                buffer = io.BytesIO(csv_content) if isinstance(csv_content, bytes) else io.StringIO(csv_content)
            else:
                buffer = csv_content

            df = pd.read_csv(buffer)
            # Replace NaN with None
            records = df.where(pd.notna(df), None).to_dict(orient="records")
            return self.ingest_records(records, project_id=project_id, project_name=project_name)
        except Exception as e:
            return ImportSummary(
                imported=0,
                failed=1,
                errors=[ImportErrorItem(row=0, reason=f"Failed to parse CSV file: {str(e)}")]
            )

    def ingest_json(
        self,
        json_data: Union[str, bytes, List[Dict[str, Any]], Dict[str, Any]],
        project_id: str,
        project_name: Optional[str] = None
    ) -> ImportSummary:
        """Parses JSON content into records and ingests them."""
        try:
            if isinstance(json_data, (str, bytes)):
                parsed = json.loads(json_data)
            else:
                parsed = json_data

            if isinstance(parsed, dict):
                # If wrapped in {"experiments": [...]}, extract
                if "experiments" in parsed and isinstance(parsed["experiments"], list):
                    records = parsed["experiments"]
                else:
                    records = [parsed]
            elif isinstance(parsed, list):
                records = parsed
            else:
                return ImportSummary(
                    imported=0,
                    failed=1,
                    errors=[ImportErrorItem(row=0, reason="JSON root must be an object or array")]
                )

            return self.ingest_records(records, project_id=project_id, project_name=project_name)
        except Exception as e:
            return ImportSummary(
                imported=0,
                failed=1,
                errors=[ImportErrorItem(row=0, reason=f"Failed to parse JSON content: {str(e)}")]
            )
