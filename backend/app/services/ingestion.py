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

Trust rules implemented here (PRD 5 FR-1 #8, ranking_rules.md section 4):

* A value the source never stated is stored as ``None``. It is never replaced
  by a plausible-looking default (``"generic"``, ``"completed"``, ``"v1"``,
  ``0.0``, ``"validation"``, ``"maximize"``). The only exceptions are the
  documented name table in :data:`METRIC_DIRECTIONS` and the human labels the
  researcher explicitly typed, neither of which is written to the graph as an
  observation when the source said nothing.
* A blank measurement is unrecorded, not zero. It is reported with its row
  number and no Metric node is written; a number nobody measured never reaches
  the graph.
* Anything that cannot be understood is reported (FR-1 #4) instead of being
  swallowed in a ``pass``. Only a record without an experiment identifier, a
  record that conflicts with the target project, or a record that cannot be
  represented at all is a hard failure; a bad metric is reported and the rest
  of the record is still imported.
* A row may never write outside the project named in the request. A row that
  declares a different ``project_id`` is rejected and reported.
* Re-importing an unchanged record is a no-op counted as ``skipped``
  (FR-1 #5); two rows in one batch may not claim the same id.

Two fields are passed to the graph as ``None`` even though
``ExperimentIngestRecord`` cannot express that today (``model_family`` is a
required ``str`` and ``status`` defaults to ``"completed"``). They are assigned
after construction so the graph never receives a fabricated value; see
:func:`IngestionService._parse_record`.
"""

import io
import json
import hashlib
import logging
import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from app.graph.repository import GraphRepository
from app.models.contracts import (
    ImportSummary,
    ImportErrorItem,
    MetricModel,
    ExperimentIngestRecord
)

logger = logging.getLogger(__name__)

# Flexible column name aliases for CSV mapping.
#
# Node ids and human labels are deliberately kept in separate namespaces
# (HANDOFF.md lines 115-120, ranking_rules.md section 1, docs/GRAPH_SCHEMA.md
# section 2): `dataset_id` / `dataset_version_id` identify nodes,
# `dataset_name` / `dataset_version` are the labels a researcher reads. They are
# never compared against each other, so an id column may not populate a label.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "id": ["id", "experiment_id", "exp_id", "run_id", "trial_id"],
    "project_id": ["project_id", "project", "proj_id"],
    "name": ["name", "experiment_name", "exp_name", "run_name", "title"],
    "model_family": ["model_family", "model", "architecture", "family", "model_type"],
    "status": ["status", "run_status", "state"],
    # Node identifiers.
    "dataset_id": ["dataset_id", "dataset_node_id"],
    "dataset_version_id": ["dataset_version_id", "version_id"],
    # Human labels.
    "dataset_name": ["dataset_name", "dataset", "data"],
    "dataset_version": ["dataset_version", "ds_version", "version", "data_version"],
    "duration_seconds": ["duration_seconds", "duration", "runtime_seconds", "runtime", "time_seconds"],
    "compute_cost_gpu_hours": ["compute_cost_gpu_hours", "gpu_hours", "compute_cost", "gpu_time_hours", "cost"],
    "random_seed": ["random_seed", "seed"],
    "parent_experiment_id": ["parent_experiment_id", "parent_id", "parent_exp_id", "parent", "parent_run_id"],
    "source_reference": ["source_reference", "source_file", "source_record", "provenance", "source"],
    "description": ["description", "desc", "objective"],
    "notes": ["notes", "researcher_notes", "comments"],
    # Provenance (FR-1 #6): without this the source timestamp is lost and the
    # repository substitutes the import time, which is a fabricated
    # observation date.
    "created_at": ["created_at", "timestamp", "start_time", "run_time", "logged_at"],
}

# Known metric naming hints and directions.
#
# This table is a *normalisation of well-known metric names*, not a default: a
# name in the table yields the direction that name universally means, and a name
# outside it yields ``None`` so the direction stays unrecorded instead of being
# invented. It is also the list of column names treated as metrics in a CSV.
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
    "rouge": "maximize",
}

METRIC_NAME_HINTS = frozenset(METRIC_DIRECTIONS) | {
    "accuracy", "acc", "val_acc", "loss", "f1", "val_loss"
}

# Configuration containers that are merged into the configuration rather than
# kept as separate columns.
CONFIG_BLOCK_KEYS = ("configuration", "hyperparameters", "parameters")

# Sentinel for "this attribute is not present at all", which is different from
# "this attribute is present and is None".
_MISSING = object()


def compute_canonical_config(config_dict: Dict[str, Any]) -> Tuple[str, str, str]:
    """
    Computes a canonical JSON representation and SHA256 hash for a configuration dictionary.
    Returns: (config_id, canonical_json, config_hash)
    """
    sorted_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), default=str)
    config_hash = hashlib.sha256(sorted_json.encode("utf-8")).hexdigest()
    config_id = f"cfg_{config_hash[:12]}"
    return config_id, sorted_json, config_hash


def _is_blank(value: Any) -> bool:
    """True when a source cell carries no observation (None, empty text, NaN)."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, tuple, set, dict, bytes, bytearray)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _as_text(value: Any) -> Optional[str]:
    """Stripped source text, or ``None`` when the source stated nothing."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace").strip() or None
    if _is_blank(value):
        return None
    return str(value).strip() or None


def _as_float(value: Any) -> Optional[float]:
    """The number a source cell states, or ``None`` when it states none."""
    if _is_blank(value) or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> Optional[int]:
    """The integer a source cell states, or ``None`` when it states none."""
    number = _as_float(value)
    if number is None or math.isnan(number) or math.isinf(number):
        return None
    try:
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _describe(value: Any) -> str:
    """Short, safe rendering of a rejected source value for an error message."""
    if _is_blank(value):
        return "blank"
    text = str(value).strip()
    if len(text) > 40:
        text = text[:37] + "..."
    return repr(text)


def _contract_reason(exc: Exception) -> str:
    """Readable one-line reason for a contract rejection.

    A pydantic error carries the offending field and message; using them keeps
    the report useful instead of "Validation error for Model[model_family=...]".
    """
    errors = getattr(exc, "errors", None)
    if callable(errors):
        try:
            details = errors()
        except Exception:  # pragma: no cover - defensive
            details = None
        if details:
            first = details[0]
            location = ".".join(str(part) for part in first.get("loc", ()) or ())
            message = str(first.get("msg", "")) or _describe(first)
            return f"{location}: {message}" if location else message
    return _describe(exc)


def _slug(value: str) -> str:
    return str(value).strip().replace(" ", "_").lower()


def _plain(value: Any) -> Any:
    """A read value as a plain value: enums answer with their value."""
    return getattr(value, "value", value)


def _to_datetime(value: Any) -> Optional[datetime]:
    """Parse a timestamp that may arrive as text or as a datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        text = value.strip().replace("Z", "+00:00").replace("z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None
    return None


def _same_instant(stored: Any, incoming: Any) -> bool:
    """Do two timestamps denote the same moment, whatever shape they arrive in?

    Only used to decide whether a write would change anything; a timezone-less
    source timestamp is read as UTC for the comparison and stored verbatim.
    """
    left = _to_datetime(stored)
    right = _to_datetime(incoming)
    if left is None or right is None:
        return str(stored) == str(incoming)
    if (left.tzinfo is None) != (right.tzinfo is None):
        left = left.replace(tzinfo=None)
        right = right.replace(tzinfo=None)
    return left == right


def _stored_field(existing: Any, name: str) -> Any:
    """A field of the stored record, or :data:`_MISSING` when it is not exposed."""
    return getattr(existing, name, _MISSING)


def _stored_matches(existing: Any, name: str, incoming: Optional[str]) -> bool:
    """Would writing ``incoming`` change the stored field ``name``?

    An absent incoming value changes nothing. A stored field the read model does
    not expose is treated as a change, because skipping a real update would lose
    it.
    """
    if incoming is None:
        return True
    stored = _stored_field(existing, name)
    if stored is _MISSING:
        return False
    return _plain(stored) == incoming


def _looks_like_metric_column(column: str) -> bool:
    return any(hint in column for hint in METRIC_NAME_HINTS)


def _metric_direction_for_write(metric: MetricModel) -> Optional[str]:
    """The direction to hand to the repository, which may be ``None``.

    ``MetricValue`` normalises a missing direction to ``maximize`` (a documented
    contract behaviour). Ingestion must not be the thing that decided that, so
    the direction the source actually stated is carried alongside and it is the
    stated value -- or ``None`` -- that reaches the graph.
    """
    stated = getattr(metric, "source_direction", _MISSING)
    if stated is _MISSING:
        # The contract dropped the extra field; fall back to the model value.
        return metric.direction
    return stated


class IngestionService:
    def __init__(self, repository: Optional[GraphRepository] = None):
        self.repository = repository or GraphRepository()

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _resolve_aliases(
        self,
        row_dict: Dict[str, Any],
        default_project_id: str
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Maps varying input keys to normalized field names using aliases.

        Returns ``(normalized, row_project_id)``. The row's own ``project_id``
        is reported separately rather than merged in: a record may not choose
        which project it is written to.
        """
        normalized: Dict[str, Any] = {}
        row_keys_lower = {str(k).strip().lower(): k for k in row_dict.keys()}

        for canonical_name, aliases in COLUMN_ALIASES.items():
            for alias in aliases:
                if alias not in row_keys_lower:
                    continue
                val = row_dict[row_keys_lower[alias]]
                if not _is_blank(val):
                    normalized[canonical_name] = val
                    break
                # A blank value still means the source uses this spelling, but a
                # more specific alias may carry the value.

        row_project_id = _as_text(normalized.get("project_id"))
        return normalized, row_project_id

    @staticmethod
    def _reserved_column_keys() -> set:
        reserved = {"metrics", *CONFIG_BLOCK_KEYS}
        for aliases in COLUMN_ALIASES.values():
            reserved.update(aliases)
        return reserved

    def _make_metric(
        self,
        *,
        name: Any,
        value: Any,
        source_direction: Any,
        evaluation_split: Any,
        evaluation_protocol: Any,
        row_num: Optional[int],
        errors: List[ImportErrorItem],
    ) -> Optional[MetricModel]:
        """Build one metric, reporting whatever cannot be understood.

        A metric whose value the source never recorded is returned with
        ``value=None`` and an ``ImportErrorItem``: the record keeps the shape of
        the measurement, the graph never gains a number nobody measured, and the
        gap is visible to the caller.
        """
        metric_name = _as_text(name)
        if not metric_name:
            errors.append(ImportErrorItem(
                row=row_num,
                reason=f"Metric without a name ({_describe(name)}) was omitted"
            ))
            return None

        number = _as_float(value)
        if number is None:
            errors.append(ImportErrorItem(
                row=row_num,
                reason=(
                    f"Metric '{metric_name}' has no usable numeric value "
                    f"({_describe(value)}); recorded as unrecorded, never as 0.0"
                )
            ))

        stated_direction = _as_text(source_direction)
        if stated_direction is None:
            # Documented name table, not a default: an unrecognised metric name
            # simply leaves the direction unrecorded.
            stated_direction = METRIC_DIRECTIONS.get(metric_name.lower())

        try:
            metric = MetricModel(
                name=metric_name,
                value=number,
                direction=stated_direction,
                evaluation_split=_as_text(evaluation_split),
                evaluation_protocol=_as_text(evaluation_protocol),
            )
        except Exception as exc:  # unknown direction spelling, and friends
            errors.append(ImportErrorItem(
                row=row_num,
                reason=f"Metric '{metric_name}' was omitted: {_contract_reason(exc)}"
            ))
            return None

        # Keep the direction the source stated (or did not state) for the write.
        metric.source_direction = stated_direction
        return metric

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_record(
        self,
        raw_record: Dict[str, Any],
        row_num: Optional[int],
        default_project_id: str
    ) -> Tuple[Optional[ExperimentIngestRecord], Optional[ImportErrorItem]]:
        """Validates and normalizes an experiment record.

        Returns ``(record, error)``. When the record is returned, ``error`` is
        ``None`` even if the row carried problems: those are row-level warnings
        surfaced through :meth:`ingest_records` with :meth:`_parse_record`.
        """
        record, errors = self._parse_record(raw_record, row_num, default_project_id)
        if record is None:
            return None, (errors[0] if errors else ImportErrorItem(
                row=row_num, reason="Record was rejected"
            ))
        return record, None

    def _parse_record(
        self,
        raw_record: Dict[str, Any],
        row_num: Optional[int],
        default_project_id: str
    ) -> Tuple[Optional[ExperimentIngestRecord], List[ImportErrorItem]]:
        """Returns ``(record, errors)``.

        ``record`` is ``None`` only for a rejected row, in which case every
        error explains the rejection. When the record survives, the errors are
        row-level problems (a malformed metric, an unparsable number) and the
        record is still imported without the offending piece.
        """
        errors: List[ImportErrorItem] = []
        fields, row_project_id = self._resolve_aliases(raw_record, default_project_id)

        exp_id = _as_text(fields.get("id"))
        if not exp_id:
            errors.append(ImportErrorItem(
                row=row_num,
                reason="Missing required experiment identifier ('id' or 'experiment_id')"
            ))
            return None, errors

        if row_project_id is not None and row_project_id != default_project_id:
            errors.append(ImportErrorItem(
                row=row_num,
                experiment_id=exp_id,
                reason=(
                    f"Row declares project_id '{row_project_id}' but this import targets "
                    f"'{default_project_id}'; a row may not be written into another project"
                )
            ))
            return None, errors

        name = _as_text(fields.get("name")) or exp_id
        model_family = _as_text(fields.get("model_family"))
        status = _as_text(fields.get("status"))

        # Datasets: node ids and labels stay apart.
        dataset_label = _as_text(fields.get("dataset_name"))
        dataset_node_id = _as_text(fields.get("dataset_id"))
        dataset_version_label = _as_text(fields.get("dataset_version"))
        dataset_version_node_id = _as_text(fields.get("dataset_version_id"))

        if dataset_node_id is None and dataset_label:
            dataset_node_id = _slug(dataset_label)

        if dataset_version_node_id is None and dataset_node_id and dataset_version_label:
            # A version node id may only be derived when the source supplied
            # both halves of it. Inventing "v1" would fabricate a snapshot.
            dataset_version_node_id = (
                f"{dataset_node_id}_{dataset_version_label}".replace(" ", "_").lower()
            )

        # -- metrics ----------------------------------------------------
        metrics: List[MetricModel] = []
        raw_metrics = raw_record.get("metrics")

        if isinstance(raw_metrics, list):
            for entry in raw_metrics:
                if not isinstance(entry, dict):
                    errors.append(ImportErrorItem(
                        row=row_num,
                        reason=f"Metric entry {_describe(entry)} is not an object; omitted"
                    ))
                    continue
                if "name" not in entry or "value" not in entry:
                    errors.append(ImportErrorItem(
                        row=row_num,
                        reason=(
                            f"Metric entry {_describe(entry)} needs both 'name' and 'value'; omitted"
                        )
                    ))
                    continue
                metric = self._make_metric(
                    name=entry["name"],
                    value=entry["value"],
                    source_direction=entry.get("direction"),
                    evaluation_split=entry.get("evaluation_split"),
                    evaluation_protocol=entry.get("evaluation_protocol"),
                    row_num=row_num,
                    errors=errors,
                )
                if metric is not None:
                    metrics.append(metric)
        elif isinstance(raw_metrics, dict):
            for metric_name, metric_value in raw_metrics.items():
                metric = self._make_metric(
                    name=metric_name,
                    value=metric_value,
                    source_direction=None,
                    evaluation_split=None,
                    evaluation_protocol=None,
                    row_num=row_num,
                    errors=errors,
                )
                if metric is not None:
                    metrics.append(metric)
        elif raw_metrics is not None:
            errors.append(ImportErrorItem(
                row=row_num,
                reason=(
                    f"'metrics' must be a list or an object, got "
                    f"{type(raw_metrics).__name__}; no metrics were imported from it"
                )
            ))

        # -- in-line columns -------------------------------------------
        reserved_keys = self._reserved_column_keys()
        config_params: Dict[str, Any] = {}

        for key, value in raw_record.items():
            key_clean = str(key).strip()
            key_lower = key_clean.lower()

            if key_lower in reserved_keys:
                continue

            if _looks_like_metric_column(key_lower) and not isinstance(value, (list, dict)):
                metric = self._make_metric(
                    name=key_clean,
                    value=value,
                    source_direction=None,
                    # A column name says nothing about the split the number came
                    # from, so the split stays unrecorded unless stated.
                    evaluation_split=None,
                    evaluation_protocol=None,
                    row_num=row_num,
                    errors=errors,
                )
                if metric is not None and not any(m.name == metric.name for m in metrics):
                    metrics.append(metric)
                continue

            if _is_blank(value):
                continue

            # Not a recognised metric and not a known field: preserve it, FR-1 #3.
            config_params[key_clean] = value

        for block_key in CONFIG_BLOCK_KEYS:
            block = raw_record.get(block_key)
            if isinstance(block, dict):
                config_params.update(block)
            elif block is not None:
                errors.append(ImportErrorItem(
                    row=row_num,
                    reason=f"'{block_key}' must be a JSON object; it was not merged into the configuration"
                ))

        # -- numeric fields --------------------------------------------
        duration_s = self._number_field(
            fields, "duration_seconds", row_num, exp_id, errors, _as_float
        )
        gpu_cost = self._number_field(
            fields, "compute_cost_gpu_hours", row_num, exp_id, errors, _as_float
        )
        seed = self._number_field(fields, "random_seed", row_num, exp_id, errors, _as_int)

        created_at = _as_text(fields.get("created_at"))

        try:
            record = ExperimentIngestRecord(
                id=exp_id,
                project_id=default_project_id,
                name=name,
                # Placeholders only; the absent values are assigned below.
                model_family=model_family if model_family is not None else "",
                status=status if status is not None else "",
                description=_as_text(fields.get("description")),
                dataset_name=dataset_label,
                dataset_version=dataset_version_label,
                dataset_version_id=dataset_version_node_id,
                configuration=config_params,
                metrics=metrics,
                duration_seconds=duration_s,
                compute_cost_gpu_hours=gpu_cost,
                random_seed=seed,
                source_reference=_as_text(fields.get("source_reference")),
                notes=_as_text(fields.get("notes")),
                parent_experiment_id=_as_text(fields.get("parent_experiment_id")),
                created_at=created_at,
                extra_fields={"dataset_id": dataset_node_id} if dataset_node_id else {},
            )
        except Exception as exc:
            errors.append(ImportErrorItem(
                row=row_num,
                experiment_id=exp_id,
                reason=f"Record does not satisfy the ingest contract: {_contract_reason(exc)}"
            ))
            return None, errors

        # ExperimentIngestRecord declares ``model_family`` as a required str and
        # defaults ``status`` to "completed", so a missing value cannot be
        # passed through the constructor. Assigning afterwards keeps it absent
        # on the way to the graph instead of shipping a fabricated fact.
        # Ask the contract owner to make both ``str | None = None``.
        record.model_family = model_family
        record.status = status

        return record, errors

    @staticmethod
    def _number_field(
        fields: Dict[str, Any],
        key: str,
        row_num: Optional[int],
        exp_id: str,
        errors: List[ImportErrorItem],
        parse,
    ) -> Optional[Any]:
        """Reads a numeric field, reporting a value that is not a number."""
        raw_value = fields.get(key)
        if _is_blank(raw_value):
            return None
        number = parse(raw_value)
        if number is None or (isinstance(number, float) and math.isnan(number)):
            errors.append(ImportErrorItem(
                row=row_num,
                experiment_id=exp_id,
                reason=(
                    f"'{key}' value {_describe(raw_value)} is not a number; "
                    "stored as unrecorded"
                )
            ))
            return None
        return number

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist(self, record: ExperimentIngestRecord) -> None:
        """Writes one record. Never invents a value the source did not state."""
        dataset_node_id = (record.extra_fields or {}).get("dataset_id")

        if dataset_node_id:
            # A node id is enough to write the Dataset node; when the source
            # gave no label, none is invented and the name stays unset.
            # No description either: nobody supplied one.
            self.repository.create_or_update_dataset(
                dataset_id=dataset_node_id,
                name=record.dataset_name,
                description=None,
            )

        if record.dataset_version_id:
            if dataset_node_id:
                self.repository.create_or_update_dataset_version(
                    version_id=record.dataset_version_id,
                    dataset_id=dataset_node_id,
                    version=record.dataset_version,
                    quality_status="valid",
                )
            else:
                logger.warning(
                    "DatasetVersion %s has no dataset node id in the source record; "
                    "no DatasetVersion node was written for experiment %s",
                    record.dataset_version_id, record.id,
                )

        config_id, canonical_json, config_hash = compute_canonical_config(record.configuration or {})
        self.repository.create_or_update_configuration(
            config_id=config_id,
            canonical_json=canonical_json,
            config_hash=config_hash,
        )

        exp_dict = {
            "id": record.id,
            "project_id": record.project_id,
            "name": record.name,
            # None means "the source did not say", and the graph stores no value.
            "model_family": record.model_family,
            "status": record.status,
            "quality_status": "valid",
            "description": record.description,
            "config_id": config_id,
            "dataset_version_id": record.dataset_version_id,
            "parent_experiment_id": record.parent_experiment_id,
            "duration_seconds": record.duration_seconds,
            "compute_cost_gpu_hours": record.compute_cost_gpu_hours,
            "random_seed": record.random_seed,
            "source_reference": record.source_reference,
            "notes": record.notes,
            "created_at": record.created_at,
        }
        self.repository.create_or_update_experiment(exp_dict)

        for metric in record.metrics:
            if metric.value is None:
                # A blank is unrecorded. Writing 0.0 here would invent a result.
                logger.info(
                    "Metric '%s' of experiment %s has no recorded value; "
                    "no Metric node was written", metric.name, record.id,
                )
                continue
            metric_id = f"metric_{record.id}_{metric.name}".replace(" ", "_").lower()
            self.repository.create_or_update_metric(
                metric_id=metric_id,
                experiment_id=record.id,
                name=metric.name,
                value=metric.value,
                direction=_metric_direction_for_write(metric),
                evaluation_split=metric.evaluation_split,
                evaluation_protocol=metric.evaluation_protocol,
            )

    def _read_stored_record(self, record_id: str):
        """Reads a stored experiment, or ``None`` when it cannot be read.

        ``get_experiment`` is the port method; ``get_experiment_detail`` is
        accepted as an alias so a renamed graph read helper degrades to "cannot
        tell" (import) instead of raising.
        """
        for reader_name in ("get_experiment", "get_experiment_detail"):
            reader = getattr(self.repository, reader_name, None)
            if reader is None:
                continue
            try:
                return reader(record_id)
            except Exception as exc:
                logger.warning(
                    "Could not read stored experiment %s (%s); importing it",
                    record_id, exc, exc_info=True,
                )
                return None
        logger.warning(
            "The graph repository exposes no experiment read method; importing %s",
            record_id,
        )
        return None

    def _already_stored_unchanged(self, record: ExperimentIngestRecord) -> bool:
        """True when writing this record would not change the graph (FR-1 #5).

        Only the fields this importer actually writes are compared, and a field
        the source left out is treated as "nothing to change" rather than as a
        difference -- that is what makes a re-import idempotent.
        """
        existing = self._read_stored_record(record.id)
        if existing is None:
            return False

        try:
            return _matches_stored_record(record, existing)
        except Exception as exc:  # never let a comparison break the import
            logger.warning(
                "Could not compare experiment %s with its stored copy (%s); importing it",
                record.id, exc, exc_info=True,
            )
            return False

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
        target_project = str(project_id).strip()

        # Create the project only when it does not exist yet. An import into an
        # existing project must not rename it or replace a description somebody
        # wrote, so the fallback name and description are not passed as updates.
        if self.repository.get_project(target_project) is None:
            self.repository.create_or_update_project(
                project_id=target_project,
                name=project_name or target_project,
                description=(
                    project_name
                    and f"Created by import of {project_name}"
                    or None
                ),
            )

        valid_records: List[Tuple[int, ExperimentIngestRecord]] = []
        batch_ids: Dict[str, int] = {}

        # Phase 1: Parse and validate all records
        for idx, raw_record in enumerate(records, start=1):
            if not isinstance(raw_record, dict):
                summary.failed += 1
                summary.errors.append(ImportErrorItem(
                    row=idx, reason="Record must be a JSON/dict object"
                ))
                continue

            try:
                parsed, errors = self._parse_record(
                    raw_record, row_num=idx, default_project_id=target_project
                )
            except Exception as exc:  # one hostile row may not sink the batch
                logger.error(f"Could not parse row {idx}: {exc}", exc_info=True)
                summary.failed += 1
                summary.errors.append(ImportErrorItem(
                    row=idx,
                    reason=f"Record could not be parsed: {_contract_reason(exc)}",
                ))
                continue

            if parsed is None:
                summary.failed += 1
                summary.errors.extend(errors or [ImportErrorItem(
                    row=idx, reason="Record was rejected"
                )])
                continue

            # Row-level problems do not stop the record from being imported.
            summary.errors.extend(errors)

            first_row = batch_ids.get(parsed.id)
            if first_row is not None:
                summary.skipped += 1
                summary.errors.append(ImportErrorItem(
                    row=idx,
                    experiment_id=parsed.id,
                    reason=(
                        f"Duplicate experiment id '{parsed.id}' already imported from row "
                        f"{first_row} of this batch; the later row was skipped"
                    )
                ))
                continue

            batch_ids[parsed.id] = idx
            valid_records.append((idx, parsed))

        # Phase 2: Persist valid records to FalkorDB
        for idx, rec in valid_records:
            if self._already_stored_unchanged(rec):
                summary.skipped += 1
                summary.errors.append(ImportErrorItem(
                    row=idx,
                    experiment_id=rec.id,
                    reason=(
                        f"Experiment '{rec.id}' already exists in project '{rec.project_id}' "
                        "with identical values; skipped as an unchanged re-import"
                    )
                ))
                continue

            try:
                self._persist(rec)
                summary.imported += 1
            except Exception as e:
                logger.error(f"Failed to persist experiment {rec.id}: {e}")
                summary.failed += 1
                summary.errors.append(ImportErrorItem(
                    row=idx,
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


def _matches_stored_record(record: ExperimentIngestRecord, existing) -> bool:
    """Would persisting ``record`` change anything already in the graph?

    Compared field by field, and only where the importer actually writes:
    a field the source left out never overwrites a stored value, so it can
    never make the record "changed".
    """
    if _plain(existing.project_id) != _plain(record.project_id):
        return False
    if _plain(existing.name) != record.name:
        return False

    # Written on every write, so a stated value must match.
    if not _stored_matches(existing, "model_family", record.model_family):
        return False
    if not _stored_matches(existing, "status", record.status):
        return False

    # Coalesced on match: a stated value must match, an absent one changes nothing.
    if not _stored_matches(existing, "description", record.description):
        return False
    if not _stored_matches(existing, "notes", record.notes):
        return False
    if not _stored_matches(existing, "source_reference", record.source_reference):
        return False
    if not _stored_matches(existing, "dataset_version_id", record.dataset_version_id):
        return False
    for field, incoming in (
        ("duration_seconds", record.duration_seconds),
        ("compute_cost_gpu_hours", record.compute_cost_gpu_hours),
    ):
        if incoming is None:
            continue
        stored = _stored_field(existing, field)
        if stored is _MISSING or _as_float(_plain(stored)) != incoming:
            return False
    if record.random_seed is not None:
        stored = _stored_field(existing, "random_seed")
        if stored is _MISSING or _as_int(_plain(stored)) != record.random_seed:
            return False

    if record.parent_experiment_id:
        parents = _stored_field(existing, "parent_experiment_ids")
        if parents is not _MISSING and record.parent_experiment_id not in (parents or []):
            return False

    if record.created_at is not None:
        stored_created_at = _stored_field(existing, "created_at")
        if stored_created_at is _MISSING or not _same_instant(stored_created_at, record.created_at):
            return False

    stored_config = _stored_field(existing, "configuration")
    if stored_config is _MISSING:
        if record.configuration:
            return False
        stored_config = None
    if not isinstance(stored_config, dict):
        stored_config = {} if stored_config is None else {"raw": stored_config}
    if compute_canonical_config(record.configuration or {})[1] != compute_canonical_config(stored_config)[1]:
        return False

    stored_metrics = {
        m.name: m for m in (_stored_field(existing, "metrics") or [])
    }
    for metric in record.metrics:
        if metric.value is None:
            # Never written, so it cannot make the stored record stale.
            continue
        stored_metric = stored_metrics.get(metric.name)
        if stored_metric is None:
            return False
        if _as_float(_plain(stored_metric.value)) != metric.value:
            return False
        if metric.evaluation_split is not None and _plain(stored_metric.evaluation_split) != metric.evaluation_split:
            return False
        if metric.evaluation_protocol is not None and (_plain(stored_metric.evaluation_protocol) or "") != metric.evaluation_protocol:
            return False
        direction = _metric_direction_for_write(metric)
        if direction is not None and _plain(stored_metric.direction) != direction:
            return False

    return True