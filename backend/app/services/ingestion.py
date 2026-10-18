"""Static CSV intake: one file in, one profiled dataset out.

Responsibilities, in order:

1. reject a file that is not usable tabular data, loudly and specifically;
2. decide which column is the target, and say that it was inferred;
3. drop columns that cannot help (constants, nulls, identifiers, target leaks) and
   record *why* each one went, so the drop is visible rather than silent;
4. stratify and seed the split, and fingerprint it, so that two trials are
   comparable or neither is;
5. cache the frame in memory and on disk under `settings.DATA_DIR`.

The graph write is not done here. `ingest` returns a profile; persisting it is the
graph layer's job, which keeps this module testable with no database.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.core.config import settings
from app.models.contracts import ColumnProfile, DatasetProfile, DroppedColumn
from app.services.catalog import MAX_CLASSES, MIN_CLASS_MEMBERS

logger = logging.getLogger(__name__)

__all__ = ["IngestionError", "DatasetIngestor", "get_ingestor"]


class IngestionError(ValueError):
    """Raised when an uploaded file cannot become a usable dataset."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# Names that are a strong signal for the target column. Checked before any
# cardinality heuristic so a `target` column with many classes is still found.
_TARGET_NAME_HINTS = ("target", "label", "class", "outcome", "response", "y", "is_")

# Above this fraction of distinct values a column is treated as an identifier
# rather than a feature, as long as there are enough rows for that to mean
# something.
_ID_CARDINALITY_RATIO = 0.98
_ID_MIN_ROWS = 200

# A feature with at most this many distinct values is checked for leakage. High
# enough to catch a many-valued leak column, low enough that the per-value group
# check stays cheap.
_LEAKAGE_MAX_CARDINALITY = 200

# Rows required per distinct feature value before a one-to-one mapping counts as
# leakage rather than coincidence.
_LEAK_MIN_ROWS_PER_VALUE = 5


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return cleaned[:48] or "dataset"


class DatasetIngestor:
    """Owns the in-memory frame cache and the on-disk CSV store."""

    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = Path(data_dir or settings.DATA_DIR)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._frames: Dict[str, pd.DataFrame] = {}
        self._profiles: Dict[str, DatasetProfile] = {}
        self._paths: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest(
        self, filename: str, content: bytes, target_hint: Optional[str] = None
    ) -> DatasetProfile:
        """Parses, validates, profiles and caches one CSV."""
        if not content:
            raise IngestionError("empty_file", "The uploaded file is empty.")
        if len(content) > settings.MAX_UPLOAD_BYTES:
            raise IngestionError(
                "file_too_large",
                f"The file is larger than the {settings.MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
            )

        frame = self._parse(filename, content)

        if len(frame) > settings.MAX_UPLOAD_ROWS:
            raise IngestionError(
                "too_many_rows",
                f"The file has {len(frame):,} rows, over the {settings.MAX_UPLOAD_ROWS:,} limit.",
            )

        dataset_id = f"ds_{uuid.uuid4().hex[:8]}"
        profile = self._profile_frame(dataset_id, filename, frame, target_hint)

        stored = self._persist(dataset_id, filename, frame)
        self._frames[dataset_id] = frame
        self._profiles[dataset_id] = profile
        self._paths[dataset_id] = stored

        logger.info(
            "Ingested %s as %s: %d rows x %d cols, target=%s, task=%s",
            filename, dataset_id, profile.n_rows, profile.n_cols, profile.target, profile.task,
        )
        return profile

    def _parse(self, filename: str, content: bytes) -> pd.DataFrame:
        """Reads bytes as CSV, turning every failure mode into a specific code."""
        for encoding in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                text = content.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:  # pragma: no cover - latin-1 decodes anything
            text = content.decode("utf-8", errors="replace")

        if not text.strip():
            raise IngestionError("empty_file", "The uploaded file has no rows.")

        self._reject_ragged_rows(text)

        try:
            # `on_bad_lines="error"` is passed for defence in depth. pandas 3.0
            # accepts ragged rows even with it, which is why the structural check
            # above exists: a short row means the file is not the rectangular
            # table the models require, and silently padding it hides a data
            # problem behind a number that looks like a measurement.
            frame = pd.read_csv(io.StringIO(text), on_bad_lines="error")
        except pd.errors.EmptyDataError as exc:
            raise IngestionError("empty_file", "The uploaded file has no header row.") from exc
        except pd.errors.ParserError as exc:
            raise IngestionError(
                "unparseable_csv", f"Row lengths do not match the header: {exc}"
            ) from exc
        except UnicodeDecodeError as exc:  # pragma: no cover - covered above
            raise IngestionError("unparseable_csv", f"The file is not valid text: {exc}") from exc

        frame.columns = [str(c).strip() for c in frame.columns]
        if frame.shape[1] == 0:
            raise IngestionError("unparseable_csv", "The file has a header but no columns.")
        if frame.shape[0] == 0:
            raise IngestionError("empty_file", "The file has a header but no data rows.")

        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        if duplicates:
            raise IngestionError(
                "duplicate_columns",
                f"Duplicate column name(s): {', '.join(sorted(set(duplicates)))}.",
            )
        return frame

    @staticmethod
    def _reject_ragged_rows(text: str, tolerance: int = 1) -> None:
        """Rejects a file whose rows do not all match the header width.

        Checked with the stdlib CSV reader rather than pandas, because this is the
        one structural property the models depend on and the one pandas is
        willing to paper over. Only the first few bad lines are reported: the
        message is for a person, and listing ten thousand of them helps nobody.
        """
        reader = csv.reader(io.StringIO(text))
        try:
            header = next(reader)
        except StopIteration:
            raise IngestionError("empty_file", "The uploaded file has no header row.") from None
        except csv.Error as exc:
            raise IngestionError("unparseable_csv", f"The header row could not be parsed: {exc}") from exc

        expected = len(header)
        problems: List[int] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != expected:
                problems.append(line_number)
                if len(problems) >= tolerance + 3:
                    break

        if not problems:
            return
        shown = ", ".join(str(n) for n in problems[:tolerance])
        extra = "" if len(problems) <= tolerance else f" (and {len(problems) - tolerance} more)"
        raise IngestionError(
            "unparseable_csv",
            f"The header declares {expected} column(s) but {len(problems)} row(s) "
            f"do not: line {shown}{extra}.",
        )

    def _persist(self, dataset_id: str, filename: str, frame: pd.DataFrame) -> str:
        """Writes the frame next to the run data. Best effort.

        A read-only disk must not stop a researcher from running a trial, so a
        failure here is logged and the in-memory cache carries the run.
        """
        safe = f"{dataset_id}_{_slug(Path(filename).stem)}.csv"
        destination = self.data_dir / safe
        try:
            frame.to_csv(destination, index=False)
        except OSError as exc:
            logger.warning("Could not persist %s to %s: %s", dataset_id, destination, exc)
            return ""
        return str(destination)

    # ------------------------------------------------------------------
    # Profiling
    # ------------------------------------------------------------------

    def _profile_frame(
        self,
        dataset_id: str,
        name: str,
        frame: pd.DataFrame,
        target_hint: Optional[str] = None,
    ) -> DatasetProfile:
        """Builds the profile: target choice, dropped columns, task, split hash."""
        target = self._resolve_target(frame, target_hint)
        if target is None:
            raise IngestionError(
                "no_target_column",
                "Could not identify a target column. Upload a CSV whose last column "
                "is the label, or confirm one explicitly.",
            )

        y = frame[target]
        if y.isna().all():
            raise IngestionError(
                "empty_target", f"Target column '{target}' is entirely empty."
            )
        classes = sorted(pd.unique(y.dropna()).tolist(), key=lambda v: (str(type(v)), str(v)))
        if len(classes) < 2:
            raise IngestionError(
                "single_class_target",
                f"Target column '{target}' has one distinct value ({classes[0]!r}). "
                "There is nothing to classify.",
            )

        # A target is a label, not a measurement. Without this check the
        # structural fallback happily selects a continuous column, the profile
        # claims "multiclass, 200 classes", and every subsequent trial dies in
        # train_test_split with "the least populated class has only 1 member".
        if len(classes) > MAX_CLASSES:
            raise IngestionError(
                "target_not_a_label",
                f"Column '{target}' has {len(classes)} distinct values across "
                f"{len(frame):,} rows. That is a continuous variable, not a class "
                f"label; this harness handles classification with at most "
                f"{MAX_CLASSES} classes. Choose the label column explicitly.",
            )
        thinnest = int(y.value_counts().min())
        if thinnest < MIN_CLASS_MEMBERS:
            raise IngestionError(
                "class_too_small",
                f"Class '{classes[0] if thinnest < 2 else y.value_counts().idxmin()!r}' in "
                f"'{target}' has only {thinnest} member(s). Every class needs at least "
                f"{MIN_CLASS_MEMBERS} for a train/test split.",
            )

        dropped: List[DroppedColumn] = []
        features = frame.drop(columns=[target])
        features = self._drop_unusable(features, y, dropped)

        if features.shape[1] == 0:
            raise IngestionError(
                "no_features",
                "Every column was dropped as unusable. There are no features left to train on.",
            )

        counts = y.value_counts(normalize=True)
        class_balance = {str(k): round(float(v), 6) for k, v in counts.items()}

        columns = [
            ColumnProfile(
                name=target,
                dtype=str(y.dtype),
                role="target",
                n_missing=int(y.isna().sum()),
                n_unique=int(y.nunique(dropna=True)),
            )
        ]
        columns.extend(
            ColumnProfile(
                name=str(col),
                dtype=str(features[col].dtype),
                role="feature",
                n_missing=int(features[col].isna().sum()),
                n_unique=int(features[col].nunique(dropna=True)),
            )
            for col in features.columns
        )

        profile = DatasetProfile(
            id=dataset_id,
            name=name,
            n_rows=int(frame.shape[0]),
            n_cols=int(len(columns)),
            task="binary" if len(classes) == 2 else "multiclass",
            n_classes=len(classes),
            class_balance=class_balance,
            created_at=_utcnow(),
            target=target,
            target_inferred=target_hint is None,
            columns=columns,
            dropped_columns=dropped,
        )
        profile.split_hash = self._split_hash(profile, frame)
        return profile

    def _resolve_target(
        self, frame: pd.DataFrame, target_hint: Optional[str]
    ) -> Optional[str]:
        """Chooses the target column. An explicit hint always wins."""
        if target_hint:
            if target_hint not in frame.columns:
                raise IngestionError(
                    "unknown_target",
                    f"'{target_hint}' is not a column in this file. "
                    f"Available: {', '.join(map(str, frame.columns))}.",
                )
            return target_hint

        lowered = {str(c).strip().lower(): str(c) for c in frame.columns}
        # Exact name matches only. Prefix matching on "y" and "is_" hijacked real
        # feature columns: `columns = [age, is_fraud, churn]` resolved the target
        # to `is_fraud`, and `class_size` won over `churn` on the "class" hint.
        for hint in _TARGET_NAME_HINTS:
            for low, original in lowered.items():
                if low == hint:
                    return original

        # No name matched, so fall back to structure: the lowest-cardinality
        # column that is not obviously an identifier.
        best: Optional[str] = None
        best_cardinality = -1
        for col in frame.columns:
            cardinality = int(frame[col].nunique(dropna=True))
            if cardinality < 2:
                continue
            if (
                len(frame) >= _ID_MIN_ROWS
                and cardinality / max(len(frame), 1) > _ID_CARDINALITY_RATIO
            ):
                continue
            if best is None or cardinality < best_cardinality:
                best, best_cardinality = str(col), cardinality

        if best is not None:
            return best
        # Every column is constant or identifier-like. The last column is the
        # conventional position for a label, so take it rather than give up.
        return str(frame.columns[-1])

    def _drop_unusable(
        self, features: pd.DataFrame, y: pd.Series, dropped: List[DroppedColumn]
    ) -> pd.DataFrame:
        """Removes columns that cannot contribute, recording each reason."""
        keep: List[str] = []

        for col in features.columns:
            series = features[col]
            if series.isna().all():
                dropped.append(DroppedColumn(name=str(col), reason="all_null"))
                continue
            non_null = series.dropna()
            if non_null.nunique() <= 1:
                dropped.append(DroppedColumn(name=str(col), reason="constant"))
                continue
            if self._looks_like_identifier(series):
                dropped.append(DroppedColumn(name=str(col), reason="high_cardinality_id"))
                continue
            if self._is_leaking(series, y):
                dropped.append(DroppedColumn(name=str(col), reason="leakage"))
                continue
            keep.append(str(col))

        return features[keep]

    @staticmethod
    def _looks_like_identifier(series: pd.Series) -> bool:
        """Whether this column is a row identifier rather than a feature.

        Only integer-dtype columns are eligible. A float column whose values are
        all distinct is almost certainly a continuous measurement, and dropping it
        would discard the signal on any dataset whose features are measured to
        more precision than its rows — which is most of them.

        The monotonicity test is applied to the column **in its original order**.
        Sorting first and testing the result is vacuous: any set of distinct
        values becomes strictly increasing once sorted, so every near-unique
        column would be classified as an identifier.

        The consequence of a false positive here is a real feature silently
        deleted and recorded under a confidently wrong reason, so the test is
        deliberately narrow: high cardinality *and* integer *and* already ordered.
        """
        n_rows = len(series)
        if n_rows < _ID_MIN_ROWS:
            return False
        non_null = series.dropna()
        cardinality = int(non_null.nunique())
        if cardinality / max(n_rows, 1) <= _ID_CARDINALITY_RATIO:
            return False

        if pd.api.types.is_integer_dtype(series):
            values = non_null.to_numpy()
        elif pd.api.types.is_float_dtype(series):
            # Whole-number floats are treated as integers; genuinely continuous
            # columns are left alone.
            sample = non_null.to_numpy()[:500]
            if not np.all(np.equal(np.mod(sample, 1), 0)):
                return False
            values = non_null.to_numpy()
        else:
            labels = non_null.astype(str).tolist()
            if any(not label.lstrip("-").isdigit() for label in labels[:50]):
                return False
            values = np.array([int(label) for label in labels])

        if len(values) < 2:
            return False
        # Original order, not sorted order. This is the whole test.
        deltas = np.diff(values)
        return bool(np.all(deltas > 0) or np.all(deltas < 0))

    @staticmethod
    def _is_leaking(series: pd.Series, y: pd.Series) -> bool:
        """Whether this feature determines the target on its own.

        The test is: does every distinct value of this feature correspond to
        exactly one class? A column of `pos` / `neg_a` / `neg_b` labels does, even
        though it has three distinct values against a binary target — so an
        earlier version that required `cardinality == n_classes` missed real leaks
        of exactly this shape.

        A one-to-one feature is leakage in effect even when the column is
        legitimate, because using it gives a perfect validation score that will
        not survive on unseen data. It is reported with its reason rather than
        dropped silently, so a researcher can overrule it.
        """
        cardinality = int(series.nunique(dropna=True))
        if cardinality < 2 or cardinality > _LEAKAGE_MAX_CARDINALITY:
            return False

        # On a small frame a one-to-one mapping is what chance produces. Six rows
        # with six distinct feature values maps one-to-one onto almost any label,
        # so without this guard the rule deletes every feature of every toy
        # dataset. Requiring several rows per distinct value keeps the test
        # meaningful: a real leak column is populated across the whole dataset,
        # not once per value.
        if len(series) < _LEAK_MIN_ROWS_PER_VALUE * cardinality:
            return False

        grouped = pd.DataFrame({"feature": series, "target": y}).dropna()
        if grouped.empty:
            return False
        classes_per_value = grouped.groupby("feature", observed=True)["target"].nunique()
        return bool((classes_per_value == 1).all())

    def _split_hash(self, profile: DatasetProfile, frame: pd.DataFrame) -> str:
        """Fingerprints the split so two trials can be shown to be comparable.

        Covers the row count, the confirmed target, the feature set, the class
        distribution **and the actual cell values**. An earlier version hashed
        only the metadata, so two entirely different datasets with the same
        column names and class balance produced the same hash — which defeats
        the one job this field has, because it is what tells a reader whether
        two sets of numbers can be put side by side.
        """
        payload = {
            "n_rows": profile.n_rows,
            "target": profile.target,
            "features": sorted(c.name for c in profile.columns if c.role == "feature"),
            "task": profile.task,
            "balance": profile.class_balance,
            "test_size": settings.TRAIN_TEST_SPLIT,
            "seed": settings.RANDOM_SEED,
        }
        metadata = json.dumps(payload, sort_keys=True).encode("utf-8")
        try:
            content = int(pd.util.hash_pandas_object(
                frame[profile.target], index=False
            ).sum())
        except Exception:  # pragma: no cover - exotic dtypes
            content = 0
        return hashlib.sha1(metadata + str(content).encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Confirm target
    # ------------------------------------------------------------------

    def confirm_target(self, dataset_id: str, target: str) -> DatasetProfile:
        """Re-profiles against a user-chosen target column."""
        frame = self._require_frame(dataset_id)
        current = self._profiles[dataset_id]
        if target not in frame.columns:
            raise IngestionError(
                "unknown_target",
                f"'{target}' is not a column in '{current.name}'.",
            )

        # The previously inferred target may have been a feature; if so it goes
        # back into the feature pool under the new profile.
        reprofiled = self._profile_frame(dataset_id, current.name, frame, target_hint=target)
        reprofiled.created_at = current.created_at
        reprofiled.target_inferred = False
        self._profiles[dataset_id] = reprofiled
        logger.info(
            "Dataset %s target confirmed as '%s' (task=%s)", dataset_id, target, reprofiled.task
        )
        return reprofiled

    # ------------------------------------------------------------------
    # Read access (DataStore protocol)
    # ------------------------------------------------------------------

    def get_profile(self, dataset_id: str) -> DatasetProfile:
        return self._profiles.get(dataset_id) or self._reload(dataset_id)

    def get_frame(self, dataset_id: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Returns the feature frame and target series with unusable columns removed.

        Object columns are label-encoded here rather than at ingest so the stored
        CSV stays exactly what the researcher uploaded. The encoding is derived
        from sorted unique values, so it is stable for a given dataset.
        """
        profile = self.get_profile(dataset_id)
        raw = self._require_frame(dataset_id)

        target = profile.target
        if not isinstance(target, str) or not target or target not in raw.columns:
            raise IngestionError(
                "missing_target",
                f"Dataset {dataset_id} has no usable target column "
                f"(resolved {target!r} against {list(raw.columns)[:8]}).",
            )

        y = raw[target]
        if y.isna().any():
            y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 0)

        features = pd.DataFrame(index=raw.index)
        for column in profile.columns:
            if column.role != "feature" or column.name not in raw.columns:
                continue
            series = raw[column.name]
            if pd.api.types.is_numeric_dtype(series):
                features[column.name] = pd.to_numeric(series, errors="coerce")
            else:
                categories = sorted(series.dropna().astype(str).unique())
                mapping = {label: index for index, label in enumerate(categories)}
                features[column.name] = series.astype(str).map(mapping)
            features[column.name] = features[column.name].replace([np.inf, -np.inf], np.nan)

        # Median imputation: deterministic, and it never invents a category.
        if features.isna().any().any():
            features = features.fillna(features.median(numeric_only=True))
            features = features.fillna(0)

        return features, y

    def list_datasets(self) -> List[DatasetProfile]:
        return sorted(self._profiles.values(), key=lambda p: p.created_at, reverse=True)

    def clear(self) -> None:
        self._frames.clear()
        self._profiles.clear()
        self._paths.clear()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_frame(self, dataset_id: str) -> pd.DataFrame:
        frame = self._frames.get(dataset_id)
        if frame is not None:
            return frame
        return self._reload(dataset_id)

    def _reload(self, dataset_id: str) -> pd.DataFrame:
        """Rebuilds the cache from disk after a backend restart."""
        if dataset_id in self._profiles and self._profiles[dataset_id].split_hash:
            raise IngestionError(
                "dataset_evicted",
                f"Dataset {dataset_id} is no longer in memory. Upload the CSV again.",
            )

        candidates = sorted(self.data_dir.glob(f"{dataset_id}_*.csv"))
        if not candidates:
            raise IngestionError(
                "unknown_dataset", f"No dataset with id '{dataset_id}'."
            )
        frame = pd.read_csv(candidates[0])
        profile = self._profile_frame(dataset_id, candidates[0].name, frame)
        self._frames[dataset_id] = frame
        self._profiles[dataset_id] = profile
        self._paths[dataset_id] = str(candidates[0])
        logger.info("Reloaded dataset %s from %s", dataset_id, candidates[0].name)
        return frame

    def has(self, dataset_id: str) -> bool:
        return dataset_id in self._profiles or bool(list(self.data_dir.glob(f"{dataset_id}_*.csv")))


_ingestor: Optional[DatasetIngestor] = None


def get_ingestor() -> DatasetIngestor:
    """Process-wide ingestor. The frame cache must survive across requests."""
    global _ingestor
    if _ingestor is None:
        _ingestor = DatasetIngestor()
    return _ingestor