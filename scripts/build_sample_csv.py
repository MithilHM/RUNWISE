"""Regenerate ``sample_data/synthetic_experiments.csv`` from the JSON source of truth.

The CSV hand-written cells held Python repr text (``{'max_depth': 4}``) rather than
JSON, so every row reported a parse error on import. This script keeps the two files
in step instead of leaving them to drift.

    python scripts/build_sample_csv.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "sample_data" / "synthetic_experiments.json"
CSV_PATH = ROOT / "sample_data" / "synthetic_experiments.csv"

#: Column order, chosen to match the importer's alias table and to read as a
#: researcher's flat export.
COLUMNS = [
    "id",
    "project_id",
    "name",
    "model_family",
    "dataset_name",
    "dataset_version",
    "status",
    "duration_seconds",
    "compute_cost_gpu_hours",
    "random_seed",
    "configuration",
    "metrics",
    "description",
    "notes",
    "source_reference",
    "created_at",
    "parent_experiment_id",
]

#: Fields serialised as compact JSON in their cell rather than as a Python repr.
JSON_FIELDS = ("configuration", "hyperparameters", "preprocessing", "metrics")


def cell(record: dict, column: str) -> str:
    value = record.get(column)
    if value is None:
        return ""
    if column in JSON_FIELDS:
        return json.dumps(value, sort_keys=True, default=str)
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, default=str)
    return str(value)


def main() -> int:
    records = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        print(f"{JSON_PATH} does not contain a non-empty list of records")
        return 1

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({column: cell(record, column) for column in COLUMNS})

    print(f"Wrote {len(records)} records to {CSV_PATH.relative_to(ROOT)}")

    # Read it back the way the importer does and check every JSON cell parses, so a
    # drift between the two sample files is reported here rather than mid-demo.
    import pandas as pd  # noqa: PLC0415

    frame = pd.read_csv(CSV_PATH)
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    problems: list[str] = []
    for index, row in enumerate(rows, start=2):
        for field in JSON_FIELDS:
            raw = row.get(field)
            if raw is None or raw == "":
                continue
            try:
                json.loads(raw)
            except ValueError as error:
                problems.append(f"row {index}: {field} is not JSON ({error})")
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for problem in problems[:5]:
            print(f"  {problem}")
        return 1
    print(f"Verified {len(rows)} rows: every {', '.join(JSON_FIELDS)} cell parses as JSON")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
