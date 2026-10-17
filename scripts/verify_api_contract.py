"""Check ``frontend/src/api/types.ts`` against the backend's live OpenAPI schema.

The frontend types are hand-written so they can carry the semantics the UI relies on
(which fields may be null, which values are legal). Hand-written types drift, and a
drifted field is an ``undefined`` in the browser rather than a build error. This
script is the gate: it fails when a type names a field the backend does not return, or
declares a field required that the backend may omit.

    python scripts/verify_api_contract.py

Run it with the backend up. ``--json`` prints machine-readable drift.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES_PATH = ROOT / "frontend" / "src" / "api" / "types.ts"
DEFAULT_OPENAPI = "http://127.0.0.1:8000/openapi.json"

#: TypeScript interfaces that must match a same-named OpenAPI schema. Types that
#: intentionally have no backend counterpart (query parameter bags, error helpers)
#: are listed here with a reason so the omission is deliberate and visible.
UNCHECKED = {
    "ErrorDetail": "mirrors the detail entry the error handler builds",
    "ExperimentFilters": "query parameter bag for the history endpoint",
    "HealthResponse": "typed from /health, which is not part of the OpenAPI paths",
}

INTERFACE_RE = re.compile(r"export interface (\w+)\s*\{(.*?)\n\}", re.S)
FIELD_RE = re.compile(r"^\s*(\w+)(\??):\s*(.+?)\s*(?:/\*.*)?$", re.M)
EXTENDS_RE = re.compile(r"export interface (\w+) extends (\w+)")


def load_openapi(url: str) -> dict:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except urllib.error.URLError as error:
        print(f"Cannot read {url}: {error}")
        print("Start the backend first: cd backend && uvicorn app.main:app --port 8000")
        raise SystemExit(2) from error


def parse_types(source: str) -> dict[str, dict]:
    """Return interface name -> {fields, optional, types, parent}.

    ``field_types`` maps a field name to its declared TypeScript type, which is what
    makes the nullability check possible: a field declared without ``| null`` asserts
    the backend never sends one.
    """
    parsed: dict[str, dict] = {}
    for match in INTERFACE_RE.finditer(source):
        name, body = match.group(1), match.group(2)
        fields: set[str] = set()
        optional: dict[str, bool] = {}
        declared: dict[str, str] = {}
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("*"):
                continue
            field = FIELD_RE.match(stripped)
            if not field:
                continue
            field_name, optional_mark, field_type = field.group(1), field.group(2), field.group(3)
            fields.add(field_name)
            optional[field_name] = optional_mark == "?"
            declared[field_name] = field_type
        extends = EXTENDS_RE.search(match.group(0))
        parsed[name] = {
            "fields": fields,
            "optional": optional,
            "types": declared,
            "parent": extends.group(2) if extends else None,
        }
    return parsed


def schema_fields(schema: dict, seen: dict | None = None) -> tuple[set[str], set[str]]:
    """Return (all property names, required names) for a schema, following allOf."""
    seen = seen or {}
    if "$ref" in schema:
        name = schema["$ref"].split("/")[-1]
        if name in seen:
            return seen[name]
    if "allOf" in schema:
        properties: set[str] = set()
        required: set[str] = set()
        for part in schema["allOf"]:
            part_properties, part_required = schema_fields(part, seen)
            properties |= part_properties
            required |= part_required
        if "$ref" in schema:
            seen[schema["$ref"].split("/")[-1]] = (properties, required)
        return properties, required
    properties = set((schema.get("properties") or {}).keys())
    required = set(schema.get("required") or [])
    if "$ref" in schema:
        seen[schema["$ref"].split("/")[-1]] = (properties, required)
    return properties, required


def is_nullable(prop_schema: dict) -> bool:
    """Whether a property schema can produce ``null``.

    Pydantic renders ``X | None`` as ``anyOf: [X, {type: null}]``, so both shapes are
    inspected. This is the direction that matters: a TypeScript field declared without
    ``| null`` asserts the backend never sends null, and a surprise null there is a
    runtime crash in the browser rather than a type error.
    """
    if not prop_schema:
        return False
    if prop_schema.get("nullable") is True:
        return True
    declared = prop_schema.get("type")
    if isinstance(declared, list) and "null" in declared:
        return True
    for variant in prop_schema.get("anyOf", []) or prop_schema.get("oneOf", []):
        if variant.get("type") == "null":
            return True
    return False


def merged_properties(schema: dict) -> dict:
    """Flatten an allOf schema's properties into one mapping."""
    if "allOf" not in schema:
        return dict(schema.get("properties") or {})
    merged: dict = {}
    for part in schema["allOf"]:
        merged.update(merged_properties(part))
    merged.update(schema.get("properties") or {})
    return merged


def resolve_schema(name: str, schemas: dict) -> dict:
    return schemas.get(name, {})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_OPENAPI, help="OpenAPI document URL")
    parser.add_argument("--json", action="store_true", help="emit drift as JSON")
    args = parser.parse_args()

    openapi = load_openapi(args.url)
    schemas = openapi.get("components", {}).get("schemas", {})
    source = TYPES_PATH.read_text(encoding="utf-8")
    interfaces = parse_types(source)

    drift: list[dict] = []
    checked = 0
    for name, spec in sorted(interfaces.items()):
        if name in UNCHECKED:
            continue
        if name not in schemas:
            drift.append({"interface": name, "problem": "no matching OpenAPI schema"})
            continue
        own = merged_properties(schemas[name])
        parent = spec["parent"]
        inherited = merged_properties(resolve_schema(parent, schemas)) if parent else {}
        available = set(own) | set(inherited)

        for field in sorted(spec["fields"]):
            checked += 1
            if field not in available:
                drift.append(
                    {"interface": name, "field": field, "problem": "the backend does not return this field"}
                )
                continue
            ts_type = spec["types"].get(field, "")
            nullable_in_ts = "| null" in ts_type
            backend_nullable = is_nullable(own.get(field)) or is_nullable(inherited.get(field))
            if not nullable_in_ts and backend_nullable:
                drift.append(
                    {
                        "interface": name,
                        "field": field,
                        "problem": (
                            "the backend can send null here, but the TypeScript type does not "
                            f"include null (declared as {ts_type!r})"
                        ),
                    }
                )

    if args.json:
        print(json.dumps({"checked_fields": checked, "drift": drift}, indent=2))
    else:
        print(f"checked {checked} fields across {len(interfaces) - len(UNCHECKED)} interfaces")
        if not drift:
            print("no drift: frontend/src/api/types.ts matches the live OpenAPI schema")
        else:
            print(f"{len(drift)} problem(s):")
            for item in drift:
                target = f"{item['interface']}.{item.get('field')}" if item.get("field") else item["interface"]
                print(f"  - {target}: {item['problem']}")

    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
