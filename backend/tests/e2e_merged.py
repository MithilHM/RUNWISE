"""End-to-end check across the merged stack: upload -> run -> graph -> why.

Exercises Member B's HTTP layer against Member A's real FalkorDB-backed
implementations (not the fakes). Run from `backend/`:

    python -m tests.e2e_merged
"""

from __future__ import annotations

import io
import sys
import time

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def make_csv(n: int = 600, seed: int = 21) -> bytes:
    rng = np.random.RandomState(seed)
    signal = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        np.round(rng.normal(0, 1, (n, 6)), 3),
        columns=["f1", "f2", "f3", "f4", "f5", "f6"],
    )
    frame["tenure"] = rng.randint(1, 60, n)
    frame["target"] = ((signal * 0.8 + frame["f1"] * 0.6 + rng.normal(0, 0.3, n)) > 0).astype(int)
    return frame.to_csv(index=False).encode("utf-8")


def main() -> int:
    with TestClient(app) as client:
        print("\n== upload (F1) ==")
        response = client.post(
            "/api/v1/datasets",
            files={"file": ("customers.csv", io.BytesIO(make_csv()), "text/csv")},
        )
        check("upload accepted", response.status_code in (200, 201), f"HTTP {response.status_code}")
        profile = response.json()
        dataset_id = profile.get("id", "")
        check("dataset id returned", dataset_id.startswith("ds_"), dataset_id)
        check("target inferred", profile.get("target") == "target", str(profile.get("target")))
        check("task detected as binary", profile.get("task") == "binary", str(profile.get("task")))
        check("features survived", profile.get("n_cols", 0) > 5, str(profile.get("n_cols")))

        print("\n== catalog (F2/F3) ==")
        models = client.get("/api/v1/catalog/models").json()
        models = models.get("models", models)
        names = sorted(m["name"] for m in models)
        check("three models exposed", len(models) == 3, ", ".join(names))
        by_id = {m["id"]: m for m in models}
        check("each model has its own parameter space",
              len(by_id["xgboost"]["params"]) == 9 and len(by_id["lightgbm"]["params"]) == 9
              and len(by_id["random_forest"]["params"]) == 8,
              ", ".join(f"{k}={len(v['params'])}" for k, v in by_id.items()))

        print("\n== run (F4) ==")
        created = client.post("/api/v1/runs", json={
            "dataset_id": dataset_id,
            "model_ids": ["xgboost", "lightgbm"],
            "mode": "autonomous",
            "metric": "roc_auc",
            "budget": 3,
        })
        check("run created", created.status_code in (200, 201), f"HTTP {created.status_code} {created.text[:200]}")
        run_id = created.json().get("run_id", "")

        deadline = time.time() + 240
        state = {}
        while time.time() < deadline:
            state = client.get(f"/api/v1/runs/{run_id}").json()
            if state.get("status") in ("COMPLETED", "FAILED", "PAUSED"):
                break
            time.sleep(2)
        check("run reached a terminal state",
              state.get("status") == "COMPLETED", f"status={state.get('status')}")
        check("a best metric was recorded", state.get("best_metric") is not None,
              str(state.get("best_metric")))

        print("\n== logs (F6) ==")
        logs = client.get(f"/api/v1/runs/{run_id}/logs").json()
        entries = logs.get("entries", logs if isinstance(logs, list) else [])
        check("logs persisted", len(entries) > 0, f"{len(entries)} entries")
        stages = {e.get("stage") for e in entries}
        check("execute and propose were logged",
              {"propose", "execute"} <= stages, str(sorted(stages)))
        check("log sequence is monotonic",
              [e["seq"] for e in entries] == sorted(e["seq"] for e in entries))

        print("\n== graph (F5) ==")
        graph = client.get("/api/v1/graph", params={"run_id": run_id}).json()
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        node_types = {n.get("type") for n in nodes}
        edge_types = {e.get("type") for e in edges}
        check("graph has nodes", len(nodes) > 0, f"{len(nodes)} nodes, {len(edges)} edges")
        for label in ("Dataset", "Run", "Model", "Param", "Setting", "Trial", "Decision"):
            check(f"graph contains {label}", label in node_types, str(sorted(node_types)))
        for rel in ("ON_DATASET", "HAS_TRIAL", "USES_MODEL", "USES_SETTING", "CHOSE", "CITES"):
            check(f"graph contains -{rel}->", rel in edge_types, str(sorted(edge_types)))

        ids = [n["id"] for n in nodes]
        check("no duplicate node ids", len(ids) == len(set(ids)),
              f"{len(ids)} nodes, {len(set(ids))} distinct")
        edge_keys = [(e["source"], e["type"], e["target"]) for e in edges]
        check("no duplicate edges", len(edge_keys) == len(set(edge_keys)),
              f"{len(edge_keys)} edges, {len(set(edge_keys))} distinct")

        trials = [n for n in nodes if n.get("type") == "Trial"]
        check("trials recorded", len(trials) >= 3, f"{len(trials)} trials")

        print("\n== why (F7) ==")
        if trials:
            trial_id = trials[0]["id"]
            why = client.get("/api/v1/runs/{}/why".format(run_id),
                             params={"trial_id": trial_id})
            check("why endpoint responds", why.status_code == 200, f"HTTP {why.status_code}")
            record = why.json()
            check("decision returned", bool(record.get("decision", {}).get("why")),
                  str(record.get("decision", {}).get("why", ""))[:80])
            check("evidence resolved", len(record.get("evidence", [])) > 0,
                  f"{len(record.get('evidence', []))} items")
            highlight = set(record.get("highlight", []))
            evidence_ids = {e["id"] for e in record.get("evidence", [])}
            check("every highlighted node resolved", highlight == evidence_ids,
                  f"{highlight - evidence_ids}")

            node_id_set = set(ids)
            check("highlighted nodes exist in the graph", highlight <= node_id_set,
                  str(list(highlight - node_id_set)[:3]))

            cites = {(e["source"], e["target"]) for e in edges if e["type"] == "CITES"}
            decision_id = record.get("decision", {}).get("decision_id", "")
            check("highlight is backed by real CITES edges",
                  all((decision_id, nid) in cites for nid in highlight),
                  f"{decision_id} -> {len(highlight)} highlighted")

        print("\n== mcp ==")
        tools = client.get("/api/v1/mcp/tools").json()
        check("mcp tools listed", len(tools.get("tools", [])) > 0,
              f"{len(tools.get('tools', []))} tools")
        rpc = client.post("/api/v1/mcp/rpc", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "list_models", "arguments": {}},
        })
        check("mcp tools/call responds", rpc.status_code == 200, f"HTTP {rpc.status_code}")

    print(f"\n{'E2E PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S): ' + '; '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())