"""Smoke test for the Member A stack: ingest -> train -> graph -> why.

Runs against live FalkorDB. Run from `backend/`:

    python -m tests.smoke_member_a
"""

from __future__ import annotations

import io
import json
import sys
import traceback

import numpy as np
import pandas as pd

from app.graph.repository import get_graph_store
from app.models.contracts import RunState
from app.services import catalog
from app.services.executor import get_trainer
from app.services.ingestion import get_ingestor
from app.services.laya import LayaAgent, get_laya
from app.services.logs import get_logger_


def make_csv(n: int = 400, seed: int = 7) -> bytes:
    rng = np.random.RandomState(seed)
    signal = rng.normal(0, 1, n)
    noise = rng.normal(0, 1, (n, 4))
    frame = pd.DataFrame(
        # Coarse floats so the feature columns are not all-distinct by accident.
        np.round(noise, 3),
        columns=["f_a", "f_b", "f_c", "f_d"],
    )
    frame["target"] = ((signal * 0.9 + noise[:, 0] * 0.5 + rng.normal(0, 0.2, n)) > 0).astype(int)
    # An exact copy of the target: the unambiguous leak case.
    frame["f_leaky"] = frame["target"]
    frame["constant"] = 1
    frame["row_id"] = range(n)
    return frame.to_csv(index=False).encode("utf-8")


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {label}" + (f" :: {detail}" if detail else ""))
    return condition


def main() -> int:
    failures = 0

    print("\n== catalog ==")
    failures += not check("exactly 3 models", len(catalog.MODEL_SPECS) == 3,
                          ", ".join(catalog.MODEL_IDS))
    for spec in catalog.MODEL_SPECS:
        try:
            catalog.build_estimator(spec.id, catalog.default_params(spec.id))
            failures += not check(f"{spec.id} estimator builds from defaults", True)
        except Exception as exc:
            failures += not check(f"{spec.id} estimator builds from defaults", False, str(exc))

    # clamp / step-grid behaviour
    cases = [
        ("xgboost", {"learning_rate": 99}, "learning_rate", 0.5),
        ("xgboost", {"subsample": 0.41}, "subsample", 0.4),
        ("xgboost", {"n_estimators": 137}, "n_estimators", 150),
        ("xgboost", {"max_depth": 0}, "max_depth", 2),
        ("random_forest", {"n_estimators": 9999}, "n_estimators", 500),
        ("random_forest", {"max_depth": None}, "max_depth", None),
    ]
    for model_id, raw, key, expected in cases:
        got = catalog.coerce_params(model_id, raw)[key]
        failures += not check(f"clamp {model_id}.{key} {raw[key]!r} -> {expected}", got == expected, f"got {got!r}")

    failures += not check(
        "lightgbm num_leaves > depth ceiling is rejected",
        any("silently ignored" in p for p in catalog.validate_params(
            "lightgbm", catalog.coerce_params("lightgbm", {"num_leaves": 200, "max_depth": 3}))),
    )

    print("\n== ingestion ==")
    ingestor = get_ingestor()
    profile = ingestor.ingest("smoke.csv", make_csv())
    failures += not check("dataset id assigned", profile.id.startswith("ds_"), profile.id)
    failures += not check("task is binary", profile.task == "binary", profile.task)
    failures += not check("target inferred as 'target'", profile.target == "target", profile.target)
    reasons = {d.name: d.reason for d in profile.dropped_columns}
    failures += not check("constant column dropped", reasons.get("constant") == "constant", str(reasons))
    failures += not check("row id dropped", reasons.get("row_id") == "high_cardinality_id", str(reasons))
    failures += not check("leaky column dropped", reasons.get("f_leaky") == "leakage", str(reasons))
    failures += not check("4 real features survive", profile.n_cols == 5, f"n_cols={profile.n_cols}")
    failures += not check("split hash recorded", len(profile.split_hash) == 16, profile.split_hash)

    for bad, code in [
        (b"", "empty_file"),
        (b"a,b,c\n1,2\n3,4,5\n", "unparseable_csv"),
    ]:
        try:
            ingestor.ingest("bad.csv", bad)
            failures += not check(f"rejects {code}", False, "no error raised")
        except Exception as exc:
            failures += not check(f"rejects {code}", getattr(exc, "code", "") == code, str(exc))

    print("\n== executor ==")
    trainer = get_trainer(ingestor)
    for model_id in catalog.MODEL_IDS:
        outcome = trainer.train(profile.id, model_id, {}, "accuracy")
        failures += not check(
            f"{model_id} trains",
            outcome.status == "SUCCESS" and outcome.metric_value is not None,
            f"{outcome.status} {outcome.metric_value} {outcome.failure_reason or ''}",
        )
        failures += not check(
            f"{model_id} reports all 4 binary metrics",
            all(outcome.metrics.get(k) is not None
                for k in ("accuracy", "f1_macro", "roc_auc", "pr_auc")),
            str(outcome.metrics),
        )

    again = trainer.train(profile.id, "xgboost", {}, "accuracy")
    same = trainer.train(profile.id, "xgboost", {}, "accuracy")
    failures += not check("same config -> same score (seeded)",
                          again.metric_value == same.metric_value,
                          f"{again.metric_value} vs {same.metric_value}")

    bad = trainer.train(profile.id, "xgboost", {"learning_rate": 0.0}, "accuracy")
    failures += not check("invalid params -> FAILED", bad.status == "FAILED", bad.status)
    failures += not check("invalid params -> no metric", bad.metric_value is None, str(bad.metric_value))
    failures += not check("invalid params -> reason given", bool(bad.failure_reason), str(bad.failure_reason))

    store = get_graph_store()
    store.ensure_schema()
    failures += not check("FalkorDB reachable", store.is_healthy())

    print("\n== graph writes ==")
    run = RunState(
        run_id="run_smoke01", dataset_id=profile.id, mode="autonomous", status="RUNNING",
        step=1, budget=4, metric_name="accuracy", model_ids=list(catalog.MODEL_IDS),
        created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00",
    )
    store.write_dataset(profile)
    store.write_models(catalog.list_models())
    store.write_run(run)

    trial = trainer.train(profile.id, "xgboost", {"learning_rate": 0.05, "max_depth": 4}, "accuracy")
    trial.run_id = run.run_id
    trial.step = 1

    laya = get_laya()
    graph_before = store.read_graph(run_id=run.run_id)
    proposal = laya.propose(run, graph_before)
    failures += not check("Laya produced a proposal", bool(proposal.model_id), proposal.model_id)
    failures += not check("Laya proposal cites graph nodes", len(proposal.citations) > 0,
                          str(proposal.citations[:4]))
    failures += not check("Laya produced a rationale", len(proposal.why) > 40, proposal.why[:80])

    from app.models.contracts import Decision
    from datetime import datetime, timezone

    decision = Decision(
        decision_id=f"decision_{run.run_id}_s001_laya", run_id=run.run_id, step=1,
        trial_id=trial.trial_id, actor="laya", strategy=proposal.strategy,
        why=proposal.why, hypothesis=proposal.hypothesis,
        confidence=proposal.confidence, citations=proposal.citations,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store.write_trial(trial, decision)

    log = get_logger_()
    log.append(run.run_id, "INFO", "execute", f"trained {trial.model_id}", step=1)

    snapshot = store.read_graph(run_id=run.run_id)
    types = {n.type for n in snapshot.nodes}
    edge_types = {e.type for e in snapshot.edges}
    failures += not check("graph has Trial node", "Trial" in types, str(sorted(types)))
    failures += not check("graph has Model/Param/Setting nodes",
                          {"Model", "Param", "Setting"} <= types, str(sorted(types)))
    failures += not check("graph has Decision node", "Decision" in types)
    failures += not check("CITES edges written", "CITES" in edge_types, str(sorted(edge_types)))
    failures += not check("USES_SETTING edges written", "USES_SETTING" in edge_types)
    failures += not check("FOR_PARAM edges written", "FOR_PARAM" in edge_types)
    failures += not check("IN_RUN log edge written", "IN_RUN" in edge_types)

    # settings dedup: same value again -> same node
    before = sum(1 for n in snapshot.nodes if n.type == "Setting")
    twin = trainer.train(profile.id, "xgboost", {"learning_rate": 0.05, "max_depth": 4}, "accuracy")
    twin.run_id = run.run_id
    twin.step = 2
    twin.trial_id = "trial_smoke_dup"
    store.write_trial(twin, Decision(
        decision_id=f"decision_{run.run_id}_s002_laya", run_id=run.run_id, step=2,
        trial_id=twin.trial_id, actor="laya", strategy="replay", why="replay of an identical configuration",
        hypothesis="identical config", confidence=0.5, citations=[],
        created_at=datetime.now(timezone.utc).isoformat(),
    ))
    after_snapshot = store.read_graph(run_id=run.run_id)
    after = sum(1 for n in after_snapshot.nodes if n.type == "Setting")
    failures += not check("identical params reuse Setting nodes", before == after, f"{before} -> {after}")

    print("\n== why (F7) ==")
    why = store.explain_trial(run.run_id, trial.trial_id)
    failures += not check("why returns a decision", why.decision.decision_id == decision.decision_id)
    failures += not check("highlight == citations", why.highlight == decision.citations,
                          f"{why.highlight} vs {decision.citations}")
    failures += not check("evidence resolved", len(why.evidence) > 0, str(len(why.evidence)))
    unresolved = [c for c in why.highlight if c not in {e.id for e in why.evidence}]
    failures += not check("every highlighted node resolves", not unresolved, str(unresolved))
    print(f"       evidence: {[e.label for e in why.evidence][:5]}")

    print("\n== logs ==")
    history = log.history(run.run_id)
    failures += not check("log round-tripped from graph", len(history) >= 1, str(len(history)))
    persisted = store.read_logs(run.run_id)
    failures += not check("graph holds the log entry", len(persisted) >= 1, str(len(persisted)))

    print("\n== laya selection validation ==")
    from app.services.llm import LLMResult

    class StubLLM:
        def __init__(self, text):
            self._text = text
            self.is_available = True

        def complete(self, prompt, system=None, max_tokens=400):
            return LLMResult(text=self._text, used=True)

        @staticmethod
        def extract_json(text, prefer_keys=None):
            from app.services.llm import LLMClient

            return LLMClient.extract_json(text, prefer_keys=prefer_keys)

    stub = LayaAgent(llm_client=StubLLM('{"selected": "nope_not_real", "why": "x", "confidence": 0.9}'))
    bad_id = stub.propose(run, graph_before)
    failures += not check("LLM naming a fake candidate is rejected",
                           "llm" not in bad_id.strategy, bad_id.strategy)

    stub = LayaAgent(llm_client=StubLLM("I think cand_b sounds nicer, honestly?"))
    garbage = stub.propose(run, graph_before)
    failures += not check("LLM returning prose is rejected",
                          "llm" not in garbage.strategy, garbage.strategy)

    stub = LayaAgent(llm_client=StubLLM(
        'Here is my answer. {"selected": "%s", "why": "Because the incumbent scored best '
        'and one step of regularisation is the cheapest test.", "hypothesis": "n", "confidence": 0.77}'
        % bad_id.citations and "REALID"
    ))
    real_cid = LayaAgent(llm_client=StubLLM("{}")).propose(run, graph_before)
    # Build the agent's own candidates to learn the real ids, then answer with one.
    probe = LayaAgent(llm_client=StubLLM("{}"))
    cands = probe._build_candidates(run, [], graph_before)
    stub = LayaAgent(llm_client=StubLLM(
        'Reasoning omitted. {"selected": "%s", "why": "The incumbent scored best and one '
        'step of regularisation is the cheapest remaining test.", "hypothesis": "nearby '
        'values stay close.", "confidence": 0.77}' % cands[0].candidate_id
    ))
    picked = stub.propose(run, graph_before)
    failures += not check("LLM naming a real candidate is honoured",
                          "(llm)" in picked.strategy, picked.strategy)
    failures += not check("LLM rationale is used verbatim",
                          picked.why.startswith("The incumbent scored best"), picked.why[:60])
    failures += not check("LLM citations still come from the candidate",
                          picked.citations == cands[0].citations, str(picked.citations[:3]))

    stub = LayaAgent(llm_client=StubLLM(""))
    offline = LayaAgent(llm_client=StubLLM(""))
    offline.llm_client = type("Off", (), {"is_available": False})()
    det = offline.propose(run, graph_before)
    failures += not check("no LLM -> deterministic policy", "(llm)" not in det.strategy, det.strategy)
    failures += not check("deterministic still grounds itself in graph nodes",
                          len(det.citations) > 0, str(det.citations[:3]))

    print("\n== queries ==")
    failures += not check("list_trials", len(store.list_trials(run_id=run.run_id)) >= 2)
    best = store.best_trials(run.run_id)
    failures += not check("best_trials groups per model", isinstance(best, list), str(best)[:120])
    hood = store.graph_neighbourhood(decision.citations[0], depth=1)
    failures += not check("graph_neighbourhood", len(hood["nodes"]) > 1, str(len(hood["nodes"])))

    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} CHECK(S) FAILED'}")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)