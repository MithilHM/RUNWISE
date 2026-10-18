"""Regression tests for the defects the adversarial review found.

Each block is named after the finding it closes. Run from `backend/`:

    python -m tests.regressions
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np
import pandas as pd

from app.graph.repository import get_graph_store, param_id, setting_id
from app.models.contracts import RunState, TrialOutcome, is_better, is_number
from app.services import catalog
from app.services.executor import get_trainer
from app.services.ingestion import DatasetIngestor, IngestionError
from app.services.laya import LayaAgent
from app.services.logs import RunLogger

FAILURES: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def frame_csv(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def make_classification(n: int = 500, seed: int = 3, positives: int = 120) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    signal = rng.normal(0, 1, n)
    frame = pd.DataFrame(
        np.round(rng.normal(0, 1, (n, 5)), 3),
        columns=["a", "b", "c", "d", "e"],
    )
    frame["target"] = (signal > 0).astype(int)
    if positives < n:
        frame.loc[frame.index[: n - positives], "target"] = 0
    return frame


# ---------------------------------------------------------------------------


def c1_wrong_typed_params() -> None:
    print("\n== C1: wrong-typed params must not raise ==")
    for model_id, bad in [
        ("xgboost", {"n_estimators": "abc"}),
        ("xgboost", {"max_depth": "deep"}),
        ("xgboost", {"learning_rate": {}}),
        ("xgboost", {"subsample": [1]}),
        ("lightgbm", {"num_leaves": "many"}),
        ("random_forest", {"min_samples_split": None}),
        ("xgboost", {"subsample": True}),
    ]:
        try:
            problems = catalog.validate_params(model_id, bad, strict=True)
            check(f"strict validate rejects {model_id}.{list(bad)[0]}={bad[list(bad)[0]]!r}",
                  bool(problems), problems[0] if problems else "no problem reported")
        except Exception as exc:
            check(f"strict validate {model_id}.{list(bad)[0]}", False, f"RAISED {type(exc).__name__}: {exc}")

    ingestor = DatasetIngestor(data_dir="runs_data")
    profile = ingestor.ingest("c1.csv", frame_csv(make_classification()))
    trainer = get_trainer(ingestor)
    for bad in [{"n_estimators": "abc"}, {"learning_rate": None}, {"max_depth": "x"}]:
        try:
            outcome = trainer.train(profile.id, "xgboost", bad, "accuracy")
            ok = outcome.status == "FAILED" and outcome.metric_value is None and bool(outcome.failure_reason)
            check(f"train() returns FAILED for {list(bad)[0]}={bad[list(bad)[0]]!r}", ok,
                  f"{outcome.status}/{outcome.metric_value}")
        except Exception as exc:
            check(f"train() {list(bad)[0]}", False, f"RAISED {type(exc).__name__}: {exc}")


def c2_nan_metric() -> None:
    print("\n== C2: NaN must never enter a trial or ranking ==")
    check("is_number rejects NaN", not is_number(float("nan")))
    check("is_number rejects inf", not is_number(float("inf")))
    check("is_better rejects a NaN candidate", not is_better(float("nan"), 0.5, "accuracy"))
    check("is_better rejects a NaN incumbent",
          is_better(0.9, float("nan"), "accuracy"), "a real result must be able to win")

    try:
        TrialOutcome(trial_id="t", model_id="xgboost", status="SUCCESS", metric_value=None)
        check("SUCCESS with no metric is rejected by the model", False, "constructed")
    except Exception:
        check("SUCCESS with no metric is rejected by the model", True)
    try:
        TrialOutcome(trial_id="t", model_id="xgboost", status="FAILED", metric_value=0.9,
                     failure_reason="x")
        check("FAILED carrying a metric is rejected by the model", False, "constructed")
    except Exception:
        check("FAILED carrying a metric is rejected by the model", True)
    try:
        TrialOutcome(trial_id="t", model_id="xgboost", status="FAILED", metric_value=None,
                     failure_reason=None)
        check("FAILED without a reason is rejected by the model", False, "constructed")
    except Exception:
        check("FAILED without a reason is rejected by the model", True)

    # A two-positive dataset: roc_auc returns NaN on such a fold.
    ingestor = DatasetIngestor(data_dir="runs_data")
    frame = make_classification(n=300, positives=2)
    profile = ingestor.ingest("c2.csv", frame_csv(frame))
    trainer = get_trainer(ingestor)
    outcome = trainer.train(profile.id, "xgboost", {}, "roc_auc")
    check("unmeasurable metric -> FAILED, not a NaN SUCCESS",
          outcome.status == "FAILED" and outcome.metric_value is None,
          f"{outcome.status}/{outcome.metric_value}")
    check("failure explains the missing measurement",
          outcome.failure_reason is not None and "roc_auc" in (outcome.failure_reason or ""),
          str(outcome.failure_reason))
    check("no NaN anywhere in metrics",
          all(v is None or is_number(v) for v in outcome.metrics.values()), str(outcome.metrics))
    try:
        json.dumps(outcome.model_dump())
        check("trial is JSON-serialisable", True)
    except ValueError as exc:
        check("trial is JSON-serialisable", False, str(exc))


def c3_best_trials_groups() -> None:
    print("\n== C3: best_trials must group per model and take the max ==")
    store = get_graph_store()
    store.reset()
    store.ensure_schema()
    store.write_models(catalog.list_models())

    run_id = "run_reg_c3"
    store.write_run(RunState(run_id=run_id, dataset_id="ds_x", mode="autonomous",
                             step=0, budget=9, metric_name="accuracy",
                             model_ids=["random_forest", "xgboost"]))

    from app.models.contracts import Decision
    from datetime import datetime, timezone

    truth = {"random_forest": 0.11, "random_forest#2": 0.95, "random_forest#3": 0.42,
             "xgboost": 0.30, "xgboost#2": 0.77}
    expected = {"random_forest": 0.95, "xgboost": 0.77}
    for key, value in truth.items():
        model_id = key.split("#")[0]
        trial = TrialOutcome(trial_id=f"trial_reg_{key}", run_id=run_id, step=1,
                             model_id=model_id, params={"n_estimators": 100},
                             status="SUCCESS", metric_name="accuracy", metric_value=value)
        store.write_trial(trial, Decision(
            decision_id=f"dec_{key}", run_id=run_id, step=1, trial_id=trial.trial_id,
            actor="laya", strategy="t", why="w", hypothesis="h", confidence=0.5,
            citations=[], created_at=datetime.now(timezone.utc).isoformat()))

    best = store.best_trials(run_id)
    check("one row per model", len(best) == 2, f"got {len(best)}")
    check("returns the max, not an arbitrary trial",
          all(abs(b["metric_value"] - expected[b["model_id"]]) < 1e-9 for b in best),
          str([(b["model_id"], b["metric_value"]) for b in best]))


def c4_no_duplicate_nodes() -> None:
    print("\n== C4: repeated writes must not duplicate nodes or edges ==")
    store = get_graph_store()
    store.reset()
    store.ensure_schema()
    store.write_models(catalog.list_models())

    def count(label: str) -> int:
        rows = store._run(f"MATCH (n:{label}) RETURN count(n)")
        return int(rows[0][0])

    base = {label: count(label) for label in ("Model", "Param", "Setting", "Trial")}

    run_id = "run_reg_c4"
    store.write_run(RunState(run_id=run_id, dataset_id="ds_x", mode="autonomous",
                             step=0, budget=9, metric_name="accuracy",
                             model_ids=list(catalog.MODEL_IDS)))
    from app.models.contracts import Decision
    from datetime import datetime, timezone

    for step in range(1, 7):
        trial = TrialOutcome(trial_id=f"trial_c4_{step}", run_id=run_id, step=step,
                             model_id="xgboost", params={"n_estimators": 100 + step,
                                                         "max_depth": 3 + step % 4,
                                                         "learning_rate": 0.05},
                             status="SUCCESS", metric_name="accuracy", metric_value=0.5 + step / 100)
        store.write_trial(trial, Decision(
            decision_id=f"dec_c4_{step}", run_id=run_id, step=step, trial_id=trial.trial_id,
            actor="laya", strategy="t", why="w", hypothesis="h", confidence=0.5,
            citations=["xgboost", param_id("xgboost", "max_depth")],
            created_at=datetime.now(timezone.utc).isoformat()))

    after = {label: count(label) for label in ("Model", "Param", "Setting", "Trial")}
    check("Model nodes not duplicated", after["Model"] == base["Model"],
          f"{base['Model']} -> {after['Model']}")
    check("Param nodes not duplicated", after["Param"] == base["Param"],
          f"{base['Param']} -> {after['Param']}")
    check("6 trials recorded", after["Trial"] == 6, str(after["Trial"]))

    rows = store._run("MATCH (n:Agent) RETURN count(n)")
    check("Agent nodes deduplicated to one per actor", int(rows[0][0]) == 1, str(int(rows[0][0])))

    snapshot = store.read_graph(run_id=run_id)
    node_ids = [n.id for n in snapshot.nodes]
    edge_keys = [(e.source, e.type, e.target) for e in snapshot.edges]
    check("no duplicate node ids in the snapshot", len(node_ids) == len(set(node_ids)),
          f"{len(node_ids)} nodes, {len(set(node_ids))} distinct")
    check("no duplicate edges in the snapshot", len(edge_keys) == len(set(edge_keys)),
          f"{len(edge_keys)} edges, {len(set(edge_keys))} distinct")

    rows = store._run("MATCH ()-[r:CITES]->() RETURN count(r)")
    cites = int(rows[0][0])
    check("CITES deduplicated", cites == 12, f"{cites} edges for 6 decisions x 2 citations")


def c5_degenerate_target() -> None:
    print("\n== C5: a continuous or unusable target must be refused at ingest ==")
    ingestor = DatasetIngestor(data_dir="runs_data")
    trainer = get_trainer(ingestor)

    # The structural fallback already skips high-cardinality columns, so the
    # realistic route to a continuous label is a user confirming one explicitly.
    continuous = pd.DataFrame({"price": np.arange(300.0), "a": np.arange(300.0) % 7,
                               "b": np.arange(300.0) % 5})
    try:
        ingestor.ingest("cont.csv", frame_csv(continuous), target_hint="price")
        check("explicitly confirmed continuous target refused", False, "accepted")
    except IngestionError as exc:
        check("explicitly confirmed continuous target refused",
              exc.code == "target_not_a_label", f"{exc.code}: {exc.message}")

    # And the structural fallback must not pick one on its own either.
    profile = ingestor.ingest("cont2.csv", frame_csv(continuous))
    check("structural fallback avoids the continuous column",
          profile.target != "price", profile.target)

    # One member in the rarest class.
    frame = pd.DataFrame({"x": np.arange(20.0), "signal": np.arange(20.0) % 4,
                          "target": [0] * 19 + [1]})
    try:
        ingestor.ingest("thin.csv", frame_csv(frame))
        check("one-member class refused at ingest", False, "accepted")
    except IngestionError as exc:
        check("one-member class refused at ingest", exc.code == "class_too_small",
              f"{exc.code}: {exc.message}")

    # A dataset too small to split must never raise out of train(). It may well be
    # splittable, so the contract asserted here is the real one: no exception
    # escapes, and the outcome is self-consistent either way.
    tiny = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "target": [0, 0, 0, 1, 1, 1]})
    try:
        p = ingestor.ingest("tiny.csv", frame_csv(tiny))
        outcome = trainer.train(p.id, "xgboost", {}, "accuracy")
        if outcome.status == "SUCCESS":
            check("tiny dataset: SUCCESS carries a finite metric",
                  is_number(outcome.metric_value), str(outcome.metric_value))
        else:
            check("tiny dataset: FAILED carries a reason",
                  bool(outcome.failure_reason), str(outcome.failure_reason))
            check("tiny dataset: FAILED carries no metric",
                  outcome.metric_value is None, str(outcome.metric_value))
    except Exception as exc:
        check("tiny dataset -> no exception escapes train()", False,
              f"RAISED {type(exc).__name__}: {exc}")


def m1_identifier() -> None:
    print("\n== M1: the identifier heuristic must not eat real features ==")
    ingestor = DatasetIngestor(data_dir="runs_data")
    rng = np.random.RandomState(11)
    n = 400
    frame = pd.DataFrame({
        "measurement": rng.permutation(n),           # unique ints, shuffled: a feature
        "row_id": list(range(n)),                     # unique ints, in order: an id
        "signal": rng.normal(0, 1, n),
        "target": (rng.normal(0, 1, n) > 0).astype(int),
    })
    profile = ingestor.ingest("m1.csv", frame_csv(frame))
    reasons = {d.name: d.reason for d in profile.dropped_columns}
    check("ordered row_id dropped", reasons.get("row_id") == "high_cardinality_id", str(reasons))
    check("shuffled unique-int feature kept", "measurement" not in reasons, str(reasons))

    floats = frame.copy()
    floats["row_id"] = floats["row_id"].astype(float)
    profile2 = ingestor.ingest("m1b.csv", frame_csv(floats))
    reasons2 = {d.name: d.reason for d in profile2.dropped_columns}
    check("float row_id still dropped", reasons2.get("row_id") == "high_cardinality_id", str(reasons2))


def m2_leakage() -> None:
    print("\n== M2: leakage detection must catch one-to-one columns ==")
    ingestor = DatasetIngestor(data_dir="runs_data")
    rng = np.random.RandomState(5)
    n = 300
    y = (rng.normal(0, 1, n) > 0).astype(int)
    labels = np.array(["pos" if v == 1 else rng.choice(["neg_a", "neg_b"]) for v in y])
    frame = pd.DataFrame({"sneaky": labels, "signal": rng.normal(0, 1, n), "target": y})
    profile = ingestor.ingest("m2.csv", frame_csv(frame))
    reasons = {d.name: d.reason for d in profile.dropped_columns}
    check("3-valued perfect leak against a binary target is caught",
          reasons.get("sneaky") == "leakage", str(reasons))
    check("the honest feature survives", "signal" in [c.name for c in profile.columns])


def m3_m4_scoping() -> None:
    print("\n== M3/M4: graph scoping must include Agent and the Model layer ==")
    store = get_graph_store()
    run_id = "run_reg_c4"
    run_scope = store.read_graph(run_id=run_id)
    types = {n.type for n in run_scope.nodes}
    edge_types = {e.type for e in run_scope.edges}
    check("run scope has Agent nodes", "Agent" in types, str(sorted(types)))
    check("run scope has DECIDED edges", "DECIDED" in edge_types, str(sorted(edge_types)))
    check("run scope has Model nodes", "Model" in types)

    rows = store._run(
        "MATCH (d:Dataset)-[:HAS_COLUMN]->() OPTIONAL MATCH (r:Run)-[:ON_DATASET]->(d) "
        "RETURN d.id LIMIT 1")
    if rows:
        ds = rows[0][0]
        scope = store.read_graph(dataset_id=ds)
        dtypes = {n.type for n in scope.nodes}
        check("dataset scope has Model nodes", "Model" in dtypes, str(sorted(dtypes)))
        check("dataset scope has USES_MODEL edges",
              "USES_MODEL" in {e.type for e in scope.edges})


class OfflineLLM:
    """A client with a key configured but no network, so Laya skips the call.

    Matters: `LayaAgent.propose` calls the LLM synchronously, so leaving the real
    client in place turns a 360-draw loop into 360 network round trips.
    """

    is_available = False

    def complete(self, *args, **kwargs):
        raise AssertionError("must not be called when is_available is False")


def m5_laya_valid_configs() -> None:
    print("\n== M5: Laya must never propose a config the catalog rejects ==")
    from app.models.contracts import GraphSnapshot

    invalid: list[str] = []
    total = 0
    for seed in range(12):
        agent = LayaAgent(llm_client=OfflineLLM(), random_seed=seed)
        run = RunState(run_id=f"r{seed}", dataset_id="d", mode="autonomous", status="RUNNING",
                       step=seed % 3, budget=9, metric_name="accuracy",
                       model_ids=list(catalog.MODEL_IDS))
        for _ in range(30):
            proposal = agent.propose(run, GraphSnapshot())
            total += 1
            problems = catalog.validate_params(proposal.model_id, proposal.params)
            if problems:
                invalid.append(f"{proposal.model_id}: {problems[0]}")
    check("zero invalid proposals across 360 draws", not invalid,
          f"{len(invalid)}/{total} invalid; first: {invalid[0] if invalid else ''}")

    # And the incumbent path: a valid incumbent must not be corrupted by jitter.
    corrupted = 0
    for seed in range(8):
        agent = LayaAgent(llm_client=OfflineLLM(), random_seed=seed)
        trial = TrialOutcome(trial_id="t", model_id="lightgbm", step=1,
                             params={"num_leaves": 31, "max_depth": 5},
                             status="SUCCESS", metric_name="accuracy", metric_value=0.8,
                             metrics={"accuracy": 0.8})
        from app.models.contracts import GraphNode

        graph = GraphSnapshot(nodes=[GraphNode(
            id="t", label="t", type="Trial",
            props={"id": "t", "model_id": "lightgbm", "status": "SUCCESS", "step": 1,
                   "metric_name": "accuracy", "metric_value": 0.8,
                   "metrics": {"accuracy": 0.8}, "params": trial.params})])
        run = RunState(run_id="r", dataset_id="d", mode="autonomous", status="RUNNING",
                       step=1, budget=9, metric_name="accuracy", model_ids=list(catalog.MODEL_IDS))
        proposal = agent.propose(run, graph)
        if catalog.validate_params(proposal.model_id, proposal.params):
            corrupted += 1
    check("jitter never corrupts a valid LightGBM incumbent", corrupted == 0, str(corrupted))


def m6_target_hints() -> None:
    print("\n== M6: target-name hints must not hijack feature columns ==")
    ingestor = DatasetIngestor(data_dir="runs_data")
    rng = np.random.RandomState(9)
    n = 400
    frame = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "is_fraud": rng.normal(0, 1, n),
        "class_size": rng.normal(0, 1, n),
        "churn": (rng.normal(0, 1, n) > 0).astype(int),
    })
    profile = ingestor.ingest("m6.csv", frame_csv(frame))
    check("'is_fraud' is not mistaken for the target", profile.target == "churn", profile.target)
    check("'class_size' does not win the 'class' hint", profile.target == "churn", profile.target)
    check("the other three columns survive as features",
          len([c for c in profile.columns if c.role == "feature"]) == 3,
          str([c.name for c in profile.columns]))


def m7_split_hash() -> None:
    print("\n== M7: split_hash must distinguish different data ==")
    ingestor = DatasetIngestor(data_dir="runs_data")
    a = ingestor.ingest("m7a.csv", frame_csv(make_classification(n=300, seed=1)))
    b = ingestor.ingest("m7b.csv", frame_csv(make_classification(n=300, seed=2)))
    c = ingestor.ingest("m7c.csv", frame_csv(make_classification(n=300, seed=1)))
    check("different data -> different hash", a.split_hash != b.split_hash,
          f"{a.split_hash} vs {b.split_hash}")
    check("identical data -> identical hash", a.split_hash == c.split_hash)


def m8_m9_m10_graph_edges() -> None:
    print("\n== M8/M9/M10: settings, run scoping and CITES labels ==")
    store = get_graph_store()
    run_id = "run_reg_c4"

    rows = store._run("MATCH (s:Setting) WHERE s.value = '\"-99\"' OR s.value = '0.0' RETURN s.id")
    check("no Setting node for a rejected value", len(rows) == 0, str([r[0] for r in rows]))

    from app.models.contracts import Decision
    from datetime import datetime, timezone

    orphan = TrialOutcome(trial_id="trial_orphan", run_id="", step=0, model_id="xgboost",
                          params={"n_estimators": 120}, status="SUCCESS",
                          metric_name="accuracy", metric_value=0.6)
    store.write_trial(orphan, Decision(decision_id="dec_orphan", run_id="", step=0,
                                       trial_id=orphan.trial_id, actor="laya", strategy="t",
                                       why="w", hypothesis="h", confidence=0.5, citations=[],
                                       created_at=datetime.now(timezone.utc).isoformat()))
    rows = store._run("MATCH (r:Run)-[:HAS_TRIAL]->(t:Trial {id: 'trial_orphan'}) RETURN count(t)")
    check("orphan trial is not attached to a run", int(rows[0][0]) == 0)

    # A citation to a Dataset must produce a real edge.
    profile_rows = store._run("MATCH (d:Dataset) RETURN d.id LIMIT 1")
    if profile_rows:
        ds_id = profile_rows[0][0]
        dec = Decision(decision_id="dec_cite_ds", run_id=run_id, step=1, actor="laya",
                       strategy="t", why="w", hypothesis="h", confidence=0.5,
                       citations=[ds_id], created_at=datetime.now(timezone.utc).isoformat())
        trial = TrialOutcome(trial_id="trial_cite_ds", run_id=run_id, step=1, model_id="xgboost",
                             params={"n_estimators": 111}, status="SUCCESS",
                             metric_name="accuracy", metric_value=0.61)
        store.write_trial(trial, dec)
        rows = store._run(
            "MATCH (d:Decision {id: 'dec_cite_ds'})-[:CITES]->(t) WHERE t.id = $id RETURN count(t)",
            {"id": ds_id})
        check("a Dataset citation produces a CITES edge", int(rows[0][0]) == 1, str(int(rows[0][0])))
        why = store.explain_trial(run_id, trial.trial_id)
        check("Dataset citation resolves as evidence",
              any(e.id == ds_id for e in why.evidence), str([e.kind for e in why.evidence]))
        check("Dataset citation is in the highlight", ds_id in why.highlight)


def minors() -> None:
    print("\n== minors: seq, whitespace rationale, unknown task ==")
    logger = RunLogger(store=None)
    logger._store = _SilentStore()
    logger.append("r", "INFO", "execute", "one")
    logger.append("r", "INFO", "execute", "two")
    logger.clear("r")
    entry = logger.append("r", "INFO", "execute", "three")
    check("seq does not restart after clear()", entry.seq == 3, f"got {entry.seq}")

    try:
        catalog.available_metrics("regression")
        check("unknown task raises", False, "returned a list")
    except ValueError as exc:
        check("unknown task raises", "unknown task" in str(exc), str(exc)[:60])

    class Stub:
        is_available = True

        def complete(self, *a, **k):
            from app.services.llm import LLMResult

            return LLMResult(text='{"selected": "REAL", "why": "   "}', used=True)

        @staticmethod
        def extract_json(text, prefer_keys=None):
            from app.services.llm import LLMClient

            return LLMClient.extract_json(text, prefer_keys=prefer_keys)

    agent = LayaAgent(llm_client=Stub())
    run = RunState(run_id="r", dataset_id="d", mode="autonomous", status="RUNNING",
                   step=0, budget=3, metric_name="accuracy", model_ids=["xgboost"])
    from app.models.contracts import GraphSnapshot

    probe = LayaAgent(llm_client=Stub())
    candidates = probe._build_candidates(run, [], GraphSnapshot())
    agent.llm_client = type("S2", (), {
        "is_available": True,
        "complete": lambda self, *a, **k: __import__(
            "app.services.llm", fromlist=["LLMResult"]
        ).LLMResult(text='{"selected": "%s", "why": "   "}' % candidates[0].candidate_id, used=True),
        "extract_json": staticmethod(lambda text, prefer_keys=None: __import__(
            "app.services.llm", fromlist=["LLMClient"]
        ).LLMClient.extract_json(text, prefer_keys=prefer_keys)),
    })()
    proposal = agent.propose(run, GraphSnapshot())
    check("whitespace-only rationale falls back", len(proposal.why.strip()) > 10,
          repr(proposal.why[:40]))


class _SilentStore:
    """A graph sink that accepts writes and returns nothing."""

    def write_log(self, entry):
        return None


def main() -> int:
    for step in (c1_wrong_typed_params, c2_nan_metric, c3_best_trials_groups, c4_no_duplicate_nodes,
                 c5_degenerate_target, m1_identifier, m2_leakage, m3_m4_scoping, m5_laya_valid_configs,
                 m6_target_hints, m7_split_hash, m8_m9_m10_graph_edges, minors):
        try:
            step()
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"{step.__name__} raised")

    print(f"\n{'ALL REGRESSIONS PASSED' if not FAILURES else f'{len(FAILURES)} FAILURE(S): ' + '; '.join(FAILURES)}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())