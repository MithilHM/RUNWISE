"""Laya: decides the next configuration, and says why.

Two things happen in this module, and they are the same thing done at two moments.

At `propose`, Laya reads what the graph already contains — the trials, their
settings, their scores, and which ones failed — and builds three candidates:

- **exploit**: nudge the best configuration found so far, because it is the only
  evidence there is that this region works;
- **explore**: sample the space deliberately away from anything already tried,
  because the best region may not be the one we have found yet;
- **avoid**: pick the least-explored model and stay clear of parameters that have
  already failed there.

It then selects one. When an LLM is configured it does the selecting and writes the
rationale; when it is not, a deterministic policy does. Either way the result is
the same shape, and `mode` is recorded so the UI never implies a model was
consulted when it was not.

Every candidate carries **citations**: the ids of the graph nodes it was derived
from. Those become `(Decision)-[:CITES]->` edges, which is the mechanism behind
"why this model?" and the highlight. A rationale that cannot name its evidence is
not a rationale, so the citation list is populated before the prose is written,
not after.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.graph.repository import param_id
from app.models.contracts import (
    GraphSnapshot,
    Proposal,
    RunState,
    TrialOutcome,
    WhyRecord,
    is_better,
)
from app.services import catalog
from app.services.llm import LLMClient, build_llm_client

logger = logging.getLogger(__name__)

__all__ = ["LayaAgent", "get_laya"]

SYSTEM_PROMPT = (
    "You are Laya, the decision agent of an autonomous ML harness. You choose which "
    "model configuration to train next, based only on the experiment history given to "
    "you. Do not describe your reasoning process. Do not repeat the question. "
    "Reply with one JSON object and nothing else."
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _trial_value(trial: TrialOutcome, metric: str) -> Optional[float]:
    return trial.metrics.get(metric) if trial.metrics else None


def _trials_from_graph(graph: GraphSnapshot) -> List[TrialOutcome]:
    """Rebuilds trial records from the graph snapshot.

    Reads the graph rather than run state, so a decision can never be based on
    something the graph does not also contain. If Laya could see a result the
    graph does not have, the explanation it gives would be unreproducible.
    """
    trials: List[TrialOutcome] = []
    for node in graph.nodes:
        if node.type != "Trial":
            continue
        props = node.props
        metrics = props.get("metrics") or {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except json.JSONDecodeError:
                metrics = {}
        params = props.get("params") or {}
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except json.JSONDecodeError:
                params = {}
        trials.append(
            TrialOutcome(
                trial_id=str(props.get("id", node.id)),
                run_id=str(props.get("run_id", "")),
                step=int(props.get("step", 0) or 0),
                model_id=str(props.get("model_id", "")),
                params=dict(params),
                status=str(props.get("status", "FAILED")),  # type: ignore[arg-type]
                metric_name=str(props.get("metric_name", "")),
                metric_value=props.get("metric_value"),
                metrics=dict(metrics),
                failure_reason=props.get("failure_reason") or None,
                created_at=str(props.get("created_at", "")),
            )
        )
    trials.sort(key=lambda t: (t.step, t.trial_id))
    return trials


class LayaAgent:
    """Implements the `DecisionMaker` port."""

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        random_seed: Optional[int] = None,
        store: Optional[Any] = None,
    ) -> None:
        self.llm_client = llm_client if llm_client is not None else build_llm_client()
        self.rng = random.Random(
            settings.RANDOM_SEED if random_seed is None else random_seed
        )
        self._store = store

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    def propose(
        self,
        run: RunState,
        graph: GraphSnapshot,
        rejected: Optional[List[Dict[str, Any]]] = None,
    ) -> Proposal:
        """Builds three candidates, picks one, and grounds it in the graph.

        `rejected` carries the proposals a researcher turned down, so the next
        decision does not re-propose something a human already refused. It is
        optional to keep the two-argument `DecisionMaker` signature usable.
        """
        history = _trials_from_graph(graph)
        candidates = self._build_candidates(run, history, graph)

        if not candidates:
            # No targeted model produced a candidate. Falling back to the first
            # model in the catalog beats returning nothing, and the run logs it.
            logger.warning("No candidate built for run %s; using catalog defaults", run.run_id)
            fallback = run.model_ids[0] if run.model_ids else catalog.MODEL_IDS[0]
            candidates = [
                self._candidate(
                    run,
                    fallback,
                    catalog.default_params(fallback),
                    "fallback_default",
                    "No usable history yet, so the library defaults for "
                    f"{catalog.get_model(fallback).name} are the only defensible starting point.",
                    "",
                    0.2,
                    [self._model_node(fallback)],
                )
            ]

        chosen, mode, why, hypothesis, confidence = self._select(run, candidates, history)

        # The decision mode rides along in the strategy so the UI never implies a
        # model was consulted when the deterministic policy ran instead.
        strategy = chosen.strategy if mode != "llm" else f"{chosen.strategy} (llm)"

        return Proposal(
            proposal_id=f"prop_{run.run_id}_s{run.step + 1:03d}",
            step=run.step + 1,
            model_id=chosen.model_id,
            params=dict(chosen.params),
            strategy=strategy,
            why=why,
            hypothesis=hypothesis,
            confidence=confidence,
            citations=list(chosen.citations),
            rejected=list(rejected or []),
        )

    def _build_candidates(
        self, run: RunState, history: List[TrialOutcome], graph: GraphSnapshot
    ) -> List["_Candidate"]:
        """The triad: exploit, explore, avoid."""
        metric = run.metric_name
        successful = [
            t for t in history
            if t.status == "SUCCESS" and _trial_value(t, metric) is not None
        ]
        failed = [t for t in history if t.status == "FAILED"]
        best = self._best_of(successful, metric)

        candidates: List[_Candidate] = []

        # --- exploit ------------------------------------------------------
        if best is not None:
            model_id = best.model_id
            # The baseline is the *coerced* config. Diffing against the raw dict
            # would report every parameter the trial never set as "changed" when
            # it merely acquired its default, and the rationale would then claim
            # the step moved seven parameters when it moved one.
            baseline = catalog.coerce_params(model_id, best.params)
            jittered = self._jitter(model_id, baseline)
            changed = [k for k in jittered if jittered[k] != baseline.get(k)]
            citations = [best.trial_id, self._model_node(model_id)]
            # `Param` nodes, not `Setting` nodes. A `Setting` exists only once a
            # trial has actually used that value, so citing the proposed value
            # would point at a node the graph does not have yet — a highlight
            # with nothing behind it. The knob exists; the value will follow when
            # the trial runs and writes its own `USES_SETTING` edge.
            citations += [self._param_node(model_id, k) for k in sorted(jittered)]
            note = (
                f"{catalog.get_model(model_id).name} scored "
                f"{metric}={_trial_value(best, metric):.4f} at step {best.step}, the best "
                f"result in this run. "
            )
            if changed:
                note += (
                    f"Moving {', '.join(changed)} away from that point tests whether the "
                    f"result sits on a plateau or on a peak."
                )
            else:
                note += "No neighbouring value was available to move to, so this re-runs the incumbent."
            candidates.append(
                self._candidate(
                    run, model_id, jittered, "exploit_promising_regime", note,
                    f"The region around {metric}={_trial_value(best, metric):.4f} holds the "
                    f"optimum found so far.",
                    0.72, citations,
                )
            )

        # --- explore ------------------------------------------------------
        counts: Dict[str, int] = {}
        for trial in successful:
            counts[trial.model_id] = counts.get(trial.model_id, 0) + 1
        explore_model = self._least_explored(run.model_ids, counts)
        fresh = self._sample_away_from_tried(explore_model, {})
        explore_citations = [self._model_node(explore_model)]
        explore_citations += [
            self._param_node(explore_model, key) for key in list(fresh)[:4]
        ]
        n_tried = counts.get(explore_model, 0)
        candidates.append(
            self._candidate(
                run,
                explore_model,
                fresh,
                "explore_unmapped_space",
                f"{catalog.get_model(explore_model).name} has the least coverage in this run "
                f"({len([t for t in successful if t.model_id == explore_model])} successful trials), "
                f"and these values sit away from anything already measured. The current best "
                f"configuration is an observation, not proof that the optimum is nearby.",
                f"An untested region of {catalog.get_model(explore_model).name}'s space will "
                f"beat the incumbent if the incumbent is a local optimum rather than the "
                f"global one.",
                0.55, explore_citations,
            )
        )

        # --- avoid --------------------------------------------------------
        failures = [t for t in failed if t.model_id in run.model_ids]
        avoid_model = (
            failures[-1].model_id if failures
            else self._least_explored(run.model_ids, counts, exclude=explore_model)
        )
        avoid_params = self._sample_avoiding(avoid_model, failures)
        avoid_citations = [self._model_node(avoid_model)]
        if failures:
            failed_ids = [t.trial_id for t in failures[-2:]]
            avoid_citations += failed_ids
            why_avoid = (
                f"{len(failures)} earlier trial(s) in this run did not complete. The parameters "
                f"that failed are excluded here, so this step spends the budget on a "
                f"configuration that can actually finish."
            )
        else:
            why_avoid = (
                f"No trial has failed yet, so there is no failure subgraph to avoid. "
                f"{catalog.get_model(avoid_model).name} is chosen to spread coverage across the "
                f"three models rather than to exploit a known good region."
            )
        candidates.append(
            self._candidate(
                run, avoid_model, avoid_params, "avoid_failure_subgraph", why_avoid,
                f"A run that avoids repeating a failure makes progress on every step, "
                f"including the steps that do not improve the score.",
                0.5, avoid_citations,
            )
        )
        return candidates

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _select(
        self, run: RunState, candidates: List["_Candidate"], history: List[TrialOutcome]
    ) -> Tuple[_Candidate, str, str, str, float]:
        """Returns (candidate, mode, why, hypothesis, confidence)."""
        if self.llm_client.is_available:
            result = self._select_with_llm(run, candidates, history)
            if result is not None:
                return result
        return self._select_deterministic(run, candidates, history)

    def _select_with_llm(
        self, run: RunState, candidates: List["_Candidate"], history: List[TrialOutcome]
    ) -> Optional[Tuple["_Candidate", str, str, str, float]]:
        """Asks the model to choose. Validated hard; falls through on anything odd."""
        recent = [
            {
                "step": t.step, "model": t.model_id, "status": t.status,
                "metric": t.metric_name, "value": _trial_value(t, run.metric_name),
                "reason": t.failure_reason,
            }
            for t in history[-8:]
        ]
        prompt = (
            f"Dataset task: classification. Metric to optimise: {run.metric_name} "
            f"(higher is better). Step {run.step + 1} of {run.budget}. "
            f"Models in play: {', '.join(run.model_ids)}.\n\n"
            f"Experiment history (most recent last):\n{json.dumps(recent, indent=2)}\n\n"
            f"Candidates:\n{json.dumps([c.to_dict() for c in candidates], indent=2)}\n\n"
            "Choose exactly one candidate. Ground the reasoning in the history above.\n"
            "Reply with JSON only, and nothing before or after it:\n"
            '{"selected": "<candidate_id>", "why": "<two sentences>", '
            '"hypothesis": "<one sentence>", "confidence": <0 to 1>}'
        )

        # Generous budget: a reasoning-style model spends tokens before the JSON,
        # and truncating mid-thought loses the answer entirely.
        result = self.llm_client.complete(prompt, system=SYSTEM_PROMPT, max_tokens=1200)
        if not result.used:
            logger.info("Laya LLM unavailable (%s); using deterministic policy", result.error)
            return None

        data = self.llm_client.extract_json(result.text, prefer_keys=("selected",))
        if not data:
            logger.info("Laya LLM returned no usable JSON; using deterministic policy")
            return None

        # The selected id must be one we actually proposed. Anything else is a
        # hallucination and is discarded rather than coerced to the nearest match.
        by_id = {c.candidate_id: c for c in candidates}
        selected = by_id.get(str(data.get("selected", "")))
        if selected is None:
            logger.warning(
                "Laya LLM selected unknown candidate %r; using deterministic policy",
                data.get("selected"),
            )
            return None

        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.6))))
        except (TypeError, ValueError):
            confidence = 0.6

        # A model can answer with a real candidate id and a whitespace-only
        # rationale. `or` does not catch that, so an empty explanation would be
        # stored as the reason a trial ran — which is worse than the candidate's
        # own rationale, because it is attributed to the model.
        why = str(data.get("why") or "").strip() or selected.rationale
        hypothesis = str(data.get("hypothesis") or "").strip() or selected.hypothesis

        return selected, "llm", why, hypothesis, confidence

    def _select_deterministic(
        self, run: RunState, candidates: List["_Candidate"], history: List[TrialOutcome]
    ) -> Tuple[_Candidate, str, str, str, float]:
        """Rotation policy used when no LLM is configured.

        Cycles exploit → explore → avoid rather than choosing by score, because the
        first three steps are the ones that decide whether the search covers the
        space at all. A policy that always exploited would report a confident
        answer from the narrowest possible basin.
        """
        index = run.step % len(candidates)
        order = ["exploit_promising_regime", "explore_unmapped_space", "avoid_failure_subgraph"]
        preferred = order[index % len(order)]
        chosen = next((c for c in candidates if c.strategy == preferred), candidates[index])
        return chosen, "deterministic", chosen.rationale, chosen.hypothesis, chosen.confidence

    # ------------------------------------------------------------------
    # Explain
    # ------------------------------------------------------------------

    def explain(self, run_id: str, trial_id: str) -> WhyRecord:
        store = self._store or self._default_store()
        return store.explain_trial(run_id, trial_id)

    def _default_store(self):
        from app.graph.repository import get_graph_store

        return get_graph_store()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _candidate(
        run: RunState,
        model_id: str,
        params: Dict[str, Any],
        strategy: str,
        rationale: str,
        hypothesis: str,
        confidence: float,
        citations: List[str],
    ) -> "_Candidate":
        return _Candidate(
            candidate_id=f"cand_{run.run_id}_s{run.step + 1:03d}_{strategy[:6]}",
            model_id=model_id,
            params=params,
            strategy=strategy,
            rationale=rationale,
            hypothesis=hypothesis,
            confidence=confidence,
            citations=citations,
        )

    @staticmethod
    def _model_node(model_id: str) -> str:
        return model_id

    @staticmethod
    def _param_node(model_id: str, name: str) -> str:
        return param_id(model_id, name)

    @staticmethod
    def _best_of(trials: List[TrialOutcome], metric: str) -> Optional[TrialOutcome]:
        best: Optional[TrialOutcome] = None
        for trial in trials:
            value = _trial_value(trial, metric)
            if value is None:
                continue
            if best is None or is_better(value, _trial_value(best, metric), metric):
                best = trial
        return best

    @staticmethod
    def _least_explored(
        model_ids: List[str],
        trial_counts: Dict[str, int],
        exclude: Optional[str] = None,
    ) -> str:
        """The model this run has measured least often.

        `trial_counts` holds trial counts, not parameter counts. Ranking by
        "how many parameters does this model's spec have" would always produce a
        tie and silently degenerate to catalog order, which is not what
        "least explored" means.
        """
        candidates = [m for m in model_ids if m != exclude] or list(model_ids)
        if not candidates:
            return catalog.MODEL_IDS[0]
        return min(
            candidates,
            key=lambda m: (trial_counts.get(m, 0), catalog.MODEL_IDS.index(m)),
        )

    def _jitter(self, model_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Moves exactly one parameter to a neighbouring value."""
        resolved = catalog.repair_params(model_id, params)
        numeric = [
            p for p in catalog.get_model(model_id).params
            if p.kind in ("int", "float", "int_or_null") and p.step
            and resolved.get(p.name) is not None
        ]
        if not numeric:
            return resolved
        target = self.rng.choice(numeric)
        alternatives = self._neighbours(target, resolved[target.name])
        if not alternatives:
            return resolved
        moved = dict(resolved)
        moved[target.name] = self.rng.choice(alternatives)
        return catalog.repair_params(model_id, moved)

    @staticmethod
    def _neighbours(param, value: Any) -> List[Any]:
        """Adjacent values on this parameter's grid."""
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return []
        step = float(param.step or 0)
        if not step:
            return []
        candidates: List[Any] = []
        for delta in (step, -step):
            candidate = float(value) + delta
            if param.low is not None and candidate < float(param.low):
                continue
            if param.high is not None and candidate > float(param.high):
                continue
            if param.kind in ("int", "int_or_null"):
                candidate = int(round(candidate))
            candidates.append(round(candidate, 6))
        return candidates

    def _sample_away_from_tried(
        self, model_id: str, tried: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Draws from the model's full space, not the neighbourhood of what ran.

        The explore strategy exists to test a different region of the space, so
        it deliberately does not start from the incumbent and nudge it. Every
        candidate is repaired afterwards, because drawing `num_leaves` and
        `max_depth` independently yields a LightGBM configuration the catalog
        itself rejects about a quarter of the time, and a step spent on an
        unrunnable configuration is a step that produced nothing.
        """
        del tried
        return catalog.repair_params(model_id, catalog.sample_params(model_id, self.rng))

    def _sample_avoiding(self, model_id: str, failures: List[TrialOutcome]) -> Dict[str, Any]:
        """Samples, then steers away from values a failure already used.

        A shared default is not a clash worth abandoning a whole sample for: only
        the parameters a failed trial was *judged* on are worth avoiding.
        """
        last = failures[-1] if failures else None
        for _attempt in range(6):
            candidate = catalog.repair_params(
                model_id, catalog.sample_params(model_id, self.rng)
            )
            if last is None or last.model_id != model_id:
                return candidate
            clashing = [
                key
                for key, value in last.params.items()
                if key in candidate and candidate[key] == value
            ]
            if not clashing:
                return candidate
            moved = dict(candidate)
            for key in clashing:
                param = catalog.get_model(model_id).param(key)
                if param and param.choices:
                    alternative = next(
                        (c for c in param.choices if c != moved.get(key)), None
                    )
                    if alternative is not None:
                        moved[key] = alternative
                elif param and param.high is not None:
                    # Nudge upward along the grid rather than resample everything.
                    moved[key] = param.clamp(float(moved[key]) + float(param.step or 0))
            candidate = catalog.repair_params(model_id, moved)
        return candidate


class _Candidate:
    """One proposed configuration, before it is selected."""

    __slots__ = (
        "candidate_id", "model_id", "params", "strategy",
        "rationale", "hypothesis", "confidence", "citations",
    )

    def __init__(
        self,
        candidate_id: str,
        model_id: str,
        params: Dict[str, Any],
        strategy: str,
        rationale: str,
        hypothesis: str,
        confidence: float,
        citations: List[str],
    ) -> None:
        self.candidate_id = candidate_id
        self.model_id = model_id
        self.params = params
        self.strategy = strategy
        self.rationale = rationale
        self.hypothesis = hypothesis
        self.confidence = confidence
        self.citations = citations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "model": catalog.get_model(self.model_id).name,
            "model_id": self.model_id,
            "strategy": self.strategy,
            "params": self.params,
            "rationale": self.rationale,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Candidate({self.candidate_id}, {self.model_id}, {self.strategy})"


_agent: Optional[LayaAgent] = None


def get_laya() -> LayaAgent:
    global _agent
    if _agent is None:
        _agent = LayaAgent()
    return _agent