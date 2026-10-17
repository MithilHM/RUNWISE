"""Pluggable LLM adapter with a deterministic fallback.

PRD 18 assigns this area: "Add a pluggable LLM adapter with a deterministic
fallback."

The product must work with no API key at all (PRD 15), and it must never imply
a model was consulted when none was. So every explanation path is built on the
same rule:

  * The LLM is a *prose* layer over facts the deterministic engine already
    computed. It never decides lineage, comparability, ranking or metrics.
  * Its output is only ever used after passing through the evidence ledger's
    citation check, so a hallucinated experiment ID fails loudly instead of
    reaching a researcher.
  * With no key configured, the deterministic template runs and the response
    carries ``mode="deterministic"`` and ``llm_used=False``.

Environment variables are declared here but consolidated by the integration owner into
``.env.example`` (PRD 19).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.models.contracts import AnswerMode


class LLMUnavailable(RuntimeError):
    """Raised when an LLM is required but none is configured or reachable."""


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    used: bool


class LLMClient(Protocol):
    """Minimal surface. Kept narrow so a provider can be swapped in one class."""

    @property
    def is_available(self) -> bool: ...

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 700) -> LLMResult: ...


class DeterministicLLMClient:
    """The no-key default. Reports itself unavailable so callers skip it."""

    @property
    def is_available(self) -> bool:
        return False

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 700) -> LLMResult:
        raise LLMUnavailable(
            "no LLM configured; the deterministic explanation path was used instead"
        )


class OpenAICompatibleLLMClient:
    """Any endpoint speaking the OpenAI chat-completions shape.

    Uses ``urllib`` from the standard library so this module adds no dependency
    to ``backend/pyproject.toml``, which the integration owner owns.
    """

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 20.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def complete(self, prompt: str, *, system: str | None = None,
                 max_tokens: int = 700) -> LLMResult:
        if not self._api_key:
            raise LLMUnavailable("RUNWISE_LLM_API_KEY is not set")

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = json.dumps(
            {
                "model": self._model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            raise LLMUnavailable(f"LLM request failed: {exc}") from exc

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable("LLM response had an unexpected shape") from exc

        return LLMResult(text=str(text).strip(), model=self._model, used=True)


def build_llm_client(env: Mapping[str, str] | None = None) -> LLMClient:
    """Construct the client described by the environment.

    Returns :class:`DeterministicLLMClient` whenever the key is absent or blank,
    which is the default demo path (PRD 15).
    """
    source = env if env is not None else os.environ
    api_key = (source.get("RUNWISE_LLM_API_KEY") or "").strip()
    if not api_key:
        return DeterministicLLMClient()

    model = (source.get("RUNWISE_LLM_MODEL") or "gpt-4o-mini").strip()
    base_url = (source.get("RUNWISE_LLM_BASE_URL")
                or "https://api.openai.com/v1").strip()
    return OpenAICompatibleLLMClient(api_key=api_key, model=model, base_url=base_url)


# ---------------------------------------------------------------------------
# Explanation layer
# ---------------------------------------------------------------------------

EXPLANATION_SYSTEM = """You explain ML experiment history to a researcher.

Hard rules:
- Use ONLY the records supplied in the user message. Never name an experiment, \
metric, dataset version or number that is not in that data.
- Cite supporting runs by their exact identifiers.
- Distinguish measured results from your own interpretation. Label anything that \
is your interpretation as an inference.
- If the data does not answer the question, say exactly what is missing.
- Do not invent a metric value, a runtime, a cost, or a causal explanation.
- Be concise and concrete. No preamble."""


_UUID = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)


def _identifier_tokens(text: str, allowed_ids: Sequence[str]) -> set[str]:
    """Every id-shaped token in ``text`` that is not a known id.

    Token shapes come from the ids we actually know about, so the check follows the
    graph's naming convention instead of assuming ``exp_``. UUIDs are also matched
    outright, since those have no prefix to derive.

    Known limitation: an id with no separator at all (``abc123``) has no prefix to
    search for, so a fabricated sibling of it would not be detected here. Graphs
    should use a separator; this is documented rather than papered over.
    """
    known = set(allowed_ids)
    prefixes = {
        token.split("_", 1)[0].split("-", 1)[0]
        for token in known
        if token
    }
    prefixes.discard("")

    found: set[str] = set()
    for prefix in prefixes:
        found.update(
            re.findall(rf"\b{re.escape(prefix)}[_-][A-Za-z0-9_\-]+", text)
        )
    found.update(_UUID.findall(text))

    # Anything already known is legitimate regardless of prefix matching.
    return found - known


_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _unsupported_numbers(text: str, facts: Mapping[str, Any]) -> set[str]:
    """Figures in the prose that are absent from the evidence.

    Only ``1`` is exempt: it is too common to be informative ("step 1", "1 of 3").
    Everything else, decimals included, must already appear in the facts payload.
    A rejection here falls back to the deterministic text, so the cost of a false
    positive is a plainer answer, not a wrong one.
    """
    try:
        serialised = json.dumps(facts, default=str)
    except (TypeError, ValueError):
        return set()

    allowed: set[str] = set()
    for match in _NUMBER.finditer(serialised):
        allowed.add(_normalise_number(match.group(0)))

    unsupported: set[str] = set()
    for match in _NUMBER.finditer(text):
        raw = match.group(0)
        if "." not in raw and abs(int(raw)) < 2:
            continue
        if _normalise_number(raw) in allowed:
            continue
        unsupported.add(raw)
    return unsupported


def _normalise_number(raw: str) -> str:
    """Compare figures by value, so ``0.911`` and ``0.9110`` are the same."""
    try:
        return f"{float(raw):.6g}"
    except ValueError:
        return raw


@dataclass
class ExplanationResult:
    """Prose plus the truthful record of how it was produced."""

    text: str
    mode: AnswerMode
    llm_used: bool
    model: str | None = None
    fallback_reason: str | None = None


class ExplanationWriter:
    """Turns structured engine output into prose.

    ``render`` produces the deterministic text. ``write`` tries the LLM first and
    falls back on any failure, so a bad key degrades the writing but never the
    numbers.
    """

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or DeterministicLLMClient()

    def write(
        self,
        *,
        question: str,
        facts: Mapping[str, Any],
        deterministic_text: str,
        allowed_ids: Sequence[str],
    ) -> ExplanationResult:
        """Return the best available explanation for ``facts``.

        ``allowed_ids`` is the citation whitelist. The LLM output is accepted only
        if it stays inside it, which keeps an external model from widening the
        evidence base.
        """
        if not self.client.is_available:
            return ExplanationResult(
                text=deterministic_text,
                mode=AnswerMode.DETERMINISTIC,
                llm_used=False,
                fallback_reason="no LLM configured; deterministic template used",
            )

        try:
            result = self.client.complete(
                self._build_prompt(question, facts),
                system=EXPLANATION_SYSTEM,
            )
        except LLMUnavailable as exc:
            return ExplanationResult(
                text=deterministic_text,
                mode=AnswerMode.DETERMINISTIC,
                llm_used=False,
                fallback_reason=str(exc),
            )

        # A client that reports it made no call must never be presented as one,
        # whatever text it returned (PRD 15).
        if not result.used:
            return ExplanationResult(
                text=deterministic_text,
                mode=AnswerMode.DETERMINISTIC,
                llm_used=False,
                model=result.model,
                fallback_reason=(
                    "the client returned no completed call; the deterministic "
                    "explanation was used instead"
                ),
            )

        rejection = self._rejection_reason(result.text, allowed_ids, facts)
        if rejection is not None:
            return ExplanationResult(
                text=deterministic_text,
                mode=AnswerMode.DETERMINISTIC,
                llm_used=False,
                model=result.model,
                fallback_reason=rejection,
            )

        return ExplanationResult(
            text=result.text,
            mode=AnswerMode.LLM,
            llm_used=True,
            model=result.model,
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _build_prompt(question: str, facts: Mapping[str, Any]) -> str:
        return (
            f"Researcher question:\n{question}\n\n"
            f"Retrieved experiment records (this is the entire evidence base):\n"
            f"{json.dumps(facts, indent=2, default=str)}\n\n"
            "Answer the question using only these records."
        )

    @classmethod
    def _rejection_reason(
        cls,
        text: str,
        allowed_ids: Sequence[str],
        facts: Mapping[str, Any],
    ) -> str | None:
        """Why generated prose must not be served, or ``None`` if it is fine.

        Two checks, both deliberately conservative:

        * identifiers. Any id-shaped token must be in ``allowed_ids``. Token
          shapes are derived from the allowed ids rather than hardcoded to
          ``exp_``, so a graph using ``run-0042`` is still checked, and UUIDs are
          matched outright. Ids with no separator cannot be pattern-matched and
          are a documented gap.
        * numbers. Every figure in the prose except ``1`` must already appear in
          the facts. A metric, runtime or count that is not in the evidence is a
          fabrication, which FR-3 and PRD 12 forbid outright.

        Both checks are lexical. They cannot catch a qualitative claim invented
        from supplied words; see ``ranking_rules.md`` section 5.
        """
        allowed = set(allowed_ids)
        referenced = set(_identifier_tokens(text, allowed))
        unknown = sorted(referenced - allowed)
        if unknown:
            return (
                "the model referenced identifiers outside the retrieved evidence "
                f"({', '.join(unknown)}); the deterministic explanation was used instead"
            )

        unsupported = _unsupported_numbers(text, facts)
        if unsupported:
            return (
                "the model stated figures that do not appear in the retrieved evidence "
                f"({', '.join(sorted(unsupported))}); the deterministic explanation was "
                "used instead"
            )

        return None