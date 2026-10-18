"""Minimal LLM client for Laya.

Optional by design. With no API key the harness still produces every decision,
every rationale and every citation — it just does it with a deterministic graph
policy instead of a language model, and it says which one ran.

The reason for the strictness below is that a language model's output is
untrusted input here. It proposes a candidate id and some prose; the id is
validated against the candidates that actually exist before anything runs, and a
parse failure falls through to the deterministic policy rather than propagating.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings

logger = logging.getLogger(__name__)

__all__ = ["LLMResult", "LLMClient", "build_llm_client"]


@dataclass
class LLMResult:
    text: str
    used: bool
    error: Optional[str] = None


class LLMClient:
    """Chat-completions client. Deliberately minimal."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else settings.RUNWISE_LLM_API_KEY
        self.model = model or settings.RUNWISE_LLM_MODEL
        self.base_url = (base_url or settings.RUNWISE_LLM_BASE_URL).rstrip("/")
        self.timeout = timeout or settings.LAYA_TIMEOUT_SECONDS
        self.last_error: Optional[str] = None

    @property
    def is_available(self) -> bool:
        return bool(self.api_key)

    def complete(
        self, prompt: str, system: Optional[str] = None, max_tokens: int = 500
    ) -> LLMResult:
        """One completion. Never raises; a failure is a result, not an exception."""
        if not self.is_available:
            return LLMResult(text="", used=False, error="no_api_key")

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": (
                ([{"role": "system", "content": system}] if system else [])
                + [{"role": "user", "content": prompt}]
            ),
            "max_tokens": max_tokens,
            "temperature": 0.2,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
            text = body["choices"][0]["message"]["content"]
            self.last_error = None
            return LLMResult(text=text or "", used=True)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                json.JSONDecodeError, KeyError, IndexError, OSError) as exc:
            self.last_error = str(exc)
            logger.warning("LLM call failed (%s): %s", self.model, exc)
            return LLMResult(text="", used=False, error=str(exc))

    @staticmethod
    def extract_json(
        text: str, prefer_keys: Optional[Tuple[str, ...]] = None
    ) -> Optional[Dict[str, Any]]:
        """Pulls the answer object out of a response that may be wrapped or narrated.

        Reasoning-style models often echo the prompt before answering, so the
        first `{` in the text is frequently the example they were shown rather
        than their own output. This scans for every balanced top-level object and
        prefers the last one that parses and carries the expected keys, falling
        back to the last that parses at all.
        """
        if not text:
            return None

        candidates: List[Dict[str, Any]] = []
        for block in LLMClient._balanced_objects(text):
            try:
                parsed = json.loads(block)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                candidates.append(parsed)

        if not candidates:
            return None
        if prefer_keys:
            wanted = set(prefer_keys)
            preferred = [c for c in candidates if wanted <= set(c)]
            if preferred:
                return preferred[-1]
        return candidates[-1]

    @staticmethod
    def _balanced_objects(text: str) -> List[str]:
        """Every balanced `{...}` span, brace-matched and quote-aware."""
        blocks: List[str] = []
        depth = 0
        start = -1
        in_string = False
        escaped = False

        for index, char in enumerate(text):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                if depth == 0:
                    start = index
                depth += 1
            elif char == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start >= 0:
                        blocks.append(text[start : index + 1])
                        start = -1
        return blocks


_client: Optional[LLMClient] = None


def build_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client