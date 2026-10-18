"""Probe: does the configured LLM answer, and does its JSON survive validation?"""

from __future__ import annotations

import logging

from app.services.llm import build_llm_client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

client = build_llm_client()
print("available:", client.is_available, "| model:", client.model, "| base:", client.base_url)

prompt = (
    "Choose one candidate. Candidates:\n"
    '[{"candidate_id":"cand_a","params":{"max_depth":4}},'
    '{"candidate_id":"cand_b","params":{"max_depth":9}}]\n\n'
    'Reply with JSON only: {"selected": "<candidate_id>", "why": "<one sentence>", '
    '"hypothesis": "<one sentence>", "confidence": <0 to 1>}'
)

result = client.complete(prompt, system=(
    "You output one JSON object and nothing else. "
    "Do not describe your reasoning and do not repeat the question."
), max_tokens=1200)
print("used:", result.used, "| error:", result.error)
print("length:", len(result.text or ""))
print("tail:", repr((result.text or "")[-300:]))
parsed = client.extract_json(result.text, prefer_keys=("selected",))
print("parsed:", parsed)
if isinstance(parsed, dict):
    print("selected is a real candidate:", parsed.get("selected") in {"cand_a", "cand_b"})