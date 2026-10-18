"""Run logging with three sinks and one call.

`append` writes to all three of: an in-memory ring, the graph, and every live SSE
subscriber. Callers never touch any of them directly, which is what stops a log
line from existing on screen but not in the graph — the reason a researcher cannot
answer "when did that happen?" after reloading.

The ring buffer is capped so a long run cannot grow without bound. When it drops
the oldest entries, they are not lost: `history` falls through to the graph, which
is the durable sink.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.graph.repository import FalkorDBGraphStore
from app.models.contracts import LogEntry, LogEvent

logger = logging.getLogger(__name__)

__all__ = ["RunLogger", "get_logger_"]

# Enough history to cover any realistic run without unbounded growth. Anything
# older is still in the graph; only the hot path is capped.
RING_CAPACITY = 5000

_STAGES = {"ingest", "propose", "decide", "execute", "mutate", "done"}
_LEVELS = {"INFO", "WARN", "ERROR"}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLogger:
    """Implements the `RunLogger` port."""

    def __init__(self, store: Optional[FalkorDBGraphStore] = None) -> None:
        self._store = store
        self._ring: Dict[str, List[LogEntry]] = {}
        self._seq: Dict[str, int] = {}
        self._subscribers: Dict[str, List["asyncio.Queue[LogEvent]"]] = {}

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def append(
        self,
        run_id: str,
        level: str,
        stage: str,
        message: str,
        step: int = 0,
        meta: Optional[Dict[str, Any]] = None,
    ) -> LogEntry:
        """Records one line everywhere. Never raises.

        A logging call must not be able to fail a training run, so graph errors
        are caught and reported as a warning. Losing a log line is bad; losing the
        trial because of a log line is worse.
        """
        if stage not in _STAGES:
            logger.debug("Coercing unknown log stage %r to 'execute'", stage)
            stage = "execute"
        if level not in _LEVELS:
            level = "INFO"

        seq = self._seq.get(run_id, 0) + 1
        self._seq[run_id] = seq

        entry = LogEntry(
            seq=seq,
            run_id=run_id,
            step=step,
            level=level,  # type: ignore[arg-type]
            stage=stage,  # type: ignore[arg-type]
            message=message,
            meta=dict(meta or {}),
            ts=_utcnow(),
        )

        ring = self._ring.setdefault(run_id, [])
        ring.append(entry)
        if len(ring) > RING_CAPACITY:
            del ring[: len(ring) - RING_CAPACITY]

        try:
            self._graph_sink().write_log(entry)
        except Exception as exc:  # noqa: BLE001 - see docstring
            logger.warning("Could not persist log %s/%d: %s", run_id, seq, exc)

        self._broadcast(run_id, LogEvent(type="log", payload=entry.model_dump()))
        return entry

    def publish(self, run_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        """Broadcasts a non-log notification. Not persisted as a line."""
        self._broadcast(run_id, LogEvent(type=event_type, payload=payload))  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def history(self, run_id: str, after_seq: int = 0) -> List[LogEntry]:
        """Entries after `after_seq`, preferring memory and falling back to graph.

        The ring only holds recent lines, so a long-past sequence number has to be
        answered from the graph. Going to the graph unconditionally would be
        simpler and would add a round trip to every log poll for nothing.
        """
        ring = self._ring.get(run_id, [])
        if ring:
            first_in_ring = ring[0].seq
            if after_seq + 1 >= first_in_ring:
                return [entry for entry in ring if entry.seq > after_seq]

        try:
            return self._graph_sink().read_logs(run_id, after_seq)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read logs for %s from the graph: %s", run_id, exc)
            return [entry for entry in ring if entry.seq > after_seq]

    def next_seq(self, run_id: str) -> int:
        """The sequence number the next append will use."""
        return self._seq.get(run_id, 0) + 1

    def clear(self, run_id: str) -> None:
        """Drops the in-memory ring for a run.

        The sequence counter is deliberately **not** reset. Log ids in the graph
        are `log_<run_id>_<seq>`, so restarting at 1 would `MERGE` over the
        existing entries and silently destroy the run's earlier history.
        """
        self._ring.pop(run_id, None)

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(self, run_id: str) -> "asyncio.Queue[LogEvent]":
        """A queue receiving every future event for this run."""
        queue: "asyncio.Queue[LogEvent]" = asyncio.Queue()
        self._subscribers.setdefault(run_id, []).append(queue)
        return queue

    def unsubscribe(self, run_id: str, queue: "asyncio.Queue[LogEvent]") -> None:
        queues = self._subscribers.get(run_id)
        if queues and queue in queues:
            queues.remove(queue)
        if queues is not None and not queues:
            self._subscribers.pop(run_id, None)

    def subscriber_count(self, run_id: str) -> int:
        return len(self._subscribers.get(run_id, []))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _broadcast(self, run_id: str, event: LogEvent) -> None:
        for queue in list(self._subscribers.get(run_id, [])):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:  # pragma: no cover - unbounded queue
                logger.warning("Dropping event for a full subscriber queue on %s", run_id)

    def _graph_sink(self) -> FalkorDBGraphStore:
        if self._store is None:
            from app.graph.repository import get_graph_store

            self._store = get_graph_store()
        return self._store


_logger: Optional[RunLogger] = None


def get_logger_() -> RunLogger:
    """Process-wide logger. Subscriber state has to be shared across requests."""
    global _logger
    if _logger is None:
        _logger = RunLogger()
    return _logger