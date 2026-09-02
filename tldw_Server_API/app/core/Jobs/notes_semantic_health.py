"""Shared validation for durable Notes semantic health sweep totals."""

from __future__ import annotations

import json
from datetime import datetime

NOTES_SEMANTIC_HEALTH_BACKENDS = frozenset({"chromadb", "pgvector", "unavailable"})
_COUNT_FIELDS = frozenset(
    {
        "indexed_notes",
        "excluded_notes",
        "failed_notes",
        "dirty_notes",
        "pending_notes",
        "stale_generations",
        "cleanup_backlog",
        "cleanup_retries",
    }
)
_FIELDS = _COUNT_FIELDS | {"backend", "oldest_cleanup_created_at"}


def parse_notes_semantic_health_totals(value: str) -> tuple[dict[str, object], ...]:
    """Return one exact, bounded payload accepted by persistence and readers."""

    if not isinstance(value, str) or len(value) > 65_536:
        raise ValueError("semantic health totals must be bounded JSON")
    try:
        payload = json.loads(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("semantic health totals must be valid JSON") from exc
    if not isinstance(payload, list) or len(payload) > len(NOTES_SEMANTIC_HEALTH_BACKENDS):
        raise ValueError("semantic health totals must contain bounded backends")
    backends: set[str] = set()
    validated: list[dict[str, object]] = []
    for item in payload:
        if not isinstance(item, dict) or set(item) != _FIELDS:
            raise ValueError("semantic health totals must contain exact fields")
        backend = item["backend"]
        if not isinstance(backend, str) or backend not in NOTES_SEMANTIC_HEALTH_BACKENDS:
            raise ValueError("semantic health totals contain an invalid backend")
        if backend in backends:
            raise ValueError("semantic health totals contain a duplicate backend")
        backends.add(backend)
        for field_name in _COUNT_FIELDS:
            count = item[field_name]
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("semantic health totals contain an invalid count")
        oldest = item["oldest_cleanup_created_at"]
        if oldest is not None:
            if not isinstance(oldest, str) or len(oldest) > 64:
                raise ValueError("semantic health totals contain an invalid timestamp")
            try:
                parsed = datetime.fromisoformat(oldest.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("semantic health totals contain an invalid timestamp") from exc
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError("semantic health totals contain an invalid timestamp")
        validated.append(dict(item))
    if payload and backends != NOTES_SEMANTIC_HEALTH_BACKENDS:
        raise ValueError("semantic health totals must contain every backend")
    return tuple(validated)
