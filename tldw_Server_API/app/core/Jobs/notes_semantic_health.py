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


def validate_notes_semantic_health_checkpoint(
    *,
    after_owner_id: object,
    after_dataset_id: object,
    totals_json: str,
) -> tuple[dict[str, object], ...]:
    """Validate one coherent reset or in-progress health sweep checkpoint."""

    totals = parse_notes_semantic_health_totals(totals_json)
    if after_owner_id is None:
        if after_dataset_id is not None or totals:
            raise ValueError("semantic health initial checkpoint must be empty")
        return totals
    if isinstance(after_owner_id, bool) or not isinstance(after_owner_id, int) or after_owner_id <= 0:
        raise ValueError("semantic health owner id must be positive")
    if not totals:
        raise ValueError("semantic health in-progress checkpoint requires totals")
    if after_dataset_id is not None and (
        not isinstance(after_dataset_id, str) or not after_dataset_id or len(after_dataset_id.encode("utf-8")) > 256
    ):
        raise ValueError("semantic health dataset cursor must be bounded")
    return totals
