"""Store contracts and in-memory store for MCP tool-use reporting."""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import datetime, timezone
from typing import Protocol

from mcp_unified.tool_use_reporting.models import (
    ToolUseEvent,
    ToolUseEventExportFormat,
    ToolUseEventQuery,
)


class ToolUseEventStore(Protocol):
    """Async store contract for metadata-only tool-use events."""

    async def append_event(self, event: ToolUseEvent) -> None:
        """Append one immutable tool-use event."""

    async def query_events(self, query: ToolUseEventQuery) -> list[ToolUseEvent]:
        """Return events matching a bounded query, newest first."""

    async def delete_events_older_than(self, cutoff: datetime | int) -> int:
        """Delete events older than a UTC datetime or epoch microsecond cutoff."""

    async def delete_events_over_limit(self, max_events: int) -> int:
        """Delete oldest events so at most max_events remain."""

    async def export_events(
        self,
        query: ToolUseEventQuery,
        *,
        format: ToolUseEventExportFormat,
    ) -> str:
        """Export events matching a bounded query."""


def cutoff_epoch_us(cutoff: datetime | int) -> int:
    """Return a UTC epoch microsecond cutoff."""

    if isinstance(cutoff, int):
        return cutoff
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("cutoff must be timezone-aware")
    utc_cutoff = cutoff.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_cutoff - epoch
    return (((delta.days * 86_400) + delta.seconds) * 1_000_000) + delta.microseconds


def encode_event_cursor(event: ToolUseEvent) -> str:
    """Encode a newest-first pagination cursor for an event."""

    payload = json.dumps(
        [event.created_at_epoch_us, event.event_id],
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def decode_event_cursor(cursor: str | None) -> tuple[int, str] | None:
    """Decode an event cursor, returning None for malformed values."""

    if not cursor:
        return None
    try:
        padded = cursor + ("=" * (-len(cursor) % 4))
        raw = base64.urlsafe_b64decode(padded.encode("ascii"))
        epoch_us, event_id = json.loads(raw.decode("utf-8"))
        return int(epoch_us), str(event_id)
    except (ValueError, TypeError, json.JSONDecodeError):
        return None


def event_is_after_cursor(event: ToolUseEvent, cursor: tuple[int, str] | None) -> bool:
    """Return whether an event belongs after a newest-first cursor."""

    if cursor is None:
        return True
    return (event.created_at_epoch_us, event.event_id) < cursor


def event_matches_query(event: ToolUseEvent, query: ToolUseEventQuery) -> bool:
    """Return whether an event satisfies scalar query filters."""

    if not event_is_after_cursor(event, decode_event_cursor(query.cursor)):
        return False
    if query.created_at_epoch_us_gte is not None and event.created_at_epoch_us < query.created_at_epoch_us_gte:
        return False
    if query.created_at_epoch_us_lt is not None and event.created_at_epoch_us >= query.created_at_epoch_us_lt:
        return False
    for field_name in (
        "runtime_surface",
        "requested_tool_name",
        "effective_tool_name",
        "profile_id",
        "mode_id",
        "model_id",
        "tool_prompt_id",
        "status",
    ):
        expected = getattr(query, field_name)
        if expected is not None and getattr(event, field_name) != expected:
            return False
    return True


class InMemoryToolUseEventStore:
    """In-memory tool-use event store for tests and ephemeral gateways."""

    def __init__(self) -> None:
        self._events: list[ToolUseEvent] = []
        self._lock = asyncio.Lock()

    async def append_event(self, event: ToolUseEvent) -> None:
        """Append a copy-isolated event."""

        async with self._lock:
            self._events.append(event.model_copy(deep=True))

    async def query_events(self, query: ToolUseEventQuery) -> list[ToolUseEvent]:
        """Return copy-isolated events matching a bounded query, newest first."""

        async with self._lock:
            rows = [event for event in self._events if event_matches_query(event, query)]
        rows.sort(key=lambda event: (event.created_at_epoch_us, event.event_id), reverse=True)
        return [event.model_copy(deep=True) for event in rows[: query.limit]]

    async def delete_events_older_than(self, cutoff: datetime | int) -> int:
        """Delete events older than the cutoff and return the number removed."""

        cutoff_us = cutoff_epoch_us(cutoff)
        async with self._lock:
            before = len(self._events)
            self._events = [event for event in self._events if event.created_at_epoch_us >= cutoff_us]
            return before - len(self._events)

    async def delete_events_over_limit(self, max_events: int) -> int:
        """Keep the newest max_events and delete the rest."""

        max_events = max(0, int(max_events))
        async with self._lock:
            rows = sorted(
                self._events,
                key=lambda event: (event.created_at_epoch_us, event.event_id),
                reverse=True,
            )
            self._events = rows[:max_events]
            return len(rows) - len(self._events)

    async def export_events(
        self,
        query: ToolUseEventQuery,
        *,
        format: ToolUseEventExportFormat,
    ) -> str:
        """Export events matching the query as JSON or JSON Lines."""

        rows = await self.query_events(query)
        if format == "jsonl":
            return "\n".join(event.model_dump_json() for event in rows)
        return json.dumps(
            [event.model_dump(mode="json") for event in rows],
            separators=(",", ":"),
        )
