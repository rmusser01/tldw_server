"""Aggregate reporting service for MCP tool-use events."""

from __future__ import annotations

from collections import Counter, defaultdict

from mcp_unified.tool_use_reporting.models import (
    MAX_EVENT_QUERY_LIMIT,
    ToolUseEvent,
    ToolUseReport,
    ToolUseReportQuery,
    ToolUseReportRow,
)
from mcp_unified.tool_use_reporting.store import ToolUseEventStore

_GROUP_FIELDS = {
    "profile": "profile_id",
    "tool_prompt": "tool_prompt_id",
    "model": "model_id",
    "tool": "effective_tool_name",
}


def _percentile(values: list[float], percentile: float) -> float | None:
    """Return a nearest-rank percentile for a sorted or unsorted list."""

    if not values:
        return None
    sorted_values = sorted(values)
    rank = int((len(sorted_values) - 1) * percentile + 0.5)
    index = max(0, min(len(sorted_values) - 1, rank))
    return sorted_values[index]


class ToolUseReportService:
    """Build bounded aggregate reports from a tool-use event store."""

    def __init__(self, store: ToolUseEventStore) -> None:
        self._store = store

    async def build_report(self, query: ToolUseReportQuery) -> ToolUseReport:
        """Build a bounded aggregate report."""

        fetch_limit = min(query.event_limit + 1, MAX_EVENT_QUERY_LIMIT)
        events = await self._store.query_events(query.to_event_query(limit=fetch_limit))
        truncated = len(events) > query.event_limit
        scanned = events[: query.event_limit]
        group_field = _GROUP_FIELDS[query.group_by]

        grouped: dict[str, list[ToolUseEvent]] = defaultdict(list)
        for event in scanned:
            group_key = getattr(event, group_field) or "unknown"
            grouped[group_key].append(event)

        rows = [
            self._build_row(
                group_key=group_key,
                events=group_events,
                top_reason_code_limit=query.top_reason_code_limit,
            )
            for group_key, group_events in grouped.items()
        ]
        rows.sort(key=lambda row: (-row.call_count, row.group_key))

        return ToolUseReport(
            rows=rows[: query.group_limit],
            events_scanned=len(scanned),
            event_limit=query.event_limit,
            truncated=truncated,
        )

    def _build_row(
        self,
        *,
        group_key: str,
        events: list[ToolUseEvent],
        top_reason_code_limit: int,
    ) -> ToolUseReportRow:
        """Build one aggregate row."""

        call_count = len(events)
        success_count = sum(1 for event in events if event.status == "success")
        reason_counts = Counter(event.reason_code for event in events if event.reason_code)
        durations = [event.duration_ms for event in events if event.duration_ms is not None]

        return ToolUseReportRow(
            group_key=group_key,
            call_count=call_count,
            tool_call_success_rate=success_count / call_count if call_count else 0.0,
            top_reason_codes=[
                {"reason_code": reason_code, "count": count}
                for reason_code, count in reason_counts.most_common(top_reason_code_limit)
            ],
            p50_duration_ms=_percentile(durations, 0.50),
            p95_duration_ms=_percentile(durations, 0.95),
        )
