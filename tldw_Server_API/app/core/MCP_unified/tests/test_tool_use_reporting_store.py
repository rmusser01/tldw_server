"""Tests for MCP tool-use reporting stores and aggregate reports."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import mcp_unified.tool_use_reporting.recorder as recorder_module
import mcp_unified.tool_use_reporting.store as store_module
from mcp_unified.tool_use_reporting.models import (
    ToolUseEvent,
    ToolUseEventQuery,
    ToolUseReportQuery,
)
from mcp_unified.tool_use_reporting.reporting import ToolUseReportService
from mcp_unified.tool_use_reporting.sqlite import SQLiteToolUseEventStore
from mcp_unified.tool_use_reporting.store import (
    InMemoryToolUseEventStore,
    cutoff_epoch_us,
    encode_event_cursor,
)


StoreFactory = Callable[[Path], Any]


async def _close_store(store: Any) -> None:
    close = getattr(store, "aclose", None)
    if close is not None:
        await close()
        return
    close = getattr(store, "close", None)
    if close is not None:
        close()


class _FailingToolUseEventStore:
    async def append_event(self, _event: ToolUseEvent) -> None:
        raise RuntimeError("sink unavailable")


class _FailingToolUseRecorder:
    async def record_tool_use(self, _event: ToolUseEvent) -> None:
        raise RuntimeError("recorder unavailable")


@pytest.fixture(
    params=[
        lambda tmp_path: InMemoryToolUseEventStore(),
        lambda tmp_path: SQLiteToolUseEventStore(tmp_path / "tool-use.sqlite3"),
    ],
    ids=["memory", "sqlite"],
)
def store_factory(request: pytest.FixtureRequest) -> StoreFactory:
    return request.param


@pytest.mark.asyncio
async def test_store_queries_events_by_epoch_newest_first(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    store = store_factory(tmp_path)
    older = ToolUseEvent(
        created_at=datetime(2026, 6, 6, 10, 0, tzinfo=timezone.utc),
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )
    newer = ToolUseEvent(
        created_at=datetime(2026, 6, 6, 7, 30, tzinfo=timezone(timedelta(hours=-7))),
        runtime_surface="protocol",
        requested_tool_name="git.status",
        status="success",
    )

    try:
        await store.append_event(older)
        await store.append_event(newer)

        rows = await store.query_events(ToolUseEventQuery(limit=10))

        assert [row.event_id for row in rows] == [newer.event_id, older.event_id]
        assert rows[0] is not newer
    finally:
        await _close_store(store)


@pytest.mark.asyncio
async def test_store_filters_and_exports_jsonl(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    store = store_factory(tmp_path)
    try:
        await store.append_event(
            ToolUseEvent(
                runtime_surface="protocol",
                requested_tool_name="fs.read",
                profile_id="architect",
                status="success",
            )
        )
        await store.append_event(
            ToolUseEvent(
                runtime_surface="gateway",
                requested_tool_name="fs.write",
                profile_id="owner",
                status="denied",
            )
        )

        query = ToolUseEventQuery(profile_id="architect", limit=10)
        rows = await store.query_events(query)
        exported = await store.export_events(query, format="jsonl")
        exported_json = await store.export_events(query, format="json")

        assert [row.requested_tool_name for row in rows] == ["fs.read"]
        exported_rows = [json.loads(line) for line in exported.splitlines() if line.strip()]
        assert exported_rows[0]["profile_id"] == "architect"
        assert exported_json == json.dumps(
            [row.model_dump(mode="json") for row in rows],
            separators=(",", ":"),
        )
    finally:
        await _close_store(store)


@pytest.mark.asyncio
async def test_store_cursor_returns_next_page_after_cursor(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    store = store_factory(tmp_path)
    events = [
        ToolUseEvent(
            created_at=datetime(2026, 1, day, tzinfo=timezone.utc),
            runtime_surface="protocol",
            requested_tool_name=f"tool.{day}",
            status="success",
        )
        for day in range(1, 4)
    ]

    try:
        for event in events:
            await store.append_event(event)

        first_page = await store.query_events(ToolUseEventQuery(limit=1))
        second_page = await store.query_events(
            ToolUseEventQuery(
                limit=10,
                cursor=encode_event_cursor(first_page[0]),
            )
        )

        assert [row.requested_tool_name for row in first_page] == ["tool.3"]
        assert [row.requested_tool_name for row in second_page] == [
            "tool.2",
            "tool.1",
        ]
    finally:
        await _close_store(store)


@pytest.mark.asyncio
async def test_store_ignores_malformed_cursor(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    store = store_factory(tmp_path)
    try:
        await store.append_event(
            ToolUseEvent(
                runtime_surface="protocol",
                requested_tool_name="fs.read",
                status="success",
            )
        )

        rows = await store.query_events(ToolUseEventQuery(cursor="a", limit=10))

        assert [row.requested_tool_name for row in rows] == ["fs.read"]
    finally:
        await _close_store(store)


@pytest.mark.asyncio
async def test_recorder_failure_logs_safe_event_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[tuple[str, tuple[Any, ...]]] = []

    def fake_warning(message: str, *args: Any, **_kwargs: Any) -> None:
        captured.append((message, args))

    monkeypatch.setattr(recorder_module.logger, "warning", fake_warning)
    event = ToolUseEvent(
        runtime_surface="protocol",
        requested_tool_name="fs.read",
        status="success",
    )

    recorder = recorder_module.StoreBackedToolUseRecorder(_FailingToolUseEventStore())
    await recorder.record_tool_use(event)
    await recorder_module.record_tool_use_safely(_FailingToolUseRecorder(), event)

    assert len(captured) == 2
    rendered = [
        " ".join([message, *(str(arg) for arg in args)])
        for message, args in captured
    ]
    for message in rendered:
        assert event.event_id in message
        assert "protocol" in message
        assert "fs.read" in message


@pytest.mark.asyncio
async def test_store_deletes_old_and_over_limit_events(
    tmp_path: Path,
    store_factory: StoreFactory,
) -> None:
    store = store_factory(tmp_path)
    old = ToolUseEvent(
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        runtime_surface="protocol",
        requested_tool_name="old.tool",
        status="success",
    )
    middle = ToolUseEvent(
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        runtime_surface="protocol",
        requested_tool_name="middle.tool",
        status="success",
    )
    newest = ToolUseEvent(
        created_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        runtime_surface="protocol",
        requested_tool_name="newest.tool",
        status="success",
    )

    try:
        await store.append_event(old)
        await store.append_event(middle)
        await store.append_event(newest)

        deleted_old = await store.delete_events_older_than(datetime(2026, 1, 2, tzinfo=timezone.utc))
        deleted_over_limit = await store.delete_events_over_limit(1)
        rows = await store.query_events(ToolUseEventQuery(limit=10))

        assert deleted_old == 1
        assert deleted_over_limit == 1
        assert [row.requested_tool_name for row in rows] == ["newest.tool"]
    finally:
        await _close_store(store)


def test_cutoff_epoch_us_calculates_directly_without_event_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_event_model(**_kwargs: Any) -> ToolUseEvent:
        raise AssertionError("ToolUseEvent should not be built for cutoff math")

    monkeypatch.setattr(store_module, "ToolUseEvent", fail_event_model)
    cutoff = datetime(2026, 6, 6, 7, 30, tzinfo=timezone(timedelta(hours=-7)))
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = cutoff.astimezone(timezone.utc) - epoch
    expected = (((delta.days * 86_400) + delta.seconds) * 1_000_000) + delta.microseconds

    assert cutoff_epoch_us(cutoff) == expected


@pytest.mark.asyncio
async def test_report_groups_by_tool_prompt_with_tool_call_rates() -> None:
    store = InMemoryToolUseEventStore()
    await store.append_event(
        ToolUseEvent(
            runtime_surface="protocol",
            requested_tool_name="fs.read",
            tool_prompt_id="fs.read.default",
            status="success",
        )
    )
    await store.append_event(
        ToolUseEvent(
            runtime_surface="protocol",
            requested_tool_name="fs.read",
            tool_prompt_id="fs.read.default",
            status="denied",
            reason_code="permission_denied",
        )
    )

    report = await ToolUseReportService(store).build_report(ToolUseReportQuery(group_by="tool_prompt"))

    row = report.rows[0]
    assert row.group_key == "fs.read.default"
    assert row.call_count == 2
    assert row.tool_call_success_rate == 0.5
    assert row.top_reason_codes[0]["reason_code"] == "permission_denied"


@pytest.mark.asyncio
async def test_report_percentiles_use_half_up_nearest_index() -> None:
    store = InMemoryToolUseEventStore()
    await store.append_event(
        ToolUseEvent(
            runtime_surface="protocol",
            requested_tool_name="fs.read",
            tool_prompt_id="fs.read.default",
            status="success",
            duration_ms=10,
        )
    )
    await store.append_event(
        ToolUseEvent(
            runtime_surface="protocol",
            requested_tool_name="fs.read",
            tool_prompt_id="fs.read.default",
            status="success",
            duration_ms=20,
        )
    )

    report = await ToolUseReportService(store).build_report(ToolUseReportQuery(group_by="tool_prompt"))

    assert report.rows[0].p50_duration_ms == 20
    assert report.rows[0].p95_duration_ms == 20


@pytest.mark.asyncio
async def test_report_discloses_when_event_limit_truncates_aggregates() -> None:
    store = InMemoryToolUseEventStore()
    for index in range(5):
        await store.append_event(
            ToolUseEvent(
                runtime_surface="protocol",
                requested_tool_name=f"tool.{index}",
                status="success",
            )
        )

    report = await ToolUseReportService(store).build_report(ToolUseReportQuery(group_by="tool", event_limit=2))

    assert report.events_scanned == 2
    assert report.event_limit == 2
    assert report.truncated is True
