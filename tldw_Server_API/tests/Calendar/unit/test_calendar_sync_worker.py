from __future__ import annotations

import asyncio
import base64
import json
import multiprocessing
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any

import pytest
from anyio import CancelScope

from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
from tldw_Server_API.app.core.Calendar.errors import (
    CalendarNotFound,
    CalendarPermissionDenied,
    CalendarSyncError,
    CalendarValidationError,
)
from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavEvent
from tldw_Server_API.app.core.Calendar.secret_store import CalendarSecretStore
from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables

pytestmark = pytest.mark.unit


_CUSTOM_VTIMEZONE = """BEGIN:VTIMEZONE
TZID:Custom/Research
BEGIN:STANDARD
DTSTART:19701101T020000
TZOFFSETFROM:-0700
TZOFFSETTO:-0800
RRULE:FREQ=YEARLY;BYMONTH=11;BYDAY=1SU
END:STANDARD
BEGIN:DAYLIGHT
DTSTART:19700308T020000
TZOFFSETFROM:-0800
TZOFFSETTO:-0700
RRULE:FREQ=YEARLY;BYMONTH=3;BYDAY=2SU
END:DAYLIGHT
END:VTIMEZONE"""


@pytest.mark.asyncio
@pytest.mark.parametrize("view_name", ["agenda", "week"])
@pytest.mark.parametrize("zone_name", ["America/Los_Angeles", "Custom/Research"])
async def test_floating_recurrence_dates_survive_import_refresh_and_dst_views(
    calendar_db: CalendarDatabase, view_name: str, zone_name: str,
) -> None:
    """Floating additions/exclusions use the persisted series zone across a DST change."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewFilters, CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    definition = _CUSTOM_VTIMEZONE if zone_name == "Custom/Research" else ""
    payload = (
        f"BEGIN:VCALENDAR\n{definition}\nBEGIN:VEVENT\nUID:floating-dst\n"
        f"DTSTART;TZID={zone_name}:20260306T090000\nDURATION:PT1H\nRRULE:FREQ=DAILY;COUNT=3\n"
        "RDATE:20260309T090000\nRDATE:20260310T160000Z\nEXDATE:20260308T090000\n"
        "END:VEVENT\nEND:VCALENDAR"
    )
    binding = calendar_db.get_external_binding(fixture.binding_id)
    _upsert_events(calendar_db, binding=binding, events=CalDavProvider().parse_vevents(payload))
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-03-07", window_end="2026-03-11",
    )
    item_id = rows[0].id
    _upsert_events(calendar_db, binding=binding, events=CalDavProvider().parse_vevents(payload))
    reopened = CalendarDatabase(db_path=calendar_db.db_path)
    view = CalendarViewService(calendar_service=CalendarService(db=reopened))
    filters = CalendarViewFilters(include_scheduled_tasks=False)
    if view_name == "agenda":
        result = await view.agenda(
            actor_user_id=1, start_at="2026-03-07", end_at="2026-03-11", filters=filters,
        )
    else:
        result = await view.week(actor_user_id=1, week_start="2026-03-07", timezone="UTC", filters=filters)
    assert [item.start_at for item in result.items] == [
        "2026-03-07T09:00:00-08:00", "2026-03-09T09:00:00-07:00", "2026-03-10T09:00:00-07:00",
    ]
    assert [item.end_at for item in result.items] == [
        "2026-03-07T10:00:00-08:00", "2026-03-09T10:00:00-07:00", "2026-03-10T10:00:00-07:00",
    ]
    assert result.partial is False and result.warnings == []
    assert all(item.calendar_item_id == item_id and item.read_only_reason == "provider" for item in result.items)
    stored = json.loads(rows[0].source_payload_json)
    assert stored["rdate"] == ["2026-03-09T09:00:00", "2026-03-10T16:00:00+00:00"]
    assert stored["exdate"] == ["2026-03-08T09:00:00"]


@pytest.mark.asyncio
@pytest.mark.parametrize("view_name", ["agenda", "week"])
@pytest.mark.parametrize(("master", "window_start", "window_end", "expected"), [
    ("20260306T120000", "2026-03-07", "2026-03-10", [
        "2026-03-07T12:00:00-08:00", "2026-03-08T12:00:00-07:00", "2026-03-09T12:00:00-07:00",
    ]),
    ("20261030T120000", "2026-10-31", "2026-11-03", [
        "2026-10-31T12:00:00-07:00", "2026-11-01T12:00:00-08:00", "2026-11-02T12:00:00-08:00",
    ]),
])
async def test_custom_vtimezone_recurrence_survives_import_in_later_views(
    calendar_db: CalendarDatabase, view_name: str, master: str,
    window_start: str, window_end: str, expected: list[str],
) -> None:
    """Persisted custom rules retain wall time and both DST offsets beyond the master window."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewFilters, CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    events = CalDavProvider().parse_vevents(
        f"BEGIN:VCALENDAR\n{_CUSTOM_VTIMEZONE}\nBEGIN:VEVENT\nUID:custom-series\n"
        f"DTSTART;TZID=Custom/Research:{master}\nDURATION:PT1H\n"
        "RRULE:FREQ=DAILY;COUNT=4\nEND:VEVENT\nEND:VCALENDAR"
    )
    binding = calendar_db.get_external_binding(fixture.binding_id)
    _upsert_events(calendar_db, binding=binding, events=events)
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start=window_start, window_end=window_end,
    )
    item_id = rows[0].id
    _upsert_events(calendar_db, binding=binding, events=events)
    # Re-open the repository: no in-memory tzinfo can satisfy the round trip.
    view = CalendarViewService(calendar_service=CalendarService(db=CalendarDatabase(db_path=calendar_db.db_path)))
    filters = CalendarViewFilters(include_scheduled_tasks=False)
    if view_name == "agenda":
        result = await view.agenda(
            actor_user_id=1, start_at=window_start, end_at=window_end, filters=filters,
        )
    else:
        result = await view.week(actor_user_id=1, week_start=window_start, timezone="UTC", filters=filters)
    assert [item.start_at for item in result.items] == expected
    assert result.partial is False
    assert result.warnings == []
    assert all(item.calendar_item_id == item_id and item.read_only_reason == "provider" for item in result.items)
    assert json.loads(rows[0].source_payload_json)["rrule"] == "FREQ=DAILY;COUNT=4"


@pytest.mark.asyncio
@pytest.mark.parametrize(("master", "window_start", "window_end", "duration", "expected_ends"), [
    ("20260306T120000", "2026-03-07T12:00:00-08:00", "2026-03-09", "P1D", [
        "2026-03-08T12:00:00-07:00", "2026-03-09T12:00:00-07:00",
    ]),
    ("20260306T120000", "2026-03-07T12:00:00-08:00", "2026-03-09", "PT24H", [
        "2026-03-08T13:00:00-07:00", "2026-03-09T12:00:00-07:00",
    ]),
    ("20261030T120000", "2026-10-31T12:00:00-07:00", "2026-11-02", "P1D", [
        "2026-11-01T12:00:00-08:00", "2026-11-02T12:00:00-08:00",
    ]),
    ("20261030T120000", "2026-10-31T12:00:00-07:00", "2026-11-02", "PT24H", [
        "2026-11-01T11:00:00-08:00", "2026-11-02T12:00:00-08:00",
    ]),
])
async def test_custom_vtimezone_keeps_nominal_days_distinct_from_elapsed_hours(
    calendar_db: CalendarDatabase, master: str, window_start: str,
    window_end: str, duration: str, expected_ends: list[str],
) -> None:
    """Each later occurrence reapplies the lexical duration using the embedded DST rules."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewFilters, CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    events = CalDavProvider().parse_vevents(
        f"BEGIN:VCALENDAR\n{_CUSTOM_VTIMEZONE}\nBEGIN:VEVENT\nUID:custom-duration\n"
        f"DTSTART;TZID=Custom/Research:{master}\nDURATION:{duration}\n"
        "RRULE:FREQ=DAILY;COUNT=4\nEND:VEVENT\nEND:VCALENDAR"
    )
    _upsert_events(calendar_db, binding=calendar_db.get_external_binding(fixture.binding_id), events=events)
    result = await CalendarViewService(calendar_service=CalendarService(db=calendar_db)).agenda(
        actor_user_id=1, start_at=window_start, end_at=window_end,
        filters=CalendarViewFilters(include_scheduled_tasks=False),
    )
    assert [item.end_at for item in result.items] == expected_ends
    assert result.partial is False


@pytest.mark.asyncio
@pytest.mark.parametrize("exclude_second_fold", [False, True])
async def test_custom_vtimezone_preserves_dates_detached_identity_and_fold_order(
    calendar_db: CalendarDatabase, exclude_second_fold: bool,
) -> None:
    """UTC set identity distinguishes folds and suppresses only the detached original instant."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewFilters, CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    fold_exdate = "EXDATE:20261101T093000Z\n" if exclude_second_fold else ""
    events = CalDavProvider().parse_vevents(
        f"BEGIN:VCALENDAR\n{_CUSTOM_VTIMEZONE}\nBEGIN:VEVENT\nUID:custom-fold\nSUMMARY:Master\n"
        "DTSTART;TZID=Custom/Research:20261030T013000\nDURATION:PT30M\nRRULE:FREQ=DAILY;COUNT=5\n"
        "RDATE:20261101T093000Z\nEXDATE;TZID=Custom/Research:20261031T013000\n"
        f"{fold_exdate}END:VEVENT\nBEGIN:VEVENT\nUID:custom-fold\nSUMMARY:Moved\n"
        "RECURRENCE-ID;TZID=Custom/Research:20261102T013000\n"
        "DTSTART;TZID=Custom/Research:20261102T050000\nDURATION:PT30M\nEND:VEVENT\nEND:VCALENDAR"
    )
    binding = calendar_db.get_external_binding(fixture.binding_id)
    _upsert_events(calendar_db, binding=binding, events=events)
    _upsert_events(calendar_db, binding=binding, events=list(reversed(events)))
    result = await CalendarViewService(calendar_service=CalendarService(db=calendar_db)).agenda(
        actor_user_id=1, start_at="2026-10-31", end_at="2026-11-04",
        filters=CalendarViewFilters(include_scheduled_tasks=False),
    )
    expected = [
        ("Master", "2026-11-01T01:30:00-07:00", "2026-11-01T01:00:00-08:00"),
        ("Moved", "2026-11-02T05:00:00-08:00", "2026-11-02T05:30:00-08:00"),
        ("Master", "2026-11-03T01:30:00-08:00", "2026-11-03T02:00:00-08:00"),
    ]
    if not exclude_second_fold:
        expected.insert(1, ("Master", "2026-11-01T01:30:00-08:00", "2026-11-01T02:00:00-08:00"))
    assert [(item.title, item.start_at, item.end_at) for item in result.items] == expected
    assert len({item.id for item in result.items}) == len(expected)
    assert result.partial is False
    assert events[1].recurrence_id == "2026-11-02T09:30:00+00:00"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata_problem", ["nonproductive", "oversized"])
async def test_unsafe_persisted_custom_timezone_reports_partial_later_view(
    calendar_db: CalendarDatabase, metadata_problem: str,
) -> None:
    """Unusable metadata is rejected safely and cannot silently manufacture fixed-offset instances."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewFilters, CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    events = CalDavProvider().parse_vevents(
        f"BEGIN:VCALENDAR\n{_CUSTOM_VTIMEZONE}\nBEGIN:VEVENT\nUID:unsafe-stored-zone\n"
        "DTSTART;TZID=Custom/Research:20260306T120000\nDURATION:PT1H\n"
        "RRULE:FREQ=DAILY;COUNT=4\nEND:VEVENT\nEND:VCALENDAR"
    )
    metadata = events[0].provider_payload
    assert metadata is not None
    metadata["vtimezone"] = (
        _CUSTOM_VTIMEZONE.replace("BYMONTH=3;BYDAY=2SU", "BYMONTH=2;BYMONTHDAY=30")
        if metadata_problem == "nonproductive" else "x" * (64 * 1024 + 1)
    )
    _upsert_events(calendar_db, binding=calendar_db.get_external_binding(fixture.binding_id), events=events)
    result = await CalendarViewService(calendar_service=CalendarService(db=calendar_db)).agenda(
        actor_user_id=1, start_at="2026-03-07", end_at="2026-03-10",
        filters=CalendarViewFilters(include_scheduled_tasks=False),
    )
    assert result.items == []
    assert result.partial is True
    assert result.warnings


@pytest.mark.asyncio
async def test_import_preserves_master_and_detached_occurrences_across_refresh(calendar_db: CalendarDatabase) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    events = CalDavProvider().parse_vevents("""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:series
SUMMARY:Daily
DTSTART:20260605T090000Z
DTEND:20260605T100000Z
RRULE:FREQ=DAILY;COUNT=4
RDATE:20260610T090000Z
EXDATE:20260606T090000Z
END:VEVENT
BEGIN:VEVENT
UID:series
RECURRENCE-ID:20260607T090000Z
SUMMARY:Moved
DTSTART:20260607T110000Z
DTEND:20260607T120000Z
END:VEVENT
BEGIN:VEVENT
UID:series
RECURRENCE-ID:20260608T090000Z
SUMMARY:Moved outside window
DTSTART:20260708T090000Z
DTEND:20260708T100000Z
END:VEVENT
END:VCALENDAR""")
    binding = calendar_db.get_external_binding(fixture.binding_id)
    _upsert_events(calendar_db, binding=binding, events=events)
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-07-15"
    )
    first_ids = {row.source_uid: row.id for row in rows}
    assert len(rows) == 3
    _upsert_events(calendar_db, binding=binding, events=list(reversed(events)))
    refreshed = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-07-15"
    )
    assert {row.source_uid: row.id for row in refreshed} == first_ids
    result = await CalendarViewService(calendar_service=CalendarService(db=calendar_db)).agenda(
        actor_user_id=1, start_at="2026-06-01T00:00:00Z", end_at="2026-06-12T00:00:00Z"
    )
    assert [(entry.title, entry.start_at) for entry in result.items] == [
        ("Daily", "2026-06-05T09:00:00+00:00"),
        ("Moved", "2026-06-07T11:00:00+00:00"),
        ("Daily", "2026-06-10T09:00:00+00:00"),
    ]


def test_import_preserves_all_day_flag(calendar_db: CalendarDatabase) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider

    fixture = _create_sync_fixture(calendar_db)
    events = CalDavProvider().parse_vevents("""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:holiday
DTSTART;VALUE=DATE:20260605
DTEND;VALUE=DATE:20260607
END:VEVENT
END:VCALENDAR""")
    _upsert_events(calendar_db, binding=calendar_db.get_external_binding(fixture.binding_id), events=events)
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-06-08"
    )
    assert rows[0].all_day is True
    assert rows[0].start_at == "2026-06-05"
    assert rows[0].end_at == "2026-06-07"


@pytest.mark.asyncio
async def test_imported_all_day_until_dates_expand_with_exclusions(calendar_db: CalendarDatabase) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import _upsert_events
    from tldw_Server_API.app.core.Calendar.providers.caldav import CalDavProvider
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewService

    fixture = _create_sync_fixture(calendar_db)
    events = CalDavProvider().parse_vevents("""BEGIN:VCALENDAR
BEGIN:VEVENT
UID:holiday-series
DTSTART;VALUE=DATE:20260605
RRULE:FREQ=DAILY;UNTIL=20260608
EXDATE;VALUE=DATE:20260606
END:VEVENT
END:VCALENDAR""")
    _upsert_events(calendar_db, binding=calendar_db.get_external_binding(fixture.binding_id), events=events)
    result = await CalendarViewService(calendar_service=CalendarService(db=calendar_db)).agenda(
        actor_user_id=1, start_at="2026-06-05T12:00:00Z", end_at="2026-06-09T00:00:00Z"
    )
    assert [item.start_at for item in result.items] == ["2026-06-05", "2026-06-07", "2026-06-08"]
    assert result.partial is False


@pytest.mark.asyncio
async def test_sync_provider_runs_off_event_loop(calendar_db: CalendarDatabase, jobs_manager: JobManager) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    queued = CalendarService(db=calendar_db, job_manager=jobs_manager).queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00Z",
        window_end="2026-06-08T00:00:00Z",
    )
    call_threads: list[int] = []

    class Provider:
        def fetch_vevents(self, **_kwargs: object) -> list[CalDavEvent]:
            call_threads.append(threading.get_ident())
            return []

    await handle_calendar_sync_job(jobs_manager.get_job(queued.job_id), db=calendar_db, provider=Provider())
    assert call_threads and call_threads[0] != threading.get_ident()


def _trace_db_operation(
    monkeypatch: pytest.MonkeyPatch, operation: str, *, target: Any = CalendarDatabase
) -> list[tuple[str, int]]:
    """Trace synchronous DB work or full context lifetimes without replacing their work."""
    calls: list[tuple[str, int]] = []
    original = getattr(target, operation)

    @wraps(original)
    def traced(*args: Any, **kwargs: Any) -> Any:
        calls.append(("enter", threading.get_ident()))
        try:
            return original(*args, **kwargs)
        finally:
            calls.append(("exit", threading.get_ident()))

    @contextmanager
    def traced_context(*args: Any, **kwargs: Any) -> Iterator[Any]:
        calls.append(("enter", threading.get_ident()))
        try:
            with original(*args, **kwargs) as connection:
                calls.append(("body", threading.get_ident()))
                yield connection
        finally:
            calls.append(("exit", threading.get_ident()))

    monkeypatch.setattr(
        target, operation, traced_context if operation in {"connection", "transaction"} else traced
    )
    return calls


def _sync_job(binding_id: int) -> dict[str, Any]:
    """Build a secret-free worker input without involving the unrelated Jobs database."""
    return {
        "id": 42,
        "job_type": "calendar_sync",
        "owner_user_id": "1",
        "payload": {
            "binding_id": binding_id,
            "window_start": "2026-06-01T00:00:00Z",
            "window_end": "2026-06-08T00:00:00Z",
            "reason": "manual",
        },
    }


def _sync_event(uid: str, *, rrule: str | None = None) -> CalDavEvent:
    """Provide a complete timed provider event for real persistence tests."""
    return CalDavEvent(
        uid=uid, title=uid, start_at="2026-06-05T09:00:00Z", end_at="2026-06-05T10:00:00Z",
        location=None, description=None, rrule=rrule,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        "__init__",
        "ensure_schema",
        "get_external_binding",
        "get_external_account",
        "resolve_secret_ref_for_user",
        "resolve_caldav_credentials",
        "connection",
        "transaction",
        "upsert_provider_item",
        "upsert_recurrence",
        "delete_recurrence",
        "_upsert_events",
        "update_binding_sync_state",
        "record_sync_event",
    ],
)
async def test_worker_complete_db_phases_run_off_event_loop(
    calendar_db: CalendarDatabase, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Every DB phase, including default schema creation and transaction exits, stays off-loop."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker
    from tldw_Server_API.app.core.DB_Management import Calendar_DB

    fixture = _create_sync_fixture(calendar_db)
    loop_thread = threading.get_ident()
    provider = _FakeProvider(events=[
        _sync_event("master", rrule="FREQ=DAILY;COUNT=2"),
        _sync_event("single"),
    ])
    with monkeypatch.context() as patch:
        patch.setattr(Calendar_DB, "_default_calendar_db_path", lambda: calendar_db.db_path)
        target = worker if operation in {"resolve_caldav_credentials", "_upsert_events"} else CalendarDatabase
        calls = _trace_db_operation(patch, operation, target=target)
        result = await worker.handle_calendar_sync_job(
            _sync_job(fixture.binding_id),
            db=None if operation in {"__init__", "ensure_schema"} else calendar_db,
            provider=provider,
        )

    assert calls, f"The worker did not exercise {operation}"
    assert all(thread_id != loop_thread for _, thread_id in calls), f"{operation} ran on the event loop: {calls}"
    assert result == {"items_seen": 2, "items_upserted": 2, "items_tombstoned": 0}
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-06-08"
    )
    assert {row.source_uid for row in rows} == {"master", "single"}
    assert calendar_db.list_sync_events(binding_id=fixture.binding_id)[0].status == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["credentials", "provider", "import", "success_state", "success_audit"])
async def test_worker_failure_bookkeeping_runs_off_event_loop(
    calendar_db: CalendarDatabase, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    """Failures from each protected phase retain their exception and persist off-loop diagnostics."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    fixture = _create_sync_fixture(calendar_db)
    failure = RuntimeError(f"{failure_phase} unavailable")
    provider = _FakeProvider(events=[_sync_event("event")])
    loop_thread = threading.get_ident()
    with monkeypatch.context() as patch:
        state_calls = _trace_db_operation(patch, "update_binding_sync_state")
        audit_calls = _trace_db_operation(patch, "record_sync_event")
        if failure_phase == "provider":
            provider.exc = failure
        else:
            target: Any = worker if failure_phase == "credentials" else calendar_db
            operation = {
                "credentials": "resolve_caldav_credentials",
                "import": "upsert_provider_item",
                "success_state": "update_binding_sync_state",
                "success_audit": "record_sync_event",
            }[failure_phase]
            original = getattr(target, operation)

            def fail_phase(*args: Any, **kwargs: Any) -> Any:
                """Fail only the requested phase, leaving failure bookkeeping real."""
                if kwargs.get("last_error") is not None or kwargs.get("status") == "failed":
                    return original(*args, **kwargs)
                raise failure

            patch.setattr(target, operation, fail_phase)

        with pytest.raises(RuntimeError) as raised:
            await worker.handle_calendar_sync_job(_sync_job(fixture.binding_id), db=calendar_db, provider=provider)

    assert raised.value is failure
    assert state_calls and audit_calls
    assert all(thread_id != loop_thread for _, thread_id in state_calls + audit_calls)
    binding = calendar_db.get_external_binding(fixture.binding_id)
    audit = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    assert binding.last_error == str(failure)
    assert audit[0].status == "failed"
    assert audit[0].error_message == str(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_import", [False, True], ids=["success", "failure"])
@pytest.mark.parametrize("cancel_count", [1, 3], ids=["cancel-once", "cancel-repeatedly"])
async def test_worker_native_cancellation_drains_import_and_records_outcome(
    calendar_db: CalendarDatabase,
    monkeypatch: pytest.MonkeyPatch,
    fail_import: bool,
    cancel_count: int,
) -> None:
    """Native cancellation must wait for real batch commit/rollback and its sync audit."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    fixture = _create_sync_fixture(calendar_db)
    failure = RuntimeError("blocked import failed")
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    import_started = asyncio.Event()
    import_returned = asyncio.Event()
    release_import = threading.Event()
    original = worker._upsert_events

    def blocked_import(*args: Any, **kwargs: Any) -> dict[str, int]:
        """Hold actual writes inside the outer transaction, then succeed or force rollback."""
        result = original(*args, **kwargs)
        loop.call_soon_threadsafe(import_started.set)
        try:
            if not release_import.wait(timeout=10):
                raise TimeoutError("Test did not release the blocked import")
            if fail_import:
                raise failure
            return result
        finally:
            loop.call_soon_threadsafe(import_returned.set)

    with monkeypatch.context() as patch:
        transactions = _trace_db_operation(patch, "transaction")
        state_calls = _trace_db_operation(patch, "update_binding_sync_state")
        audit_calls = _trace_db_operation(patch, "record_sync_event")
        patch.setattr(worker, "_upsert_events", blocked_import)
        task = asyncio.create_task(worker.handle_calendar_sync_job(
            _sync_job(fixture.binding_id), db=calendar_db,
            provider=_FakeProvider(events=[_sync_event("first"), _sync_event("second", rrule="FREQ=DAILY;COUNT=2")]),
        ))
        exited_while_blocked: list[bool] = []
        try:
            await asyncio.wait_for(import_started.wait(), timeout=5)
            for _ in range(cancel_count):
                task.cancel("native shutdown")
                # Deliver cancellation and run any handler continuation without releasing the DB thread.
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                exited_while_blocked.append(task.done())
        finally:
            release_import.set()
            with pytest.raises(asyncio.CancelledError) as cancelled:
                await task
            await asyncio.wait_for(import_returned.wait(), timeout=5)

    binding = calendar_db.get_external_binding(fixture.binding_id)
    audits = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-06-08"
    )
    assert exited_while_blocked == [False] * cancel_count, "Handler exited before its DB work finished"
    assert str(cancelled.value) == "native shutdown"
    assert all(thread_id != loop_thread for _, thread_id in transactions + state_calls + audit_calls)
    assert len({thread_id for _, thread_id in transactions}) == 1
    assert len(audits) == 1
    if fail_import:
        assert rows == []
        assert binding.last_error == str(failure)
        assert audits[0].status == "failed"
        assert audits[0].error_message == str(failure)
        assert cancelled.value.__cause__ is failure
    else:
        assert {row.source_uid for row in rows} == {"first", "second"}
        assert binding.last_sync_at is not None
        assert binding.last_error is None
        assert audits[0].status == "success"
        assert audits[0].items_upserted == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_phase", ["provider", "import"])
async def test_worker_native_cancellation_drains_failure_bookkeeping(
    calendar_db: CalendarDatabase, monkeypatch: pytest.MonkeyPatch, failure_phase: str
) -> None:
    """Repeated native cancellation cannot leave an in-flight failure audit behind."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    fixture = _create_sync_fixture(calendar_db)
    failure = RuntimeError(f"{failure_phase} failed before audit")
    provider = _FakeProvider(events=[_sync_event("event")], exc=failure if failure_phase == "provider" else None)
    loop = asyncio.get_running_loop()
    audit_started = asyncio.Event()
    audit_finished = asyncio.Event()
    release_audit = threading.Event()
    original = calendar_db.record_sync_event

    def blocked_audit(**kwargs: Any) -> Any:
        """Pause failure bookkeeping before its real audit write."""
        loop.call_soon_threadsafe(audit_started.set)
        try:
            if not release_audit.wait(timeout=10):
                raise TimeoutError("Test did not release the blocked failure audit")
            return original(**kwargs)
        finally:
            loop.call_soon_threadsafe(audit_finished.set)

    def fail_import(**_kwargs: Any) -> Any:
        """Trigger the import failure path without replacing its bookkeeping."""
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(calendar_db, "record_sync_event", blocked_audit)
        if failure_phase == "import":
            patch.setattr(calendar_db, "upsert_provider_item", fail_import)
        task = asyncio.create_task(worker.handle_calendar_sync_job(
            _sync_job(fixture.binding_id), db=calendar_db, provider=provider,
        ))
        exited_while_blocked: list[bool] = []
        try:
            await asyncio.wait_for(audit_started.wait(), timeout=5)
            for _ in range(3):
                task.cancel("native shutdown during audit")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                exited_while_blocked.append(task.done())
        finally:
            release_audit.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.wait_for(audit_finished.wait(), timeout=5)

    assert exited_while_blocked == [False, False, False], "Handler exited before failure bookkeeping finished"
    assert calendar_db.get_external_binding(fixture.binding_id).last_error == str(failure)
    audits = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    assert len(audits) == 1
    assert audits[0].status == "failed"
    assert audits[0].error_message == str(failure)


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_import", [False, True], ids=["success", "failure"])
async def test_worker_scope_cancellation_drains_without_hot_retries(
    calendar_db: CalendarDatabase, monkeypatch: pytest.MonkeyPatch, fail_import: bool,
) -> None:
    """Level-triggered cancellation waits for real DB work without spinning through shield retries."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    fixture = _create_sync_fixture(calendar_db)
    failure = RuntimeError("scope-cancelled import failed")
    loop = asyncio.get_running_loop()
    import_started = asyncio.Event()
    release_import = threading.Event()
    scopes: list[CancelScope] = []
    retries: list[None] = []
    task: asyncio.Task[None] | None = None
    original_import = worker._upsert_events
    original_shield = asyncio.shield

    def blocked_import(*args: Any, **kwargs: Any) -> dict[str, int]:
        """Hold real writes until the event loop has exercised scope cancellation."""
        result = original_import(*args, **kwargs)
        loop.call_soon_threadsafe(import_started.set)
        if not release_import.wait(timeout=10):
            raise TimeoutError("Test did not release the blocked import")
        if fail_import:
            raise failure
        return result

    def counted_shield(awaitable: Any) -> asyncio.Future[Any]:
        """Count actual drain attempts only after the import thread is blocked."""
        if asyncio.current_task() is task and import_started.is_set():
            retries.append(None)
        return original_shield(awaitable)

    async def scoped_handler() -> None:
        """Cancel the real handler inside an AnyIO scope rather than cancelling its native task."""
        with CancelScope() as scope:
            scopes.append(scope)
            await worker.handle_calendar_sync_job(
                _sync_job(fixture.binding_id), db=calendar_db, provider=_FakeProvider(events=[_sync_event("event")]),
            )

    with monkeypatch.context() as patch:
        patch.setattr(worker, "_upsert_events", blocked_import)
        patch.setattr(asyncio, "shield", counted_shield)
        task = asyncio.create_task(scoped_handler())
        try:
            await asyncio.wait_for(import_started.wait(), timeout=5)
            scopes[0].cancel()
            for _ in range(20):
                await asyncio.sleep(0)
            finished_early = task.done()
            retry_count = len(retries)
        finally:
            release_import.set()
            await asyncio.wait_for(task, timeout=5)

    audits = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    rows = calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-06-08",
    )
    assert not finished_early, "Handler exited before its DB phase finished"
    assert retry_count <= 3, f"Scope cancellation spun through {retry_count} drain attempts"
    assert scopes[0].cancelled_caught
    assert len(audits) == 1
    assert audits[0].status == ("failed" if fail_import else "success")
    assert len(rows) == (0 if fail_import else 1)


@pytest.mark.asyncio
async def test_worker_import_rollback_keeps_full_transaction_on_one_worker_thread(
    calendar_db: CalendarDatabase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recurrence failure rolls back the complete event batch on its owning worker thread."""
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    failure = RuntimeError("recurrence write failed")
    write_threads: list[int] = []
    original = calendar_db.upsert_recurrence
    loop_thread = threading.get_ident()

    def fail_recurrence(**kwargs: Any) -> Any:
        """Raise after a real nested write to prove the outer transaction rolls it back."""
        write_threads.append(threading.get_ident())
        original(**kwargs)
        raise failure

    with monkeypatch.context() as patch:
        transactions = _trace_db_operation(patch, "transaction")
        patch.setattr(calendar_db, "upsert_recurrence", fail_recurrence)
        with pytest.raises(RuntimeError) as raised:
            await handle_calendar_sync_job(
                _sync_job(fixture.binding_id),
                db=calendar_db,
                provider=_FakeProvider(events=[
                    _sync_event("first"),
                    _sync_event("second", rrule="FREQ=DAILY;COUNT=2"),
                ]),
            )

    assert raised.value is failure
    assert write_threads and write_threads[0] != loop_thread
    # The batch includes item writes, nested recurrence work, and outer rollback before bookkeeping.
    depth = 0
    for label, thread_id in transactions:
        assert thread_id == write_threads[0]
        depth += 1 if label == "enter" else -1 if label == "exit" else 0
        if depth == 0:
            break
    assert depth == 0
    assert calendar_db.list_items_for_expansion(
        calendar_ids=[fixture.calendar_id], window_start="2026-06-01", window_end="2026-06-08"
    ) == []
    assert calendar_db.list_sync_events(binding_id=fixture.binding_id)[0].status == "failed"


@pytest.mark.asyncio
async def test_scheduler_skips_disappeared_account_and_continues(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    missing = _create_sync_fixture(calendar_db)
    healthy = _create_sync_fixture(calendar_db)
    original = calendar_db.get_external_account

    def get_account(account_id: int, **kwargs: object) -> object:
        if account_id == missing.account_id:
            raise CalendarNotFound("account deleted during scan")
        return original(account_id, **kwargs)

    monkeypatch.setattr(calendar_db, "get_external_account", get_account)
    queued = await queue_due_calendar_sync_jobs(db=calendar_db, job_manager=jobs_manager)
    assert [item.binding_id for item in queued] == [healthy.binding_id]


@pytest.mark.asyncio
async def test_scheduler_retries_after_scan_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import calendar_sync_scheduler as scheduler

    stop = asyncio.Event()
    scans: list[int] = []

    async def scan(**_kwargs: object) -> list[object]:
        scans.append(1)
        if len(scans) == 1:
            raise RuntimeError("database temporarily unavailable")
        stop.set()
        return []

    monkeypatch.setattr(scheduler, "queue_due_calendar_sync_jobs", scan)
    await scheduler.run_calendar_sync_scheduler(stop, interval_seconds=0.001)
    assert len(scans) == 2


def test_shared_credentials_enforce_scope_and_override_precedence(calendar_db: CalendarDatabase) -> None:
    from tldw_Server_API.app.core.Calendar.errors import CalendarPermissionDenied
    from tldw_Server_API.app.core.Calendar.provider_operations import resolve_caldav_credentials

    fixture = _create_sync_fixture(calendar_db)
    with pytest.raises(CalendarPermissionDenied):
        resolve_caldav_credentials(calendar_db, account_id=fixture.account_id, actor_user_id=2, tenant_id="default")
    with pytest.raises(CalendarPermissionDenied):
        resolve_caldav_credentials(calendar_db, account_id=fixture.account_id, actor_user_id=1, tenant_id="other")
    credentials = resolve_caldav_credentials(
        calendar_db,
        account_id=fixture.account_id,
        actor_user_id=1,
        tenant_id="default",
        overrides={"username": "override-user", "token": "override-token"},
    )
    assert credentials == {
        "server_url": "https://caldav.example.test/dav/",
        "username": "override-user",
        "password": "override-token",
    }


@pytest.fixture
def calendar_db(tmp_path: Path) -> CalendarDatabase:
    db = CalendarDatabase(db_path=tmp_path / "calendar_sync.db")
    db.ensure_schema()
    return db


@pytest.fixture
def jobs_manager(tmp_path: Path) -> JobManager:
    db_path = tmp_path / "calendar_jobs.db"
    ensure_jobs_tables(db_path)
    return JobManager(db_path)


@pytest.fixture(autouse=True)
def calendar_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key = base64.b64encode(b"c" * 32).decode("ascii")
    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", key)


@dataclass
class _SyncFixture:
    calendar_id: int
    account_id: int
    binding_id: int
    secret_ref: str


class _FakeProvider:
    def __init__(self, *, events: list[CalDavEvent] | None = None, exc: Exception | None = None) -> None:
        self.events = events or []
        self.exc = exc
        self.calls: list[dict[str, object]] = []

    def fetch_vevents(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc is not None:
            raise self.exc
        return self.events


def _create_sync_fixture(
    calendar_db: CalendarDatabase,
    *,
    owner_user_id: int = 1,
    remote_calendar_url: str = "https://caldav.example.test/calendars/user/work/",
) -> _SyncFixture:
    calendar = calendar_db.create_calendar(
        tenant_id="default",
        owner_user_id=owner_user_id,
        org_id=None,
        name="Imported",
        timezone="UTC",
        color="#2563eb",
    )
    secret_ref = CalendarSecretStore(db=calendar_db, tenant_id="default").create_secret(
        owner_user_id=owner_user_id,
        provider="caldav",
        payload={
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
            "password": "app-secret",
        },
    )
    account = calendar_db.create_external_account(
        tenant_id="default",
        user_id=owner_user_id,
        provider="caldav",
        display_name="Fastmail",
        secret_ref=secret_ref,
        account_metadata_json={
            "server_url": "https://caldav.example.test/dav/",
            "username": "reader@example.test",
        },
    )
    binding = calendar_db.create_external_binding(
        account_id=account.id,
        calendar_id=calendar.id,
        remote_calendar_id=remote_calendar_url,
        remote_display_name="Work",
        lookback_days=14,
        lookahead_days=30,
    )
    return _SyncFixture(
        calendar_id=calendar.id,
        account_id=account.id,
        binding_id=binding.id,
        secret_ref=secret_ref,
    )


def test_queue_binding_sync_creates_sanitized_jobs_payload(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)

    response = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    job = jobs_manager.get_job(response.job_id)
    assert response.queued is True
    assert response.status == "queued"
    assert job is not None
    assert job["domain"] == "calendar"
    assert job["queue"] == "default"
    assert job["job_type"] == "calendar_sync"
    assert job["owner_user_id"] == "1"
    assert job["batch_group"] == f"calendar:sync:binding:{fixture.binding_id}"
    assert response.idempotency_key.startswith("calendar:sync:admission:")
    assert job["payload"] == {
        "binding_id": fixture.binding_id,
        "window_start": "2026-06-01T00:00:00+00:00",
        "window_end": "2026-06-08T00:00:00+00:00",
        "reason": "manual",
    }
    assert "app-secret" not in json.dumps(job["payload"])
    assert "secret_ref" not in json.dumps(job["payload"])


def test_queue_binding_sync_reuses_active_binding_job(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    first = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    second = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-07-01T00:00:00+00:00",
        window_end="2026-07-08T00:00:00+00:00",
    )

    assert second.job_id == first.job_id
    assert second.queued is False
    assert second.status == "already_active"
    assert jobs_manager.count_jobs(domain="calendar", queue="default", job_type="calendar_sync") == 1


def _queue_admission(service: CalendarService, binding_id: int, reason: str = "manual") -> Any:
    """Submit a fixed canonical window through the real service."""
    return service.queue_binding_sync(
        actor_user_id=1, binding_id=binding_id, reason=reason,
        window_start="2026-06-01T00:00:00+00:00", window_end="2026-06-08T00:00:00+00:00",
    )


def test_durable_admission_audit_failure_creates_no_runnable_job(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Calendar audit cannot leave runnable work in the separate Jobs store."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)

    def fail(**kwargs: Any) -> Any:
        raise RuntimeError("queued audit failed")

    with monkeypatch.context() as patch:
        patch.setattr(calendar_db, "record_sync_event", fail)
        with pytest.raises(RuntimeError, match="queued audit failed"):
            _queue_admission(service, fixture.binding_id)
    assert jobs_manager.count_jobs(domain="calendar") == 0
    assert calendar_db.list_sync_events(binding_id=fixture.binding_id) == []
    assert _queue_admission(service, fixture.binding_id).queued
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


def test_durable_admission_dispatch_failure_retains_audit_and_recovers_identity(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch failure retains one committed intent that retry dispatches exactly once."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    def fail(command: Any) -> Any:
        raise RuntimeError("Jobs unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(jobs_manager, "admit_idempotent_operation", fail)
        with pytest.raises(RuntimeError, match="Jobs unavailable"):
            _queue_admission(service, fixture.binding_id)
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1
    pending = calendar_db.get_sync_admission(fixture.binding_id)
    assert pending is not None and pending.job_id is None
    reopened = CalendarService(db=CalendarDatabase(calendar_db.db_path), job_manager=JobManager(jobs_manager.db_path))
    recovered = _queue_admission(reopened, fixture.binding_id, "scheduled")
    assert recovered.idempotency_key == pending.idempotency_key
    assert recovered.queued
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


@pytest.mark.parametrize("newer_status", ["queued", "cancelled"])
def test_durable_admission_finds_legacy_active_work_behind_newer_jobs(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, newer_status: str,
) -> None:
    """The authoritative legacy scan must not stop at the newest hundred jobs."""
    fixture = _create_sync_fixture(calendar_db)
    legacy = jobs_manager.create_job(
        domain="calendar", queue="default", job_type="calendar_sync", owner_user_id="1",
        payload={"binding_id": fixture.binding_id}, idempotency_key="legacy-active-calendar",
    )
    for index in range(101):
        newer = jobs_manager.create_job(
            domain="calendar", queue="default", job_type="calendar_sync", owner_user_id="1",
            payload={"binding_id": 999}, idempotency_key=f"unrelated-{index}",
        )
        if newer_status == "cancelled":
            assert jobs_manager.cancel_job(newer["id"])
    response = _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager), fixture.binding_id)
    assert response.job_id == legacy["id"]
    assert not response.queued
    assert calendar_db.list_sync_events(binding_id=fixture.binding_id) == []


def test_durable_admission_decodes_encrypted_dispatch_and_receipt_replay(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Encrypted Jobs envelopes use canonical reads before binding correlation."""
    monkeypatch.setenv("JOBS_ENCRYPT_CALENDAR", "true")
    monkeypatch.setenv("WORKFLOWS_ARTIFACT_ENC_KEY", base64.b64encode(b"c" * 32).decode("ascii"))
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    original = jobs_manager.admit_idempotent_operation
    envelopes = []

    def dispatch(command: Any) -> Any:
        result = original(command)
        envelopes.append(result.job["payload"])
        return result

    monkeypatch.setattr(jobs_manager, "admit_idempotent_operation", dispatch)

    def crash(*args: Any) -> Any:
        raise RuntimeError("encrypted dispatch crash")

    with monkeypatch.context() as patch:
        patch.setattr(calendar_db, "attach_sync_admission_job", crash)
        with pytest.raises(RuntimeError, match="encrypted dispatch crash"):
            _queue_admission(service, fixture.binding_id)
    assert "_encrypted" in envelopes[0]
    first = _queue_admission(service, fixture.binding_id)
    second = _queue_admission(service, fixture.binding_id)
    assert second.job_id == first.job_id and not second.queued
    assert jobs_manager.get_job(first.job_id)["payload"]["binding_id"] == fixture.binding_id
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


def test_durable_admission_pending_receipt_with_missing_job_fails_closed(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A committed receipt without readable Jobs authority must not dispatch again."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)

    def crash(*args: Any) -> Any:
        raise RuntimeError("post-dispatch crash")

    with monkeypatch.context() as patch:
        patch.setattr(calendar_db, "attach_sync_admission_job", crash)
        with pytest.raises(RuntimeError, match="post-dispatch crash"):
            _queue_admission(service, fixture.binding_id)
    pending = calendar_db.get_sync_admission(fixture.binding_id)
    monkeypatch.setattr(jobs_manager, "get_job_or_archived_by_uuid", lambda *args, **kwargs: None)
    monkeypatch.setattr(jobs_manager, "admit_idempotent_operation", lambda *args: pytest.fail("Receipt replay redispatched"))
    with pytest.raises(CalendarSyncError, match="unavailable"):
        _queue_admission(service, fixture.binding_id)
    assert calendar_db.get_sync_admission(fixture.binding_id) == pending
    assert jobs_manager.count_jobs(domain="calendar") == 1


def test_durable_admission_upgrades_existing_calendar_database(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """An existing database gains admission storage without changing saved binding state."""
    fixture = _create_sync_fixture(calendar_db)
    binding = calendar_db.get_external_binding(fixture.binding_id)
    with calendar_db.connection() as connection:
        connection.execute("DROP TABLE calendar_sync_admissions")
        connection.commit()
    reopened = CalendarDatabase(calendar_db.db_path)
    assert reopened.get_external_binding(fixture.binding_id) == binding
    result = _queue_admission(CalendarService(db=reopened, job_manager=jobs_manager), fixture.binding_id)
    assert result.queued and reopened.get_sync_admission(fixture.binding_id).job_id == result.job_id


def _independent_admission_process(calendar_path: str, jobs_path: str, binding_id: int, barrier: Any, output: Any) -> None:
    """Synchronize independent empty legacy reads, retaining real database writes."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    original = worker._active_job_for_binding
    first_lookup = True

    def synchronized_lookup(**kwargs: Any) -> Any:
        nonlocal first_lookup
        existing = original(**kwargs)
        if first_lookup:
            first_lookup = False
            barrier.wait(timeout=15)
        return existing

    worker._active_job_for_binding = synchronized_lookup
    try:
        response = _queue_admission(
            CalendarService(db=CalendarDatabase(calendar_path), job_manager=JobManager(jobs_path)), binding_id,
            f"process-{multiprocessing.current_process().name}",
        )
        output.put((response.job_id, response.queued, response.idempotency_key))
    except Exception as exc:
        output.put(("error", type(exc).__name__))
        raise


def test_durable_admission_independent_processes_admit_one_job_and_audit(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """Real independent processes with stale empty reads share durable binding authority."""
    fixture = _create_sync_fixture(calendar_db)
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    barrier = context.Barrier(2)
    processes = [context.Process(target=_independent_admission_process, args=(
        str(calendar_db.db_path), str(jobs_manager.db_path), fixture.binding_id, barrier, output,
    )) for _ in range(2)]
    try:
        for process in processes:
            process.start()
        results = [output.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=10)
        assert [process.exitcode for process in processes] == [0, 0], results
        assert jobs_manager.count_jobs(domain="calendar") == 1, results
        assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1
        assert len({result[0] for result in results}) == 1
        assert sum(result[1] for result in results) == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        output.close()
        output.join_thread()


def test_durable_admission_terminal_same_window_creates_fresh_job(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """A terminal job cannot permanently suppress a later request for the same window."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    first = _queue_admission(service, fixture.binding_id)
    assert jobs_manager.cancel_job(first.job_id, reason="finished admission")
    second = _queue_admission(service, fixture.binding_id)
    assert second.queued and second.job_id != first.job_id
    assert second.idempotency_key != first.idempotency_key
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 2


@pytest.mark.parametrize("terminal", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("archived", [False, True])
def test_durable_admission_reconciles_terminal_live_and_archived_jobs(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
    terminal: str, archived: bool,
) -> None:
    """Only confirmed terminal authority permits another identity for the same window."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    first = _queue_admission(service, fixture.binding_id)
    if terminal == "completed":
        claimed = jobs_manager.acquire_next_job(domain="calendar", queue="default", worker_id="terminal-test", lease_seconds=60)
        assert claimed is not None and claimed["id"] == first.job_id
        assert jobs_manager.complete_job(first.job_id, enforce=False)
    elif terminal == "failed":
        claimed = jobs_manager.acquire_next_job(domain="calendar", queue="default", worker_id="terminal-test", lease_seconds=60)
        assert claimed is not None and claimed["id"] == first.job_id
        assert jobs_manager.fail_job(first.job_id, error="terminal", retryable=False, enforce=False)
    else:
        assert jobs_manager.cancel_job(first.job_id, reason="terminal")
    if archived:
        monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")
        assert jobs_manager.prune_jobs(statuses=[terminal], older_than_days=0, domain="calendar") == 1
        assert jobs_manager.get_job(first.job_id) is None
    second = _queue_admission(service, fixture.binding_id)
    assert second.queued and second.idempotency_key != first.idempotency_key
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 2


def test_durable_admission_missing_recorded_job_retains_authority(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temporarily unavailable recorded Job cannot be interpreted as terminal."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    first = _queue_admission(service, fixture.binding_id)
    monkeypatch.setattr(jobs_manager, "get_job_or_archived_by_uuid", lambda *args, **kwargs: None)
    with pytest.raises(CalendarSyncError, match="unavailable"):
        _queue_admission(service, fixture.binding_id)
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert calendar_db.get_sync_admission(fixture.binding_id).idempotency_key == first.idempotency_key


def test_durable_admission_cannot_be_dispatched_from_another_tenant(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recoverable intent preserves its original tenant and owner authority."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    with monkeypatch.context() as patch:
        def fail(command: Any) -> Any:
            raise RuntimeError("dispatch")
        patch.setattr(jobs_manager, "admit_idempotent_operation", fail)
        with pytest.raises(RuntimeError, match="dispatch"):
            _queue_admission(service, fixture.binding_id)
    with pytest.raises(CalendarPermissionDenied):
        _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager, tenant_id="other"), fixture.binding_id)
    assert jobs_manager.count_jobs(domain="calendar") == 0
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


def test_durable_admission_late_correlation_cannot_overwrite_successor(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """A late retry may not attach or release a subsequent binding reservation."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    first = _queue_admission(service, fixture.binding_id)
    old = calendar_db.get_sync_admission(fixture.binding_id)
    assert jobs_manager.cancel_job(first.job_id)
    second = _queue_admission(service, fixture.binding_id)
    assert not calendar_db.attach_sync_admission_job(old.admission_id, first.job_id, old.job_uuid)
    calendar_db.release_sync_admission(old.admission_id)
    assert calendar_db.get_sync_admission(fixture.binding_id).job_id == second.job_id


def test_durable_admission_audit_write_then_failure_rolls_back_reservation(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nested audit write must roll back with admission when its readback fails."""
    fixture = _create_sync_fixture(calendar_db)
    original = calendar_db.record_sync_event

    def fail_after_write(**kwargs: Any) -> Any:
        original(**kwargs)
        raise RuntimeError("audit readback")

    monkeypatch.setattr(calendar_db, "record_sync_event", fail_after_write)
    with pytest.raises(RuntimeError, match="audit readback"):
        _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager), fixture.binding_id)
    assert calendar_db.get_sync_admission(fixture.binding_id) is None
    assert calendar_db.list_sync_events(binding_id=fixture.binding_id) == []
    assert jobs_manager.count_jobs(domain="calendar") == 0


@pytest.mark.asyncio
async def test_durable_admission_recovery_rotates_past_failing_bindings(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bounded recovery batch cannot starve later bindings behind repeated failures."""
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixtures = [_create_sync_fixture(calendar_db) for _ in range(3)]
    create = jobs_manager.admit_idempotent_operation
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    for fixture in fixtures:
        calendar_db.update_external_binding(fixture.binding_id, {"sync_interval_minutes": None})
        with monkeypatch.context() as patch:
            def fail(command: Any) -> Any:
                raise RuntimeError("dispatch")
            patch.setattr(jobs_manager, "admit_idempotent_operation", fail)
            with pytest.raises(RuntimeError, match="dispatch"):
                _queue_admission(service, fixture.binding_id)

    def fail_first_two(command: Any) -> Any:
        if command.job.payload["binding_id"] != fixtures[-1].binding_id:
            raise RuntimeError("persistent dispatch failure")
        return create(command)

    monkeypatch.setattr(jobs_manager, "admit_idempotent_operation", fail_first_two)
    recovered = []
    for _ in range(3):
        recovered.extend(await queue_due_calendar_sync_jobs(db=calendar_db, job_manager=jobs_manager, limit=1))
    assert [response.binding_id for response in recovered] == [fixtures[-1].binding_id]


@pytest.mark.asyncio
async def test_durable_admission_scheduler_recovers_manual_only_pending_dispatch(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A manually admitted dispatch crash recovers without enabling periodic polling."""
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixture = _create_sync_fixture(calendar_db)
    calendar_db.update_external_binding(fixture.binding_id, {"sync_interval_minutes": None})

    def fail(command: Any) -> Any:
        raise RuntimeError("dispatch crash")

    with monkeypatch.context() as patch:
        patch.setattr(jobs_manager, "admit_idempotent_operation", fail)
        with pytest.raises(RuntimeError, match="dispatch crash"):
            _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager), fixture.binding_id)
    recovered = await queue_due_calendar_sync_jobs(db=calendar_db, job_manager=jobs_manager)
    assert len(recovered) == 1 and recovered[0].queued
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert calendar_db.get_external_binding(fixture.binding_id).sync_interval_minutes is None
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


def test_durable_admission_post_dispatch_crash_recovers_existing_job(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash after Jobs commit repairs correlation without another dispatch or audit."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)

    def crash(*args: Any) -> Any:
        raise RuntimeError("post-dispatch crash")

    with monkeypatch.context() as patch:
        patch.setattr(calendar_db, "attach_sync_admission_job", crash)
        with pytest.raises(RuntimeError, match="post-dispatch crash"):
            _queue_admission(service, fixture.binding_id)
    admission = calendar_db.get_sync_admission(fixture.binding_id)
    assert admission is not None and admission.job_id is None
    recovered = _queue_admission(service, fixture.binding_id, "scheduled")
    assert recovered.idempotency_key == admission.idempotency_key
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


def test_durable_admission_revalidates_account_before_committing_reservation(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Revocation between service lookup and reservation cannot admit runnable work."""
    fixture = _create_sync_fixture(calendar_db)
    reserve = calendar_db.reserve_sync_admission

    def revoked(**kwargs: Any) -> Any:
        calendar_db.revoke_external_account(fixture.account_id)
        return reserve(**kwargs)

    monkeypatch.setattr(calendar_db, "reserve_sync_admission", revoked)
    with pytest.raises(CalendarValidationError, match="not active"):
        _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager), fixture.binding_id)
    assert jobs_manager.count_jobs(domain="calendar") == 0
    assert calendar_db.get_sync_admission(fixture.binding_id) is None


def test_durable_admission_stale_dispatch_cannot_recreate_archived_work(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed dispatcher cannot resurrect an admission retired by another caller."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    other_manager = JobManager(jobs_manager.db_path)
    other = CalendarService(db=calendar_db, job_manager=other_manager)
    original = jobs_manager.admit_idempotent_operation
    successor: list[Any] = []
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "1")

    def delayed(command: Any) -> Any:
        first = _queue_admission(other, fixture.binding_id)
        assert other_manager.cancel_job(first.job_id)
        assert other_manager.prune_jobs(statuses=["cancelled"], older_than_days=0, domain="calendar") == 1
        successor.append(_queue_admission(other, fixture.binding_id))
        return original(command)

    monkeypatch.setattr(jobs_manager, "admit_idempotent_operation", delayed)
    stale = _queue_admission(service, fixture.binding_id)
    queued = jobs_manager.list_jobs(domain="calendar", status="queued")
    assert len(queued) == 1 and queued[0]["id"] == successor[0].job_id
    assert not stale.queued
    assert calendar_db.get_sync_admission(fixture.binding_id).job_id == successor[0].job_id


def test_durable_admission_legacy_processing_to_retry_transition_cannot_escape_scan(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy active Job moving between active states stays in the same scan population."""
    fixture = _create_sync_fixture(calendar_db)
    legacy = jobs_manager.create_job(domain="calendar", queue="default", job_type="calendar_sync", owner_user_id="1",
                                     payload=_sync_job(fixture.binding_id)["payload"], idempotency_key="legacy-retry")
    claimed = jobs_manager.acquire_next_job(domain="calendar", queue="default", worker_id="legacy", lease_seconds=60)
    assert claimed is not None and claimed["id"] == legacy["id"]
    original = jobs_manager.list_jobs
    transitioned = False

    def move_after_scan(**kwargs: Any) -> Any:
        nonlocal transitioned
        rows = original(**kwargs)
        if not transitioned:
            transitioned = True
            assert jobs_manager.fail_job(legacy["id"], error="retry", retryable=True, backoff_seconds=60, enforce=False)
        return rows

    monkeypatch.setattr(jobs_manager, "list_jobs", move_after_scan)
    response = _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager), fixture.binding_id)
    assert response.job_id == legacy["id"] and not response.queued
    assert jobs_manager.count_jobs(domain="calendar") == 1


@pytest.mark.asyncio
async def test_durable_admission_recovery_does_not_resubmit_completed_manual_work(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recovery reconciles the admitted identity, not a fresh request after terminal completion."""
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixture = _create_sync_fixture(calendar_db)
    calendar_db.update_external_binding(fixture.binding_id, {"sync_interval_minutes": None})

    def crash(*args: Any) -> Any:
        raise RuntimeError("correlation crash")

    with monkeypatch.context() as patch:
        patch.setattr(calendar_db, "attach_sync_admission_job", crash)
        with pytest.raises(RuntimeError, match="correlation crash"):
            _queue_admission(CalendarService(db=calendar_db, job_manager=jobs_manager), fixture.binding_id)
    claimed = jobs_manager.acquire_next_job(domain="calendar", queue="default", worker_id="recovery", lease_seconds=60)
    assert claimed is not None
    assert jobs_manager.complete_job(claimed["id"], enforce=False)
    await queue_due_calendar_sync_jobs(db=calendar_db, job_manager=jobs_manager)
    assert jobs_manager.list_jobs(domain="calendar", status="queued") == []
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1
    assert calendar_db.get_sync_admission(fixture.binding_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("different_window", [False, True], ids=["same-window", "different-window"])
async def test_concurrent_manual_and_scheduled_sync_admit_one_binding_job_and_audit(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
    different_window: bool,
) -> None:
    """A stale scheduler lookup cannot admit another job after concurrent manual admission."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixture = _create_sync_fixture(calendar_db)
    scheduler_path = calendar_db.db_path.parent / "calendar_alias.db"
    scheduler_path.symlink_to(calendar_db.db_path)
    scheduler_db = CalendarDatabase(db_path=scheduler_path)
    scheduler_jobs = JobManager(jobs_manager.db_path)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    loop = asyncio.get_running_loop()
    empty_lookup = asyncio.Event()
    manual_finished = threading.Event()
    original_lookup = worker._active_job_for_binding

    def pause_empty_scheduler_lookup(**kwargs: Any) -> dict[str, Any] | None:
        """Force the reviewer's stale read without replacing real Jobs queries."""
        existing = original_lookup(**kwargs)
        if kwargs["job_manager"] is scheduler_jobs and existing is None:
            loop.call_soon_threadsafe(empty_lookup.set)
            # The stale pre-reservation scan must not defeat the unique binding insert.
            manual_finished.wait(1)
        return existing

    def submit_manual() -> Any:
        """Call the real service from a different DB thread and always release the holder."""
        try:
            return service.queue_binding_sync(
                actor_user_id=1, binding_id=fixture.binding_id, reason="manual",
                window_start="2026-06-01T00:00:00+00:00" if different_window else "2026-05-27T12:00:00+00:00",
                window_end="2026-06-08T00:00:00+00:00" if different_window else "2026-07-10T12:00:00+00:00",
            )
        finally:
            manual_finished.set()

    monkeypatch.setattr(worker, "_active_job_for_binding", pause_empty_scheduler_lookup)
    scheduled = asyncio.create_task(queue_due_calendar_sync_jobs(
        db=scheduler_db, job_manager=scheduler_jobs, now=datetime(2026, 6, 10, 12, tzinfo=timezone.utc),
    ))
    manual: asyncio.Task[Any] | None = None
    try:
        await asyncio.wait_for(empty_lookup.wait(), timeout=3)
        manual = asyncio.create_task(asyncio.to_thread(submit_manual))
        scheduled_responses, manual_response = await asyncio.wait_for(
            asyncio.gather(scheduled, manual), timeout=5,
        )
    finally:
        manual_finished.set()
        await asyncio.gather(scheduled, *([manual] if manual is not None else []), return_exceptions=True)

    jobs = jobs_manager.list_jobs(domain="calendar", job_type="calendar_sync")
    audits = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    assert (len(jobs), len(audits)) == (1, 1), "Concurrent admission persisted duplicate Jobs/audits"
    responses = [*scheduled_responses, manual_response]
    assert len(responses) == 2
    assert {response.job_id for response in responses} == {jobs[0]["id"]}
    assert sorted(response.status for response in responses) == ["already_active", "queued"]
    assert sum(response.queued for response in responses) == 1
    assert {response.idempotency_key for response in responses} == {
        calendar_db.get_sync_admission(fixture.binding_id).idempotency_key,
    }
    assert audits[0].event_type == "sync_queued"
    assert json.loads(audits[0].metadata_json) == {
        "job_id": jobs[0]["id"], "reason": jobs[0]["payload"]["reason"],
        "admission_id": calendar_db.get_sync_admission(fixture.binding_id).admission_id,
    }
    assert "app-secret" not in json.dumps(jobs[0]["payload"])


@pytest.mark.asyncio
async def test_unrelated_binding_admission_progresses_while_first_jobs_dispatch_is_blocked(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent bindings must not hold SQLite write authority across Jobs dispatch."""
    first = _create_sync_fixture(calendar_db)
    second = _create_sync_fixture(calendar_db, remote_calendar_url="https://caldav.example.test/second/")
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    loop = asyncio.get_running_loop()
    audit_started = asyncio.Event()
    release = threading.Event()
    original_dispatch = jobs_manager.admit_idempotent_operation

    def blocked_dispatch(command: Any) -> Any:
        if command.job.payload["binding_id"] == first.binding_id:
            loop.call_soon_threadsafe(audit_started.set)
            if not release.wait(5):
                raise TimeoutError("First binding audit was not released")
        return original_dispatch(command)

    def queue(binding_id: int) -> Any:
        return service.queue_binding_sync(
            actor_user_id=1, binding_id=binding_id, reason="manual",
            window_start="2026-06-01T00:00:00+00:00", window_end="2026-06-08T00:00:00+00:00",
        )

    monkeypatch.setattr(jobs_manager, "admit_idempotent_operation", blocked_dispatch)
    first_task = asyncio.create_task(asyncio.to_thread(queue, first.binding_id))
    second_task: asyncio.Task[Any] | None = None
    try:
        await asyncio.wait_for(audit_started.wait(), timeout=3)
        second_task = asyncio.create_task(asyncio.to_thread(queue, second.binding_id))
        second_response = await asyncio.wait_for(asyncio.shield(second_task), timeout=2)
        assert not first_task.done()
        assert second_response.queued
        assert len(calendar_db.list_sync_events(binding_id=second.binding_id)) == 1
    finally:
        release.set()
        await asyncio.gather(first_task, *([second_task] if second_task is not None else []), return_exceptions=True)

    assert jobs_manager.count_jobs(domain="calendar", job_type="calendar_sync") == 2
    assert len(calendar_db.list_sync_events(binding_id=first.binding_id)) == 1


@pytest.mark.parametrize("failure_phase", [None, "create", "audit"], ids=["success", "create-failure", "audit-failure"])
def test_binding_admission_retains_only_recoverable_state_after_failure(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
    failure_phase: str | None,
) -> None:
    """Audit rollback and dispatch recovery cannot wedge or duplicate later admission."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    def queue() -> Any:
        return service.queue_binding_sync(
            actor_user_id=1, binding_id=fixture.binding_id, reason="manual",
            window_start="2026-06-01T00:00:00+00:00", window_end="2026-06-08T00:00:00+00:00",
        )

    if failure_phase is not None:
        repository = jobs_manager if failure_phase == "create" else calendar_db
        operation_name = "admit_idempotent_operation" if failure_phase == "create" else "record_sync_event"

        def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Admission persistence failed")

        with monkeypatch.context() as patch:
            patch.setattr(repository, operation_name, fail)
            with pytest.raises(RuntimeError, match="Admission persistence failed"):
                queue()
    else:
        assert queue().queued
    response = queue()
    assert response.status == ("already_active" if failure_phase is None else "queued")
    assert jobs_manager.count_jobs(domain="calendar") == 1
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 1


def test_binding_admission_preserves_processing_retry_and_later_window_jobs(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """An active retry still suppresses other windows/reasons; terminal work permits a later window."""
    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)

    def queue(reason: str, month: str) -> Any:
        return service.queue_binding_sync(
            actor_user_id=1, binding_id=fixture.binding_id, reason=reason,
            window_start=f"2026-{month}-01T00:00:00+00:00", window_end=f"2026-{month}-08T00:00:00+00:00",
        )

    first = queue("scheduled", "06")
    claimed = jobs_manager.acquire_next_job(
        domain="calendar", queue="default", job_type="calendar_sync", worker_id="calendar-test", lease_seconds=60,
    )
    assert claimed is not None and claimed["id"] == first.job_id and claimed["status"] == "processing"
    processing = queue("manual", "07")
    assert processing.job_id == first.job_id and processing.status == "already_active"
    assert jobs_manager.fail_job(first.job_id, error="retry sync", retryable=True, backoff_seconds=60, enforce=False)
    retried = jobs_manager.get_job(first.job_id)
    assert retried is not None and retried["status"] == "queued" and retried["retry_count"] == 1
    retry_response = queue("manual", "07")
    assert retry_response.job_id == first.job_id and not retry_response.queued
    assert jobs_manager.cancel_job(first.job_id, reason="terminal retry cancellation")
    later = queue("manual", "07")
    assert later.queued and later.job_id != first.job_id
    assert jobs_manager.count_jobs(domain="calendar") == 2
    assert len(calendar_db.list_sync_events(binding_id=fixture.binding_id)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(("binding_patch", "next_scan_at", "due_at"), [
    ({}, "2026-06-05T18:00:00+00:00", "2026-06-05T18:00:00+00:00"),
    ({"sync_interval_minutes": 15}, "2026-06-05T17:15:00+00:00", "2026-06-05T17:15:00+00:00"),
    ({"sync_interval_minutes": 120}, "2026-06-05T19:00:00+00:00", "2026-06-05T19:00:00+00:00"),
    ({"sync_interval_minutes": None}, None, "2026-06-06T17:00:00+00:00"),
    ({"sync_interval_minutes": 0}, None, "2026-06-06T17:00:00+00:00"),
    ({"sync_interval_minutes": -1}, None, "2026-06-06T17:00:00+00:00"),
], ids=["omitted-hourly", "fifteen-minutes", "two-hours", "manual-null", "legacy-zero", "legacy-negative"])
async def test_polling_cadence_success_schedules_only_positive_intervals(
    calendar_db: CalendarDatabase, jobs_manager: JobManager, monkeypatch: pytest.MonkeyPatch,
    binding_patch: dict[str, int | None], next_scan_at: str | None, due_at: str,
) -> None:
    """Manual jobs remain available, but success cannot cause immediate periodic resync."""
    from tldw_Server_API.app.core.Calendar import calendar_sync_worker as worker

    fixture = _create_sync_fixture(calendar_db)
    if binding_patch:
        calendar_db.update_external_binding(fixture.binding_id, binding_patch)
    monkeypatch.setattr(worker, "_utcnow_iso", lambda: "2026-06-05T17:00:00+00:00")
    queued = worker.queue_calendar_binding_sync(
        db=calendar_db, job_manager=jobs_manager, actor_user_id=1, tenant_id="default",
        binding_id=fixture.binding_id, reason="manual",
        window_start="2026-06-01T00:00:00+00:00", window_end="2026-06-08T00:00:00+00:00",
    )
    await worker.handle_calendar_sync_job(
        jobs_manager.get_job(queued.job_id), db=calendar_db, provider=_FakeProvider(events=[_sync_event("cadence")]),
    )

    binding = calendar_db.get_external_binding(fixture.binding_id)
    assert binding.last_sync_at == "2026-06-05T17:00:00+00:00"
    assert binding.next_scan_at == next_scan_at
    assert calendar_db.list_sync_events(binding_id=binding.id)[0].status == "success"
    assert calendar_db.list_sync_enabled_bindings_due_for_scan(now_iso="2026-06-05T17:00:01+00:00") == []
    due = calendar_db.list_sync_enabled_bindings_due_for_scan(now_iso=due_at)
    assert [row.id for row in due] == ([binding.id] if next_scan_at is not None else [])


@pytest.mark.asyncio
async def test_polling_cadence_scheduler_skips_manual_binding_without_jobs(
    calendar_db: CalendarDatabase, jobs_manager: JobManager,
) -> None:
    """Repeated periodic scans of a legacy null interval must create no Jobs or audits."""
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixture = _create_sync_fixture(calendar_db)
    calendar_db.update_external_binding(fixture.binding_id, sync_interval_minutes=None)
    for minute in [0, 1]:
        queued = await queue_due_calendar_sync_jobs(
            db=calendar_db, job_manager=jobs_manager, now=datetime(2026, 6, 5, 17, minute, tzinfo=timezone.utc),
        )
        assert queued == []
    assert jobs_manager.count_jobs(domain="calendar", queue="default", job_type="calendar_sync") == 0
    assert calendar_db.list_sync_events(binding_id=fixture.binding_id) == []


@pytest.mark.asyncio
async def test_due_calendar_sync_scheduler_queues_sanitized_scheduled_job(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.services.calendar_sync_scheduler import queue_due_calendar_sync_jobs

    fixture = _create_sync_fixture(calendar_db)

    queued = await queue_due_calendar_sync_jobs(
        db=calendar_db,
        job_manager=jobs_manager,
        now=datetime(2026, 6, 10, 12, 0, tzinfo=timezone.utc),
    )

    assert len(queued) == 1
    assert queued[0].binding_id == fixture.binding_id
    assert queued[0].queued is True
    job = jobs_manager.get_job(queued[0].job_id)
    assert job["payload"] == {
        "binding_id": fixture.binding_id,
        "window_start": "2026-05-27T12:00:00+00:00",
        "window_end": "2026-07-10T12:00:00+00:00",
        "reason": "scheduled",
    }
    assert "app-secret" not in json.dumps(job["payload"])
    assert "secret_ref" not in json.dumps(job["payload"])


@pytest.mark.asyncio
async def test_worker_resolves_credentials_and_upserts_provider_items_preserving_local_context(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    provider_item = calendar_db.upsert_provider_item(
        calendar_id=fixture.calendar_id,
        external_binding_id=fixture.binding_id,
        source_uid="event-1",
        title="Old title",
        start_at="2026-06-05T16:00:00+00:00",
        end_at="2026-06-05T17:00:00+00:00",
        source_etag="old-etag",
        source_ctag="old-ctag",
    )
    local_tags = CalendarService(db=calendar_db).update_local_tags(
        actor_user_id=1,
        item_id=provider_item.id,
        tags=["important"],
    )
    annotation = calendar_db.create_annotation(
        calendar_item_id=provider_item.id,
        author_user_id=1,
        body="Keep local note",
    )
    link = calendar_db.create_link(
        calendar_item_id=provider_item.id,
        target_type="note",
        target_id="note-1",
        label="Notes",
    )
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    queued = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )
    job = jobs_manager.get_job(queued.job_id)
    provider = _FakeProvider(
        events=[
            CalDavEvent(
                uid="event-1",
                title="Updated title",
                start_at="2026-06-05T18:00:00+00:00",
                end_at="2026-06-05T19:00:00+00:00",
                location="Room 2",
                description="Remote description",
                source_updated_at="2026-06-04T12:00:00+00:00",
                provider_payload={"etag": "new-etag", "ctag": "new-ctag"},
            )
        ]
    )

    result = await handle_calendar_sync_job(job, db=calendar_db, provider=provider)

    updated = calendar_db.get_item(provider_item.id)
    assert result["items_seen"] == 1
    assert result["items_upserted"] == 1
    assert result["items_tombstoned"] == 0
    assert provider.calls[0]["password"] == "app-secret"
    assert provider.calls[0]["window_start"] == "2026-06-01T00:00:00+00:00"
    assert updated.title == "Updated title"
    assert updated.location == "Room 2"
    assert updated.source_etag == "new-etag"
    assert updated.source_ctag == "new-ctag"
    assert calendar_db.get_annotation(local_tags.id).tags_json == '["important"]'
    assert calendar_db.get_annotation(annotation.id).body == "Keep local note"
    assert calendar_db.get_link(link.id).label == "Notes"


@pytest.mark.asyncio
async def test_worker_does_not_infer_remote_deletion_from_bounded_poll(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db, owner_user_id=1)
    stale_item = calendar_db.upsert_provider_item(
        calendar_id=fixture.calendar_id,
        external_binding_id=fixture.binding_id,
        source_uid="stale-event",
        title="Stale event",
        start_at="2026-06-05T16:00:00+00:00",
    )
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    queued = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    await handle_calendar_sync_job(
        jobs_manager.get_job(queued.job_id), db=calendar_db, provider=_FakeProvider(events=[])
    )

    preserved = calendar_db.get_item(stale_item.id, include_deleted=True)
    assert preserved.remote_deleted_at is None
    other_user_service = CalendarService(db=calendar_db)
    assert other_user_service.list_calendars(actor_user_id=2) == []


@pytest.mark.asyncio
async def test_queued_job_does_not_fetch_after_binding_is_disabled(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    queued = CalendarService(db=calendar_db, job_manager=jobs_manager).queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )
    calendar_db.update_external_binding(fixture.binding_id, {"sync_enabled": False})
    provider = _FakeProvider(events=[])

    with pytest.raises(CalendarValidationError, match="not enabled for sync"):
        await handle_calendar_sync_job(jobs_manager.get_job(queued.job_id), db=calendar_db, provider=provider)

    assert provider.calls == []


@pytest.mark.asyncio
async def test_worker_rejects_binding_on_different_origin_before_fetch(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db, remote_calendar_url="https://other.example.test/calendar/")
    queued = CalendarService(db=calendar_db, job_manager=jobs_manager).queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )
    provider = _FakeProvider(events=[])

    with pytest.raises(CalendarValidationError, match="same origin"):
        await handle_calendar_sync_job(jobs_manager.get_job(queued.job_id), db=calendar_db, provider=provider)

    assert provider.calls == []


@pytest.mark.asyncio
async def test_worker_records_failure_and_updates_binding_error(
    calendar_db: CalendarDatabase,
    jobs_manager: JobManager,
) -> None:
    from tldw_Server_API.app.core.Calendar.calendar_sync_worker import handle_calendar_sync_job

    fixture = _create_sync_fixture(calendar_db)
    service = CalendarService(db=calendar_db, job_manager=jobs_manager)
    queued = service.queue_binding_sync(
        actor_user_id=1,
        binding_id=fixture.binding_id,
        reason="manual",
        window_start="2026-06-01T00:00:00+00:00",
        window_end="2026-06-08T00:00:00+00:00",
    )

    with pytest.raises(RuntimeError, match="provider down"):
        await handle_calendar_sync_job(
            jobs_manager.get_job(queued.job_id),
            db=calendar_db,
            provider=_FakeProvider(exc=RuntimeError("provider down")),
        )

    binding = calendar_db.get_external_binding(fixture.binding_id)
    events = calendar_db.list_sync_events(binding_id=fixture.binding_id)
    assert binding.last_error == "provider down"
    assert events[0].status == "failed"
    assert events[0].error_message == "provider down"
