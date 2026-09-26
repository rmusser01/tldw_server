"""Regressions for exact second-fold seeds and instant-based query windows."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from tldw_Server_API.app.core.Calendar.constants import MAX_QUERY_WINDOW_DAYS
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.recurrence import (
    LocalRecurrenceRule,
    RecurrenceOccurrence,
    expand_recurrence,
    expand_recurrence_set,
    validate_query_window,
)

ZONE = "America/Los_Angeles"
START = "2026-11-01T01:30:00-08:00"
END = "2026-11-01T02:00:00-08:00"
pytestmark = pytest.mark.unit


def _instants(occurrences: list[RecurrenceOccurrence]) -> list[tuple[str, str | None]]:
    """Read UTC instants so same-zone datetime equality cannot mask fold loss."""
    return [
        (
            value.start_at.astimezone(timezone.utc).isoformat(),
            value.end_at.astimezone(timezone.utc).isoformat() if value.end_at else None,
        )
        for value in occurrences
    ]


@pytest.mark.parametrize("api", ["direct", "set", "provider"])
@pytest.mark.parametrize(
    ("window_start", "window_end"),
    [("2026-11-01T08:00:00Z", "2026-11-01T10:30:00Z"), ("2026-11-01T09:20:00Z", "2026-11-01T10:10:00Z")],
)
def test_second_fold_count_one_keeps_only_exact_seed(api: str, window_start: str, window_end: str) -> None:
    """Neither broad nor narrow windows may substitute or duplicate the DTSTART instant."""
    arguments = {
        "master_start": START,
        "master_end": END,
        "timezone_name": ZONE,
        "window_start": window_start,
        "window_end": window_end,
    }
    if api == "direct":
        occurrences = expand_recurrence(**arguments, recurrence=LocalRecurrenceRule(frequency="daily", count=1))
    else:
        occurrences = expand_recurrence_set(
            **arguments,
            rrule_text="FREQ=DAILY;COUNT=1",
            rdates=[],
            exdates=[],
            provider_rule=api == "provider",
        )
    assert _instants(occurrences) == [("2026-11-01T09:30:00+00:00", "2026-11-01T10:00:00+00:00")]


@pytest.mark.parametrize("api", ["direct", "set", "provider"])
def test_second_fold_datetime_seed_preserves_subsecond_instant(api: str) -> None:
    """Rebuilding DTSTART cannot replace its exact datetime with dateutil's rounded seed."""
    start = datetime(2026, 11, 1, 1, 30, microsecond=123456, tzinfo=ZoneInfo(ZONE), fold=1)
    arguments = {
        "master_start": start,
        "master_end": END,
        "timezone_name": ZONE,
        "window_start": "2026-11-01T08:00:00Z",
        "window_end": "2026-11-01T10:30:00Z",
    }
    if api == "direct":
        occurrences = expand_recurrence(**arguments, recurrence=LocalRecurrenceRule(frequency="daily", count=1))
    else:
        occurrences = expand_recurrence_set(
            **arguments,
            rrule_text="FREQ=DAILY;COUNT=1",
            rdates=[],
            exdates=[],
            provider_rule=api == "provider",
        )
    assert _instants(occurrences) == [("2026-11-01T09:30:00.123456+00:00", "2026-11-01T10:00:00+00:00")]


@pytest.mark.parametrize("api", ["direct", "set", "provider"])
@pytest.mark.parametrize(
    ("frequency", "next_start"),
    [
        ("daily", "2026-11-02T09:30:00+00:00"),
        ("weekly", "2026-11-08T09:30:00+00:00"),
        ("monthly", "2026-12-01T09:30:00+00:00"),
    ],
)
def test_second_fold_seed_consumes_one_count_slot(api: str, frequency: str, next_start: str) -> None:
    """Repairing the seed cannot add an occurrence or consume the next valid local date."""
    arguments = {
        "master_start": START,
        "master_end": None,
        "timezone_name": ZONE,
        "window_start": "2026-11-01T08:00:00Z",
        "window_end": "2026-12-02T00:00:00Z",
    }
    if api == "direct":
        occurrences = expand_recurrence(**arguments, recurrence=LocalRecurrenceRule(frequency=frequency, count=2))
    else:
        occurrences = expand_recurrence_set(
            **arguments,
            rrule_text=f"FREQ={frequency.upper()};COUNT=2",
            rdates=[],
            exdates=[],
            provider_rule=api == "provider",
        )
    assert _instants(occurrences) == [("2026-11-01T09:30:00+00:00", None), (next_start, None)]


@pytest.mark.parametrize(
    ("frequency", "next_start"),
    [
        ("SECONDLY", "2026-11-01T09:30:01+00:00"),
        ("MINUTELY", "2026-11-01T09:31:00+00:00"),
        ("HOURLY", "2026-11-01T10:30:00+00:00"),
    ],
)
def test_productive_subday_provider_rule_stays_after_second_fold_seed(frequency: str, next_start: str) -> None:
    """A counted subday series cannot rewind into the first repeated hour."""
    occurrences = expand_recurrence_set(
        master_start=START,
        master_end=None,
        timezone_name=ZONE,
        rrule_text=f"FREQ={frequency};COUNT=2",
        rdates=[],
        exdates=[],
        provider_rule=True,
        window_start="2026-11-01T08:00:00Z",
        window_end="2026-11-01T11:00:00Z",
    )
    assert _instants(occurrences) == [("2026-11-01T09:30:00+00:00", None), (next_start, None)]


@pytest.mark.parametrize("provider_rule", [False, True])
@pytest.mark.parametrize(
    ("rdates", "exdates", "expected"),
    [
        (["2026-11-01T09:30:00Z"], [], ["2026-11-01T09:30:00+00:00"]),
        ([], ["2026-11-01T08:30:00Z"], ["2026-11-01T09:30:00+00:00"]),
        ([], ["2026-11-01T09:30:00Z"], []),
        (["2026-11-01T08:30:00Z"], [], ["2026-11-01T08:30:00+00:00", "2026-11-01T09:30:00+00:00"]),
        (["2026-11-01T08:30:00Z"], ["2026-11-01T09:30:00Z"], ["2026-11-01T08:30:00+00:00"]),
    ],
)
def test_second_fold_additions_and_exclusions_match_exact_instants(
    provider_rule: bool,
    rdates: list[str],
    exdates: list[str],
    expected: list[str],
) -> None:
    """Explicit first-fold additions remain distinct; excluding the seed removes no other instant."""
    occurrences = expand_recurrence_set(
        master_start=START,
        master_end=END,
        timezone_name=ZONE,
        rrule_text="FREQ=DAILY;COUNT=1",
        rdates=rdates,
        exdates=exdates,
        provider_rule=provider_rule,
        window_start="2026-11-01T08:00:00Z",
        window_end="2026-11-01T10:30:00Z",
    )
    assert [value[0] for value in _instants(occurrences)] == expected


def test_positive_utc_window_with_reversed_fold_wall_times_is_valid() -> None:
    """08:45Z to 09:15Z is positive despite a backwards same-zone wall clock."""
    zone = ZoneInfo(ZONE)
    validate_query_window(datetime(2026, 11, 1, 1, 45, tzinfo=zone), datetime(2026, 11, 1, 1, 15, tzinfo=zone, fold=1))


def test_negative_utc_window_with_positive_fold_wall_times_is_rejected() -> None:
    """09:15Z to 08:45Z is reversed despite a forwards same-zone wall clock."""
    zone = ZoneInfo(ZONE)
    with pytest.raises(CalendarValidationError):
        validate_query_window(
            datetime(2026, 11, 1, 1, 15, tzinfo=zone, fold=1), datetime(2026, 11, 1, 1, 45, tzinfo=zone)
        )


@pytest.mark.parametrize("shorter_utc", [False, True])
def test_maximum_query_span_is_measured_in_utc(shorter_utc: bool) -> None:
    """The span limit uses UTC even when next year's DST transition falls later."""
    start = datetime(2026, 11, 2, tzinfo=ZoneInfo(ZONE)) if shorter_utc else datetime(2026, 3, 9, tzinfo=ZoneInfo(ZONE))
    end = start + timedelta(days=MAX_QUERY_WINDOW_DAYS, hours=1 if shorter_utc else 0)
    if shorter_utc:
        validate_query_window(start, end)
    else:
        with pytest.raises(CalendarValidationError):
            validate_query_window(start, end)


@pytest.mark.parametrize("api", ["direct", "set"])
@pytest.mark.parametrize("at_start", [False, True])
def test_fold_fix_retains_intentional_overlap_boundaries(api: str, at_start: bool) -> None:
    """Direct overlap is inclusive; recurrence-set overlap remains end-exclusive."""
    arguments = {
        "master_start": START,
        "master_end": END,
        "timezone_name": ZONE,
        "window_start": "2026-11-01T10:00:00Z" if at_start else "2026-11-01T09:00:00Z",
        "window_end": "2026-11-01T10:30:00Z" if at_start else "2026-11-01T09:30:00Z",
    }
    if api == "direct":
        occurrences = expand_recurrence(**arguments, recurrence=LocalRecurrenceRule(frequency="daily", count=1))
    else:
        occurrences = expand_recurrence_set(**arguments, rrule_text="FREQ=DAILY;COUNT=1", rdates=[], exdates=[])
    assert _instants(occurrences) == (
        [("2026-11-01T09:30:00+00:00", "2026-11-01T10:00:00+00:00")] if api == "direct" else []
    )


def test_direct_fold_fix_does_not_normalize_unrelated_spring_wall_candidates() -> None:
    """Direct expansion retains its existing local start representation in a spring gap."""
    occurrences = expand_recurrence(
        master_start="2026-03-07T02:30:00-08:00",
        master_end=None,
        timezone_name=ZONE,
        recurrence=LocalRecurrenceRule(frequency="daily", count=2),
        window_start="2026-03-08T00:00:00Z",
        window_end="2026-03-09T00:00:00Z",
    )
    assert [value.start_at.isoformat() for value in occurrences] == ["2026-03-08T02:30:00-08:00"]


@pytest.mark.parametrize("api", ["direct", "set", "provider"])
def test_until_before_second_fold_seed_cannot_admit_rebuilt_first_fold(api: str) -> None:
    """UNTIL bounds rule instants; sets still include their independent explicit DTSTART."""
    arguments = {
        "master_start": START,
        "master_end": None,
        "timezone_name": ZONE,
        "window_start": "2026-11-01T08:00:00Z",
        "window_end": "2026-11-01T11:00:00Z",
    }
    if api == "direct":
        occurrences = expand_recurrence(
            **arguments,
            recurrence=LocalRecurrenceRule(frequency="daily", until="2026-11-01T08:45:00Z"),
        )
    else:
        occurrences = expand_recurrence_set(
            **arguments,
            rrule_text="FREQ=DAILY;UNTIL=20261101T084500Z",
            rdates=[],
            exdates=[],
            provider_rule=api == "provider",
        )
    assert _instants(occurrences) == ([] if api == "direct" else [("2026-11-01T09:30:00+00:00", None)])


@pytest.mark.parametrize("until", ["20261101T083100Z", "20261101T093100Z"])
def test_subday_provider_until_stops_at_exact_second_fold_instant(until: str) -> None:
    """Correcting subday folds cannot leak candidates past UNTIL in the repeated hour."""
    occurrences = expand_recurrence_set(
        master_start=START,
        master_end=None,
        timezone_name=ZONE,
        provider_rule=True,
        rrule_text=f"FREQ=MINUTELY;COUNT=3;UNTIL={until}",
        rdates=[],
        exdates=[],
        window_start="2026-11-01T08:00:00Z",
        window_end="2026-11-01T11:00:00Z",
    )
    assert _instants(occurrences) == (
        [("2026-11-01T09:30:00+00:00", None)]
        if until == "20261101T083100Z"
        else [
            ("2026-11-01T09:30:00+00:00", None),
            ("2026-11-01T09:31:00+00:00", None),
        ]
    )


@pytest.mark.parametrize("api", ["direct", "set", "provider"])
@pytest.mark.parametrize("previous_day", [False, True])
def test_second_fold_until_keeps_valid_first_fold_daily_candidate(api: str, previous_day: bool) -> None:
    """08:30Z is before a 09:15Z limit despite later-looking local wall time."""
    arguments = {
        "master_start": "2026-10-31T01:30:00-07:00" if previous_day else "2026-11-01T01:30:00-07:00",
        "master_end": None,
        "timezone_name": ZONE,
        "window_start": "2026-11-01T08:00:00Z",
        "window_end": "2026-11-01T11:00:00Z",
    }
    if api == "direct":
        occurrences = expand_recurrence(
            **arguments,
            recurrence=LocalRecurrenceRule(frequency="daily", count=2, until="2026-11-01T09:15:00Z"),
        )
    else:
        occurrences = expand_recurrence_set(
            **arguments,
            rrule_text="FREQ=DAILY;COUNT=2;UNTIL=20261101T091500Z",
            rdates=[],
            exdates=[],
            provider_rule=api == "provider",
        )
    assert _instants(occurrences) == [("2026-11-01T08:30:00+00:00", None)]


def test_second_fold_until_keeps_counted_first_fold_subday_candidates() -> None:
    """The explicit seed cannot hide premature exclusion of later counted minutes."""
    occurrences = expand_recurrence_set(
        master_start="2026-11-01T01:30:00-07:00",
        master_end=None,
        timezone_name=ZONE,
        provider_rule=True,
        rrule_text="FREQ=MINUTELY;COUNT=3;UNTIL=20261101T091500Z",
        rdates=[],
        exdates=[],
        window_start="2026-11-01T08:00:00Z",
        window_end="2026-11-01T11:00:00Z",
    )
    assert _instants(occurrences) == [
        ("2026-11-01T08:30:00+00:00", None),
        ("2026-11-01T08:31:00+00:00", None),
        ("2026-11-01T08:32:00+00:00", None),
    ]


@pytest.mark.parametrize("api", ["direct", "set"])
def test_negative_duration_validation_contract_remains_api_specific(api: str) -> None:
    """The direct API's legacy signed interval is not replaced by set validation."""
    arguments = {
        "master_start": START,
        "master_end": "2026-11-01T01:20:00-08:00",
        "timezone_name": ZONE,
        "window_start": "2026-11-01T09:00:00Z",
        "window_end": "2026-11-01T11:00:00Z",
    }
    if api == "direct":
        occurrences = expand_recurrence(**arguments, recurrence=LocalRecurrenceRule(frequency="daily", count=1))
        assert _instants(occurrences) == [("2026-11-01T09:30:00+00:00", "2026-11-01T09:20:00+00:00")]
    else:
        with pytest.raises(CalendarValidationError):
            expand_recurrence_set(**arguments, rrule_text="FREQ=DAILY;COUNT=1", rdates=[], exdates=[])


@pytest.mark.asyncio
@pytest.mark.parametrize("view_name", ["agenda", "week"])
async def test_reopened_persisted_second_fold_seed_has_one_view_occurrence(tmp_path: Path, view_name: str) -> None:
    """Real persisted agenda/week expansion must not resurrect dateutil's first-fold seed."""
    from tldw_Server_API.app.core.Calendar.calendar_service import CalendarService
    from tldw_Server_API.app.core.Calendar.view_service import CalendarViewFilters, CalendarViewService
    from tldw_Server_API.app.core.DB_Management.Calendar_DB import CalendarDatabase

    path = tmp_path / "folds.db"
    db = CalendarDatabase(db_path=path)
    service = CalendarService(db=db)
    calendar = service.create_calendar(actor_user_id=1, name="Fold", timezone=ZONE)
    item = service.create_item(
        actor_user_id=1,
        calendar_id=calendar.id,
        kind="event",
        title="Second fold",
        start_at=START,
        end_at=END,
        timezone=ZONE,
    )
    db.upsert_recurrence(calendar_item_id=item.id, rrule="FREQ=DAILY;COUNT=1", timezone=ZONE)
    reopened = CalendarDatabase(db_path=path)
    view = CalendarViewService(calendar_service=CalendarService(db=reopened))
    filters = CalendarViewFilters(include_scheduled_tasks=False)
    if view_name == "agenda":
        result = await view.agenda(
            actor_user_id=1,
            start_at="2026-11-01T08:00:00Z",
            end_at="2026-11-01T10:30:00Z",
            filters=filters,
        )
    else:
        result = await view.week(actor_user_id=1, week_start=date(2026, 11, 1), timezone=ZONE, filters=filters)
    assert (result.partial, [(entry.start_at, entry.end_at) for entry in result.items]) == (
        False,
        [(START, END)],
    )
