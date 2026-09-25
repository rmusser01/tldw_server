from __future__ import annotations

import importlib
from datetime import date, datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.Calendar.constants import (
    MAX_EXPANDED_OCCURRENCES,
    MAX_QUERY_WINDOW_DAYS,
)
from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError


def _recurrence_module():
    try:
        return importlib.import_module("tldw_Server_API.app.core.Calendar.recurrence")
    except ModuleNotFoundError as exc:
        pytest.fail(f"calendar recurrence module is missing: {exc}")


def _dates(occurrences):
    values = []
    for occurrence in occurrences:
        value = occurrence.start_at
        values.append(value if isinstance(value, date) and not isinstance(value, datetime) else value.date())
    return values


def test_daily_recurrence_respects_count() -> None:
    recurrence = _recurrence_module()
    rule = recurrence.LocalRecurrenceRule(frequency="daily", count=3)
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(hours=1),
        recurrence=rule,
        window_start=start - timedelta(days=1),
        window_end=start + timedelta(days=10),
    )

    assert _dates(occurrences) == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]


def test_weekly_recurrence_respects_weekday_list() -> None:
    recurrence = _recurrence_module()
    rule = recurrence.LocalRecurrenceRule(
        frequency="weekly",
        weekdays=("MO", "WE"),
        count=4,
    )
    start = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(minutes=30),
        recurrence=rule,
        window_start=start - timedelta(days=1),
        window_end=start + timedelta(days=14),
    )

    assert _dates(occurrences) == [
        date(2026, 1, 5),
        date(2026, 1, 7),
        date(2026, 1, 12),
        date(2026, 1, 14),
    ]


def test_monthly_by_date_skips_impossible_dates() -> None:
    recurrence = _recurrence_module()
    rule = recurrence.LocalRecurrenceRule(frequency="monthly", count=4)
    start = datetime(2026, 1, 31, 9, 0, tzinfo=timezone.utc)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(hours=1),
        recurrence=rule,
        window_start=start - timedelta(days=1),
        window_end=datetime(2026, 8, 1, tzinfo=timezone.utc),
    )

    assert _dates(occurrences) == [
        date(2026, 1, 31),
        date(2026, 3, 31),
        date(2026, 5, 31),
        date(2026, 7, 31),
    ]


def test_until_bounds_occurrences() -> None:
    recurrence = _recurrence_module()
    rule = recurrence.LocalRecurrenceRule(
        frequency="daily",
        until=datetime(2026, 1, 3, 23, 59, tzinfo=timezone.utc),
    )
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(hours=1),
        recurrence=rule,
        window_start=start,
        window_end=start + timedelta(days=10),
    )

    assert _dates(occurrences) == [
        date(2026, 1, 1),
        date(2026, 1, 2),
        date(2026, 1, 3),
    ]


def test_all_day_recurrence_remains_date_stable_across_dst() -> None:
    recurrence = _recurrence_module()
    rule = recurrence.LocalRecurrenceRule(frequency="daily", count=4)

    occurrences = recurrence.expand_recurrence(
        master_start=date(2026, 3, 7),
        master_end=date(2026, 3, 8),
        recurrence=rule,
        window_start=date(2026, 3, 7),
        window_end=date(2026, 3, 12),
        timezone_name="America/Los_Angeles",
        all_day=True,
    )

    assert [occurrence.start_at for occurrence in occurrences] == [
        date(2026, 3, 7),
        date(2026, 3, 8),
        date(2026, 3, 9),
        date(2026, 3, 10),
    ]


def test_expansion_rejects_query_windows_over_max_days() -> None:
    recurrence = _recurrence_module()
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

    with pytest.raises(CalendarValidationError):
        recurrence.expand_recurrence(
            master_start=start,
            master_end=start + timedelta(hours=1),
            recurrence=recurrence.LocalRecurrenceRule(frequency="daily"),
            window_start=start,
            window_end=start + timedelta(days=MAX_QUERY_WINDOW_DAYS + 1),
        )


def test_expansion_stops_at_max_expanded_occurrences(monkeypatch) -> None:
    recurrence = _recurrence_module()
    monkeypatch.setattr(recurrence, "MAX_EXPANDED_OCCURRENCES", 5)
    start = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(hours=1),
        recurrence=recurrence.LocalRecurrenceRule(frequency="daily"),
        window_start=start,
        window_end=start + timedelta(days=30),
    )

    assert len(occurrences) == 5
    assert len(occurrences) < MAX_EXPANDED_OCCURRENCES
