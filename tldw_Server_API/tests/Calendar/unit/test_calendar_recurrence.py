from __future__ import annotations

import importlib
from datetime import date, datetime, timedelta, timezone
from typing import Any

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


@pytest.mark.unit
@pytest.mark.parametrize("rule", ["FREQ=SECONDLY;BYSETPOS=2", "FREQ=DAILY;BYMONTH=2;BYMONTHDAY=30"])
def test_provider_no_yield_rules_are_rejected_before_dateutil(monkeypatch: pytest.MonkeyPatch, rule: str) -> None:
    """Impossible complex rules fail before entering a potentially non-yielding parser."""
    recurrence = _recurrence_module()

    def must_not_parse(*args: Any, **kwargs: Any) -> None:
        """Fail if validation passes an unsupported rule to dateutil."""
        pytest.fail("unsafe provider rule reached dateutil")

    monkeypatch.setattr(recurrence.rrule, "rrulestr", must_not_parse)
    with pytest.raises(CalendarValidationError):
        recurrence.expand_recurrence_set(
            master_start="2026-06-05T09:00:00Z", master_end=None,
            rrule_text=rule, rdates=[], exdates=[], provider_rule=True,
            window_start="2026-06-05T00:00:00Z", window_end="2026-06-06T00:00:00Z",
        )


def _dates(occurrences):
    values = []
    for occurrence in occurrences:
        value = occurrence.start_at
        values.append(value if isinstance(value, date) and not isinstance(value, datetime) else value.date())
    return values


@pytest.mark.unit
@pytest.mark.parametrize("rule", ["FREQ=MINUTELY", "FREQ=SECONDLY;INTERVAL=60"])
def test_old_frequent_provider_recurrence_seeks_current_window(rule: str) -> None:
    """High-frequency provider rules retain current occurrences without scanning their entire history."""
    recurrence = _recurrence_module()
    occurrences = recurrence.expand_recurrence_set(
        master_start="2020-01-01T00:00:00Z", master_end=None, rrule_text=rule,
        rdates=[], exdates=[], provider_rule=True,
        window_start="2026-06-05T09:00:00Z", window_end="2026-06-05T09:03:00Z",
    )
    assert [value.start_at.isoformat() for value in occurrences] == [
        "2026-06-05T09:00:00+00:00", "2026-06-05T09:01:00+00:00", "2026-06-05T09:02:00+00:00",
    ]


@pytest.mark.unit
def test_seeking_preserves_provider_count_and_interval_phase() -> None:
    """Seeking a counted rule retains its original interval alignment and series end."""
    recurrence = _recurrence_module()
    occurrences = recurrence.expand_recurrence_set(
        master_start="2026-06-05T09:00:00Z", master_end=None,
        rrule_text="FREQ=MINUTELY;INTERVAL=3;COUNT=4", rdates=[], exdates=[], provider_rule=True,
        window_start="2026-06-05T09:07:00Z", window_end="2026-06-05T09:20:00Z",
    )
    assert [value.start_at.isoformat() for value in occurrences] == ["2026-06-05T09:09:00+00:00"]


@pytest.mark.unit
def test_provider_duration_remains_elapsed_hours_for_each_occurrence() -> None:
    """A DST-crossing master must not turn a two-hour duration into three hours on later dates."""
    recurrence = _recurrence_module()
    occurrences = recurrence.expand_recurrence_set(
        master_start="2026-03-08T01:30:00-08:00", master_end="2026-03-08T04:30:00-07:00",
        rrule_text="FREQ=DAILY;COUNT=2", rdates=[], exdates=[], provider_rule=True,
        timezone_name="America/Los_Angeles", duration_text="PT2H",
        window_start="2026-03-08T00:00:00-08:00", window_end="2026-03-10T00:00:00-07:00",
    )
    assert [value.end_at.isoformat() for value in occurrences] == [
        "2026-03-08T04:30:00-07:00", "2026-03-09T03:30:00-07:00",
    ]


@pytest.mark.unit
def test_fall_fold_duration_is_validated_and_filtered_as_instants() -> None:
    """The repeated clock hour cannot make a positive elapsed duration negative or hide overlap."""
    recurrence = _recurrence_module()
    occurrences = recurrence.expand_recurrence_set(
        master_start="2026-11-01T01:45:00-07:00", master_end="2026-11-01T01:15:00-08:00",
        rrule_text="FREQ=DAILY;COUNT=2", rdates=[], exdates=[], provider_rule=True,
        timezone_name="America/Los_Angeles", duration_text="PT30M",
        window_start="2026-11-01T08:50:00Z", window_end="2026-11-03T00:00:00Z",
    )
    assert [value.end_at.isoformat() for value in occurrences] == [
        "2026-11-01T01:15:00-08:00", "2026-11-02T02:15:00-08:00",
    ]


@pytest.mark.unit
def test_rdate_duration_preserves_second_fold_instant() -> None:
    """A duration with zero nominal days retains the explicit second-fold occurrence."""
    recurrence = _recurrence_module()
    occurrences = recurrence.expand_recurrence_set(
        master_start="2026-10-31T01:30:00-07:00", master_end="2026-10-31T02:00:00-07:00",
        rrule_text=None, rdates=["2026-11-01T09:30:00Z"], exdates=[], provider_rule=True,
        timezone_name="America/Los_Angeles", duration_text="PT30M",
        window_start="2026-11-01T09:20:00Z", window_end="2026-11-01T10:10:00Z",
    )
    assert [value.end_at.isoformat() for value in occurrences] == ["2026-11-01T02:00:00-08:00"]


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
