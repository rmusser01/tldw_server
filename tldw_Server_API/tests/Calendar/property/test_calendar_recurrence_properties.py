from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone

import pytest
from hypothesis import given, settings, strategies as st


def _recurrence_module():
    try:
        return importlib.import_module("tldw_Server_API.app.core.Calendar.recurrence")
    except ModuleNotFoundError as exc:
        pytest.fail(f"calendar recurrence module is missing: {exc}")


@given(
    start_day=st.integers(min_value=1, max_value=25),
    interval=st.integers(min_value=1, max_value=14),
    count=st.integers(min_value=1, max_value=40),
    window_offset_days=st.integers(min_value=0, max_value=20),
    window_length_days=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=40, deadline=None)
def test_daily_occurrences_are_sorted_unique_and_inside_query_window(
    start_day: int,
    interval: int,
    count: int,
    window_offset_days: int,
    window_length_days: int,
) -> None:
    recurrence = _recurrence_module()
    start = datetime(2026, 1, start_day, 9, 0, tzinfo=timezone.utc)
    window_start = start + timedelta(days=window_offset_days)
    window_end = window_start + timedelta(days=window_length_days)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(hours=1),
        recurrence=recurrence.LocalRecurrenceRule(
            frequency="daily",
            interval=interval,
            count=count,
        ),
        window_start=window_start,
        window_end=window_end,
    )

    starts = [occurrence.start_at for occurrence in occurrences]
    assert starts == sorted(starts)
    assert len(starts) == len(set(starts))
    assert all(window_start <= occurrence.start_at <= window_end for occurrence in occurrences)
    assert len(occurrences) <= count


@given(
    interval=st.integers(min_value=1, max_value=6),
    count=st.integers(min_value=1, max_value=30),
)
@settings(max_examples=25, deadline=None)
def test_weekly_weekday_recurrence_only_emits_requested_weekdays(interval: int, count: int) -> None:
    recurrence = _recurrence_module()
    start = datetime(2026, 1, 5, 9, 0, tzinfo=timezone.utc)

    occurrences = recurrence.expand_recurrence(
        master_start=start,
        master_end=start + timedelta(hours=1),
        recurrence=recurrence.LocalRecurrenceRule(
            frequency="weekly",
            interval=interval,
            weekdays=("MO", "FR"),
            count=count,
        ),
        window_start=start,
        window_end=start + timedelta(days=180),
    )

    assert {occurrence.start_at.weekday() for occurrence in occurrences} <= {0, 4}
