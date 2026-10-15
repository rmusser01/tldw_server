"""Direct duration arithmetic and validation at civil/elapsed time boundaries."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from tldw_Server_API.app.core.Calendar.errors import CalendarValidationError
from tldw_Server_API.app.core.Calendar.temporal import add_ical_duration

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("start", "duration", "expected"),
    [
        (date(2026, 1, 1), "P1D", date(2026, 1, 2)),
        (date(2026, 1, 1), "P2W", date(2026, 1, 15)),
        (datetime(2026, 1, 1, 9), "+P1D", datetime(2026, 1, 2, 9)),
        (datetime(2026, 1, 1, 9), "P2W", datetime(2026, 1, 15, 9)),
        (datetime(2026, 1, 1, 9), "P1DT2H30M15S", datetime(2026, 1, 2, 11, 30, 15)),
        (datetime(2026, 1, 1, 9), "PT24H", datetime(2026, 1, 2, 9)),
        (datetime(2026, 1, 1, 9), "PT90M", datetime(2026, 1, 1, 10, 30)),
        (datetime(2026, 1, 1, 9), "PT15S", datetime(2026, 1, 1, 9, 0, 15)),
    ],
)
def test_add_ical_duration_valid_day_time_and_week_forms(
    start: date | datetime, duration: str, expected: date | datetime,
) -> None:
    """Valid provider durations derive exact ends for dates and naive datetimes."""
    assert add_ical_duration(start, duration) == expected


@pytest.mark.parametrize(
    "duration",
    ["", "garbage", "P", "PT", "P1M", "P1Y", "P1W1D", "PT1.5H", "-P1D", "-PT1H", "PT0S", "P0D", "P1DT"],
)
def test_add_ical_duration_rejects_malformed_negative_and_zero(duration: str) -> None:
    """Invalid, unsupported and nonpositive durations raise the domain validation error."""
    with pytest.raises(CalendarValidationError):
        add_ical_duration(datetime(2026, 1, 1, 9), duration)


@pytest.mark.parametrize("duration", ["PT24H", "P1DT1H", "P1DT0H", "PT1S", "-P1D", "P0D"])
def test_add_ical_duration_rejects_invalid_all_day_forms(duration: str) -> None:
    """All-day durations require positive civil days/weeks, never a time component."""
    with pytest.raises(CalendarValidationError):
        add_ical_duration(date(2026, 1, 1), duration)


@pytest.mark.parametrize(
    ("start", "duration", "expected_end", "expected_fold"),
    [
        ("2026-03-07T12:00:00-08:00", "P1D", "2026-03-08T12:00:00-07:00", 0),
        ("2026-03-07T12:00:00-08:00", "PT24H", "2026-03-08T13:00:00-07:00", 0),
        ("2026-03-01T12:00:00-08:00", "P1W", "2026-03-08T12:00:00-07:00", 0),
        ("2026-03-07T12:00:00-08:00", "P1DT2H", "2026-03-08T14:00:00-07:00", 0),
        ("2026-03-08T01:30:00-08:00", "PT2H", "2026-03-08T04:30:00-07:00", 0),
        ("2026-10-31T12:00:00-07:00", "P1D", "2026-11-01T12:00:00-08:00", 0),
        ("2026-10-31T12:00:00-07:00", "PT24H", "2026-11-01T11:00:00-08:00", 0),
        ("2026-11-01T01:45:00-07:00", "PT30M", "2026-11-01T01:15:00-08:00", 1),
        ("2026-11-01T01:15:00-08:00", "PT30M", "2026-11-01T01:45:00-08:00", 1),
    ],
)
def test_add_ical_duration_preserves_nominal_days_elapsed_time_and_folds(
    start: str, duration: str, expected_end: str, expected_fold: int,
) -> None:
    """Day/week steps preserve wall time; subday steps use instants, including the second fold."""
    zone = ZoneInfo("America/Los_Angeles")
    result = add_ical_duration(datetime.fromisoformat(start).astimezone(zone), duration)
    assert (result.isoformat(), result.fold, result.tzinfo) == (expected_end, expected_fold, zone)


@pytest.mark.parametrize(
    "start",
    [date(9999, 12, 31), datetime(9999, 12, 31, 23, 59)],
)
def test_add_ical_duration_translates_out_of_range_end(start: date | datetime) -> None:
    """Date overflow is exposed as domain validation, not a raw arithmetic exception."""
    with pytest.raises(CalendarValidationError):
        add_ical_duration(start, "P1D")
