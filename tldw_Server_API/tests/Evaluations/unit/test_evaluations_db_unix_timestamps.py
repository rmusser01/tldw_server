"""Regression guard for TASK-13299.

`EvaluationsDatabase._ensure_unix_timestamp` converts stored timestamps to Unix
epoch seconds for the OpenAI-compatible `created` field that ADR-014 preserves.

Stored timestamps are UTC — SQLite's `CURRENT_TIMESTAMP` emits `YYYY-MM-DD HH:MM:SS`
with no offset, and a driver may hand back a naive `datetime` for a
`timestamp without time zone` column. `datetime.timestamp()` interprets a *naive*
datetime in the **host local zone**, so on any non-UTC deployment every converted
value was wrong by the host's UTC offset. CI runs UTC, where the error is exactly
zero, so no existing test could see it.

These tests pin the conversion under a non-UTC TZ.

Note on scope: the `except` fallback `int(datetime.now().timestamp())` is **correct**
and is deliberately not changed here. `datetime.now()` returns naive *local* time and
`.timestamp()` interprets naive as local, so the two cancel and it yields the true
epoch. An earlier draft of this finding claimed the fallback carried the same defect;
it does not, and `test_fallback_now_is_correct` pins that.
"""

import time
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

# 2026-09-21 20:00:15 UTC
_UTC_WALL = "2026-09-21 20:00:15"
_TRUE_EPOCH = int(datetime(2026, 9, 21, 20, 0, 15, tzinfo=timezone.utc).timestamp())


@pytest.fixture
def la_timezone(monkeypatch: pytest.MonkeyPatch):
    """Run the body under America/Los_Angeles (UTC-7 on this date)."""
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def _db() -> EvaluationsDatabase:
    return EvaluationsDatabase.__new__(EvaluationsDatabase)


@pytest.mark.parametrize(
    "stored",
    [
        _UTC_WALL,                     # SQLite CURRENT_TIMESTAMP shape
        "2026-09-21T20:00:15",         # ISO, naive
        "2026-09-21T20:00:15Z",        # ISO, explicit UTC
        "2026-09-21T20:00:15+00:00",   # ISO, explicit offset
    ],
    ids=["sqlite_naive", "iso_naive", "iso_zulu", "iso_offset"],
)
def test_string_timestamps_are_utc(la_timezone, stored: str) -> None:
    got = _db()._ensure_unix_timestamp(stored)
    drift = got - _TRUE_EPOCH
    assert got == _TRUE_EPOCH, (
        f"{stored!r} converted to {got}, off by {drift:+d} s "
        f"({drift / 3600:+g} h) -- a naive stored timestamp is being read as "
        "host-local time instead of UTC"
    )


def test_naive_datetime_instance_is_utc(la_timezone) -> None:
    """A driver may return a naive datetime for a timestamp-without-tz column."""
    naive = datetime(2026, 9, 21, 20, 0, 15)
    got = _db()._ensure_unix_timestamp(naive)
    assert got == _TRUE_EPOCH, (
        f"naive datetime converted to {got}, off by {got - _TRUE_EPOCH:+d} s"
    )


def test_aware_datetime_instance_is_unchanged(la_timezone) -> None:
    """Control: tz-aware input was already correct and must stay correct."""
    aware = datetime(2026, 9, 21, 20, 0, 15, tzinfo=timezone.utc)
    assert _db()._ensure_unix_timestamp(aware) == _TRUE_EPOCH


def test_numeric_passthrough_is_unchanged(la_timezone) -> None:
    """Control: an epoch value must not be re-interpreted."""
    assert _db()._ensure_unix_timestamp(_TRUE_EPOCH) == _TRUE_EPOCH
    assert _db()._ensure_unix_timestamp(str(_TRUE_EPOCH)) == _TRUE_EPOCH


def test_fallback_now_is_correct(la_timezone) -> None:
    """`datetime.now().timestamp()` is correct: naive-local and local-interpretation cancel."""
    got = _db()._ensure_unix_timestamp("not a timestamp", fallback_now=True)
    assert got is not None
    assert abs(got - int(time.time())) <= 2, (
        "the unparseable-input fallback drifted from the true epoch"
    )


def test_unparsable_returns_none_without_fallback(la_timezone) -> None:
    assert _db()._ensure_unix_timestamp("not a timestamp") is None
