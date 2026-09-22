"""Two retry schedules, deliberately not one.

Decorrelated jitter de-synchronises a fleet of HTTP clients retrying the same failing
endpoint. It is the wrong shape for in-process SQLite lock contention, where prev*3
grows faster than capped exponential and the contention window is milliseconds.

These tests pin the schedules the existing call sites already used, so the
consolidation is behaviour-preserving rather than a timing change.
"""

from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Utils.backoff import (
    capped_exponential_delay,
    decorrelated_jitter_delay,
    is_sqlite_locked_error,
    parse_retry_after_seconds,
)


def test_capped_exponential_reproduces_the_sqlite_loop_schedule() -> None:
    """0.05 * 2**n -- the schedule the 28 inlined Prompt Studio loops used."""
    got = [capped_exponential_delay(i, base_s=0.05, jitter=False) for i in range(5)]
    assert got == [0.05, 0.1, 0.2, 0.4, 0.8]


def test_capped_exponential_reproduces_the_transaction_utils_schedule() -> None:
    """0.1 * 2**n, no jitter -- transaction_utils' current behaviour."""
    got = [capped_exponential_delay(i, base_s=0.1, jitter=False) for i in range(1, 5)]
    assert got == [0.2, 0.4, 0.8, 1.6]


def test_capped_exponential_is_bounded() -> None:
    assert capped_exponential_delay(50, base_s=0.05, cap_s=2.0, jitter=False) == 2.0


def test_capped_exponential_jitter_stays_in_band() -> None:
    """Multiplicative 0.5..1.5 around the nominal delay, as 23 of 28 loops had."""
    nominal = capped_exponential_delay(3, base_s=0.05, jitter=False)
    for _ in range(200):
        d = capped_exponential_delay(3, base_s=0.05)
        assert 0.5 * nominal <= d <= 1.5 * nominal


def test_decorrelated_jitter_first_delay_is_the_base() -> None:
    assert decorrelated_jitter_delay(0, 100, 30) == pytest.approx(0.1)


def test_decorrelated_jitter_is_bounded_by_cap_and_grows_from_base() -> None:
    for _ in range(200):
        d = decorrelated_jitter_delay(1.0, 100, 5)
        assert 0.1 <= d <= 5.0


def test_decorrelated_jitter_grows_faster_than_capped_exponential() -> None:
    """The reason the two schedules stay separate."""
    prev = 0.4
    samples = [decorrelated_jitter_delay(prev, 50, 60) for _ in range(500)]
    assert max(samples) > capped_exponential_delay(3, base_s=0.05, jitter=False)


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("database is locked", True),
        ("Database Is Locked", True),   # one inlined copy compared without .lower()
        ("DATABASE IS LOCKED", True),
        ("no such table", False),
    ],
)
def test_sqlite_locked_predicate_is_case_insensitive(msg: str, expected: bool) -> None:
    assert is_sqlite_locked_error(Exception(msg)) is expected


def test_retry_after_accepts_delta_seconds_and_http_date() -> None:
    from datetime import datetime, timezone

    assert parse_retry_after_seconds("5") == 5.0
    assert parse_retry_after_seconds(None) is None
    assert parse_retry_after_seconds("garbage") is None
    now = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after_seconds("Thu, 01 Jan 2026 00:00:10 GMT", now=now) == pytest.approx(10.0)
