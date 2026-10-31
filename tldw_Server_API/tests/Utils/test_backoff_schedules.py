"""Two retry schedules, deliberately not one.

Decorrelated jitter de-synchronises a fleet of HTTP clients retrying the same failing
endpoint. It is the wrong shape for in-process SQLite lock contention, where prev*3
grows faster than capped exponential and the contention window is milliseconds.

These tests pin the schedules the existing call sites already used, so the
consolidation is behaviour-preserving rather than a timing change.
"""

from __future__ import annotations

import socket
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Utils.backoff import (
    capped_exponential_delay,
    classify_http_retry,
    decorrelated_jitter_delay,
    is_dns_resolution_error,
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


# HTTP retriability classification, moved verbatim from core/http_client.py.
_POLICY = SimpleNamespace(
    retry_on_status=(408, 429, 500, 502, 503, 504),
    retry_on_methods=("GET", "HEAD", "OPTIONS"),
    retry_on_unsafe=False,
)


def _flagged() -> Exception:
    exc = ConnectionError("boom")
    exc._tldw_dns_resolution = True  # type: ignore[attr-defined]
    return exc


def _chained_gaierror() -> Exception:
    try:
        raise socket.gaierror(8, "lookup failed")
    except socket.gaierror as inner:
        outer = ConnectionError("connect failed")
        outer.__cause__ = inner
        return outer


@pytest.mark.parametrize(
    "exc, expected",
    [
        (socket.gaierror(8, "x"), True),
        (_chained_gaierror(), True),
        (_flagged(), True),
        (OSError("nodename nor servname provided, or not known"), True),
        (OSError("[Errno -2] Name or service not known"), True),
        (OSError("Temporary failure in name resolution"), True),
        (Exception("DNSResolutionError: example.invalid"), True),
        (ConnectionError("Connection refused"), False),
        (TimeoutError("read timed out"), False),
    ],
)
def test_dns_resolution_detection(exc: Exception, expected: bool) -> None:
    assert is_dns_resolution_error(exc) is expected


@pytest.mark.parametrize(
    "method, status, expected",
    [
        ("get", 429, (True, "429")),
        ("GET", 503, (True, "503")),
        ("HEAD", 408, (True, "408")),
        ("GET", 404, (False, "status_not_retriable")),
        ("POST", 503, (False, "status_not_retriable")),
        ("GET", None, (False, "no_status")),
    ],
)
def test_status_classification(method: str, status: int | None, expected: tuple[bool, str]) -> None:
    assert classify_http_retry(method, status, None, _POLICY) == expected


def test_dns_failure_is_permanent_but_other_network_errors_retry() -> None:
    assert classify_http_retry("GET", None, socket.gaierror(8, "x"), _POLICY) == (False, "gaierror")
    assert classify_http_retry("GET", None, ConnectionError("reset"), _POLICY) == (True, "ConnectionError")


def test_unsafe_methods_retry_only_when_policy_allows() -> None:
    assert classify_http_retry("POST", None, ConnectionError("x"), _POLICY) == (False, "method_not_retriable")
    unsafe = SimpleNamespace(**{**vars(_POLICY), "retry_on_unsafe": True})
    assert classify_http_retry("POST", None, ConnectionError("x"), unsafe) == (True, "ConnectionError")
    assert classify_http_retry("POST", 503, None, unsafe) == (True, "503")


def test_http_client_binds_the_moved_classifiers() -> None:
    from tldw_Server_API.app.core import http_client

    assert http_client._should_retry is classify_http_retry
    assert http_client._is_dns_resolution_error is is_dns_resolution_error
