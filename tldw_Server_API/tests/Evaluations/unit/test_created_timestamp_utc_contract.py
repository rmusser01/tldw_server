"""ADR-014: Evaluations `created` is a Unix timestamp on a public, OpenAI-compatible surface.

SQLite writes CURRENT_TIMESTAMP as naive UTC ("2026-09-21 21:06:55"). The converter did
`fromisoformat(s.replace("Z","+00:00"))` -- a no-op on that format, since it has no Z --
producing a NAIVE datetime whose .timestamp() is interpreted in the HOST's local zone.
On a host at UTC-7 that is 25200 seconds wrong.

CI runs UTC, where the error is exactly zero, so no existing test can observe it. This
test forces a non-UTC zone itself rather than depending on the CI timezone.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import pytest

from tldw_Server_API.app.core.DB_Management.Evaluations_DB import EvaluationsDatabase

# The converter does not touch `self`; call it unbound so the test needs no DB.
def _ensure_unix_timestamp(value, **kwargs):
    return EvaluationsDatabase._ensure_unix_timestamp(None, value, **kwargs)

# A fixed instant, written the way SQLite's CURRENT_TIMESTAMP writes it (naive UTC).
SQLITE_NAIVE_UTC = "2026-09-21 21:06:55"
EXPECTED_EPOCH = int(
    datetime(2026, 9, 21, 21, 6, 55, tzinfo=timezone.utc).timestamp()
)


@pytest.fixture
def non_utc_timezone():
    """Run the body under a fixed non-UTC zone, restoring the previous one after."""
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset() unavailable on this platform")
    previous = os.environ.get("TZ")
    os.environ["TZ"] = "America/Los_Angeles"
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def test_naive_sqlite_timestamp_is_read_as_utc(non_utc_timezone) -> None:
    got = _ensure_unix_timestamp(SQLITE_NAIVE_UTC)
    assert got == EXPECTED_EPOCH, (
        f"naive SQLite timestamp converted to {got}, expected {EXPECTED_EPOCH} "
        f"(off by {got - EXPECTED_EPOCH}s = the host UTC offset). ADR-014 makes this "
        "a public API contract."
    )


def test_conversion_is_timezone_independent(non_utc_timezone) -> None:
    """The same stored value must convert identically regardless of host zone."""
    under_non_utc = _ensure_unix_timestamp(SQLITE_NAIVE_UTC)

    previous = os.environ.get("TZ")
    os.environ["TZ"] = "UTC"
    time.tzset()
    try:
        under_utc = _ensure_unix_timestamp(SQLITE_NAIVE_UTC)
    finally:
        if previous is not None:
            os.environ["TZ"] = previous
        time.tzset()

    assert under_non_utc == under_utc, (
        f"same stored value converted to {under_non_utc} under America/Los_Angeles and "
        f"{under_utc} under UTC"
    )


def test_offset_aware_input_still_correct(non_utc_timezone) -> None:
    assert _ensure_unix_timestamp("2026-09-21T21:06:55+00:00") == EXPECTED_EPOCH
    assert _ensure_unix_timestamp("2026-09-21T21:06:55Z") == EXPECTED_EPOCH


def test_numeric_passthrough_unchanged(non_utc_timezone) -> None:
    assert _ensure_unix_timestamp("1790024815") == 1790024815
    assert _ensure_unix_timestamp(1790024815) == 1790024815
