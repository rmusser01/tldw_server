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

from tldw_Server_API.app.api.v1.endpoints.evaluations.evaluations_datasets import _normalize_dataset_payload
from tldw_Server_API.app.core.DB_Management.Evaluations_DB import to_unix_timestamp as _ensure_unix_timestamp

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


def test_postgres_naive_datetime_is_read_as_utc(non_utc_timezone) -> None:
    """PostgreSQL hands back datetimes, not strings; naive ones are UTC too."""
    assert _ensure_unix_timestamp(datetime(2026, 9, 21, 21, 6, 55)) == EXPECTED_EPOCH


def test_dataset_payload_converts_stored_created_at(non_utc_timezone) -> None:
    """The datasets endpoint and pipeline presets use the same converter, not now()."""
    for stored in (SQLITE_NAIVE_UTC, datetime(2026, 9, 21, 21, 6, 55)):
        assert _normalize_dataset_payload({"created_at": stored})["created"] == EXPECTED_EPOCH
