"""_strip_tzinfo stores naive UTC; an offset must be converted, not discarded (TASK-13324)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.datetime_utils import _strip_tzinfo

pytestmark = pytest.mark.unit


def test_aware_non_utc_is_converted_to_utc_before_stripping() -> None:
    pacific = timezone(timedelta(hours=-7))
    assert _strip_tzinfo(datetime(2026, 9, 21, 14, 6, 55, tzinfo=pacific)) == datetime(2026, 9, 21, 21, 6, 55)


def test_aware_utc_and_naive_are_unchanged() -> None:
    assert _strip_tzinfo(datetime(2026, 9, 21, 21, 6, 55, tzinfo=timezone.utc)) == datetime(2026, 9, 21, 21, 6, 55)
    assert _strip_tzinfo(datetime(2026, 9, 21, 21, 6, 55)) == datetime(2026, 9, 21, 21, 6, 55)
