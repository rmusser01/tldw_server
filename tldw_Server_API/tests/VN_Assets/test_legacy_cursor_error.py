"""Public activity-reader pagination errors have a safe central domain type."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core import exceptions
from tldw_Server_API.app.core.VN_Assets.jobs import build_legacy_activity_reader

pytestmark = pytest.mark.unit


class ActivityPages:
    """Supply supported Jobs pages without relevant slot rows or DB side effects."""

    def __init__(self, *, stalled: bool, failure: Exception | None = None) -> None:
        """Select repeated-page, advancing-page or native read-error behavior."""
        self.stalled = stalled
        self.failure = failure
        self.calls = 0

    def list_jobs(self, **filters: Any) -> list[dict[str, Any]]:
        """Return full pages with distinct or repeated stable cursor endpoints."""
        self.calls += 1
        if self.failure is not None:
            raise self.failure
        if not self.stalled and filters.get("before_id") is not None:
            return []
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        return [{"id": index, "created_at": (created - timedelta(seconds=index)).isoformat(),
                 "payload": {}} for index in range(100, 0, -1)]


def test_stalled_activity_cursor_has_central_runtime_error() -> None:
    """Retain safe text and RuntimeError compatibility while identifying VN failure.

    Returns:
        None; asserts the central exception identity, safe code and bounded reader.
    """
    pages = ActivityPages(stalled=True)
    with pytest.raises(RuntimeError) as rejected:
        build_legacy_activity_reader(pages)(1, 2, 3, {4: "processing"}, set(), None)
    error_type = getattr(exceptions, "VNLegacyActivityCursorError", None)
    assert error_type is not None, "cursor failure must have a central VN error type"
    assert type(rejected.value) is error_type
    assert str(rejected.value) == "vn_asset_legacy_jobs_cursor_stalled"
    assert pages.calls == 2
    wrapped = exceptions.LegacyDisplayReconciliationError(rejected.value)
    assert wrapped.error_type is error_type
    assert wrapped.error_traceback is rejected.value.__traceback__
    assert str(wrapped) == "VN legacy display reconciliation failed"


def test_advancing_activity_cursor_preserves_no_activity_result() -> None:
    """Valid exhausted pagination remains a read-only no-activity result.

    Returns:
        None; asserts that ordinary full pages do not become domain failures.
    """
    assert build_legacy_activity_reader(ActivityPages(stalled=False))(
        1, 2, 3, {4: "processing"}, set(), None,
    ) == (False, False)


def test_activity_reader_keeps_native_failure_identity() -> None:
    """Do not wrap an unrelated Jobs read failure as a stalled cursor.

    Returns:
        None; asserts the original error object propagates with its native class.
    """
    original = OSError("PRIVATE_NATIVE_READ_ERROR")
    with pytest.raises(OSError) as rejected:
        build_legacy_activity_reader(ActivityPages(stalled=False, failure=original))(
            1, 2, 3, {4: "processing"}, set(), None,
        )
    assert rejected.value is original
