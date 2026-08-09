from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_analytics_exports as exports
from tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports import (
    CSV_COLUMNS,
    DEFAULT_EXPORT_MAX_BYTES,
    DEFAULT_EXPORT_ORPHAN_GRACE_SEC,
    DEFAULT_EXPORT_RETENTION_HOURS,
    EXPORT_ID_RE,
    EXPORT_SCAN_PAGE_SIZE,
    ClaimsAnalyticsExportError,
    create_queued_artifact,
    create_ready_artifact,
    export_max_bytes,
    export_retention_hours,
    normalize_export_request,
    orphan_grace_seconds,
    process_export_artifact,
    render_export,
    spreadsheet_safe,
    validate_export_id,
)

FIXED_NOW = datetime(2026, 8, 8, 12, 0, 0, 123456, tzinfo=timezone.utc)
FIXED_SNAPSHOT = "2026-08-08T12:00:00.123Z"


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _event(
    event_id: int,
    *,
    owner_user_id: str = "7",
    created_at: str = "2026-08-08T11:00:00.000Z",
    event_type: str = "claim_reviewed",
    severity: str = "info",
    payload: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "id": event_id,
        "user_id": owner_user_id,
        "event_type": event_type,
        "severity": severity,
        "payload_json": json.dumps(
            {} if payload is None else payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "created_at": created_at,
        "delivered_at": None,
    }
    row.update(extra)
    return row


class FakeMonitoringDB:
    def __init__(self, rows: list[dict[str, Any]], *, expected_owner: str = "7") -> None:
        self.rows = list(rows)
        self.expected_owner = expected_owner
        self.calls: list[dict[str, Any]] = []

    def list_claims_monitoring_events_page(
        self,
        *,
        user_id: str,
        event_type: str | None = None,
        severity: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        after_created_at: Any = None,
        after_id: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        assert user_id == self.expected_owner
        assert limit == EXPORT_SCAN_PAGE_SIZE
        assert (after_created_at is None) == (after_id is None)
        self.calls.append(
            {
                "user_id": user_id,
                "event_type": event_type,
                "severity": severity,
                "start_time": start_time,
                "end_time": end_time,
                "after_created_at": after_created_at,
                "after_id": after_id,
                "limit": limit,
            }
        )

        rows = [row for row in self.rows if str(row.get("user_id")) == user_id]
        if event_type is not None:
            rows = [row for row in rows if row.get("event_type") == event_type]
        if severity is not None:
            rows = [row for row in rows if row.get("severity") == severity]
        if start_time is not None:
            start = _parse_time(start_time)
            rows = [row for row in rows if _parse_time(str(row["created_at"])) >= start]
        if end_time is not None:
            end = _parse_time(end_time)
            rows = [row for row in rows if _parse_time(str(row["created_at"])) <= end]

        rows.sort(key=lambda row: (_parse_time(str(row["created_at"])), int(row["id"])))
        if after_created_at is not None and after_id is not None:
            cursor = (_parse_time(str(after_created_at)), int(after_id))
            rows = [row for row in rows if (_parse_time(str(row["created_at"])), int(row["id"])) > cursor]
        return [dict(row) for row in rows[:limit]]


class ScriptedPageDB:
    def __init__(
        self,
        pages: list[list[dict[str, Any]] | BaseException],
        *,
        expected_owner: str = "7",
    ) -> None:
        self.pages = pages
        self.expected_owner = expected_owner
        self.calls: list[dict[str, Any]] = []

    def list_claims_monitoring_events_page(
        self,
        *,
        user_id: str,
        event_type: str | None = None,
        severity: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        after_created_at: Any = None,
        after_id: int | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        assert user_id == self.expected_owner
        assert limit == EXPORT_SCAN_PAGE_SIZE
        assert (after_created_at is None) == (after_id is None)
        self.calls.append(
            {
                "user_id": user_id,
                "event_type": event_type,
                "severity": severity,
                "start_time": start_time,
                "end_time": end_time,
                "after_created_at": after_created_at,
                "after_id": after_id,
                "limit": limit,
            }
        )
        page_index = len(self.calls) - 1
        scripted = self.pages[page_index] if page_index < len(self.pages) else []
        if isinstance(scripted, BaseException):
            raise scripted
        return scripted


class ArtifactDB(FakeMonitoringDB):
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        super().__init__(rows or [], expected_owner="7")
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.mark_ready_hook: Any = None
        self.failure_transition_hook: Any = None
        self.force_wrong_owner_get = False

    def create_claims_analytics_export(self, **values: Any) -> dict[str, Any]:
        row = {
            "payload_json": None,
            "payload_csv": None,
            "filters_json": None,
            "pagination_json": None,
            "error_message": None,
            "job_id": None,
            "error_code": None,
            "snapshot_at": None,
            "created_at": FIXED_SNAPSHOT,
            "updated_at": FIXED_SNAPSHOT,
            **values,
        }
        row["user_id"] = str(row["user_id"])
        self.artifacts[row["export_id"]] = row
        return dict(row)

    def get_claims_analytics_export(self, export_id: str, *, user_id: str) -> dict[str, Any]:
        row = self.artifacts.get(export_id)
        if row is None:
            return {}
        if not self.force_wrong_owner_get and row["user_id"] != user_id:
            return {}
        return dict(row)

    def attach_claims_analytics_export_job(self, *, export_id: str, user_id: str, job_id: int) -> bool:
        row = self.artifacts.get(export_id)
        if row is None or row["user_id"] != user_id:
            return False
        if row["job_id"] not in (None, job_id):
            return False
        row["job_id"] = job_id
        return True

    def transition_claims_analytics_export_status(
        self,
        *,
        export_id: str,
        user_id: str,
        from_statuses: tuple[str, ...],
        to_status: str,
        error_code: str | None = None,
        error_message: str | None = None,
    ) -> bool:
        row = self.artifacts.get(export_id)
        if row is None or row["user_id"] != user_id:
            return False
        if to_status == "failed" and self.failure_transition_hook is not None:
            self.failure_transition_hook(row)
        if row["status"] not in from_statuses:
            return False
        row["status"] = to_status
        row["error_code"] = error_code
        row["error_message"] = error_message
        return True

    def mark_claims_analytics_export_ready(
        self,
        *,
        export_id: str,
        user_id: str,
        payload_json: str | None,
        payload_csv: str | None,
    ) -> bool:
        row = self.artifacts[export_id]
        if self.mark_ready_hook is not None:
            self.mark_ready_hook(row)
        if row["user_id"] != user_id or row["status"] != "processing":
            return False
        row.update(
            status="ready",
            payload_json=payload_json,
            payload_csv=payload_csv,
            error_code=None,
            error_message=None,
        )
        return True


def _normalized(
    *,
    format: str = "json",
    filters: dict[str, Any] | None = None,
    pagination: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"format": format}
    if filters is not None:
        payload["filters"] = filters
    if pagination is not None:
        payload["pagination"] = pagination
    return normalize_export_request(payload, owner_user_id="7", now=FIXED_NOW)


def _render(
    db: FakeMonitoringDB,
    normalized: dict[str, Any],
    *,
    max_bytes: int = DEFAULT_EXPORT_MAX_BYTES,
) -> dict[str, Any]:
    return render_export(
        db,
        owner_user_id=normalized["owner_user_id"],
        format=normalized["format"],
        filters=normalized["filters"],
        pagination=normalized["pagination"],
        snapshot_at=normalized["snapshot_at"],
        max_bytes=max_bytes,
    )


def test_public_constants_and_error_attributes_are_stable() -> None:
    assert DEFAULT_EXPORT_MAX_BYTES == 10_485_760
    assert DEFAULT_EXPORT_ORPHAN_GRACE_SEC == 300
    assert EXPORT_SCAN_PAGE_SIZE == 1000
    assert EXPORT_ID_RE.pattern == r"^[0-9a-f]{32}$"
    assert CSV_COLUMNS == ("id", "event_type", "severity", "created_at", "payload_json")

    error = ClaimsAnalyticsExportError(
        "safe message",
        code="stable_code",
        retryable=True,
        http_status=503,
    )
    assert str(error) == "safe message"
    assert error.public_message == "safe message"
    assert error.code == "stable_code"
    assert error.retryable is True
    assert error.http_status == 503


def test_normalize_export_request_matches_plan_example() -> None:
    normalized = normalize_export_request(
        {
            "format": "json",
            "filters": {
                "workspace_id": "7",
                "start_time": "2026-08-01T01:00:00-07:00",
                "end_time": "2026-08-10T00:00:00Z",
            },
            "pagination": {"limit": 20_000, "offset": -4},
        },
        owner_user_id="7",
        now=datetime(2026, 8, 8, 12, tzinfo=timezone.utc),
    )

    assert normalized == {
        "owner_user_id": "7",
        "format": "json",
        "filters": {
            "start_time": "2026-08-01T08:00:00.000Z",
            "end_time": "2026-08-08T12:00:00.000Z",
        },
        "pagination": {"limit": 10_000, "offset": 0},
        "snapshot_at": "2026-08-08T12:00:00.000Z",
    }


def test_normalize_defaults_format_filters_pagination_and_snapshot_end() -> None:
    normalized = normalize_export_request({}, owner_user_id="7", now=FIXED_NOW)

    assert normalized == {
        "owner_user_id": "7",
        "format": "json",
        "filters": {"end_time": FIXED_SNAPSHOT},
        "pagination": {"limit": 1000, "offset": 0},
        "snapshot_at": FIXED_SNAPSHOT,
    }


def test_normalize_lowercases_csv_and_keeps_only_known_filters() -> None:
    normalized = normalize_export_request(
        {
            "format": "CSV",
            "filters": {
                "event_type": "claim_reviewed",
                "severity": "warning",
                "provider": "local",
                "model": "model-a",
                "workspace_id": "999",
                "unknown_filter": "drop-me",
            },
        },
        owner_user_id="7",
        now=FIXED_NOW,
    )

    assert normalized["format"] == "csv"
    assert normalized["filters"] == {
        "event_type": "claim_reviewed",
        "severity": "warning",
        "provider": "local",
        "model": "model-a",
        "end_time": FIXED_SNAPSHOT,
    }


@pytest.mark.parametrize("payload", [[], "json", 1, True, None])
def test_normalize_rejects_non_mapping_payload(payload: Any) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        normalize_export_request(payload, owner_user_id="7", now=FIXED_NOW)

    assert exc_info.value.code == "claims_export_invalid_payload"
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "payload",
    [
        {"filters": None},
        {"filters": []},
        {"filters": "event_type=claim_reviewed"},
        {"pagination": None},
        {"pagination": []},
        {"pagination": "limit=10"},
    ],
)
def test_normalize_rejects_nonsensical_filter_and_pagination_containers(
    payload: dict[str, Any],
) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        normalize_export_request(payload, owner_user_id="7", now=FIXED_NOW)

    assert exc_info.value.code == "claims_export_invalid_payload"
    assert exc_info.value.retryable is False
    assert exc_info.value.http_status == 400


@pytest.mark.parametrize(
    "invalid_format",
    ["", None, False, True, 0, 1, [], {}, "xlsx"],
)
def test_normalize_rejects_explicit_invalid_format_with_stable_code(
    invalid_format: Any,
) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        normalize_export_request({"format": invalid_format}, owner_user_id="7", now=FIXED_NOW)

    assert exc_info.value.code == "claims_export_unsupported_format"
    assert exc_info.value.retryable is False
    assert exc_info.value.http_status == 400


@pytest.mark.parametrize(
    "owner_user_id",
    [1, True, "01", "0", "-1", "+1", " 1", "1 ", "", None],
)
def test_normalize_rejects_noncanonical_owner_ids(owner_user_id: Any) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        normalize_export_request({}, owner_user_id=owner_user_id, now=FIXED_NOW)

    assert exc_info.value.code == "claims_owner_scope_violation"
    assert exc_info.value.retryable is False


def test_normalize_accepts_naive_and_offset_timestamps_as_utc_milliseconds() -> None:
    naive = _normalized(filters={"start_time": "2026-08-01T01:02:03.456789"})
    offset = _normalized(filters={"start_time": "2026-08-01T01:02:03.456789-07:00"})

    assert naive["filters"]["start_time"] == "2026-08-01T01:02:03.456Z"
    assert offset["filters"]["start_time"] == "2026-08-01T08:02:03.456Z"


def test_normalize_caps_future_end_at_snapshot() -> None:
    normalized = _normalized(filters={"end_time": "2099-01-01T00:00:00+00:00"})

    assert normalized["filters"]["end_time"] == FIXED_SNAPSHOT
    assert normalized["snapshot_at"] == FIXED_SNAPSHOT


@pytest.mark.parametrize(
    "filters",
    [
        {"start_time": "not-a-time"},
        {"end_time": "2026-99-99T00:00:00Z"},
        {"start_time": "2026-08-09T00:00:00Z"},
        {
            "start_time": "2026-08-08T10:00:00Z",
            "end_time": "2026-08-08T09:59:59Z",
        },
    ],
)
def test_normalize_rejects_malformed_times_and_invalid_ranges(
    filters: dict[str, Any],
) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _normalized(filters=filters)

    assert exc_info.value.code == "claims_export_invalid_payload"
    assert exc_info.value.retryable is False
    assert exc_info.value.http_status == 400


def test_normalize_coerces_pagination_values_consistently() -> None:
    numeric_strings = _normalized(pagination={"limit": "42", "offset": "3"})
    numeric_values = _normalized(pagination={"limit": 42.9, "offset": -3.8})
    invalid_values = _normalized(pagination={"limit": "many", "offset": {"bad": True}})

    assert numeric_strings["pagination"] == {"limit": 42, "offset": 3}
    assert numeric_values["pagination"] == {"limit": 42, "offset": 0}
    assert invalid_values["pagination"] == {"limit": 1000, "offset": 0}


def test_validate_export_id_requires_exact_lowercase_uuid_hex() -> None:
    valid = "0123456789abcdef0123456789abcdef"

    assert validate_export_id(valid) == valid
    for value in (
        valid.upper(),
        valid[:-1],
        valid + "0",
        valid + "\n",
        "g" * 32,
        123,
        True,
        None,
    ):
        with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
            validate_export_id(value)
        assert exc_info.value.code == "claims_export_invalid_payload"
        assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    ("settings_obj", "expected"),
    [
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": "2048"}, 2048),
        (SimpleNamespace(CLAIMS_ANALYTICS_EXPORT_MAX_BYTES=4096), 4096),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": 0}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": "0"}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": -1}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": "-1"}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": "bad"}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": True}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": False}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": 1.5}, DEFAULT_EXPORT_MAX_BYTES),
        ({"CLAIMS_ANALYTICS_EXPORT_MAX_BYTES": "1.5"}, DEFAULT_EXPORT_MAX_BYTES),
        ({}, DEFAULT_EXPORT_MAX_BYTES),
    ],
)
def test_export_max_bytes_reads_mapping_or_attributes_with_safe_fallback(
    settings_obj: Any,
    expected: int,
) -> None:
    assert export_max_bytes(settings_obj) == expected


@pytest.mark.parametrize(
    ("key", "resolver", "env_value", "expected"),
    [
        ("CLAIMS_ANALYTICS_EXPORT_MAX_BYTES", export_max_bytes, "2048", 2048),
        ("CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", orphan_grace_seconds, "30", 30),
        ("CLAIMS_ANALYTICS_EXPORT_MAX_BYTES", export_max_bytes, "", DEFAULT_EXPORT_MAX_BYTES),
        ("CLAIMS_ANALYTICS_EXPORT_MAX_BYTES", export_max_bytes, "true", DEFAULT_EXPORT_MAX_BYTES),
        ("CLAIMS_ANALYTICS_EXPORT_MAX_BYTES", export_max_bytes, "0", DEFAULT_EXPORT_MAX_BYTES),
        (
            "CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC",
            orphan_grace_seconds,
            "-1",
            DEFAULT_EXPORT_ORPHAN_GRACE_SEC,
        ),
    ],
)
def test_export_settings_read_environment_before_cached_settings(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    resolver: Any,
    env_value: str,
    expected: int,
) -> None:
    monkeypatch.setitem(exports.settings, key, "999")
    monkeypatch.setenv(key, env_value)

    assert resolver() == expected


@pytest.mark.parametrize(
    ("key", "resolver", "env_value", "settings_value"),
    [
        ("CLAIMS_ANALYTICS_EXPORT_MAX_BYTES", export_max_bytes, "2048", 4096),
        ("CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC", orphan_grace_seconds, "30", 45),
        ("CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS", export_retention_hours, "2", 48),
    ],
)
def test_explicit_export_settings_override_process_environment(
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    resolver: Any,
    env_value: str,
    settings_value: int,
) -> None:
    monkeypatch.setenv(key, env_value)

    assert resolver({key: settings_value}) == settings_value


@pytest.mark.parametrize(
    ("settings_value", "expected"),
    [
        (1, 1.0),
        ("1", 1.0),
        (0.5, 0.5),
        ("0.5", 0.5),
        (True, DEFAULT_EXPORT_RETENTION_HOURS),
        (False, DEFAULT_EXPORT_RETENTION_HOURS),
        ("", DEFAULT_EXPORT_RETENTION_HOURS),
        ("not-a-number", DEFAULT_EXPORT_RETENTION_HOURS),
        (float("inf"), DEFAULT_EXPORT_RETENTION_HOURS),
        (float("-inf"), DEFAULT_EXPORT_RETENTION_HOURS),
        (float("nan"), DEFAULT_EXPORT_RETENTION_HOURS),
        (0, DEFAULT_EXPORT_RETENTION_HOURS),
        ("0", DEFAULT_EXPORT_RETENTION_HOURS),
        (-0.5, DEFAULT_EXPORT_RETENTION_HOURS),
        ("-0.5", DEFAULT_EXPORT_RETENTION_HOURS),
    ],
)
def test_export_retention_hours_accepts_only_finite_positive_numbers(
    settings_value: Any,
    expected: float,
) -> None:
    assert export_retention_hours({"CLAIMS_ANALYTICS_EXPORT_RETENTION_HOURS": settings_value}) == expected


@pytest.mark.parametrize(
    ("settings_obj", "expected"),
    [
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": "30"}, 30),
        (SimpleNamespace(CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC=45), 45),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": 0}, 0),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": "0"}, 0),
        (SimpleNamespace(CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC=0), 0),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": -1}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": "-1"}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": object()}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": True}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": False}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": 1.5}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({"CLAIMS_ANALYTICS_EXPORT_ORPHAN_GRACE_SEC": "1.5"}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
        ({}, DEFAULT_EXPORT_ORPHAN_GRACE_SEC),
    ],
)
def test_orphan_grace_seconds_reads_mapping_or_attributes_with_safe_fallback(
    settings_obj: Any,
    expected: int,
) -> None:
    assert orphan_grace_seconds(settings_obj) == expected


def test_render_scans_tied_timestamps_in_keyset_pages_without_gaps_or_duplicates() -> None:
    rows = [_event(event_id) for event_id in range(1, 1004)]
    db = FakeMonitoringDB(list(reversed(rows)))
    normalized = _normalized(pagination={"limit": 5, "offset": 998})

    result = _render(db, normalized)
    payload = json.loads(result["payload_json"])

    assert [event["id"] for event in payload["events"]] == [999, 1000, 1001, 1002, 1003]
    assert len({event["id"] for event in payload["events"]}) == 5
    assert payload["pagination"] == {"limit": 5, "offset": 998, "total": 1003}
    assert result["event_count"] == 5
    assert len(db.calls) == 2
    assert db.calls[0]["after_created_at"] is None
    assert db.calls[0]["after_id"] is None
    assert db.calls[1]["after_created_at"] == rows[999]["created_at"]
    assert db.calls[1]["after_id"] == 1000


def test_render_rejects_repeated_full_page_with_non_advancing_cursor() -> None:
    page = [_event(event_id) for event_id in range(1, EXPORT_SCAN_PAGE_SIZE + 1)]
    db = ScriptedPageDB([page, page])

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _render(db, _normalized())

    assert exc_info.value.code == "claims_export_serialization_failed"
    assert exc_info.value.retryable is False
    assert len(db.calls) == 2


@pytest.mark.parametrize(
    "rows",
    [
        [_event(1), _event(1)],
        [_event(2), _event(1)],
        [
            _event(1, created_at="2026-08-08T11:00:01.000Z"),
            _event(2, created_at="2026-08-08T11:00:00.000Z"),
        ],
    ],
    ids=["duplicate-key", "descending-id", "descending-time"],
)
def test_render_rejects_duplicate_or_out_of_order_rows(rows: list[dict[str, Any]]) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _render(ScriptedPageDB([rows]), _normalized())

    assert exc_info.value.code == "claims_export_serialization_failed"
    assert exc_info.value.retryable is False


def test_render_rejects_page_larger_than_scan_limit() -> None:
    page = [_event(event_id) for event_id in range(1, EXPORT_SCAN_PAGE_SIZE + 2)]

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _render(ScriptedPageDB([page]), _normalized())

    assert exc_info.value.code == "claims_export_serialization_failed"
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "row",
    [
        {key: value for key, value in _event(1).items() if key != "id"},
        _event(0),
        _event(True),
        _event("1"),
        {key: value for key, value in _event(1).items() if key != "event_type"},
        _event(1, event_type=""),
        {key: value for key, value in _event(1).items() if key != "user_id"},
        _event(1, owner_user_id="8"),
        {key: value for key, value in _event(1).items() if key != "created_at"},
        _event(1, created_at="not-a-timestamp"),
    ],
    ids=[
        "missing-id",
        "zero-id",
        "bool-id",
        "string-id",
        "missing-event-type",
        "empty-event-type",
        "missing-owner",
        "wrong-owner",
        "missing-created-at",
        "malformed-created-at",
    ],
)
def test_render_rejects_rows_without_deterministic_identity_and_time(
    row: dict[str, Any],
) -> None:
    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _render(ScriptedPageDB([[row]]), _normalized())

    assert exc_info.value.code == "claims_export_serialization_failed"
    assert exc_info.value.retryable is False


def test_render_canonicalizes_native_datetimes_and_uses_native_cursor() -> None:
    native_created_at = datetime.fromisoformat("2026-08-08T04:05:06.789123-07:00")
    native_delivered_at = datetime.fromisoformat("2026-08-08T04:06:07.987654-07:00")
    page = [
        _event(
            event_id,
            created_at=native_created_at,
            delivered_at=native_delivered_at,
        )
        for event_id in range(1, EXPORT_SCAN_PAGE_SIZE + 1)
    ]
    db = ScriptedPageDB([page, []])
    result = _render(db, _normalized(pagination={"limit": 1, "offset": 0}))
    event = json.loads(result["payload_json"])["events"][0]

    assert event["created_at"] == "2026-08-08T11:05:06.789Z"
    assert event["delivered_at"] == "2026-08-08T11:06:07.987Z"
    assert db.calls[1]["after_created_at"] is native_created_at
    assert db.calls[1]["after_id"] == EXPORT_SCAN_PAGE_SIZE


def test_render_canonicalizes_sqlite_timestamp_strings_for_json_and_csv() -> None:
    row = _event(
        1,
        created_at="2026-08-08 04:05:06.789123-07:00",
        delivered_at="2026-08-08 04:06:07.987654-07:00",
    )
    json_result = _render(ScriptedPageDB([[row]]), _normalized())
    json_event = json.loads(json_result["payload_json"])["events"][0]
    csv_result = _render(ScriptedPageDB([[row]]), _normalized(format="csv"))
    csv_rows = list(csv.reader(io.StringIO(csv_result["payload_csv"], newline="")))

    assert json_event["created_at"] == "2026-08-08T11:05:06.789Z"
    assert json_event["delivered_at"] == "2026-08-08T11:06:07.987Z"
    assert csv_rows[1][3] == "2026-08-08T11:05:06.789Z"


def test_render_preserves_none_delivered_at() -> None:
    result = _render(ScriptedPageDB([[_event(1, delivered_at=None)]]), _normalized())

    assert json.loads(result["payload_json"])["events"][0]["delivered_at"] is None


def test_render_keeps_malformed_payload_json_tolerant_as_empty_object() -> None:
    row = _event(1)
    row["payload_json"] = "{malformed"

    result = _render(ScriptedPageDB([[row]]), _normalized())

    assert json.loads(result["payload_json"])["events"][0]["payload"] == {}


def test_render_does_not_wrap_raw_database_exceptions() -> None:
    database_error = RuntimeError("database temporarily unavailable")

    with pytest.raises(RuntimeError) as exc_info:
        _render(ScriptedPageDB([database_error]), _normalized())

    assert exc_info.value is database_error


def test_render_applies_provider_and_model_after_payload_decode_and_counts_all_matches() -> None:
    rows = [
        _event(
            event_id,
            payload={
                "provider": "local" if event_id % 2 == 0 else "remote",
                "model": "model-a" if event_id % 4 == 0 else "model-b",
            },
        )
        for event_id in range(1, 1205)
    ]
    db = FakeMonitoringDB(rows)
    normalized = _normalized(
        filters={"provider": "local", "model": "model-a"},
        pagination={"limit": 3, "offset": 2},
    )

    result = _render(db, normalized)
    payload = json.loads(result["payload_json"])

    assert [event["id"] for event in payload["events"]] == [12, 16, 20]
    assert payload["pagination"] == {"limit": 3, "offset": 2, "total": 301}
    assert all("payload_json" not in event for event in payload["events"])
    assert all(event["payload"] == {"provider": "local", "model": "model-a"} for event in payload["events"])
    assert len(db.calls) == 2
    assert all(call["event_type"] is None and call["severity"] is None for call in db.calls)


def test_render_passes_database_filters_and_snapshot_cutoff_on_every_page() -> None:
    rows = [
        _event(
            event_id,
            event_type="claim_reviewed" if event_id <= 1000 else "other",
            severity="warning",
        )
        for event_id in range(1, 1002)
    ]
    db = FakeMonitoringDB(rows)
    normalized = _normalized(
        filters={
            "event_type": "claim_reviewed",
            "severity": "warning",
            "start_time": "2026-08-01T00:00:00Z",
            "end_time": "2099-01-01T00:00:00Z",
        },
        pagination={"limit": 1, "offset": 0},
    )

    result = _render(db, normalized)

    assert json.loads(result["payload_json"])["pagination"]["total"] == 1000
    assert len(db.calls) == 2
    for call in db.calls:
        assert call["event_type"] == "claim_reviewed"
        assert call["severity"] == "warning"
        assert call["start_time"] == "2026-08-01T00:00:00.000Z"
        assert call["end_time"] == FIXED_SNAPSHOT


def test_render_fixed_snapshot_ignores_rows_inserted_between_equivalent_calls() -> None:
    db = FakeMonitoringDB([_event(1)])
    normalized = _normalized()

    first = _render(db, normalized)
    db.rows.append(_event(2, created_at="2026-08-08T12:00:00.124Z"))
    second = _render(db, normalized)

    assert first == second
    assert [event["id"] for event in json.loads(second["payload_json"])["events"]] == [1]


def test_render_is_deterministic_for_sync_and_worker_equivalent_calls() -> None:
    rows = [
        _event(2, payload={"text": "café", "provider": "local"}),
        _event(1, payload={"text": "東京", "provider": "local"}),
    ]
    normalized = _normalized(filters={"provider": "local"})

    sync_result = _render(FakeMonitoringDB(rows), normalized)
    worker_result = _render(FakeMonitoringDB(list(reversed(rows))), dict(normalized))

    assert sync_result == worker_result


def test_render_json_is_compact_unicode_preserving_and_drops_payload_json() -> None:
    db = FakeMonitoringDB([_event(1, payload={"text": "café / 東京"})])
    normalized = _normalized(filters={"provider": "local"})
    db.rows[0]["payload_json"] = '{"provider":"local","text":"café / 東京"}'

    result = _render(db, normalized)
    payload_text = result["payload_json"]
    payload = json.loads(payload_text)

    assert "café / 東京" in payload_text
    assert "\\u00e9" not in payload_text
    assert ": " not in payload_text
    assert ", " not in payload_text
    assert payload["events"][0]["payload"] == {"provider": "local", "text": "café / 東京"}
    assert "payload_json" not in payload["events"][0]
    assert result["size_bytes"] == len(payload_text.encode("utf-8"))


def test_spreadsheet_safe_prefixes_each_dangerous_string() -> None:
    assert spreadsheet_safe("=SUM(A1:A2)") == "'=SUM(A1:A2)"
    assert spreadsheet_safe("+1") == "'+1"
    assert spreadsheet_safe("-1") == "'-1"
    assert spreadsheet_safe("@cmd") == "'@cmd"
    assert spreadsheet_safe("\tcmd") == "'\tcmd"
    assert spreadsheet_safe("\rcmd") == "'\rcmd"
    assert spreadsheet_safe("safe") == "safe"
    assert spreadsheet_safe(7) == 7
    assert spreadsheet_safe(None) is None


def test_render_csv_preserves_unicode_delimiters_quotes_newlines_and_formula_safety() -> None:
    db = FakeMonitoringDB(
        [
            _event(
                1,
                event_type="=SUM(A1:A2)",
                severity='+critical,"quoted"\nnext',
                payload={"formula": "-1", "place": "Montréal, 東京", "text": "line 1\nline 2"},
            )
        ]
    )
    normalized = _normalized(format="csv")

    result = _render(db, normalized)
    payload_csv = result["payload_csv"]
    parsed_rows = list(csv.reader(io.StringIO(payload_csv, newline="")))

    assert payload_csv.endswith("\r\n")
    assert parsed_rows[0] == list(CSV_COLUMNS)
    assert parsed_rows[1] == [
        "1",
        "'=SUM(A1:A2)",
        '\'+critical,"quoted"\nnext',
        "2026-08-08T11:00:00.000Z",
        '{"formula":"-1","place":"Montréal, 東京","text":"line 1\\nline 2"}',
    ]
    assert result["payload_json"] is None
    assert result["event_count"] == 1
    assert result["size_bytes"] == len(payload_csv.encode("utf-8"))


def test_render_enforces_exact_utf8_byte_boundary_and_one_byte_over() -> None:
    db = FakeMonitoringDB([_event(1, payload={"text": "東京"})])
    normalized = _normalized()
    baseline = _render(db, normalized)
    exact_size = baseline["size_bytes"]

    exact = _render(FakeMonitoringDB(db.rows), normalized, max_bytes=exact_size)
    assert exact["size_bytes"] == exact_size

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _render(FakeMonitoringDB(db.rows), normalized, max_bytes=exact_size - 1)

    assert exact_size == (exact_size - 1) + 1
    assert exc_info.value.code == "claims_export_too_large"
    assert exc_info.value.retryable is False
    assert exc_info.value.http_status == 413


def test_render_wraps_serialization_failures_without_raw_public_text() -> None:
    secret = object()
    db = FakeMonitoringDB([_event(1, severity=secret)])  # type: ignore[arg-type]
    normalized = _normalized()

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        _render(db, normalized)

    assert exc_info.value.code == "claims_export_serialization_failed"
    assert exc_info.value.retryable is False
    assert exc_info.value.http_status == 400
    assert "object" not in exc_info.value.public_message.lower()
    assert hex(id(secret)) not in exc_info.value.public_message


@pytest.mark.parametrize(
    "overrides",
    [
        {"format": "xlsx"},
        {"filters": []},
        {"pagination": []},
        {"snapshot_at": "not-a-time"},
        {"max_bytes": 0},
        {"owner_user_id": "07"},
    ],
)
def test_render_revalidates_persisted_inputs(overrides: dict[str, Any]) -> None:
    db = FakeMonitoringDB([])
    normalized = _normalized()
    arguments = {
        "owner_user_id": normalized["owner_user_id"],
        "format": normalized["format"],
        "filters": normalized["filters"],
        "pagination": normalized["pagination"],
        "snapshot_at": normalized["snapshot_at"],
        "max_bytes": DEFAULT_EXPORT_MAX_BYTES,
    }
    arguments.update(overrides)

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        render_export(db, **arguments)

    expected_code = (
        "claims_export_unsupported_format"
        if overrides.get("format") == "xlsx"
        else "claims_owner_scope_violation"
        if overrides.get("owner_user_id") == "07"
        else "claims_export_invalid_payload"
    )
    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "invalid_format",
    ["", None, False, True, 0, 1, [], {}, "xlsx"],
)
def test_render_rejects_explicit_invalid_format(invalid_format: Any) -> None:
    db = FakeMonitoringDB([])
    normalized = _normalized()

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        render_export(
            db,
            owner_user_id=normalized["owner_user_id"],
            format=invalid_format,
            filters=normalized["filters"],
            pagination=normalized["pagination"],
            snapshot_at=normalized["snapshot_at"],
            max_bytes=DEFAULT_EXPORT_MAX_BYTES,
        )

    assert exc_info.value.code == "claims_export_unsupported_format"
    assert exc_info.value.retryable is False


def test_render_result_has_only_compact_artifact_fields() -> None:
    db = FakeMonitoringDB([_event(1, db_path="/private/owner-7.db")])
    normalized = _normalized(filters={"provider": "local", "workspace_id": "/private/owner-7.db"})
    db.rows[0]["payload_json"] = '{"provider":"local"}'

    result = _render(db, normalized)

    assert set(result) == {"payload_json", "payload_csv", "format", "event_count", "size_bytes"}
    assert result["payload_csv"] is None
    assert "filters" not in result
    assert "pagination" not in result
    assert "owner_user_id" not in result
    assert "db_path" not in result
    assert "/private/owner-7.db" not in json.dumps(result, sort_keys=True)


def test_create_queued_artifact_persists_only_compact_normalized_request() -> None:
    db = ArtifactDB()
    normalized = _normalized(
        filters={"severity": "warning"},
        pagination={"limit": 25, "offset": 2},
    )

    row = create_queued_artifact(db, owner_user_id="7", normalized=normalized)

    assert EXPORT_ID_RE.fullmatch(row["export_id"])
    assert row["user_id"] == "7"
    assert row["format"] == "json"
    assert row["status"] == "queued"
    assert row["job_id"] is None
    assert row["payload_json"] is None
    assert row["payload_csv"] is None
    assert row["snapshot_at"] == FIXED_SNAPSHOT
    assert row["filters_json"] == json.dumps(normalized["filters"], ensure_ascii=False, separators=(",", ":"))
    assert row["pagination_json"] == '{"limit":25,"offset":2}'
    assert " " not in row["filters_json"]


@pytest.mark.parametrize(
    ("owner_user_id", "normalized_owner"),
    [("07", "7"), ("7", "8"), ("7", 7)],
)
def test_artifact_creation_rejects_invalid_or_mismatched_normalized_owner(
    owner_user_id: Any,
    normalized_owner: Any,
) -> None:
    normalized = _normalized()
    normalized["owner_user_id"] = normalized_owner

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        create_queued_artifact(
            ArtifactDB(),
            owner_user_id=owner_user_id,
            normalized=normalized,
        )

    assert exc_info.value.code == "claims_owner_scope_violation"


def test_synchronous_and_worker_artifacts_use_identical_renderer_content() -> None:
    normalized = _normalized(filters={"severity": "warning"})
    events = [_event(1, severity="warning", payload={"model": "model-a"})]
    sync_db = ArtifactDB(events)
    worker_db = ArtifactDB(events)

    sync_row = create_ready_artifact(
        sync_db,
        owner_user_id="7",
        normalized=normalized,
    )
    queued = create_queued_artifact(
        worker_db,
        owner_user_id="7",
        normalized=normalized,
    )
    worker_result = process_export_artifact(
        worker_db,
        owner_user_id="7",
        export_id=queued["export_id"],
        job_id=42,
    )
    worker_row = worker_db.get_claims_analytics_export(queued["export_id"], user_id="7")

    assert sync_row["status"] == "ready"
    assert sync_row["payload_json"] == worker_row["payload_json"]
    assert worker_result == {
        "outcome": "ok",
        "export_id": queued["export_id"],
        "format": "json",
        "event_count": 1,
        "size_bytes": len(worker_row["payload_json"].encode("utf-8")),
    }
    assert "content" not in worker_result
    assert "filters" not in worker_result
    assert "path" not in worker_result


@pytest.mark.parametrize("starting_status", ["queued", "failed", "processing"])
def test_process_repairs_missing_job_and_resumes_retryable_states(
    starting_status: str,
) -> None:
    db = ArtifactDB([_event(1)])
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())
    db.artifacts[row["export_id"]]["status"] = starting_status

    result = process_export_artifact(
        db,
        owner_user_id="7",
        export_id=row["export_id"],
        job_id=42,
    )

    stored = db.get_claims_analytics_export(row["export_id"], user_id="7")
    assert result["outcome"] == "ok"
    assert stored["job_id"] == 42
    assert stored["status"] == "ready"


def test_process_rejects_conflicting_job_without_mutating_artifact() -> None:
    db = ArtifactDB()
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())
    db.artifacts[row["export_id"]]["job_id"] = 41

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        process_export_artifact(
            db,
            owner_user_id="7",
            export_id=row["export_id"],
            job_id=42,
        )

    assert exc_info.value.code == "claims_export_invalid_artifact"
    assert db.artifacts[row["export_id"]]["status"] == "queued"
    assert db.artifacts[row["export_id"]]["job_id"] == 41


def test_process_ready_artifact_returns_exact_non_mutating_skip() -> None:
    db = ArtifactDB([_event(1)])
    row = create_ready_artifact(db, owner_user_id="7", normalized=_normalized())
    before = dict(row)

    result = process_export_artifact(
        db,
        owner_user_id="7",
        export_id=row["export_id"],
        job_id=42,
    )

    assert result == {
        "outcome": "skipped",
        "reason": "already_ready",
        "export_id": row["export_id"],
    }
    assert db.artifacts[row["export_id"]] == before


def test_process_late_ready_race_returns_skip_without_overwrite() -> None:
    db = ArtifactDB([_event(1)])
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())

    def winner(artifact: dict[str, Any]) -> None:
        artifact["status"] = "ready"
        artifact["payload_json"] = '{"winner":true}'

    db.mark_ready_hook = winner
    result = process_export_artifact(
        db,
        owner_user_id="7",
        export_id=row["export_id"],
        job_id=42,
    )

    assert result == {
        "outcome": "skipped",
        "reason": "already_ready",
        "export_id": row["export_id"],
    }
    assert db.artifacts[row["export_id"]]["payload_json"] == '{"winner":true}'


def test_process_late_failure_race_raises_stable_retryable_failure() -> None:
    db = ArtifactDB([_event(1)])
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())
    db.mark_ready_hook = lambda artifact: artifact.update(status="failed")

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        process_export_artifact(
            db,
            owner_user_id="7",
            export_id=row["export_id"],
            job_id=42,
        )

    assert exc_info.value.code == "claims_export_storage_unavailable"
    assert exc_info.value.retryable is True
    assert "failed" not in exc_info.value.public_message.lower()


def test_process_transition_race_to_failed_is_retryable_not_malformed() -> None:
    db = ArtifactDB([_event(1)])
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())
    transition = db.transition_claims_analytics_export_status

    def losing_transition(**values: Any) -> bool:
        changed = transition(**values)
        if values["to_status"] == "processing":
            db.artifacts[row["export_id"]]["status"] = "failed"
        return changed

    db.transition_claims_analytics_export_status = losing_transition  # type: ignore[method-assign]

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        process_export_artifact(
            db,
            owner_user_id="7",
            export_id=row["export_id"],
            job_id=42,
        )

    assert exc_info.value.code == "claims_export_storage_unavailable"
    assert exc_info.value.retryable is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("filters_json", "[]"),
        ("filters_json", "not-json"),
        ("pagination_json", "[]"),
        ("format", "JSON"),
        ("snapshot_at", "2026-08-08T12:00:00.123456Z"),
    ],
)
def test_process_rejects_malformed_or_noncanonical_persisted_request(
    field: str,
    value: Any,
) -> None:
    db = ArtifactDB()
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())
    db.artifacts[row["export_id"]][field] = value

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        process_export_artifact(
            db,
            owner_user_id="7",
            export_id=row["export_id"],
            job_id=42,
        )

    stored = db.artifacts[row["export_id"]]
    assert exc_info.value.code == "claims_export_invalid_artifact"
    assert stored["status"] == "failed"
    assert stored["error_code"] == "claims_export_invalid_artifact"
    assert "not-json" not in (stored["error_message"] or "")


def test_process_missing_or_wrong_owner_artifact_uses_same_safe_code() -> None:
    db = ArtifactDB()
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())
    db.artifacts[row["export_id"]]["user_id"] = "8"
    db.force_wrong_owner_get = True

    for export_id in ("0" * 32, row["export_id"]):
        with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
            process_export_artifact(
                db,
                owner_user_id="7",
                export_id=export_id,
                job_id=42,
            )
        assert exc_info.value.code == "claims_export_missing"
        assert exc_info.value.http_status == 404


@pytest.mark.parametrize(
    ("failure_kind", "expected_code"),
    [
        ("too_large", "claims_export_too_large"),
        ("serialization", "claims_export_serialization_failed"),
    ],
)
def test_deterministic_render_failures_persist_only_safe_fields(
    monkeypatch: pytest.MonkeyPatch,
    failure_kind: str,
    expected_code: str,
) -> None:
    event = _event(1)
    if failure_kind == "serialization":
        event["severity"] = object()
    else:
        monkeypatch.setitem(
            __import__(
                "tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports",
                fromlist=["settings"],
            ).settings,
            "CLAIMS_ANALYTICS_EXPORT_MAX_BYTES",
            1,
        )
    db = ArtifactDB([event])
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())

    with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
        process_export_artifact(
            db,
            owner_user_id="7",
            export_id=row["export_id"],
            job_id=42,
        )

    stored = db.artifacts[row["export_id"]]
    assert exc_info.value.code == expected_code
    assert stored["status"] == "failed"
    assert stored["error_code"] == expected_code
    assert stored["error_message"] == exc_info.value.public_message
    assert stored["payload_json"] is None
    assert stored["payload_csv"] is None


def test_transient_sqlite_error_remains_concrete_and_raw_text_is_not_persisted() -> None:
    secret = "database is locked at /private/owner-7.db"
    db = ArtifactDB()
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())

    def locked_page(**_: Any) -> list[dict[str, Any]]:
        raise sqlite3.OperationalError(secret)

    db.list_claims_monitoring_events_page = locked_page  # type: ignore[method-assign]

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        process_export_artifact(
            db,
            owner_user_id="7",
            export_id=row["export_id"],
            job_id=42,
        )

    stored = db.artifacts[row["export_id"]]
    assert stored["status"] == "failed"
    assert stored["error_code"] == "claims_export_storage_unavailable"
    assert stored["error_message"] == "Claims analytics export storage is temporarily unavailable."
    assert secret not in json.dumps(stored, default=str)


def test_domain_failure_losing_to_ready_returns_exact_skip() -> None:
    db = ArtifactDB()
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())

    def serialization_failure(**_: Any) -> list[dict[str, Any]]:
        raise ClaimsAnalyticsExportError(
            "Safe deterministic failure.",
            code="claims_export_serialization_failed",
        )

    def ready_winner(artifact: dict[str, Any]) -> None:
        artifact.update(status="ready", payload_json='{"winner":true}')

    db.list_claims_monitoring_events_page = serialization_failure  # type: ignore[method-assign]
    db.failure_transition_hook = ready_winner

    result = process_export_artifact(
        db,
        owner_user_id="7",
        export_id=row["export_id"],
        job_id=42,
    )

    stored = db.artifacts[row["export_id"]]
    assert result == {
        "outcome": "skipped",
        "reason": "already_ready",
        "export_id": row["export_id"],
    }
    assert stored["status"] == "ready"
    assert stored["payload_json"] == '{"winner":true}'
    assert stored["error_code"] is None


def test_transient_db_failure_losing_to_ready_returns_exact_skip() -> None:
    db = ArtifactDB()
    row = create_queued_artifact(db, owner_user_id="7", normalized=_normalized())

    def transient_failure(**_: Any) -> list[dict[str, Any]]:
        raise sqlite3.OperationalError("database is locked at /private/owner-7.db")

    def ready_winner(artifact: dict[str, Any]) -> None:
        artifact.update(status="ready", payload_json='{"winner":true}')

    db.list_claims_monitoring_events_page = transient_failure  # type: ignore[method-assign]
    db.failure_transition_hook = ready_winner

    result = process_export_artifact(
        db,
        owner_user_id="7",
        export_id=row["export_id"],
        job_id=42,
    )

    stored = db.artifacts[row["export_id"]]
    assert result == {
        "outcome": "skipped",
        "reason": "already_ready",
        "export_id": row["export_id"],
    }
    assert stored["status"] == "ready"
    assert stored["payload_json"] == '{"winner":true}'
    assert stored["error_code"] is None
