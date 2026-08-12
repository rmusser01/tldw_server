from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Claims_Extraction.claims_analytics_exports import (
    ClaimsAnalyticsExportError,
    process_export_artifact,
    render_export,
)
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import (
    MediaDatabase,
)

pytestmark = pytest.mark.unit


ALLOWED_EXPORT_TRANSITIONS = {
    ("queued", "processing"),
    ("queued", "failed"),
    ("processing", "ready"),
    ("processing", "failed"),
    ("failed", "processing"),
    ("ready", "ready"),
}

GENERIC_STATUS_TRANSITIONS = ALLOWED_EXPORT_TRANSITIONS - {("processing", "ready")}

EXPORT_STATUSES = ("queued", "processing", "ready", "failed")
_CANONICAL_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$"
)


def _make_db(tmp_path: Path, name: str) -> MediaDatabase:
    db = MediaDatabase(db_path=str(tmp_path / name), client_id="claims-analytics-helper")
    db.initialize_db()
    return db


def _seed_export(
    db: MediaDatabase,
    *,
    export_id: str,
    user_id: str = "1",
    format: str = "json",
    status: str = "queued",
    job_id: int | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    snapshot_at: str | None = "2026-08-08T12:00:00.000Z",
) -> dict[str, object]:
    return db.create_claims_analytics_export(
        export_id=export_id,
        user_id=user_id,
        format=format,
        status=status,
        payload_json='{"events":[]}' if status == "ready" and format == "json" else None,
        payload_csv="id\n" if status == "ready" and format == "csv" else None,
        filters_json="{}",
        pagination_json='{"limit":10,"offset":0}',
        job_id=job_id,
        error_code=error_code,
        error_message=error_message,
        snapshot_at=snapshot_at,
    )


def test_claims_analytics_export_row_projections_canonicalize_native_datetimes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-native-datetimes.db")

    class _Cursor:
        def __init__(
            self,
            *,
            one: dict[str, object] | None = None,
            rows: list[dict[str, object]] | None = None,
        ) -> None:
            self._one = one
            self._rows = rows or []

        def fetchone(self) -> dict[str, object] | None:
            return self._one

        def fetchall(self) -> list[dict[str, object]]:
            return self._rows

    aware = datetime(2026, 8, 8, 5, 6, 7, 987654, tzinfo=timezone(timedelta(hours=-7)))
    naive = datetime(2026, 8, 8, 12, 6, 7, 123456)
    get_row = {
        "export_id": "exp-get-native-datetimes",
        "user_id": "1",
        "format": "json",
        "status": "queued",
        "payload_json": '{"events":[]}',
        "snapshot_at": aware,
        "snapshot_event_id": 19,
        "created_at": naive,
        "updated_at": "preserve-this-string",
    }
    list_row = {
        "export_id": "exp-list-native-datetimes",
        "user_id": "1",
        "format": "json",
        "status": "queued",
        "snapshot_at": None,
        "snapshot_event_id": None,
        "created_at": aware,
        "updated_at": naive,
    }
    maintenance_row = {
        "export_id": "exp-maintenance-native-datetimes",
        "user_id": "1",
        "format": "json",
        "status": "queued",
        "snapshot_at": "preserve-this-string",
        "snapshot_event_id": None,
        "created_at": None,
        "updated_at": aware,
    }

    def _execute_query(query: str | tuple[str, ...], *args: object, **kwargs: object) -> _Cursor:
        sql = query[0] if isinstance(query, tuple) else query
        if "LIMIT 1" in sql:
            return _Cursor(one=get_row)
        if "OFFSET ?" in sql:
            return _Cursor(rows=[list_row])
        return _Cursor(rows=[maintenance_row])

    try:
        monkeypatch.setattr(db, "execute_query", _execute_query)

        fetched = db.get_claims_analytics_export("exp-get-native-datetimes", user_id="1")
        listed = db.list_claims_analytics_exports("1")
        maintenance = db.list_claims_analytics_exports_for_maintenance(user_id="1")

        assert fetched["snapshot_at"] == "2026-08-08T12:06:07.987Z"
        assert fetched["snapshot_event_id"] == 19
        assert fetched["created_at"] == "2026-08-08T12:06:07.123Z"
        assert fetched["updated_at"] == "preserve-this-string"
        assert fetched["payload_json"] == '{"events":[]}'
        assert listed[0]["snapshot_at"] is None
        assert listed[0]["snapshot_event_id"] is None
        assert listed[0]["created_at"] == "2026-08-08T12:06:07.987Z"
        assert listed[0]["updated_at"] == "2026-08-08T12:06:07.123Z"
        assert maintenance[0]["snapshot_at"] == "preserve-this-string"
        assert maintenance[0]["snapshot_event_id"] is None
        assert maintenance[0]["created_at"] is None
        assert maintenance[0]["updated_at"] == "2026-08-08T12:06:07.987Z"
    finally:
        db.close_connection()


def test_create_claims_analytics_export_returns_freshly_readable_row(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-create.db")
    try:
        row = db.create_claims_analytics_export(
            export_id="exp-create-1",
            user_id="1",
            format="json",
            status="ready",
            payload_json='{"events":[]}',
            filters_json='{"severity":"high"}',
            pagination_json='{"limit":10}',
            job_id=37,
            snapshot_at="2026-08-08T12:00:00.000Z",
            snapshot_event_id=19,
        )

        assert row["export_id"] == "exp-create-1"
        assert row["user_id"] == "1"
        assert row["format"] == "json"
        assert row["status"] == "ready"
        assert row["payload_json"] == '{"events":[]}'
        assert row["filters_json"] == '{"severity":"high"}'
        assert row["pagination_json"] == '{"limit":10}'
        assert row["job_id"] == 37
        assert row["error_code"] is None
        assert row["snapshot_at"] == "2026-08-08T12:00:00.000Z"
        assert row["snapshot_event_id"] == 19
        assert row["created_at"]
        assert row["updated_at"]
    finally:
        db.close_connection()


def test_get_claims_analytics_export_requires_matching_owner(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-get.db")
    try:
        db.create_claims_analytics_export(
            export_id="exp-get-1",
            user_id="1",
            format="csv",
            status="ready",
            payload_csv="id,value\n1,test\n",
        )

        with pytest.raises(TypeError):
            db.get_claims_analytics_export("exp-get-1")  # type: ignore[call-arg]
        assert db.get_claims_analytics_export("exp-get-1", user_id="2") == {}
        row = db.get_claims_analytics_export("exp-get-1", user_id="1")

        assert row["export_id"] == "exp-get-1"
        assert row["user_id"] == "1"
        assert row["payload_csv"] == "id,value\n1,test\n"
    finally:
        db.close_connection()


def test_claims_analytics_export_list_and_count_stay_in_filter_parity(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-list.db")
    try:
        db.create_claims_analytics_export(
            export_id="exp-ready-json",
            user_id="1",
            format="json",
            status="ready",
            payload_json='{"events":[]}',
        )
        db.create_claims_analytics_export(
            export_id="exp-ready-csv",
            user_id="1",
            format="csv",
            status="ready",
            payload_csv="id\n",
        )
        db.create_claims_analytics_export(
            export_id="exp-failed-json",
            user_id="1",
            format="json",
            status="failed",
        )
        db.create_claims_analytics_export(
            export_id="exp-other-user",
            user_id="2",
            format="json",
            status="ready",
            payload_json='{"events":[]}',
        )

        rows = db.list_claims_analytics_exports(
            "1",
            status="ready",
            format="json",
            limit=50,
            offset=0,
        )
        total = db.count_claims_analytics_exports("1", status="ready", format="json")

        assert [row["export_id"] for row in rows] == ["exp-ready-json"]
        assert total == len(rows) == 1
    finally:
        db.close_connection()


def test_claims_analytics_export_list_projects_job_fields_without_payloads(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-list-projection.db")
    try:
        db.create_claims_analytics_export(
            export_id="exp-projection",
            user_id="1",
            format="json",
            status="failed",
            filters_json="{}",
            pagination_json='{"limit":10,"offset":0}',
            job_id=41,
            error_code="claims_export_safe_code",
            snapshot_at="2026-08-08T12:00:00.000Z",
            snapshot_event_id=19,
        )

        rows = db.list_claims_analytics_exports("1")

        assert db.count_claims_analytics_exports("1") == 1
        assert len(rows) == 1
        assert rows[0]["job_id"] == 41
        assert rows[0]["error_code"] == "claims_export_safe_code"
        assert rows[0]["snapshot_at"] == "2026-08-08T12:00:00.000Z"
        assert rows[0]["snapshot_event_id"] == 19
        assert "payload_json" not in rows[0]
        assert "payload_csv" not in rows[0]
    finally:
        db.close_connection()


def test_claims_analytics_export_list_is_stable_for_tied_creation_timestamps(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-tied-list.db")
    try:
        for export_id in ("exp-a", "exp-c", "exp-b"):
            _seed_export(db, export_id=export_id)
        db.execute_query(
            "UPDATE claims_analytics_exports SET created_at = ? WHERE user_id = ?",
            ("2026-08-08T12:00:00.000Z", "1"),
            commit=True,
        )

        first_page = db.list_claims_analytics_exports("1", limit=2, offset=0)
        second_page = db.list_claims_analytics_exports("1", limit=2, offset=2)

        assert [row["export_id"] for row in first_page] == ["exp-c", "exp-b"]
        assert [row["export_id"] for row in second_page] == ["exp-a"]
    finally:
        db.close_connection()


def test_attach_claims_analytics_export_job_is_owner_scoped_and_idempotent(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-attach.db")
    try:
        _seed_export(db, export_id="exp-attach")

        assert db.attach_claims_analytics_export_job(
            export_id="exp-attach", user_id="2", job_id=51
        ) is False
        assert db.attach_claims_analytics_export_job(
            export_id="exp-attach", user_id="1", job_id=51
        ) is True
        assert db.attach_claims_analytics_export_job(
            export_id="exp-attach", user_id="1", job_id=51
        ) is True
        assert db.attach_claims_analytics_export_job(
            export_id="exp-attach", user_id="1", job_id=52
        ) is False
        assert db.get_claims_analytics_export(
            "exp-attach", user_id="1"
        )["job_id"] == 51
    finally:
        db.close_connection()


@pytest.mark.parametrize(("from_status", "to_status"), sorted(GENERIC_STATUS_TRANSITIONS))
def test_claims_analytics_export_allows_only_declared_transition(
    tmp_path: Path,
    from_status: str,
    to_status: str,
) -> None:
    db = _make_db(tmp_path, f"claims-analytics-transition-{from_status}-{to_status}.db")
    try:
        export_id = f"exp-{from_status}-{to_status}"
        _seed_export(db, export_id=export_id, status=from_status)

        assert db.transition_claims_analytics_export_status(
            export_id=export_id,
            user_id="1",
            from_statuses=(from_status,),
            to_status=to_status,
        ) is True
        assert db.get_claims_analytics_export(
            export_id, user_id="1"
        )["status"] == to_status
    finally:
        db.close_connection()


def test_generic_transition_cannot_mark_processing_export_ready_without_payload(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-processing-ready.db")
    try:
        _seed_export(
            db,
            export_id="exp-processing-ready",
            status="processing",
            error_code="original_error",
            error_message="original safe error",
        )

        assert db.transition_claims_analytics_export_status(
            export_id="exp-processing-ready",
            user_id="1",
            from_statuses=("processing",),
            to_status="ready",
        ) is False
        row = db.get_claims_analytics_export("exp-processing-ready", user_id="1")
        assert row["status"] == "processing"
        assert row["payload_json"] is None
        assert row["payload_csv"] is None
        assert row["error_code"] == "original_error"
        assert row["error_message"] == "original safe error"
    finally:
        db.close_connection()


def test_ready_to_ready_is_non_mutating_owner_scoped_observation(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-ready-observation.db")
    try:
        _seed_export(db, export_id="exp-ready-observation", status="ready")
        db.execute_query(
            (
                "UPDATE claims_analytics_exports "
                "SET updated_at = ?, error_code = ?, error_message = ? "
                "WHERE export_id = ? AND user_id = ?"
            ),
            (
                "2026-08-08T12:34:56.000Z",
                "legacy_error",
                "legacy safe error",
                "exp-ready-observation",
                "1",
            ),
            commit=True,
        )
        before = db.get_claims_analytics_export("exp-ready-observation", user_id="1")

        assert db.transition_claims_analytics_export_status(
            export_id="exp-ready-observation",
            user_id="2",
            from_statuses=("ready",),
            to_status="ready",
            error_code="replacement_error",
            error_message="replacement safe error",
        ) is False
        assert db.transition_claims_analytics_export_status(
            export_id="exp-ready-observation",
            user_id="1",
            from_statuses=("ready", "ready"),
            to_status="ready",
            error_code="replacement_error",
            error_message="replacement safe error",
        ) is True

        after = db.get_claims_analytics_export("exp-ready-observation", user_id="1")
        assert after["updated_at"] == before["updated_at"]
        assert after["error_code"] == before["error_code"]
        assert after["error_message"] == before["error_message"]
        assert after["payload_json"] == before["payload_json"]
        assert after["payload_csv"] == before["payload_csv"]
    finally:
        db.close_connection()


def test_claims_analytics_export_rejects_mixed_valid_and_invalid_source_statuses(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-reject-mixed-sources.db")
    try:
        _seed_export(
            db,
            export_id="exp-reject-mixed-sources",
            status="queued",
            error_code="original_error",
            error_message="original safe error",
        )

        assert db.transition_claims_analytics_export_status(
            export_id="exp-reject-mixed-sources",
            user_id="1",
            from_statuses=("queued", "ready"),
            to_status="processing",
            error_code="replacement_error",
            error_message="replacement safe error",
        ) is False
        row = db.get_claims_analytics_export(
            "exp-reject-mixed-sources",
            user_id="1",
        )
        assert row["status"] == "queued"
        assert row["error_code"] == "original_error"
        assert row["error_message"] == "original safe error"
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("from_status", "to_status"),
    [
        (from_status, to_status)
        for from_status in EXPORT_STATUSES
        for to_status in EXPORT_STATUSES
        if (from_status, to_status) not in ALLOWED_EXPORT_TRANSITIONS
    ],
)
def test_claims_analytics_export_rejects_undeclared_transition(
    tmp_path: Path,
    from_status: str,
    to_status: str,
) -> None:
    db = _make_db(tmp_path, f"claims-analytics-reject-{from_status}-{to_status}.db")
    try:
        export_id = f"exp-reject-{from_status}-{to_status}"
        _seed_export(db, export_id=export_id, status=from_status)

        assert db.transition_claims_analytics_export_status(
            export_id=export_id,
            user_id="1",
            from_statuses=(from_status,),
            to_status=to_status,
            error_code="late_failure",
        ) is False
        assert db.get_claims_analytics_export(
            export_id, user_id="1"
        )["status"] == from_status
    finally:
        db.close_connection()


def test_claims_analytics_export_transition_is_owner_scoped(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-transition-owner.db")
    try:
        _seed_export(db, export_id="exp-transition-owner", status="queued")

        assert db.transition_claims_analytics_export_status(
            export_id="exp-transition-owner",
            user_id="2",
            from_statuses=("queued",),
            to_status="failed",
            error_code="claims_export_safe_code",
        ) is False
        row = db.get_claims_analytics_export("exp-transition-owner", user_id="1")
        assert row["status"] == "queued"
        assert row["error_code"] is None
    finally:
        db.close_connection()


def test_mark_claims_analytics_export_ready_requires_processing_and_one_payload(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-ready.db")
    try:
        _seed_export(
            db,
            export_id="exp-ready",
            status="processing",
            error_code="old_error",
            error_message="safe old error",
        )

        assert db.mark_claims_analytics_export_ready(
            export_id="exp-ready",
            user_id="1",
            payload_json=None,
            payload_csv=None,
        ) is False
        assert db.mark_claims_analytics_export_ready(
            export_id="exp-ready",
            user_id="1",
            payload_json='{"events":[]}',
            payload_csv="id\n",
        ) is False
        assert db.mark_claims_analytics_export_ready(
            export_id="exp-ready",
            user_id="2",
            payload_json='{"events":[]}',
            payload_csv=None,
        ) is False
        assert db.mark_claims_analytics_export_ready(
            export_id="exp-ready",
            user_id="1",
            payload_json='{"events":[]}',
            payload_csv=None,
        ) is True

        row = db.get_claims_analytics_export("exp-ready", user_id="1")
        assert row["status"] == "ready"
        assert row["payload_json"] == '{"events":[]}'
        assert row["payload_csv"] is None
        assert row["error_code"] is None
        assert row["error_message"] is None
        assert db.mark_claims_analytics_export_ready(
            export_id="exp-ready",
            user_id="1",
            payload_json=None,
            payload_csv="replacement\n",
        ) is False
        assert db.get_claims_analytics_export(
            "exp-ready", user_id="1"
        )["payload_json"] == '{"events":[]}'
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    ("format", "payload_json", "payload_csv"),
    [
        ("json", None, "id\n"),
        ("csv", '{"events":[]}', None),
    ],
)
def test_mark_claims_analytics_export_ready_rejects_payload_for_wrong_format(
    tmp_path: Path,
    format: str,
    payload_json: str | None,
    payload_csv: str | None,
) -> None:
    db = _make_db(tmp_path, f"claims-analytics-ready-format-{format}.db")
    try:
        _seed_export(
            db,
            export_id=f"exp-ready-format-{format}",
            format=format,
            status="processing",
            error_code="original_error",
            error_message="original safe error",
        )

        assert db.mark_claims_analytics_export_ready(
            export_id=f"exp-ready-format-{format}",
            user_id="1",
            payload_json=payload_json,
            payload_csv=payload_csv,
        ) is False
        row = db.get_claims_analytics_export(
            f"exp-ready-format-{format}",
            user_id="1",
        )
        assert row["status"] == "processing"
        assert row["payload_json"] is None
        assert row["payload_csv"] is None
        assert row["error_code"] == "original_error"
        assert row["error_message"] == "original safe error"
    finally:
        db.close_connection()


@pytest.mark.parametrize(
    (
        "format",
        "status",
        "payload_json",
        "payload_csv",
        "error_code",
        "error_message",
    ),
    [
        ("xml", "queued", None, None, None, None),
        ("json", "unknown", None, None, None, None),
        ("json", "ready", None, None, None, None),
        ("json", "ready", None, "id\n", None, None),
        ("csv", "ready", '{"events":[]}', None, None, None),
        ("json", "ready", '{"events":[]}', "id\n", None, None),
        ("json", "processing", '{"events":[]}', None, None, None),
        ("csv", "failed", None, "id\n", "failed", "safe failure"),
        ("json", "ready", '{"events":[]}', None, "stale", None),
        ("csv", "ready", None, "id\n", None, "stale safe error"),
    ],
)
def test_create_claims_analytics_export_rejects_contradictory_artifact(
    tmp_path: Path,
    format: str,
    status: str,
    payload_json: str | None,
    payload_csv: str | None,
    error_code: str | None,
    error_message: str | None,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-create-invalid.db")
    try:
        with pytest.raises(ValueError):
            db.create_claims_analytics_export(
                export_id="exp-create-invalid",
                user_id="1",
                format=format,
                status=status,
                payload_json=payload_json,
                payload_csv=payload_csv,
                error_code=error_code,
                error_message=error_message,
            )
        assert db.get_claims_analytics_export("exp-create-invalid", user_id="1") == {}
    finally:
        db.close_connection()


@pytest.mark.parametrize("invalid_job_id", [True, 0, -1, "1", 1.0])
def test_claims_analytics_export_rejects_invalid_job_ids(
    tmp_path: Path,
    invalid_job_id: object,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-invalid-job-id.db")
    try:
        _seed_export(db, export_id="exp-attach-invalid-job")

        with pytest.raises(ValueError, match="job_id must be a positive integer"):
            db.attach_claims_analytics_export_job(
                export_id="exp-attach-invalid-job",
                user_id="1",
                job_id=invalid_job_id,  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="job_id must be a positive integer"):
            db.create_claims_analytics_export(
                export_id="exp-create-invalid-job",
                user_id="1",
                format="json",
                status="queued",
                job_id=invalid_job_id,  # type: ignore[arg-type]
            )
        assert db.get_claims_analytics_export("exp-attach-invalid-job", user_id="1")[
            "job_id"
        ] is None
        assert db.get_claims_analytics_export("exp-create-invalid-job", user_id="1") == {}
    finally:
        db.close_connection()


def test_claims_analytics_export_accepts_minimum_positive_job_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-minimum-job-id.db")
    try:
        created = _seed_export(db, export_id="exp-create-minimum-job", job_id=1)
        _seed_export(db, export_id="exp-attach-minimum-job")

        assert created["job_id"] == 1
        assert db.attach_claims_analytics_export_job(
            export_id="exp-attach-minimum-job",
            user_id="1",
            job_id=1,
        ) is True
        assert db.get_claims_analytics_export(
            "exp-attach-minimum-job",
            user_id="1",
        )["job_id"] == 1
    finally:
        db.close_connection()


def test_claims_analytics_export_maintenance_list_is_owner_scoped_and_deterministic(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-maintenance.db")
    try:
        _seed_export(db, export_id="exp-b", job_id=2)
        _seed_export(db, export_id="exp-a", job_id=1)
        _seed_export(db, export_id="exp-c", job_id=3)
        _seed_export(db, export_id="exp-other", user_id="2", job_id=4)
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id IN (?, ?)",
            ("2026-08-08T12:00:00.000Z", "exp-a", "exp-b"),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id IN (?, ?)",
            ("2026-08-08T12:00:01.000Z", "exp-c", "exp-other"),
            commit=True,
        )

        rows = db.list_claims_analytics_exports_for_maintenance(user_id="1", limit=10)

        assert [row["export_id"] for row in rows] == ["exp-a", "exp-b", "exp-c"]
        assert all(row["user_id"] == "1" for row in rows)
        assert all(
            "job_id" in row
            and "error_code" in row
            and "snapshot_at" in row
            and "snapshot_event_id" in row
            for row in rows
        )
        assert all("payload_json" not in row and "payload_csv" not in row for row in rows)
    finally:
        db.close_connection()


def test_claims_analytics_export_maintenance_queries_use_composite_indexes(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-maintenance-indexes.db")
    try:
        rotation_plan = db.get_connection().execute(
            "EXPLAIN QUERY PLAN SELECT export_id FROM claims_analytics_exports "
            "WHERE user_id = ? AND status = ? AND export_id > ? "
            "ORDER BY export_id ASC LIMIT ?",
            ("1", "failed", "0", 10),
        ).fetchall()
        age_plan = db.get_connection().execute(
            "EXPLAIN QUERY PLAN SELECT export_id FROM claims_analytics_exports "
            "WHERE user_id = ? AND status = ? AND updated_at < ? "
            "ORDER BY updated_at ASC, export_id ASC LIMIT ?",
            ("1", "ready", "2026-08-08T12:00:00.000Z", 10),
        ).fetchall()

        assert any(
            "idx_claims_analytics_exports_user_status_export_id" in str(row["detail"])
            for row in rotation_plan
        )
        assert any(
            "idx_claims_analytics_exports_user_status_updated_export_id"
            in str(row["detail"])
            for row in age_plan
        )
    finally:
        db.close_connection()


def test_maintenance_list_filters_reconciliation_candidates_before_limit(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-maintenance-reconciliation-filter.db")
    try:
        for index in range(3):
            _seed_export(
                db,
                export_id=f"exp-attached-{index}",
                status="queued",
                job_id=100 + index,
            )
        _seed_export(db, export_id="exp-orphan", status="queued")
        _seed_export(db, export_id="exp-failed", status="failed")
        _seed_export(db, export_id="exp-other-owner", user_id="2", status="queued")
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id LIKE ?",
            ("2026-08-08T12:00:00.000Z", "exp-attached-%"),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id IN (?, ?, ?)",
            (
                "2026-08-08T12:00:01.000Z",
                "exp-orphan",
                "exp-failed",
                "exp-other-owner",
            ),
            commit=True,
        )

        rows = db.list_claims_analytics_exports_for_maintenance(
            user_id="1",
            statuses=("queued",),
            job_id_missing=True,
            limit=2,
        )

        assert [row["export_id"] for row in rows] == ["exp-orphan"]
    finally:
        db.close_connection()


def test_maintenance_list_filters_cleanup_candidates_before_limit(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-maintenance-cleanup-filter.db")
    try:
        for index, status in enumerate(("queued", "processing", "queued")):
            _seed_export(db, export_id=f"exp-active-{index}", status=status)
        _seed_export(db, export_id="exp-old-ready", status="ready")
        _seed_export(db, export_id="exp-fresh-ready", status="ready")
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id LIKE ?",
            ("2026-08-08T12:00:00.000Z", "exp-active-%"),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id = ?",
            ("2026-08-08T12:00:01.000Z", "exp-old-ready"),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id = ?",
            ("2026-08-08T12:00:03.000Z", "exp-fresh-ready"),
            commit=True,
        )

        rows = db.list_claims_analytics_exports_for_maintenance(
            user_id="1",
            statuses=("ready", "failed"),
            updated_before="2026-08-08T12:00:02.000Z",
            limit=1,
        )

        assert [row["export_id"] for row in rows] == ["exp-old-ready"]
    finally:
        db.close_connection()


def test_maintenance_list_supports_bounded_export_id_rotation(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-maintenance-rotation.db")
    try:
        for export_number in (1, 2, 100, 101):
            _seed_export(
                db,
                export_id=f"{export_number:032x}",
                status="failed",
            )

        anchor = f"{50:032x}"
        after = db.list_claims_analytics_exports_for_maintenance(
            user_id="1",
            statuses=("failed",),
            export_id_after=anchor,
            limit=1,
        )
        wrapped = db.list_claims_analytics_exports_for_maintenance(
            user_id="1",
            statuses=("failed",),
            export_id_at_or_before=anchor,
            limit=2,
        )

        assert [row["export_id"] for row in after] == [f"{100:032x}"]
        assert [row["export_id"] for row in wrapped] == [
            f"{1:032x}",
            f"{2:032x}",
        ]
    finally:
        db.close_connection()


def test_delete_claims_analytics_exports_is_exact_owner_scoped_and_uses_updated_at(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-delete.db")
    try:
        _seed_export(db, export_id="exp-old")
        _seed_export(db, export_id="exp-fresh")
        _seed_export(db, export_id="exp-other", user_id="2")
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id IN (?, ?)",
            ("2026-08-08T12:00:00.000Z", "exp-old", "exp-other"),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id = ?",
            ("2026-08-08T12:00:02.000Z", "exp-fresh"),
            commit=True,
        )

        deleted = db.delete_claims_analytics_exports(
            user_id="1",
            export_ids=["exp-old", "exp-fresh", "exp-other"],
            updated_before="2026-08-08T12:00:01.000Z",
        )

        assert deleted == 1
        assert db.get_claims_analytics_export("exp-old", user_id="1") == {}
        assert db.get_claims_analytics_export("exp-fresh", user_id="1")
        assert db.get_claims_analytics_export("exp-other", user_id="2")
        assert db.delete_claims_analytics_exports(
            user_id="1", export_ids=[], updated_before="2026-08-08T12:00:03.000Z"
        ) == 0
    finally:
        db.close_connection()


def test_delete_claims_analytics_exports_chunks_more_than_four_hundred_unique_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-delete-chunks.db")
    calls: list[tuple[str, tuple[object, ...], bool]] = []

    class _Cursor:
        def __init__(self, rowcount: int) -> None:
            self.rowcount = rowcount

    def _capture_query(
        sql: str,
        params: tuple[object, ...],
        *,
        commit: bool = False,
    ) -> _Cursor:
        calls.append((sql, params, commit))
        return _Cursor(len(params) - 2)

    try:
        monkeypatch.setattr(db, "execute_query", _capture_query)
        export_ids = [f"exp-{index}" for index in range(405)]

        deleted = db.delete_claims_analytics_exports(
            user_id="1",
            export_ids=export_ids + export_ids[:10],
            updated_before="2026-08-08T12:00:00.000Z",
        )

        assert deleted == 405
        assert len(calls) == 2
        assert [len(params) - 2 for _, params, _ in calls] == [400, 5]
        assert all("WHERE user_id = ?" in sql for sql, _, _ in calls)
        assert all("AND updated_at < ?" in sql for sql, _, _ in calls)
        assert all(params[0] == "1" for _, params, _ in calls)
        assert all(params[-1] == "2026-08-08T12:00:00.000Z" for _, params, _ in calls)
        assert all(commit is True for _, _, commit in calls)
    finally:
        db.close_connection()


def test_cleanup_claims_analytics_exports_compares_updated_at(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-cleanup-updated.db")
    try:
        _seed_export(db, export_id="exp-touched", status="ready")
        _seed_export(db, export_id="exp-stale", status="ready")
        db.execute_query(
            "UPDATE claims_analytics_exports SET created_at = ?, updated_at = ? WHERE export_id = ?",
            (
                "2000-01-01T00:00:00.000Z",
                "2999-01-01T00:00:00.000Z",
                "exp-touched",
            ),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET created_at = ?, updated_at = ? WHERE export_id = ?",
            (
                "2999-01-01T00:00:00.000Z",
                "2000-01-01T00:00:00.000Z",
                "exp-stale",
            ),
            commit=True,
        )

        assert db.cleanup_claims_analytics_exports(user_id="1", retention_hours=1) == 1
        assert db.get_claims_analytics_export("exp-touched", user_id="1")
        assert db.get_claims_analytics_export("exp-stale", user_id="1") == {}
    finally:
        db.close_connection()


def test_cleanup_claims_analytics_exports_preserves_old_non_ready_rows(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-analytics-cleanup-status.db")
    try:
        for status in ("queued", "processing", "failed", "ready"):
            _seed_export(db, export_id=f"exp-old-{status}", status=status)
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE user_id = ?",
            ("2000-01-01T00:00:00.000Z", "1"),
            commit=True,
        )

        assert db.cleanup_claims_analytics_exports(user_id="1", retention_hours=1) == 1
        assert db.get_claims_analytics_export("exp-old-ready", user_id="1") == {}
        for status in ("queued", "processing", "failed"):
            assert db.get_claims_analytics_export(f"exp-old-{status}", user_id="1")
    finally:
        db.close_connection()


@pytest.mark.parametrize("retention_hours", ["oops", None, 0, -4])
def test_cleanup_claims_analytics_exports_rejects_invalid_retention(
    tmp_path: Path,
    retention_hours: object,
) -> None:
    db = _make_db(tmp_path, "claims-analytics-cleanup.db")
    try:
        db.create_claims_analytics_export(
            export_id="exp-cleanup-1",
            user_id="1",
            format="json",
            status="ready",
            payload_json='{"events":[]}',
        )

        assert db.cleanup_claims_analytics_exports(
            user_id="1",
            retention_hours=retention_hours,  # type: ignore[arg-type]
        ) == 0
        assert db.count_claims_analytics_exports("1") == 1
    finally:
        db.close_connection()


@pytest.mark.integration
def test_claims_analytics_exports_postgres_owner_scoped_crud_and_v24_fields(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(
        db_path=":memory:",
        client_id="claims-analytics-postgres-crud",
        backend=backend,
    )
    try:
        created = db.create_claims_analytics_export(
            export_id="pg-crud-owner-1",
            user_id="owner-1",
            format="csv",
            status="queued",
            filters_json='{"severity":"high"}',
            pagination_json='{"limit":25,"offset":0}',
            snapshot_at="2026-08-08T12:00:00.000Z",
            snapshot_event_id=19,
        )
        db.create_claims_analytics_export(
            export_id="pg-crud-owner-2",
            user_id="owner-2",
            format="json",
            status="queued",
            payload_json=None,
            snapshot_at="2026-08-08T12:00:00.000Z",
        )

        assert created["export_id"] == "pg-crud-owner-1"
        assert created["user_id"] == "owner-1"
        assert created["filters_json"] == '{"severity":"high"}'
        assert created["pagination_json"] == '{"limit":25,"offset":0}'
        assert created["job_id"] is None
        assert created["error_code"] is None
        assert created["snapshot_at"] == "2026-08-08T12:00:00.000Z"
        assert created["snapshot_event_id"] == 19
        assert all(
            isinstance(created[field], str)
            and _CANONICAL_TIMESTAMP_RE.fullmatch(created[field])
            for field in ("snapshot_at", "created_at", "updated_at")
        )

        assert db.get_claims_analytics_export(
            "pg-crud-owner-1", user_id="owner-2"
        ) == {}
        owner_one_rows = db.list_claims_analytics_exports("owner-1")
        assert [row["export_id"] for row in owner_one_rows] == ["pg-crud-owner-1"]
        assert all(
            isinstance(owner_one_rows[0][field], str)
            and _CANONICAL_TIMESTAMP_RE.fullmatch(owner_one_rows[0][field])
            for field in ("snapshot_at", "created_at", "updated_at")
        )
        assert [row["export_id"] for row in db.list_claims_analytics_exports("owner-2")] == [
            "pg-crud-owner-2"
        ]
        assert db.count_claims_analytics_exports("owner-1") == 1
        assert db.count_claims_analytics_exports("owner-2") == 1

        assert db.attach_claims_analytics_export_job(
            export_id="pg-crud-owner-1", user_id="owner-2", job_id=101
        ) is False
        assert db.attach_claims_analytics_export_job(
            export_id="pg-crud-owner-1", user_id="owner-1", job_id=101
        ) is True
        assert db.attach_claims_analytics_export_job(
            export_id="pg-crud-owner-1", user_id="owner-1", job_id=101
        ) is True
        assert db.attach_claims_analytics_export_job(
            export_id="pg-crud-owner-1", user_id="owner-1", job_id=102
        ) is False
        assert db.get_claims_analytics_export(
            "pg-crud-owner-1", user_id="owner-1"
        )["job_id"] == 101

        queued = db.create_claims_analytics_export(
            export_id="a" * 32,
            user_id="1",
            format="json",
            status="queued",
            filters_json='{"end_time":"2026-08-08T12:00:00.000Z"}',
            pagination_json='{"limit":10,"offset":0}',
            snapshot_at="2026-08-08T12:00:00.000Z",
        )
        assert db.attach_claims_analytics_export_job(
            export_id=queued["export_id"], user_id="1", job_id=102
        ) is True

        result = process_export_artifact(
            db,
            owner_user_id="1",
            export_id=queued["export_id"],
            job_id=102,
        )
        ready = db.get_claims_analytics_export(queued["export_id"], user_id="1")

        assert result["outcome"] == "ok"
        assert ready["status"] == "ready"
        assert ready["job_id"] == 102
        assert all(
            isinstance(ready[field], str)
            and _CANONICAL_TIMESTAMP_RE.fullmatch(ready[field])
            for field in ("snapshot_at", "created_at", "updated_at")
        )
        assert json.loads(ready["payload_json"]) == {
            "events": [],
            "filters": {"end_time": "2026-08-08T12:00:00.000Z"},
            "pagination": {"limit": 10, "offset": 0, "total": 0},
        }
    finally:
        db.close_connection()


@pytest.mark.integration
def test_claims_analytics_exports_postgres_transitions_stay_ready_and_delete_by_updated_at(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(
        db_path=":memory:",
        client_id="claims-analytics-postgres-lifecycle",
        backend=backend,
    )
    try:
        _seed_export(db, export_id="pg-lifecycle", user_id="owner-1")
        assert db.transition_claims_analytics_export_status(
            export_id="pg-lifecycle",
            user_id="owner-1",
            from_statuses=("queued",),
            to_status="ready",
        ) is False
        assert db.transition_claims_analytics_export_status(
            export_id="pg-lifecycle",
            user_id="owner-2",
            from_statuses=("queued",),
            to_status="processing",
        ) is False
        assert db.transition_claims_analytics_export_status(
            export_id="pg-lifecycle",
            user_id="owner-1",
            from_statuses=("queued",),
            to_status="processing",
        ) is True
        assert db.mark_claims_analytics_export_ready(
            export_id="pg-lifecycle",
            user_id="owner-1",
            payload_json='{"events":[]}',
            payload_csv=None,
        ) is True
        ready = db.get_claims_analytics_export("pg-lifecycle", user_id="owner-1")

        assert db.transition_claims_analytics_export_status(
            export_id="pg-lifecycle",
            user_id="owner-1",
            from_statuses=("processing",),
            to_status="failed",
            error_code="late_failure",
        ) is False
        assert db.mark_claims_analytics_export_ready(
            export_id="pg-lifecycle",
            user_id="owner-1",
            payload_json=None,
            payload_csv="replacement\n",
        ) is False
        after_late_attempt = db.get_claims_analytics_export(
            "pg-lifecycle", user_id="owner-1"
        )
        assert after_late_attempt["status"] == "ready"
        assert after_late_attempt["payload_json"] == ready["payload_json"]
        assert after_late_attempt["updated_at"] == ready["updated_at"]

        _seed_export(db, export_id="pg-delete-equal", user_id="owner-1", status="ready")
        _seed_export(db, export_id="pg-delete-old", user_id="owner-1", status="ready")
        _seed_export(db, export_id="pg-delete-other-owner", user_id="owner-2", status="ready")
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id IN (?, ?, ?)",
            (
                "2026-08-08T12:00:00.000Z",
                "pg-delete-equal",
                "pg-delete-old",
                "pg-delete-other-owner",
            ),
            commit=True,
        )
        db.execute_query(
            "UPDATE claims_analytics_exports SET updated_at = ? WHERE export_id = ?",
            ("2026-08-08T11:59:59.999Z", "pg-delete-old"),
            commit=True,
        )

        assert db.delete_claims_analytics_exports(
            user_id="owner-1",
            export_ids=[
                "pg-delete-equal",
                "pg-delete-old",
                "pg-delete-other-owner",
            ],
            updated_before="2026-08-08T12:00:00.000Z",
        ) == 1
        assert db.get_claims_analytics_export("pg-delete-equal", user_id="owner-1")
        assert db.get_claims_analytics_export("pg-delete-old", user_id="owner-1") == {}
        assert db.get_claims_analytics_export(
            "pg-delete-other-owner", user_id="owner-2"
        )
    finally:
        db.close_connection()


@pytest.mark.integration
def test_claims_monitoring_event_postgres_pages_are_bounded_with_equal_timestamps(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = MediaDatabase(
        db_path=":memory:",
        client_id="claims-analytics-postgres-events",
        backend=backend,
    )
    try:
        payload = '{"provider":"true","model":"7","text":"東京"}'
        payload_size = len(payload.encode("utf-8"))
        event_ids = [
            int(
                db.insert_claims_monitoring_event(
                    user_id="owner-1",
                    event_type="unsupported_ratio",
                    severity="high",
                    payload_json=payload,
                )["id"]
            )
            for _ in range(5)
        ]
        equal_timestamp = "2026-08-08T12:00:00.000Z"
        db.execute_query(
            "UPDATE claims_monitoring_events SET created_at = ? WHERE id IN (?, ?, ?, ?, ?)",
            (equal_timestamp, *event_ids),
            commit=True,
        )

        first_page = db.list_claims_monitoring_events_page(
            user_id="owner-1", limit=2
        )
        second_page = db.list_claims_monitoring_events_page(
            user_id="owner-1",
            after_created_at=first_page[-1]["created_at"],
            after_id=int(first_page[-1]["id"]),
            limit=2,
        )
        third_page = db.list_claims_monitoring_events_page(
            user_id="owner-1",
            after_created_at=second_page[-1]["created_at"],
            after_id=int(second_page[-1]["id"]),
            limit=2,
        )

        assert [row["id"] for row in first_page] == event_ids[:2]
        assert [row["id"] for row in second_page] == event_ids[2:4]
        assert [row["id"] for row in third_page] == event_ids[4:]
        assert len({row["id"] for row in first_page + second_page + third_page}) == 5
        assert all(
            set(row) == {"id", "user_id", "created_at"}
            for row in first_page + second_page + third_page
        )
        assert db.get_claims_monitoring_event_payload_bounded(
            user_id="owner-2",
            event_id=event_ids[0],
            max_bytes=payload_size,
        ) == {}
        assert db.get_claims_monitoring_event_payload_bounded(
            user_id="owner-1",
            event_id=event_ids[0],
            max_bytes=payload_size - 1,
        ) == {
            "payload_json": None,
            "payload_size_bytes": payload_size,
        }
        assert db.get_claims_monitoring_event_payload_bounded(
            user_id="owner-1",
            event_id=event_ids[0],
            max_bytes=payload_size,
        ) == {
            "payload_json": payload,
            "payload_size_bytes": payload_size,
        }
        db.insert_claims_monitoring_event(
            user_id="owner-1",
            event_type="unsupported_ratio",
            severity="high",
            payload_json='{"provider":true,"model":7}',
        )
        filtered = db.list_claims_monitoring_events_page(
            user_id="owner-1",
            provider="true",
            model="7",
            limit=10,
        )
        assert [row["id"] for row in filtered] == event_ids

        escaped_payload = '{"text":"\\u6771\\u4eac"}'
        canonical_payload = '{"text":"東京"}'
        canonical_size = len(canonical_payload.encode("utf-8"))
        escaped_event = db.insert_claims_monitoring_event(
            user_id="owner-1",
            event_type="escaped-unicode",
            payload_json=escaped_payload,
        )
        assert db.get_claims_monitoring_event_payload_bounded(
            user_id="owner-1",
            event_id=int(escaped_event["id"]),
            max_bytes=canonical_size,
        ) == {
            "payload_json": canonical_payload,
            "payload_size_bytes": canonical_size,
        }
    finally:
        db.close_connection()


@pytest.mark.parametrize("format", ["json", "csv"])
def test_sqlite_render_accepts_escaped_unicode_at_exact_artifact_boundary(
    tmp_path: Path,
    format: str,
) -> None:
    db = _make_db(tmp_path, f"claims-analytics-escaped-unicode-{format}.db")
    try:
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="escaped-unicode",
            severity="info",
            payload_json='{"text":"' + ("\\u6771\\u4eac" * 32) + '"}',
        )
        render_kwargs = {
            "owner_user_id": "1",
            "format": format,
            "filters": {},
            "pagination": {"limit": 1, "offset": 0},
            "snapshot_at": "2099-01-01T00:00:00.000Z",
            "snapshot_event_id": db.get_claims_monitoring_event_high_water(user_id="1"),
        }
        baseline = render_export(db, max_bytes=100_000, **render_kwargs)
        exact_size = baseline["size_bytes"]

        exact = render_export(db, max_bytes=exact_size, **render_kwargs)
        assert exact["size_bytes"] == exact_size

        with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
            render_export(db, max_bytes=exact_size - 1, **render_kwargs)
        assert exc_info.value.code == "claims_export_too_large"
    finally:
        db.close_connection()


@pytest.mark.parametrize("format", ["json", "csv"])
def test_sqlite_render_rejects_non_finite_numbers_in_monitoring_payload(
    tmp_path: Path,
    format: str,
) -> None:
    db = _make_db(tmp_path, f"claims-analytics-non-finite-{format}.db")
    try:
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="large-exponent",
            payload_json='{"value":1e999}',
        )

        with pytest.raises(ClaimsAnalyticsExportError) as exc_info:
            render_export(
                db,
                owner_user_id="1",
                format=format,
                filters={},
                pagination={"limit": 1, "offset": 0},
                snapshot_at="2099-01-01T00:00:00.000Z",
                snapshot_event_id=db.get_claims_monitoring_event_high_water(user_id="1"),
                max_bytes=100_000,
            )
        assert exc_info.value.code == "claims_export_serialization_failed"
    finally:
        db.close_connection()
