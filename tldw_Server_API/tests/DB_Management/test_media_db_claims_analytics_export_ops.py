from __future__ import annotations

from pathlib import Path

import pytest

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
        )

        rows = db.list_claims_analytics_exports("1")

        assert db.count_claims_analytics_exports("1") == 1
        assert len(rows) == 1
        assert rows[0]["job_id"] == 41
        assert rows[0]["error_code"] == "claims_export_safe_code"
        assert rows[0]["snapshot_at"] == "2026-08-08T12:00:00.000Z"
        assert "payload_json" not in rows[0]
        assert "payload_csv" not in rows[0]
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
        assert all("job_id" in row and "error_code" in row and "snapshot_at" in row for row in rows)
        assert all("payload_json" not in row and "payload_csv" not in row for row in rows)
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
