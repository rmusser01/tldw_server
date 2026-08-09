from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import (
    MediaDatabase,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime import (
    claims_monitoring_event_ops,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    get_claims_monitoring_event as helper_get_claims_monitoring_event,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    get_latest_claims_monitoring_event_delivery as helper_get_latest_claims_monitoring_event_delivery,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    has_successful_claims_monitoring_event_delivery as helper_has_successful_claims_monitoring_event_delivery,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    insert_claims_monitoring_event as helper_insert_claims_monitoring_event,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    list_claims_monitoring_events as helper_list_claims_monitoring_events,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    list_undelivered_claims_monitoring_events as helper_list_undelivered_claims_monitoring_events,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    mark_claims_monitoring_events_delivered as helper_mark_claims_monitoring_events_delivered,
)

pytestmark = pytest.mark.unit


def _make_db(tmp_path: Path, name: str) -> MediaDatabase:
    db = MediaDatabase(db_path=str(tmp_path / name), client_id="claims-monitoring-event-helper")
    db.initialize_db()
    return db


def test_insert_claims_monitoring_event_writes_null_delivered_at_and_rebinds_method(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-insert.db")
    try:
        assert db.insert_claims_monitoring_event.__func__ is helper_insert_claims_monitoring_event

        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="warning",
            payload_json='{"status":"failure"}',
        )

        row = db.execute_query(
            "SELECT user_id, event_type, severity, payload_json, created_at, delivered_at "
            "FROM claims_monitoring_events ORDER BY id ASC"
        ).fetchone()

        assert row is not None
        assert row["user_id"] == "1"
        assert row["event_type"] == "webhook_delivery"
        assert row["severity"] == "warning"
        assert row["payload_json"] == '{"status":"failure"}'
        assert row["created_at"]
        assert row["delivered_at"] is None
    finally:
        db.close_connection()


def test_insert_claims_monitoring_event_returns_inserted_row_and_gets_by_id(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-get.db")
    try:
        assert db.get_claims_monitoring_event.__func__ is helper_get_claims_monitoring_event

        created = db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"alert_id":9}',
        )

        assert isinstance(created["id"], int)
        loaded = db.get_claims_monitoring_event(int(created["id"]))
        assert loaded["id"] == created["id"]
        assert loaded["user_id"] == "1"
        assert loaded["event_type"] == "unsupported_ratio"
        assert loaded["severity"] == "warning"
        assert loaded["payload_json"] == '{"alert_id":9}'
        assert loaded["delivered_at"] is None
    finally:
        db.close_connection()


def test_claims_monitoring_event_high_water_is_owner_scoped(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-high-water.db")
    try:
        assert db.get_claims_monitoring_event_high_water(user_id="1") == 0
        first = db.insert_claims_monitoring_event(user_id="1", event_type="first")
        second = db.insert_claims_monitoring_event(user_id="2", event_type="other")
        third = db.insert_claims_monitoring_event(user_id="1", event_type="last")

        assert db.get_claims_monitoring_event_high_water(user_id="1") == third["id"]
        assert db.get_claims_monitoring_event_high_water(user_id="2") == second["id"]
        assert first["id"] < third["id"]
    finally:
        db.close_connection()


def test_insert_claims_monitoring_event_postgres_returning_id_reloads_inserted_row() -> None:
    loaded_row = {
        "id": 17,
        "user_id": "1",
        "event_type": "unsupported_ratio",
        "severity": "warning",
        "payload_json": '{"alert_id":9}',
        "created_at": "2026-03-22T00:00:01Z",
        "delivered_at": None,
    }
    execute_calls: list[tuple[str, tuple[object, ...], bool]] = []

    class _Cursor:
        def __init__(self, row: dict[str, object]) -> None:
            self._row = row

        def fetchone(self) -> dict[str, object]:
            return self._row

    class _FakePostgresDB:
        backend_type = BackendType.POSTGRESQL

        def _get_current_utc_timestamp_str(self) -> str:
            return "2026-03-22T00:00:01Z"

        def execute_query(
            self,
            sql: str,
            params: tuple[object, ...] | None = None,
            *,
            commit: bool = False,
        ) -> _Cursor:
            execute_calls.append((sql, tuple(params or ()), commit))
            if sql.startswith("INSERT INTO claims_monitoring_events"):
                return _Cursor({"id": 17})
            return _Cursor(loaded_row)

    created = helper_insert_claims_monitoring_event(
        _FakePostgresDB(),
        user_id="1",
        event_type="unsupported_ratio",
        severity="warning",
        payload_json='{"alert_id":9}',
    )

    assert created == loaded_row
    assert execute_calls[0][0].endswith(" RETURNING id")
    assert execute_calls[0][1] == (
        "1",
        "unsupported_ratio",
        "warning",
        '{"alert_id":9}',
        "2026-03-22T00:00:01Z",
        None,
    )
    assert execute_calls[0][2] is True
    assert execute_calls[1][0].startswith(
        "SELECT id, user_id, event_type, severity, payload_json, created_at, delivered_at "
    )
    assert execute_calls[1][1] == (17,)
    assert execute_calls[1][2] is False


def test_list_claims_monitoring_events_filters_and_preserves_created_at_order(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-list.db")
    timestamps = iter(
        [
            "2026-03-22T00:00:01Z",
            "2026-03-22T00:00:02Z",
            "2026-03-22T00:00:03Z",
        ]
    )
    db._get_current_utc_timestamp_str = lambda: next(timestamps)  # type: ignore[method-assign]
    try:
        assert db.list_claims_monitoring_events.__func__ is helper_list_claims_monitoring_events

        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"kind":"first"}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"kind":"second"}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="info",
            payload_json='{"kind":"third"}',
        )

        rows = db.list_claims_monitoring_events(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            start_time="2026-03-22T00:00:01Z",
            end_time="2026-03-22T00:00:02Z",
        )

        assert [row["payload_json"] for row in rows] == ['{"kind":"first"}', '{"kind":"second"}']
        assert [row["created_at"] for row in rows] == [
            "2026-03-22T00:00:01Z",
            "2026-03-22T00:00:02Z",
        ]
    finally:
        db.close_connection()


def test_list_undelivered_claims_monitoring_events_clamps_limit_and_filters_event_type(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-undelivered.db")
    timestamps = iter(
        [
            "2026-03-22T00:00:01Z",
            "2026-03-22T00:00:02Z",
            "2026-03-22T00:00:03Z",
            "2026-03-22T00:00:04Z",
        ]
    )
    db._get_current_utc_timestamp_str = lambda: next(timestamps)  # type: ignore[method-assign]
    try:
        assert (
            db.list_undelivered_claims_monitoring_events.__func__
            is helper_list_undelivered_claims_monitoring_events
        )
        assert (
            db.mark_claims_monitoring_events_delivered.__func__
            is helper_mark_claims_monitoring_events_delivered
        )

        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"id":1}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"id":2}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="info",
            payload_json='{"id":3}',
        )

        first = db.execute_query(
            "SELECT id FROM claims_monitoring_events WHERE payload_json = ?",
            ('{"id":1}',),
        ).fetchone()
        db.mark_claims_monitoring_events_delivered([int(first["id"])])

        rows = db.list_undelivered_claims_monitoring_events(
            user_id="1",
            event_type="unsupported_ratio",
            limit=0,
        )

        assert len(rows) == 1
        assert rows[0]["payload_json"] == '{"id":2}'
    finally:
        db.close_connection()


def test_mark_claims_monitoring_events_delivered_handles_empty_ids_and_returns_rowcount(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-mark.db")
    try:
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"id":1}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"id":2}',
        )
        rows = db.execute_query(
            "SELECT id FROM claims_monitoring_events ORDER BY id ASC"
        ).fetchall()

        assert db.mark_claims_monitoring_events_delivered([]) == 0
        assert db.mark_claims_monitoring_events_delivered([int(rows[0]["id"]), int(rows[1]["id"])]) == 2
    finally:
        db.close_connection()


def test_get_latest_claims_monitoring_event_delivery_returns_none_and_supports_tuple_row_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-latest.db")

    class _Cursor:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    try:
        assert (
            db.get_latest_claims_monitoring_event_delivery.__func__
            is helper_get_latest_claims_monitoring_event_delivery
        )

        monkeypatch.setattr(db, "execute_query", lambda *_args, **_kwargs: _Cursor(None))
        assert (
            db.get_latest_claims_monitoring_event_delivery(
                user_id="1",
                event_type="unsupported_ratio",
            )
            is None
        )

        monkeypatch.setattr(
            db,
            "execute_query",
            lambda *_args, **_kwargs: _Cursor(("2026-03-22T12:00:00Z",)),
        )
        assert db.get_latest_claims_monitoring_event_delivery(
            user_id="1",
            event_type="unsupported_ratio",
        ) == "2026-03-22T12:00:00Z"
    finally:
        db.close_connection()


def test_has_successful_claims_monitoring_event_delivery_checks_bounded_recent_events(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-dedupe.db")
    try:
        assert (
            db.has_successful_claims_monitoring_event_delivery.__func__
            is helper_has_successful_claims_monitoring_event_delivery
        )

        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="warning",
            payload_json="{not-json",
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="warning",
            payload_json='{"status":"failure","event_id":7,"alert_id":3,"channel":"webhook"}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="info",
            payload_json='{"status":"success","event_id":7,"alert_id":3,"channel":"webhook"}',
        )

        assert db.has_successful_claims_monitoring_event_delivery(
            user_id="1",
            event_id=7,
            alert_id=3,
            channel="webhook",
        ) is True
        assert db.has_successful_claims_monitoring_event_delivery(
            user_id="1",
            event_id=7,
            alert_id=3,
            channel="slack",
        ) is False
    finally:
        db.close_connection()


def test_list_claims_monitoring_events_page_uses_paired_keyset_without_gaps(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-page.db")
    timestamp = "2026-08-08T12:00:00.000Z"
    db._get_current_utc_timestamp_str = lambda: timestamp  # type: ignore[method-assign]
    try:
        wanted_ids = []
        for payload in ('{"index":1}', '{"index":2}', '{"index":3}', '{"index":4}'):
            row = db.insert_claims_monitoring_event(
                user_id="1",
                event_type="unsupported_ratio",
                severity="warning",
                payload_json=payload,
            )
            wanted_ids.append(int(row["id"]))
        db.insert_claims_monitoring_event(
            user_id="2",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"owner":2}',
        )
        db.insert_claims_monitoring_event(
            user_id="1",
            event_type="webhook_delivery",
            severity="info",
            payload_json='{"filtered":true}',
        )

        assert (
            db.list_claims_monitoring_events_page.__func__
            is claims_monitoring_event_ops.list_claims_monitoring_events_page
        )
        first = db.list_claims_monitoring_events_page(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            start_time=timestamp,
            end_time=timestamp,
            limit=2,
        )
        second = db.list_claims_monitoring_events_page(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            start_time=timestamp,
            end_time=timestamp,
            after_created_at=first[-1]["created_at"],
            after_id=int(first[-1]["id"]),
            limit=2,
        )

        rows = first + second
        assert [int(row["id"]) for row in rows] == wanted_ids
        assert [int(row["id"]) for row in rows] == sorted(int(row["id"]) for row in rows)
        assert all(row["user_id"] == "1" for row in rows)
        assert all(row["event_type"] == "unsupported_ratio" for row in rows)
        assert all(row["severity"] == "warning" for row in rows)
    finally:
        db.close_connection()


def test_list_claims_monitoring_events_page_requires_both_cursor_parts(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-page-cursor.db")
    try:
        with pytest.raises(ValueError, match="after_created_at and after_id"):
            db.list_claims_monitoring_events_page(
                user_id="1",
                after_created_at="2026-08-08T12:00:00.000Z",
            )
        with pytest.raises(ValueError, match="after_created_at and after_id"):
            db.list_claims_monitoring_events_page(user_id="1", after_id=1)
    finally:
        db.close_connection()


def test_list_claims_monitoring_events_page_clamps_limit_to_one_through_one_thousand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-page-limit.db")
    calls: list[tuple[object, ...]] = []

    class _Cursor:
        def fetchall(self) -> list[object]:
            return []

    def _capture_query(_sql: str, params: tuple[object, ...]) -> _Cursor:
        calls.append(params)
        return _Cursor()

    try:
        monkeypatch.setattr(db, "execute_query", _capture_query)

        assert db.list_claims_monitoring_events_page(user_id="1", limit=0) == []
        assert calls[-1][-1] == 1
        assert db.list_claims_monitoring_events_page(user_id="1", limit=1001) == []
        assert calls[-1][-1] == 1000
    finally:
        db.close_connection()
