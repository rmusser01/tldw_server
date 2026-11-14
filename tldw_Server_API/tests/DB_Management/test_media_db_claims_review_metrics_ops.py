from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import (
    MediaDatabase,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_metrics_ops import (
    get_claims_review_extractor_metrics_daily as helper_get_claims_review_extractor_metrics_daily,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_metrics_ops import (
    list_claims_review_extractor_metrics_daily as helper_list_claims_review_extractor_metrics_daily,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_metrics_ops import (
    list_claims_review_user_ids as helper_list_claims_review_user_ids,
)
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_review_metrics_ops import (
    upsert_claims_review_extractor_metrics_daily as helper_upsert_claims_review_extractor_metrics_daily,
)

pytestmark = pytest.mark.unit


def _make_db(tmp_path: Path, name: str) -> MediaDatabase:
    db = MediaDatabase(db_path=str(tmp_path / name), client_id="claims-review-metrics-helper")
    db.initialize_db()
    return db


def test_get_claims_review_extractor_metrics_daily_normalizes_none_version_and_missing_row(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-review-metrics-missing.db")
    try:
        assert db.get_claims_review_extractor_metrics_daily.__func__ is helper_get_claims_review_extractor_metrics_daily
        assert (
            db.get_claims_review_extractor_metrics_daily(
                user_id="1",
                report_date="2024-01-10",
                extractor="heuristic",
                extractor_version=None,
            )
            == {}
        )
    finally:
        db.close_connection()


def test_upsert_claims_review_extractor_metrics_daily_inserts_updates_and_normalizes_version(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-review-metrics-upsert.db")
    try:
        assert (
            db.upsert_claims_review_extractor_metrics_daily.__func__
            is helper_upsert_claims_review_extractor_metrics_daily
        )

        created = db.upsert_claims_review_extractor_metrics_daily(
            user_id="1",
            report_date="2024-01-10",
            extractor="heuristic",
            extractor_version=None,
            total_reviewed=10,
            approved_count=7,
            rejected_count=2,
            flagged_count=1,
            reassigned_count=0,
            edited_count=1,
            reason_code_counts_json='{"spam": 2}',
        )
        updated = db.upsert_claims_review_extractor_metrics_daily(
            user_id="1",
            report_date="2024-01-10",
            extractor="heuristic",
            extractor_version=None,
            total_reviewed=12,
            approved_count=8,
            rejected_count=2,
            flagged_count=1,
            reassigned_count=1,
            edited_count=2,
            reason_code_counts_json='{"spam": 3}',
        )

        assert created["extractor_version"] == ""
        assert updated["extractor_version"] == ""
        assert int(updated["id"]) == int(created["id"])
        assert int(updated["total_reviewed"]) == 12
        assert int(updated["edited_count"]) == 2
        assert updated["reason_code_counts_json"] == '{"spam": 3}'
    finally:
        db.close_connection()


def test_list_claims_review_extractor_metrics_daily_clamps_limit_and_offset(
    tmp_path: Path,
) -> None:
    db = _make_db(tmp_path, "claims-review-metrics-list.db")
    try:
        assert (
            db.list_claims_review_extractor_metrics_daily.__func__ is helper_list_claims_review_extractor_metrics_daily
        )

        db.upsert_claims_review_extractor_metrics_daily(
            user_id="1",
            report_date="2024-01-10",
            extractor="heuristic",
            extractor_version="v1",
            total_reviewed=1,
        )
        db.upsert_claims_review_extractor_metrics_daily(
            user_id="1",
            report_date="2024-01-11",
            extractor="heuristic",
            extractor_version="v1",
            total_reviewed=2,
        )

        rows = db.list_claims_review_extractor_metrics_daily(
            user_id="1",
            limit=0,
            offset=-5,
        )

        assert len(rows) == 1
        assert rows[0]["report_date"] == "2024-01-11"
    finally:
        db.close_connection()


def test_list_claims_review_user_ids_returns_empty_for_non_postgres(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-review-user-ids-sqlite.db")
    try:
        assert db.list_claims_review_user_ids.__func__ is helper_list_claims_review_user_ids
        assert db.list_claims_review_user_ids() == []
    finally:
        db.close_connection()


def test_upsert_joins_outer_transaction_and_rolls_back_all_rows(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-review-metrics-rollback.db")
    try:
        with pytest.raises(RuntimeError, match="abort window"):
            with db.transaction():
                for extractor in ("alpha", "beta"):
                    db.upsert_claims_review_extractor_metrics_daily(
                        user_id="42",
                        report_date="2026-09-06",
                        extractor=extractor,
                        total_reviewed=3,
                    )
                raise RuntimeError("abort window")
        assert db.list_claims_review_extractor_metrics_daily(user_id="42") == []
    finally:
        db.close_connection()


def test_upsert_update_preserves_original_created_at(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-review-metrics-created.db")
    try:
        db.upsert_claims_review_extractor_metrics_daily(
            user_id="42",
            report_date="2026-09-06",
            extractor="alpha",
            total_reviewed=1,
        )
        db.execute_query(
            "UPDATE claims_review_extractor_metrics_daily SET created_at = ?",
            ("2020-01-01 00:00:00",),
            commit=True,
        )
        updated = db.upsert_claims_review_extractor_metrics_daily(
            user_id="42",
            report_date="2026-09-06",
            extractor="alpha",
            total_reviewed=3,
        )
        assert updated["created_at"] == "2020-01-01 00:00:00"
        assert updated["total_reviewed"] == 3
    finally:
        db.close_connection()


def test_concurrent_sqlite_initial_upserts_converge_on_one_row(tmp_path: Path) -> None:
    path = tmp_path / "claims-review-metrics-concurrent.db"
    first = _make_db(tmp_path, path.name)
    second = MediaDatabase(db_path=str(path), client_id="42")
    barrier = threading.Barrier(2)

    def write(db):
        barrier.wait(timeout=5)
        return db.upsert_claims_review_extractor_metrics_daily(
            user_id="42",
            report_date="2026-09-06",
            extractor="heuristic",
            total_reviewed=3,
        )

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(write, db) for db in (first, second)]
            rows = [future.result(timeout=10) for future in futures]
        assert rows[0]["id"] == rows[1]["id"]
        assert first.count_claims_review_extractor_metrics_daily(user_id="42") == 1
    finally:
        first.close_connection()
        second.close_connection()


def test_upsert_is_one_atomic_statement_and_review_user_ids_preserve_tuple_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Cursor:
        def __init__(self, *, one=None, all_rows=None):
            self._one = one
            self._all = all_rows or []

        def fetchone(self):
            return self._one

        def fetchall(self):
            return self._all

    execute_calls: list[tuple[str, tuple[object, ...] | None, bool]] = []

    connection = object()

    def _execute_query(sql, params=None, commit=False, **kwargs):
        execute_calls.append((sql, params, commit))
        if sql.startswith("INSERT INTO claims_review_extractor_metrics_daily"):
            assert kwargs["connection"] is connection
            return _Cursor(
                one={
                    "id": 9,
                    "user_id": "1",
                    "report_date": "2024-01-10",
                    "extractor": "heuristic",
                    "extractor_version": "",
                    "total_reviewed": 12,
                    "approved_count": 8,
                    "rejected_count": 2,
                    "flagged_count": 1,
                    "reassigned_count": 1,
                    "edited_count": 2,
                    "reason_code_counts_json": '{"spam": 3}',
                    "created_at": "2026-03-22T00:00:00Z",
                    "updated_at": "2026-03-22T00:00:01Z",
                }
            )
        if sql.startswith("SELECT DISTINCT COALESCE(CAST(m.owner_user_id AS TEXT), m.client_id) AS user_id"):
            return _Cursor(all_rows=[("1",), (None,), ("",), ("2",)])
        raise AssertionError(f"Unexpected SQL: {sql}")

    fake_db = SimpleNamespace(
        _get_current_utc_timestamp_str=lambda: "2026-03-22T00:00:01Z",
        execute_query=_execute_query,
        transaction=lambda: nullcontext(connection),
        backend_type=BackendType.POSTGRESQL,
    )
    monkeypatch.setattr(
        fake_db,
        "get_claims_review_extractor_metrics_daily",
        lambda **kwargs: helper_get_claims_review_extractor_metrics_daily(fake_db, **kwargs),
        raising=False,
    )

    updated = helper_upsert_claims_review_extractor_metrics_daily(
        fake_db,
        user_id="1",
        report_date="2024-01-10",
        extractor="heuristic",
        extractor_version=None,
        total_reviewed=12,
        approved_count=8,
        rejected_count=2,
        flagged_count=1,
        reassigned_count=1,
        edited_count=2,
        reason_code_counts_json='{"spam": 3}',
    )
    user_ids = helper_list_claims_review_user_ids(fake_db)

    assert int(updated["id"]) == 9
    assert updated["extractor_version"] == ""
    writes = [call for call in execute_calls if call[0].startswith("INSERT")]
    assert len(writes) == 1
    assert "ON CONFLICT" in writes[0][0]
    assert writes[0][2] is False
    assert user_ids == ["1", "2"]


def test_postgres_source_statement_uses_utc_bounds_and_explicit_owner():
    calls = []

    def execute(query, params=None, **kwargs):
        calls.append((query, params))
        return SimpleNamespace(fetchall=lambda: [])

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.execute_query = execute
    assert (
        db.get_claims_review_metrics_window_rows(
            owner_user_id="42",
            start_date=date(2026, 9, 6),
            end_date=date(2026, 9, 7),
        )
        == []
    )
    assert len(calls) == 1
    query, params = calls[0]
    assert "AT TIME ZONE 'UTC'" in query
    assert "UNION ALL" in query
    assert params == (
        datetime(2026, 9, 6, tzinfo=timezone.utc),
        datetime(2026, 9, 8, tzinfo=timezone.utc),
        "42",
    )


@pytest.mark.parametrize("limit,expected", [(1000, 100), (0, 1), ("invalid", 100), (5, 5)])
def test_postgres_owner_page_is_date_filtered_keyset_and_bounded(limit, expected):
    calls = []

    def execute(query, params=None, **kwargs):
        calls.append((query, params))
        return SimpleNamespace(fetchall=lambda: [{"user_id": "43"}])

    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db.execute_query = execute
    assert db.list_claims_review_user_ids_page(
        start_date=date(2026, 9, 6),
        end_date=date(2026, 9, 7),
        after_user_id="42",
        limit=limit,
    ) == ["43"]
    query, params = calls[0]
    assert " > ?" in query
    assert "ORDER BY user_id" in query
    assert params == (
        datetime(2026, 9, 6, tzinfo=timezone.utc),
        datetime(2026, 9, 8, tzinfo=timezone.utc),
        "42",
        expected,
    )


def test_postgres_owner_lock_uses_active_transaction():
    calls = []
    connection = object()
    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db._get_txn_conn = lambda: connection
    db.execute_query = lambda query, params, **kwargs: calls.append((query, params, kwargs))
    db.lock_claims_review_metrics_owner(owner_user_id="42")
    assert calls == [
        (
            "SELECT pg_advisory_xact_lock(hashtextextended(?, 0))",
            ("claims-review-metrics:42",),
            {"connection": connection},
        )
    ]


def test_postgres_owner_lock_rejects_calls_without_transaction():
    db = MediaDatabase.__new__(MediaDatabase)
    db.backend_type = BackendType.POSTGRESQL
    db._get_txn_conn = lambda: None
    with pytest.raises(RuntimeError, match="transaction"):
        db.lock_claims_review_metrics_owner(owner_user_id="42")
