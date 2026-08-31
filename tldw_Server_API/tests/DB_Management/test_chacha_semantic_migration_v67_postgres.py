"""PostgreSQL schema-v67 contracts for semantic operation authority."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticIndexingError,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = [pytest.mark.integration, pytest.mark.timeout(30)]

_RECEIPTS = "note_semantic_operation_receipts"


class _FakeTransaction:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    def table_exists(self, _name: str, connection: object = None) -> bool:
        return True


def test_postgres_v67_ddl_has_model_authority_and_forced_owner_dataset_rls() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V66_TO_V67_POSTGRES.split())

    assert "ADD COLUMN IF NOT EXISTS model_revision" in sql
    assert f"CREATE TABLE IF NOT EXISTS {_RECEIPTS}" in sql
    assert f"ALTER TABLE {_RECEIPTS} ENABLE ROW LEVEL SECURITY" in sql
    assert f"ALTER TABLE {_RECEIPTS} FORCE ROW LEVEL SECURITY" in sql
    assert f"CREATE POLICY {_RECEIPTS}_tenant_isolation" in sql
    assert "idx_note_semantic_operation_receipts_scope" in sql
    assert "idx_note_semantic_operation_receipts_expiry" in sql


def test_postgres_initializer_routes_schema_v66_through_v67(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()
    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 67)
    monkeypatch.setattr(
        db,
        "_get_schema_version_postgres",
        lambda _conn, lock=False: 66,
    )
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_notes_moodboard_studio_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_graph_suggestion_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_ensure_note_semantic_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )

    def reached(_conn: object) -> None:
        raise RuntimeError("reached-v67")

    monkeypatch.setattr(db, "_migrate_from_v66_to_v67_postgres", reached, raising=False)

    with pytest.raises(RuntimeError, match="^reached-v67$"):
        db._initialize_schema_postgres()


def test_postgres_v67_live_receipts_are_forced_rls(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name=%s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        relation = backend.execute(
            "SELECT relrowsecurity,relforcerowsecurity FROM pg_class WHERE oid=to_regclass(%s)",
            (_RECEIPTS,),
        ).rows[0]
        columns = {
            str(row["column_name"])
            for row in backend.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema=current_schema() AND table_name=%s",
                (_RECEIPTS,),
            ).rows
        }
        indexes = {
            str(row["indexname"])
            for row in backend.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname=current_schema() "
                "AND tablename=%s",
                (_RECEIPTS,),
            ).rows
        }

        assert int(version) == 67
        assert relation == {"relrowsecurity": True, "relforcerowsecurity": True}
        assert {"owner_user_id", "dataset_id", "key_digest", "request_fingerprint"} <= columns
        assert "idx_note_semantic_operation_receipts_expiry" in indexes
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_postgres_v67_live_receipt_expiry_allows_reuse_and_fences_completion(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    now = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    try:
        store = db.note_semantic_store
        store.begin_operation_receipt(
            dataset_id="dataset-a",
            key_digest="a" * 64,
            action="enable",
            request_fingerprint="b" * 64,
            run_id=None,
            expected_revision=0,
            expires_at=now + timedelta(seconds=1),
            now=now,
        )

        replacement, replayed = store.begin_operation_receipt(
            dataset_id="dataset-a",
            key_digest="a" * 64,
            action="enable",
            request_fingerprint="c" * 64,
            run_id=None,
            expected_revision=0,
            expires_at=now + timedelta(days=1),
            now=now + timedelta(seconds=2),
        )
        assert replayed is False
        assert replacement.request_fingerprint == "c" * 64

        store.begin_operation_receipt(
            dataset_id="dataset-a",
            key_digest="d" * 64,
            action="cancel",
            request_fingerprint="e" * 64,
            run_id="run-a",
            expected_revision=2,
            expires_at=now + timedelta(seconds=1),
            now=now,
        )
        with pytest.raises(SemanticIndexingError) as exc_info:
            store.complete_operation_receipt(
                dataset_id="dataset-a",
                key_digest="d" * 64,
                request_fingerprint="e" * 64,
                run_id="run-a",
                response={"status": "cancelled"},
                now=now + timedelta(seconds=2),
            )
        assert exc_info.value.code == "notes_semantic_operation_receipt_conflict"
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("with_expired_predecessor", [False, True])
def test_postgres_v67_serializes_same_fingerprint_admission_across_connections(
    pg_database_config: DatabaseConfig,
    with_expired_predecessor: bool,
) -> None:
    suffix = "expired" if with_expired_predecessor else "fresh"
    owner = f"owner-receipt-race-{suffix}"
    dataset = f"dataset-receipt-race-{suffix}"
    fingerprint = "f" * 64
    first_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    second_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    observer_backend = DatabaseBackendFactory.create_backend(pg_database_config)
    first_db = CharactersRAGDB(":memory:", client_id=owner, backend=first_backend)
    second_db = CharactersRAGDB(":memory:", client_id=owner, backend=second_backend)
    first_admitted = threading.Event()
    release_first_commit = threading.Event()
    second_connection_ready = threading.Event()
    second_finished = threading.Event()
    results: dict[str, object] = {}
    workers: list[threading.Thread] = []

    if with_expired_predecessor:
        first_db.note_semantic_store.begin_operation_receipt(
            dataset_id=dataset,
            key_digest="0" * 64,
            action="enable",
            request_fingerprint=fingerprint,
            run_id=None,
            expected_revision=0,
            expires_at=datetime(2026, 8, 30, 12, 0, 1, tzinfo=timezone.utc),
            now=datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc),
        )

    admission_now = datetime(2026, 8, 30, 12, 0, 2, tzinfo=timezone.utc)

    def admit_first() -> None:
        try:
            with first_db.transaction() as conn:
                results["first_pid"] = int(
                    conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
                )
                results["first"] = first_db.note_semantic_store.begin_operation_receipt(
                    dataset_id=dataset,
                    key_digest="a" * 64,
                    action="enable",
                    request_fingerprint=fingerprint,
                    run_id=None,
                    expected_revision=0,
                    expires_at=admission_now + timedelta(days=1),
                    now=admission_now,
                )
                first_admitted.set()
                if not release_first_commit.wait(timeout=10):
                    raise AssertionError("first receipt commit was not released")
        except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
            results["first_error"] = exc
            first_admitted.set()
        finally:
            first_db.close_connection()

    def admit_second() -> None:
        try:
            with second_db.transaction() as conn:
                results["second_pid"] = int(
                    conn.execute("SELECT pg_backend_pid() AS pid").fetchone()["pid"]
                )
                second_connection_ready.set()
                results["second"] = second_db.note_semantic_store.begin_operation_receipt(
                    dataset_id=dataset,
                    key_digest="b" * 64,
                    action="enable",
                    request_fingerprint=fingerprint,
                    run_id=None,
                    expected_revision=0,
                    expires_at=admission_now + timedelta(days=1),
                    now=admission_now,
                )
        except BaseException as exc:  # noqa: BLE001 - surfaced on the test thread
            results["second_error"] = exc
            second_connection_ready.set()
        finally:
            second_finished.set()
            second_db.close_connection()

    try:
        first = threading.Thread(target=admit_first, name=f"receipt-first-{suffix}")
        second = threading.Thread(target=admit_second, name=f"receipt-second-{suffix}")
        workers = [first, second]
        first.start()
        assert first_admitted.wait(timeout=10)
        assert "first_error" not in results
        second.start()
        assert second_connection_ready.wait(timeout=10)
        assert "second_error" not in results

        first_pid = int(results["first_pid"])
        second_pid = int(results["second_pid"])
        deadline = time.monotonic() + 10
        lock_wait = None
        last_activity = None
        while time.monotonic() < deadline:
            rows = observer_backend.execute(
                "SELECT state,wait_event_type,wait_event,pg_blocking_pids(pid) AS blocking_pids "
                "FROM pg_stat_activity WHERE datname=current_database() AND pid=%s",
                (second_pid,),
            ).rows
            if rows:
                last_activity = dict(rows[0])
                blocking_pids = {
                    int(pid) for pid in (last_activity["blocking_pids"] or [])
                }
                if (
                    last_activity["state"] == "active"
                    and last_activity["wait_event_type"] == "Lock"
                    and last_activity["wait_event"] == "advisory"
                    and first_pid in blocking_pids
                ):
                    lock_wait = last_activity
                    break
            if second_finished.wait(timeout=0.01):
                break
        assert lock_wait is not None, (
            "second receipt admission never waited on the first dataset mutation lock; "
            f"finished={second_finished.is_set()}, activity={last_activity!r}, "
            f"results={results!r}"
        )

        release_first_commit.set()
        for worker in workers:
            worker.join(timeout=10)
            assert not worker.is_alive()

        assert "first_error" not in results
        assert "first" in results
        assert "second" not in results
        assert isinstance(results.get("second_error"), SemanticIndexingError)
        assert results["second_error"].code == "notes_semantic_idempotency_conflict"
        with first_db.transaction() as conn:
            first_db.note_semantic_store._set_scope(conn, dataset)
            active = conn.execute(
                "SELECT key_digest FROM note_semantic_operation_receipts "
                "WHERE owner_user_id=? AND dataset_id=? AND action='enable' "
                "AND request_fingerprint=? AND expires_at>?",
                (owner, dataset, fingerprint, admission_now),
            ).fetchall()
        assert [str(row["key_digest"]) for row in active] == ["a" * 64]
    finally:
        release_first_commit.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=10)
        assert all(not worker.is_alive() for worker in workers)
        first_db.close_all_connections()
        second_db.close_all_connections()
        first_backend.get_pool().close_all()
        second_backend.get_pool().close_all()
        observer_backend.get_pool().close_all()
