"""PostgreSQL schema-v67 contracts for semantic operation authority."""

from __future__ import annotations

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
