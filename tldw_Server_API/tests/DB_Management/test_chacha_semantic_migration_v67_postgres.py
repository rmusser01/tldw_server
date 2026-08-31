"""PostgreSQL schema-v67 contracts for semantic operation authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
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

        assert int(version) == 67
        assert relation == {"relrowsecurity": True, "relforcerowsecurity": True}
        assert {"owner_user_id", "dataset_id", "key_digest", "request_fingerprint"} <= columns
    finally:
        db.close_connection()
        backend.get_pool().close_all()
