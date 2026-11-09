from __future__ import annotations

import types
from unittest.mock import Mock, call

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.media_database_impl import MediaDatabase


def test_bootstrap_lifecycle_methods_rebind_to_runtime_wrapper() -> None:
    expected_module = (
        "tldw_Server_API.app.core.DB_Management.media_db.runtime.bootstrap_lifecycle_ops"
    )

    assert MediaDatabase.__dict__["__init__"].__globals__["__name__"] == expected_module
    assert MediaDatabase.__dict__["initialize_db"].__globals__["__name__"] == expected_module
    assert MediaDatabase.__dict__["_ensure_sqlite_backend"].__globals__["__name__"] == expected_module


def test_initialize_db_returns_self_after_reinitializing_schema(monkeypatch) -> None:
    db = MediaDatabase.__new__(MediaDatabase)
    calls: list[str] = []

    monkeypatch.setattr(db, "_initialize_schema", lambda: calls.append("initialize_schema"), raising=False)

    result = MediaDatabase.__dict__["initialize_db"](db)

    assert result is db
    assert calls == ["initialize_schema"]


def test_initialize_db_wraps_failures_in_database_error(monkeypatch) -> None:
    db = MediaDatabase.__new__(MediaDatabase)

    def _boom():
        raise RuntimeError("boom")

    monkeypatch.setattr(db, "_initialize_schema", _boom, raising=False)

    with pytest.raises(DatabaseError, match="Database initialization failed: boom"):
        MediaDatabase.__dict__["initialize_db"](db)


def test_ensure_sqlite_backend_is_harmless_for_sqlite_and_postgres() -> None:
    sqlite_db = MediaDatabase.__new__(MediaDatabase)
    sqlite_db.backend_type = BackendType.SQLITE

    postgres_db = MediaDatabase.__new__(MediaDatabase)
    postgres_db.backend_type = BackendType.POSTGRESQL

    assert MediaDatabase.__dict__["_ensure_sqlite_backend"](sqlite_db) is None
    assert MediaDatabase.__dict__["_ensure_sqlite_backend"](postgres_db) is None


def test_constructor_uses_explicit_backend_and_initializes_schema(monkeypatch) -> None:
    explicit_backend = types.SimpleNamespace(backend_type=BackendType.POSTGRESQL)
    calls: list[str] = []

    monkeypatch.setattr(MediaDatabase, "_initialize_schema", lambda self: calls.append("initialize_schema"))

    db = MediaDatabase(
        db_path="explicit-bootstrap.db",
        client_id="bootstrap-runtime",
        backend=explicit_backend,
        config=types.SimpleNamespace(),
    )

    assert db.backend is explicit_backend
    assert db.backend_type == BackendType.POSTGRESQL
    assert calls == ["initialize_schema"]


def test_memory_constructor_creates_persistent_sqlite_connection(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(MediaDatabase, "_initialize_schema", lambda self: None)
    monkeypatch.setattr(
        MediaDatabase,
        "_apply_sqlite_connection_pragmas",
        lambda self, conn: calls.append(conn),
    )

    db = MediaDatabase(db_path=":memory:", client_id="bootstrap-memory")
    try:
        assert db._persistent_conn is not None
        assert db._persistent_conn in calls
    finally:
        if db._persistent_conn is not None:
            db._persistent_conn.close()


def test_constructor_failure_calls_close_connection_before_raising(monkeypatch) -> None:
    close_calls: list[str] = []

    def _failing_initialize(self):
        raise DatabaseError("boom")

    def _record_close(self):
        close_calls.append("close_connection")

    monkeypatch.setattr(MediaDatabase, "_initialize_schema", _failing_initialize)
    monkeypatch.setattr(MediaDatabase, "close_connection", _record_close)

    with pytest.raises(DatabaseError, match="Database initialization failed"):
        MediaDatabase(db_path="bootstrap-boom.db", client_id="bootstrap-boom")

    assert close_calls == ["close_connection"]


@pytest.mark.unit
def test_ensure_postgres_post_core_structures_runs_followup_ensures(
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.DB_Management.media_db.schema.backends import (
        postgres_helpers as postgres_helpers_module,
    )

    from tldw_Server_API.app.core.DB_Management.media_db.schema.migration_bodies.postgres_claims_cluster_rebuild_publication import (
        POSTGRES_PUBLICATION_DDL,
    )

    conn = object()
    calls = Mock()
    backend = types.SimpleNamespace(execute=calls.safe_transcript_extractor)
    db = types.SimpleNamespace(
        backend=backend,
        _ensure_postgres_collections_tables=calls.collections,
        _ensure_postgres_tts_history=calls.tts_history,
        _ensure_postgres_audio_presets=calls.audio_presets,
        _ensure_postgres_data_tables=calls.data_tables,
        _ensure_postgres_source_hash_column=calls.source_hash,
        _ensure_postgres_claims_extensions=calls.claims_extensions,
        _ensure_postgres_email_schema=calls.email_schema,
        _ensure_postgres_rls=calls.policies,
        _sync_postgres_sequences=lambda value: pytest.fail(
            "routine bootstrap must not rewind RLS-scoped sequences"
        ),
    )

    monkeypatch.setattr(
        postgres_helpers_module,
        "ensure_postgres_document_workspace_schema",
        calls.document_workspace,
        raising=False,
    )

    monkeypatch.setattr(
        postgres_helpers_module,
        "ensure_postgres_sync_log_contract",
        calls.sync_contract,
    )
    monkeypatch.setattr(
        postgres_helpers_module,
        "ensure_postgres_keyword_identity",
        calls.keyword_identity,
    )
    monkeypatch.setattr(
        postgres_helpers_module,
        "ensure_postgres_media_owner_identity",
        calls.media_owner_identity,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.DB_Management.media_db.schema.postgres_runtime_validation.validate_postgres_annotation_replay_index",
        calls.annotation_replay,
    )
    postgres_helpers_module.ensure_postgres_post_core_structures(db, conn)

    assert (  # nosec B101 -- Preserve the real coordinator order and arguments.
        calls.mock_calls
        == [
            call.sync_contract(db, conn),
            call.media_owner_identity(db, conn),
            call.collections(conn),
            call.tts_history(conn),
            call.audio_presets(conn),
            call.data_tables(conn),
            call.safe_transcript_extractor(
                postgres_helpers_module._SAFE_TRANSCRIPT_TEXT_FUNCTION_SQL,
                connection=conn,
            ),
            call.document_workspace(conn),
            call.annotation_replay(db, conn),
            call.source_hash(conn),
            call.claims_extensions(conn),
            *(call.safe_transcript_extractor(statement, connection=conn) for statement in POSTGRES_PUBLICATION_DDL),
            call.email_schema(conn),
            call.policies(conn),
            call.keyword_identity(db, conn),
        ]
    )
