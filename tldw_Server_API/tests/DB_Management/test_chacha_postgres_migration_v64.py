"""PostgreSQL schema-v64 parity contracts for Notes graph suggestions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

pytestmark = pytest.mark.unit


class _ReachedV64(Exception):
    pass


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


def _set_tenant_scope(
    backend: object,
    conn: object,
    owner_user_id: str,
    dataset_id: str,
) -> None:
    backend.execute(  # type: ignore[attr-defined]
        "SELECT set_config('app.current_user_id', %s, true)",
        (owner_user_id,),
        connection=conn,
    )
    backend.execute(  # type: ignore[attr-defined]
        "SELECT set_config('app.current_dataset_id', %s, true)",
        (dataset_id,),
        connection=conn,
    )


def _insert_note(
    backend: object,
    conn: object,
    note_id: str,
    owner_user_id: str,
) -> None:
    backend.execute(  # type: ignore[attr-defined]
        "INSERT INTO notes(id, title, content, client_id) VALUES (%s, %s, %s, %s)",
        (note_id, note_id, "content", owner_user_id),
        connection=conn,
    )


def test_postgres_initializer_routes_schema_v63_through_v64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 64)
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn, lock=False: 63)
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_notes_moodboard_studio_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )

    def _reached_v64(_conn: object) -> None:
        raise _ReachedV64

    monkeypatch.setattr(db, "_migrate_from_v63_to_v64_postgres", _reached_v64, raising=False)

    with pytest.raises(_ReachedV64):
        db._initialize_schema_postgres()


def test_postgres_v64_migration_versions_after_applying_ddl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    applied: list[tuple[str, int]] = []
    versions = iter((63, 64))
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn: next(versions))
    monkeypatch.setattr(
        db,
        "_apply_postgres_migration_script",
        lambda script, _conn, *, expected_version: applied.append((script, expected_version)),
    )
    monkeypatch.setattr(
        db,
        "_configure_note_graph_suggestion_receipt_delete_trigger_postgres",
        lambda _conn: None,
    )

    db._migrate_from_v63_to_v64_postgres(object())

    assert applied == [(CharactersRAGDB._MIGRATION_SQL_V63_TO_V64_POSTGRES, 64)]


def test_postgres_v64_ddl_has_owner_scoped_tables_checks_indexes_and_forced_rls() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V63_TO_V64_POSTGRES.split())

    for table in (
        "note_graph_suggestion_runs",
        "note_graph_suggestion_operation_receipts",
        "note_graph_suggestion_rejection_sets",
        "note_graph_suggestions",
        "note_graph_suggestion_evidence",
    ):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql
        assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY" in sql
        assert f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY" in sql
        assert f"CREATE POLICY {table}_tenant_isolation ON {table}" in sql
        assert "owner_user_id = current_setting('app.current_user_id', true)" in sql
        assert "dataset_id = current_setting('app.current_dataset_id', true)" in sql

    for clause in (
        "CHECK(state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing', 'succeeded', 'failed', 'cancelled', 'stale'))",
        "CHECK(operation_kind IN ('run_admit', 'run_cancel', 'suggestion_accept', 'suggestion_reject', 'rejections_reset'))",
        "CHECK(state IN ('staged', 'pending', 'accepting', 'accepted', 'rejected', 'stale'))",
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_note_graph_suggestion_runs_active_source",
        "WHERE state IN ('admitting', 'queued', 'running', 'cancelling', 'publishing')",
        "idx_note_graph_suggestions_acceptance_lease",
        "idx_note_graph_suggestion_operation_receipts_retention",
        "FOREIGN KEY(owner_user_id, source_note_id) REFERENCES notes(client_id, id)",
        "FOREIGN KEY(owner_user_id, dataset_id, admission_receipt_id)",
        "FOREIGN KEY(owner_user_id, dataset_id, run_id)",
        "FOREIGN KEY(owner_user_id, dataset_id, suggestion_id)",
        "ON DELETE CASCADE",
    ):
        assert clause in sql


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v64_live_schema_has_owner_scoped_graph_suggestion_contract(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    tables = (
        "note_graph_suggestion_operation_receipts",
        "note_graph_suggestion_runs",
        "note_graph_suggestion_rejection_sets",
        "note_graph_suggestions",
        "note_graph_suggestion_evidence",
    )
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        columns = backend.execute(
            """
            SELECT table_name, column_name, data_type
              FROM information_schema.columns
             WHERE table_schema = current_schema()
               AND table_name = ANY(%s)
               AND column_name IN (
                   'owner_user_id', 'dataset_id', 'state', 'source_note_id',
                   'expires_at', 'acceptance_lease_expires_at', 'keyword_sync_id'
               )
            """,
            (list(tables),),
        ).rows
        constraints = backend.execute(
            """
            SELECT relation.relname AS table_name, pg_get_constraintdef(constraint_row.oid) AS definition
              FROM pg_constraint AS constraint_row
              JOIN pg_class AS relation ON relation.oid = constraint_row.conrelid
              JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname = ANY(%s)
            """,
            (list(tables),),
        ).rows
        indexes = backend.execute(
            """
            SELECT indexname
              FROM pg_indexes
             WHERE schemaname = current_schema()
               AND tablename = ANY(%s)
            """,
            (list(tables),),
        ).rows
        relations = backend.execute(
            """
            SELECT relname, relrowsecurity, relforcerowsecurity
              FROM pg_class AS relation
              JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relname = ANY(%s)
            """,
            (list(tables),),
        ).rows
        policies = backend.execute(
            """
            SELECT tablename, qual, with_check
              FROM pg_policies
             WHERE schemaname = current_schema()
               AND tablename = ANY(%s)
               AND policyname = tablename || '_tenant_isolation'
            """,
            (list(tables),),
        ).rows

        assert int(version) == 64
        assert {(row["table_name"], row["column_name"]) for row in columns} >= {
            (table, "owner_user_id") for table in tables
        } | {(table, "dataset_id") for table in tables}
        assert ("note_graph_suggestions", "keyword_sync_id") in {
            (row["table_name"], row["column_name"]) for row in columns
        }
        assert all(
            row["data_type"] == "timestamp with time zone"
            for row in columns
            if row["column_name"].endswith("_at")
        )
        definitions = " ".join(str(row["definition"]) for row in constraints)
        assert "CHECK ((state = ANY" in definitions
        assert (
            "FOREIGN KEY (owner_user_id, source_note_id) REFERENCES notes(client_id, id) "
            "ON UPDATE CASCADE ON DELETE CASCADE"
        ) in definitions
        assert {
            "idx_note_graph_suggestion_runs_active_source",
            "idx_note_graph_suggestion_runs_retention",
            "idx_note_graph_suggestions_acceptance_lease",
            "idx_note_graph_suggestions_retention",
            "idx_note_graph_suggestion_operation_receipts_retention",
        } <= {str(row["indexname"]) for row in indexes}
        assert {str(row["relname"]) for row in relations} == set(tables)
        assert all(row["relrowsecurity"] is True for row in relations)
        assert all(row["relforcerowsecurity"] is True for row in relations)
        assert len(policies) == len(tables)
        assert all("owner_user_id" in str(row["qual"]) for row in policies)
        assert all("dataset_id" in str(row["with_check"]) for row in policies)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v64_rejects_cross_scope_source_note_reference(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
            _insert_note(backend, conn, "owner-a-note", "owner-a")
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn, "owner-b", "dataset-b")
            _insert_note(backend, conn, "owner-b-note", "owner-b")

        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
                backend.execute(
                    """
                    INSERT INTO note_graph_suggestion_runs(
                        id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                        state, revision, created_at, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        "cross-owner-run",
                        "owner-a",
                        "dataset-a",
                        "owner-b-note",
                        "fingerprint-owner-b",
                        "succeeded",
                        1,
                    ),
                    connection=conn,
                )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v64_note_hard_delete_cascades_receipt_graph(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
            _insert_note(backend, conn, "source-note", "owner-a")
            backend.execute(
                """
                INSERT INTO note_graph_suggestion_operation_receipts(
                    id, operation_kind, owner_user_id, dataset_id, source_note_id,
                    resource_identity, idempotency_key_digest, request_fingerprint,
                    state, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                """,
                (
                    "receipt-a",
                    "run_admit",
                    "owner-a",
                    "dataset-a",
                    "source-note",
                    "resource-a",
                    "key-a",
                    "request-a",
                    "completed",
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    admission_receipt_id, state, revision, created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "source-note",
                    "source-fingerprint",
                    "receipt-a",
                    "succeeded",
                    1,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_graph_suggestions(
                    id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                    source_fingerprint, normalized_tag, display_tag, state, revision,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "suggestion-a",
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "tag",
                    "source-note",
                    "source-fingerprint",
                    "research",
                    "Research",
                    "pending",
                    1,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_graph_suggestion_evidence(
                    suggestion_id, owner_user_id, dataset_id, side, ordinal, note_id,
                    field, content_fingerprint, start_offset, end_offset
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    "suggestion-a",
                    "owner-a",
                    "dataset-a",
                    "source",
                    0,
                    "source-note",
                    "content",
                    "source-fingerprint",
                    0,
                    1,
                ),
                connection=conn,
            )
            backend.execute(
                "DELETE FROM notes WHERE client_id = %s AND id = %s",
                ("owner-a", "source-note"),
                connection=conn,
            )
            for table in (
                "note_graph_suggestion_operation_receipts",
                "note_graph_suggestion_runs",
                "note_graph_suggestions",
                "note_graph_suggestion_evidence",
            ):
                assert backend.execute(  # nosec B608
                    f"SELECT COUNT(*) FROM {table}", connection=conn
                ).scalar == 0
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v64_receipt_delete_clears_only_scoped_receipt_references(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
            _insert_note(backend, conn, "source-note", "owner-a")
            for receipt_id in ("admission-receipt", "decision-receipt"):
                backend.execute(
                    """
                    INSERT INTO note_graph_suggestion_operation_receipts(
                        id, operation_kind, owner_user_id, dataset_id, source_note_id,
                        resource_identity, idempotency_key_digest, request_fingerprint,
                        state, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    """,
                    (
                        receipt_id,
                        "run_admit",
                        "owner-a",
                        "dataset-a",
                        "source-note",
                        f"resource-{receipt_id}",
                        f"key-{receipt_id}",
                        f"request-{receipt_id}",
                        "completed",
                    ),
                    connection=conn,
                )
            backend.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    admission_receipt_id, state, revision, created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "source-note",
                    "source-fingerprint",
                    "admission-receipt",
                    "succeeded",
                    1,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_graph_suggestions(
                    id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                    source_fingerprint, normalized_tag, display_tag, state, revision,
                    decision_receipt_id, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "suggestion-a",
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "tag",
                    "source-note",
                    "source-fingerprint",
                    "research",
                    "Research",
                    "rejected",
                    1,
                    "decision-receipt",
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_graph_suggestions(
                    id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                    source_fingerprint, normalized_tag, display_tag, state, revision,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "suggestion-pending",
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "tag",
                    "source-note",
                    "source-fingerprint",
                    "planning",
                    "Planning",
                    "pending",
                    1,
                ),
                connection=conn,
            )

            backend.execute(
                "DELETE FROM note_graph_suggestion_operation_receipts WHERE id = %s",
                ("admission-receipt",),
                connection=conn,
            )
            run = backend.execute(
                """
                SELECT owner_user_id, dataset_id, source_note_id, admission_receipt_id, state
                  FROM note_graph_suggestion_runs
                 WHERE id = %s
                """,
                ("run-a",),
                connection=conn,
            ).rows[0]
            assert tuple(run.values()) == (
                "owner-a",
                "dataset-a",
                "source-note",
                None,
                "succeeded",
            )
            assert backend.execute(
                "SELECT state FROM note_graph_suggestions WHERE id = %s",
                ("suggestion-a",),
                connection=conn,
            ).scalar == "rejected"
            assert backend.execute(
                "SELECT state FROM note_graph_suggestions WHERE id = %s",
                ("suggestion-pending",),
                connection=conn,
            ).scalar == "pending"

            backend.execute(
                "DELETE FROM note_graph_suggestion_operation_receipts WHERE id = %s",
                ("decision-receipt",),
                connection=conn,
            )
            suggestion = backend.execute(
                """
                SELECT owner_user_id, dataset_id, run_id, source_note_id, decision_receipt_id, state
                  FROM note_graph_suggestions
                 WHERE id = %s
                """,
                ("suggestion-a",),
                connection=conn,
            ).rows[0]
            assert tuple(suggestion.values()) == (
                "owner-a",
                "dataset-a",
                "run-a",
                "source-note",
                None,
                "rejected",
            )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v64_rejects_duplicate_tag_and_reverse_related_pair(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
            _insert_note(backend, conn, "alpha", "owner-a")
            _insert_note(backend, conn, "beta", "owner-a")
            for run_id, source_note_id in (("run-alpha", "alpha"), ("run-beta", "beta")):
                backend.execute(
                    """
                    INSERT INTO note_graph_suggestion_runs(
                        id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                        state, revision, created_at, expires_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        run_id,
                        "owner-a",
                        "dataset-a",
                        source_note_id,
                        f"fingerprint-{source_note_id}",
                        "succeeded",
                        1,
                    ),
                    connection=conn,
                )
            backend.execute(
                """
                INSERT INTO note_graph_suggestions(
                    id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                    source_fingerprint, target_note_id, target_fingerprint, state,
                    revision, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "related-alpha-beta",
                    "run-alpha",
                    "owner-a",
                    "dataset-a",
                    "related_note",
                    "alpha",
                    "fingerprint-alpha",
                    "beta",
                    "fingerprint-beta",
                    "pending",
                    1,
                ),
                connection=conn,
            )
        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
                backend.execute(
                    """
                    INSERT INTO note_graph_suggestions(
                        id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                        source_fingerprint, target_note_id, target_fingerprint, state,
                        revision, created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        "related-beta-alpha",
                        "run-beta",
                        "owner-a",
                        "dataset-a",
                        "related_note",
                        "beta",
                        "fingerprint-beta",
                        "alpha",
                        "fingerprint-alpha",
                        "pending",
                        1,
                    ),
                    connection=conn,
                )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v64_rejects_duplicate_tag_identity(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    try:
        with backend.transaction() as conn:
            _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
            _insert_note(backend, conn, "source-note", "owner-a")
            backend.execute(
                """
                INSERT INTO note_graph_suggestion_runs(
                    id, owner_user_id, dataset_id, source_note_id, source_fingerprint,
                    state, revision, created_at, expires_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "source-note",
                    "source-fingerprint",
                    "succeeded",
                    1,
                ),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO note_graph_suggestions(
                    id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                    source_fingerprint, normalized_tag, display_tag, state, revision,
                    created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """,
                (
                    "tag-one",
                    "run-a",
                    "owner-a",
                    "dataset-a",
                    "tag",
                    "source-note",
                    "source-fingerprint",
                    "research",
                    "Research",
                    "pending",
                    1,
                ),
                connection=conn,
            )

        with pytest.raises(BackendDatabaseError):
            with backend.transaction() as conn:
                _set_tenant_scope(backend, conn, "owner-a", "dataset-a")
                backend.execute(
                    """
                    INSERT INTO note_graph_suggestions(
                        id, run_id, owner_user_id, dataset_id, kind, source_note_id,
                        source_fingerprint, normalized_tag, display_tag, state, revision,
                        created_at, updated_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        "tag-two",
                        "run-a",
                        "owner-a",
                        "dataset-a",
                        "tag",
                        "source-note",
                        "source-fingerprint",
                        "research",
                        "Research",
                        "pending",
                        1,
                    ),
                    connection=conn,
                )
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v63_to_v64_upgrade_creates_graph_suggestion_schema(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="owner-a", backend=backend)
    tables = (
        "note_graph_suggestion_evidence",
        "note_graph_suggestions",
        "note_graph_suggestion_rejection_sets",
        "note_graph_suggestion_runs",
        "note_graph_suggestion_operation_receipts",
    )
    try:
        with backend.transaction() as conn:
            for table in tables:
                backend.execute(f"DROP TABLE {table}", connection=conn)  # nosec B608
            backend.execute(
                "UPDATE db_schema_version SET version = %s WHERE schema_name = %s",
                (63, CharactersRAGDB._SCHEMA_NAME),
                connection=conn,
            )

        db.close_connection()
        db._initialize_schema_postgres()

        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        migrated_tables = {
            str(row["tablename"])
            for row in backend.execute(
                """
                SELECT tablename
                  FROM pg_tables
                 WHERE schemaname = current_schema()
                   AND tablename = ANY(%s)
                """,
                (list(reversed(tables)),),
            ).rows
        }

        assert int(version) == 64
        assert migrated_tables == set(tables)
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
