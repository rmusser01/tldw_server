"""PostgreSQL schema-v61 contracts for recipient-owned shared chat state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends import pg_rls_policies
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _ReachedV61(Exception):
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


def test_postgres_initializer_routes_schema_v60_through_v61(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    db._backend = _FakeBackend()
    db._uses_shared_content_backend = False
    db._backend_refresh_suspended = False
    db._local = SimpleNamespace()

    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn, lock=False: 60)
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)

    def _reached_v61(_conn: object) -> None:
        raise _ReachedV61

    monkeypatch.setattr(db, "_migrate_from_v60_to_v61_postgres", _reached_v61, raising=False)

    with pytest.raises(_ReachedV61):
        db._initialize_schema_postgres()


def test_postgres_v61_migration_uses_only_reviewed_policy_block_and_versions_last(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = CharactersRAGDB.__new__(CharactersRAGDB)
    applied: list[tuple[str, int]] = []
    versions = iter((60, 61))
    monkeypatch.setattr(db, "_get_schema_version_postgres", lambda _conn: next(versions))
    monkeypatch.setattr(
        db,
        "_apply_postgres_migration_script",
        lambda script, _conn, *, expected_version: applied.append((script, expected_version)),
    )

    db._migrate_from_v60_to_v61_postgres(object())

    assert len(applied) == 1
    script, expected_version = applied[0]
    assert expected_version == 61
    for policy_block in pg_rls_policies.build_shared_workspace_chat_rls_sql():
        assert policy_block in script
    assert "notes_tenant_isolation" not in script
    assert script.index("CREATE TABLE IF NOT EXISTS shared_workspace_chat_threads") < script.index(
        "CREATE POLICY shared_workspace_chat_threads_tenant_isolation"
    )
    assert script.index("relforcerowsecurity") > script.index(
        "CREATE POLICY shared_workspace_chat_requests_tenant_isolation"
    )
    normalized_script = " ".join(script.split())
    assert (
        "( 'shared_workspace_chat_threads', "
        "'shared_workspace_chat_threads_tenant_isolation' )"
    ) in normalized_script
    assert (
        "( 'shared_workspace_chat_requests', "
        "'shared_workspace_chat_requests_tenant_isolation' )"
    ) in normalized_script


def test_postgres_v61_ddl_matches_sqlite_constraint_and_index_contract() -> None:
    sql = " ".join(CharactersRAGDB._MIGRATION_SQL_V60_TO_V61_POSTGRES.split())

    for clause in (
        "recipient_user_id TEXT NOT NULL CHECK(char_length(btrim(recipient_user_id)) > 0)",
        "share_id BIGINT NOT NULL CHECK(share_id > 0)",
        "conversation_id TEXT NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE",
        "owner_user_id TEXT NOT NULL CHECK(char_length(btrim(owner_user_id)) > 0)",
        "status TEXT NOT NULL CHECK(status IN ('in_progress','retryable','completed','conflicted'))",
        "lease_epoch INTEGER NOT NULL DEFAULT 1 CHECK(lease_epoch >= 1)",
        "source_mode TEXT CHECK(source_mode IN ('all','include'))",
        "user_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL",
        "assistant_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL",
        "PRIMARY KEY (recipient_user_id, share_id, request_id)",
        "FOREIGN KEY (recipient_user_id, share_id, conversation_id) REFERENCES shared_workspace_chat_threads(recipient_user_id, share_id, conversation_id) ON DELETE CASCADE",
        "idx_shared_workspace_chat_threads_conversation ON shared_workspace_chat_threads(conversation_id)",
        "idx_shared_workspace_chat_requests_status_lease ON shared_workspace_chat_requests(status, lease_expires_at)",
        "idx_shared_workspace_chat_requests_status_updated ON shared_workspace_chat_requests(status, updated_at)",
        "idx_shared_workspace_chat_requests_share_updated ON shared_workspace_chat_requests(share_id, updated_at)",
    ):
        assert clause in sql
    assert sql.count("TIMESTAMPTZ") == 6
    assert sql.count("PRIMARY KEY (recipient_user_id, share_id)") == 1
    assert sql.count("UNIQUE (recipient_user_id, share_id, conversation_id)") == 1


def _policy_catalog(backend: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    relations = list(
        backend.execute(
            """
            SELECT relation.relname AS table_name,
                   relation.relrowsecurity,
                   relation.relforcerowsecurity
              FROM pg_class AS relation
              JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
             WHERE namespace.nspname = current_schema()
               AND relation.relname IN (
                   'shared_workspace_chat_threads',
                   'shared_workspace_chat_requests'
               )
             ORDER BY relation.relname
            """
        )
    )
    policies = list(
        backend.execute(
            """
            SELECT tablename AS table_name, policyname, qual, with_check
              FROM pg_policies
             WHERE schemaname = current_schema()
               AND policyname IN (
                   'shared_workspace_chat_threads_tenant_isolation',
                   'shared_workspace_chat_requests_tenant_isolation'
               )
             ORDER BY tablename
            """
        )
    )
    return relations, policies


@pytest.mark.integration
@pytest.mark.timeout(30)
def test_postgres_v61_fresh_upgrade_constraints_forced_rls_and_rerun(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="recipient-a", backend=backend)
    try:
        version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        columns = list(
            backend.execute(
                """
                SELECT table_name, column_name, data_type
                  FROM information_schema.columns
                 WHERE table_schema = current_schema()
                   AND table_name IN (
                       'shared_workspace_chat_threads',
                       'shared_workspace_chat_requests'
                   )
                   AND column_name IN (
                       'recipient_user_id', 'owner_user_id', 'share_id',
                       'created_at', 'updated_at', 'lease_expires_at', 'completed_at'
                   )
                 ORDER BY table_name, column_name
                """
            )
        )
        relations, policies = _policy_catalog(backend)

        assert int(version) == 61
        types = {(row["table_name"], row["column_name"]): row["data_type"] for row in columns}
        assert types[("shared_workspace_chat_threads", "recipient_user_id")] == "text"
        assert types[("shared_workspace_chat_threads", "owner_user_id")] == "text"
        assert types[("shared_workspace_chat_threads", "share_id")] == "bigint"
        assert types[("shared_workspace_chat_requests", "share_id")] == "bigint"
        for key, data_type in types.items():
            if key[1].endswith("_at"):
                assert data_type == "timestamp with time zone"
        assert len(relations) == 2
        assert all(row["relrowsecurity"] is True for row in relations)
        assert all(row["relforcerowsecurity"] is True for row in relations)
        assert len(policies) == 2
        assert all(row["qual"] for row in policies)
        assert all(row["with_check"] for row in policies)

        conversation_id = db.add_conversation({"id": "conversation-a", "title": "Shared"})
        assert conversation_id == "conversation-a"
        with db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO shared_workspace_chat_threads(
                    recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("recipient-a", 1, conversation_id, "owner-a", "workspace-a"),
            )
            conn.execute(
                """
                INSERT INTO shared_workspace_chat_requests(
                    recipient_user_id, share_id, request_id, request_fingerprint,
                    conversation_id, status, source_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("recipient-a", 1, "request-a", "fingerprint-a", conversation_id, "completed", "include"),
            )

        with pytest.raises(BackendDatabaseError):
            with db.transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO shared_workspace_chat_requests(
                        recipient_user_id, share_id, request_id, request_fingerprint,
                        conversation_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("recipient-a", 2, "wrong-share", "fingerprint-b", conversation_id, "completed"),
                )

        with backend.transaction() as conn:
            backend.execute("DROP TABLE shared_workspace_chat_requests", connection=conn)
            backend.execute("DROP TABLE shared_workspace_chat_threads", connection=conn)
            backend.execute(
                "UPDATE db_schema_version SET version = %s WHERE schema_name = %s",
                (60, CharactersRAGDB._SCHEMA_NAME),
                connection=conn,
            )

        db.close_connection()
        db._initialize_schema_postgres()
        db._initialize_schema_postgres()

        rerun_version = backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar
        rerun_relations, rerun_policies = _policy_catalog(backend)
        assert int(rerun_version) == 61
        assert len(rerun_relations) == 2
        assert all(row["relrowsecurity"] is True for row in rerun_relations)
        assert all(row["relforcerowsecurity"] is True for row in rerun_relations)
        assert len(rerun_policies) == 2
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
