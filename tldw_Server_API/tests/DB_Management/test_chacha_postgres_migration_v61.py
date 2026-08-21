"""PostgreSQL schema-v61 contracts for recipient-owned shared chat state."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

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


def _insert_conversation(
    backend: Any,
    conn: Any,
    conversation_id: str,
    recipient_user_id: str,
    *,
    deleted: bool = False,
) -> None:
    backend.execute(
        """
        INSERT INTO conversations(id, root_id, client_id, deleted)
        VALUES (?, ?, ?, ?)
        """,
        (conversation_id, conversation_id, recipient_user_id, deleted),
        connection=conn,
    )


def _insert_message(
    backend: Any,
    conn: Any,
    message_id: str,
    conversation_id: str,
    recipient_user_id: str,
) -> None:
    backend.execute(
        """
        INSERT INTO messages(id, conversation_id, sender, content, client_id)
        VALUES (?, ?, 'user', 'message', ?)
        """,
        (message_id, conversation_id, recipient_user_id),
        connection=conn,
    )


def _insert_thread(
    backend: Any,
    conn: Any,
    *,
    recipient_user_id: str,
    share_id: int,
    conversation_id: str,
    owner_user_id: str = "owner-a",
    workspace_id: str = "workspace-a",
) -> None:
    backend.execute(
        """
        INSERT INTO shared_workspace_chat_threads(
            recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
        ) VALUES (?, ?, ?, ?, ?)
        """,
        (recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id),
        connection=conn,
    )


def _insert_request(
    backend: Any,
    conn: Any,
    *,
    recipient_user_id: str,
    share_id: int,
    request_id: str,
    conversation_id: str,
    status: str = "in_progress",
    lease_epoch: int = 1,
    source_mode: str | None = "all",
    user_message_id: str | None = None,
    assistant_message_id: str | None = None,
) -> None:
    backend.execute(
        """
        INSERT INTO shared_workspace_chat_requests(
            recipient_user_id, share_id, request_id, request_fingerprint,
            conversation_id, status, lease_epoch, source_mode,
            user_message_id, assistant_message_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            recipient_user_id,
            share_id,
            request_id,
            f"fingerprint-{request_id}",
            conversation_id,
            status,
            lease_epoch,
            source_mode,
            user_message_id,
            assistant_message_id,
        ),
        connection=conn,
    )


def _assert_write_rejected(
    backend: Any,
    query: str,
    params: tuple[Any, ...],
    *,
    match: str,
) -> None:
    with pytest.raises(BackendDatabaseError, match=match):
        with backend.transaction() as conn:
            backend.execute(query, params, connection=conn)


def _set_restricted_recipient(
    backend: Any,
    conn: Any,
    role_name: str,
    recipient_user_id: str | None,
) -> None:
    backend.execute(
        f"SET LOCAL ROLE {backend.escape_identifier(role_name)}",
        connection=conn,
    )
    backend.execute("SET LOCAL row_security = on", connection=conn)
    if recipient_user_id is None:
        backend.execute("SET LOCAL app.current_user_id TO DEFAULT", connection=conn)
    else:
        backend.execute(
            "SELECT set_config('app.current_user_id', ?, true)",
            (recipient_user_id,),
            connection=conn,
        )


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_postgres_v61_executes_constraints_defaults_and_cascades(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="recipient-a", backend=backend)
    try:
        bypasses_rls = bool(
            backend.execute(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).scalar
        )
        assert bypasses_rls, "The PostgreSQL constraint test requires the fixture admin role"

        with backend.transaction() as conn:
            for conversation_id in (
                "constraint-conversation",
                "constraint-conversation-2",
                "defaults-conversation",
                "set-null-conversation",
                "cascade-conversation",
            ):
                _insert_conversation(backend, conn, conversation_id, "recipient-a")
            _insert_thread(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=1,
                conversation_id="constraint-conversation",
            )

        thread_insert = """
            INSERT INTO shared_workspace_chat_threads(
                recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
            ) VALUES (?, ?, ?, ?, ?)
        """
        for params in (
            ("", 2, "constraint-conversation-2", "owner-a", "workspace-a"),
            ("   ", 2, "constraint-conversation-2", "owner-a", "workspace-a"),
            ("recipient-a", 0, "constraint-conversation-2", "owner-a", "workspace-a"),
            ("recipient-a", -1, "constraint-conversation-2", "owner-a", "workspace-a"),
            ("recipient-a", 2, "constraint-conversation-2", "", "workspace-a"),
            ("recipient-a", 2, "constraint-conversation-2", "   ", "workspace-a"),
        ):
            _assert_write_rejected(backend, thread_insert, params, match="check constraint")

        request_insert = """
            INSERT INTO shared_workspace_chat_requests(
                recipient_user_id, share_id, request_id, request_fingerprint,
                conversation_id, status, lease_epoch, source_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        for params in (
            ("", 1, "blank-recipient", "fingerprint", "constraint-conversation", "in_progress", 1, "all"),
            ("recipient-a", 0, "zero-share", "fingerprint", "constraint-conversation", "in_progress", 1, "all"),
            ("recipient-a", 1, "bad-status", "fingerprint", "constraint-conversation", "unknown", 1, "all"),
            ("recipient-a", 1, "bad-lease", "fingerprint", "constraint-conversation", "in_progress", 0, "all"),
            ("recipient-a", 1, "bad-source", "fingerprint", "constraint-conversation", "in_progress", 1, "exclude"),
        ):
            _assert_write_rejected(backend, request_insert, params, match="check constraint")

        _assert_write_rejected(
            backend,
            thread_insert,
            ("recipient-a", 1, "constraint-conversation-2", "owner-a", "workspace-a"),
            match="duplicate key value",
        )
        _assert_write_rejected(
            backend,
            thread_insert,
            ("recipient-a", 2, "constraint-conversation", "owner-a", "workspace-a"),
            match="duplicate key value",
        )

        with backend.transaction() as conn:
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=1,
                request_id="duplicate-request",
                conversation_id="constraint-conversation",
            )
        _assert_write_rejected(
            backend,
            request_insert,
            (
                "recipient-a",
                1,
                "duplicate-request",
                "fingerprint-duplicate",
                "constraint-conversation",
                "in_progress",
                1,
                "all",
            ),
            match="duplicate key value",
        )

        with backend.transaction() as conn:
            _insert_thread(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=10,
                conversation_id="defaults-conversation",
            )
            backend.execute(
                """
                INSERT INTO shared_workspace_chat_requests(
                    recipient_user_id, share_id, request_id, request_fingerprint,
                    conversation_id, status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    "recipient-a",
                    10,
                    "defaults-request",
                    "fingerprint-defaults",
                    "defaults-conversation",
                    "in_progress",
                ),
                connection=conn,
            )
        defaults = backend.execute(
            """
            SELECT thread.created_at AS thread_created_at,
                   thread.updated_at AS thread_updated_at,
                   request.lease_epoch,
                   request.created_at AS request_created_at,
                   request.updated_at AS request_updated_at
              FROM shared_workspace_chat_threads AS thread
              JOIN shared_workspace_chat_requests AS request
                ON request.recipient_user_id = thread.recipient_user_id
               AND request.share_id = thread.share_id
               AND request.conversation_id = thread.conversation_id
             WHERE request.request_id = ?
            """,
            ("defaults-request",),
        ).rows[0]
        assert defaults["lease_epoch"] == 1
        assert all(value is not None for key, value in defaults.items() if key != "lease_epoch")

        with backend.transaction() as conn:
            _insert_thread(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=20,
                conversation_id="set-null-conversation",
            )
            _insert_message(backend, conn, "user-message", "set-null-conversation", "recipient-a")
            _insert_message(
                backend,
                conn,
                "assistant-message",
                "set-null-conversation",
                "recipient-a",
            )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=20,
                request_id="set-null-request",
                conversation_id="set-null-conversation",
                user_message_id="user-message",
                assistant_message_id="assistant-message",
            )
            backend.execute(
                "DELETE FROM messages WHERE id IN (?, ?)",
                ("user-message", "assistant-message"),
                connection=conn,
            )
        message_refs = backend.execute(
            """
            SELECT user_message_id, assistant_message_id
              FROM shared_workspace_chat_requests
             WHERE request_id = ?
            """,
            ("set-null-request",),
        ).rows[0]
        assert message_refs == {"user_message_id": None, "assistant_message_id": None}

        with backend.transaction() as conn:
            _insert_thread(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=30,
                conversation_id="cascade-conversation",
            )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=30,
                request_id="cascade-request",
                conversation_id="cascade-conversation",
            )
            backend.execute(
                "DELETE FROM conversations WHERE id = ?",
                ("cascade-conversation",),
                connection=conn,
            )
        assert backend.execute(
            "SELECT count(*) FROM shared_workspace_chat_threads WHERE share_id = 30"
        ).scalar == 0
        assert backend.execute(
            "SELECT count(*) FROM shared_workspace_chat_requests WHERE share_id = 30"
        ).scalar == 0
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()


@pytest.mark.integration
@pytest.mark.timeout(60)
def test_postgres_v61_restricted_role_enforces_recipient_rls_predicates(
    pg_database_config: DatabaseConfig,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(":memory:", client_id="recipient-a", backend=backend)
    role_name = f"shared_chat_rls_{uuid4().hex[:12]}"
    ident = backend.escape_identifier
    role_created = False
    try:
        bypasses_rls = bool(
            backend.execute(
                "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
            ).scalar
        )
        assert bypasses_rls, "The PostgreSQL RLS test requires the fixture admin role for seeding"

        with backend.transaction() as conn:
            for conversation_id, recipient_user_id, deleted in (
                ("conversation-a", "recipient-a", False),
                ("conversation-b", "recipient-b", False),
                ("conversation-deleted", "recipient-a", True),
                ("conversation-other", "recipient-a", False),
                ("conversation-unthreaded", "recipient-a", False),
                ("conversation-new-a", "recipient-a", False),
                ("conversation-new-b", "recipient-b", False),
            ):
                _insert_conversation(
                    backend,
                    conn,
                    conversation_id,
                    recipient_user_id,
                    deleted=deleted,
                )
            _insert_message(backend, conn, "message-a", "conversation-a", "recipient-a")
            _insert_message(backend, conn, "message-b", "conversation-b", "recipient-b")
            _insert_message(
                backend,
                conn,
                "message-other",
                "conversation-other",
                "recipient-a",
            )
            for recipient_user_id, share_id, conversation_id in (
                ("recipient-a", 1, "conversation-a"),
                ("recipient-b", 2, "conversation-b"),
                ("recipient-a", 3, "conversation-deleted"),
                ("recipient-a", 4, "conversation-other"),
            ):
                _insert_thread(
                    backend,
                    conn,
                    recipient_user_id=recipient_user_id,
                    share_id=share_id,
                    conversation_id=conversation_id,
                )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=1,
                request_id="visible-request",
                conversation_id="conversation-a",
                user_message_id="message-a",
            )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-b",
                share_id=2,
                request_id="foreign-recipient-request",
                conversation_id="conversation-b",
            )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=3,
                request_id="deleted-conversation-request",
                conversation_id="conversation-deleted",
            )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=1,
                request_id="foreign-message-request",
                conversation_id="conversation-a",
                user_message_id="message-b",
            )
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=1,
                request_id="cross-conversation-message-request",
                conversation_id="conversation-a",
                assistant_message_id="message-other",
            )
            backend.execute(
                f"CREATE ROLE {ident(role_name)} NOLOGIN NOSUPERUSER NOBYPASSRLS",
                connection=conn,
            )
            backend.execute(f"GRANT USAGE ON SCHEMA public TO {ident(role_name)}", connection=conn)
            backend.execute(
                f"GRANT SELECT ON conversations, messages TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(
                "GRANT SELECT, INSERT, UPDATE ON "
                "shared_workspace_chat_threads, shared_workspace_chat_requests "
                f"TO {ident(role_name)}",
                connection=conn,
            )
            backend.execute(f"GRANT {ident(role_name)} TO CURRENT_USER", connection=conn)
        role_created = True

        with backend.transaction() as conn:
            _set_restricted_recipient(backend, conn, role_name, None)
            principal = backend.execute(
                "SELECT current_user AS role_name, rolsuper, rolbypassrls "
                "FROM pg_roles WHERE rolname = current_user",
                connection=conn,
            ).rows[0]
            setting = backend.execute(
                "SELECT current_setting('app.current_user_id', true) AS value",
                connection=conn,
            ).scalar
            assert principal == {
                "role_name": role_name,
                "rolsuper": False,
                "rolbypassrls": False,
            }
            assert setting in (None, "")
            assert backend.execute(
                "SELECT share_id FROM shared_workspace_chat_threads",
                connection=conn,
            ).rows == []
            assert backend.execute(
                "SELECT request_id FROM shared_workspace_chat_requests",
                connection=conn,
            ).rows == []

        def assert_rls_denied(
            query: str,
            params: tuple[Any, ...],
            *,
            recipient_user_id: str | None = "recipient-a",
        ) -> None:
            with pytest.raises(BackendDatabaseError, match="row-level security"):
                with backend.transaction() as conn:
                    _set_restricted_recipient(
                        backend,
                        conn,
                        role_name,
                        recipient_user_id,
                    )
                    backend.execute(query, params, connection=conn)

        assert_rls_denied(
            """
            INSERT INTO shared_workspace_chat_threads(
                recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("recipient-a", 10, "conversation-new-a", "owner-a", "workspace-a"),
            recipient_user_id=None,
        )

        with backend.transaction() as conn:
            _set_restricted_recipient(backend, conn, role_name, "recipient-a")
            visible_threads = backend.execute(
                "SELECT share_id FROM shared_workspace_chat_threads ORDER BY share_id",
                connection=conn,
            ).rows
            visible_requests = backend.execute(
                "SELECT request_id FROM shared_workspace_chat_requests ORDER BY request_id",
                connection=conn,
            ).rows
            hidden_update = backend.execute(
                "UPDATE shared_workspace_chat_threads SET workspace_id = ? WHERE share_id = ?",
                ("cross-recipient-overwrite", 2),
                connection=conn,
            )
            assert visible_threads == [{"share_id": 1}, {"share_id": 4}]
            assert visible_requests == [{"request_id": "visible-request"}]
            assert hidden_update.rowcount == 0

        assert_rls_denied(
            """
            INSERT INTO shared_workspace_chat_threads(
                recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("recipient-b", 11, "conversation-new-b", "owner-b", "workspace-b"),
        )
        for request_id, share_id, conversation_id, user_message_id, assistant_message_id in (
            ("deleted-write", 3, "conversation-deleted", None, None),
            ("mismatched-thread-write", 1, "conversation-unthreaded", None, None),
            ("foreign-message-write", 1, "conversation-a", "message-b", None),
            ("cross-conversation-message-write", 1, "conversation-a", None, "message-other"),
        ):
            assert_rls_denied(
                """
                INSERT INTO shared_workspace_chat_requests(
                    recipient_user_id, share_id, request_id, request_fingerprint,
                    conversation_id, status, user_message_id, assistant_message_id
                ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?)
                """,
                (
                    "recipient-a",
                    share_id,
                    request_id,
                    f"fingerprint-{request_id}",
                    conversation_id,
                    user_message_id,
                    assistant_message_id,
                ),
            )

        with backend.transaction() as conn:
            _set_restricted_recipient(backend, conn, role_name, "recipient-a")
            _insert_request(
                backend,
                conn,
                recipient_user_id="recipient-a",
                share_id=1,
                request_id="restricted-valid-write",
                conversation_id="conversation-a",
                user_message_id="message-a",
            )
        assert backend.execute(
            "SELECT count(*) FROM shared_workspace_chat_requests WHERE request_id = ?",
            ("restricted-valid-write",),
        ).scalar == 1
    finally:
        if role_created:
            with backend.transaction() as conn:
                backend.execute(f"REVOKE {ident(role_name)} FROM CURRENT_USER", connection=conn)
                backend.execute(f"DROP OWNED BY {ident(role_name)}", connection=conn)
                backend.execute(f"DROP ROLE {ident(role_name)}", connection=conn)
        db.close_all_connections()
        backend.get_pool().close_all()


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
