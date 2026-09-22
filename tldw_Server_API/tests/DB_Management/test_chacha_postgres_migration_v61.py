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
    UniqueConstraintError,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha import schema_bootstrap
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


class _ReachedV61(Exception):
    pass


class _FakeTransaction:
    def __init__(self, connection: object) -> None:
        self.connection = connection
        self.exit_exception = None

    def __enter__(self) -> object:
        return self.connection

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        self.exit_exception = exc_type
        return False


class _FakeBackend:
    backend_type = BackendType.POSTGRESQL

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction(self)

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

    # Keep version routing separate from the real PostgreSQL session-lock tests.
    migration_transaction = _FakeTransaction(object())
    coordinator_calls: list[tuple[object, str]] = []
    schema_version_reads: list[tuple[object, bool]] = []
    migration_calls: list[object] = []

    def _schema_migration(backend, lock_timeout):
        coordinator_calls.append((backend, lock_timeout))
        return migration_transaction

    def _schema_version(conn: object, *, lock: bool = False) -> int:
        schema_version_reads.append((conn, lock))
        return 60

    monkeypatch.setattr(schema_bootstrap, "postgres_schema_migration", _schema_migration)
    monkeypatch.setattr(db, "_get_schema_version_postgres", _schema_version)
    monkeypatch.setattr(db, "_verify_note_attachment_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(db, "_verify_note_task_schema_postgres", lambda _conn: None)
    monkeypatch.setattr(
        db,
        "_configure_notes_moodboard_studio_v61_postgres_transaction",
        lambda _conn: None,
    )

    def _reached_v61(_conn: object) -> None:
        migration_calls.append(_conn)
        raise _ReachedV61

    monkeypatch.setattr(db, "_migrate_from_v60_to_v61_postgres", _reached_v61, raising=False)

    with pytest.raises(_ReachedV61):
        db._initialize_schema_postgres()

    assert migration_calls == [migration_transaction.connection]
    assert schema_version_reads == [
        (db._backend, False),
        (migration_transaction.connection, False),
        (migration_transaction.connection, True),
    ]
    assert coordinator_calls == [
        (db._backend, db._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT),
    ]
    assert migration_transaction.exit_exception is _ReachedV61


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


def _shared_chat_rows(backend: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read complete rows as the fixture admin to observe rejected-write rollback."""
    return (
        backend.execute(
            "SELECT * FROM shared_workspace_chat_threads ORDER BY recipient_user_id, share_id"
        ).rows,
        backend.execute(
            "SELECT * FROM shared_workspace_chat_requests ORDER BY recipient_user_id, share_id, request_id"
        ).rows,
    )


def _assert_write_rejected(
    backend: Any,
    query: str,
    params: tuple[Any, ...],
    *,
    error_type: type[BackendDatabaseError] = BackendDatabaseError,
) -> None:
    before = _shared_chat_rows(backend)
    # Driver diagnostics are intentionally private; verify the public error and
    # actual database state after transaction rollback instead of their text.
    with pytest.raises(error_type, match="^PostgreSQL query execution failed$"):
        with backend.transaction() as conn:
            backend.execute(query, params, connection=conn)
    assert _shared_chat_rows(backend) == before


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
            _assert_write_rejected(backend, thread_insert, params)

        # These invalid values would also fail the composite thread FK. Verify
        # their exact validated CHECKs so a generic FK error cannot mask a gap.
        request_checks = backend.execute(
            "SELECT conname, convalidated, pg_get_constraintdef(oid) AS definition "
            "FROM pg_constraint WHERE conrelid = 'shared_workspace_chat_requests'::regclass "
            "AND conname IN ('shared_workspace_chat_requests_recipient_user_id_check', "
            "'shared_workspace_chat_requests_share_id_check')"
        ).rows
        assert {row["conname"]: (row["convalidated"], row["definition"]) for row in request_checks} == {
            "shared_workspace_chat_requests_recipient_user_id_check": (
                True, "CHECK ((char_length(btrim(recipient_user_id)) > 0))",
            ),
            "shared_workspace_chat_requests_share_id_check": (True, "CHECK ((share_id > 0))"),
        }
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
            _assert_write_rejected(backend, request_insert, params)

        _assert_write_rejected(
            backend,
            thread_insert,
            ("recipient-a", 1, "constraint-conversation-2", "owner-a", "workspace-a"),
            error_type=UniqueConstraintError,
        )
        _assert_write_rejected(
            backend,
            thread_insert,
            ("recipient-a", 2, "constraint-conversation", "owner-a", "workspace-a"),
            error_type=UniqueConstraintError,
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
            error_type=UniqueConstraintError,
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
            before = _shared_chat_rows(backend)
            with pytest.raises(BackendDatabaseError, match="^PostgreSQL query execution failed$"):
                with backend.transaction() as conn:
                    _set_restricted_recipient(
                        backend,
                        conn,
                        role_name,
                        recipient_user_id,
                    )
                    backend.execute(query, params, connection=conn)
            assert _shared_chat_rows(backend) == before

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
def test_postgres_v60_to_v61_constraints_forced_rls_and_head_rerun(
    pg_database_config: DatabaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_initialize_schema", lambda self: None)
        db = CharactersRAGDB(":memory:", client_id="recipient-a", backend=backend)
    try:
        class _ReachedHistoricalV60(Exception):
            pass

        stopped_connections = []
        stopped_process_ids = []

        def stop_before_v61(conn: Any) -> None:
            assert db._get_schema_version_postgres(conn) == 60
            stopped_connections.append(conn)
            stopped_process_ids.append(conn.info.backend_pid)
            # Test-only historical boundary: persist the real v4-to-v60 prefix
            # before stopping initialization. The real coordinator must unwind
            # the sentinel, release its session lock, and discard this checkout.
            conn.commit()
            raise _ReachedHistoricalV60

        with monkeypatch.context() as patch:
            patch.setattr(db, "_migrate_from_v60_to_v61_postgres", stop_before_v61)
            with pytest.raises(_ReachedHistoricalV60):
                db._initialize_schema_postgres()
        assert len(stopped_connections) == 1
        assert stopped_connections[0].closed
        assert backend.execute(
            "SELECT count(*) FROM pg_locks WHERE pid=%s AND locktype='advisory' AND granted",
            (stopped_process_ids[0],),
        ).scalar == 0
        assert backend.execute(
            "SELECT version FROM db_schema_version WHERE schema_name = %s",
            (CharactersRAGDB._SCHEMA_NAME,),
        ).scalar == 60
        assert not backend.table_exists("shared_workspace_chat_threads")
        assert not backend.table_exists("shared_workspace_chat_requests")

        conversation_id = "conversation-a"
        with backend.transaction() as conn:
            _insert_conversation(backend, conn, conversation_id, "recipient-a")
        conversation_before = backend.execute(
            "SELECT * FROM conversations WHERE id = %s", (conversation_id,)
        ).rows
        assert "history_version" not in conversation_before[0]
        assert "conversations_fts_tsv" not in conversation_before[0]
        with schema_bootstrap.postgres_schema_migration(
            backend, db._NOTES_MOODBOARD_STUDIO_V61_POSTGRES_LOCK_TIMEOUT
        ) as conn:
            db._migrate_from_v60_to_v61_postgres(conn)
        assert backend.execute(
            "SELECT * FROM conversations WHERE id = %s", (conversation_id,)
        ).rows == conversation_before

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

        with backend.transaction() as conn:
            backend.execute(
                """
                INSERT INTO shared_workspace_chat_threads(
                    recipient_user_id, share_id, conversation_id, owner_user_id, workspace_id
                ) VALUES (?, ?, ?, ?, ?)
                """,
                ("recipient-a", 1, conversation_id, "owner-a", "workspace-a"),
                connection=conn,
            )
            backend.execute(
                """
                INSERT INTO shared_workspace_chat_requests(
                    recipient_user_id, share_id, request_id, request_fingerprint,
                    conversation_id, status, source_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                ("recipient-a", 1, "request-a", "fingerprint-a", conversation_id, "completed", "include"),
                connection=conn,
            )
        shared_rows_before = _shared_chat_rows(backend)

        with pytest.raises(BackendDatabaseError, match="^PostgreSQL query execution failed$"):
            with backend.transaction() as conn:
                backend.execute(
                    """
                    INSERT INTO shared_workspace_chat_requests(
                        recipient_user_id, share_id, request_id, request_fingerprint,
                        conversation_id, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("recipient-a", 2, "wrong-share", "fingerprint-b", conversation_id, "completed"),
                    connection=conn,
                )
        assert _shared_chat_rows(backend) == shared_rows_before

        for _ in range(2):
            db._initialize_schema_postgres()
            rerun_version = backend.execute(
                "SELECT version FROM db_schema_version WHERE schema_name = %s",
                (CharactersRAGDB._SCHEMA_NAME,),
            ).scalar
            rerun_relations, rerun_policies = _policy_catalog(backend)
            assert int(rerun_version) == CharactersRAGDB._POSTGRES_SCHEMA_VERSION
            assert rerun_relations == relations
            assert rerun_policies == policies
            assert _shared_chat_rows(backend) == shared_rows_before
            # v65 adds history_version; head reconciliation adds the title's
            # FTS vector. Every historical column and value remains unchanged.
            assert backend.execute(
                "SELECT * FROM conversations WHERE id = %s", (conversation_id,)
            ).rows == [
                {**conversation_before[0], "history_version": 1, "conversations_fts_tsv": ""},
            ]
    finally:
        db.close_all_connections()
        backend.get_pool().close_all()
