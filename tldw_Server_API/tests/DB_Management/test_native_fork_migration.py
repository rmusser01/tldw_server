"""Native fork storage upgrades keep prior chats and durable operation keys."""

from __future__ import annotations

from pathlib import Path

import pytest
from psycopg import sql

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB

NATIVE_TABLES = (
    "native_chat_operations",
    "native_chat_asset_candidates",
    "native_chat_asset_claims",
    "native_chat_asset_references",
    "native_chat_quota_intents",
)

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_upgrade_reopen_preserves_chat_and_installs_native_storage(
    backend_name: str, request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = {"db_path": str(tmp_path / "native.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 69)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 73)
        previous = CharactersRAGDB(**kwargs)
        cid = previous.add_conversation({"character_id": 1, "title": "Keep this chat"})
        mid = previous.add_message({"conversation_id": cid, "sender": "user", "content": "keep this turn"})
        previous.close_connection()

    for _ in range(2):
        upgraded = CharactersRAGDB(**kwargs)
        assert upgraded.execute_query("SELECT content FROM messages WHERE id = ?", (mid,)).fetchone()["content"] == "keep this turn"
        expected_version = 74 if backend_name == "postgres" else 70
        assert upgraded.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (CharactersRAGDB._SCHEMA_NAME,)
        ).fetchone()["version"] == expected_version
        for table in NATIVE_TABLES:
            assert upgraded.backend.table_exists(table)
        workspace_columns = (
            {row["column_name"] for row in upgraded.execute_query(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'workspaces'"
            ).fetchall()}
            if backend_name == "postgres"
            else {row["name"] for row in upgraded.execute_query("PRAGMA table_info(workspaces)").fetchall()}
        )
        assert "native_chat_admission_closed" in workspace_columns
        upgraded.close_connection()
    if backend_name == "postgres":
        kwargs["backend"].get_pool().close_all()


def test_sqlite_operation_receipt_has_no_cascading_chat_fk(tmp_path: Path) -> None:
    db = CharactersRAGDB(db_path=str(tmp_path / "native.sqlite"), client_id="alice")
    assert db.backend.table_exists("native_chat_operations")
    foreign_keys = db.execute_query("PRAGMA foreign_key_list(native_chat_operations)").fetchall()
    assert all(row["table"] not in {"conversations", "messages"} for row in foreign_keys)
    db.close_connection()


def test_postgres_native_tables_enforce_direct_owner_rls(pg_database_config: object) -> None:
    """An app role sees only its own receipts even without a live chat row."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    role = "h2_probe_owner_rls"

    class RollbackProbe(Exception):
        pass

    try:
        with pytest.raises(RollbackProbe):
            with db.transaction() as conn:
                for client_id in ("alice", "bob"):
                    conn.execute(
                        "INSERT INTO native_chat_operations "
                        "(client_id, operation_kind, operation_id, owner_key, scope_type, "
                        "request_digest, projection_version, state, created_at, updated_at) "
                        "VALUES (?, 'native_fork_v1', ?, ?, 'global', ?, 'native-fork-v1', "
                        "'gone', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
                        (client_id, f"operation-{client_id}", f"native:{client_id}", f"sha256:{client_id}"),
                    )
                for table in NATIVE_TABLES:
                    flags = conn.execute(
                        "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE oid = ?::regclass",
                        (table,),
                    ).fetchone()
                    assert flags["relrowsecurity"] and flags["relforcerowsecurity"]
                role_sql = sql.Identifier(role).as_string(conn._connection)
                conn.execute(f"CREATE ROLE {role_sql} NOLOGIN NOSUPERUSER")  # nosec B608 - quoted identifier
                conn.execute(f"GRANT USAGE ON SCHEMA public TO {role_sql}")  # nosec B608 - quoted identifier
                for table in NATIVE_TABLES:
                    table_sql = sql.Identifier(table).as_string(conn._connection)
                    conn.execute(f"GRANT SELECT ON {table_sql} TO {role_sql}")  # nosec B608 - quoted identifiers
                conn.execute(f"SET LOCAL ROLE {role_sql}")  # nosec B608 - quoted identifier
                for client_id in ("alice", "bob"):
                    conn.execute("SELECT set_config('app.current_user_id', ?, true)", (client_id,))
                    rows = conn.execute("SELECT client_id FROM native_chat_operations").fetchall()
                    assert [row["client_id"] for row in rows] == [client_id]
                raise RollbackProbe()
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_native_schema_failure_rolls_back_tables_and_version(
    backend_name: str, request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    kwargs = {"db_path": str(tmp_path / "rollback.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    old_version = 73 if backend_name == "postgres" else 69
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 69)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 73)
        old = CharactersRAGDB(**kwargs)
        old.close_connection()
    method_name = "_migrate_from_v73_to_v74_postgres" if backend_name == "postgres" else "_migrate_from_v69_to_v70"
    migrate = getattr(CharactersRAGDB, method_name)

    def interrupted(self, conn):
        migrate(self, conn)
        raise RuntimeError("native migration interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, method_name, interrupted)
        with pytest.raises(Exception, match="native migration interrupted"):
            CharactersRAGDB(**kwargs)
    assert old.execute_query(
        "SELECT version FROM db_schema_version WHERE schema_name = ?", (CharactersRAGDB._SCHEMA_NAME,)
    ).fetchone()["version"] == old_version
    assert not old.backend.table_exists("native_chat_operations")
    if backend_name == "postgres":
        kwargs["backend"].get_pool().close_all()
