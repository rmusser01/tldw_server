"""Native fork storage upgrades keep prior chats and durable operation keys."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.chacha.native_fork_schema import native_fork_schema_statements
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError

NATIVE_TABLES = (
    "native_chat_operations",
    "native_chat_asset_candidates",
    "native_chat_asset_claims",
    "native_chat_asset_references",
    "native_chat_quota_intents",
)


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_upgrade_reopen_preserves_chat_and_installs_native_storage(backend_name, request, tmp_path, monkeypatch):
    kwargs = {"db_path": str(tmp_path / "native.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 71)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 75)
        previous = CharactersRAGDB(**kwargs)
        cid = previous.add_conversation({"character_id": 1, "title": "Keep this chat"})
        mid = previous.add_message({"conversation_id": cid, "sender": "user", "content": "keep this turn"})
        previous.close_connection()

    for _ in range(2):
        upgraded = CharactersRAGDB(**kwargs)
        assert upgraded.execute_query("SELECT content FROM messages WHERE id = ?", (mid,)).fetchone()["content"] == "keep this turn"
        expected_version = 76 if backend_name == "postgres" else 72
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


def test_sqlite_operation_receipt_has_no_cascading_chat_fk(tmp_path):
    db = CharactersRAGDB(db_path=str(tmp_path / "native.sqlite"), client_id="alice")
    assert db.backend.table_exists("native_chat_operations")
    foreign_keys = db.execute_query("PRAGMA foreign_key_list(native_chat_operations)").fetchall()
    assert all(row["table"] not in {"conversations", "messages"} for row in foreign_keys)
    db.close_connection()


def test_postgres_native_tables_enforce_direct_owner_rls(pg_database_config):
    """An app role sees only its own receipts even without a live chat row."""
    backend = DatabaseBackendFactory.create_backend(pg_database_config)
    db = CharactersRAGDB(db_path=":memory:", client_id="alice", backend=backend)
    role = "h2_probe_" + uuid.uuid4().hex

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
                conn.execute(f"CREATE ROLE {role} NOLOGIN NOSUPERUSER")  # nosec B608 - generated hex identifier
                conn.execute(f"GRANT USAGE ON SCHEMA public TO {role}")  # nosec B608
                for table in NATIVE_TABLES:
                    conn.execute(f"GRANT SELECT ON {table} TO {role}")  # nosec B608 - fixed table list
                conn.execute(f"SET LOCAL ROLE {role}")  # nosec B608
                for client_id in ("alice", "bob"):
                    conn.execute("SELECT set_config('app.current_user_id', ?, true)", (client_id,))
                    rows = conn.execute("SELECT client_id FROM native_chat_operations").fetchall()
                    assert [row["client_id"] for row in rows] == [client_id]
                raise RollbackProbe()
    finally:
        db.close_connection()
        backend.get_pool().close_all()


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
def test_native_schema_failure_rolls_back_tables_and_version(backend_name, request, tmp_path, monkeypatch):
    kwargs = {"db_path": str(tmp_path / "rollback.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    old_version = 75 if backend_name == "postgres" else 71
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 71)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 75)
        old = CharactersRAGDB(**kwargs)
        old.close_connection()
    method_name = "_migrate_from_v75_to_v76_postgres" if backend_name == "postgres" else "_migrate_from_v71_to_v72"
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


@pytest.mark.parametrize("backend_name", ["sqlite", "postgres"])
@pytest.mark.parametrize("partial", [False, True])
def test_colliding_native_lineage_preserves_receipts_or_rolls_back(
    backend_name: str, partial: bool, request: pytest.FixtureRequest,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Old H2 v70/v74 retains its receipts while gaining Persona provenance."""
    kwargs: dict[str, Any] = {"db_path": str(tmp_path / "old-native.sqlite"), "client_id": "alice"}
    if backend_name == "postgres":
        kwargs["backend"] = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    with monkeypatch.context() as patch:
        patch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 68)
        patch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 72)
        old = CharactersRAGDB(**kwargs)
        with old.transaction() as conn:
            conn.execute(
                "INSERT INTO conversations (id, root_id, character_id, title, client_id) VALUES (?, ?, ?, ?, ?)",
                ("child", "child", 1, "Preserved", "alice"),
            )
            conn.execute("INSERT INTO workspaces (id, name, client_id) VALUES (?, ?, ?)",
                         ("workspace", "Preserved", "alice"))
            conn.execute(CharactersRAGDB._HISTORY_PROJECTIONS_SCHEMA_SQL)
            conn.execute("ALTER TABLE messages ADD COLUMN history_admission_json TEXT")
            conn.execute(
                "INSERT INTO conversation_history_projections VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("projection", "child", "alice", "owner", 1, "digest", "fence", "[]", "[]", "{}", "digest", "2026-01-01"),
            )
            for statement in native_fork_schema_statements(postgres=backend_name == "postgres"):
                conn.execute(statement)
            conn.execute(
                "INSERT INTO native_chat_operations "
                "(client_id, operation_kind, operation_id, owner_key, scope_type, request_digest, "
                "projection_version, state, child_conversation_id, created_at, updated_at) "
                "VALUES (?, 'native_fork_v1', ?, ?, 'global', ?, 'native-fork-v1', 'committed', ?, ?, ?)",
                ("alice", "operation", "native:alice", "digest", "child", "2026-01-01", "2026-01-01"),
            )
            conn.execute("UPDATE workspaces SET native_chat_admission_closed = ? WHERE id = ?", (True, "workspace"))
            conn.execute(
                "UPDATE conversations SET required_projection_version = ?, native_creation_operation_kind = ?, "
                "native_creation_operation_id = ?, native_bundle_json = ? WHERE id = ?",
                ("native-fork-v1", "native_fork_v1", "operation", '{"preserved":true}', "child"),
            )
            if partial:
                conn.execute("DROP TABLE native_chat_quota_intents")
            original_version = 74 if backend_name == "postgres" else 70
            conn.execute("UPDATE db_schema_version SET version = ? WHERE schema_name = ?",
                         (original_version, old._SCHEMA_NAME))
        receipt = dict(old.execute_query("SELECT * FROM native_chat_operations").fetchone())
        projection = dict(old.execute_query("SELECT * FROM conversation_history_projections").fetchone())
        old.close_connection()
    try:
        if partial:
            with pytest.raises(CharactersRAGDBError, match="Incomplete native fork schema"):
                CharactersRAGDB(**kwargs)
            assert old.execute_query("SELECT version FROM db_schema_version").fetchone()["version"] == original_version
            assert dict(old.execute_query("SELECT * FROM native_chat_operations").fetchone()) == receipt
        else:
            for _ in range(2):
                upgraded = CharactersRAGDB(**kwargs)
                try:
                    assert dict(upgraded.execute_query("SELECT * FROM native_chat_operations").fetchone()) == receipt
                    assert dict(upgraded.execute_query("SELECT * FROM conversation_history_projections").fetchone()) == projection
                    workspace = upgraded.execute_query("SELECT * FROM workspaces WHERE id = ?", ("workspace",)).fetchone()
                    assert workspace["native_chat_admission_closed"]
                    assert workspace["assistant_defaults_explicit_none"]
                    child = upgraded.execute_query("SELECT * FROM conversations WHERE id = ?", ("child",)).fetchone()
                    assert tuple(child[field] for field in (
                        "required_projection_version", "native_creation_operation_kind",
                        "native_creation_operation_id", "native_bundle_json",
                    )) == ("native-fork-v1", "native_fork_v1", "operation", '{"preserved":true}')
                    assert child["assistant_startup_json"] is None
                    assert upgraded.execute_query("SELECT version FROM db_schema_version").fetchone()["version"] == (
                        76 if backend_name == "postgres" else 72
                    )
                finally:
                    upgraded.close_connection()
    finally:
        old.close_connection()
        if backend_name == "postgres":
            kwargs["backend"].get_pool().close_all()
