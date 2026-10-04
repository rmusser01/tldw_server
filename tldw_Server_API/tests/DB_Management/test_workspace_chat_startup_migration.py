"""Real previous-schema upgrades retain private Workspace retry tombstones."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, CharactersRAGDBError
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory

pytestmark = pytest.mark.integration
TABLE = "workspace_chat_startup_receipts"


def test_upgrade_and_reopen_retains_existing_chat(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Upgrade actual prior storage without replaying completed PostgreSQL DDL."""
    with monkeypatch.context() as old_code:
        old_code.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 73)
        old_code.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 77)
        old = db_factory()
        cid = old.add_conversation({"title": "Retained"})
        assert not old.backend.table_exists(TABLE)
        assert old.backend.table_exists("persona_buddy_preferences")
        assert old.backend.table_exists("persona_visual_pack_reviews")
        old.close_all_connections()
    backend_class = type(old.backend)
    execute = backend_class.execute
    ddl: list[str] = []

    def observe_ddl(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        """Observe real execution without replacing migration or catalog checks."""
        if query.lstrip().upper().startswith(("CREATE TABLE", "ALTER TABLE", "DROP TABLE")):
            ddl.append(query)
        return execute(self, query, *args, **kwargs)

    for _ in range(2):
        ddl.clear()
        with monkeypatch.context() as upgrade:
            if old.backend_type.value == "postgresql":
                upgrade.setattr(backend_class, "execute", observe_ddl)
            upgraded = db_factory()
        assert all(TABLE in statement or "message_insertion_order" in statement for statement in ddl), ddl
        assert upgraded.backend.table_exists(TABLE)
        assert upgraded.get_conversation_by_id(cid)["title"] == "Retained"
        assert upgraded.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (upgraded._SCHEMA_NAME,), read_only=True
        ).fetchone()["version"] == (
            upgraded._POSTGRES_SCHEMA_VERSION
            if upgraded.backend_type.value == "postgresql"
            else upgraded._CURRENT_SCHEMA_VERSION
        )
        upgraded.close_all_connections()


def test_interrupted_upgrade_rolls_back_table_and_version(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Failed registered DDL leaves the previous schema reopenable and unchanged."""
    with monkeypatch.context() as old_code:
        old_code.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 73)
        old_code.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 77)
        old = db_factory()
        old.close_all_connections()
    method = (
        "_migrate_from_v77_to_v78_postgres" if old.backend_type.value == "postgresql" else "_migrate_from_v73_to_v74"
    )
    assert hasattr(CharactersRAGDB, method), "Workspace startup migration must be registered"
    migrate = getattr(CharactersRAGDB, method)

    def fail_after_ddl(self: CharactersRAGDB, conn: Any) -> None:
        """Interrupt after the real migration, including its version bump."""
        migrate(self, conn)
        raise CharactersRAGDBError("injected receipt migration failure")

    with monkeypatch.context() as failing:
        failing.setattr(CharactersRAGDB, method, fail_after_ddl)
        with pytest.raises(CharactersRAGDBError, match="injected receipt migration failure"):
            db_factory()
    with monkeypatch.context() as old_code:
        old_code.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 73)
        old_code.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 77)
        reopened = db_factory()
        assert not reopened.backend.table_exists(TABLE)
        assert reopened.execute_query(
            "SELECT version FROM db_schema_version WHERE schema_name = ?", (reopened._SCHEMA_NAME,), read_only=True
        ).fetchone()["version"] == (77 if reopened.backend_type.value == "postgresql" else 73)


@pytest.mark.parametrize(
    "bad_hash",
    [
        None,
        "",
        "a" * 63,
        "a" * 65,
        "z" * 64,
        "A" * 64,
        "a" * 64 + "\x00" + "snapshot" * 1000,
        bytes(64),
        "a" * 63 + "\x00",
        "a" * 31 + "\x00" + "!" * 32,
    ],
    ids=["null", "empty", "short", "long", "not-hex", "uppercase", "nul-suffix", "blob", "nul-exact", "nul-middle"],
)
def test_receipts_reject_missing_or_unbounded_digests(
    db_factory: Callable[[], CharactersRAGDB], bad_hash: str | bytes | None
) -> None:
    """All persisted hash fields are non-null lowercase SHA256 encodings."""
    db = db_factory()
    assert db.backend.table_exists(TABLE)
    for column in ("key_digest", "request_fingerprint", "binding_digest"):
        values = {"key_digest": "a" * 64, "request_fingerprint": "b" * 64, "binding_digest": "c" * 64}
        values[column] = bad_hash
        with pytest.raises((sqlite3.IntegrityError, DatabaseError, CharactersRAGDBError)):
            with db.transaction() as conn:
                conn.execute(
                    "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id) VALUES (?, ?, ?, ?, ?)",
                    ("user-1", values["key_digest"], values["request_fingerprint"], values["binding_digest"], "ws"),
                )


def test_conversation_fk_has_a_leading_index(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """Parent deletion can find references without scanning all owners' lifetime keys."""
    db = db_factory()
    if db.backend_type.value == "postgresql":
        definition = db.execute_query(
            "SELECT indexdef FROM pg_indexes WHERE schemaname = current_schema() AND indexname = ?",
            ("workspace_chat_startup_conversation",),
            read_only=True,
        ).fetchone()["indexdef"]
        assert "(conversation_id, owner_user_id)" in definition
    else:
        columns = db.execute_query("PRAGMA index_info(workspace_chat_startup_conversation)", read_only=True).fetchall()
        assert [column["name"] for column in columns] == ["conversation_id", "owner_user_id"]


def test_private_schema_and_delete_tombstones(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """No raw snapshots or Workspace cascade; direct chat deletion nulls its reference."""
    db = db_factory()
    assert db.backend.table_exists(TABLE)
    assert {col["name"] for col in db.backend.get_table_info(TABLE)} == {
        "owner_user_id",
        "key_digest",
        "request_fingerprint",
        "binding_digest",
        "workspace_id",
        "conversation_id",
        "created_at",
        "invalidated_at",
    }
    db.upsert_workspace("ws", "Workspace")
    cid = db.add_conversation({"title": "Accepted", "scope_type": "workspace", "workspace_id": "ws"})
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id, conversation_id) VALUES (?, ?, ?, ?, ?, ?)",
            ("user-1", "a" * 64, "b" * 64, "c" * 64, "ws", cid),
        )
        conn.execute("DELETE FROM conversations WHERE id = ?", (cid,))
        conn.execute("DELETE FROM workspaces WHERE id = ?", ("ws",))
        receipt = conn.execute("SELECT * FROM workspace_chat_startup_receipts").fetchone()
        assert receipt["conversation_id"] is None
        assert receipt["workspace_id"] == "ws"
    db.close_all_connections()
    with db_factory().transaction() as conn:
        assert conn.execute("SELECT COUNT(*) AS n FROM workspace_chat_startup_receipts").fetchone()["n"] == 1


def test_owner_key_unique_across_workspaces(db_factory: Callable[[], CharactersRAGDB]) -> None:
    """One owner's key cannot be accepted again in a different Workspace."""
    db = db_factory()
    assert db.backend.table_exists(TABLE)
    with db.transaction() as conn:
        conn.execute(
            "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id) VALUES (?, ?, ?, ?, ?)",
            ("user-1", "a" * 64, "b" * 64, "c" * 64, "ws"),
        )
    with pytest.raises((sqlite3.IntegrityError, DatabaseError, CharactersRAGDBError)):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO workspace_chat_startup_receipts (owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id) VALUES (?, ?, ?, ?, ?)",
                ("user-1", "a" * 64, "b" * 64, "c" * 64, "different"),
            )
