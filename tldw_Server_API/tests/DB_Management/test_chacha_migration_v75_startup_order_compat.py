"""Real old-schema compatibility for startup receipts and native message order."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import build_workspace_chat_startup_rls_sql
from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_schema import (
    workspace_chat_startup_schema_statements,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
)
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements
from tldw_Server_API.tests.DB_Management.test_conversation_assistant_startup import db_factory as db_factory

pytestmark = pytest.mark.integration
RECEIPTS = "workspace_chat_startup_receipts"
ORDER = "message_insertion_order"


@pytest.fixture(autouse=True)
def _compatibility_target(monkeypatch: pytest.MonkeyPatch) -> None:
    # Later feature migrations have their own catalog coverage.
    monkeypatch.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 75)
    monkeypatch.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 79)

# Populated mechanically from the frozen prior-native migration, not modern DDL.
_PRIOR_NATIVE_SQLITE_ORDER_DDL = (
    """
            CREATE TABLE message_insertion_order (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT UNIQUE NOT NULL REFERENCES messages(id) ON DELETE CASCADE
            )
            """,
    """
            CREATE TRIGGER messages_record_insertion_order AFTER INSERT ON messages
            BEGIN
                INSERT INTO message_insertion_order(message_id) VALUES (NEW.id);
            END
            """,
    """
            CREATE TRIGGER message_insertion_order_immutable BEFORE UPDATE ON message_insertion_order
            BEGIN
                SELECT RAISE(ABORT, 'Message insertion order is immutable.');
            END
            """,
)
_PRIOR_NATIVE_POSTGRES_ORDER_SQL = """
            CREATE TABLE message_insertion_order (
                sequence BIGSERIAL PRIMARY KEY,
                message_id TEXT UNIQUE NOT NULL REFERENCES messages(id) ON DELETE CASCADE
            );
            ALTER TABLE message_insertion_order ENABLE ROW LEVEL SECURITY;
            ALTER TABLE message_insertion_order FORCE ROW LEVEL SECURITY;
            CREATE POLICY message_insertion_order_tenant_isolation ON message_insertion_order
                USING (EXISTS (
                    SELECT 1 FROM messages AS message
                    WHERE message.id = message_insertion_order.message_id
                        AND message.client_id = current_setting('app.current_user_id', true)
                ))
                WITH CHECK (EXISTS (
                    SELECT 1 FROM messages AS message
                    WHERE message.id = message_insertion_order.message_id
                        AND message.client_id = current_setting('app.current_user_id', true)
                ));
            CREATE FUNCTION messages_lock_insertion_order() RETURNS TRIGGER AS $$
            BEGIN
                PERFORM id FROM conversations WHERE id = NEW.conversation_id FOR UPDATE;
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER messages_lock_insertion_order BEFORE INSERT ON messages
                FOR EACH ROW EXECUTE FUNCTION messages_lock_insertion_order();
            CREATE FUNCTION messages_record_insertion_order() RETURNS TRIGGER AS $$
            BEGIN
                INSERT INTO message_insertion_order(message_id) VALUES (NEW.id);
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER messages_record_insertion_order AFTER INSERT ON messages
                FOR EACH ROW EXECUTE FUNCTION messages_record_insertion_order();
            CREATE FUNCTION message_insertion_order_immutable() RETURNS TRIGGER AS $$
            BEGIN
                RAISE EXCEPTION 'Message insertion order is immutable.';
            END;
            $$ LANGUAGE plpgsql;
            CREATE TRIGGER message_insertion_order_immutable BEFORE UPDATE ON message_insertion_order
                FOR EACH ROW EXECUTE FUNCTION message_insertion_order_immutable();
        """


def _postgres(db: CharactersRAGDB) -> bool:
    return db.backend_type.value == "postgresql"


def _version(db: CharactersRAGDB) -> int:
    return db.execute_query(
        "SELECT version FROM db_schema_version WHERE schema_name = ?",
        (db._SCHEMA_NAME,),
        read_only=True,
    ).fetchone()["version"]


def _snapshot(db: CharactersRAGDB) -> dict[str, Any]:
    tables = {table: db.backend.table_exists(table) for table in (RECEIPTS, ORDER)}
    return {
        "version": _version(db),
        "tables": tables,
        "messages": [
            dict(row) for row in db.execute_query("SELECT * FROM messages ORDER BY id", read_only=True).fetchall()
        ],
        "receipts": (
            [
                dict(row)
                for row in db.execute_query(
                    "SELECT * FROM workspace_chat_startup_receipts ORDER BY owner_user_id, key_digest", read_only=True
                ).fetchall()
            ]
            if tables[RECEIPTS]
            else []
        ),
        "order": (
            [
                dict(row)
                for row in db.execute_query(
                    "SELECT * FROM message_insertion_order ORDER BY sequence", read_only=True
                ).fetchall()
            ]
            if tables[ORDER]
            else []
        ),
    }


def _seed_shape(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch, shape: str
) -> tuple[CharactersRAGDB, str, dict[str, Any]]:
    with monkeypatch.context() as old_code:
        old_code.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", 73)
        old_code.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", 77)
        old = db_factory()
    cid = old.add_conversation({"title": "Unchanged", "client_id": "user-1"})
    old.add_message({"id": "legacy", "conversation_id": cid, "sender": "user", "content": "Before ordering"})
    assert not old.backend.table_exists(RECEIPTS)
    assert not old.backend.table_exists(ORDER)
    pg = _postgres(old)
    has_startup = shape in ("upstream", "both")
    has_order = shape in ("native", "both")
    if has_startup or has_order:
        with old.backend.transaction() as conn:
            if has_startup:
                for statement in workspace_chat_startup_schema_statements(postgres=pg):
                    old.backend.execute(statement, connection=conn)
                if pg:
                    for statement in build_workspace_chat_startup_rls_sql():
                        old.backend.execute(statement, connection=conn)
            if has_order:
                statements = (
                    split_sql_statements(_PRIOR_NATIVE_POSTGRES_ORDER_SQL) if pg else _PRIOR_NATIVE_SQLITE_ORDER_DDL
                )
                for statement in statements:
                    old.backend.execute(statement, connection=conn)
            query = (
                "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = %s"
                if pg
                else "UPDATE db_schema_version SET version = ? WHERE schema_name = ? AND version = ?"
            )
            old.backend.execute(
                query,
                (78 if pg else 74, old._SCHEMA_NAME, 77 if pg else 73),
                connection=conn,
            )
    if has_startup:
        with old.transaction() as conn:
            for key, invalidated in (("a" * 64, None), ("d" * 64, "2026-09-30T00:00:00Z")):
                conn.execute(
                    "INSERT INTO workspace_chat_startup_receipts "
                    "(owner_user_id, key_digest, request_fingerprint, binding_digest, workspace_id, conversation_id, invalidated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    ("user-1", key, "b" * 64, "c" * 64, "ws", cid, invalidated),
                )
    if has_order:
        for index in range(2):
            old.add_message(
                {
                    "id": f"ordered-{index}",
                    "conversation_id": cid,
                    "sender": "user",
                    "content": f"Ordered {index}",
                }
            )
    before = _snapshot(old)
    assert before["tables"] == {RECEIPTS: has_startup, ORDER: has_order}
    assert before["version"] == ((78 if pg else 74) if has_startup or has_order else (77 if pg else 73))
    assert {row["message_id"] for row in before["order"]} == ({"ordered-0", "ordered-1"} if has_order else set())
    return old, cid, before


@pytest.mark.parametrize("shape", ["common", "upstream", "native", "both"])
def test_actual_old_shapes_preserve_foundations_and_advance_order(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    old, cid, before = _seed_shape(db_factory, monkeypatch, shape)
    old.close_all_connections()
    upgraded = db_factory()
    after = _snapshot(upgraded)
    assert after["version"] == (79 if _postgres(upgraded) else 75)
    assert after["tables"] == {RECEIPTS: True, ORDER: True}
    assert after["messages"] == before["messages"]
    assert after["receipts"] == before["receipts"]
    assert after["order"] == before["order"]
    assert upgraded.get_conversation_by_id(cid)["title"] == "Unchanged"
    with pytest.raises(ConflictError, match="insertion order"):
        upgraded.insert_or_validate_user_turn(
            cid,
            "legacy",
            "Before ordering",
            owner_client_id=upgraded.client_id,
            conversation_context=upgraded.get_conversation_by_id(cid),
        )
    upgraded.add_message({"id": "new", "conversation_id": cid, "sender": "user", "content": "After upgrade"})
    new_sequence = upgraded.execute_query(
        "SELECT sequence FROM message_insertion_order WHERE message_id = ?", ("new",), read_only=True
    ).fetchone()["sequence"]
    assert new_sequence > max((row["sequence"] for row in before["order"]), default=0)
    assert (
        upgraded.execute_query(
            "SELECT message_id FROM message_insertion_order WHERE message_id = ?", ("legacy",), read_only=True
        ).fetchone()
        is None
    )
    upgraded.close_all_connections()
    reopened = db_factory()
    assert _snapshot(reopened)["receipts"] == before["receipts"]
    assert _version(reopened) == (79 if _postgres(reopened) else 75)


@pytest.mark.parametrize("shape", ["common", "upstream", "native", "both"])
def test_failed_compatibility_step_restores_exact_old_shape(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch, shape: str
) -> None:
    old, _, before = _seed_shape(db_factory, monkeypatch, shape)
    pg = _postgres(old)
    name = "_migrate_from_v78_to_v79_postgres" if pg else "_migrate_from_v74_to_v75"
    assert hasattr(CharactersRAGDB, name), "Owned ordering must have an independent 75/79 migration"
    migrate = getattr(CharactersRAGDB, name)
    old.close_all_connections()

    def fail_after_version_write(self: CharactersRAGDB, conn: Any) -> None:
        migrate(self, conn)
        raise CharactersRAGDBError("injected compatibility failure")

    with monkeypatch.context() as failing:
        failing.setattr(CharactersRAGDB, name, fail_after_version_write)
        with pytest.raises(CharactersRAGDBError, match="injected compatibility failure"):
            db_factory()
    with monkeypatch.context() as old_code:
        old_code.setattr(CharactersRAGDB, "_CURRENT_SCHEMA_VERSION", before["version"] if not pg else 73)
        old_code.setattr(CharactersRAGDB, "_POSTGRES_SCHEMA_VERSION", before["version"] if pg else 77)
        reopened = db_factory()
    assert _snapshot(reopened) == before


def test_existing_collision_schema_skips_unrelated_postgres_ddl_and_reopen_migrations(
    db_factory: Callable[[], CharactersRAGDB], monkeypatch: pytest.MonkeyPatch
) -> None:
    old, _, _ = _seed_shape(db_factory, monkeypatch, "native")
    pg = _postgres(old)
    old.close_all_connections()
    ddl: list[str] = []
    backend_class = type(old.backend)
    execute = backend_class.execute

    def observe_real_execute(self: Any, query: str, *args: Any, **kwargs: Any) -> Any:
        if query.lstrip().upper().startswith(("CREATE ", "ALTER ", "DROP ")):
            ddl.append(query)
        return execute(self, query, *args, **kwargs)

    with monkeypatch.context() as observation:
        if pg:
            observation.setattr(backend_class, "execute", observe_real_execute)
        upgraded = db_factory()
    if pg:
        assert ddl, "Native 78 must install missing startup storage"
        fingerprint_ddl = CharactersRAGDB._CONVERSATION_CREATE_FINGERPRINT_SCHEMA_POSTGRES
        assert all("workspace_chat_startup" in query or query == fingerprint_ddl for query in ddl), ddl
        assert ddl.count(fingerprint_ddl) == 1
    assert upgraded.backend.table_exists(RECEIPTS)
    assert upgraded.backend.table_exists(ORDER)
    upgraded.close_all_connections()

    def reject_migration_replay(self: CharactersRAGDB, conn: Any) -> None:
        pytest.fail("Completed compatible schema must not replay startup/order migrations")

    with monkeypatch.context() as completed:
        for name in (
            "_migrate_from_v73_to_v74",
            "_migrate_from_v74_to_v75",
            "_migrate_from_v77_to_v78_postgres",
            "_migrate_from_v78_to_v79_postgres",
        ):
            completed.setattr(CharactersRAGDB, name, reject_migration_replay)
        reopened = db_factory()
    assert _version(reopened) == (79 if pg else 75)


def test_fresh_schema_contains_both_foundations_and_forced_postgres_rls(
    db_factory: Callable[[], CharactersRAGDB],
) -> None:
    db = db_factory()
    assert _version(db) == (79 if _postgres(db) else 75)
    assert db.backend.table_exists(RECEIPTS)
    assert db.backend.table_exists(ORDER)
    if _postgres(db):
        rows = db.execute_query(
            "SELECT relname, relrowsecurity, relforcerowsecurity FROM pg_class "
            "WHERE oid IN ('workspace_chat_startup_receipts'::regclass, 'message_insertion_order'::regclass) ORDER BY relname",
            read_only=True,
        ).fetchall()
        assert [(row["relname"], row["relrowsecurity"], row["relforcerowsecurity"]) for row in rows] == [
            (ORDER, True, True),
            (RECEIPTS, True, True),
        ]
        policies = db.execute_query(
            "SELECT tablename, policyname, qual, with_check FROM pg_policies "
            "WHERE schemaname = current_schema() AND tablename IN (?, ?) ORDER BY tablename, policyname",
            (ORDER, RECEIPTS),
            read_only=True,
        ).fetchall()
        assert {row["tablename"] for row in policies} == {ORDER, RECEIPTS}
        assert all(
            "app.current_user_id" in row["qual"] and "app.current_user_id" in row["with_check"] for row in policies
        )
