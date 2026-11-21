"""Catalog helpers for the existing chat message-order upgrade paths."""

import sqlite3
from collections.abc import Callable
from typing import Any

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseBackend
from tldw_Server_API.app.core.DB_Management.sql_utils import split_sql_statements


def ensure_message_order_sqlite(backend: DatabaseBackend, conn: sqlite3.Connection, fingerprint_sql: str) -> None:
    """Reconcile the two V75 catalogs without reconstructing old rows."""
    columns = {column["name"] for column in backend.get_table_info("conversations", connection=conn)}
    if "create_request_fingerprint" not in columns:
        conn.execute(fingerprint_sql)
    # The statement splitter does not parse SQLite trigger BEGIN/END blocks.
    statements = (
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
    if not backend.table_exists("message_insertion_order", connection=conn):
        for statement in statements:
            conn.execute(statement)


def ensure_message_order_postgres(backend: DatabaseBackend, conn: Any, fingerprint_sql: str) -> None:
    """Reconcile V79 catalogs while retaining serialized writers and forced RLS."""
    columns = {column["name"] for column in backend.get_table_info("conversations", connection=conn)}
    if "create_request_fingerprint" not in columns:
        backend.execute(fingerprint_sql, connection=conn)
    sql = """
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
    if not backend.table_exists("message_insertion_order", connection=conn):
        for statement in split_sql_statements(sql):
            backend.execute(statement, connection=conn)


def migrate_sqlite_chat_catalog(
    conn: sqlite3.Connection,
    *,
    backend: DatabaseBackend,
    fingerprint_sql: str,
    schema_name: str,
    get_version: Callable[[sqlite3.Connection], int],
) -> None:
    """Complete missing catalog pieces without deriving history or quota identities."""
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import (
        notes_provenance_receipts_schema_sql,
        notes_provenance_schema_sql,
    )
    from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_schema import (
        workspace_chat_startup_schema_statements,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

    if not backend.table_exists("workspace_chat_startup_receipts", connection=conn):
        for statement in workspace_chat_startup_schema_statements(postgres=False):
            conn.execute(statement)
    ensure_message_order_sqlite(backend, conn, fingerprint_sql)
    conn.execute(notes_provenance_schema_sql(postgres=False))
    conn.execute(notes_provenance_receipts_schema_sql())
    columns = backend.get_table_info("native_chat_quota_intents", connection=conn)
    if not any(column["name"] == "quota_owner_json" for column in columns):
        conn.execute("ALTER TABLE native_chat_quota_intents ADD COLUMN quota_owner_json TEXT")
    conn.execute(
        "UPDATE db_schema_version SET version = 86 WHERE schema_name = ? AND version = 85", (schema_name,)
    )
    if get_version(conn) != 86:
        raise SchemaError("Chat/provenance catalog SQLite migration failed version verification.")  # noqa: TRY003


def migrate_postgres_chat_catalog(
    conn: Any,
    *,
    backend: DatabaseBackend,
    fingerprint_sql: str,
    schema_name: str,
    get_version: Callable[[Any], int],
) -> None:
    """Complete both catalogs under the schema lock with forced owner RLS."""
    from tldw_Server_API.app.core.DB_Management.backends.pg_rls_policies import (
        build_notes_provenance_receipts_rls_sql,
        build_notes_provenance_rls_sql,
        build_workspace_chat_startup_rls_sql,
    )
    from tldw_Server_API.app.core.DB_Management.chacha.note_provenance_store import (
        notes_provenance_receipts_schema_sql,
        notes_provenance_schema_sql,
    )
    from tldw_Server_API.app.core.DB_Management.chacha.workspace_chat_startup_schema import (
        workspace_chat_startup_schema_statements,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

    if not backend.table_exists("workspace_chat_startup_receipts", connection=conn):
        for statement in (
            *workspace_chat_startup_schema_statements(postgres=True),
            *build_workspace_chat_startup_rls_sql(),
        ):
            backend.execute(statement, connection=conn)
    ensure_message_order_postgres(backend, conn, fingerprint_sql)
    for statement in (
        notes_provenance_schema_sql(postgres=True),
        notes_provenance_receipts_schema_sql(),
        *build_notes_provenance_rls_sql(),
        *build_notes_provenance_receipts_rls_sql(),
    ):
        backend.execute(statement, connection=conn)
    columns = backend.get_table_info("native_chat_quota_intents", connection=conn)
    if not any(column["name"] == "quota_owner_json" for column in columns):
        backend.execute("ALTER TABLE native_chat_quota_intents ADD COLUMN quota_owner_json TEXT", connection=conn)
    backend.execute(
        "UPDATE db_schema_version SET version = %s WHERE schema_name = %s AND version = 91",
        (92, schema_name), connection=conn,
    )
    if get_version(conn) != 92:
        raise SchemaError("Chat/provenance catalog PostgreSQL migration failed version verification.")  # noqa: TRY003


def verify_postgres_message_order_privileges(backend: DatabaseBackend, conn: Any) -> None:
    """Fail at startup if a separate runtime role cannot use the order ledger."""
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import SchemaError

    allowed = backend.execute(
        """SELECT (
                NOT has_any_column_privilege(current_user, 'messages', 'SELECT,INSERT')
                OR (
                    has_table_privilege(current_user, 'message_insertion_order', 'SELECT')
                    AND has_column_privilege(current_user, 'messages', 'id', 'SELECT')
                    AND has_column_privilege(current_user, 'messages', 'client_id', 'SELECT')
                )
            ) AND (
                NOT has_any_column_privilege(current_user, 'messages', 'INSERT')
                OR (
                    has_column_privilege(current_user, 'message_insertion_order', 'message_id', 'INSERT')
                    AND COALESCE(has_sequence_privilege(
                        current_user, pg_get_serial_sequence('message_insertion_order', 'sequence'), 'USAGE'
                    ), false)
                    AND has_column_privilege(current_user, 'conversations', 'id', 'SELECT')
                    AND has_any_column_privilege(current_user, 'conversations', 'UPDATE')
                )
            )""",
        connection=conn,
    ).scalar
    if not allowed:
        raise SchemaError(  # noqa: TRY003
            "Ask the PostgreSQL schema owner to grant runtime access to message insertion order "
            "before enabling Chat reads or writes."
        )
