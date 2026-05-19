import sqlite3

import pytest


pytestmark = pytest.mark.unit


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cursor.fetchall()}


def _sqlite_column(conn: sqlite3.Connection, table: str, column_name: str) -> tuple:
    cursor = conn.execute(f"PRAGMA table_info({table})")
    return next(row for row in cursor.fetchall() if row[1] == column_name)


def test_migration_017_extend_api_keys_virtual_adds_columns() -> None:


    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_003_create_api_keys_table,
        migration_004_create_api_key_audit_log,
        migration_016_create_orgs_teams,
        migration_017_extend_api_keys_virtual,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_003_create_api_keys_table(conn)
        migration_004_create_api_key_audit_log(conn)
        migration_016_create_orgs_teams(conn)
        migration_017_extend_api_keys_virtual(conn)

        cols = _sqlite_columns(conn, "api_keys")
        for expected in (
            "is_virtual",
            "parent_key_id",
            "org_id",
            "team_id",
            "llm_budget_day_tokens",
            "llm_budget_month_tokens",
            "llm_budget_day_usd",
            "llm_budget_month_usd",
            "llm_allowed_endpoints",
            "llm_allowed_providers",
            "llm_allowed_models",
        ):
            assert expected in cols
    finally:
        conn.close()


def test_migration_003_creates_api_keys_without_scope_default() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_003_create_api_keys_table,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_003_create_api_keys_table(conn)

        assert _sqlite_column(conn, "api_keys", "scope")[4] is None
    finally:
        conn.close()


def test_migration_003_adds_scope_without_default_to_legacy_api_keys_table() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import migration_003_create_api_keys_table

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute(
            """
            CREATE TABLE api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                expires_at TIMESTAMP
            )
            """
        )
        conn.commit()

        migration_003_create_api_keys_table(conn)

        assert _sqlite_column(conn, "api_keys", "scope")[4] is None
    finally:
        conn.close()


def test_migration_085_removes_api_keys_scope_default() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_003_create_api_keys_table,
        migration_004_create_api_key_audit_log,
        migration_085_remove_api_keys_scope_default,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_003_create_api_keys_table(conn)
        migration_004_create_api_key_audit_log(conn)

        migration_085_remove_api_keys_scope_default(conn)

        assert _sqlite_column(conn, "api_keys", "scope")[4] is None
    finally:
        conn.close()


def test_migration_085_preserves_rows_indexes_and_audit_refs() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_003_create_api_keys_table,
        migration_004_create_api_key_audit_log,
        migration_085_remove_api_keys_scope_default,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_003_create_api_keys_table(conn)
        migration_004_create_api_key_audit_log(conn)

        conn.execute(
            """
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            ("migration-user", "migration@example.com", "hashed"),
        )
        user_id = conn.execute("SELECT id FROM users").fetchone()[0]
        conn.execute(
            """
            INSERT INTO api_keys (
                user_id, key_hash, key_id, key_prefix, name, scope, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, "key-hash", "kid-1", "pref", "Primary", "write", "active"),
        )
        api_key_id = conn.execute("SELECT id FROM api_keys").fetchone()[0]
        conn.execute(
            """
            INSERT INTO api_key_audit_log (api_key_id, action, user_id, details)
            VALUES (?, ?, ?, ?)
            """,
            (api_key_id, "created", user_id, '{"source":"test"}'),
        )
        conn.commit()

        migration_085_remove_api_keys_scope_default(conn)

        assert _sqlite_column(conn, "api_keys", "scope")[4] is None
        assert conn.execute(
            "SELECT id, key_hash, key_id, key_prefix, name, scope FROM api_keys"
        ).fetchone() == (api_key_id, "key-hash", "kid-1", "pref", "Primary", "write")
        assert conn.execute(
            "SELECT api_key_id, action, user_id, details FROM api_key_audit_log"
        ).fetchone() == (api_key_id, "created", user_id, '{"source":"test"}')

        index_rows = conn.execute("PRAGMA index_list(api_keys)").fetchall()
        index_names = {row[1] for row in index_rows}
        assert "idx_api_keys_user_id" in index_names
        assert "idx_api_keys_key_hash" in index_names
        assert "idx_api_keys_key_id" in index_names
        assert "idx_api_keys_status" in index_names
        assert "idx_api_keys_expires_at" in index_names
    finally:
        conn.close()
