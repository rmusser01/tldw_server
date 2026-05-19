import sqlite3


def test_migration_084_scopes_account_lockouts_and_maps_legacy_rows() -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_001_create_users_table,
        migration_011_add_enhanced_auth_tables,
        migration_084_scope_account_lockouts_by_attempt_type,
    )

    conn = sqlite3.connect(":memory:")
    try:
        migration_001_create_users_table(conn)
        migration_011_add_enhanced_auth_tables(conn)
        conn.execute(
            """
            INSERT INTO account_lockouts (identifier, locked_until, reason)
            VALUES ('legacy-user', '2030-01-01T00:00:00+00:00', 'legacy')
            """
        )

        migration_084_scope_account_lockouts_by_attempt_type(conn)

        cols = {row[1] for row in conn.execute("PRAGMA table_info(account_lockouts)").fetchall()}
        assert "attempt_type" in cols

        row = conn.execute(
            """
            SELECT identifier, attempt_type, reason
            FROM account_lockouts
            WHERE identifier = 'legacy-user'
            """
        ).fetchone()
        assert row == ("legacy-user", "login", "legacy")
    finally:
        conn.close()
