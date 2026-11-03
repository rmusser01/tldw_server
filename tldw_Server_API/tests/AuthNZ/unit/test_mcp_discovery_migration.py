"""MCP discovery persistence preserves real AuthNZ databases through the current migration chain."""

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    apply_authnz_migrations,
    ensure_authnz_tables,
    get_authnz_migrations,
)
from tldw_Server_API.app.core.DB_Management.migrations import Migration, MigrationManager


@pytest.mark.parametrize("source_version", [100, 103, 104, 105])
def test_upgrade_preserves_external_server_and_adds_nullable_discovery(tmp_path, source_version):
    path = tmp_path / "users.db"
    apply_authnz_migrations(path, target_version=source_version)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO mcp_external_servers (id, name, transport, config_json) VALUES (?, ?, ?, ?)",
            ("preserved", "Existing server", "stdio", '{"stdio":{"command":"/usr/bin/true"}}'),
        )
        before = conn.execute(
            "SELECT id, name, transport, config_json, updated_at FROM mcp_external_servers"
        ).fetchone()
    ensure_authnz_tables(path)
    ensure_authnz_tables(path)
    with sqlite3.connect(path) as conn:
        after = conn.execute("SELECT id, name, transport, config_json, updated_at FROM mcp_external_servers").fetchone()
        state = conn.execute(
            "SELECT last_discovery_at, last_successful_discovery_at, last_error_category FROM mcp_external_servers"
        ).fetchone()
    assert after == before  # nosec B101 -- real upgrade preserves server configuration and revision.
    assert state == (None, None, None)  # nosec B101 -- upgrade cannot manufacture historical discovery.


def test_discovery_migration_follows_existing_versions_without_collisions():
    versions = [migration.version for migration in get_authnz_migrations()]
    assert versions == sorted(set(versions))  # nosec B101 -- duplicate IDs can skip or corrupt a real upgrade.
    assert versions[-6:] == [101, 102, 103, 104, 105, 106]  # nosec B101 -- preserve admin, native quota, semantic and provider migrations before MCP discovery.


@pytest.mark.parametrize("has_server_table", [True, False], ids=["with-mcp", "without-mcp"])
def test_old_owned_104_upgrades_usage_attribution_without_losing_discovery(tmp_path, has_server_table):
    path = tmp_path / "old-owned-104.db"
    apply_authnz_migrations(path, target_version=103)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO llm_usage_log (request_id, provider, model, total_tokens) VALUES (?, ?, ?, ?)",
            ("legacy-call", "openai", "retained-model", 17),
        )
        if has_server_table:
            conn.execute(
                "INSERT INTO mcp_external_servers (id, name, transport, config_json) VALUES (?, ?, ?, ?)",
                ("retained", "Retained MCP", "stdio", '{"stdio":{"command":"/usr/bin/true"}}'),
            )
        else:
            conn.execute("DROP TABLE mcp_external_servers")

    def legacy_discovery_104(conn):
        # Exact accepted candidate 104 shape, before credential attribution used104 upstream.
        if has_server_table:
            conn.execute("ALTER TABLE mcp_external_servers ADD COLUMN last_discovery_at TIMESTAMP")
            conn.execute("ALTER TABLE mcp_external_servers ADD COLUMN last_successful_discovery_at TIMESTAMP")
            conn.execute("ALTER TABLE mcp_external_servers ADD COLUMN last_error_category TEXT")
            conn.execute(
                "UPDATE mcp_external_servers SET last_discovery_at=?, last_successful_discovery_at=?, last_error_category=?",
                ("2026-10-20T12:00:00Z", "2026-10-19T12:00:00Z", "unreachable"),
            )

    old_manager = MigrationManager(path)
    old_manager.add_migration(Migration(104, "Persist external MCP discovery outcomes", legacy_discovery_104))
    old_manager.migrate()
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)")}
        assert "credential_source" not in columns  # nosec B101 -- The fixture is a real pre-attribution owned104 database.
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 104  # nosec B101 -- Migration selection must encounter the historical ID collision.
        before = conn.execute("SELECT request_id, provider, model, total_tokens FROM llm_usage_log").fetchall()
        server_before = conn.execute("SELECT * FROM mcp_external_servers").fetchall() if has_server_table else None

    ensure_authnz_tables(path)
    ensure_authnz_tables(path)
    with sqlite3.connect(path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(llm_usage_log)")}
        assert "credential_source" in columns  # nosec B101 -- Real upgrade must reconcile skipped upstream104, even without an MCP table.
        after = conn.execute("SELECT request_id, provider, model, total_tokens FROM llm_usage_log").fetchall()
        assert after == before  # nosec B101 -- Existing usage facts cannot be rewritten during compatibility repair.
        assert conn.execute("SELECT credential_source FROM llm_usage_log").fetchall() == [(None,)]  # nosec B101 -- Historical key ownership remains unknown.
        conn.execute(
            "INSERT INTO llm_usage_log (request_id, provider, model, total_tokens, credential_source) VALUES (?, ?, ?, ?, ?)",
            ("new-own-key-call", "openai", "retained-model", 5, "user"),
        )
        assert conn.execute("SELECT credential_source FROM llm_usage_log WHERE request_id='new-own-key-call'").fetchone() == ("user",)  # nosec B101 -- New settlement writes can retain own-key attribution.
        if has_server_table:
            assert conn.execute("SELECT * FROM mcp_external_servers").fetchall() == server_before  # nosec B101 -- Configuration, revision and real discovery history survive intact.
        assert conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 106  # nosec B101 -- Normal registry105/106 completes without renumbering incoming migrations.
