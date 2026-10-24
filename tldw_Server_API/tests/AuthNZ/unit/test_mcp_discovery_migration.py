"""MCP discovery persistence preserves real AuthNZ databases through the current migration chain."""

import sqlite3

import pytest

from tldw_Server_API.app.core.AuthNZ.migrations import (
    apply_authnz_migrations,
    ensure_authnz_tables,
    get_authnz_migrations,
)


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
