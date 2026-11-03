from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_mcp_hub_governance_pack_tables_exist_after_authnz_migrations_sqlite(
    tmp_path,
    monkeypatch, fresh_services
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(pool.db_path))

    rows = await pool.fetchall(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )
    names = {str(row["name"]) for row in rows}

    assert "mcp_governance_packs" in names
    assert "mcp_governance_pack_objects" in names

    governance_pack_columns = await pool.fetchall("PRAGMA table_info(mcp_governance_packs)")
    governance_pack_column_names = {str(row["name"]) for row in governance_pack_columns}
    assert "pack_id" in governance_pack_column_names
    assert "pack_version" in governance_pack_column_names
    assert "bundle_digest" in governance_pack_column_names
    assert "manifest_json" in governance_pack_column_names
    assert "normalized_ir_json" in governance_pack_column_names
    assert "owner_scope_type" in governance_pack_column_names
    assert "owner_scope_id" in governance_pack_column_names

    governance_object_columns = await pool.fetchall("PRAGMA table_info(mcp_governance_pack_objects)")
    governance_object_column_names = {str(row["name"]) for row in governance_object_columns}
    assert "governance_pack_id" in governance_object_column_names
    assert "object_type" in governance_object_column_names
    assert "object_id" in governance_object_column_names
    assert "source_object_id" in governance_object_column_names

    profile_columns = await pool.fetchall("PRAGMA table_info(mcp_permission_profiles)")
    profile_column_names = {str(row["name"]) for row in profile_columns}
    assert "is_immutable" in profile_column_names

    assignment_columns = await pool.fetchall("PRAGMA table_info(mcp_policy_assignments)")
    assignment_column_names = {str(row["name"]) for row in assignment_columns}
    assert "is_immutable" in assignment_column_names

    approval_columns = await pool.fetchall("PRAGMA table_info(mcp_approval_policies)")
    approval_column_names = {str(row["name"]) for row in approval_columns}
    assert "is_immutable" in approval_column_names


def test_governance_pack_upgrade_lineage_migration_normalizes_existing_active_installs(
    tmp_path,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.migrations import (
        migration_071_add_governance_pack_upgrade_lineage,
    )

    db_path = tmp_path / "legacy-users.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE mcp_governance_packs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pack_id TEXT NOT NULL,
                pack_version TEXT NOT NULL,
                pack_schema_version INTEGER NOT NULL DEFAULT 1,
                capability_taxonomy_version INTEGER NOT NULL DEFAULT 1,
                adapter_contract_version INTEGER NOT NULL DEFAULT 1,
                title TEXT NOT NULL,
                description TEXT,
                owner_scope_type TEXT NOT NULL DEFAULT 'user',
                owner_scope_id INTEGER,
                bundle_digest TEXT NOT NULL,
                manifest_json TEXT NOT NULL DEFAULT '{}',
                normalized_ir_json TEXT NOT NULL DEFAULT '{}',
                created_by INTEGER,
                updated_by INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            INSERT INTO mcp_governance_packs (
                pack_id, pack_version, title, owner_scope_type, owner_scope_id, bundle_digest
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("researcher-pack", "1.0.0", "Researcher Pack", "team", 21, "a" * 64),
        )
        conn.execute(
            """
            INSERT INTO mcp_governance_packs (
                pack_id, pack_version, title, owner_scope_type, owner_scope_id, bundle_digest
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("researcher-pack", "1.1.0", "Researcher Pack", "team", 21, "b" * 64),
        )
        conn.execute(
            """
            INSERT INTO mcp_governance_packs (
                pack_id, pack_version, title, owner_scope_type, owner_scope_id, bundle_digest
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            ("writer-pack", "1.0.0", "Writer Pack", "team", 22, "c" * 64),
        )
        conn.commit()

        migration_071_add_governance_pack_upgrade_lineage(conn)

        rows = conn.execute(
            """
            SELECT id, pack_version, is_active_install
            FROM mcp_governance_packs
            WHERE pack_id = ?
              AND owner_scope_type = ?
              AND owner_scope_id = ?
            ORDER BY id
            """,
            ("researcher-pack", "team", 21),
        ).fetchall()
        assert [(row[1], row[2]) for row in rows] == [("1.0.0", 0), ("1.1.0", 1)]

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                """
                UPDATE mcp_governance_packs
                SET is_active_install = 1
                WHERE pack_id = ?
                  AND pack_version = ?
                  AND owner_scope_type = ?
                  AND owner_scope_id = ?
                """,
                ("researcher-pack", "1.0.0", "team", 21),
            )
    finally:
        conn.close()
