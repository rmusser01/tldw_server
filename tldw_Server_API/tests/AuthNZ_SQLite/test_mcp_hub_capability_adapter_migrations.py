from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_mcp_hub_capability_adapter_tables_exist_after_authnz_migrations_sqlite(
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

    rows = await pool.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
    names = {str(row["name"]) for row in rows}

    assert "mcp_capability_adapter_mappings" in names

    mapping_columns = await pool.fetchall("PRAGMA table_info(mcp_capability_adapter_mappings)")
    mapping_column_names = {str(row["name"]) for row in mapping_columns}
    assert "mapping_id" in mapping_column_names
    assert "owner_scope_type" in mapping_column_names
    assert "owner_scope_id" in mapping_column_names
    assert "capability_name" in mapping_column_names
    assert "adapter_contract_version" in mapping_column_names
    assert "resolved_policy_document_json" in mapping_column_names
    assert "supported_environment_requirements_json" in mapping_column_names
    assert "is_active" in mapping_column_names

    index_rows = await pool.fetchall("PRAGMA index_list(mcp_capability_adapter_mappings)")
    index_names = {str(row["name"]) for row in index_rows}
    assert "idx_mcp_capability_adapter_mappings_scope" in index_names
    assert "uq_mcp_capability_adapter_mappings_mapping_id" in index_names
    assert "uq_mcp_capability_adapter_mappings_active_scope_capability" in index_names
