from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_repo_can_crud_capability_adapter_mapping(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()

    created = await repo.create_capability_adapter_mapping(
        mapping_id="research.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["web.search"]},
        supported_environment_requirements=["local_mapping_required"],
        actor_id=1,
        title="Global research mapping",
        description="Maps portable research capability to MCP search tools",
        is_active=True,
    )

    assert created["mapping_id"] == "research.global"
    assert created["capability_name"] == "tool.invoke.research"
    assert created["resolved_policy_document"]["allowed_tools"] == ["web.search"]
    assert created["supported_environment_requirements"] == ["local_mapping_required"]
    assert created["is_active"] is True

    fetched = await repo.get_capability_adapter_mapping(int(created["id"]))
    assert fetched is not None
    assert fetched["mapping_id"] == "research.global"

    listed = await repo.list_capability_adapter_mappings(owner_scope_type="global", owner_scope_id=None)
    assert len(listed) == 1
    assert listed[0]["mapping_id"] == "research.global"

    updated = await repo.update_capability_adapter_mapping(
        int(created["id"]),
        title="Updated global research mapping",
        resolved_policy_document={"allowed_tools": ["web.search", "docs.search"]},
        supported_environment_requirements=["local_mapping_required", "workspace_bounded_read"],
        actor_id=2,
    )
    assert updated is not None
    assert updated["title"] == "Updated global research mapping"
    assert updated["resolved_policy_document"]["allowed_tools"] == ["web.search", "docs.search"]
    assert updated["supported_environment_requirements"] == [
        "local_mapping_required",
        "workspace_bounded_read",
    ]

    deleted = await repo.delete_capability_adapter_mapping(int(created["id"]))
    assert deleted is True
    missing = await repo.get_capability_adapter_mapping(int(created["id"]))
    assert missing is None


@pytest.mark.asyncio
async def test_repo_rejects_duplicate_active_capability_adapter_mapping_per_scope(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()

    created = await repo.create_capability_adapter_mapping(
        mapping_id="research.global",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["web.search"]},
        supported_environment_requirements=["local_mapping_required"],
        actor_id=1,
        is_active=True,
    )
    assert created["is_active"] is True

    with pytest.raises(Exception):
        await repo.create_capability_adapter_mapping(
            mapping_id="research.global.duplicate",
            owner_scope_type="global",
            owner_scope_id=None,
            capability_name="tool.invoke.research",
            adapter_contract_version=1,
            resolved_policy_document={"allowed_tools": ["docs.search"]},
            supported_environment_requirements=[],
            actor_id=1,
            is_active=True,
        )

    deactivated = await repo.update_capability_adapter_mapping(
        int(created["id"]),
        is_active=False,
        actor_id=2,
    )
    assert deactivated is not None
    assert deactivated["is_active"] is False

    replacement = await repo.create_capability_adapter_mapping(
        mapping_id="research.global.replacement",
        owner_scope_type="global",
        owner_scope_id=None,
        capability_name="tool.invoke.research",
        adapter_contract_version=1,
        resolved_policy_document={"allowed_tools": ["docs.search"]},
        supported_environment_requirements=[],
        actor_id=2,
        is_active=True,
    )
    assert replacement["mapping_id"] == "research.global.replacement"
