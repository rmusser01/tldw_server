from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta, timezone

import pytest

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_repo_can_crud_acp_profile(tmp_path, monkeypatch) -> None:
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

    created = await repo.create_acp_profile(
        name="default-dev",
        owner_scope_type="org",
        owner_scope_id=42,
        profile_json='{"sandbox":"strict"}',
        actor_id=1,
    )
    assert created["name"] == "default-dev"

    fetched = await repo.get_acp_profile(int(created["id"]))
    assert fetched is not None
    assert fetched["name"] == "default-dev"
    assert fetched["owner_scope_type"] == "org"
    assert int(fetched["owner_scope_id"]) == 42

    listed = await repo.list_acp_profiles(owner_scope_type="org", owner_scope_id=42)
    assert len(listed) == 1
    assert listed[0]["name"] == "default-dev"

    updated = await repo.update_acp_profile(
        int(created["id"]),
        name="default-prod",
        profile_json='{"sandbox":"relaxed"}',
        is_active=False,
        actor_id=2,
    )
    assert updated is not None
    assert updated["name"] == "default-prod"
    assert updated["is_active"] is False

    deleted = await repo.delete_acp_profile(int(created["id"]))
    assert deleted is True
    missing = await repo.get_acp_profile(int(created["id"]))
    assert missing is None


@pytest.mark.asyncio
async def test_repo_external_server_secret_is_stored_separately(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="stdio",
        config_json='{"cmd":"npx"}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.upsert_external_secret(
        server_id="docs",
        encrypted_blob='{"ciphertext":"abc"}',
        key_hint="cdef",
        actor_id=1,
    )

    row = await repo.get_external_server("docs")
    assert row is not None
    assert row["id"] == "docs"
    assert row["secret_configured"] is True
    assert "encrypted_blob" not in row

    secret = await repo.get_external_secret("docs")
    assert secret is not None
    assert secret["server_id"] == "docs"
    assert secret["encrypted_blob"] == '{"ciphertext":"abc"}'


@pytest.mark.asyncio
async def test_repo_ensure_tables_requires_governance_pack_distribution_tables(tmp_path, monkeypatch) -> None:
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

    await pool.execute("DROP TABLE mcp_governance_pack_source_candidates", ())
    await pool.execute("DROP TABLE mcp_governance_pack_trust_policy", ())

    with pytest.raises(RuntimeError, match="mcp_governance_pack_source_candidates"):
        await repo.ensure_tables()


@pytest.mark.asyncio
async def test_repo_can_crud_permission_profile_and_policy_assignment(tmp_path, monkeypatch) -> None:
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

    profile = await repo.create_permission_profile(
        name="Read Current Folder",
        owner_scope_type="user",
        owner_scope_id=7,
        mode="custom",
        policy_document={
            "capabilities": ["filesystem.read"],
            "path_scope": "cwd_descendants",
        },
        actor_id=7,
        description="Read-only profile for the current workspace",
        is_active=True,
    )
    assert profile["name"] == "Read Current Folder"
    assert profile["owner_scope_type"] == "user"
    assert int(profile["owner_scope_id"]) == 7
    assert profile["policy_document"]["capabilities"] == ["filesystem.read"]

    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=int(profile["id"]),
        inline_policy_document={"approval_mode": "ask_outside_profile"},
        approval_policy_id=None,
        actor_id=7,
        is_active=True,
    )
    assert assignment["target_type"] == "persona"
    assert assignment["target_id"] == "researcher"
    assert int(assignment["profile_id"]) == int(profile["id"])
    assert assignment["inline_policy_document"]["approval_mode"] == "ask_outside_profile"

    fetched = await repo.get_policy_assignment(int(assignment["id"]))
    assert fetched is not None
    assert fetched["target_type"] == "persona"
    assert fetched["target_id"] == "researcher"

    listed = await repo.list_policy_assignments(owner_scope_type="user", owner_scope_id=7)
    assert len(listed) == 1
    assert listed[0]["target_type"] == "persona"
    assert listed[0]["inline_policy_document"]["approval_mode"] == "ask_outside_profile"

    updated = await repo.update_policy_assignment(
        int(assignment["id"]),
        inline_policy_document={"approval_mode": "ask_every_time"},
        actor_id=8,
    )
    assert updated is not None
    assert updated["inline_policy_document"]["approval_mode"] == "ask_every_time"

    deleted = await repo.delete_policy_assignment(int(assignment["id"]))
    assert deleted is True
    missing = await repo.get_policy_assignment(int(assignment["id"]))
    assert missing is None


@pytest.mark.asyncio
async def test_repo_update_policy_assignment_can_clear_nullable_fields(tmp_path, monkeypatch) -> None:
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

    profile = await repo.create_permission_profile(
        name="Runtime Profile",
        owner_scope_type="user",
        owner_scope_id=7,
        mode="custom",
        policy_document={"capabilities": ["filesystem.read"]},
        actor_id=7,
    )
    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=int(profile["id"]),
        inline_policy_document={"approval_mode": "ask_outside_profile"},
        approval_policy_id=9,
        actor_id=7,
        is_active=True,
    )

    updated = await repo.update_policy_assignment(
        int(assignment["id"]),
        profile_id=None,
        approval_policy_id=None,
        actor_id=8,
    )

    assert updated is not None
    assert updated["profile_id"] is None
    assert updated["approval_policy_id"] is None


@pytest.mark.asyncio
async def test_repo_can_crud_workspace_set_objects_and_named_assignment_source(tmp_path, monkeypatch) -> None:
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

    workspace_set = await repo.create_workspace_set_object(
        name="Primary Research Roots",
        owner_scope_type="user",
        owner_scope_id=7,
        actor_id=7,
        description="Reusable workspace set",
        is_active=True,
    )
    assert workspace_set["name"] == "Primary Research Roots"
    assert workspace_set["owner_scope_type"] == "user"
    assert int(workspace_set["owner_scope_id"]) == 7

    member = await repo.add_workspace_set_member(
        int(workspace_set["id"]),
        workspace_id="workspace-alpha",
        actor_id=7,
    )
    assert member["workspace_id"] == "workspace-alpha"

    listed_members = await repo.list_workspace_set_members(int(workspace_set["id"]))
    assert [row["workspace_id"] for row in listed_members] == ["workspace-alpha"]

    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=None,
        path_scope_object_id=None,
        workspace_source_mode="named",
        workspace_set_object_id=int(workspace_set["id"]),
        inline_policy_document={"approval_mode": "ask_outside_profile"},
        approval_policy_id=None,
        actor_id=7,
        is_active=True,
    )
    assert assignment["workspace_source_mode"] == "named"
    assert int(assignment["workspace_set_object_id"]) == int(workspace_set["id"])

    updated_assignment = await repo.update_policy_assignment(
        int(assignment["id"]),
        workspace_source_mode="inline",
        workspace_set_object_id=None,
        actor_id=8,
    )
    assert updated_assignment is not None
    assert updated_assignment["workspace_source_mode"] == "inline"
    assert updated_assignment["workspace_set_object_id"] is None

    deleted_member = await repo.delete_workspace_set_member(
        int(workspace_set["id"]),
        "workspace-alpha",
    )
    assert deleted_member is True

    deleted_workspace_set = await repo.delete_workspace_set_object(int(workspace_set["id"]))
    assert deleted_workspace_set is True


@pytest.mark.asyncio
async def test_repo_can_crud_shared_workspaces(tmp_path, monkeypatch) -> None:
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

    created = await repo.create_shared_workspace_entry(
        workspace_id="shared-docs",
        display_name="Shared Docs",
        absolute_root="/srv/shared/docs",
        owner_scope_type="team",
        owner_scope_id=21,
        actor_id=9,
        is_active=True,
    )
    assert created["workspace_id"] == "shared-docs"
    assert created["display_name"] == "Shared Docs"
    assert created["absolute_root"] == "/srv/shared/docs"
    assert created["owner_scope_type"] == "team"
    assert int(created["owner_scope_id"]) == 21

    listed = await repo.list_shared_workspace_entries(owner_scope_type="team", owner_scope_id=21)
    assert len(listed) == 1
    assert listed[0]["workspace_id"] == "shared-docs"

    updated = await repo.update_shared_workspace_entry(
        int(created["id"]),
        display_name="Shared Docs Root",
        is_active=False,
        actor_id=10,
    )
    assert updated is not None
    assert updated["display_name"] == "Shared Docs Root"
    assert updated["is_active"] is False

    deleted = await repo.delete_shared_workspace_entry(int(created["id"]))
    assert deleted is True


@pytest.mark.asyncio
async def test_repo_can_crud_approval_policy_and_match_active_decision(tmp_path, monkeypatch) -> None:
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

    policy = await repo.create_approval_policy(
        name="Outside Profile",
        description="Require approval for tools outside the active profile",
        owner_scope_type="user",
        owner_scope_id=7,
        mode="ask_outside_profile",
        rules={"duration_options": ["once", "session"]},
        actor_id=7,
        is_active=True,
    )
    assert policy["name"] == "Outside Profile"
    assert policy["mode"] == "ask_outside_profile"
    assert policy["rules"]["duration_options"] == ["once", "session"]

    listed = await repo.list_approval_policies(owner_scope_type="user", owner_scope_id=7)
    assert len(listed) == 1
    assert listed[0]["name"] == "Outside Profile"

    updated = await repo.update_approval_policy(
        int(policy["id"]),
        mode="temporary_elevation_allowed",
        rules={"duration_options": ["session"]},
        actor_id=8,
    )
    assert updated is not None
    assert updated["mode"] == "temporary_elevation_allowed"
    assert updated["rules"]["duration_options"] == ["session"]

    expires_at = datetime.now(timezone.utc) + timedelta(minutes=15)
    decision = await repo.create_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-1",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        decision="approved",
        expires_at=expires_at,
        consume_on_match=False,
        actor_id=7,
    )
    assert decision["decision"] == "approved"
    assert decision["consume_on_match"] is False

    matched = await repo.find_active_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-1",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        decision="approved",
        now=datetime.now(timezone.utc),
    )
    assert matched is not None
    assert matched["decision"] == "approved"
    assert matched["tool_name"] == "Bash"

    expired = await repo.expire_approval_decision(
        int(decision["id"]),
        expires_at=datetime.now(timezone.utc),
    )
    assert expired is not None

    after_consume = await repo.find_active_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-1",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        decision="approved",
        now=datetime.now(timezone.utc),
    )
    assert after_consume is None

    single_use = await repo.create_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-1",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        decision="approved",
        expires_at=None,
        consume_on_match=True,
        actor_id=7,
    )
    assert single_use["consume_on_match"] is True

    consumed = await repo.consume_active_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-1",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        now=datetime.now(timezone.utc),
    )
    assert consumed is not None
    assert consumed["id"] == single_use["id"]
    assert consumed["consumed_at"] is not None

    consumed_again = await repo.consume_active_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-1",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        now=datetime.now(timezone.utc),
    )
    assert consumed_again is None

    wildcard_candidate = await repo.create_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id=None,
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        decision="approved",
        expires_at=datetime.now(timezone.utc) + timedelta(minutes=5),
        consume_on_match=False,
        actor_id=7,
    )
    assert wildcard_candidate["conversation_id"] is None

    should_not_match_foreign_conversation = await repo.find_active_approval_decision(
        approval_policy_id=int(policy["id"]),
        context_key="user:7|persona:researcher",
        conversation_id="sess-2",
        tool_name="Bash",
        scope_key="tool:Bash|command:git-status",
        decision="approved",
        now=datetime.now(timezone.utc),
    )
    assert should_not_match_foreign_conversation is None

    deleted = await repo.delete_approval_policy(int(policy["id"]))
    assert deleted is True


@pytest.mark.asyncio
async def test_repo_credential_binding_is_unique_per_target_and_server(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"url":"wss://docs.example/ws"}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    first = await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id="11",
        external_server_id="docs",
        credential_ref="server",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )
    assert first["binding_mode"] == "grant"

    with pytest.raises(Exception):
        await repo.create_credential_binding(
            binding_target_type="profile",
            binding_target_id="11",
            external_server_id="docs",
            credential_ref="server",
            binding_mode="grant",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_rejects_disable_binding_for_profile_target(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"url":"wss://docs.example/ws"}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    with pytest.raises(ValueError):
        await repo.create_credential_binding(
            binding_target_type="profile",
            binding_target_id="11",
            external_server_id="docs",
            credential_ref="server",
            binding_mode="disable",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_rejects_explicit_server_credential_ref_for_disable_binding(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"url":"wss://docs.example/ws"}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    with pytest.raises(ValueError, match="disable bindings may not store explicit credential refs"):
        await repo.create_credential_binding(
            binding_target_type="assignment",
            binding_target_id="11",
            external_server_id="docs",
            credential_ref="managed_secret_ref:17",
            binding_mode="disable",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_rejects_explicit_server_credential_ref_for_disable_binding(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"url":"wss://docs.example/ws"}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    with pytest.raises(ValueError, match="disable bindings may not store explicit credential refs"):
        await repo.create_credential_binding(
            binding_target_type="assignment",
            binding_target_id="11",
            external_server_id="docs",
            credential_ref="managed:shared-docs-token",
            binding_mode="disable",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_credential_slot_name_is_unique_per_server(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"websocket":{"url":"wss://docs.example/ws"}}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    first = await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )
    assert first["slot_name"] == "token_readonly"

    with pytest.raises(ValueError):
        await repo.create_external_server_credential_slot(
            server_id="docs",
            slot_name="token_readonly",
            display_name="Duplicate read-only token",
            secret_kind="bearer_token",
            privilege_class="read",
            is_required=True,
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_credential_binding_is_unique_per_target_server_and_slot(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"websocket":{"url":"wss://docs.example/ws"}}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )

    binding = await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id="11",
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref="slot",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )
    assert binding["slot_name"] == "token_readonly"

    with pytest.raises(ValueError):
        await repo.create_credential_binding(
            binding_target_type="profile",
            binding_target_id="11",
            external_server_id="docs",
            slot_name="token_readonly",
            credential_ref="slot",
            binding_mode="grant",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_rejects_slot_binding_for_unknown_slot(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"websocket":{"url":"wss://docs.example/ws"}}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    with pytest.raises(ValueError):
        await repo.create_credential_binding(
            binding_target_type="assignment",
            binding_target_id="11",
            external_server_id="docs",
            slot_name="token_write",
            credential_ref="slot",
            binding_mode="grant",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_rejects_explicit_slot_credential_ref_for_disable_binding(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"websocket":{"url":"wss://docs.example/ws"}}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )

    with pytest.raises(ValueError, match="disable bindings may not store explicit credential refs"):
        await repo.create_credential_binding(
            binding_target_type="assignment",
            binding_target_id="11",
            external_server_id="docs",
            slot_name="token_readonly",
            credential_ref="managed_secret_ref:23",
            binding_mode="disable",
            usage_rules={},
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_repo_rejects_explicit_slot_credential_ref_for_disable_binding(tmp_path, monkeypatch) -> None:
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

    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json='{"websocket":{"url":"wss://docs.example/ws"}}',
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )

    with pytest.raises(ValueError, match="disable bindings may not store explicit credential refs"):
        await repo.create_credential_binding(
            binding_target_type="assignment",
            binding_target_id="11",
            external_server_id="docs",
            slot_name="token_readonly",
            credential_ref="managed:readonly-token",
            binding_mode="disable",
            usage_rules={},
            actor_id=1,
        )
