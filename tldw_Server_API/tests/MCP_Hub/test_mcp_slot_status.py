from __future__ import annotations

import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
    ManagedSecretRefsRepo,
)
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_assignment_binding_reports_reauth_required_when_managed_secret_ref_requires_reauth(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_credential_broker_service import (
        McpCredentialBrokerService,
    )

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()

    managed_refs_repo = ManagedSecretRefsRepo(pool)
    await managed_refs_repo.ensure_tables()
    await managed_refs_repo.ensure_backend_registration(
        name="local_encrypted_v1",
        display_name="Local Encrypted",
    )

    profile = await repo.create_permission_profile(
        name="External GitHub",
        owner_scope_type="user",
        owner_scope_id=1,
        mode="custom",
        policy_document={"capabilities": ["network.external"]},
        actor_id=1,
    )
    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=1,
        profile_id=int(profile["id"]),
        inline_policy_document={"capabilities": ["network.external"]},
        approval_policy_id=None,
        actor_id=1,
        is_active=True,
    )
    await repo.upsert_external_server(
        server_id="github",
        name="GitHub",
        transport="websocket",
        config_json=json.dumps({"websocket": {"url": "wss://github.example/ws"}}),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="github",
        slot_name="bearer_token",
        display_name="Bearer token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )
    managed_ref = await managed_refs_repo.upsert_ref(
        backend_name="local_encrypted_v1",
        owner_scope_type="user",
        owner_scope_id=1,
        provider_key="github",
        backend_ref="user:1:github",
        metadata={"purpose": "mcp_slot"},
        status="reauth_required",
        created_by=1,
        updated_by=1,
    )
    await repo.create_credential_binding(
        binding_target_type="assignment",
        binding_target_id=str(assignment["id"]),
        external_server_id="github",
        slot_name="bearer_token",
        credential_ref=f"managed_secret_ref:{managed_ref['id']}",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    service = McpCredentialBrokerService(repo=repo)

    status = await service.get_slot_status(
        server_id="github",
        slot_name="bearer_token",
        assignment_id=int(assignment["id"]),
    )

    assert status["state"] == "reauth_required"
    assert status["managed_secret_ref_id"] == int(managed_ref["id"])
    assert status["credential_ref"] == f"managed_secret_ref:{managed_ref['id']}"


@pytest.mark.asyncio
async def test_assignment_binding_reports_missing_when_granted_slot_has_no_secret(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_credential_broker_service import (
        McpCredentialBrokerService,
    )

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
        name="External Docs",
        owner_scope_type="user",
        owner_scope_id=1,
        mode="custom",
        policy_document={"capabilities": ["network.external"]},
        actor_id=1,
    )
    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=1,
        profile_id=int(profile["id"]),
        inline_policy_document={"capabilities": ["network.external"]},
        approval_policy_id=None,
        actor_id=1,
        is_active=True,
    )
    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json=json.dumps({"websocket": {"url": "wss://docs.example/ws"}}),
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
    await repo.create_credential_binding(
        binding_target_type="assignment",
        binding_target_id=str(assignment["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref="slot",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    service = McpCredentialBrokerService(repo=repo)

    status = await service.get_slot_status(
        server_id="docs",
        slot_name="token_readonly",
        assignment_id=int(assignment["id"]),
    )

    assert status["state"] == "missing"
    assert status["managed_secret_ref_id"] is None


@pytest.mark.asyncio
async def test_assignment_binding_reports_approval_required_when_slot_secret_exists_but_slot_is_not_granted(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_credential_broker_service import (
        McpCredentialBrokerService,
    )

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
        name="External Docs",
        owner_scope_type="user",
        owner_scope_id=1,
        mode="custom",
        policy_document={"capabilities": ["network.external"]},
        actor_id=1,
    )
    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=1,
        profile_id=int(profile["id"]),
        inline_policy_document={"capabilities": ["network.external"]},
        approval_policy_id=None,
        actor_id=1,
        is_active=True,
    )
    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json=json.dumps({"websocket": {"url": "wss://docs.example/ws"}}),
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
    await repo.upsert_external_server_slot_secret(
        server_id="docs",
        slot_name="token_readonly",
        encrypted_blob='{"ciphertext":"abc"}',
        key_hint="cdef",
        actor_id=1,
    )

    service = McpCredentialBrokerService(repo=repo)

    status = await service.get_slot_status(
        server_id="docs",
        slot_name="token_readonly",
        assignment_id=int(assignment["id"]),
    )

    assert status["state"] == "approval_required"
