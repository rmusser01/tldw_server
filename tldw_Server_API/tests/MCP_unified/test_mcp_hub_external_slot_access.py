from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
async def test_external_access_resolver_is_slot_aware(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
        McpHubExternalAccessResolver,
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
        config_json=json.dumps(
            {
                "websocket": {"url": "wss://docs.example/ws"},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "header",
                            "target_name": "Authorization",
                            "prefix": "Bearer ",
                            "required": True,
                        }
                    ],
                },
            }
        ),
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
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_write",
        display_name="Write token",
        secret_kind="api_key",
        privilege_class="write",
        is_required=False,
        actor_id=1,
    )
    await repo.upsert_external_server_slot_secret(
        server_id="docs",
        slot_name="token_readonly",
        encrypted_blob='{"ciphertext":"abc"}',
        key_hint="cdef",
        actor_id=1,
    )
    await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id=str(profile["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref="slot",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )
    await repo.create_credential_binding(
        binding_target_type="assignment",
        binding_target_id=str(assignment["id"]),
        external_server_id="docs",
        slot_name="token_write",
        credential_ref="slot",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )
    await repo.create_credential_binding(
        binding_target_type="assignment",
        binding_target_id=str(assignment["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref="slot",
        binding_mode="disable",
        usage_rules={},
        actor_id=1,
    )

    resolver = McpHubExternalAccessResolver(repo=repo)
    summary = await resolver.resolve(
        assignment_id=int(assignment["id"]),
        effective_policy={"capabilities": ["network.external"]},
    )

    assert summary["servers"][0]["server_id"] == "docs"
    assert summary["servers"][0]["slots"][0]["slot_name"] == "token_readonly"
    assert summary["servers"][0]["slots"][0]["blocked_reason"] == "disabled_by_assignment"
    assert summary["servers"][0]["slots"][1]["slot_name"] == "token_write"
    assert summary["servers"][0]["slots"][1]["granted_by"] == "assignment"


@pytest.mark.asyncio
async def test_external_access_resolver_marks_missing_slot_secret(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
        McpHubExternalAccessResolver,
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
        config_json=json.dumps(
            {
                "websocket": {"url": "wss://docs.example/ws"},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "header",
                            "target_name": "Authorization",
                            "prefix": "Bearer ",
                            "required": True,
                        }
                    ],
                },
            }
        ),
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
        binding_target_type="profile",
        binding_target_id=str(profile["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref="slot",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    resolver = McpHubExternalAccessResolver(repo=repo)
    summary = await resolver.resolve(
        assignment_id=int(assignment["id"]),
        effective_policy={"capabilities": ["network.external"]},
    )

    assert summary["servers"][0]["slots"][0]["slot_name"] == "token_readonly"
    assert summary["servers"][0]["slots"][0]["blocked_reason"] == "missing_secret"


@pytest.mark.asyncio
async def test_external_access_resolver_reports_requested_slot_bundle_state(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
        McpHubExternalAccessResolver,
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
        config_json=json.dumps(
            {
                "websocket": {"url": "wss://docs.example/ws"},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "header",
                            "target_name": "Authorization",
                            "prefix": "Bearer ",
                            "required": True,
                        },
                        {
                            "slot_name": "token_write",
                            "target_type": "header",
                            "target_name": "X-Write-Key",
                            "required": True,
                        },
                    ],
                },
            }
        ),
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
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_write",
        display_name="Write token",
        secret_kind="api_key",
        privilege_class="write",
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
    await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id=str(profile["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref="slot",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    resolver = McpHubExternalAccessResolver(repo=repo)
    summary = await resolver.resolve(
        assignment_id=int(assignment["id"]),
        effective_policy={"capabilities": ["network.external"]},
    )

    server = summary["servers"][0]
    assert server["requested_slots"] == ["token_readonly", "token_write"]
    assert server["bound_slots"] == ["token_readonly"]
    assert server["missing_bound_slots"] == ["token_write"]
    assert server["missing_secret_slots"] == []


@pytest.mark.asyncio
async def test_external_access_resolver_treats_expired_managed_secret_ref_as_missing_without_backend_lookup(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
        ManagedSecretRefsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.services import mcp_hub_external_access_resolver as resolver_module
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
        McpHubExternalAccessResolver,
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
    users_db = UsersDB(pool)
    await users_db.initialize()
    await users_db.create_user(
        username="resolver-expired",
        email="resolver-expired@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_verified=True,
    )

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
        config_json=json.dumps(
            {
                "websocket": {"url": "wss://docs.example/ws"},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "header",
                            "target_name": "Authorization",
                            "prefix": "Bearer ",
                            "required": True,
                        }
                    ],
                },
            }
        ),
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
    managed_ref = await managed_refs_repo.upsert_ref(
        backend_name="local_encrypted_v1",
        owner_scope_type="user",
        owner_scope_id=1,
        provider_key="docs",
        backend_ref="user:1:docs",
        metadata={"purpose": "mcp_slot"},
        status="active",
        expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
        created_by=1,
        updated_by=1,
    )
    await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id=str(profile["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref=f"managed_secret_ref:{managed_ref['id']}",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    def _unexpected_backend(*_args, **_kwargs):  # noqa: ANN001, ANN002
        raise AssertionError("Expired managed secret refs should not query the backend")

    monkeypatch.setattr(resolver_module, "get_secret_backend", _unexpected_backend)

    resolver = McpHubExternalAccessResolver(repo=repo)
    summary = await resolver.resolve(
        assignment_id=int(assignment["id"]),
        effective_policy={"capabilities": ["network.external"]},
    )

    server = summary["servers"][0]
    assert server["secret_available"] is False
    assert server["runtime_executable"] is False
    assert server["missing_secret_slots"] == ["token_readonly"]
    assert server["blocked_reason"] == "required_slot_secret_missing"


@pytest.mark.asyncio
async def test_external_access_resolver_caches_managed_secret_readiness_per_ref(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.managed_secret_refs_repo import (
        ManagedSecretRefsRepo,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB
    from tldw_Server_API.app.services import mcp_hub_external_access_resolver as resolver_module
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
        McpHubExternalAccessResolver,
    )

    class _CountingBackend:
        def __init__(self) -> None:
            self.calls: list[int] = []

        async def describe_status(self, secret_ref_id: int) -> dict[str, str]:
            self.calls.append(int(secret_ref_id))
            return {"state": "ready"}

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
    users_db = UsersDB(pool)
    await users_db.initialize()
    await users_db.create_user(
        username="resolver-cached",
        email="resolver-cached@example.com",
        password_hash="hashed-password",
        role="user",
        is_active=True,
        is_verified=True,
    )

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
        config_json=json.dumps(
            {
                "websocket": {"url": "wss://docs.example/ws"},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "header",
                            "target_name": "Authorization",
                            "prefix": "Bearer ",
                            "required": True,
                        },
                        {
                            "slot_name": "token_write",
                            "target_type": "header",
                            "target_name": "X-Write-Key",
                            "required": True,
                        },
                    ],
                },
            }
        ),
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
    await repo.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_write",
        display_name="Write token",
        secret_kind="api_key",
        privilege_class="write",
        is_required=True,
        actor_id=1,
    )
    managed_ref = await managed_refs_repo.upsert_ref(
        backend_name="local_encrypted_v1",
        owner_scope_type="user",
        owner_scope_id=1,
        provider_key="docs",
        backend_ref="user:1:docs",
        metadata={"purpose": "mcp_slot"},
        status="active",
        created_by=1,
        updated_by=1,
    )
    await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id=str(profile["id"]),
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref=f"managed_secret_ref:{managed_ref['id']}",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )
    await repo.create_credential_binding(
        binding_target_type="assignment",
        binding_target_id=str(assignment["id"]),
        external_server_id="docs",
        slot_name="token_write",
        credential_ref=f"managed_secret_ref:{managed_ref['id']}",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    counting_backend = _CountingBackend()
    monkeypatch.setattr(
        resolver_module,
        "get_secret_backend",
        lambda *_args, **_kwargs: counting_backend,
    )

    resolver = McpHubExternalAccessResolver(repo=repo)
    summary = await resolver.resolve(
        assignment_id=int(assignment["id"]),
        effective_policy={"capabilities": ["network.external"]},
    )

    server = summary["servers"][0]
    assert server["bound_slots"] == ["token_readonly", "token_write"]
    assert server["missing_secret_slots"] == []
    assert counting_backend.calls == [int(managed_ref["id"])]
