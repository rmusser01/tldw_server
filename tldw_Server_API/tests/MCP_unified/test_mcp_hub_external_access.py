from __future__ import annotations

import base64
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


@pytest.mark.asyncio
async def test_external_access_resolver_disables_inherited_server_at_assignment_level(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import (
        McpHubExternalAccessResolver,
    )

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

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
        config_json='{"websocket":{"url":"wss://docs.example/ws"}}',
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
    await repo.create_credential_binding(
        binding_target_type="profile",
        binding_target_id=str(profile["id"]),
        external_server_id="docs",
        credential_ref="server",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )
    await repo.create_credential_binding(
        binding_target_type="assignment",
        binding_target_id=str(assignment["id"]),
        external_server_id="docs",
        credential_ref="server",
        binding_mode="disable",
        usage_rules={},
        actor_id=1,
    )

    resolver = McpHubExternalAccessResolver(repo=repo)
    summary = await resolver.resolve(
        assignment_id=int(assignment["id"]),
        effective_policy={"capabilities": ["network.external"]},
    )

    assert summary["servers"][0]["disabled_by_assignment"] is True
