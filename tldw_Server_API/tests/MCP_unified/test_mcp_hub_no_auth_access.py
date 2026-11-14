"""A bound auth-free MCP server does not need an invented credential."""

from __future__ import annotations

import base64
import json

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "auth,capability,enabled,binding_mode,executable,reason",
    [
        (None, True, True, "grant", True, None),
        ({"mode": "none"}, True, True, "grant", True, None),
        (None, False, True, "grant", False, "external_capability_not_granted"),
        (None, True, False, "grant", False, "server_disabled"),
        ({"mode": "bearer_token"}, True, True, "grant", False, "missing_secret"),
        ({"mode": "api_key_header"}, True, True, "grant", False, "missing_secret"),
        (None, True, True, "disable", False, "disabled_by_assignment"),
    ],
    ids=[
        "implicit-none",
        "explicit-none",
        "missing-capability",
        "disabled-server",
        "missing-bearer",
        "missing-api-key",
        "disabled-binding",
    ],
)
async def test_bound_external_no_auth_access_keeps_existing_gates(
    tmp_path, monkeypatch, auth, capability, enabled, binding_mode, executable, reason
):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_external_access_resolver import McpHubExternalAccessResolver

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", base64.b64encode(b"k" * 32).decode("ascii"))
    reset_settings()
    await reset_db_pool()
    try:
        pool = await get_db_pool()
        ensure_authnz_tables(db_path)
        repo = McpHubRepo(pool)
        await repo.ensure_tables()
        policy = {"capabilities": ["network.external"] if capability else []}
        profile = await repo.create_permission_profile(
            name="Bound public server",
            owner_scope_type="user",
            owner_scope_id=1,
            mode="custom",
            policy_document=policy,
            actor_id=1,
        )
        assignment = await repo.create_policy_assignment(
            target_type="persona",
            target_id="public-reader",
            owner_scope_type="user",
            owner_scope_id=1,
            profile_id=int(profile["id"]),
            inline_policy_document=policy,
            approval_policy_id=None,
            actor_id=1,
            is_active=True,
        )
        config = {"stdio": {"command": "/usr/bin/true", "args": []}}
        if auth is not None:
            config["auth"] = auth
        await repo.upsert_external_server(
            server_id="public",
            name="Public readonly",
            transport="stdio",
            config_json=json.dumps(config),
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=enabled,
            actor_id=1,
        )
        await repo.create_credential_binding(
            binding_target_type="profile",
            binding_target_id=str(profile["id"]),
            external_server_id="public",
            credential_ref="server",
            binding_mode="grant",
            usage_rules={},
            actor_id=1,
        )
        if binding_mode == "disable":
            await repo.create_credential_binding(
                binding_target_type="assignment",
                binding_target_id=str(assignment["id"]),
                external_server_id="public",
                credential_ref="server",
                binding_mode="disable",
                usage_rules={},
                actor_id=1,
            )
        summary = await McpHubExternalAccessResolver(repo=repo).resolve(
            assignment_id=int(assignment["id"]),
            effective_policy=policy,
        )
        row = summary["servers"][0]
        assert (row["runtime_executable"], row["blocked_reason"], row["secret_available"]) == (
            executable,
            reason,
            False,
        )  # nosec B101 -- readiness and existing denial gates must agree without a secret.
    finally:
        await reset_db_pool()
        reset_settings()
