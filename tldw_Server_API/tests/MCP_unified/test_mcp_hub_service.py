from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.exceptions import BadRequestError

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


def test_normalize_shared_workspace_root_rejects_paths_outside_allowed_bases(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    allowed_root = tmp_path / "allowed"
    disallowed_root = tmp_path / "disallowed"
    allowed_root.mkdir()
    disallowed_root.mkdir()
    monkeypatch.setenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", str(allowed_root))

    with pytest.raises(BadRequestError, match="allowed base paths"):
        McpHubService._normalize_shared_workspace_root(str(disallowed_root))


def test_normalize_shared_workspace_root_accepts_paths_under_symlinked_allowed_base(
    monkeypatch,
    tmp_path,
) -> None:
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    real_root = tmp_path / "real-shared"
    real_root.mkdir()
    team_root = real_root / "team-a"
    team_root.mkdir()
    symlink_root = tmp_path / "allowed-link"
    symlink_root.symlink_to(real_root, target_is_directory=True)
    monkeypatch.setenv("ACP_WORKSPACE_ALLOWED_BASE_PATHS", str(symlink_root))

    normalized = McpHubService._normalize_shared_workspace_root(str(team_root))

    assert normalized == str(team_root.resolve())


@pytest.mark.asyncio
async def test_set_external_secret_encrypts_and_never_returns_plaintext(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="stdio",
        config={"cmd": "npx"},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    out = await svc.set_external_server_secret(
        server_id="docs",
        secret_value="super-secret-token",
        actor_id=1,
    )

    assert out["secret_configured"] is True
    assert "super-secret-token" not in json.dumps(out)

    secret = await repo.get_external_secret("docs")
    assert secret is not None
    assert "super-secret-token" not in json.dumps(secret)


@pytest.mark.asyncio
async def test_service_emits_audit_event_on_external_server_update(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    calls: list[dict[str, object]] = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.services.mcp_hub_service.emit_mcp_hub_audit",
        _capture,
    )

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="stdio",
        config={"cmd": "npx"},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    assert calls
    assert calls[0]["action"] == "mcp_hub.external_server.create"


@pytest.mark.asyncio
async def test_create_external_server_raises_conflict_without_allow_existing(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubConflictError, McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="stdio",
        config={"cmd": "npx"},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    with pytest.raises(McpHubConflictError):
        await svc.create_external_server(
            server_id="docs",
            name="Docs Updated",
            transport="stdio",
            config={"cmd": "npx"},
            owner_scope_type="global",
            owner_scope_id=None,
            enabled=True,
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_service_can_import_legacy_external_server(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)
    monkeypatch.setattr(
        svc,
        "_legacy_external_inventory",
        [
            {
                "id": "legacy-docs",
                "name": "Legacy Docs",
                "enabled": True,
                "transport": "websocket",
                "config": {
                    "websocket": {"url": "wss://docs.example/ws"},
                    "auth": {"mode": "bearer_env", "token_env": "DOCS_TOKEN"},
                },
                "legacy_source_ref": "yaml:legacy-docs",
            }
        ],
        raising=False,
    )

    imported = await svc.import_legacy_external_server(
        server_id="legacy-docs",
        actor_id=1,
    )

    assert imported["id"] == "legacy-docs"
    assert imported["server_source"] == "managed"


@pytest.mark.asyncio
async def test_service_resolves_effective_external_access_with_assignment_disable(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

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

    access = await svc.resolve_effective_external_access(
        assignment_id=int(assignment["id"]),
        actor_id=1,
    )

    assert access["servers"][0]["disabled_by_assignment"] is True


@pytest.mark.asyncio
async def test_service_lists_legacy_inventory_from_config_and_hides_imported_duplicates(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_external_legacy_inventory import (
        McpHubExternalLegacyInventoryService,
    )
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    cfg_path = tmp_path / "external_servers.yaml"
    cfg_path.write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "id": "legacy-docs",
                        "name": "Legacy Docs",
                        "transport": "websocket",
                        "websocket": {"url": "wss://docs.example/ws"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(
        repo=repo,
        legacy_inventory_service=McpHubExternalLegacyInventoryService(config_path=str(cfg_path)),
    )

    rows = await svc.list_external_servers()
    assert rows[0]["server_source"] == "legacy"

    imported = await svc.import_legacy_external_server(server_id="legacy-docs", actor_id=1)
    rows_after_import = await svc.list_external_servers()

    assert imported["legacy_source_ref"] == "yaml:legacy-docs"
    assert len(rows_after_import) == 1
    assert rows_after_import[0]["server_source"] == "managed"


@pytest.mark.asyncio
async def test_service_rejects_cross_scope_assignment_binding(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    assignment = await repo.create_policy_assignment(
        target_type="group",
        target_id="ops",
        owner_scope_type="team",
        owner_scope_id=5,
        profile_id=None,
        inline_policy_document={"capabilities": ["network.external"]},
        approval_policy_id=None,
        actor_id=1,
        is_active=True,
    )
    await repo.upsert_external_server(
        server_id="private-docs",
        name="Private Docs",
        transport="websocket",
        config_json=json.dumps({"websocket": {"url": "wss://docs.example/ws"}}),
        owner_scope_type="user",
        owner_scope_id=1,
        enabled=True,
        server_source="managed",
        actor_id=1,
    )

    with pytest.raises(BadRequestError):
        await svc.upsert_assignment_credential_binding(
            assignment_id=int(assignment["id"]),
            external_server_id="private-docs",
            binding_mode="grant",
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_service_lists_slot_summary_for_managed_external_server(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={
            "websocket": {"url": "wss://docs.example/ws"},
            "auth": {
                "mode": "bearer_token",
                "required_slots": ["token_readonly"],
                "slot_bindings": {
                    "token_readonly": {
                        "inject": "header",
                        "header_name": "Authorization",
                        "prefix": "Bearer ",
                    }
                },
            },
        },
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await svc.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )
    await svc.set_external_server_slot_secret(
        server_id="docs",
        slot_name="token_readonly",
        secret_value="super-secret-token",
        actor_id=1,
    )

    rows = await svc.list_external_servers()

    assert rows[0]["credential_slots"][0]["slot_name"] == "token_readonly"
    assert rows[0]["credential_slots"][0]["secret_configured"] is True


@pytest.mark.asyncio
async def test_service_rejects_server_secret_alias_for_multislot_server(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={
            "websocket": {"url": "wss://docs.example/ws"},
            "auth": {
                "mode": "bearer_token",
                "required_slots": ["token_readonly", "token_write"],
                "slot_bindings": {
                    "token_readonly": {
                        "inject": "header",
                        "header_name": "Authorization",
                        "prefix": "Bearer ",
                    },
                    "token_write": {
                        "inject": "header",
                        "header_name": "X-Write-Token",
                    },
                },
            },
        },
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await svc.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )
    await svc.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_write",
        display_name="Write token",
        secret_kind="api_key",
        privilege_class="write",
        is_required=True,
        actor_id=1,
    )

    with pytest.raises(BadRequestError):
        await svc.set_external_server_secret(
            server_id="docs",
            secret_value="ambiguous-server-secret",
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_service_rejects_unknown_slot_privilege_class(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={"websocket": {"url": "wss://docs.example/ws"}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )

    with pytest.raises(BadRequestError):
        await svc.create_external_server_credential_slot(
            server_id="docs",
            slot_name="token_super",
            display_name="Super token",
            secret_kind="bearer_token",
            privilege_class="superuser",
            is_required=True,
            actor_id=1,
        )


@pytest.mark.asyncio
async def test_service_audits_slot_binding_with_privilege_metadata(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    audit_calls: list[dict[str, object]] = []

    def _capture(**kwargs):
        audit_calls.append(kwargs)
        return None

    monkeypatch.setattr(
        "tldw_Server_API.app.services.mcp_hub_service.emit_mcp_hub_audit",
        _capture,
    )

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    profile = await repo.create_permission_profile(
        name="External Docs",
        owner_scope_type="user",
        owner_scope_id=1,
        mode="custom",
        policy_document={"capabilities": ["network.external"]},
        actor_id=1,
    )
    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={"websocket": {"url": "wss://docs.example/ws"}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=1,
    )
    await svc.create_external_server_credential_slot(
        server_id="docs",
        slot_name="token_write",
        display_name="Write token",
        secret_kind="api_key",
        privilege_class="write",
        is_required=False,
        actor_id=1,
    )

    audit_calls.clear()
    await svc.upsert_profile_credential_binding(
        profile_id=int(profile["id"]),
        external_server_id="docs",
        slot_name="token_write",
        actor_id=1,
    )

    assert audit_calls
    metadata = dict(audit_calls[-1].get("metadata") or {})
    assert metadata["slot_name"] == "token_write"
    assert metadata["privilege_class"] == "write"
    assert metadata["required_permission"] == "grant.credentials.write"


@pytest.mark.asyncio
async def test_service_rejects_binding_managed_secret_ref_from_different_scope(tmp_path, monkeypatch, fresh_services) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
    from tldw_Server_API.app.core.AuthNZ.secret_backends.local_encrypted import LocalEncryptedSecretBackend
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService
    from tldw_Server_API.tests.helpers.authnz_seed import ensure_test_user

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    actor_id = await ensure_test_user(pool, "scope-admin", role="admin")
    orgs = AuthnzOrgsTeamsRepo(pool)
    profile_org = await orgs.create_organization(name="Profile org", slug="profile-org")
    secret_org = await orgs.create_organization(name="Secret org", slug="secret-org")
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    svc = McpHubService(repo=repo)

    profile = await repo.create_permission_profile(
        name="External Docs",
        owner_scope_type="org",
        owner_scope_id=int(profile_org["id"]),
        mode="custom",
        policy_document={"capabilities": ["network.external"]},
        actor_id=actor_id,
    )
    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={"websocket": {"url": "wss://docs.example/ws"}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=actor_id,
    )
    managed_ref = await LocalEncryptedSecretBackend(db_pool=pool).store_ref(
        owner_scope_type="org",
        owner_scope_id=int(secret_org["id"]),
        provider_key="docs-token",
        payload={"api_key": "secret-token"},
        created_by=actor_id,
        updated_by=actor_id,
    )

    with pytest.raises(BadRequestError, match="different owner scope"):
        await svc.upsert_profile_credential_binding(
            profile_id=int(profile["id"]),
            external_server_id="docs",
            managed_secret_ref_id=int(managed_ref["id"]),
            actor_id=actor_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key,target",
    [(key, target) for key in ("", "CHANGE_ME_TO_SECURE_BYOK_KEY") for target in ("legacy", "slot", "default_alias")]
    + [(_b64_key(b"k"), "default_alias_second")],
)
async def test_external_secret_encryption_configuration_is_actionable_and_writes_nothing(
    tmp_path, monkeypatch, key, target
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", key)
    reset_settings()
    await reset_db_pool()
    pool = await get_db_pool()
    ensure_authnz_tables(db_path)
    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    service = McpHubService(repo=repo)
    await service.create_external_server(
        server_id="crypto-config-test",
        name="Crypto configuration test",
        transport="stdio",
        config={"cmd": "npx"},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=False,
        actor_id=1,
    )
    if target != "legacy":
        await service.create_external_server_credential_slot(  # nosec B106 -- api_key is the credential-kind fixture label.
            server_id="crypto-config-test",
            slot_name="api_key",
            display_name="Default",
            secret_kind="api_key",
            privilege_class="read",
            is_required=True,
            actor_id=1,
        )
    legacy_before = None
    slot_before = None
    if target == "default_alias_second":
        saved = await service.set_external_server_secret(  # nosec B106 -- fixed synthetic previous credential in an isolated test database.
            server_id="crypto-config-test", secret_value="c11-previous-value", actor_id=1
        )
        legacy_before = await repo.get_external_secret("crypto-config-test")
        slot_before = await repo.get_external_server_slot_secret(server_id="crypto-config-test", slot_name="api_key")
        assert saved["secret_configured"] and legacy_before and slot_before  # nosec B101 -- the valid default alias must persist both encrypted stores.
        from tldw_Server_API.app.services import mcp_hub_service

        encrypt = mcp_hub_service.encrypt_byok_payload
        calls = 0

        def reject_second_encryption(payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("c11-unit-only-canary")
            return encrypt(payload)

        monkeypatch.setattr(mcp_hub_service, "encrypt_byok_payload", reject_second_encryption)
    with pytest.raises(BadRequestError, match="Configure a valid BYOK_ENCRYPTION_KEY") as caught:
        if target == "slot":
            await service.set_external_server_slot_secret(  # nosec B106 -- fixed synthetic rejection canary, never a real credential.
                server_id="crypto-config-test",
                slot_name="api_key",
                secret_value="c11-unit-only-canary",
                actor_id=1,
            )
        else:
            await service.set_external_server_secret(  # nosec B106 -- fixed synthetic rejection canary, never a real credential.
                server_id="crypto-config-test",
                secret_value="c11-unit-only-canary",
                actor_id=1,
            )
    assert caught.value.__context__ is None and caught.value.__cause__ is None  # nosec B101 -- safe public error must not retain private crypto context.
    assert await repo.get_external_secret("crypto-config-test") == legacy_before  # nosec B101 -- failed encryption must preserve the previous legacy credential.
    if target != "legacy":
        assert (
            await repo.get_external_server_slot_secret(server_id="crypto-config-test", slot_name="api_key")
            == slot_before
        )  # nosec B101 -- failed encryption must preserve the previous slot credential.
    if target == "default_alias_second":
        assert calls == 2  # nosec B101 -- the first real encryption must succeed before the second rejects.


@pytest.mark.asyncio
@pytest.mark.parametrize("deleted", [True, False])
async def test_delete_workspace_member_reports_result_and_audits_its_owner(monkeypatch, deleted):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from tldw_Server_API.app.services import mcp_hub_service

    owner = {"id": 17, "owner_scope_type": "org", "owner_scope_id": 42}
    repo = SimpleNamespace(
        get_workspace_set_object=AsyncMock(return_value=owner),
        delete_workspace_set_member=AsyncMock(return_value=deleted),
    )
    audit = AsyncMock()
    monkeypatch.setattr(mcp_hub_service, "emit_mcp_hub_audit", audit)
    service = mcp_hub_service.McpHubService(repo=repo)
    result = await service.delete_workspace_set_member(17, "rowan", actor_id=7)
    assert result is deleted  # nosec B101 -- successful/no-op deletion must keep its actual repository result.
    repo.delete_workspace_set_member.assert_awaited_once_with(17, "rowan")
    if deleted:
        audit.assert_awaited_once_with(
            action="mcp_hub.workspace_set_object.member.delete",
            actor_id=7,
            resource_type="mcp_workspace_set_object",
            resource_id="17",
            metadata={"owner_scope_type": "org", "owner_scope_id": 42, "workspace_id": "rowan"},
        )
    else:
        audit.assert_not_awaited()
