from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)


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
async def test_set_external_secret_encrypts_and_never_returns_plaintext(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_emits_audit_event_on_external_server_update(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    await reset_db_pool()

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
async def test_create_external_server_raises_conflict_without_allow_existing(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubConflictError, McpHubService

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
async def test_service_can_import_legacy_external_server(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_resolves_effective_external_access_with_assignment_disable(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_lists_legacy_inventory_from_config_and_hides_imported_duplicates(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
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
    await reset_db_pool()
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
async def test_service_rejects_cross_scope_assignment_binding(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_lists_slot_summary_for_managed_external_server(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_rejects_server_secret_alias_for_multislot_server(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_rejects_unknown_slot_privilege_class(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.exceptions import BadRequestError
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
async def test_service_audits_slot_binding_with_privilege_metadata(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("BYOK_ENCRYPTION_KEY", _b64_key(b"k"))

    reset_settings()
    await reset_db_pool()
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
async def test_service_rejects_binding_managed_secret_ref_from_different_scope(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.secret_backends.local_encrypted import LocalEncryptedSecretBackend
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

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
    svc = McpHubService(repo=repo)

    profile = await repo.create_permission_profile(
        name="External Docs",
        owner_scope_type="org",
        owner_scope_id=7,
        mode="custom",
        policy_document={"capabilities": ["network.external"]},
        actor_id=7,
    )
    await svc.create_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config={"websocket": {"url": "wss://docs.example/ws"}},
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        actor_id=7,
    )
    managed_ref = await LocalEncryptedSecretBackend(db_pool=pool).store_ref(
        owner_scope_type="org",
        owner_scope_id=42,
        provider_key="docs-token",
        payload={"api_key": "secret-token"},
        created_by=7,
        updated_by=7,
    )

    with pytest.raises(BadRequestError, match="different owner scope"):
        await svc.upsert_profile_credential_binding(
            profile_id=int(profile["id"]),
            external_server_id="docs",
            managed_secret_ref_id=int(managed_ref["id"]),
            actor_id=7,
        )
