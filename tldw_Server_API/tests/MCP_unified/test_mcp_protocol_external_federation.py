from __future__ import annotations

import base64
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
from tldw_Server_API.app.core.MCP_unified.config import get_config
from tldw_Server_API.app.core.MCP_unified.external_servers import VirtualExternalTool
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.external_federation_module import (
    ExternalFederationModule,
)
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext
from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod
from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    ExternalServerRegistryPartialLoadError,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.base import (
    BrokeredExternalCredential,
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    ExternalToolDefinition,
)


def _b64_key(byte_char: bytes) -> str:
    return base64.b64encode(byte_char * 32).decode("ascii")


class _BrokerAwareAdapter(ExternalMCPTransportAdapter):
    def __init__(self, server_id: str) -> None:
        super().__init__(server_id=server_id)
        self.seen_runtime_auth: BrokeredExternalCredential | None = None

    @property
    def transport_name(self) -> str:
        return "websocket"

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def health_check(self) -> dict[str, bool]:
        return {"configured": True, "connected": True, "initialized": True}

    async def list_tools(self) -> list[ExternalToolDefinition]:
        return [
            ExternalToolDefinition(
                name="repo.search",
                description="Search repos",
                input_schema={"type": "object"},
                metadata={"category": "read"},
            )
        ]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        del tool_name, arguments, context
        self.seen_runtime_auth = runtime_auth
        return ExternalToolCallResult(
            content=[{"type": "text", "text": "ok"}],
            is_error=False,
            metadata={"adapter": "broker-aware"},
        )


class _RuntimeRowsRepo:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    async def list_external_servers(self) -> list[dict[str, Any]]:
        return self._rows


class _StaticExternalWriteManager:
    def __init__(self) -> None:
        self.called = False

    def list_virtual_tools(self) -> list[VirtualExternalTool]:
        return [
            VirtualExternalTool(
                virtual_name="ext.docs.repo.set",
                server_id="docs",
                upstream_tool_name="repo.set",
                description="Set repository metadata",
                input_schema={"type": "object"},
                metadata={},
                is_write=True,
            )
        ]

    def get_virtual_tool_write_flag(self, virtual_tool_name: str) -> bool | None:
        for virtual_tool in self.list_virtual_tools():
            if virtual_tool.virtual_name == virtual_tool_name:
                return bool(virtual_tool.is_write)
        return None

    async def execute_virtual_tool(
        self,
        virtual_tool_name: str,
        arguments: dict[str, Any],
        context: Any = None,
    ) -> dict[str, Any]:
        del virtual_tool_name, arguments, context
        self.called = True
        return {"ok": True}

    async def list_servers(self) -> list[dict[str, Any]]:
        return []

    async def shutdown(self) -> None:
        return None


class _StaticExternalWriteModule(ExternalFederationModule):
    async def on_initialize(self) -> None:
        self._manager = _StaticExternalWriteManager()


class _AllowAllRBAC:
    async def check_permission(self, *args: Any, **kwargs: Any) -> bool:
        return True


@pytest.mark.asyncio
async def test_managed_external_registry_service_reports_partial_invalid_rows() -> None:
    from tldw_Server_API.app.services.mcp_hub_external_registry_service import (
        McpHubExternalRegistryService,
    )

    repo = _RuntimeRowsRepo(
        [
            {
                "id": "docs",
                "name": "Docs",
                "transport": "websocket",
                "config": {"websocket": {"url": "wss://docs.example/ws"}},
                "enabled": True,
                "server_source": "managed",
            },
            {
                "id": "broken",
                "name": "Broken",
                "transport": "websocket",
                "config": {},
                "enabled": True,
                "server_source": "managed",
            },
        ]
    )
    service = McpHubExternalRegistryService(repo=repo)  # type: ignore[arg-type]

    with pytest.raises(ExternalServerRegistryPartialLoadError) as exc_info:
        await service.list_runtime_servers()

    exc = exc_info.value
    assert [server.id for server in exc.servers] == ["docs"]
    assert exc.errors == {"broken": "external_server_config_invalid"}


@pytest.mark.asyncio
async def test_managed_external_registry_service_reports_partial_invalid_rows_with_missing_ids() -> None:
    from tldw_Server_API.app.services.mcp_hub_external_registry_service import (
        McpHubExternalRegistryService,
    )

    repo = _RuntimeRowsRepo(
        [
            {
                "id": "docs",
                "name": "Docs",
                "transport": "websocket",
                "config": {"websocket": {"url": "wss://docs.example/ws"}},
                "enabled": True,
                "server_source": "managed",
            },
            {
                "id": " ",
                "name": "Broken",
                "transport": "websocket",
                "config": {},
                "enabled": True,
                "server_source": "managed",
            },
        ]
    )
    service = McpHubExternalRegistryService(repo=repo)  # type: ignore[arg-type]

    with pytest.raises(ExternalServerRegistryPartialLoadError) as exc_info:
        await service.list_runtime_servers()

    exc = exc_info.value
    assert [server.id for server in exc.servers] == ["docs"]
    assert exc.errors == {"row_2": "external_server_config_invalid"}


@pytest.mark.asyncio
async def test_managed_external_registry_service_keeps_runtime_config_auth_neutral_and_skips_legacy(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        dumps_envelope,
        encrypt_byok_payload,
    )
    from tldw_Server_API.app.services.mcp_hub_external_registry_service import (
        McpHubExternalRegistryService,
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
    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json=json.dumps(
            {
                "websocket": {"url": "wss://docs.example/ws"},
                "auth": {"mode": "bearer_token"},
            }
        ),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        server_source="managed",
        actor_id=1,
    )
    await repo.upsert_external_secret(
        server_id="docs",
        encrypted_blob=dumps_envelope(encrypt_byok_payload(build_secret_payload("super-secret-token"))),
        key_hint="oken",
        actor_id=1,
    )
    await repo.upsert_external_server(
        server_id="legacy-docs",
        name="Legacy Docs",
        transport="websocket",
        config_json=json.dumps({"websocket": {"url": "wss://legacy.example/ws"}}),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        server_source="legacy",
        actor_id=1,
    )
    await repo.upsert_external_server(
        server_id="old-docs",
        name="Old Docs",
        transport="websocket",
        config_json=json.dumps({"websocket": {"url": "wss://old.example/ws"}}),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        server_source="managed",
        superseded_by_server_id="docs",
        actor_id=1,
    )

    service = McpHubExternalRegistryService(repo=repo)
    servers = await service.list_runtime_servers()

    assert [server.id for server in servers] == ["docs"]
    assert servers[0].websocket is not None
    assert servers[0].websocket.headers == {}
    assert str(getattr(servers[0].auth.mode, "value", servers[0].auth.mode)) == "none"


@pytest.mark.asyncio
async def test_external_federation_module_ignores_legacy_file_config_for_runtime(tmp_path, monkeypatch) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

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
                        "websocket": {"url": "wss://legacy.example/ws"},
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

    module = ExternalFederationModule(
        ModuleConfig(
            name="external_federation",
            settings={"external_servers_config_path": str(cfg_path)},
        )
    )
    try:
        await module.initialize()
        tools = await module.get_tools()
    finally:
        await module.shutdown()

    tool_names = {
        str(tool.get("name") or "")
        for tool in tools
        if isinstance(tool, dict)
    }
    assert "external.servers.list" in tool_names
    assert "external.tools.refresh" in tool_names
    assert not any(name.startswith("ext.legacy-docs.") for name in tool_names)


@pytest.mark.asyncio
async def test_managed_external_registry_service_keeps_named_slot_template_auth_neutral(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        dumps_envelope,
        encrypt_byok_payload,
    )
    from tldw_Server_API.app.services.mcp_hub_external_registry_service import (
        McpHubExternalRegistryService,
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
    await repo.upsert_external_server(
        server_id="docs",
        name="Docs",
        transport="websocket",
        config_json=json.dumps(
            {
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
            }
        ),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        server_source="managed",
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
        encrypted_blob=dumps_envelope(
            encrypt_byok_payload(build_secret_payload("super-secret-token"))
        ),
        key_hint="oken",
        actor_id=1,
    )

    service = McpHubExternalRegistryService(repo=repo)
    servers = await service.list_runtime_servers()

    assert [server.id for server in servers] == ["docs"]
    assert servers[0].websocket is not None
    assert servers[0].websocket.headers == {}
    assert str(getattr(servers[0].auth.mode, "value", servers[0].auth.mode)) == "none"


@pytest.mark.asyncio
async def test_managed_external_registry_service_keeps_stdio_template_auth_neutral(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.AuthNZ.user_provider_secrets import (
        build_secret_payload,
        dumps_envelope,
        encrypt_byok_payload,
    )
    from tldw_Server_API.app.services.mcp_hub_external_registry_service import (
        McpHubExternalRegistryService,
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
    await repo.upsert_external_server(
        server_id="docs-stdio",
        name="Docs Stdio",
        transport="stdio",
        config_json=json.dumps(
            {
                "stdio": {"command": "npx", "args": ["-y", "@docs/server"]},
                "auth": {
                    "mode": "template",
                    "mappings": [
                        {
                            "slot_name": "token_readonly",
                            "target_type": "env",
                            "target_name": "DOCS_TOKEN",
                            "prefix": "",
                            "suffix": "",
                            "required": True,
                        }
                    ],
                },
            }
        ),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        server_source="managed",
        actor_id=1,
    )
    await repo.create_external_server_credential_slot(
        server_id="docs-stdio",
        slot_name="token_readonly",
        display_name="Read-only token",
        secret_kind="bearer_token",
        privilege_class="read",
        is_required=True,
        actor_id=1,
    )
    await repo.upsert_external_server_slot_secret(
        server_id="docs-stdio",
        slot_name="token_readonly",
        encrypted_blob=dumps_envelope(
            encrypt_byok_payload(build_secret_payload("super-secret-token"))
        ),
        key_hint="oken",
        actor_id=1,
    )

    service = McpHubExternalRegistryService(repo=repo)
    servers = await service.list_runtime_servers()

    assert [server.id for server in servers] == ["docs-stdio"]
    assert servers[0].stdio is not None
    assert servers[0].stdio.env == {}
    assert str(getattr(servers[0].auth.mode, "value", servers[0].auth.mode)) == "none"


@pytest.mark.asyncio
async def test_external_federation_module_brokers_managed_secret_ref_at_call_time(
    tmp_path,
    monkeypatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.secret_backends.local_encrypted import (
        LocalEncryptedSecretBackend,
    )
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
    from tldw_Server_API.app.core.DB_Management.Users_DB import create_user

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
    await create_user(
        username="alice",
        email="alice@example.com",
        password_hash="hashed-password",
        is_active=True,
        is_verified=True,
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
                "policy": {"allow_tool_patterns": ["repo.*"]},
            }
        ),
        owner_scope_type="global",
        owner_scope_id=None,
        enabled=True,
        server_source="managed",
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

    backend = LocalEncryptedSecretBackend(db_pool=pool)
    ref = await backend.store_ref(
        owner_scope_type="user",
        owner_scope_id=1,
        provider_key="docs-token",
        payload={"secret": "super-secret-token"},
        created_by=1,
        updated_by=1,
    )
    await repo.upsert_credential_binding(
        binding_target_type="assignment",
        binding_target_id="7",
        external_server_id="docs",
        slot_name="token_readonly",
        credential_ref=f"managed_secret_ref:{int(ref['id'])}",
        binding_mode="grant",
        usage_rules={},
        actor_id=1,
    )

    adapter = _BrokerAwareAdapter(server_id="docs")
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _server: adapter)

    module = ExternalFederationModule(ModuleConfig(name="external_federation", settings={}))
    try:
        await module.initialize()
        ctx = RequestContext(
            request_id="broker-runtime",
            user_id="1",
            metadata={
                "_mcp_effective_tool_policy": {
                    "enabled": True,
                    "capabilities": ["network.external"],
                    "sources": [{"assignment_id": 7, "profile_id": None}],
                    "selected_assignment_id": 7,
                },
                "mcp_policy_context_enabled": True,
            },
        )
        result = await module.execute_tool("ext.docs.repo.search", {"q": "x"}, context=ctx)
    finally:
        await module.shutdown()

    assert adapter.seen_runtime_auth is not None
    assert adapter.seen_runtime_auth.headers == {"Authorization": "Bearer super-secret-token"}
    assert result["metadata"]["credential_mode"] == "brokered_ephemeral"


def test_external_federation_validates_refresh_and_list_arguments() -> None:
    module = ExternalFederationModule(ModuleConfig(name="external_federation", settings={}))

    module.validate_tool_arguments("external.tools.refresh", {})
    module.validate_tool_arguments("external.tools.refresh", {"server_id": "docs"})
    module.validate_tool_arguments("external.servers.list", {})

    with pytest.raises(ValueError, match="arguments must be an object"):
        module.validate_tool_arguments("external.tools.refresh", [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="server_id must be a non-empty string"):
        module.validate_tool_arguments("external.tools.refresh", {"server_id": ""})
    with pytest.raises(ValueError, match="server_id must be a non-empty string"):
        module.validate_tool_arguments("external.tools.refresh", {"server_id": 7})
    with pytest.raises(ValueError, match="does not accept arguments"):
        module.validate_tool_arguments("external.servers.list", {"server_id": "docs"})
    with pytest.raises(ValueError, match="Unknown external federation management tool"):
        module.validate_tool_arguments("external.servers.delete", {})


def test_external_federation_validates_virtual_tool_argument_shape_and_confirmation() -> None:
    module = ExternalFederationModule(ModuleConfig(name="external_federation", settings={}))

    module.validate_tool_arguments("ext.docs.repo.search", {"q": "x"})
    module.validate_tool_arguments("ext.docs.repo.update", {"title": "x", "__confirm_write": True})

    with pytest.raises(ValueError, match="arguments must be an object"):
        module.validate_tool_arguments("ext.docs.repo.search", "q=x")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="__confirm_write must be a boolean"):
        module.validate_tool_arguments("ext.docs.repo.update", {"__confirm_write": "true"})


@pytest.mark.asyncio
async def test_external_federation_virtual_write_tools_respect_protocol_write_disable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_DISABLE_WRITE_TOOLS", "true")
    get_config.cache_clear()  # type: ignore[attr-defined]

    registry = get_module_registry()
    module_id = "external_write_policy_test"
    await registry.register_module(
        module_id,
        _StaticExternalWriteModule,
        ModuleConfig(name=module_id),
    )

    try:
        proto = MCPProtocol()
        proto.rbac_policy = _AllowAllRBAC()
        ctx = RequestContext(request_id="external-write-policy", user_id="u1", client_id="c1")
        req = MCPRequest(
            method="tools/call",
            params={
                "name": "ext.docs.repo.set",
                "arguments": {"value": "x", "__confirm_write": True},
            },
            id="external-write-policy",
        )

        resp = await proto.process_request(req, ctx)

        assert resp.error is not None
        assert resp.error.code == -32001
        assert "write tools are disabled" in (resp.error.message or "").lower()
        module = await registry.get_module(module_id)
        assert isinstance(module, _StaticExternalWriteModule)
        assert isinstance(module._manager, _StaticExternalWriteManager)
        assert module._manager.called is False
    finally:
        with contextlib.suppress(Exception):
            await registry.unregister_module(module_id)
        get_config.cache_clear()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_external_federation_execute_rejects_invalid_management_arguments() -> None:
    class _Manager:
        called = False

        async def list_servers(self) -> list[dict[str, Any]]:
            self.called = True
            return []

    manager = _Manager()
    module = ExternalFederationModule(ModuleConfig(name="external_federation", settings={}))
    module._manager = manager

    with pytest.raises(ValueError, match="external.servers.list does not accept arguments"):
        await module.execute_tool("external.servers.list", {"server_id": "docs"})

    assert manager.called is False


@pytest.mark.asyncio
async def test_external_federation_execute_validates_virtual_tools_after_sanitizing_arguments() -> None:
    class _Manager:
        seen_arguments: dict[str, Any] | None = None

        async def execute_virtual_tool(
            self,
            *,
            virtual_tool_name: str,
            arguments: dict[str, Any],
            context: Any = None,
        ) -> dict[str, Any]:
            del virtual_tool_name, context
            self.seen_arguments = arguments
            return {"ok": True}

    manager = _Manager()
    module = ExternalFederationModule(ModuleConfig(name="external_federation", settings={}))
    module._manager = manager

    result = await module.execute_tool("ext.docs.repo.search", {"q": "hello\x00"})

    assert result == {"ok": True}
    assert manager.seen_arguments == {"q": "hello"}

    with pytest.raises(ValueError, match="__confirm_write must be a boolean"):
        await module.execute_tool("ext.docs.repo.update", {"__confirm_write": "true"})
