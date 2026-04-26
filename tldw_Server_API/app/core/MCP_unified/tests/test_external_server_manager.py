from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.external_servers.config_schema import (
    parse_external_server_registry,
)
from tldw_Server_API.app.core.MCP_unified.external_servers.manager import ExternalServerManager
from tldw_Server_API.app.core.MCP_unified.external_servers.transports.base import (
    BrokeredExternalCredential,
    ExternalMCPTransportAdapter,
    ExternalToolCallResult,
    ExternalToolDefinition,
    adapter_supports_runtime_auth,
    call_tool_with_ephemeral_adapter,
)
from tldw_Server_API.app.core.MCP_unified.external_servers import manager as manager_mod


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _FakeAdapter(ExternalMCPTransportAdapter):
    def __init__(self, server_id: str, tools: list[ExternalToolDefinition]) -> None:
        super().__init__(server_id)
        self.connected = False
        self.tools = tools
        self.fail_list = False
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.next_call_exception: Exception | None = None
        self.next_call_is_error = False

    @property
    def transport_name(self) -> str:
        return "fake"

    async def connect(self) -> None:
        self.connected = True

    async def close(self) -> None:
        self.connected = False

    async def health_check(self) -> dict[str, bool]:
        return {"configured": True, "connected": self.connected}

    async def list_tools(self) -> list[ExternalToolDefinition]:
        if self.fail_list:
            raise RuntimeError("discovery failed")
        return list(self.tools)

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context=None,
    ) -> ExternalToolCallResult:
        del context
        self.calls.append((tool_name, dict(arguments)))
        if self.next_call_exception is not None:
            exc = self.next_call_exception
            self.next_call_exception = None
            raise exc
        if self.next_call_is_error:
            self.next_call_is_error = False
            return ExternalToolCallResult(
                content=[{"type": "text", "text": "upstream failed"}],
                is_error=True,
                metadata={"adapter": "fake"},
            )
        return ExternalToolCallResult(
            content={"ok": True, "tool": tool_name, "args": dict(arguments)},
            is_error=False,
            metadata={"adapter": "fake"},
        )


def _registry_payload(*, policy: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "servers": [
            {
                "id": "docs",
                "name": "Docs",
                "transport": "websocket",
                "websocket": {"url": "wss://example.test/ws"},
                "policy": policy or {},
            }
        ]
    }


def _patch_loader_and_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    payload: dict[str, Any],
    adapter: _FakeAdapter,
) -> None:
    cfg = parse_external_server_registry(payload)
    monkeypatch.setattr(manager_mod, "load_external_server_registry", lambda _path=None: cfg)
    monkeypatch.setattr(manager_mod, "build_transport_adapter", lambda _server: adapter)


def test_parse_virtual_tool_name_routing_contract() -> None:
    server_id, tool_name = ExternalServerManager.parse_virtual_tool_name("ext.docs.docs.search")
    _ensure(server_id == "docs", f"Unexpected server id parse result: {server_id!r}")
    _ensure(tool_name == "docs.search", f"Unexpected tool name parse result: {tool_name!r}")

    with pytest.raises(ValueError, match="must start with 'ext.'"):
        ExternalServerManager.parse_virtual_tool_name("docs.search")
    with pytest.raises(ValueError, match="must match 'ext.<server_id>.<tool_name>'"):
        ExternalServerManager.parse_virtual_tool_name("ext.docs")


def test_adapter_runtime_auth_compatibility_helper_handles_legacy_signature() -> None:
    adapter = _FakeAdapter(server_id="docs", tools=[])
    _ensure(
        adapter_supports_runtime_auth(adapter) is False,
        "legacy adapter signature should not be treated as runtime-auth aware",
    )


@pytest.mark.asyncio
async def test_legacy_adapter_omits_runtime_auth_metadata_when_runtime_auth_is_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _FakeAdapter(
        server_id="docs",
        tools=[ExternalToolDefinition(name="docs.search", description="Search")],
    )
    _patch_loader_and_adapter(
        monkeypatch,
        payload=_registry_payload(policy={"allow_tool_patterns": ["docs.*"]}),
        adapter=adapter,
    )

    async def _broker(**_kwargs) -> BrokeredExternalCredential:
        return BrokeredExternalCredential(
            headers={"Authorization": "Bearer ephemeral-token"},
            metadata={
                "credential_mode": "brokered_ephemeral",
                "credential_source": "mcp_hub_managed_binding",
            },
        )

    manager = ExternalServerManager().with_credential_broker(_broker)
    try:
        await manager.initialize()
        result = await manager.execute_virtual_tool(
            "ext.docs.docs.search",
            {"q": "runtime"},
            context={"request_id": "r-legacy"},
        )
    finally:
        await manager.shutdown()

    _ensure(
        result["metadata"] == {"adapter": "fake"},
        f"Legacy adapter metadata should not expose runtime auth internals: {result!r}",
    )


@pytest.mark.asyncio
async def test_ephemeral_adapter_helper_skips_runtime_auth_for_legacy_adapter_signature() -> None:
    cfg = parse_external_server_registry(_registry_payload()).servers[0]
    created_adapters: list[_FakeAdapter] = []

    def _adapter_factory(server_cfg) -> _FakeAdapter:
        adapter = _FakeAdapter(server_id=server_cfg.id, tools=[])
        created_adapters.append(adapter)
        return adapter

    result = await call_tool_with_ephemeral_adapter(
        server_config=cfg,
        adapter_factory=_adapter_factory,
        prepare_config=lambda _cfg: None,
        tool_name="docs.search",
        arguments={"q": "hello"},
    )

    _ensure(result.is_error is False, f"Unexpected ephemeral adapter result: {result!r}")
    _ensure(
        created_adapters[0].calls == [("docs.search", {"q": "hello"})],
        f"Legacy adapter did not receive the expected upstream call: {created_adapters[0].calls!r}",
    )


@pytest.mark.asyncio
async def test_discovery_filters_tools_and_unknown_virtual_tool_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    adapter = _FakeAdapter(
        server_id="docs",
        tools=[
            ExternalToolDefinition(name="docs.search", description="Search"),
            ExternalToolDefinition(name="docs.delete", description="Delete"),
        ],
    )
    _patch_loader_and_adapter(
        monkeypatch,
        payload=_registry_payload(
            policy={
                "allow_tool_patterns": ["docs.*"],
                "deny_tool_patterns": ["docs.delete"],
                "allow_writes": True,
                "require_write_confirmation": False,
            }
        ),
        adapter=adapter,
    )

    unused_config = tmp_path / "unused.yaml"
    unused_config.write_text("servers: []\n", encoding="utf-8")
    manager = ExternalServerManager(config_path=str(unused_config))
    try:
        await manager.initialize()
        virtual_names = [tool.virtual_name for tool in manager.list_virtual_tools()]
        _ensure(
            virtual_names == ["ext.docs.docs.search"],
            f"Unexpected discovered virtual tools: {virtual_names!r}",
        )

        with pytest.raises(ValueError, match="Unknown external virtual tool"):
            await manager.execute_virtual_tool("ext.docs.docs.delete", {})
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_write_tool_blocked_when_allow_writes_false(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(
        server_id="docs",
        tools=[
            ExternalToolDefinition(
                name="docs.update",
                description="Update",
                metadata={"category": "management"},
            )
        ],
    )
    _patch_loader_and_adapter(
        monkeypatch,
        payload=_registry_payload(
            policy={
                "allow_tool_patterns": ["docs.*"],
                "allow_writes": False,
                "require_write_confirmation": True,
            }
        ),
        adapter=adapter,
    )

    manager = ExternalServerManager()
    try:
        await manager.initialize()
        with pytest.raises(PermissionError, match="write tool"):
            await manager.execute_virtual_tool("ext.docs.docs.update", {})
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_write_tool_requires_confirmation_and_strips_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(
        server_id="docs",
        tools=[
            ExternalToolDefinition(
                name="docs.update",
                description="Update",
                metadata={"category": "management"},
            )
        ],
    )
    _patch_loader_and_adapter(
        monkeypatch,
        payload=_registry_payload(
            policy={
                "allow_tool_patterns": ["docs.*"],
                "allow_writes": True,
                "require_write_confirmation": True,
            }
        ),
        adapter=adapter,
    )

    manager = ExternalServerManager()
    try:
        await manager.initialize()
        with pytest.raises(PermissionError, match="Write confirmation required"):
            await manager.execute_virtual_tool("ext.docs.docs.update", {"title": "x"})

        result = await manager.execute_virtual_tool(
            "ext.docs.docs.update",
            {"title": "x", "__confirm_write": True},
        )
        _ensure(result["is_error"] is False, f"Unexpected write result payload: {result!r}")
        _ensure(adapter.calls[-1][0] == "docs.update", f"Unexpected upstream tool call: {adapter.calls[-1]!r}")
        _ensure("__confirm_write" not in adapter.calls[-1][1], f"Confirmation marker leaked upstream: {adapter.calls[-1]!r}")
        _ensure(adapter.calls[-1][1]["title"] == "x", f"Unexpected upstream arguments: {adapter.calls[-1]!r}")
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_refresh_partial_failure_clears_server_tools_and_reports_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(
        server_id="docs",
        tools=[ExternalToolDefinition(name="docs.search", description="Search")],
    )
    _patch_loader_and_adapter(monkeypatch, payload=_registry_payload(), adapter=adapter)

    manager = ExternalServerManager()
    try:
        await manager.initialize()
        _ensure(
            [tool.virtual_name for tool in manager.list_virtual_tools()] == ["ext.docs.docs.search"],
            f"Unexpected initial virtual tools: {manager.list_virtual_tools()!r}",
        )

        adapter.fail_list = True
        refresh = await manager.refresh_discovery(server_id="docs")
        _ensure(refresh["errors"].get("docs") == "discovery failed", f"Unexpected refresh payload: {refresh!r}")
        _ensure(manager.list_virtual_tools() == [], f"Virtual tools should be cleared after failed refresh: {manager.list_virtual_tools()!r}")

        servers = await manager.list_servers()
        _ensure(len(servers) == 1, f"Unexpected server listing: {servers!r}")
        row = servers[0]
        _ensure(row["id"] == "docs", f"Unexpected server row: {row!r}")
        _ensure(row["discovery_ok"] is False, f"Discovery failure should mark row degraded: {row!r}")
        _ensure(row["status"] == "degraded", f"Unexpected server status after refresh failure: {row!r}")
        _ensure(row["tool_count"] == 0, f"Tool count should be cleared after refresh failure: {row!r}")
        _ensure(row["last_error"] == "discovery failed", f"Unexpected discovery error row: {row!r}")
        telemetry = row["telemetry"]
        _ensure(telemetry["connect_attempts"] == 1, f"Unexpected connect telemetry: {telemetry!r}")
        _ensure(telemetry["connect_successes"] == 1, f"Unexpected connect telemetry: {telemetry!r}")
        _ensure(telemetry["discovery_attempts"] == 2, f"Unexpected discovery telemetry: {telemetry!r}")
        _ensure(telemetry["discovery_successes"] == 1, f"Unexpected discovery telemetry: {telemetry!r}")
        _ensure(telemetry["discovery_failures"] == 1, f"Unexpected discovery telemetry: {telemetry!r}")
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_telemetry_tracks_call_outcomes_and_policy_denials(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _FakeAdapter(
        server_id="docs",
        tools=[
            ExternalToolDefinition(name="docs.search", description="Search"),
            ExternalToolDefinition(
                name="docs.update",
                description="Update",
                metadata={"category": "management"},
            ),
        ],
    )
    _patch_loader_and_adapter(
        monkeypatch,
        payload=_registry_payload(
            policy={
                "allow_tool_patterns": ["docs.*"],
                "allow_writes": True,
                "require_write_confirmation": True,
            }
        ),
        adapter=adapter,
    )

    manager = ExternalServerManager()
    try:
        await manager.initialize()

        ok = await manager.execute_virtual_tool("ext.docs.docs.search", {"q": "x"})
        _ensure(ok["is_error"] is False, f"Unexpected successful search payload: {ok!r}")

        adapter.next_call_is_error = True
        upstream_err = await manager.execute_virtual_tool("ext.docs.docs.search", {"q": "y"})
        _ensure(upstream_err["is_error"] is True, f"Unexpected upstream error payload: {upstream_err!r}")

        adapter.next_call_exception = TimeoutError("upstream timeout")
        with pytest.raises(TimeoutError, match="upstream timeout"):
            await manager.execute_virtual_tool("ext.docs.docs.search", {"q": "z"})

        with pytest.raises(PermissionError, match="Write confirmation required"):
            await manager.execute_virtual_tool("ext.docs.docs.update", {"title": "no-confirm"})

        servers = await manager.list_servers()
        _ensure(len(servers) == 1, f"Unexpected server listing: {servers!r}")
        telemetry = servers[0]["telemetry"]
        _ensure(telemetry["connect_attempts"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["connect_successes"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["connect_failures"] == 0, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["discovery_attempts"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["discovery_successes"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["discovery_failures"] == 0, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["call_attempts"] == 3, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["call_successes"] == 2, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["call_failures"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["call_timeouts"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["call_upstream_errors"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["policy_denials"] == 1, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["last_discovered_tool_count"] == 2, f"Unexpected telemetry snapshot: {telemetry!r}")
        _ensure(telemetry["last_call_latency_ms"] is not None, f"Missing last call latency: {telemetry!r}")
        _ensure(telemetry["avg_call_latency_ms"] is not None, f"Missing average call latency: {telemetry!r}")
        _ensure(telemetry["last_error"] is not None, f"Missing last error in telemetry snapshot: {telemetry!r}")
    finally:
        await manager.shutdown()
