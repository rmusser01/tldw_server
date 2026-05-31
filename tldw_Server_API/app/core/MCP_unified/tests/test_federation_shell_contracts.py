"""Tests for the standalone MCP external federation shell contracts."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

import pytest
from mcp_unified.profiles.resolution import EffectivePolicy
from mcp_unified.storage import AuditEvent, ExternalServerDefinition


def _tldw_imports_for(path: Path) -> list[str]:
    """Return imports from a Python file that cross into the host package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


class _MemoryExternalRegistryStore:
    """In-memory external registry test double for package contract tests."""

    def __init__(self, servers: list[ExternalServerDefinition]) -> None:
        self._servers = {server.id: server for server in servers}

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
        return self._servers.get(server_id)

    async def list_servers(self) -> list[ExternalServerDefinition]:
        return list(self._servers.values())

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        rows = list(self._servers.values())
        if enabled is None:
            return rows
        return [server for server in rows if server.enabled is enabled]

    async def upsert_server(
        self,
        server: ExternalServerDefinition,
    ) -> ExternalServerDefinition:
        self._servers[server.id] = server
        return server

    async def delete_server(self, server_id: str) -> bool:
        return self._servers.pop(server_id, None) is not None


class _MemoryAuditStore:
    """In-memory audit sink that preserves appended events for assertions."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        self.events.append(event)
        return event

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        rows = [
            event
            for event in self.events
            if (actor_id is None or event.actor_id == actor_id)
            and (profile_id is None or event.profile_id == profile_id)
            and (event_type is None or event.event_type == event_type)
        ]
        return rows[:limit] if limit is not None else rows


def _research_server(**overrides: Any) -> ExternalServerDefinition:
    values: dict[str, Any] = {
        "id": "research",
        "name": "Research",
        "transport": "stdio",
        "command": ["fake-mcp-research"],
    }
    values.update(overrides)
    return ExternalServerDefinition(**values)


def test_federation_package_has_no_tldw_server_imports() -> None:
    federation = importlib.import_module("mcp_unified.federation")

    assert federation.__file__ is not None
    federation_root = Path(federation.__file__).resolve().parent
    offenders: dict[str, list[str]] = {}
    for path in federation_root.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


@pytest.mark.asyncio
async def test_non_spawning_manager_loads_registry_and_virtual_tools() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolCallResult,
        ExternalToolDefinition,
        FakeExternalTransport,
    )

    store = _MemoryExternalRegistryStore([_research_server()])
    transports: dict[str, FakeExternalTransport] = {}

    def _transport_factory(server: ExternalServerDefinition) -> FakeExternalTransport:
        transport = FakeExternalTransport(
            server_id=server.id,
            tools=[
                ExternalToolDefinition(
                    name="search",
                    description="Search research indexes",
                    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
                    metadata={"category": "search"},
                )
            ],
            results={
                "search": ExternalToolCallResult(content={"matches": ["paper-1"]}),
            },
        )
        transports[server.id] = transport
        return transport

    manager = ExternalFederationManager(
        registry_store=store,
        transport_factory=_transport_factory,
    )

    await manager.start()

    server_rows = await manager.list_servers()
    virtual_tools = await manager.list_virtual_tools()

    assert server_rows == [
        {
            "id": "research",
            "name": "Research",
            "transport": "stdio",
            "tool_count": 1,
            "status": "healthy",
            "checks": {"configured": True, "connected": True, "spawns_process": False},
            "last_error": None,
        }
    ]
    assert [tool.virtual_name for tool in virtual_tools] == ["ext.research.search"]
    assert virtual_tools[0].server_id == "research"
    assert virtual_tools[0].upstream_tool_name == "search"
    assert virtual_tools[0].metadata["category"] == "search"
    assert transports["research"].connect_count == 1
    assert transports["research"].spawn_count == 0


@pytest.mark.asyncio
async def test_external_execution_requires_profile_server_grant_and_audits_denial() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolCallResult,
        ExternalToolDefinition,
        FakeExternalTransport,
        FederationPolicyDenied,
    )

    audit = _MemoryAuditStore()
    transport = FakeExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
        results={"search": ExternalToolCallResult(content={"matches": []})},
    )
    manager = ExternalFederationManager(
        registry_store=_MemoryExternalRegistryStore([_research_server()]),
        transport_factory=lambda _server: transport,
        audit_store=audit,
    )
    await manager.start()

    with pytest.raises(FederationPolicyDenied) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.search",
            {"query": "MCP"},
            effective_policy=EffectivePolicy(profile_id="project-researcher"),
            actor_id="user-1",
        )

    assert exc_info.value.reason_code == "external_server_not_granted"
    assert transport.calls == []
    assert [event.event_type for event in audit.events] == [
        "external_server.lifecycle",
        "external_server.discovery",
        "external_tool.denied",
    ]
    deny_event = audit.events[-1]
    assert deny_event.actor_id == "user-1"
    assert deny_event.profile_id == "project-researcher"
    assert deny_event.payload["reason_code"] == "external_server_not_granted"
    assert deny_event.payload["virtual_tool_name"] == "ext.research.search"


@pytest.mark.asyncio
async def test_external_execution_allows_granted_tool_and_audits_success() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolCallResult,
        ExternalToolDefinition,
        FakeExternalTransport,
    )

    audit = _MemoryAuditStore()
    transport = FakeExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
        results={"search": ExternalToolCallResult(content={"matches": ["paper-1"]})},
    )
    manager = ExternalFederationManager(
        registry_store=_MemoryExternalRegistryStore([_research_server()]),
        transport_factory=lambda _server: transport,
        audit_store=audit,
    )
    await manager.start()

    result = await manager.execute_virtual_tool(
        "ext.research.search",
        {"query": "MCP"},
        effective_policy=EffectivePolicy(
            profile_id="deep-researcher",
            allowed_tools=["ext.research.search"],
            external_server_grants=[{"server_id": "research"}],
        ),
        actor_id="user-1",
    )

    assert result.content == {"matches": ["paper-1"]}
    assert result.server_id == "research"
    assert result.upstream_tool_name == "search"
    assert transport.calls == [("search", {"query": "MCP"})]
    assert audit.events[-1].event_type == "external_tool.allowed"
    assert audit.events[-1].payload["reason_code"] == "allowed"


@pytest.mark.asyncio
async def test_start_rolls_back_connected_transports_on_discovery_failure() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolDefinition,
        FakeExternalTransport,
    )

    class _FailingDiscoveryTransport(FakeExternalTransport):
        """Fake transport that connects but fails during tool discovery."""

        async def list_tools(self) -> list[ExternalToolDefinition]:
            raise RuntimeError("discovery failed")

    research_transport = FakeExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    broken_transport = _FailingDiscoveryTransport(server_id="broken")
    transports = {
        "research": research_transport,
        "broken": broken_transport,
    }
    manager = ExternalFederationManager(
        registry_store=_MemoryExternalRegistryStore(
            [
                _research_server(),
                _research_server(id="broken", name="Broken"),
            ]
        ),
        transport_factory=lambda server: transports[server.id],
    )

    with pytest.raises(RuntimeError, match="discovery failed"):
        await manager.start()

    assert manager.started is False
    assert research_transport.connected is False
    assert broken_transport.connected is False
    assert research_transport.close_count == 1
    assert broken_transport.close_count == 1
    assert await manager.list_servers() == []


@pytest.mark.asyncio
async def test_stop_clears_state_when_close_and_audit_failures_occur() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolDefinition,
        FakeExternalTransport,
    )

    class _FailingCloseTransport(FakeExternalTransport):
        """Fake transport that fails during close after marking itself closed."""

        async def close(self) -> None:
            await super().close()
            raise RuntimeError("close failed")

    class _FailingStopAuditStore(_MemoryAuditStore):
        """Audit sink that fails only for stop lifecycle events."""

        async def append_event(self, event: AuditEvent) -> AuditEvent:
            if (
                event.event_type == "external_server.lifecycle"
                and event.payload.get("reason_code") == "stopped"
            ):
                raise RuntimeError("audit failed")
            return await super().append_event(event)

    transport = _FailingCloseTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = ExternalFederationManager(
        registry_store=_MemoryExternalRegistryStore([_research_server()]),
        transport_factory=lambda _server: transport,
        audit_store=_FailingStopAuditStore(),
    )
    await manager.start()

    await manager.stop()

    assert manager.started is False
    assert transport.connected is False
    assert await manager.list_servers() == []
    assert [error["operation"] for error in manager.last_lifecycle_errors] == [
        "close",
        "audit",
    ]


@pytest.mark.asyncio
async def test_refresh_audits_discovery_error_details() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolDefinition,
        FakeExternalTransport,
    )

    class _ToggleDiscoveryTransport(FakeExternalTransport):
        """Fake transport that can fail discovery after initial startup."""

        fail_discovery = False

        async def list_tools(self) -> list[ExternalToolDefinition]:
            if self.fail_discovery:
                raise RuntimeError("refresh failed")
            return await super().list_tools()

    audit = _MemoryAuditStore()
    transport = _ToggleDiscoveryTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = ExternalFederationManager(
        registry_store=_MemoryExternalRegistryStore([_research_server()]),
        transport_factory=lambda _server: transport,
        audit_store=audit,
    )
    await manager.start()
    transport.fail_discovery = True

    result = await manager.refresh()

    assert result["errors"] == {"research": "external_server_discovery_failed"}
    assert audit.events[-1].event_type == "external_server.discovery"
    assert audit.events[-1].payload["reason_code"] == "external_server_discovery_failed"
    assert audit.events[-1].payload["error_type"] == "RuntimeError"
    assert audit.events[-1].payload["error_message"] == "refresh failed"


@pytest.mark.asyncio
async def test_external_execution_requires_credential_grants_for_required_slots() -> None:
    from mcp_unified.federation import (
        ExternalFederationManager,
        ExternalToolCallResult,
        ExternalToolDefinition,
        FakeExternalTransport,
        FederationPolicyDenied,
    )

    transport = FakeExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
        results={"search": ExternalToolCallResult(content={"matches": ["paper-1"]})},
    )
    manager = ExternalFederationManager(
        registry_store=_MemoryExternalRegistryStore(
            [_research_server(credential_slots=["research_api"])]
        ),
        transport_factory=lambda _server: transport,
    )
    await manager.start()

    with pytest.raises(FederationPolicyDenied) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.search",
            {"query": "MCP"},
            effective_policy=EffectivePolicy(
                profile_id="deep-researcher",
                external_server_grants=[{"server_id": "research"}],
            ),
        )

    assert exc_info.value.reason_code == "required_credential_grant_missing"
    assert transport.calls == []

    result = await manager.execute_virtual_tool(
        "ext.research.search",
        {"query": "MCP"},
        effective_policy=EffectivePolicy(
            profile_id="deep-researcher",
            external_server_grants=[{"server_id": "research"}],
            credential_grants=[
                {
                    "external_server_id": "research",
                    "credential_slot": "research_api",
                }
            ],
        ),
    )

    assert result.content == {"matches": ["paper-1"]}
