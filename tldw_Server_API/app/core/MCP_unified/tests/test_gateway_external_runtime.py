from __future__ import annotations

from typing import Any

import pytest

from mcp_unified.federation.models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
)
from mcp_unified.gateway.external_runtime import GatewayExternalRuntimeManager
from mcp_unified.storage.models import AuditEvent, ExternalServerDefinition


class InMemoryExternalRegistryStore:
    """Copy-isolated external registry store for runtime manager tests."""

    def __init__(self, servers: list[ExternalServerDefinition] | None = None) -> None:
        self.servers = {
            server.id: server.model_copy(deep=True)
            for server in servers or ()
        }

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
        """Return one server definition by id."""
        server = self.servers.get(server_id)
        return None if server is None else server.model_copy(deep=True)

    async def list_servers(self) -> list[ExternalServerDefinition]:
        """Return runtime-compatible server rows."""
        return await self.list_server_definitions()

    async def list_server_definitions(
        self,
        *,
        enabled: bool | None = None,
    ) -> list[ExternalServerDefinition]:
        """Return typed server definitions matching the enabled filter."""
        rows = [
            server
            for server in self.servers.values()
            if enabled is None or server.enabled is enabled
        ]
        return [
            server.model_copy(deep=True)
            for server in sorted(rows, key=lambda item: item.id)
        ]


class RecordingAuditStore:
    """Audit sink that preserves appended events for assertions."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append_event(self, event: AuditEvent) -> AuditEvent:
        """Record and return a copy of the event."""
        stored = event.model_copy(deep=True)
        self.events.append(stored)
        return stored.model_copy(deep=True)

    async def query_events(
        self,
        *,
        actor_id: str | None = None,
        profile_id: str | None = None,
        event_type: str | None = None,
        limit: int | None = None,
    ) -> list[AuditEvent]:
        """Return matching audit events."""
        rows = [
            event
            for event in self.events
            if (actor_id is None or event.actor_id == actor_id)
            and (profile_id is None or event.profile_id == profile_id)
            and (event_type is None or event.event_type == event_type)
        ]
        if limit is not None:
            rows = rows[:limit]
        return [event.model_copy(deep=True) for event in rows]


class RecordingExternalTransport:
    """Fake external transport that records lifecycle and call behavior."""

    transport_name = "fake"

    def __init__(
        self,
        *,
        server_id: str,
        tools: list[ExternalToolDefinition] | None = None,
        result: ExternalToolCallResult | None = None,
    ) -> None:
        self.server_id = server_id
        self.connected = False
        self.connect_count = 0
        self.close_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.runtime_auth_seen: BrokeredExternalCredential | None = None
        self.fail_list = False
        self.tools = [tool.copy() for tool in tools or ()]
        self.result = result or ExternalToolCallResult(content={"ok": True})

    async def connect(self) -> None:
        """Mark the fake transport connected."""
        self.connect_count += 1
        self.connected = True

    async def close(self) -> None:
        """Mark the fake transport closed."""
        self.close_count += 1
        self.connected = False

    async def health_check(self) -> dict[str, bool]:
        """Return deterministic fake health."""
        return {
            "configured": True,
            "connected": self.connected,
            "initialized": self.connected,
        }

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Return configured tools or raise a discovery failure."""
        if self.fail_list:
            raise RuntimeError("discovery failed")
        return [tool.copy() for tool in self.tools]

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        context: Any = None,
        runtime_auth: BrokeredExternalCredential | None = None,
    ) -> ExternalToolCallResult:
        """Record and return the configured call result."""
        del context
        self.calls.append((tool_name, dict(arguments or {})))
        self.runtime_auth_seen = None if runtime_auth is None else runtime_auth.copy()
        return self.result.copy()


def _server(**overrides: Any) -> ExternalServerDefinition:
    values: dict[str, Any] = {
        "id": "research",
        "name": "Research",
        "transport": "stdio",
        "command": ["fake-research-mcp"],
    }
    values.update(overrides)
    return ExternalServerDefinition(**values)


@pytest.mark.asyncio
async def test_external_runtime_start_discovers_tools_and_reports_healthy_status() -> None:
    audit = RecordingAuditStore()
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search", description="Search papers")],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        audit_store=audit,
    )

    payload = await manager.start_server("research")
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    assert payload["ok"] is True
    assert payload["reason_code"] == "external_server_started"
    assert rows["servers"][0]["id"] == "research"
    assert rows["servers"][0]["status"] == "healthy"
    assert rows["servers"][0]["tool_count"] == 1
    assert [tool.virtual_name for tool in tools] == ["ext.research.search"]
    assert transport.connect_count == 1
    assert [event.payload["reason_code"] for event in audit.events] == [
        "external_server_started",
        "external_server_discovered",
    ]


@pytest.mark.asyncio
async def test_external_runtime_stop_is_idempotent_and_clears_tools() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")

    first = await manager.stop_server("research")
    second = await manager.stop_server("research")
    rows = await manager.list_runtime_servers()

    assert first["reason_code"] == "external_server_stopped"
    assert second["reason_code"] == "external_server_already_stopped"
    assert await manager.list_virtual_tools() == []
    assert rows["servers"][0]["status"] == "stopped"
    assert rows["servers"][0]["tool_count"] == 0
    assert transport.close_count == 1
