from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp_unified.federation.installers import ExternalServerInstaller
from mcp_unified.federation.models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
    FederationPolicyDenied,
)
from mcp_unified.gateway.external_runtime import (
    GatewayExternalRuntimeError,
    GatewayExternalRuntimeManager,
)
from mcp_unified.storage.models import AuditEvent, ExternalServerDefinition


class InMemoryExternalRegistryStore:
    """Copy-isolated external registry store for runtime manager tests."""

    def __init__(self, servers: list[ExternalServerDefinition] | None = None) -> None:
        self.servers = {
            server.id: server.model_copy(deep=True)
            for server in servers or ()
        }
        self.mutations = 0

    def set_server(self, server: ExternalServerDefinition) -> None:
        """Store a caller-owned server definition."""
        self.mutations += 1
        self.servers[server.id] = server.model_copy(deep=True)

    def delete_server(self, server_id: str) -> None:
        """Delete a server definition from the fake registry."""
        self.mutations += 1
        self.servers.pop(server_id, None)

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
        self.list_tools_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.runtime_auth_seen: BrokeredExternalCredential | None = None
        self.fail_list = False
        self.fail_health = False
        self.fail_call = False
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
        if self.fail_health:
            raise RuntimeError("health failed")
        return {
            "configured": True,
            "connected": self.connected,
            "initialized": self.connected,
        }

    async def list_tools(self) -> list[ExternalToolDefinition]:
        """Return configured tools or raise a discovery failure."""
        self.list_tools_count += 1
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
        if self.fail_call:
            raise RuntimeError("call failed")
        self.calls.append((tool_name, dict(arguments or {})))
        self.runtime_auth_seen = None if runtime_auth is None else runtime_auth.copy()
        return self.result.copy()


class BlockingConnectTransport(RecordingExternalTransport):
    """Fake transport that pauses connection until the test releases it."""

    def __init__(
        self,
        *,
        server_id: str,
        tools: list[ExternalToolDefinition] | None = None,
    ) -> None:
        super().__init__(server_id=server_id, tools=tools)
        self.connect_started = asyncio.Event()
        self.allow_connect = asyncio.Event()

    async def connect(self) -> None:
        """Wait for the test to allow connection completion."""
        self.connect_count += 1
        self.connect_started.set()
        await self.allow_connect.wait()
        self.connected = True


class RecordingCredentialBroker:
    """Credential broker fake that records resolution requests."""

    def __init__(
        self,
        result: BrokeredExternalCredential | None,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def resolve_external_credential(
        self,
        **kwargs: Any,
    ) -> BrokeredExternalCredential | None:
        """Record one broker request and return the configured result."""
        self.calls.append(dict(kwargs))
        return None if self.result is None else self.result.copy()


class UnsupportedInstaller(ExternalServerInstaller):
    """Installer fake that reports unsupported operations."""

    def __init__(self) -> None:
        self.install_calls: list[tuple[str, Any]] = []
        self.update_calls: list[tuple[str, Any]] = []

    async def install_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return an unsupported install response."""
        self.install_calls.append((server.id, context))
        return {
            "ok": False,
            "reason_code": "external_server_install_unsupported",
            "server_id": server.id,
        }

    async def update_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return an unsupported update response."""
        self.update_calls.append((server.id, context))
        return {
            "ok": False,
            "reason_code": "external_server_update_unsupported",
            "server_id": server.id,
        }

    async def get_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Return unavailable fake installer status."""
        return {"available": False, "server_id": server.id}


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
async def test_external_runtime_start_does_not_block_status_snapshot_during_connect() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    transport = BlockingConnectTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )

    start_task = asyncio.create_task(manager.start_server("research"))
    try:
        await asyncio.wait_for(transport.connect_started.wait(), timeout=1.0)
        rows = await asyncio.wait_for(manager.list_runtime_servers(), timeout=0.2)
    finally:
        transport.allow_connect.set()
        await start_task

    assert rows["servers"][0]["id"] == "research"
    assert rows["servers"][0]["status"] == "stopped"


@pytest.mark.asyncio
async def test_external_runtime_list_status_handles_health_check_failure() -> None:
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

    transport.fail_health = True
    rows = await manager.list_runtime_servers()

    row = rows["servers"][0]
    assert row["id"] == "research"
    assert row["status"] == "unhealthy"
    assert row["checks"]["error"] is True
    assert row["checks"]["error_type"] == "RuntimeError"
    assert row["last_error"] == "RuntimeError: health failed"


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


@pytest.mark.asyncio
async def test_external_runtime_refresh_failure_isolates_one_server() -> None:
    research = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    docs = RecordingExternalTransport(
        server_id="docs",
        tools=[ExternalToolDefinition(name="lookup")],
    )
    store = InMemoryExternalRegistryStore(
        [
            _server(),
            _server(id="docs", name="Docs", command=["fake-docs-mcp"]),
        ]
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: {"research": research, "docs": docs}[server.id],
    )
    await manager.start_server("research")
    await manager.start_server("docs")

    research.fail_list = True
    payload = await manager.refresh_server()
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    statuses = {row["id"]: row for row in rows["servers"]}
    assert payload["reason_code"] == "external_server_refreshed"
    assert payload["refreshed_servers"] == 1
    assert payload["errors"] == {"research": "external_server_discovery_failed"}
    assert statuses["research"]["status"] in {"degraded", "unhealthy"}
    assert statuses["research"]["tool_count"] == 0
    assert statuses["docs"]["status"] == "healthy"
    assert [tool.virtual_name for tool in tools] == ["ext.docs.lookup"]


@pytest.mark.asyncio
async def test_external_runtime_restart_reloads_registry_definition() -> None:
    first = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search", description="Old search")],
    )
    second = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="lookup", description="New lookup")],
    )
    transports = [first, second]
    store = InMemoryExternalRegistryStore([_server(name="Research v1")])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transports.pop(0),
    )
    await manager.start_server("research")
    store.set_server(_server(name="Research v2", command=["fake-research-mcp-v2"]))

    payload = await manager.restart_server("research")
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    assert payload["reason_code"] == "external_server_restarted"
    assert first.close_count == 1
    assert second.connect_count == 1
    assert rows["servers"][0]["name"] == "Research v2"
    assert [tool.virtual_name for tool in tools] == ["ext.research.lookup"]


@pytest.mark.asyncio
async def test_external_runtime_reconcile_starts_stops_restarts_and_refreshes() -> None:
    existing_servers = [
        _server(id="unchanged", name="Unchanged", command=["fake-unchanged"]),
        _server(id="changed", name="Changed", command=["fake-changed-v1"]),
        _server(id="to_disable", name="To Disable", command=["fake-disable"]),
        _server(id="to_delete", name="To Delete", command=["fake-delete"]),
    ]
    store = InMemoryExternalRegistryStore(existing_servers)
    unchanged = RecordingExternalTransport(
        server_id="unchanged",
        tools=[ExternalToolDefinition(name="read")],
    )
    changed_old = RecordingExternalTransport(
        server_id="changed",
        tools=[ExternalToolDefinition(name="old")],
    )
    changed_new = RecordingExternalTransport(
        server_id="changed",
        tools=[ExternalToolDefinition(name="new")],
    )
    to_disable = RecordingExternalTransport(
        server_id="to_disable",
        tools=[ExternalToolDefinition(name="stop_me")],
    )
    to_delete = RecordingExternalTransport(
        server_id="to_delete",
        tools=[ExternalToolDefinition(name="remove_me")],
    )
    added = RecordingExternalTransport(
        server_id="added",
        tools=[ExternalToolDefinition(name="created")],
    )
    transport_queues: dict[str, list[RecordingExternalTransport]] = {
        "unchanged": [unchanged],
        "changed": [changed_old, changed_new],
        "to_disable": [to_disable],
        "to_delete": [to_delete],
        "added": [added],
    }

    def factory(server: ExternalServerDefinition) -> RecordingExternalTransport:
        return transport_queues[server.id].pop(0)

    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=factory,
    )
    for server in existing_servers:
        await manager.start_server(server.id)

    unchanged.tools = [ExternalToolDefinition(name="read_again")]
    store.set_server(_server(id="changed", name="Changed", command=["fake-changed-v2"]))
    store.set_server(
        _server(
            id="to_disable",
            name="To Disable",
            command=["fake-disable"],
            enabled=False,
        )
    )
    store.delete_server("to_delete")
    store.set_server(
        _server(
            id="added",
            name="Added",
            command=["fake-added"],
            auto_start=True,
        )
    )

    payload = await manager.reconcile()
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    statuses = {row["id"]: row["status"] for row in rows["servers"]}
    assert payload["reason_code"] == "external_server_reconciled"
    assert payload["started_servers"] == 1
    assert payload["stopped_servers"] == 2
    assert payload["restarted_servers"] == 1
    assert payload["refreshed_servers"] == 1
    assert payload["errors"] == {}
    assert unchanged.list_tools_count == 2
    assert changed_old.close_count == 1
    assert changed_new.connect_count == 1
    assert to_disable.close_count == 1
    assert to_delete.close_count == 1
    assert added.connect_count == 1
    assert statuses["added"] == "healthy"
    assert statuses["changed"] == "healthy"
    assert statuses["to_disable"] == "disabled"
    assert [tool.virtual_name for tool in tools] == [
        "ext.added.created",
        "ext.changed.new",
        "ext.unchanged.read_again",
    ]


@pytest.mark.asyncio
async def test_external_runtime_execution_uses_brokered_credentials_without_leaking_secrets() -> None:
    secret_header = "Bearer do-not-leak-header"
    secret_env = "do-not-leak-env"
    audit = RecordingAuditStore()
    store = InMemoryExternalRegistryStore([_server(credential_slots=["api_key"])])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
        result=ExternalToolCallResult(
            content={"matches": ["paper-1"]},
            metadata={"adapter": "fake"},
        ),
    )
    broker = RecordingCredentialBroker(
        BrokeredExternalCredential(
            headers={"Authorization": secret_header},
            env={"TOKEN": secret_env},
            metadata={
                "credential_mode": "brokered_ephemeral",
                "credential_source": "test",
                "unsafe_note": secret_env,
            },
        )
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        audit_store=audit,
        credential_broker=broker,
    )
    await manager.start_server("research")

    result = await manager.execute_virtual_tool(
        "ext.research.search",
        {"query": "MCP"},
        effective_policy={
            "external_server_grants": [{"server_id": "research"}],
            "credential_grants": [
                {
                    "external_server_id": "research",
                    "credential_slot": "api_key",
                }
            ],
        },
        actor_id="user-1",
        context={"request_id": "r1"},
    )

    assert transport.runtime_auth_seen is not None
    assert transport.runtime_auth_seen.headers == {"Authorization": secret_header}
    assert transport.runtime_auth_seen.env == {"TOKEN": secret_env}
    assert result.content == {"matches": ["paper-1"]}
    assert result.metadata["adapter"] == "fake"
    assert result.metadata["credential_mode"] == "brokered_ephemeral"
    assert result.metadata["credential_source"] == "test"
    assert result.metadata["credential_injection"] == {
        "headers": ["Authorization"],
        "env": ["TOKEN"],
    }
    assert "unsafe_note" not in result.metadata
    serialized_public_data = json.dumps(
        {
            "result_metadata": result.metadata,
            "audit_payloads": [event.payload for event in audit.events],
        },
        sort_keys=True,
    )
    assert secret_header not in serialized_public_data
    assert secret_env not in serialized_public_data
    assert (await store.get_server("research")).credential_slots == ["api_key"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("effective_policy", "reason_code"),
    [
        ({"profile_id": "project-researcher"}, "external_server_not_granted"),
        (
            {
                "profile_id": "project-researcher",
                "external_server_grants": [{"server_id": "research"}],
                "denied_tools": ["ext.research.search"],
            },
            "tool_denied",
        ),
        (
            {
                "profile_id": "project-researcher",
                "external_server_grants": [{"server_id": "research"}],
                "allowed_tools": ["ext.research.lookup"],
            },
            "tool_not_allowed",
        ),
    ],
)
async def test_external_runtime_execution_enforces_policy_before_transport_call(
    effective_policy: dict[str, Any],
    reason_code: str,
) -> None:
    audit = RecordingAuditStore()
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        audit_store=audit,
    )
    await manager.start_server("research")

    with pytest.raises(FederationPolicyDenied) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.search",
            {"query": "MCP"},
            effective_policy=effective_policy,
            actor_id="user-1",
        )

    assert exc_info.value.reason_code == reason_code
    assert transport.calls == []
    denied = audit.events[-1]
    assert denied.event_type == "external_tool.denied"
    assert denied.actor_id == "user-1"
    assert denied.profile_id == "project-researcher"
    assert denied.payload["reason_code"] == reason_code
    assert denied.payload["virtual_tool_name"] == "ext.research.search"


@pytest.mark.asyncio
async def test_external_runtime_unknown_virtual_tool_reports_not_found() -> None:
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

    with pytest.raises(GatewayExternalRuntimeError) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.missing",
            {"query": "MCP"},
            effective_policy={"external_server_grants": [{"server_id": "research"}]},
        )

    assert exc_info.value.reason_code == "external_virtual_tool_not_found"
    assert transport.calls == []


@pytest.mark.asyncio
async def test_external_runtime_execution_wraps_transport_call_errors() -> None:
    audit = RecordingAuditStore()
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        audit_store=audit,
    )
    await manager.start_server("research")

    transport.fail_call = True
    with pytest.raises(GatewayExternalRuntimeError) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.search",
            {"query": "MCP"},
            effective_policy={"external_server_grants": [{"server_id": "research"}]},
            actor_id="user-1",
        )

    assert exc_info.value.reason_code == "external_tool_call_failed"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    failed = audit.events[-1]
    assert failed.event_type == "external_tool.failed"
    assert failed.actor_id == "user-1"
    assert failed.payload["reason_code"] == "external_tool_call_failed"
    assert failed.payload["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_external_runtime_credential_broker_accepts_noncopyable_policy_and_context() -> None:
    class NonCopyablePolicy:
        profile_id = "project-researcher"
        external_server_grants = [{"server_id": "research"}]
        credential_grants = [
            {
                "external_server_id": "research",
                "credential_slot": "api_key",
            }
        ]
        allowed_tools: list[str] = []
        denied_tools: list[str] = []

        def __deepcopy__(self, memo: dict[int, Any]) -> NonCopyablePolicy:
            raise RuntimeError("policy cannot be copied")

    class NonCopyableContext:
        def __deepcopy__(self, memo: dict[int, Any]) -> NonCopyableContext:
            raise RuntimeError("context cannot be copied")

    policy = NonCopyablePolicy()
    context = NonCopyableContext()
    store = InMemoryExternalRegistryStore([_server(credential_slots=["api_key"])])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    broker = RecordingCredentialBroker(
        BrokeredExternalCredential(headers={"Authorization": "Bearer test"})
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        credential_broker=broker,
    )
    await manager.start_server("research")

    result = await manager.execute_virtual_tool(
        "ext.research.search",
        {"query": "MCP"},
        effective_policy=policy,
        context=context,
    )

    assert result.content == {"ok": True}
    assert broker.calls[0]["effective_policy"] is policy
    assert broker.calls[0]["context"] is context
    assert transport.runtime_auth_seen is not None


@pytest.mark.asyncio
async def test_external_runtime_required_credentials_fail_closed_without_broker() -> None:
    store = InMemoryExternalRegistryStore([_server(credential_slots=["api_key"])])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")

    with pytest.raises(GatewayExternalRuntimeError) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.search",
            {"query": "MCP"},
            effective_policy={
                "external_server_grants": [{"server_id": "research"}],
                "credential_grants": [
                    {
                        "external_server_id": "research",
                        "credential_slot": "api_key",
                    }
                ],
            },
        )

    assert exc_info.value.reason_code == "credential_broker_unavailable"
    assert transport.calls == []


@pytest.mark.asyncio
async def test_external_runtime_required_credentials_fail_closed_when_broker_returns_none() -> None:
    store = InMemoryExternalRegistryStore([_server(credential_slots=["api_key"])])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    broker = RecordingCredentialBroker(None)
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        credential_broker=broker,
    )
    await manager.start_server("research")

    with pytest.raises(FederationPolicyDenied) as exc_info:
        await manager.execute_virtual_tool(
            "ext.research.search",
            {"query": "MCP"},
            effective_policy={
                "external_server_grants": [{"server_id": "research"}],
                "credential_grants": [
                    {
                        "external_server_id": "research",
                        "credential_slot": "api_key",
                    }
                ],
            },
        )

    assert exc_info.value.reason_code == "required_credential_grant_missing"
    assert broker.calls
    assert transport.calls == []


@pytest.mark.asyncio
async def test_external_runtime_default_install_update_are_not_configured() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
    )

    install = await manager.install_server("research")
    update = await manager.update_server("research")

    assert install["reason_code"] == "external_server_install_not_configured"
    assert update["reason_code"] == "external_server_update_not_configured"
    assert install["available"] is False
    assert update["available"] is False
    assert store.mutations == 0


@pytest.mark.asyncio
async def test_external_runtime_installer_unsupported_does_not_mutate_runtime_state() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        tools=[ExternalToolDefinition(name="search")],
    )
    installer = UnsupportedInstaller()
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
        installer=installer,
    )
    await manager.start_server("research")

    install = await manager.install_server("research", context={"request_id": "install"})
    update = await manager.update_server("research", context={"request_id": "update"})
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    assert install["reason_code"] == "external_server_install_unsupported"
    assert update["reason_code"] == "external_server_update_unsupported"
    assert installer.install_calls == [("research", {"request_id": "install"})]
    assert installer.update_calls == [("research", {"request_id": "update"})]
    assert store.mutations == 0
    assert transport.connect_count == 1
    assert transport.close_count == 0
    assert rows["servers"][0]["status"] == "healthy"
    assert [tool.virtual_name for tool in tools] == ["ext.research.search"]
