from __future__ import annotations

import asyncio
import json
from typing import Any

import mcp_unified.gateway.external_runtime as gateway_external_runtime
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
from mcp_unified.gateway.external_runtime_adapter import ExternalRuntimeGatewayRuntime
from mcp_unified.gateway.runtime import GatewayRequestContext
from mcp_unified.interfaces.storage import ExternalRegistryStoreUnavailableError
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


class ToggleOutageExternalRegistryStore(InMemoryExternalRegistryStore):
    """Registry store fake that can fail get_server calls after startup."""

    def __init__(self, servers: list[ExternalServerDefinition] | None = None) -> None:
        super().__init__(servers)
        self.fail_get_server = False

    async def get_server(self, server_id: str) -> ExternalServerDefinition | None:
        """Return one server definition or simulate store unavailability."""
        if self.fail_get_server:
            raise ExternalRegistryStoreUnavailableError("registry unavailable")
        return await super().get_server(server_id)


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
        resources: list[dict[str, Any]] | None = None,
        resource_reads: dict[str, dict[str, Any]] | None = None,
        result: ExternalToolCallResult | None = None,
    ) -> None:
        self.server_id = server_id
        self.connected = False
        self.connect_count = 0
        self.close_count = 0
        self.list_tools_count = 0
        self.list_resources_count = 0
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.resource_reads: list[str] = []
        self.runtime_auth_seen: BrokeredExternalCredential | None = None
        self.fail_list = False
        self.fail_resources = False
        self.fail_health = False
        self.fail_call = False
        self.tools = [tool.copy() for tool in tools or ()]
        self.resources = [dict(resource) for resource in resources or ()]
        self.resource_results = {
            uri: dict(result)
            for uri, result in (resource_reads or {}).items()
        }
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

    async def list_resources(self) -> list[dict[str, Any]]:
        """Return configured resources or raise a discovery failure."""
        self.list_resources_count += 1
        if self.fail_resources:
            raise RuntimeError("resource discovery failed")
        return [dict(resource) for resource in self.resources]

    async def read_resource(self, uri: str, *, context: Any = None) -> dict[str, Any]:
        """Record and return a configured resource read."""
        del context
        self.resource_reads.append(uri)
        if uri not in self.resource_results:
            raise RuntimeError("resource missing")
        return dict(self.resource_results[uri])

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


class FakeLogger:
    """Logger fake that records structured exception log calls."""

    def __init__(self) -> None:
        self.opt_calls: list[dict[str, Any]] = []
        self.error_calls: list[tuple[str, tuple[Any, ...]]] = []

    def opt(self, **kwargs: Any) -> FakeLogger:
        self.opt_calls.append(kwargs)
        return self

    def error(self, message: str, *args: Any) -> None:
        self.error_calls.append((message, args))


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


class BlockingResourceListTransport(RecordingExternalTransport):
    """Fake transport that pauses resource discovery until released."""

    def __init__(
        self,
        *,
        server_id: str,
        resources: list[dict[str, Any]],
        allow_list: asyncio.Event,
    ) -> None:
        super().__init__(server_id=server_id, resources=resources)
        self.list_started = asyncio.Event()
        self.allow_list = allow_list

    async def list_resources(self) -> list[dict[str, Any]]:
        """Wait for the test to release resource discovery."""
        self.list_resources_count += 1
        self.list_started.set()
        await self.allow_list.wait()
        return [dict(resource) for resource in self.resources]


class BaseResourceRuntime:
    """Base gateway runtime fake used by adapter resource tests."""

    name = "base-runtime"
    version = "0.0-test"

    def __init__(self) -> None:
        """Initialize request recording state."""
        self.read_requests: list[str] = []

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no base tools for resource-focused tests."""
        del context
        return []

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Return an empty tool result for unused tool calls."""
        del name, arguments, context
        return {}

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return one local resource before external resources."""
        del context
        return [{"uri": "resource://local/doc", "name": "Local Doc"}]

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Record the local read request and return local content."""
        del context
        self.read_requests.append(uri)
        return {"contents": [{"uri": uri, "text": "local"}]}

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no prompts for resource-focused tests."""
        del context
        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Return an empty prompt payload for unused prompt calls."""
        del name, arguments, context
        return {}

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no modules for resource-focused tests."""
        del context
        return []

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Return an empty module health payload."""
        del context
        return {"modules": []}


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


class SuccessfulInstaller(ExternalServerInstaller):
    """Installer fake that returns rich public and secret-looking metadata."""

    def __init__(self) -> None:
        self.install_calls: list[tuple[str, Any]] = []
        self.update_calls: list[tuple[str, Any]] = []

    async def install_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return a successful install payload with nested sensitive fields."""
        self.install_calls.append((server.id, context))
        return self._payload(
            server.id,
            reason_code="external_server_installed",
            version="1.2.3",
        )

    async def update_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return a successful update payload with nested sensitive fields."""
        self.update_calls.append((server.id, context))
        return self._payload(
            server.id,
            reason_code="external_server_updated",
            installed_version="1.2.4",
        )

    async def get_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Return available fake installer status with nested sensitive fields."""
        return self._payload(
            server.id,
            ok=True,
            reason_code="external_server_installer_available",
            latest_version="1.2.4",
        )

    @staticmethod
    def _payload(
        server_id: str,
        *,
        reason_code: str,
        ok: bool = True,
        version: str | None = None,
        installed_version: str | None = None,
        latest_version: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": ok,
            "available": True,
            "reason_code": reason_code,
            "server_id": server_id,
            "installer": "fake",
            "message": "ready",
            "details": {
                "channel": "stable",
                "authorization": "Bearer do-not-leak",
                "headers": {"Authorization": "Bearer do-not-leak"},
                "env": {"TOKEN": "do-not-leak-env"},
                "nested": [
                    {
                        "safe": "kept",
                        "password": "do-not-leak-password",
                    }
                ],
            },
            "command": ["do-not-leak-command"],
            "credential_slots": ["api_key"],
        }
        if version is not None:
            payload["version"] = version
        if installed_version is not None:
            payload["installed_version"] = installed_version
        if latest_version is not None:
            payload["latest_version"] = latest_version
        return payload


class ExplodingInstaller(ExternalServerInstaller):
    """Installer fake that raises from all adapter methods."""

    async def install_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Raise a failure containing text that must not become public."""
        del server, context
        raise RuntimeError("install failed with do-not-leak-token")

    async def update_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Raise a failure containing text that must not become public."""
        del server, context
        raise RuntimeError("update failed with do-not-leak-token")

    async def get_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Raise a failure containing text that must not become public."""
        del server
        raise RuntimeError("status failed with do-not-leak-token")


class HangingInstaller(ExternalServerInstaller):
    """Installer fake that never returns status for timeout coverage."""

    async def install_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return an unused install payload."""
        del context
        return {"ok": True, "available": True, "server_id": server.id}

    async def update_server(
        self,
        server: ExternalServerDefinition,
        *,
        context: Any = None,
    ) -> dict[str, Any]:
        """Return an unused update payload."""
        del context
        return {"ok": True, "available": True, "server_id": server.id}

    async def get_status(
        self,
        server: ExternalServerDefinition,
    ) -> dict[str, Any]:
        """Block until cancelled by the runtime status timeout."""
        del server
        await asyncio.Event().wait()
        return {"available": True}


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
async def test_external_runtime_lists_and_reads_redacted_resources() -> None:
    """External resources and read payloads expose only redacted virtual URIs."""
    upstream_uri = "secret://docs/source?token=do-not-leak"
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        resources=[
            {
                "uri": upstream_uri,
                "name": "Secret Doc",
                "description": "Safe description",
                "mimeType": "text/plain",
                "metadata": {
                    "category": "docs",
                    "api_key": "do-not-leak",
                },
            }
        ],
        resource_reads={
            upstream_uri: {
                "uri": upstream_uri,
                "sourceUri": f"{upstream_uri}#source",
                "metadata": {"href": "secret://docs/related?token=do-not-leak"},
                "contents": [
                    {
                        "uri": upstream_uri,
                        "mimeType": "text/plain",
                        "text": f"external contents from {upstream_uri}",
                    },
                    {
                        "uri": "secret://docs/related?token=do-not-leak",
                        "type": "text",
                        "text": "see secret://docs/related?token=do-not-leak",
                    }
                ]
            }
        },
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")

    resources = await manager.list_virtual_resources()
    public_payload = json.dumps(resources, sort_keys=True)
    resource = resources[0]
    result = await manager.read_virtual_resource(resource["uri"])

    assert resource["uri"].startswith("external://research/")
    assert resource["name"] == "Secret Doc"
    assert resource["mimeType"] == "text/plain"
    assert resource["metadata"] == {
        "external_server_id": "research",
        "source": "external_runtime",
    }
    assert "do-not-leak" not in public_payload
    assert upstream_uri not in public_payload
    private_payload = json.dumps(result, sort_keys=True)
    assert "do-not-leak" not in private_payload
    assert "secret://docs/" not in private_payload
    assert result["contents"][0]["uri"] == resource["uri"]
    assert [content["uri"] for content in result["contents"]] == [
        resource["uri"],
        resource["uri"],
    ]
    assert transport.resource_reads == [upstream_uri]


@pytest.mark.asyncio
async def test_external_runtime_redacts_related_file_uris_in_resource_read_text() -> None:
    upstream_uri = "file:///Users/example/private/source.txt"
    related_uri = "file:///Users/example/private/related.txt"
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        resources=[{"uri": upstream_uri, "name": "Local secret"}],
        resource_reads={
            upstream_uri: {
                "contents": [
                    {
                        "uri": upstream_uri,
                        "mimeType": "text/plain",
                        "text": f"see {related_uri}",
                    }
                ]
            }
        },
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")
    resource = (await manager.list_virtual_resources())[0]

    result = await manager.read_virtual_resource(resource["uri"])
    payload = json.dumps(result, sort_keys=True)

    assert "file:///" not in payload
    assert result["contents"][0]["uri"] == resource["uri"]
    assert result["contents"][0]["text"] == f"see {resource['uri']}"


@pytest.mark.asyncio
async def test_external_runtime_resource_listing_filters_ungranted_servers_before_discovery() -> None:
    """Resource discovery skips active servers outside the effective profile policy."""
    research_uri = "resource://research/one"
    docs_uri = "resource://docs/one"
    store = InMemoryExternalRegistryStore(
        [
            _server(id="research", name="Research"),
            _server(id="docs", name="Docs"),
        ]
    )
    research = RecordingExternalTransport(
        server_id="research",
        resources=[{"uri": research_uri, "name": "Research Doc"}],
    )
    docs = RecordingExternalTransport(
        server_id="docs",
        resources=[{"uri": docs_uri, "name": "Docs Doc"}],
    )
    transports = {"research": research, "docs": docs}
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: transports[server.id],
    )
    await manager.start_server("research")
    await manager.start_server("docs")

    resources = await manager.list_virtual_resources(
        effective_policy={"external_server_grants": [{"server_id": "research"}]}
    )

    assert [resource["name"] for resource in resources] == ["Research Doc"]
    assert research.list_resources_count == 1
    assert docs.list_resources_count == 0


@pytest.mark.asyncio
async def test_external_runtime_resource_discovery_runs_active_servers_concurrently() -> None:
    """Resource discovery starts all allowed active servers before awaiting completion."""
    allow_list = asyncio.Event()
    store = InMemoryExternalRegistryStore(
        [
            _server(id="research", name="Research"),
            _server(id="docs", name="Docs"),
        ]
    )
    research = BlockingResourceListTransport(
        server_id="research",
        resources=[{"uri": "resource://research/one", "name": "Research Doc"}],
        allow_list=allow_list,
    )
    docs = BlockingResourceListTransport(
        server_id="docs",
        resources=[{"uri": "resource://docs/one", "name": "Docs Doc"}],
        allow_list=allow_list,
    )
    transports = {"research": research, "docs": docs}
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: transports[server.id],
    )
    await manager.start_server("research")
    await manager.start_server("docs")

    list_task = asyncio.create_task(manager.list_virtual_resources())
    try:
        await asyncio.wait_for(research.list_started.wait(), timeout=0.2)
        await asyncio.wait_for(docs.list_started.wait(), timeout=0.2)
    finally:
        allow_list.set()
    resources = await asyncio.wait_for(list_task, timeout=0.2)

    assert [resource["name"] for resource in resources] == ["Docs Doc", "Research Doc"]


@pytest.mark.asyncio
async def test_external_runtime_resource_read_requires_credential_grants_before_transport_call() -> None:
    """External resource reads fail before transport calls when credentials are ungranted."""
    upstream_uri = "resource://docs/secret"
    store = InMemoryExternalRegistryStore([_server(credential_slots=["api_key"])])
    transport = RecordingExternalTransport(
        server_id="research",
        resources=[{"uri": upstream_uri, "name": "Secret"}],
        resource_reads={upstream_uri: {"contents": [{"uri": upstream_uri, "text": "secret"}]}},
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")
    resources = await manager.list_virtual_resources()

    with pytest.raises(FederationPolicyDenied) as exc_info:
        await manager.read_virtual_resource(
            resources[0]["uri"],
            effective_policy={"external_server_grants": [{"server_id": "research"}]},
        )

    assert exc_info.value.reason_code == "required_credential_grant_missing"
    assert transport.resource_reads == []


@pytest.mark.asyncio
async def test_external_runtime_resource_discovery_success_clears_last_error() -> None:
    """A successful rediscovery clears the server's stale resource discovery error."""
    upstream_uri = "resource://docs/one"
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        resources=[{"uri": upstream_uri, "name": "Doc"}],
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")

    transport.fail_resources = True
    assert await manager.list_virtual_resources() == []
    degraded = await manager.wait_for_servers(
        ["research"],
        timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )

    transport.fail_resources = False
    resources = await manager.list_virtual_resources()
    ready = await manager.wait_for_servers(
        ["research"],
        timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )

    assert degraded["ok"] is False
    assert resources[0]["name"] == "Doc"
    assert ready == {
        "ok": True,
        "ready_servers": ["research"],
        "unavailable_servers": [],
        "unknown_servers": [],
    }


@pytest.mark.asyncio
async def test_external_runtime_resource_read_reports_missing_or_stopped() -> None:
    upstream_uri = "resource://docs/one"
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        resources=[{"uri": upstream_uri, "name": "Doc"}],
        resource_reads={upstream_uri: {"contents": [{"uri": upstream_uri, "text": "ok"}]}},
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")
    resources = await manager.list_virtual_resources()

    with pytest.raises(GatewayExternalRuntimeError) as missing:
        await manager.read_virtual_resource("external://research/missing")
    assert missing.value.reason_code == "external_resource_not_found"

    await manager.stop_server("research")
    with pytest.raises(GatewayExternalRuntimeError) as stopped:
        await manager.read_virtual_resource(resources[0]["uri"])
    assert stopped.value.reason_code == "external_resource_not_found"


@pytest.mark.asyncio
async def test_external_runtime_adapter_merges_base_and_external_resources() -> None:
    """Gateway runtime lists base resources first, then external resources."""
    upstream_uri = "resource://docs/one"
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(
        server_id="research",
        resources=[{"uri": upstream_uri, "name": "External Doc"}],
        resource_reads={upstream_uri: {"contents": [{"uri": upstream_uri, "text": "external"}]}},
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")
    base_runtime = BaseResourceRuntime()
    runtime = ExternalRuntimeGatewayRuntime(
        external_runtime_manager=manager,
        base_runtime=base_runtime,
    )
    context = GatewayRequestContext(
        request_id="resources",
        metadata={
            "_gateway_effective_policy": {
                "external_server_grants": [{"server_id": "research"}],
            }
        },
    )

    resources = await runtime.list_resources(context)
    external_uri = next(
        resource["uri"]
        for resource in resources
        if resource.get("metadata", {}).get("source") == "external_runtime"
    )
    external = await runtime.read_resource(external_uri, context)
    local = await runtime.read_resource("resource://local/doc", context)

    assert [resource["uri"] for resource in resources] == [
        "resource://local/doc",
        external_uri,
    ]
    assert external["contents"][0]["text"] == "external"
    assert local["contents"][0]["text"] == "local"
    assert base_runtime.read_requests == ["resource://local/doc"]


@pytest.mark.asyncio
async def test_external_runtime_wait_for_servers_reports_ready_and_unavailable() -> None:
    """Explicit server ids report unavailable before start and ready after start."""
    store = InMemoryExternalRegistryStore([_server()])
    transport = RecordingExternalTransport(server_id="research")
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )

    unavailable = await manager.wait_for_servers(
        ["research"],
        timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )
    await manager.start_server("research")
    ready = await manager.wait_for_servers(
        ["research"],
        timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )

    assert unavailable == {
        "ok": False,
        "ready_servers": [],
        "unavailable_servers": ["research"],
        "unknown_servers": [],
    }
    assert ready == {
        "ok": True,
        "ready_servers": ["research"],
        "unavailable_servers": [],
        "unknown_servers": [],
    }


@pytest.mark.asyncio
async def test_external_runtime_wait_for_servers_tolerates_malformed_runtime_rows() -> None:
    class MalformedRuntimeRowsManager(GatewayExternalRuntimeManager):
        async def list_runtime_servers(self) -> dict[str, Any]:
            return {
                "servers": [
                    None,
                    "not-a-row",
                    {"id": "research", "status": "healthy"},
                    {"id": None, "status": "healthy"},
                ]
            }

    manager = MalformedRuntimeRowsManager(
        external_registry_store=InMemoryExternalRegistryStore([_server()]),
        transport_factory=lambda _server: RecordingExternalTransport(server_id="research"),
    )

    ready = await manager.wait_for_servers(
        ["research"],
        timeout_seconds=0,
        poll_interval_seconds=0.001,
    )

    assert ready == {
        "ok": True,
        "ready_servers": ["research"],
        "unavailable_servers": [],
        "unknown_servers": [],
    }


@pytest.mark.asyncio
async def test_external_runtime_wait_for_servers_defaults_to_active_servers() -> None:
    """Omitted server ids wait for active transports rather than all registry rows."""
    store = InMemoryExternalRegistryStore(
        [
            _server(id="research", name="Research"),
            _server(id="docs", name="Docs"),
        ]
    )
    transport = RecordingExternalTransport(server_id="research")
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda _server: transport,
    )
    await manager.start_server("research")

    ready = await manager.wait_for_servers(
        timeout_seconds=0.01,
        poll_interval_seconds=0.001,
    )

    assert ready == {
        "ok": True,
        "ready_servers": ["research"],
        "unavailable_servers": [],
        "unknown_servers": [],
    }


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
async def test_external_runtime_stop_all_stops_active_transports_and_clears_tools() -> None:
    store = InMemoryExternalRegistryStore(
        [
            _server(),
            _server(id="docs", name="Docs", command=["fake-docs-mcp"]),
        ]
    )
    transports = {
        "research": RecordingExternalTransport(
            server_id="research",
            tools=[ExternalToolDefinition(name="search")],
        ),
        "docs": RecordingExternalTransport(
            server_id="docs",
            tools=[ExternalToolDefinition(name="lookup")],
        ),
    }
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: transports[server.id],
    )
    await manager.start_server("research")
    await manager.start_server("docs")

    payload = await manager.stop_all()
    rows = await manager.list_runtime_servers()
    tools = await manager.list_virtual_tools()

    assert payload == {
        "ok": True,
        "reason_code": "external_runtime_stopped",
        "stopped_servers": 2,
        "total_servers": 2,
        "errors": {},
    }
    assert {row["id"]: row["status"] for row in rows["servers"]} == {
        "docs": "stopped",
        "research": "stopped",
    }
    assert tools == []
    assert transports["research"].close_count == 1
    assert transports["docs"].close_count == 1


@pytest.mark.asyncio
async def test_external_runtime_stop_all_does_not_require_registry_store_for_active_transports() -> None:
    store = ToggleOutageExternalRegistryStore(
        [
            _server(),
            _server(id="docs", name="Docs", command=["fake-docs-mcp"]),
        ]
    )
    transports = {
        "research": RecordingExternalTransport(
            server_id="research",
            tools=[ExternalToolDefinition(name="search")],
        ),
        "docs": RecordingExternalTransport(
            server_id="docs",
            tools=[ExternalToolDefinition(name="lookup")],
        ),
    }
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: transports[server.id],
    )
    await manager.start_server("research")
    await manager.start_server("docs")
    store.fail_get_server = True

    payload = await manager.stop_all()
    tools = await manager.list_virtual_tools()

    assert payload == {
        "ok": True,
        "reason_code": "external_runtime_stopped",
        "stopped_servers": 2,
        "total_servers": 2,
        "errors": {},
    }
    assert tools == []
    assert transports["research"].close_count == 1
    assert transports["docs"].close_count == 1


@pytest.mark.asyncio
async def test_external_runtime_stop_all_continues_after_unexpected_stop_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingStopRuntimeManager(GatewayExternalRuntimeManager):
        async def stop_server(self, server_id: str) -> dict[str, Any]:
            if server_id == "research":
                raise RuntimeError("stop failed")
            return await super().stop_server(server_id)

    fake_logger = FakeLogger()
    monkeypatch.setattr(gateway_external_runtime, "logger", fake_logger)
    store = InMemoryExternalRegistryStore(
        [
            _server(),
            _server(id="docs", name="Docs", command=["fake-docs-mcp"]),
        ]
    )
    transports = {
        "research": RecordingExternalTransport(
            server_id="research",
            tools=[ExternalToolDefinition(name="search")],
        ),
        "docs": RecordingExternalTransport(
            server_id="docs",
            tools=[ExternalToolDefinition(name="lookup")],
        ),
    }
    manager = ExplodingStopRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: transports[server.id],
    )
    await manager.start_server("research")
    await manager.start_server("docs")

    payload = await manager.stop_all()

    assert payload["ok"] is False
    assert payload["reason_code"] == "external_runtime_stopped"
    assert payload["stopped_servers"] == 1
    assert payload["total_servers"] == 2
    assert payload["errors"]["research"]["reason_code"] == "external_server_stop_failed"
    assert payload["errors"]["research"]["error_type"] == "RuntimeError"
    assert fake_logger.opt_calls == [{"exception": True}]
    assert fake_logger.error_calls == [
        (
            "External runtime stop failed server_id={!r} error_type={!r}",
            ("research", "RuntimeError"),
        )
    ]
    assert transports["docs"].close_count == 1


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
async def test_external_runtime_has_virtual_tool_uses_active_catalog() -> None:
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

    assert await manager.has_virtual_tool("ext.research.search") is True
    assert await manager.has_virtual_tool("ext.research.missing") is False


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
async def test_external_runtime_default_installer_status_is_not_configured() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
    )

    rows = await manager.list_runtime_servers()

    assert rows["servers"][0]["installer"] == {
        "available": False,
        "reason_code": "external_server_installer_not_configured",
        "server_id": "research",
    }


@pytest.mark.asyncio
async def test_external_runtime_installer_status_is_sanitized() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
        installer=SuccessfulInstaller(),
    )

    rows = await manager.list_runtime_servers()

    installer = rows["servers"][0]["installer"]
    assert installer == {
        "available": True,
        "reason_code": "external_server_installer_available",
        "server_id": "research",
        "installer": "fake",
        "latest_version": "1.2.4",
        "message": "ready",
        "details": {
            "channel": "stable",
            "nested": [{"safe": "kept"}],
        },
    }
    assert "do-not-leak" not in json.dumps(installer, sort_keys=True)


@pytest.mark.asyncio
async def test_external_runtime_installer_status_failure_is_row_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = FakeLogger()
    monkeypatch.setattr(gateway_external_runtime, "logger", fake_logger)
    store = InMemoryExternalRegistryStore([_server()])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
        installer=ExplodingInstaller(),
    )

    rows = await manager.list_runtime_servers()

    assert rows["servers"][0]["installer"] == {
        "available": False,
        "reason_code": "external_server_installer_status_unavailable",
        "server_id": "research",
        "error_type": "RuntimeError",
    }
    assert "do-not-leak" not in json.dumps(rows, sort_keys=True)
    assert fake_logger.opt_calls == []
    assert len(fake_logger.error_calls) == 1
    message, args = fake_logger.error_calls[0]
    assert message == "External installer status failed server_id={!r} error_type={!r} traceback_frames={!r}"
    assert args[:2] == ("research", "RuntimeError")
    frame_functions = [frame["function"] for frame in args[2]]
    assert frame_functions[0] == "_installer_status"
    assert frame_functions[-1] == "get_status"
    assert "do-not-leak" not in json.dumps(fake_logger.error_calls, sort_keys=True)


@pytest.mark.asyncio
async def test_external_runtime_installer_status_timeout_is_row_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = FakeLogger()
    monkeypatch.setattr(gateway_external_runtime, "logger", fake_logger)
    store = InMemoryExternalRegistryStore([_server()])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
        installer=HangingInstaller(),
        installer_status_timeout_seconds=0.01,
    )

    rows = await asyncio.wait_for(manager.list_runtime_servers(), timeout=0.2)

    assert rows["servers"][0]["installer"] == {
        "available": False,
        "reason_code": "external_server_installer_status_timeout",
        "server_id": "research",
        "error_type": "TimeoutError",
    }
    assert fake_logger.opt_calls == [{"exception": True}]
    assert fake_logger.error_calls == [
        (
            "External installer status timed out server_id={!r}",
            ("research",),
        )
    ]


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


@pytest.mark.asyncio
async def test_external_runtime_installer_operations_reject_disabled_and_missing_servers() -> None:
    store = InMemoryExternalRegistryStore([_server(enabled=False)])
    installer = SuccessfulInstaller()
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
        installer=installer,
    )

    with pytest.raises(GatewayExternalRuntimeError) as disabled_exc:
        await manager.install_server("research")
    with pytest.raises(GatewayExternalRuntimeError) as missing_exc:
        await manager.update_server("missing")

    assert disabled_exc.value.reason_code == "external_server_disabled"
    assert missing_exc.value.reason_code == "external_server_not_found"
    assert installer.install_calls == []
    assert installer.update_calls == []


@pytest.mark.asyncio
async def test_external_runtime_installer_operations_are_sanitized() -> None:
    store = InMemoryExternalRegistryStore([_server()])
    installer = SuccessfulInstaller()
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
        installer=installer,
    )

    install = await manager.install_server("research", context={"request_id": "install"})
    update = await manager.update_server("research", context={"request_id": "update"})

    assert install == {
        "ok": True,
        "available": True,
        "reason_code": "external_server_installed",
        "server_id": "research",
        "installer": "fake",
        "version": "1.2.3",
        "message": "ready",
        "details": {
            "channel": "stable",
            "nested": [{"safe": "kept"}],
        },
    }
    assert update == {
        "ok": True,
        "available": True,
        "reason_code": "external_server_updated",
        "server_id": "research",
        "installer": "fake",
        "installed_version": "1.2.4",
        "message": "ready",
        "details": {
            "channel": "stable",
            "nested": [{"safe": "kept"}],
        },
    }
    assert installer.install_calls == [("research", {"request_id": "install"})]
    assert installer.update_calls == [("research", {"request_id": "update"})]
    assert "do-not-leak" not in json.dumps(
        {"install": install, "update": update},
        sort_keys=True,
    )


@pytest.mark.asyncio
async def test_external_runtime_installer_operation_failures_are_wrapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_logger = FakeLogger()
    monkeypatch.setattr(gateway_external_runtime, "logger", fake_logger)
    store = InMemoryExternalRegistryStore([_server()])
    manager = GatewayExternalRuntimeManager(
        external_registry_store=store,
        transport_factory=lambda server: RecordingExternalTransport(server_id=server.id),
        installer=ExplodingInstaller(),
    )

    with pytest.raises(GatewayExternalRuntimeError) as install_exc:
        await manager.install_server("research")
    with pytest.raises(GatewayExternalRuntimeError) as update_exc:
        await manager.update_server("research")

    assert install_exc.value.reason_code == "external_server_install_failed"
    assert install_exc.value.server_id == "research"
    assert str(install_exc.value) == "External server install failed"
    assert install_exc.value.__cause__ is not None
    assert str(install_exc.value.__cause__) == "RuntimeError"
    assert "do-not-leak" not in str(install_exc.value.__cause__)
    assert update_exc.value.reason_code == "external_server_update_failed"
    assert update_exc.value.server_id == "research"
    assert str(update_exc.value) == "External server update failed"
    assert update_exc.value.__cause__ is not None
    assert str(update_exc.value.__cause__) == "RuntimeError"
    assert "do-not-leak" not in str(update_exc.value.__cause__)
    public_payloads = {
        "install": install_exc.value.to_payload(),
        "update": update_exc.value.to_payload(),
    }
    assert "do-not-leak" not in json.dumps(public_payloads, sort_keys=True)
    assert fake_logger.opt_calls == []
    assert len(fake_logger.error_calls) == 2
    for expected_operation, expected_function, error_call in (
        ("install", "install_server", fake_logger.error_calls[0]),
        ("update", "update_server", fake_logger.error_calls[1]),
    ):
        message, args = error_call
        assert message == (
            "External installer operation failed operation={!r} server_id={!r} "
            "error_type={!r} traceback_frames={!r}"
        )
        assert args[:3] == (expected_operation, "research", "RuntimeError")
        assert [frame["function"] for frame in args[3]] == [
            "_installer_operation_payload",
            expected_function,
        ]
    assert "do-not-leak" not in json.dumps(fake_logger.error_calls, sort_keys=True)


def test_runtime_signature_detects_header_changes() -> None:
    """A headers-only change must alter the reconcile signature so patched tokens restart transports."""
    base = ExternalServerDefinition(
        id="linear",
        name="Linear",
        transport="streamable_http",
        url="https://mcp.linear.example.test/mcp",
        headers={"Authorization": "Bearer old"},
    )
    rotated = base.model_copy(update={"headers": {"Authorization": "Bearer new"}})
    assert GatewayExternalRuntimeManager._definition_changed(base, rotated)
    assert not GatewayExternalRuntimeManager._definition_changed(base, base.model_copy(deep=True))
