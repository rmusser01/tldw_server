from __future__ import annotations

from typing import Any

import pytest
from mcp_unified.federation.models import (
    BrokeredExternalCredential,
    ExternalToolCallResult,
    ExternalToolDefinition,
    FederatedToolResult,
    VirtualExternalTool,
)
from mcp_unified.federation.transports import FakeExternalTransport
from mcp_unified.gateway.external_runtime import GatewayExternalRuntimeManager
from mcp_unified.gateway.external_runtime_adapter import ExternalRuntimeGatewayRuntime
from mcp_unified.gateway.jsonrpc import handle_jsonrpc
from mcp_unified.gateway.profile_runtime import (
    EFFECTIVE_POLICY_METADATA_KEY,
    ProfileAwareGatewayRuntime,
)
from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy
from mcp_unified.profiles.store import InMemoryProfileStore
from mcp_unified.storage.models import ExternalServerDefinition


class InMemoryExternalRegistryStore:
    """Copy-isolated external registry store for adapter tests."""

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


class BaseGatewayRuntime:
    """Small local gateway runtime used to prove delegation behavior."""

    name = "base-gateway"
    version = "1.2.3"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], GatewayRequestContext]] = []

    async def list_tools(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return one local tool descriptor."""

        del context
        return [
            {
                "name": "local.echo",
                "description": "Echo a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
                "metadata": {"source": "base"},
            }
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Record a local tool call and return a simple text response."""

        self.calls.append((name, dict(arguments), context))
        return {"content": [{"type": "text", "text": arguments["query"]}]}

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no local resources."""

        del context
        return []

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Reject local resource reads in this fake runtime."""

        del context
        raise ValueError(f"Unknown local resource: {uri}")

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no local prompts."""

        del context
        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        """Reject local prompt reads in this fake runtime."""

        del arguments, context
        raise ValueError(f"Unknown local prompt: {name}")

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        """Return no local modules."""

        del context
        return []

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        """Return empty local health."""

        del context
        return {"modules": []}


class RecordingCredentialBroker:
    """Credential broker fake that records resolution requests."""

    def __init__(self, result: BrokeredExternalCredential) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def resolve_external_credential(
        self,
        **kwargs: Any,
    ) -> BrokeredExternalCredential:
        """Record one credential lookup and return a copied credential."""

        self.calls.append(dict(kwargs))
        return self.result.copy()


class NullableVirtualToolManager:
    """Minimal manager fake that returns nullable virtual tool fields."""

    async def list_virtual_tools(self) -> list[VirtualExternalTool]:
        """Return one virtual tool with nullable schema and metadata values."""

        return [
            VirtualExternalTool(
                virtual_name="ext.null.search",
                server_id="null-server",
                upstream_tool_name="search",
                input_schema=None,  # type: ignore[arg-type]
                metadata=None,  # type: ignore[arg-type]
            )
        ]


class DirectRoutingVirtualToolManager:
    """Manager fake that fails if tool calls route through full discovery."""

    def __init__(self) -> None:
        self.has_calls: list[str] = []
        self.execute_calls: list[tuple[str, dict[str, Any]]] = []

    async def list_virtual_tools(self) -> list[VirtualExternalTool]:
        """Reject call-time discovery scans."""

        raise AssertionError("call_tool should not list virtual tools for routing")

    async def has_virtual_tool(self, name: str) -> bool:
        """Return whether the fake external catalog owns the requested name."""

        self.has_calls.append(name)
        return name == "ext.fast.search"

    async def execute_virtual_tool(
        self,
        virtual_tool_name: str,
        arguments: dict[str, Any],
        **_kwargs: Any,
    ) -> FederatedToolResult:
        """Record direct execution and return a gateway-compatible result."""

        self.execute_calls.append((virtual_tool_name, dict(arguments)))
        return FederatedToolResult(
            content={"matches": ["paper-1"]},
            server_id="fast",
            upstream_tool_name="search",
            virtual_tool_name=virtual_tool_name,
        )


def _server(
    server_id: str = "research",
    *,
    credential_slots: list[str] | None = None,
) -> ExternalServerDefinition:
    """Build a valid non-spawning stdio server definition for tests."""

    return ExternalServerDefinition(
        id=server_id,
        name=server_id.title(),
        transport="stdio",
        command=["python", "-m", "fake_mcp_server"],
        enabled=True,
        credential_slots=credential_slots or [],
    )


def _search_tool() -> ExternalToolDefinition:
    """Return one read-only external search tool definition."""

    return ExternalToolDefinition(
        name="search",
        description="Search papers.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
        metadata={
            "capability": "research.search",
            "annotations": {"readOnlyHint": True},
        },
    )


async def _started_runtime(
    *,
    base_runtime: BaseGatewayRuntime | None = None,
    credential_slots: list[str] | None = None,
    credential_broker: RecordingCredentialBroker | None = None,
) -> tuple[ExternalRuntimeGatewayRuntime, FakeExternalTransport]:
    """Create an adapter around a started external runtime manager."""

    transport = FakeExternalTransport(
        server_id="research",
        tools=[_search_tool()],
        results={
            "search": ExternalToolCallResult(
                content={"matches": ["paper-1"]},
                metadata={"transport": "fake"},
            )
        },
    )
    manager = GatewayExternalRuntimeManager(
        external_registry_store=InMemoryExternalRegistryStore(
            [_server(credential_slots=credential_slots)]
        ),
        transport_factory=lambda _server: transport,
        credential_broker=credential_broker,
    )
    await manager.start_server("research")
    return (
        ExternalRuntimeGatewayRuntime(
            base_runtime=base_runtime,
            external_runtime_manager=manager,
        ),
        transport,
    )


async def test_external_runtime_gateway_lists_base_and_external_tools() -> None:
    """External virtual tools should appear beside base gateway tools."""

    runtime, _transport = await _started_runtime(base_runtime=BaseGatewayRuntime())

    tools = await runtime.list_tools(GatewayRequestContext(request_id="list"))

    assert [tool["name"] for tool in tools] == ["local.echo", "ext.research.search"]
    external = tools[1]
    assert external["description"] == "Search papers."
    assert external["inputSchema"]["properties"]["query"]["type"] == "string"
    assert external["metadata"]["external_server_id"] == "research"
    assert external["metadata"]["upstream_tool_name"] == "search"
    assert external["metadata"]["source"] == "external_runtime"
    assert external["metadata"]["is_write"] is False


async def test_external_runtime_gateway_defaults_nullable_input_schema() -> None:
    """Nullable upstream schemas should still expose a JSON object descriptor."""

    runtime = ExternalRuntimeGatewayRuntime(
        external_runtime_manager=NullableVirtualToolManager(),  # type: ignore[arg-type]
    )

    tools = await runtime.list_tools(GatewayRequestContext(request_id="list-nullable"))

    assert tools[0]["name"] == "ext.null.search"
    assert tools[0]["inputSchema"] == {}
    assert tools[0]["metadata"]["external_server_id"] == "null-server"


async def test_external_runtime_gateway_dispatches_external_and_local_tools() -> None:
    """External names call the manager while local names still use the base runtime."""

    base_runtime = BaseGatewayRuntime()
    runtime, transport = await _started_runtime(base_runtime=base_runtime)
    context = GatewayRequestContext(
        request_id="call",
        user_id="user-1",
        metadata={
            EFFECTIVE_POLICY_METADATA_KEY: {
                "profile_id": "researcher",
                "external_server_grants": [{"server_id": "research"}],
            }
        },
    )

    external = await runtime.call_tool("ext.research.search", {"query": "mcp"}, context)
    local = await runtime.call_tool("local.echo", {"query": "hello"}, context)

    assert external == {
        "content": {"matches": ["paper-1"]},
        "isError": False,
        "metadata": {
            "server_id": "research",
            "transport": "fake",
            "upstream_tool_name": "search",
            "virtual_tool_name": "ext.research.search",
        },
    }
    assert transport.calls == [("search", {"query": "mcp"})]
    assert local["content"][0]["text"] == "hello"
    assert base_runtime.calls[0][0] == "local.echo"


async def test_external_runtime_gateway_routes_calls_without_full_tool_listing() -> None:
    """Call routing should use a direct manager membership check."""

    manager = DirectRoutingVirtualToolManager()
    base_runtime = BaseGatewayRuntime()
    runtime = ExternalRuntimeGatewayRuntime(
        external_runtime_manager=manager,  # type: ignore[arg-type]
        base_runtime=base_runtime,
    )
    context = GatewayRequestContext(request_id="direct-route")

    external = await runtime.call_tool("ext.fast.search", {"query": "mcp"}, context)
    local = await runtime.call_tool("local.echo", {"query": "hello"}, context)

    assert external["content"] == {"matches": ["paper-1"]}
    assert local["content"][0]["text"] == "hello"
    assert manager.has_calls == ["ext.fast.search", "local.echo"]
    assert manager.execute_calls == [("ext.fast.search", {"query": "mcp"})]
    assert base_runtime.calls[0][0] == "local.echo"


async def test_external_runtime_gateway_handles_missing_context_metadata() -> None:
    """Missing metadata should not crash effective-policy extraction."""

    runtime, _transport = await _started_runtime()
    context = GatewayRequestContext(
        request_id="call-null-metadata",
        metadata=None,  # type: ignore[arg-type]
    )

    with pytest.raises(GatewayPolicyDenied) as exc_info:
        await runtime.call_tool("ext.research.search", {"query": "mcp"}, context)

    assert exc_info.value.reason_code == "external_server_not_granted"


async def test_external_runtime_gateway_unknown_tool_without_base_fails() -> None:
    """An adapter without a base runtime should fail closed for unknown names."""

    runtime, _transport = await _started_runtime()

    with pytest.raises(ValueError, match="Unknown gateway tool"):
        await runtime.call_tool(
            "missing.tool",
            {},
            GatewayRequestContext(request_id="missing"),
        )


async def test_profile_runtime_passes_external_grants_to_adapter() -> None:
    """Resolved profile grants should reach the external runtime manager."""

    broker = RecordingCredentialBroker(
        BrokeredExternalCredential(
            headers={"Authorization": "Bearer test"},
            metadata={"credential_mode": "brokered_ephemeral"},
        )
    )
    adapter, transport = await _started_runtime(
        credential_slots=["api_key"],
        credential_broker=broker,
    )
    profile_store = InMemoryProfileStore()
    await profile_store.upsert_profile(
        MCPProfile(
            id="researcher",
            name="Researcher",
            policy_document=ProfilePolicy(allowed_tools=["ext.research.search"]),
            external_server_grants=[{"server_id": "research"}],
            credential_grants=[
                {"server_id": "research", "credential_slots": ["api_key"]}
            ],
        )
    )
    runtime = ProfileAwareGatewayRuntime(
        adapter,
        profile_store=profile_store,
        default_profile_id="researcher",
    )
    context = GatewayRequestContext(request_id="profile-call", user_id="user-1")

    result = await runtime.call_tool("ext.research.search", {"query": "mcp"}, context)

    assert result["content"] == {"matches": ["paper-1"]}
    assert transport.runtime_auth_seen is not None
    assert broker.calls[0]["effective_policy"]["profile_id"] == "researcher"
    assert broker.calls[0]["effective_policy"]["credential_grants"] == [
        {"server_id": "research", "credential_slots": ["api_key"]}
    ]
    assert EFFECTIVE_POLICY_METADATA_KEY not in context.metadata


async def test_profile_runtime_enriches_context_with_missing_metadata() -> None:
    """Profile policy metadata enrichment should tolerate a null metadata mapping."""

    base_runtime = BaseGatewayRuntime()
    profile_store = InMemoryProfileStore()
    await profile_store.upsert_profile(
        MCPProfile(
            id="researcher",
            name="Researcher",
            policy_document=ProfilePolicy(allowed_tools=["local.echo"]),
        )
    )
    runtime = ProfileAwareGatewayRuntime(
        base_runtime,
        profile_store=profile_store,
        default_profile_id="researcher",
    )
    context = GatewayRequestContext(
        request_id="profile-null-metadata",
        metadata=None,  # type: ignore[arg-type]
    )

    result = await runtime.call_tool("local.echo", {"query": "hello"}, context)

    assert result["content"][0]["text"] == "hello"
    delegated_context = base_runtime.calls[0][2]
    assert delegated_context.metadata[EFFECTIVE_POLICY_METADATA_KEY]["profile_id"] == "researcher"
    assert context.metadata is None


async def test_jsonrpc_maps_external_policy_denial_to_policy_error() -> None:
    """External grant denial should surface as JSON-RPC policy denial."""

    adapter, _transport = await _started_runtime()
    profile_store = InMemoryProfileStore()
    await profile_store.upsert_profile(
        MCPProfile(
            id="researcher",
            name="Researcher",
            policy_document=ProfilePolicy(allowed_tools=["ext.research.search"]),
        )
    )
    runtime = ProfileAwareGatewayRuntime(
        adapter,
        profile_store=profile_store,
        default_profile_id="researcher",
    )

    response = await handle_jsonrpc(
        runtime,
        {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": "ext.research.search",
                "arguments": {"query": "mcp"},
            },
            "id": "denied",
        },
        path="/mcp",
    )

    assert not isinstance(response, list)
    assert response.error is not None
    assert response.error.code == -32001
    assert response.error.data["reason_code"] == "external_server_not_granted"
