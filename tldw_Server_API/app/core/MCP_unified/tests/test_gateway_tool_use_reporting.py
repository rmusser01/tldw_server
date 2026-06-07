from __future__ import annotations

from typing import Any

import pytest

from mcp_unified.gateway.config import (
    GatewayProfileBootstrapConfig,
    bootstrap_profile_gateway_from_config,
)
from mcp_unified.gateway.profile_runtime import ProfileAwareGatewayRuntime
from mcp_unified.gateway.runtime import GatewayPolicyDenied, GatewayRequestContext
from mcp_unified.gateway.tool_use_reporting import ToolUseReportingGatewayRuntime
from mcp_unified.profiles.models import MCPProfile, ProfilePolicy
from mcp_unified.profiles.store import InMemoryProfileStore
from mcp_unified.tool_use_reporting.models import ToolUseEvent


class _MemoryToolUseRecorder:
    """Small recorder double that exposes captured events for assertions."""

    def __init__(self) -> None:
        self.events: list[ToolUseEvent] = []

    async def record_tool_use(self, event: ToolUseEvent) -> None:
        self.events.append(event)


class _FakeGatewayRuntime:
    name = "unit-gateway"
    version = "0.0-test"

    def __init__(self) -> None:
        self.call_requests: list[tuple[str, dict[str, Any], GatewayRequestContext]] = []

    async def list_tools(
        self,
        context: GatewayRequestContext,
    ) -> list[dict[str, Any]]:
        del context
        return [
            {
                "name": "git.status",
                "description": "Show repository status.",
                "inputSchema": {"type": "object", "properties": {}},
                "metadata": {"category": "git"},
            },
            {
                "name": "echo.search",
                "description": "Echo a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                "metadata": {"category": "test", "capability": "code_search"},
            },
        ]

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        self.call_requests.append((name, arguments, context))
        return {"content": [{"type": "text", "text": name}]}

    async def list_resources(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        del context
        return []

    async def read_resource(
        self,
        uri: str,
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        del context
        return {"contents": [{"uri": uri}]}

    async def list_prompts(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        del context
        return []

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        del arguments, context
        return {"messages": [{"role": "user", "content": name}]}

    async def list_modules(self, context: GatewayRequestContext) -> list[dict[str, Any]]:
        del context
        return []

    async def get_modules_health(self, context: GatewayRequestContext) -> dict[str, Any]:
        del context
        return {}


class _DenyingGatewayRuntime(_FakeGatewayRuntime):
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        del arguments, context
        raise GatewayPolicyDenied(
            f"Denied {name}",
            reason_code="profile_tool_denied",
            provenance={"tool_name": name},
        )


class _ExplodingGatewayRuntime(_FakeGatewayRuntime):
    class BackendFailure(RuntimeError):
        pass

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: GatewayRequestContext,
    ) -> dict[str, Any]:
        del name, arguments, context
        raise self.BackendFailure("raw /Users/me/secret path")


def _profile_with_deferred_tools() -> MCPProfile:
    return MCPProfile(
        id="researcher",
        name="Researcher",
        policy_document=ProfilePolicy(capabilities=["code_search"]),
        metadata={
            "tooling": {
                "recommended_tools": [],
                "progressive_disclosure": {
                    "direct_categories": [],
                    "deferred_categories": ["test"],
                    "max_direct_tools": 24,
                },
            }
        },
    )


@pytest.mark.asyncio
async def test_gateway_wrapper_records_direct_call_with_profile_and_model() -> None:
    recorder = _MemoryToolUseRecorder()
    backend = _FakeGatewayRuntime()
    runtime = ToolUseReportingGatewayRuntime(backend, recorder=recorder)
    context = GatewayRequestContext(
        request_id="req-1",
        metadata={"profile_id": "devops", "model_id": "gpt-4.1"},
    )

    await runtime.call_tool("git.status", {}, context)

    event = recorder.events[-1]
    assert event.runtime_surface == "gateway"
    assert event.requested_tool_name == "git.status"
    assert event.effective_tool_name == "git.status"
    assert event.profile_id == "devops"
    assert event.model_id == "gpt-4.1"
    assert event.status == "success"
    assert event.execution_origin == "executed"
    assert backend.call_requests[-1][2].metadata["mcp_tool_use_observed"] is True
    assert "mcp_tool_use_observed" not in context.metadata


@pytest.mark.asyncio
async def test_gateway_wrapper_skips_when_context_already_observed() -> None:
    recorder = _MemoryToolUseRecorder()
    backend = _FakeGatewayRuntime()
    runtime = ToolUseReportingGatewayRuntime(backend, recorder=recorder)

    await runtime.call_tool(
        "git.status",
        {},
        GatewayRequestContext(
            request_id="req-1",
            metadata={"mcp_tool_use_observed": True},
        ),
    )

    assert recorder.events == []
    assert backend.call_requests[-1][0] == "git.status"


@pytest.mark.asyncio
async def test_gateway_wrapper_delegates_non_tool_call_methods_without_recording() -> None:
    recorder = _MemoryToolUseRecorder()
    runtime = ToolUseReportingGatewayRuntime(_FakeGatewayRuntime(), recorder=recorder)
    context = GatewayRequestContext(request_id="req-1")

    tools = await runtime.list_tools(context)
    resources = await runtime.list_resources(context)
    resource = await runtime.read_resource("resource://unit/doc", context)
    prompts = await runtime.list_prompts(context)
    prompt = await runtime.get_prompt("review.prompt", {}, context)
    modules = await runtime.list_modules(context)
    health = await runtime.get_modules_health(context)

    assert tools[0]["name"] == "git.status"
    assert resources == []
    assert resource["contents"][0]["uri"] == "resource://unit/doc"
    assert prompts == []
    assert prompt["messages"][0]["content"] == "review.prompt"
    assert modules == []
    assert health == {}
    assert recorder.events == []


@pytest.mark.asyncio
async def test_gateway_wrapper_records_policy_denial() -> None:
    recorder = _MemoryToolUseRecorder()
    runtime = ToolUseReportingGatewayRuntime(
        _DenyingGatewayRuntime(),
        recorder=recorder,
    )

    with pytest.raises(GatewayPolicyDenied):
        await runtime.call_tool(
            "fs.write",
            {},
            GatewayRequestContext(request_id="req-1"),
        )

    event = recorder.events[-1]
    assert event.status == "denied"
    assert event.reason_code == "profile_tool_denied"
    assert event.execution_origin == "denied"
    assert event.requested_tool_name == "fs.write"


@pytest.mark.asyncio
async def test_gateway_wrapper_records_sanitized_backend_failure() -> None:
    recorder = _MemoryToolUseRecorder()
    runtime = ToolUseReportingGatewayRuntime(
        _ExplodingGatewayRuntime(),
        recorder=recorder,
    )

    with pytest.raises(_ExplodingGatewayRuntime.BackendFailure):
        await runtime.call_tool(
            "fs.read",
            {"path": "/Users/me/secret"},
            GatewayRequestContext(request_id="req-1"),
        )

    event = recorder.events[-1]
    assert event.status == "error"
    assert event.reason_code == "BackendFailure"
    assert event.requested_tool_name == "fs.read"
    assert event.capture_ref is None


@pytest.mark.asyncio
async def test_gateway_bridge_call_records_effective_tool_name_when_tool_id_differs() -> None:
    recorder = _MemoryToolUseRecorder()
    backend = _FakeGatewayRuntime()
    profile_runtime = ProfileAwareGatewayRuntime(
        backend,
        profile_store=InMemoryProfileStore([_profile_with_deferred_tools()]),
        default_profile_id="researcher",
    )
    wrapped = ToolUseReportingGatewayRuntime(profile_runtime, recorder=recorder)
    context = GatewayRequestContext(request_id="req-1")

    await wrapped.call_tool(
        "tool_call",
        {
            "tool_id": "echo.search",
            "arguments": {"query": "bridge", "path": "/Users/me/secret"},
        },
        context,
    )

    event = recorder.events[-1]
    assert event.requested_tool_name == "tool_call"
    assert event.effective_tool_name == "echo.search"
    assert event.source_kind == "bridge"
    assert event.capture_ref == "echo.search"
    assert event.status == "success"
    assert backend.call_requests[-1][0] == "echo.search"
    assert "mcp_tool_use_effective_tool_name" not in context.metadata


def test_gateway_config_parses_tool_use_reporting_defaults() -> None:
    config = GatewayProfileBootstrapConfig()

    assert config.tool_use_reporting.enabled is False
    assert config.tool_use_reporting.store.kind == "memory"


def test_gateway_config_allows_disabled_sqlite_reporting_without_path() -> None:
    config = GatewayProfileBootstrapConfig(
        tool_use_reporting={
            "enabled": False,
            "store": {"kind": "sqlite"},
        }
    )

    assert config.tool_use_reporting.store.kind == "sqlite"
    assert config.tool_use_reporting.store.sqlite_path is None


def test_gateway_config_rejects_enabled_sqlite_reporting_without_path() -> None:
    with pytest.raises(ValueError, match="sqlite_path is required"):
        GatewayProfileBootstrapConfig(
            tool_use_reporting={
                "enabled": True,
                "store": {"kind": "sqlite"},
            }
        )


@pytest.mark.asyncio
async def test_bootstrap_wraps_runtime_when_tool_use_reporting_enabled(tmp_path) -> None:
    config = GatewayProfileBootstrapConfig(
        tool_use_reporting={
            "enabled": True,
            "store": {
                "kind": "sqlite",
                "sqlite_path": str(tmp_path / "events.sqlite3"),
            },
        }
    )

    bootstrap = await bootstrap_profile_gateway_from_config(_FakeGatewayRuntime(), config)

    assert isinstance(bootstrap.runtime, ToolUseReportingGatewayRuntime)
