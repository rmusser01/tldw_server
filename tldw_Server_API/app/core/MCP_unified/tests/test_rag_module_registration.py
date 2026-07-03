from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from tldw_Server_API.app.core.MCP_unified.module_surface import MODULE_RISK_TIERS
from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, MCPRequest, RequestContext


def test_rag_module_is_in_default_module_config() -> None:
    config = yaml.safe_load(Path("tldw_Server_API/Config_Files/mcp_modules.yaml").read_text())

    rag = next(module for module in config["modules"] if module["id"] == "rag")

    assert rag["enabled"] is True  # nosec B101
    assert rag["class"].endswith("rag_module:RagModule")  # nosec B101
    assert rag["department"] == "knowledge"  # nosec B101


def test_rag_tool_category_config_separates_generation() -> None:
    mapping = yaml.safe_load(Path("tldw_Server_API/Config_Files/mcp_tool_categories.yaml").read_text())

    assert mapping["rag.search"] == "search"  # nosec B101
    assert mapping["rag.answer"] == "rag_generation"  # nosec B101


def test_rag_search_and_answer_have_concrete_mcp_policies() -> None:
    policies = yaml.safe_load(Path("tldw_Server_API/Config_Files/resource_governor_policies.yaml").read_text())

    assert "mcp.search" in policies["policies"]  # nosec B101
    assert "mcp.rag_generation" in policies["policies"]  # nosec B101
    assert policies["policies"]["mcp.rag_generation"]["scopes"] == ["user", "api_key"]  # nosec B101


def test_module_surface_classifies_rag_as_read_only() -> None:
    assert MODULE_RISK_TIERS["rag"][0] == "read_only"  # nosec B101


class _CategoryProbeRateLimiter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def check_rate_limit(self, key: str, *, category: str = "default") -> None:
        self.calls.append((key, category))


class _AllowAllRBAC:
    async def check_permission(self, *args: Any, **kwargs: Any) -> bool:
        del args, kwargs
        return True


class _NoopMetrics:
    def __getattr__(self, _name: str):
        return lambda *args, **kwargs: None


class _NoopTelemetry:
    def trace_context(self, _operation_name: str, _attributes: dict[str, Any] | None = None) -> Any:
        class _Span:
            def set_attribute(self, _key: str, _value: Any) -> None:
                return None

        class _Context:
            def __enter__(self) -> _Span:
                return _Span()

            def __exit__(self, *_exc_info: Any) -> None:
                return None

        return _Context()


class _RagCategoryProbeModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="rag.search",
                description="probe search",
                parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="rag.answer",
                description="probe answer",
                parameters={"properties": {"query": {"type": "string"}}, "required": ["query"]},
                metadata={"category": "rag_generation", "readOnlyHint": True},
            ),
        ]

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        del tool_name
        if "query" not in arguments:
            raise ValueError("query required")

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: RequestContext | None = None,
    ) -> dict[str, str]:
        del arguments, context
        return {"tool": tool_name}


def _protocol_with_probe_limiter(registry: ModuleRegistry, limiter: _CategoryProbeRateLimiter) -> MCPProtocol:
    return MCPProtocol(
        dependencies=SimpleNamespace(
            module_registry=registry,
            rbac_policy=_AllowAllRBAC(),
            rate_limiter=limiter,
            metrics_collector=_NoopMetrics(),
            telemetry_provider=_NoopTelemetry(),
            redis_client_factory=lambda **_kwargs: None,
            tool_catalog_provider=object(),
            effective_policy_resolver=object(),
            approval_evaluator=object(),
            path_scope_enforcer=object(),
            external_access_evaluator=object(),
        )
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_rag_tool_categories_reach_tools_call_rate_limiter() -> None:
    registry = ModuleRegistry()
    await registry.register_module(
        "rag_category_probe",
        _RagCategoryProbeModule,
        ModuleConfig(name="rag_category_probe"),
    )
    limiter = _CategoryProbeRateLimiter()
    proto = _protocol_with_probe_limiter(registry, limiter)
    ctx = RequestContext(request_id="rag-categories", user_id="1", client_id="unit")

    search = await proto.process_request(
        MCPRequest(method="tools/call", params={"name": "rag.search", "arguments": {"query": "q"}}, id=1),
        ctx,
    )
    answer = await proto.process_request(
        MCPRequest(method="tools/call", params={"name": "rag.answer", "arguments": {"query": "q"}}, id=2),
        ctx,
    )

    assert search.error is None  # nosec B101
    assert answer.error is None  # nosec B101
    tool_categories = [category for key, category in limiter.calls if ":tool:" in key]
    assert tool_categories == ["search", "rag_generation"]  # nosec B101
