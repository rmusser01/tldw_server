from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry, register_module, reset_module_registry
from tldw_Server_API.app.services.mcp_hub_tool_registry import McpHubToolRegistryService


class _RegistryProbeModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="probe.explicit",
                description="Explicitly annotated read tool.",
                parameters={"properties": {}, "required": []},
                metadata={
                    "category": "search",
                    "risk_class": "low",
                    "capabilities": ["filesystem.read"],
                    "uses_filesystem": True,
                    "path_boundable": True,
                    "path_argument_hints": ["path"],
                    "readOnlyHint": True,
                },
            ),
            create_tool_definition(
                name="probe.execute",
                description="Execution tool derived from category.",
                parameters={"properties": {}, "required": []},
                metadata={"category": "execution"},
            ),
            create_tool_definition(
                name="probe.unknown",
                description="Tool with no explicit metadata.",
                parameters={"properties": {}, "required": []},
            ),
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        return {"tool": tool_name, "arguments": arguments}


class _FailingInitializeModule(_RegistryProbeModule):
    async def on_initialize(self) -> None:
        raise RuntimeError("module init failed at /private/mcp-init.json")


class _FailingHealthModule(_RegistryProbeModule):
    async def check_health(self) -> dict[str, bool]:
        raise RuntimeError("module health failed at /private/mcp-health.json")


@pytest.fixture(autouse=True)
async def _reset_registry() -> None:
    await reset_module_registry()
    yield
    await reset_module_registry()


@pytest.mark.asyncio
async def test_tool_registry_normalizes_explicit_and_fallback_metadata() -> None:
    await register_module("probe", _RegistryProbeModule, ModuleConfig(name="probe", description="Probe module"))

    service = McpHubToolRegistryService(module_registry=get_module_registry())
    entries = await service.list_entries()
    by_name = {entry["tool_name"]: entry for entry in entries}

    explicit = by_name["probe.explicit"]
    assert explicit["module"] == "probe"
    assert explicit["category"] == "search"
    assert explicit["risk_class"] == "low"
    assert explicit["capabilities"] == ["filesystem.read"]
    assert explicit["uses_filesystem"] is True
    assert explicit["path_boundable"] is True
    assert explicit["path_argument_hints"] == ["path"]
    assert explicit["metadata_source"] == "explicit"
    assert explicit["metadata_warnings"] == []

    unknown = by_name["probe.unknown"]
    assert unknown["module"] == "probe"
    assert unknown["risk_class"] == "unclassified"
    assert unknown["path_argument_hints"] == []
    assert unknown["metadata_source"] in {"heuristic", "fallback"}
    assert unknown["metadata_warnings"]


@pytest.mark.asyncio
async def test_tool_registry_derives_execution_risk_and_groups_modules() -> None:
    await register_module("probe", _RegistryProbeModule, ModuleConfig(name="probe", description="Probe module"))

    service = McpHubToolRegistryService(module_registry=get_module_registry())
    groups = await service.list_modules()
    entries = await service.list_entries()
    by_name = {entry["tool_name"]: entry for entry in entries}

    execute = by_name["probe.execute"]
    assert execute["module"] == "probe"
    assert execute["category"] == "execution"
    assert execute["risk_class"] == "high"
    assert "process.execute" in execute["capabilities"]

    assert len(groups) == 1
    assert groups[0]["module"] == "probe"
    assert groups[0]["tool_count"] == 3
    assert groups[0]["risk_summary"]["high"] == 1


@pytest.mark.asyncio
async def test_base_module_initialization_failure_sets_safe_health_message() -> None:
    module = _FailingInitializeModule(ModuleConfig(name="probe", description="Probe module"))

    with pytest.raises(RuntimeError):
        await module.initialize()

    assert module._health.status.value == "unhealthy"
    assert module._health.message == "Initialization failed"
    assert "module init failed" not in module._health.message
    assert "/private/mcp-init.json" not in module._health.message


@pytest.mark.asyncio
async def test_base_module_health_check_sanitizes_backend_failures() -> None:
    module = _FailingHealthModule(ModuleConfig(name="probe", description="Probe module"))

    health = await module.health_check()

    assert health.status.value == "unhealthy"
    assert health.message == "Health check error"
    assert "module health failed" not in health.message
    assert "/private/mcp-health.json" not in health.message
