import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _DynamicToolModule(BaseModule):
    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.tool_names = ["dyn.old"]

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": name, "description": "", "inputSchema": {"type": "object"}}
            for name in self.tool_names
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        return None


class _LifecycleGateModule(BaseModule):
    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)
        self.initialize_enter = asyncio.Event()
        self.initialize_release = asyncio.Event()
        self.shutdown_enter = asyncio.Event()
        self.shutdown_release = asyncio.Event()

    async def on_initialize(self) -> None:
        self.initialize_enter.set()
        await self.initialize_release.wait()

    async def on_shutdown(self) -> None:
        self.shutdown_enter.set()
        await self.shutdown_release.wait()

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return []

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        return None


@pytest.mark.asyncio
async def test_registry_drops_stale_tool_mapping_after_cache_invalidation():
    registry = ModuleRegistry()
    await registry.register_module("dyn", _DynamicToolModule, ModuleConfig(name="dyn"))
    module = await registry.get_module("dyn")

    _ensure(module is not None, "dynamic module failed to register")
    _ensure(
        await registry.find_module_for_tool("dyn.old") is module,
        "expected old dynamic tool to resolve before invalidation",
    )

    module.tool_names = ["dyn.new"]
    module.invalidate_capability_caches()

    _ensure(
        await registry.find_module_for_tool("dyn.old") is None,
        "stale dynamic tool mapping should be dropped after cache invalidation",
    )
    _ensure(
        await registry.find_module_for_tool("dyn.new") is module,
        "new dynamic tool mapping should resolve after cache invalidation",
    )


@pytest.mark.asyncio
async def test_initialize_and_shutdown_do_not_overlap():
    module = _LifecycleGateModule(ModuleConfig(name="life"))

    initialize_task = asyncio.create_task(module.initialize())
    await asyncio.wait_for(module.initialize_enter.wait(), timeout=2.0)

    shutdown_task = asyncio.create_task(module.shutdown())
    await asyncio.sleep(0.05)
    _ensure(
        not module.shutdown_enter.is_set(),
        "shutdown should wait until initialization finishes",
    )

    module.initialize_release.set()
    await asyncio.wait_for(initialize_task, timeout=2.0)
    await asyncio.wait_for(module.shutdown_enter.wait(), timeout=2.0)

    module.shutdown_release.set()
    await asyncio.wait_for(shutdown_task, timeout=2.0)
    _ensure(not module._initialized, "module should be shut down at the end")


def test_default_mcp_modules_config_declares_codegraph_disabled():
    config_path = Path("tldw_Server_API/Config_Files/mcp_modules.yaml")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    entry = next((module for module in data["modules"] if module.get("id") == "codegraph"), None)

    assert entry is not None  # nosec B101
    assert entry["enabled"] is False  # nosec B101
    assert entry["class"] == (  # nosec B101
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module:CodeGraphModule"
    )


@pytest.mark.asyncio
async def test_server_skips_disabled_codegraph_module_from_config(monkeypatch, tmp_path: Path) -> None:
    from tldw_Server_API.app.core.MCP_unified.server import MCPServer

    config_path = tmp_path / "mcp_modules.yaml"
    config_path.write_text(
        """
modules:
  - id: codegraph
    class: tldw_Server_API.app.core.MCP_unified.modules.implementations.codegraph_module:CodeGraphModule
    enabled: false
    name: CodeGraph
    version: "0.1.0"
    department: code
    settings: {}
""".strip(),
        encoding="utf-8",
    )

    server = MCPServer()
    registered_module_ids: list[str] = []

    async def _capture_registration(module_id, module_type, config):  # noqa: ANN001, ARG001
        registered_module_ids.append(str(module_id))

    monkeypatch.setattr(server.module_registry, "register_module", _capture_registration)
    monkeypatch.setenv("MCP_MODULES_CONFIG", str(config_path))
    monkeypatch.setenv("MCP_MODULES", "")
    monkeypatch.setenv("MCP_ENABLE_MEDIA_MODULE", "0")
    monkeypatch.setenv("MCP_ENABLE_FILESYSTEM_MODULE", "0")

    await server._register_default_modules()

    assert "codegraph" not in registered_module_ids  # nosec B101
