from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig, create_tool_definition
from tldw_Server_API.app.core.MCP_unified.modules.implementations.mcp_discovery_module import MCPDiscoveryModule
from tldw_Server_API.app.core.MCP_unified.modules.registry import register_module, reset_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext


class DummyModule(BaseModule):
    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="dummy.echo",
                description="Echo tool",
                parameters={"properties": {"message": {"type": "string"}}, "required": ["message"]},
            )
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        return {"ok": True, "tool": tool_name, "args": arguments}


@pytest.mark.asyncio
async def test_discovery_lists_modules_and_tools(monkeypatch):
    await reset_module_registry()
    await register_module("dummy", DummyModule, ModuleConfig(name="dummy"))

    async def _allow(*_args, **_kwargs):
        return True

    monkeypatch.setattr(MCPProtocol, "_has_module_permission", _allow)
    monkeypatch.setattr(MCPProtocol, "_has_tool_permission", _allow)

    mod = MCPDiscoveryModule(ModuleConfig(name="mcp_discovery"))
    ctx = RequestContext(request_id="mcp-discovery-test", user_id="1", metadata={})

    modules_res = await mod.execute_tool("mcp.modules.list", {}, context=ctx)
    assert any(m.get("module_id") == "dummy" for m in modules_res.get("modules", []))

    tools_res = await mod.execute_tool("mcp.tools.list", {"modules": ["dummy"]}, context=ctx)
    assert any(t.get("name") == "dummy.echo" for t in tools_res.get("tools", []))


@pytest.mark.asyncio
async def test_discovery_tools_list_parses_catalog_strict_string(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _handle_tools_list(
        _self: MCPProtocol,
        params: dict[str, Any],
        _context: RequestContext,
    ) -> dict[str, Any]:
        captured.update(params)
        return {"tools": []}

    monkeypatch.setattr(MCPProtocol, "_handle_tools_list", _handle_tools_list)

    mod = MCPDiscoveryModule(ModuleConfig(name="mcp_discovery"))
    ctx = RequestContext(request_id="mcp-discovery-strict", user_id="1", metadata={})

    await mod.execute_tool("mcp.tools.list", {"catalog_strict": "yes"}, context=ctx)

    assert captured["catalog_strict"] is True


@pytest.mark.asyncio
async def test_discovery_lists_catalogs_admin(monkeypatch):
    class FakePool:
        async def fetchall(self, query: str, *args):
            if "org_id IS NULL AND team_id IS NULL" in query:
                return [
                    {
                        "id": 1,
                        "name": "global-cat",
                        "description": "global",
                        "org_id": None,
                        "team_id": None,
                        "is_active": 1,
                        "created_at": None,
                        "updated_at": None,
                    }
                ]
            if "org_id IS NOT NULL" in query:
                return [
                    {
                        "id": 2,
                        "name": "org-cat",
                        "description": "org",
                        "org_id": 10,
                        "team_id": None,
                        "is_active": 1,
                        "created_at": None,
                        "updated_at": None,
                    }
                ]
            if "team_id IS NOT NULL" in query:
                return [
                    {
                        "id": 3,
                        "name": "team-cat",
                        "description": "team",
                        "org_id": 10,
                        "team_id": 99,
                        "is_active": 1,
                        "created_at": None,
                        "updated_at": None,
                    }
                ]
            return []

        async def fetchone(self, query: str, *args):
            return None

    async def _fake_get_db_pool():
        return FakePool()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.mcp_discovery_module.get_db_pool",
        _fake_get_db_pool,
    )

    mod = MCPDiscoveryModule(ModuleConfig(name="mcp_discovery"))
    ctx = RequestContext(
        request_id="mcp-discovery-catalogs",
        user_id="1",
        metadata={"roles": ["admin"]},
    )

    result = await mod.execute_tool("mcp.catalogs.list", {}, context=ctx)
    catalogs = result.get("catalogs", {})
    assert result.get("count") == 3
    assert catalogs["global"][0]["name"] == "global-cat"
    assert catalogs["org"][0]["name"] == "org-cat"
    assert catalogs["team"][0]["name"] == "team-cat"
