"""Stage 3 module tests (unit-level, stubbed): knowledge.search aggregation with stub sources."""

import pytest
from typing import Dict, Any, List

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import ModuleRegistry
from tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module import KnowledgeModule
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class StubNotesModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}
    async def get_tools(self) -> List[Dict[str, Any]]:
        return [{"name": "notes.search", "description": "", "inputSchema": {"type": "object"}}]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        return {
            "results": [{
                "id": "n1", "source": "notes", "title": "Note A", "snippet": "aaa", "uri": "notes://n1",
                "score": 0.9, "score_type": "fts", "created_at": None, "last_modified": None, "version": 1, "tags": None, "loc": None,
            }],
            "has_more": False,
            "next_offset": None,
            "total_estimated": 1,
        }


class StubMediaModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}
    async def get_tools(self) -> List[Dict[str, Any]]:
        return [{"name": "media.search", "description": "", "inputSchema": {"type": "object"}}]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        return {
            "results": [{
                "id": 1, "source": "media", "title": "Media A", "snippet": "bbb", "uri": "media://1",
                "score": 0.8, "score_type": "fts", "created_at": None, "last_modified": None, "version": 1, "tags": None,
                "media_type": "pdf", "url": None, "loc": None,
            }],
            "has_more": False,
            "next_offset": None,
            "total_estimated": 1,
        }


@pytest.mark.asyncio
@pytest.mark.usefixtures("allow_knowledge_source_tools")
async def test_knowledge_aggregates_stub_sources(monkeypatch):
    registry = ModuleRegistry()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module.get_module_registry",
        lambda: registry,
    )
    await registry.register_module("notes", StubNotesModule, ModuleConfig(name="notes"))
    await registry.register_module("media", StubMediaModule, ModuleConfig(name="media"))
    km = KnowledgeModule(ModuleConfig(name="knowledge"))
    await km.on_initialize()
    ctx = RequestContext(request_id="rx", user_id="1", client_id="cli")

    out = await km.execute_tool("knowledge.search", {"query": "x", "limit": 10}, context=ctx)
    assert isinstance(out, dict)
    assert isinstance(out.get("results"), list)
    uris = [r.get("uri") for r in out["results"]]
    assert "notes://n1" in uris
    assert "media://1" in uris
