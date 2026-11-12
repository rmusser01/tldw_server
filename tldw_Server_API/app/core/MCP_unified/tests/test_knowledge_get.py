"""Knowledge.get tests using stub source modules for chats, characters, and prompts."""

import pytest
from typing import Dict, Any, List

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.registry import (
    get_module_registry,
    reset_module_registry,
)
from tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module import KnowledgeModule
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext



class StubChatsModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}
    async def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "chats.get", "description": "", "inputSchema": {"type": "object"}},
            {"name": "chats.search", "description": "", "inputSchema": {"type": "object"}},
        ]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        if tool_name == "chats.get":
            cid = arguments.get("conversation_id")
            return {
                "meta": {
                    "id": cid,
                    "source": "chats",
                    "title": "Chat Title",
                    "snippet": "hello",
                    "uri": f"chats://{cid}",
                    "score": 1.0,
                    "score_type": "fts",
                    "created_at": None,
                    "last_modified": None,
                    "version": 1,
                    "tags": None,
                    "loc": None,
                },
                "content": "hello world",
                "attachments": None,
            }
        raise ValueError(tool_name)


class StubCharactersModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}
    async def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "characters.get", "description": "", "inputSchema": {"type": "object"}},
            {"name": "characters.search", "description": "", "inputSchema": {"type": "object"}},
        ]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        if tool_name == "characters.get":
            char_id = int(arguments.get("character_id"))
            return {
                "meta": {
                    "id": char_id,
                    "source": "characters",
                    "title": "Alice",
                    "snippet": "desc",
                    "uri": f"characters://{char_id}",
                    "score": 1.0,
                    "score_type": "fts",
                    "created_at": None,
                    "last_modified": None,
                    "version": 1,
                    "tags": None,
                    "loc": None,
                },
                "content": {"name": "Alice"},
                "attachments": None,
            }
        raise ValueError(tool_name)


class StubPromptsModule(BaseModule):
    async def on_initialize(self) -> None: ...
    async def on_shutdown(self) -> None: ...
    async def check_health(self) -> Dict[str, bool]:
        return {"ok": True}
    async def get_tools(self) -> List[Dict[str, Any]]:
        return [
            {"name": "prompts.get", "description": "", "inputSchema": {"type": "object"}},
            {"name": "prompts.search", "description": "", "inputSchema": {"type": "object"}},
        ]
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any], context: Any | None = None) -> Any:
        if tool_name == "prompts.get":
            ident = str(arguments.get("prompt_id_or_name"))
            return {
                "meta": {
                    "id": 123,
                    "source": "prompts",
                    "title": ident,
                    "snippet": "s",
                    "uri": f"prompts://123",
                    "score": 1.0,
                    "score_type": "fts",
                    "created_at": None,
                    "last_modified": None,
                    "version": 1,
                    "tags": None,
                    "loc": None,
                },
                "content": {"name": ident},
                "attachments": None,
            }
        raise ValueError(tool_name)


@pytest.mark.asyncio
@pytest.mark.usefixtures("allow_knowledge_source_tools")
async def test_knowledge_get_for_additional_sources():
    await reset_module_registry()
    registry = get_module_registry()
    await registry.register_module("chats", StubChatsModule, ModuleConfig(name="chats"))
    await registry.register_module("characters", StubCharactersModule, ModuleConfig(name="characters"))
    await registry.register_module("prompts", StubPromptsModule, ModuleConfig(name="prompts"))

    km = KnowledgeModule(ModuleConfig(name="knowledge"))
    await km.on_initialize()
    ctx = RequestContext(request_id="r1", user_id="1", client_id="cli")

    # chats
    out_chats = await km.execute_tool("knowledge.get", {"source": "chats", "id": "conv1"}, context=ctx)
    assert out_chats["meta"]["source"] == "chats"
    assert out_chats["meta"]["id"] == "conv1"

    # characters
    out_chars = await km.execute_tool("knowledge.get", {"source": "characters", "id": 42}, context=ctx)
    assert out_chars["meta"]["source"] == "characters"
    assert out_chars["meta"]["id"] == 42

    # prompts
    out_prompts = await km.execute_tool("knowledge.get", {"source": "prompts", "id": "greeting"}, context=ctx)
    assert out_prompts["meta"]["source"] == "prompts"
    assert out_prompts["content"]["name"] == "greeting"
    await reset_module_registry()
