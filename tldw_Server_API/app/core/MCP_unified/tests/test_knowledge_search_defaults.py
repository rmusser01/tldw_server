import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module import KnowledgeModule
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry, reset_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext, _trusted_compat_claims_metadata


class _NotesSearchModule(BaseModule):
    calls = []

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict]:
        return [{"name": "notes.search", "description": "", "inputSchema": {"type": "object"}}]

    async def execute_tool(self, tool_name, arguments, context=None):
        self.__class__.calls.append(tool_name)
        source = tool_name.split(".", 1)[0]
        return {
            "results": [
                {
                    "id": f"{source}-1",
                    "uri": f"{source}://1",
                    "source": source,
                    "score": 1.0,
                    "score_type": "fts",
                }
            ],
            "has_more": False,
            "next_offset": None,
            "total_estimated": 1,
        }


class _MediaSearchModule(_NotesSearchModule):
    calls = []

    async def get_tools(self) -> list[dict]:
        return [{"name": "media.search", "description": "", "inputSchema": {"type": "object"}}]


class _ChatsSearchModule(_NotesSearchModule):
    calls = []

    async def get_tools(self) -> list[dict]:
        return [{"name": "chats.search", "description": "", "inputSchema": {"type": "object"}}]


class _CharactersSearchModule(_NotesSearchModule):
    calls = []

    async def get_tools(self) -> list[dict]:
        return [{"name": "characters.search", "description": "", "inputSchema": {"type": "object"}}]


class _PromptsSearchModule(_NotesSearchModule):
    calls = []

    async def get_tools(self) -> list[dict]:
        return [{"name": "prompts.search", "description": "", "inputSchema": {"type": "object"}}]


class _RagSearchSentinelModule(_NotesSearchModule):
    calls = []

    async def get_tools(self) -> list[dict]:
        return [{"name": "rag.search", "description": "", "inputSchema": {"type": "object"}}]

    async def execute_tool(self, tool_name, arguments, context=None):
        self.__class__.calls.append(tool_name)
        raise AssertionError("knowledge.search must not invoke rag.search")


@pytest.mark.asyncio
async def test_knowledge_search_defaults_to_all_advertised_sources():
    await reset_module_registry()
    try:
        _NotesSearchModule.calls = []
        _MediaSearchModule.calls = []
        _ChatsSearchModule.calls = []
        _CharactersSearchModule.calls = []
        _PromptsSearchModule.calls = []
        _RagSearchSentinelModule.calls = []
        registry = get_module_registry()
        await registry.register_module("notes", _NotesSearchModule, ModuleConfig(name="notes"))
        await registry.register_module("media", _MediaSearchModule, ModuleConfig(name="media"))
        await registry.register_module("chats", _ChatsSearchModule, ModuleConfig(name="chats"))
        await registry.register_module("characters", _CharactersSearchModule, ModuleConfig(name="characters"))
        await registry.register_module("prompts", _PromptsSearchModule, ModuleConfig(name="prompts"))
        await registry.register_module("rag", _RagSearchSentinelModule, ModuleConfig(name="rag"))

        km = KnowledgeModule(ModuleConfig(name="knowledge"))
        await km.on_initialize()
        metadata = _trusted_compat_claims_metadata(
            auth_via="single_user_test_api_key",
            compat_claims_source="mounted_http",
        )
        metadata["roles"] = ["admin"]
        metadata["permissions"] = ["*"]
        ctx = RequestContext(request_id="knowledge-defaults", user_id="1", client_id="cli", metadata=metadata)

        result = await km.execute_tool("knowledge.search", {"query": "hello"}, context=ctx)

        assert _NotesSearchModule.calls == ["notes.search"]  # nosec B101
        assert _MediaSearchModule.calls == ["media.search"]  # nosec B101
        assert _ChatsSearchModule.calls == ["chats.search"]  # nosec B101
        assert _CharactersSearchModule.calls == ["characters.search"]  # nosec B101
        assert _PromptsSearchModule.calls == ["prompts.search"]  # nosec B101
        assert _RagSearchSentinelModule.calls == []  # nosec B101
        assert result["total_estimated"] == 5  # nosec B101
        assert {item["score_type"] for item in result["results"]} == {"fts"}  # nosec B101
    finally:
        await reset_module_registry()
