"""Stage 3 persona scope enforcement tests for MCP retrieval surfaces."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import BaseModule, ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.chats_module import ChatsModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.knowledge_module import KnowledgeModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.media_module import MediaModule
from tldw_Server_API.app.core.MCP_unified.modules.implementations.notes_module import NotesModule
from tldw_Server_API.app.core.MCP_unified.modules.registry import get_module_registry, reset_module_registry
from tldw_Server_API.app.core.MCP_unified.protocol import RequestContext


class _ChatsScopeDB:
    def __init__(self) -> None:
        self._conversations = {
            "conv_a": {"id": "conv_a", "title": "A", "character_id": 1, "created_at": None, "last_modified": None, "version": 1},
            "conv_b": {"id": "conv_b", "title": "B", "character_id": 2, "created_at": None, "last_modified": None, "version": 1},
        }

    def search_conversations_by_title(self, _query: str, character_id: int | None = None, limit: int = 10, client_id: Any = None):
        rows = list(self._conversations.values())
        if character_id is not None:
            rows = [r for r in rows if int(r.get("character_id") or 0) == int(character_id)]
        return rows[:limit]

    def search_messages_by_content(self, _query: str, conversation_id: str | None = None, limit: int = 10):
        rows = [
            {"id": "m_a", "conversation_id": "conv_a", "content": "hello a", "sender": "user", "timestamp": None, "last_modified": None, "version": 1},
            {"id": "m_b", "conversation_id": "conv_b", "content": "hello b", "sender": "user", "timestamp": None, "last_modified": None, "version": 1},
        ]
        if conversation_id is not None:
            rows = [r for r in rows if str(r.get("conversation_id")) == str(conversation_id)]
        return rows[:limit]

    def get_conversation_by_id(self, conversation_id: str):
        return self._conversations.get(str(conversation_id))

    def get_messages_for_conversation(self, conversation_id: str, limit: int = 1000, offset: int = 0, order_by_timestamp: str = "ASC"):
        rows = [
            {"id": "m1", "sender": "user", "content": f"content for {conversation_id}"},
        ]
        return rows[offset : offset + limit]

    def close_all_connections(self):
        return None


class _MediaScopeDB:
    def __init__(self) -> None:
        self.last_media_ids_filter: Any = None
        self._rows = {
            1: {
                "id": 1,
                "title": "One",
                "content": "alpha",
                "type": "text",
                "url": None,
                "ingestion_date": None,
                "last_modified": None,
                "version": 1,
                "owner_user_id": "1",
            },
            2: {
                "id": 2,
                "title": "Two",
                "content": "beta",
                "type": "text",
                "url": None,
                "ingestion_date": None,
                "last_modified": None,
                "version": 1,
                "owner_user_id": "1",
            },
        }

    def search_media_db(
        self,
        search_query: Any = None,
        search_fields: Any = None,
        media_types: Any = None,
        date_range: Any = None,
        must_have_keywords: Any = None,
        must_not_have_keywords: Any = None,
        sort_by: Any = None,
        media_ids_filter: Any = None,
        page: int = 1,
        results_per_page: int = 20,
        include_trash: bool = False,
        include_deleted: bool = False,
    ):
        self.last_media_ids_filter = media_ids_filter
        rows = [dict(v) for v in self._rows.values()]
        if media_ids_filter is not None:
            allowed = {int(v) for v in media_ids_filter}
            rows = [r for r in rows if int(r.get("id") or 0) in allowed]
        return rows[:results_per_page], len(rows)

    def get_media_by_id(self, media_id: int, include_deleted: bool = False, include_trash: bool = False):
        row = self._rows.get(int(media_id))
        if row is None:
            return None
        return dict(row)

    def has_unvectorized_chunks(self, media_id: int) -> bool:
        return False


class _NotesScopeDB:
    def __init__(self) -> None:
        self._rows = {
            "n1": {"id": "n1", "title": "N1", "content": "note one", "created_at": None, "last_modified": None, "version": 1},
            "n2": {"id": "n2", "title": "N2", "content": "note two", "created_at": None, "last_modified": None, "version": 1},
        }

    def search_notes(self, _query: str, limit: int = 10, offset: int = 0):
        rows = [dict(self._rows["n1"]), dict(self._rows["n2"])]
        return rows[offset : offset + limit]

    def count_notes_matching(self, _query: str) -> int:
        return 2

    def get_note_by_id(self, note_id: str):
        row = self._rows.get(str(note_id))
        if row is None:
            return None
        return dict(row)

    def close_all_connections(self):
        return None


@pytest.mark.asyncio
async def test_chats_scope_enforces_conversation_and_character_filters():
    mod = ChatsModule(ModuleConfig(name="chats"))
    db = _ChatsScopeDB()
    mod._open_db = lambda context: db  # type: ignore[assignment]

    ctx_conversation = SimpleNamespace(metadata={"persona_scope": {"explicit_ids": {"conversation_id": ["conv_a"]}}})
    out_conversation = await mod.execute_tool(
        "chats.search",
        {"query": "hello", "by": "both", "limit": 10, "offset": 0},
        context=ctx_conversation,
    )
    assert out_conversation["results"]  # nosec B101
    assert all(str(r.get("conversation_id")) == "conv_a" for r in out_conversation["results"])  # nosec B101
    with pytest.raises(PermissionError):
        await mod.execute_tool(
            "chats.get",
            {"conversation_id": "conv_b", "retrieval": {"mode": "snippet"}},
            context=ctx_conversation,
        )

    ctx_character = SimpleNamespace(metadata={"persona_scope": {"explicit_ids": {"character_id": ["2"]}}})
    out_character = await mod.execute_tool(
        "chats.search",
        {"query": "hello", "by": "both", "limit": 10, "offset": 0},
        context=ctx_character,
    )
    assert out_character["results"]  # nosec B101
    assert all(str(r.get("conversation_id")) == "conv_b" for r in out_character["results"])  # nosec B101
    with pytest.raises(PermissionError):
        await mod.execute_tool(
            "chats.get",
            {"conversation_id": "conv_a", "retrieval": {"mode": "snippet"}},
            context=ctx_character,
        )


@pytest.mark.asyncio
async def test_chats_scope_enforces_message_branch_filters():
    mod = ChatsModule(ModuleConfig(name="chats"))
    db = _ChatsScopeDB()
    mod._open_db = lambda context: db  # type: ignore[assignment]

    ctx_character = SimpleNamespace(metadata={"persona_scope": {"explicit_ids": {"character_id": ["2"]}}})
    out_character = await mod.execute_tool(
        "chats.search",
        {"query": "hello", "by": "message", "limit": 10, "offset": 0},
        context=ctx_character,
    )
    assert out_character["results"]  # nosec B101
    assert all(str(r.get("conversation_id")) == "conv_b" for r in out_character["results"])  # nosec B101

    out_character_mismatch = await mod.execute_tool(
        "chats.search",
        {"query": "hello", "by": "message", "limit": 10, "offset": 0, "character_id": 1},
        context=ctx_character,
    )
    assert out_character_mismatch["results"] == []  # nosec B101

    ctx_conversation = SimpleNamespace(metadata={"persona_scope": {"explicit_ids": {"conversation_id": ["conv_a"]}}})
    out_conversation = await mod.execute_tool(
        "chats.search",
        {"query": "hello", "by": "message", "limit": 10, "offset": 0},
        context=ctx_conversation,
    )
    assert out_conversation["results"]  # nosec B101
    assert all(str(r.get("conversation_id")) == "conv_a" for r in out_conversation["results"])  # nosec B101


@pytest.mark.asyncio
async def test_media_scope_enforces_media_search_and_get():
    mod = MediaModule(ModuleConfig(name="media"))
    db = _MediaScopeDB()
    mod._open_media_db = lambda context: db  # type: ignore[assignment]

    ctx = SimpleNamespace(user_id="1", metadata={"persona_scope": {"explicit_ids": {"media_id": ["2"]}}})
    out = await mod.execute_tool(
        "media.search",
        {"query": "alpha", "limit": 10, "offset": 0},
        context=ctx,
    )
    assert db.last_media_ids_filter == [2]  # nosec B101
    assert [int(r.get("id")) for r in out["results"]] == [2]  # nosec B101

    with pytest.raises(PermissionError):
        await mod.execute_tool("media.get", {"media_id": 1, "retrieval": {"mode": "snippet"}}, context=ctx)

    allowed = await mod.execute_tool("media.get", {"media_id": 2, "retrieval": {"mode": "snippet"}}, context=ctx)
    assert int(allowed["meta"]["id"]) == 2  # nosec B101


@pytest.mark.asyncio
async def test_notes_scope_enforces_notes_search_and_get():
    mod = NotesModule(ModuleConfig(name="notes"))
    db = _NotesScopeDB()
    mod._open_db = lambda context: db  # type: ignore[assignment]

    ctx = SimpleNamespace(metadata={"persona_scope": {"explicit_ids": {"note_id": ["n2"]}}})
    out = await mod.execute_tool(
        "notes.search",
        {"query": "note", "limit": 10, "offset": 0},
        context=ctx,
    )
    assert [str(r.get("id")) for r in out["results"]] == ["n2"]  # nosec B101

    with pytest.raises(PermissionError):
        await mod.execute_tool("notes.get", {"note_id": "n1", "retrieval": {"mode": "snippet"}}, context=ctx)

    allowed = await mod.execute_tool("notes.get", {"note_id": "n2", "retrieval": {"mode": "snippet"}}, context=ctx)
    assert str(allowed["meta"]["id"]) == "n2"  # nosec B101


@pytest.mark.asyncio
async def test_direct_characters_and_prompts_tools_honor_persona_scope():
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.characters_module import CharactersModule
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module import PromptsModule

    class _CharactersDbForScopeTests:
        def search_character_cards(self, query, limit=10):
            return [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]

        def get_character_card_by_id(self, character_id):
            return {"id": character_id, "name": f"char-{character_id}", "description": "desc", "version": 1}

        def close_all_connections(self):
            return None

    class _PromptsDbForScopeTests:
        def search_prompts(self, search_query, search_fields=None, page=1, results_per_page=10, include_deleted=False):
            rows = [
                {"id": 1, "name": "p1", "details": "desc", "version": 1},
                {"id": 2, "name": "p2", "details": "desc", "version": 1},
            ]
            return rows, len(rows)

        def get_prompt_by_id(self, prompt_id):
            return {"id": prompt_id, "name": f"p{prompt_id}", "details": "desc", "version": 1}

        def get_prompt_by_name(self, name):
            prompt_id = 2 if str(name) == "2" else 1
            return {"id": prompt_id, "name": str(name), "details": "desc", "version": 1}

        def close_connection(self):
            return None

    ctx = RequestContext(
        request_id="persona-direct-tools",
        user_id="1",
        metadata={
            "persona_scope": {
                "scope_snapshot_id": "snap-1",
                "materialized_scope": {
                    "explicit_ids": {
                        "character_id": ["2"],
                        "prompt_id": ["2"],
                    }
                },
            }
        },
    )

    characters = CharactersModule(ModuleConfig(name="characters"))
    prompts = PromptsModule(ModuleConfig(name="prompts"))
    characters._open_db = lambda _ctx: _CharactersDbForScopeTests()  # type: ignore[attr-defined]
    prompts._open_db = lambda _ctx: _PromptsDbForScopeTests()  # type: ignore[attr-defined]

    with pytest.raises(PermissionError, match="Character access denied by persona scope"):
        await characters.execute_tool("characters.get", {"character_id": 1}, context=ctx)

    with pytest.raises(PermissionError, match="Prompt access denied by persona scope"):
        await prompts.execute_tool("prompts.get", {"prompt_id_or_name": "1"}, context=ctx)


@pytest.mark.asyncio
async def test_prompts_search_filters_out_of_scope_results():
    from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
    from tldw_Server_API.app.core.MCP_unified.modules.implementations.prompts_module import PromptsModule

    class _PromptsDbForScopeSearchTests:
        def search_prompts(self, search_query, search_fields=None, page=1, results_per_page=10, include_deleted=False):
            rows = [
                {"id": 1, "name": "p1", "details": "desc 1", "version": 1},
                {"id": 2, "name": "p2", "details": "desc 2", "version": 1},
            ]
            return rows, len(rows)

        def close_connection(self):
            return None

    ctx = RequestContext(
        request_id="persona-prompts-search",
        user_id="1",
        metadata={
            "persona_scope": {
                "scope_snapshot_id": "snap-1",
                "materialized_scope": {"explicit_ids": {"prompt_id": ["2"]}},
            }
        },
    )

    prompts = PromptsModule(ModuleConfig(name="prompts"))
    prompts._open_db = lambda _ctx: _PromptsDbForScopeSearchTests()  # type: ignore[attr-defined]

    result = await prompts.execute_tool("prompts.search", {"query": "p"}, context=ctx)

    assert [row["id"] for row in result["results"]] == [2]  # nosec B101


class _ScopeBypassNotesModule(BaseModule):
    last_args: dict[str, Any] | None = None

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "notes.search", "description": "", "inputSchema": {"type": "object"}},
            {"name": "notes.get", "description": "", "inputSchema": {"type": "object"}},
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        type(self).last_args = dict(arguments)
        if tool_name == "notes.search":
            return {
                "results": [
                    {"id": "n1", "source": "notes", "title": "N1", "snippet": "one", "uri": "notes://n1", "score": 1.0},
                    {"id": "n2", "source": "notes", "title": "N2", "snippet": "two", "uri": "notes://n2", "score": 0.9},
                ],
                "has_more": False,
                "next_offset": None,
                "total_estimated": 2,
            }
        if tool_name == "notes.get":
            note_id = str(arguments.get("note_id"))
            return {"meta": {"id": note_id, "source": "notes", "uri": f"notes://{note_id}"}, "content": note_id}
        raise ValueError(tool_name)


class _ScopeBypassMediaModule(BaseModule):
    last_args: dict[str, Any] | None = None

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "media.search", "description": "", "inputSchema": {"type": "object"}},
            {"name": "media.get", "description": "", "inputSchema": {"type": "object"}},
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        type(self).last_args = dict(arguments)
        if tool_name == "media.search":
            return {
                "results": [
                    {"id": 1, "source": "media", "title": "M1", "snippet": "one", "uri": "media://1", "score": 1.0},
                    {"id": 2, "source": "media", "title": "M2", "snippet": "two", "uri": "media://2", "score": 0.9},
                ],
                "has_more": False,
                "next_offset": None,
                "total_estimated": 2,
            }
        if tool_name == "media.get":
            media_id = int(arguments.get("media_id"))
            return {"meta": {"id": media_id, "source": "media", "uri": f"media://{media_id}"}, "content": str(media_id)}
        raise ValueError(tool_name)


class _ScopeBypassChatsModule(BaseModule):
    last_args: dict[str, Any] | None = None

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "chats.search", "description": "", "inputSchema": {"type": "object"}},
            {"name": "chats.get", "description": "", "inputSchema": {"type": "object"}},
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        type(self).last_args = dict(arguments)
        if tool_name == "chats.search":
            return {
                "results": [
                    {"id": "conv_a", "conversation_id": "conv_a", "source": "chats", "title": "A", "snippet": "one", "uri": "chats://conv_a", "score": 1.0},
                    {"id": "conv_b", "conversation_id": "conv_b", "source": "chats", "title": "B", "snippet": "two", "uri": "chats://conv_b", "score": 0.9},
                ],
                "has_more": False,
                "next_offset": None,
                "total_estimated": 2,
            }
        if tool_name == "chats.get":
            conv_id = str(arguments.get("conversation_id"))
            return {"meta": {"id": conv_id, "source": "chats", "uri": f"chats://{conv_id}"}, "content": conv_id}
        raise ValueError(tool_name)


class _ScopeBypassCharactersModule(BaseModule):
    last_args: dict[str, Any] | None = None

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "characters.search", "description": "", "inputSchema": {"type": "object"}},
            {"name": "characters.get", "description": "", "inputSchema": {"type": "object"}},
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        type(self).last_args = dict(arguments)
        if tool_name == "characters.search":
            return {
                "results": [
                    {
                        "id": 1,
                        "source": "characters",
                        "title": "C1",
                        "snippet": "one",
                        "uri": "characters://1",
                        "score": 1.0,
                    },
                    {
                        "id": 2,
                        "source": "characters",
                        "title": "C2",
                        "snippet": "two",
                        "uri": "characters://2",
                        "score": 0.9,
                    },
                ],
                "has_more": False,
                "next_offset": None,
                "total_estimated": 2,
            }
        if tool_name == "characters.get":
            character_id = int(arguments.get("character_id"))
            return {
                "meta": {"id": character_id, "source": "characters", "uri": f"characters://{character_id}"},
                "content": str(character_id),
            }
        raise ValueError(tool_name)


class _ScopeBypassPromptsModule(BaseModule):
    last_args: dict[str, Any] | None = None

    async def on_initialize(self) -> None:
        return None

    async def on_shutdown(self) -> None:
        return None

    async def check_health(self) -> dict[str, bool]:
        return {"ok": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            {"name": "prompts.search", "description": "", "inputSchema": {"type": "object"}},
            {"name": "prompts.get", "description": "", "inputSchema": {"type": "object"}},
        ]

    async def execute_tool(self, tool_name: str, arguments: dict[str, Any], context: Any | None = None) -> Any:
        type(self).last_args = dict(arguments)
        if tool_name == "prompts.search":
            return {
                "results": [
                    {
                        "id": 1,
                        "source": "prompts",
                        "title": "P1",
                        "snippet": "one",
                        "uri": "prompts://1",
                        "score": 1.0,
                    },
                    {
                        "id": 2,
                        "source": "prompts",
                        "title": "P2",
                        "snippet": "two",
                        "uri": "prompts://2",
                        "score": 0.9,
                    },
                ],
                "has_more": False,
                "next_offset": None,
                "total_estimated": 2,
            }
        if tool_name == "prompts.get":
            ident = str(arguments.get("prompt_id_or_name"))
            try:
                prompt_id = int(ident)
            except (TypeError, ValueError):
                prompt_id = 2 if ident.lower() == "p2" else 1
            return {
                "meta": {"id": prompt_id, "source": "prompts", "uri": f"prompts://{prompt_id}"},
                "content": str(prompt_id),
            }
        raise ValueError(tool_name)


@pytest.mark.asyncio
async def test_knowledge_scope_propagates_filters_and_blocks_bypass():
    await reset_module_registry()
    try:
        registry = get_module_registry()
        await registry.register_module("notes_scope_bypass", _ScopeBypassNotesModule, ModuleConfig(name="notes_scope_bypass"))
        await registry.register_module("media_scope_bypass", _ScopeBypassMediaModule, ModuleConfig(name="media_scope_bypass"))
        await registry.register_module("chats_scope_bypass", _ScopeBypassChatsModule, ModuleConfig(name="chats_scope_bypass"))
        await registry.register_module(
            "characters_scope_bypass",
            _ScopeBypassCharactersModule,
            ModuleConfig(name="characters_scope_bypass"),
        )
        await registry.register_module("prompts_scope_bypass", _ScopeBypassPromptsModule, ModuleConfig(name="prompts_scope_bypass"))

        km = KnowledgeModule(ModuleConfig(name="knowledge"))
        await km.on_initialize()
        ctx = RequestContext(
            request_id="persona-scope-knowledge",
            user_id="1",
            client_id="cli",
            metadata={
                "persona_scope": {
                    "explicit_ids": {
                        "note_id": ["n2"],
                        "media_id": ["2"],
                        "conversation_id": ["conv_b"],
                        "character_id": ["2"],
                        "prompt_id": ["2"],
                    }
                }
            },
        )

        out = await km.execute_tool(
            "knowledge.search",
            {"query": "x", "limit": 20, "sources": ["notes", "media", "chats", "characters", "prompts"]},
            context=ctx,
        )

        assert _ScopeBypassNotesModule.last_args is not None  # nosec B101
        assert _ScopeBypassNotesModule.last_args.get("note_ids_filter") == ["n2"]  # nosec B101
        assert _ScopeBypassMediaModule.last_args is not None  # nosec B101
        assert _ScopeBypassMediaModule.last_args.get("media_ids_filter") == ["2"]  # nosec B101
        assert _ScopeBypassChatsModule.last_args is not None  # nosec B101
        assert _ScopeBypassChatsModule.last_args.get("conversation_ids_filter") == ["conv_b"]  # nosec B101
        assert _ScopeBypassCharactersModule.last_args is not None  # nosec B101
        assert _ScopeBypassCharactersModule.last_args.get("character_ids_filter") == ["2"]  # nosec B101
        assert _ScopeBypassPromptsModule.last_args is not None  # nosec B101
        assert _ScopeBypassPromptsModule.last_args.get("prompt_ids_filter") == ["2"]  # nosec B101

        uris = {str(item.get("uri")) for item in out.get("results", [])}
        assert "notes://n1" not in uris  # nosec B101
        assert "media://1" not in uris  # nosec B101
        assert "chats://conv_a" not in uris  # nosec B101
        assert "characters://1" not in uris  # nosec B101
        assert "prompts://1" not in uris  # nosec B101
        assert "notes://n2" in uris  # nosec B101
        assert "media://2" in uris  # nosec B101
        assert "chats://conv_b" in uris  # nosec B101
        assert "characters://2" in uris  # nosec B101
        assert "prompts://2" in uris  # nosec B101

        with pytest.raises(PermissionError):
            await km.execute_tool("knowledge.get", {"source": "media", "id": 1}, context=ctx)

        with pytest.raises(PermissionError):
            await km.execute_tool("knowledge.get", {"source": "notes", "id": "n1"}, context=ctx)

        with pytest.raises(PermissionError):
            await km.execute_tool("knowledge.get", {"source": "chats", "id": "conv_a"}, context=ctx)

        with pytest.raises(PermissionError):
            await km.execute_tool("knowledge.get", {"source": "characters", "id": 1}, context=ctx)

        with pytest.raises(PermissionError):
            await km.execute_tool("knowledge.get", {"source": "prompts", "id": 1}, context=ctx)

        allowed = await km.execute_tool("knowledge.get", {"source": "media", "id": 2}, context=ctx)
        assert int(allowed["meta"]["id"]) == 2  # nosec B101

        allowed_character = await km.execute_tool("knowledge.get", {"source": "characters", "id": 2}, context=ctx)
        assert int(allowed_character["meta"]["id"]) == 2  # nosec B101

        allowed_prompt = await km.execute_tool("knowledge.get", {"source": "prompts", "id": 2}, context=ctx)
        assert int(allowed_prompt["meta"]["id"]) == 2  # nosec B101

        allowed_prompt_by_name = await km.execute_tool("knowledge.get", {"source": "prompts", "id": "p2"}, context=ctx)
        assert int(allowed_prompt_by_name["meta"]["id"]) == 2  # nosec B101
    finally:
        await reset_module_registry()
