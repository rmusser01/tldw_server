"""Unit tests for chats.get window selection around an anchor message."""

import pytest
from typing import Any, Dict, List

from tldw_Server_API.app.core.MCP_unified.modules.implementations.chats_module import ChatsModule
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig


class FakeChatsDB:
    def __init__(self) -> None:
        # 7 messages with varying content sizes
        self._msgs = [
            {"id": "m0", "conversation_id": "c1", "sender": "u", "content": "a" * 5},
            {"id": "m1", "conversation_id": "c1", "sender": "u", "content": "b" * 5},
            {"id": "m2", "conversation_id": "c1", "sender": "u", "content": "c" * 10},
            {"id": "m3", "conversation_id": "c1", "sender": "u", "content": "d" * 12},
            {"id": "m4", "conversation_id": "c1", "sender": "u", "content": "e" * 7},
            {"id": "m5", "conversation_id": "c1", "sender": "u", "content": "f" * 9},
            {"id": "m6", "conversation_id": "c1", "sender": "u", "content": "g" * 3},
        ]

    def get_conversation_by_id(self, conversation_id: str) -> Dict[str, Any]:
        return {"id": conversation_id, "title": "T", "created_at": None, "last_modified": None, "version": 1}

    def get_messages_for_conversation(self, conversation_id: str, limit: int = 1000, offset: int = 0, order_by_timestamp: str = "ASC") -> List[Dict[str, Any]]:
        return list(self._msgs)


@pytest.mark.asyncio
async def test_chats_get_window_selection_budget():
    mod = ChatsModule(ModuleConfig(name="chats"))
    mod._open_db = lambda ctx: FakeChatsDB()  # type: ignore[attr-defined]

    # Anchor at m3 (index 3). cpt=1. Budget 25 tokens should include m3 (12), then m2 (10) to reach 22, and stop.
    out = await mod.execute_tool(
        "chats.get",
        {
            "conversation_id": "c1",
            "retrieval": {
                "mode": "chunk_with_siblings",
                "max_tokens": 25,
                "chars_per_token": 1,  # nosec B105
                "loc": {"message_id": "m3"},
            },
        },
        context=None,
    )

    assert isinstance(out, dict)  # nosec B101
    assert out["meta"]["loc"]["message_id"] == "m3"  # nosec B101
    attachments = out["attachments"]
    # Ensure anchor present
    ids = [m.get("id") for m in attachments]
    assert "m3" in ids  # nosec B101
    # With greedy left/right and budget=25, we expect two messages: m2 and m3 (10 + 12)
    assert set(ids) <= {"m2", "m3", "m4"}  # nosec B101
    # Combined token length <= 25
    total_chars = sum(len(m.get("content") or "") for m in attachments)
    assert total_chars <= 25  # nosec B101


@pytest.mark.asyncio
async def test_chats_title_search_uses_request_client_scope():
    from types import SimpleNamespace

    class _FakeSearchDb(FakeChatsDB):
        def __init__(self) -> None:
            super().__init__()
            self.last_title_search = None

        def search_conversations_by_title(self, query, character_id=None, limit=10, client_id=None):
            self.last_title_search = {
                "query": query,
                "character_id": character_id,
                "limit": limit,
                "client_id": client_id,
            }
            return [{"id": "c1", "title": "Hello", "created_at": None, "last_modified": None, "version": 1, "character_id": 3}]

    db = _FakeSearchDb()
    mod = ChatsModule(ModuleConfig(name="chats"))
    mod._open_db = lambda ctx: db  # type: ignore[attr-defined]

    out = await mod.execute_tool(
        "chats.search",
        {"query": "hello", "by": "title", "limit": 5},
        context=SimpleNamespace(client_id="tenant-7", metadata={}),
    )

    assert out["results"][0]["conversation_id"] == "c1"  # nosec B101
    assert db.last_title_search["client_id"] == "tenant-7"  # nosec B101
