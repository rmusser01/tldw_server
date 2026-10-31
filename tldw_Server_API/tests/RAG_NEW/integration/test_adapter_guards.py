"""
Integration-style tests validating production-mode guards and adapter-backed retrieval
for Notes and Character retrievers in the RAG module.

These tests ensure:
- Raw SQL fallback is blocked in production when no adapter is provided
- When an adapter is provided, retrieval succeeds and does not use raw SQL
"""

import os
import pytest
from typing import List, Dict, Any

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    NotesDBRetriever,
    CharacterCardsRetriever,
    RetrievalConfig,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notes_retriever_requires_adapter_in_production(monkeypatch):
    """In production, NotesDBRetriever must not fall back to raw SQL."""
    monkeypatch.setenv("tldw_production", "true")
    retr = NotesDBRetriever(db_path=":memory:", chacha_db=None)
    with pytest.raises(RuntimeError):
        await retr.retrieve("test query", notebook_id=None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_notes_retriever_uses_adapter_without_raw_sql(monkeypatch):
    """With adapter provided, NotesDBRetriever should not use raw SQL fallback."""
    monkeypatch.setenv("tldw_production", "true")

    # Patch BaseRetriever._execute_query to fail if called (to prove adapter path is used)
    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import BaseRetriever

    async_calls: Dict[str, int] = {"executed": 0}

    def _no_sql(*args, **kwargs):  # noqa: ANN001, ANN002
        async_calls["executed"] += 1
        raise AssertionError("_execute_query should not be called when adapter is provided")

    monkeypatch.setattr(BaseRetriever, "_execute_query", _no_sql, raising=True)

    class FakeChaCha:
        def search_notes(self, query: str, limit: int, *, match_any: bool = False) -> List[Dict[str, Any]]:  # noqa: D401
            return [
                {
                    "id": 1,
                    "title": "Test Note",
                    "content": "This is a test note",
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-02T00:00:00",
                    "notebook_id": 1,
                    "notebook_name": "Default",
                    "rank": 0.9,
                }
            ]

    retr = NotesDBRetriever(db_path=":memory:", chacha_db=FakeChaCha())
    docs = await retr.retrieve("test query")
    assert len(docs) == 1
    assert str(docs[0].id).startswith("note_")
    assert docs[0].source == DataSource.NOTES
    # Ensure raw SQL path was not hit
    assert async_calls["executed"] == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_character_retriever_uses_adapter_without_raw_sql(monkeypatch):
    """With adapter provided, CharacterCardsRetriever should not use raw SQL fallback."""
    monkeypatch.setenv("tldw_production", "true")

    from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import BaseRetriever

    async_calls: Dict[str, int] = {"executed": 0}

    def _no_sql(*args, **kwargs):  # noqa: ANN001, ANN002
        async_calls["executed"] += 1
        raise AssertionError("_execute_query should not be called when adapter is provided")

    monkeypatch.setattr(BaseRetriever, "_execute_query", _no_sql, raising=True)

    class FakeChaCha:
        def search_character_cards(self, query: str, limit: int) -> List[Dict[str, Any]]:  # noqa: D401
            return [
                {
                    "id": 10,
                    "name": "Alice",
                    "description": "Helper character",
                    "personality": "Friendly",
                    "scenario": "Guide",
                    "first_message": "Hello!",
                    "creator": "system",
                    "version": 1,
                    "rank": 0.8,
                }
            ]

        def search_messages_by_content(self, query: str, limit: int) -> List[Dict[str, Any]]:  # noqa: D401
            return [
                {
                    "id": 5,
                    "content": "Hi there",
                    "sender": "Alice",
                    "timestamp": "2024-01-01T12:00:00",
                    "conversation_id": 1,
                }
            ]

        def get_conversation_by_id(self, cid: int) -> Dict[str, Any]:  # noqa: D401
            return {"id": cid, "character_id": 10}

        def get_character_card_by_id(self, cid: int) -> Dict[str, Any]:  # noqa: D401
            return {"id": cid, "name": "Alice"}

    retr = CharacterCardsRetriever(db_path=":memory:", chacha_db=FakeChaCha())
    docs = await retr.retrieve("hello", include_chats=True)
    assert len(docs) >= 1
    assert docs[0].source == DataSource.CHARACTER_CARDS
    assert async_calls["executed"] == 0
