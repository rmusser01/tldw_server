from __future__ import annotations

import pytest

from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    CharacterCardsRetriever,
    RetrievalConfig,
)
from tldw_Server_API.tests.RAG.conftest import DualBackendEnv


def _seed_character(env: DualBackendEnv) -> int:
    card_id = env.chacha_db.add_character_card(
        {
            "name": "Parity Tester",
            "description": "Character for dual-backend retrieval tests",
            "personality": "curious",
            "scenario": "evaluating systems",
            "system_prompt": "Be precise",
            "first_message": "Hello!",
            "creator": "pytest",
            "character_version": 1,
            "tags": ["parity", "postgres"],
        }
    )
    assert card_id is not None
    env.chacha_db.rebuild_full_text_indexes()
    return int(card_id)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dual_backend_character_cards_retrieval(dual_backend_env: DualBackendEnv) -> None:
    env = dual_backend_env
    _seed_character(env)

    retriever = CharacterCardsRetriever(
        db_path=env.chacha_db.db_path_str,
        config=RetrievalConfig(max_results=5),
        chacha_db=env.chacha_db,
    )

    documents = await retriever.retrieve("Parity", include_chats=False)
    assert documents, f"Character retriever returned no results for {env.label}"
    top = documents[0]
    assert "Parity Tester" in top.content or "Parity Tester" in (top.metadata or {}).get("name", "")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_dual_backend_character_chats_retrieval(dual_backend_env: DualBackendEnv) -> None:
    """Ensure chats-on retrieval returns chat message documents for both SQLite and Postgres backends."""
    env = dual_backend_env
    card_id = _seed_character(env)

    # Create a conversation and a chat message tied to the character
    conv_id = env.chacha_db.add_conversation({
        "character_id": card_id,
        "title": "Parity Conversation",
        "client_id": env.chacha_db.client_id,
    })
    assert conv_id is not None

    msg_id = env.chacha_db.add_message({
        "conversation_id": conv_id,
        "sender": "Tester",
        "content": "This chat mentions Parity retrieval explicitly.",
        "client_id": env.chacha_db.client_id,
    })
    assert msg_id is not None

    # Ensure FTS structures are ready for both backends
    env.chacha_db.rebuild_full_text_indexes()

    retriever = CharacterCardsRetriever(
        db_path=env.chacha_db.db_path_str,
        config=RetrievalConfig(max_results=8),
        chacha_db=env.chacha_db,
    )

    docs = await retriever.retrieve("Parity", include_chats=True)
    assert docs, f"No documents returned for {env.label} with chats enabled"
    # Expect at least one chat doc referencing our message
    has_chat = any(d.id.startswith("chat_") or (d.metadata or {}).get("type") == "chat_message" for d in docs)
    assert has_chat, f"Expected at least one chat document in results for {env.label}"
