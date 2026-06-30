import pytest

from tldw_Server_API.app.core.RAG.rag_service.request_resolution import ResolvedRAGRequest
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import (
    _normalize_plan_sources,
)
from tldw_Server_API.app.core.RAG.rag_service.retrieval_plan import (
    RetrievalPlan,
    build_retrieval_plan,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


pytestmark = pytest.mark.unit


def test_build_retrieval_plan_centralizes_namespace_and_sources() -> None:
    resolved = ResolvedRAGRequest(
        query="retrieval plan",
        strategy="standard",
        payload={
            "query": "retrieval plan",
            "sources": ["media_db", "notes"],
            "search_mode": "hybrid",
            "top_k": 5,
            "min_score": 0.25,
        },
        index_namespace="tenant-a",
        rag_profile=None,
        user_id="9",
        feedback_user_id="9",
    )

    plan = build_retrieval_plan(resolved)

    assert plan.index_namespace == "tenant-a"
    assert plan.sources == ("media_db", "notes")
    assert plan.collection_names["media_db"] == "user_9_media_embeddings"
    assert plan.collection_names["notes"] == "user_9_notes_embeddings"
    assert plan.search_mode == "hybrid"
    assert plan.top_k == 5
    assert plan.min_score == 0.25


def test_build_retrieval_plan_keeps_chats_and_characters_distinct() -> None:
    resolved = ResolvedRAGRequest(
        query="character retrieval",
        strategy="standard",
        payload={
            "query": "character retrieval",
            "sources": ["characters", "chats"],
            "search_mode": "hybrid",
            "top_k": 5,
            "min_score": 0.0,
        },
        index_namespace=None,
        rag_profile=None,
        user_id="9",
        feedback_user_id="9",
    )

    plan = build_retrieval_plan(resolved)

    assert plan.sources == ("characters", "chats")
    assert plan.collection_names["character_cards"] == "user_9_character_embeddings"


def test_build_retrieval_plan_normalizes_public_knowledge_source_aliases() -> None:
    resolved = ResolvedRAGRequest(
        query="all knowledge sources",
        strategy="standard",
        payload={
            "query": "all knowledge sources",
            "sources": [
                "media",
                "notes_db",
                "chat_history",
                "character_cards",
                "task_boards",
                "prompts_db",
                "worldbooks",
                "chat_dictionaries",
            ],
        },
        index_namespace=None,
        rag_profile=None,
        user_id="9",
        feedback_user_id="9",
    )

    plan = build_retrieval_plan(resolved)

    assert plan.sources == (
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    )


def test_database_retriever_plan_sources_keep_chats_separate_from_characters() -> None:
    plan = RetrievalPlan(
        query="source map",
        sources=("chats", "characters", "prompts", "world_books", "dictionaries"),
        search_mode="hybrid",
        top_k=5,
        min_score=0.0,
        index_namespace=None,
    )

    assert _normalize_plan_sources(plan) == [
        DataSource.CHAT_HISTORY,
        DataSource.CHARACTER_CARDS,
        DataSource.PROMPTS,
        DataSource.WORLD_BOOKS,
        DataSource.DICTIONARIES,
    ]
