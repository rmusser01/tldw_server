import pytest

from tldw_Server_API.app.core.RAG.rag_service.source_health import (
    CANONICAL_KNOWLEDGE_SOURCE_IDS,
    build_source_health_entries,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


pytestmark = pytest.mark.unit


def test_source_health_returns_all_canonical_knowledge_sources() -> None:
    entries = build_source_health_entries(configured_sources=[])

    assert [entry.source_id for entry in entries] == list(CANONICAL_KNOWLEDGE_SOURCE_IDS)
    assert [entry.source_id for entry in entries] == [
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    ]


def test_source_health_marks_configured_sources_searchable() -> None:
    entries = build_source_health_entries(
        configured_sources=[
            DataSource.MEDIA_DB,
            "notes",
            DataSource.CHAT_HISTORY,
        ]
    )
    by_id = {entry.source_id: entry for entry in entries}

    assert by_id["media_db"].available is True
    assert by_id["media_db"].searchable is True
    assert by_id["media_db"].index_status == "ready"
    assert by_id["notes"].available is True
    assert by_id["notes"].searchable is True
    assert by_id["chats"].available is True
    assert by_id["chats"].searchable is True

    assert by_id["characters"].available is False
    assert by_id["characters"].searchable is False
    assert by_id["characters"].index_status == "unavailable"
    assert by_id["characters"].disabled_reason == "no_retriever_configured"


def test_source_health_marks_lazy_sources_empty_without_disabling_search() -> None:
    entries = build_source_health_entries(
        configured_sources=[DataSource.MEDIA_DB],
        empty_sources=["notes", DataSource.PROMPTS],
    )
    by_id = {entry.source_id: entry for entry in entries}

    assert by_id["notes"].available is True
    assert by_id["notes"].searchable is True
    assert by_id["notes"].index_status == "empty"
    assert by_id["notes"].disabled_reason is None
    assert by_id["prompts"].available is True
    assert by_id["prompts"].searchable is True
    assert by_id["prompts"].index_status == "empty"
    assert by_id["prompts"].disabled_reason is None

    assert by_id["characters"].available is False
    assert by_id["characters"].searchable is False
    assert by_id["characters"].index_status == "unavailable"


def test_source_health_ignores_unsafe_source_metadata() -> None:
    entries = build_source_health_entries(
        configured_sources={DataSource.MEDIA_DB},
        unsafe_metadata={
            "media_db": {
                "title": "Sensitive imported title",
                "metadata": {"provider_key": "sk-secret"},
                "user_id": "42",
                "db_path": "/private/var/folders/media.db",
                "item_count": 999,
            }
        }
    )
    media = {entry.source_id: entry for entry in entries}["media_db"]

    assert media.available is True
    assert media.searchable is True
    assert media.item_count is None
    payload = media.model_dump() if hasattr(media, "model_dump") else media.dict()
    assert "title" not in payload
    assert "metadata" not in payload
    assert "user_id" not in payload
    assert "db_path" not in payload
    rendered = repr([payload])
    assert "Sensitive imported title" not in rendered
    assert "sk-secret" not in rendered
    assert "/private/var/folders/media.db" not in rendered
