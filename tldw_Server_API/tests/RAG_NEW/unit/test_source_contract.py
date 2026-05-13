import pytest

from tldw_Server_API.app.core.RAG.rag_service.unified_pipeline import (
    _normalize_pipeline_sources,
    _sources_to_data_sources,
)
from tldw_Server_API.app.core.RAG.rag_service.types import DataSource


pytestmark = pytest.mark.unit


def test_pipeline_normalizes_all_public_knowledge_source_aliases() -> None:
    assert _normalize_pipeline_sources(
        [
            "media",
            "notes_db",
            "chat_history",
            "character_cards",
            "task_boards",
            "prompts_db",
            "worldbooks",
            "chat_dictionaries",
        ]
    ) == [
        "media_db",
        "notes",
        "chats",
        "characters",
        "kanban",
        "prompts",
        "world_books",
        "dictionaries",
    ]


def test_pipeline_maps_chats_and_characters_to_distinct_data_sources() -> None:
    sources = _normalize_pipeline_sources(["chats", "characters"])

    assert _sources_to_data_sources(sources) == [
        DataSource.CHAT_HISTORY,
        DataSource.CHARACTER_CARDS,
    ]


def test_pipeline_maps_world_books_and_dictionaries_to_distinct_data_sources() -> None:
    sources = _normalize_pipeline_sources(["world_books", "dictionaries"])

    assert _sources_to_data_sources(sources) == [
        DataSource.WORLD_BOOKS,
        DataSource.DICTIONARIES,
    ]
