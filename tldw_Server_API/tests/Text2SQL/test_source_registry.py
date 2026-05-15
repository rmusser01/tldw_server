import pytest

from tldw_Server_API.app.core.Text2SQL.source_registry import (
    normalize_source,
    normalize_sources_internal,
    normalize_sources_public,
)


def test_normalize_source_accepts_sql_alias() -> None:
    assert normalize_source("sql") == "sql"


def test_normalize_source_normalizes_media_alias() -> None:
    assert normalize_source("media") == "media_db"


def test_normalize_source_rejects_unknown_source() -> None:
    with pytest.raises(ValueError):
        normalize_source("unknown_source")


def test_public_sources_accept_all_knowledge_qa_surfaces() -> None:
    assert normalize_sources_public(
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


def test_public_sources_reject_internal_claims_source() -> None:
    with pytest.raises(ValueError):
        normalize_sources_public(["claims"])


def test_internal_sources_allow_internal_claims_source() -> None:
    assert normalize_sources_internal(["claims"]) == ["claims"]


def test_sources_default_to_media_db() -> None:
    assert normalize_sources_public(None) == ["media_db"]
