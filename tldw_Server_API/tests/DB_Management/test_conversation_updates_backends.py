"""Conversation updates use portable columns and preserve existing state contracts."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, ConflictError

pytestmark = pytest.mark.integration


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def conversation_db(request, tmp_path):
    """Exercise each backend, using only the official PostgreSQL fixture."""
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    db = CharactersRAGDB(tmp_path / "updates.db", client_id="3", backend=backend)
    try:
        yield db
    finally:
        db.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


@pytest.fixture
def conversation(conversation_db):
    """Create an owned persona conversation without unrelated character setup."""
    return conversation_db.add_conversation(
        {"title": "Zircon original", "assistant_kind": "persona", "assistant_id": "fixture-persona"}
    )


def test_touch_advances_version_and_preserves_identity(conversation_db, conversation):
    before = conversation_db.get_conversation_by_id(conversation)
    assert conversation_db.update_conversation(conversation, {}, before["version"]) is True
    after = conversation_db.get_conversation_by_id(conversation)
    assert after["version"] == before["version"] + 1
    assert after["last_modified"] >= before["last_modified"]
    assert (after["title"], after["client_id"], after["assistant_id"]) == (before["title"], "3", "fixture-persona")


def test_edit_updates_content_and_title_search(conversation_db, conversation):
    before = conversation_db.get_conversation_by_id(conversation)
    assert (
        conversation_db.update_conversation(
            conversation,
            {"title": "Topaz revised", "rating": 4, "persona_memory_mode": "read_only"},
            before["version"],
        )
        is True
    )
    after = conversation_db.get_conversation_by_id(conversation)
    assert (after["title"], after["rating"], after["persona_memory_mode"]) == ("Topaz revised", 4, "read_only")
    assert [row["id"] for row in conversation_db.search_conversations_by_title("Topaz")] == [conversation]
    assert conversation_db.search_conversations_by_title("Zircon") == []


def test_stale_version_cannot_change_conversation(conversation_db, conversation):
    before = conversation_db.get_conversation_by_id(conversation)
    with pytest.raises(ConflictError, match="version mismatch"):
        conversation_db.update_conversation(conversation, {"title": "Rejected"}, before["version"] - 1)
    assert conversation_db.get_conversation_by_id(conversation) == before


def test_missing_conversation_reports_conflict(conversation_db):
    with pytest.raises(ConflictError, match="not found"):
        conversation_db.update_conversation("missing-conversation", {"title": "Rejected"}, 1)


def test_deleted_conversation_cannot_be_updated(conversation_db, conversation):
    before = conversation_db.get_conversation_by_id(conversation)
    assert conversation_db.soft_delete_conversation(conversation, before["version"]) is True
    with pytest.raises(ConflictError, match="is deleted"):
        conversation_db.update_conversation(conversation, {"title": "Rejected"}, before["version"] + 1)
    assert conversation_db.get_conversation_by_id(conversation) is None


def test_outer_rollback_restores_update_and_search(conversation_db, conversation):
    before = conversation_db.get_conversation_by_id(conversation)
    with pytest.raises(RuntimeError, match="rollback control"):
        with conversation_db.transaction():
            conversation_db.update_conversation(conversation, {"title": "Topaz temporary"}, before["version"])
            raise RuntimeError("rollback control")
    assert conversation_db.get_conversation_by_id(conversation) == before
    assert conversation_db.search_conversations_by_title("Topaz") == []
    assert [row["id"] for row in conversation_db.search_conversations_by_title("Zircon")] == [conversation]
