"""Conversation-bound message helpers scope to the instance owner on PostgreSQL."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB, InputError
from tldw_Server_API.app.core.DB_Management.db_errors import NotFoundError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

pytestmark = [pytest.mark.integration, pytest.mark.owner_isolation]


@pytest.fixture
def owners(request, tmp_path):
    backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    media = MediaDatabase(tmp_path / "media.db", client_id="930201", backend=backend)
    a = CharactersRAGDB(tmp_path / "a.db", client_id="930201", backend=backend)
    b = CharactersRAGDB(tmp_path / "b.db", client_id="930202", backend=backend)
    try:
        yield a, b
    finally:
        for db in (a, b, media):
            db.close_connection()


def _conversation(db):
    character_id = db.add_character_card({"name": "Owner Scope Character"})
    return db.add_conversation({"character_id": character_id, "title": "Owner scope"})


@pytest.mark.postgres
def test_owner_edits_a_device_labelled_message_on_postgres(owners):
    a, b = owners
    conversation_id = _conversation(a)
    message_id = a.add_message(
        {"conversation_id": conversation_id, "sender": "user", "content": "hi", "client_id": "device-x"}
    )
    with pytest.raises(NotFoundError):
        b.update_message(message_id, {"content": "stolen"}, 1)
    assert a.update_message(message_id, {"content": "mine"}, 1)
    assert a.soft_delete_message(message_id, 2)


@pytest.mark.postgres
def test_other_owner_cannot_add_a_message_to_a_conversation_on_postgres(owners):
    a, b = owners
    conversation_id = _conversation(a)
    with pytest.raises(InputError):
        b.add_message({"conversation_id": conversation_id, "sender": "user", "content": "intruder"})


@pytest.mark.postgres
def test_sync_append_and_tombstone_still_work_for_the_owner_on_postgres(owners):
    a, _ = owners
    conversation_id = _conversation(a)
    result = a.append_message_from_sync(
        stable_message_id="stable-930201",
        conversation_id=conversation_id,
        sender="user",
        content="synced",
        timestamp=None,
        sync_client_id="device-x",
        object_revision=1,
        payload_hash="h1",
    )
    assert result
    assert a.tombstone_message_from_sync(
        stable_message_id="stable-930201", sync_client_id="device-x", object_revision=2, object_hash="h1"
    )


def test_sqlite_message_edit_ignores_device_labels(tmp_path):
    db = CharactersRAGDB(tmp_path / "s.db", client_id="1")
    try:
        conversation_id = _conversation(db)
        message_id = db.add_message(
            {"conversation_id": conversation_id, "sender": "user", "content": "hi", "client_id": "device-x"}
        )
        assert db.update_message(message_id, {"content": "edited"}, 1)
    finally:
        db.close_connection()
