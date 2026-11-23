"""Conversation-bound message helpers scope to the instance owner on PostgreSQL."""

import pytest

from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    ConflictError,
    InputError,
    OwnerScope,
    OwnerScopeRequiredError,
)
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


_PG_HELPERS = {
    "_selected_owner_filter": lambda db, conn: db._selected_owner_filter(None),
    "_selected_keyword_link_filter": lambda db, conn: db.ctx.keyword_link_filter(
        "note_keywords", "nk", owner_client_id=None
    ),
    "_require_selected_owner_row": lambda db, conn: db._require_selected_owner_row(conn, "notes", 1, None),
    "manage_link": lambda db, conn: db.manage_link(
        "note_keywords", "note_id", "n", "keyword_id", 1, "link", owner_client_id=None
    ),
}


@pytest.mark.postgres
@pytest.mark.parametrize("helper", sorted(_PG_HELPERS))
def test_owner_none_raises_on_postgres(pg_restricted_backend, tmp_path, helper):
    db = CharactersRAGDB(tmp_path / "r.db", client_id="930299", backend=pg_restricted_backend)
    try:
        with db.transaction() as conn, pytest.raises(OwnerScopeRequiredError) as raised:
            _PG_HELPERS[helper](db, conn)
        assert helper in str(raised.value)
        assert "930299" not in str(raised.value)
    finally:
        db.close_connection()


@pytest.mark.postgres
def test_not_applicable_skips_the_predicate_on_postgres(pg_restricted_backend, tmp_path):
    db = CharactersRAGDB(tmp_path / "n.db", client_id="930298", backend=pg_restricted_backend)
    try:
        assert db._selected_owner_filter(OwnerScope.NOT_APPLICABLE) == ("", ())
        assert db.ctx.keyword_link_filter(
            "note_keywords", "nk", owner_client_id=OwnerScope.NOT_APPLICABLE
        ) == ("", ())
    finally:
        db.close_connection()


@pytest.mark.postgres
def test_bypass_role_no_longer_reaches_another_owners_writing_rows(owners):
    a, b = owners
    b.add_writing_template("b-template", {"x": 1})
    session_id = b.add_writing_session("b-session", {"x": 1})
    assert a.get_writing_template_by_name("b-template") is None
    with pytest.raises(ConflictError):
        a.update_writing_session(session_id, {"name": "stolen"}, expected_version=1)
    assert b.get_writing_template_by_name("b-template")


@pytest.mark.postgres
def test_restricted_role_paths_return_only_the_owners_rows(pg_restricted_backend, tmp_path):
    a = CharactersRAGDB(tmp_path / "ra.db", client_id="930201", backend=pg_restricted_backend)
    b = CharactersRAGDB(tmp_path / "rb.db", client_id="930202", backend=pg_restricted_backend)
    try:
        b.add_writing_template("b-template", {"x": 1})
        session_id = b.add_writing_session("b-session", {"x": 1})
        moodboard_id = b.add_moodboard("b-board")
        b.add_keyword("b-only-keyword")
        assert a.get_writing_template_by_name("b-template") is None
        with pytest.raises(ConflictError):
            a.update_writing_session(session_id, {"name": "stolen"}, expected_version=1)
        with pytest.raises(ConflictError):
            a.update_moodboard(moodboard_id, {"name": "stolen"}, 1)
        assert a.search_keywords("b-only") == []
        assert [k["keyword"] for k in b.search_keywords("b-only")] == ["b-only-keyword"]
        with a.transaction() as conn, pytest.raises(OwnerScopeRequiredError):
            a._require_selected_owner_row(conn, "notes", 1, None)
    finally:
        for db in (a, b):
            db.close_connection()


@pytest.mark.postgres
def test_skill_registry_version_lookup_still_works_on_postgres(pg_restricted_backend, tmp_path):
    db = CharactersRAGDB(tmp_path / "sk.db", client_id="930203", backend=pg_restricted_backend)
    try:
        with pytest.raises(ConflictError):  # reaches the version lookup, not an owner guard
            db.update_skill_registry("missing-skill", {"description": "x"}, 1)
    finally:
        db.close_connection()


def test_sqlite_owner_none_stays_unscoped(tmp_path):
    db = CharactersRAGDB(tmp_path / "q.db", client_id="1")
    try:
        assert db._selected_owner_filter(None) == ("", ())
        with db.transaction() as conn:
            assert db._require_selected_owner_row(conn, "notes", 1, None) is None
    finally:
        db.close_connection()
