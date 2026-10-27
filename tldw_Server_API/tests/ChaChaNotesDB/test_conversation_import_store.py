"""Storage contract for importing a local chat in one transaction (D7 P8).

The import store writes a conversation and its whole message graph, or nothing.
It composes the ordinary conversation and message stores on one connection, so
an imported chat has the same rows, triggers and search index as a chat that
was written message by message.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat.conversation_import import (
    DEFAULT_CONVERSATION_STATE,
    import_fingerprint_from_authority,
)
from tldw_Server_API.app.core.DB_Management.backends.base import UniqueConstraintError
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.integration

CHAT_ID = "3f1c9a52-6a2b-4c1e-9d7f-0b8e2a4c6d10"
FINGERPRINT = "a" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
TABLES = ("conversations", "messages", "message_metadata", "message_images")


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.postgres)])
def db(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[CharactersRAGDB]:
    backend = None
    if request.param == "postgres":
        backend = DatabaseBackendFactory.create_backend(request.getfixturevalue("pg_database_config"))
    database = CharactersRAGDB(tmp_path / "import.db", client_id="1", backend=backend)
    try:
        yield database
    finally:
        database.close_all_connections()
        if backend is not None:
            backend.get_pool().close_all()


def _instant(value: Any) -> datetime:
    """A stored timestamp as a point in time: SQLite keeps text, PostgreSQL returns datetimes."""
    moment = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def _conversation(**extra: Any) -> dict[str, Any]:
    return {
        "id": CHAT_ID,
        "title": "Trip planning",
        "state": "resolved",
        "created_at": "2026-09-01T10:00:00.000Z",
        "last_modified": "2026-09-01T10:04:00.000Z",
        **extra,
    }


def _message(mid: str, parent: str | None, sender: str = "user", minute: int = 0, **extra: Any) -> dict[str, Any]:
    return {
        "id": mid,
        "parent_message_id": parent,
        "sender": sender,
        "content": f"text of {mid}",
        "timestamp": f"2026-09-01T10:{minute:02d}:00.000Z",
        "images": [],
        "extra_metadata": {"sender_role": sender},
        **extra,
    }


def _branching() -> list[dict[str, Any]]:
    """One question, two answers, and a follow-up under the second answer."""
    return [
        _message("m1", None, "user", 0, images=[{"data": PNG, "mime": "image/png"}]),
        _message("m2", "m1", "assistant", 1, extra_metadata={"sender_role": "assistant", "model_id": "gpt-4o"}),
        _message("m3", "m1", "assistant", 2),
        _message("m4", "m3", "user", 3),
    ]


def _counts(db: CharactersRAGDB) -> dict[str, int]:
    return {
        table: int(db.execute_query(f"SELECT COUNT(*) AS n FROM {table}", read_only=True).fetchone()["n"])  # nosec B608
        for table in TABLES
    }


def _import(db: CharactersRAGDB, messages: list[dict[str, Any]] | None = None, **kwargs: Any) -> str:
    return db.chat_imports.import_conversation(
        kwargs.pop("conversation", _conversation()),
        _branching() if messages is None else messages,
        owner_client_id=kwargs.pop("owner_client_id", "1"),
        request_fingerprint=kwargs.pop("request_fingerprint", FINGERPRINT),
    )


def test_import_writes_the_conversation_with_its_own_dates(db: CharactersRAGDB) -> None:
    assert _import(db) == CHAT_ID
    row = db.get_conversation_by_id(CHAT_ID)
    assert (row["title"], row["state"], row["client_id"], row["version"]) == ("Trip planning", "resolved", "1", 1)
    assert _instant(row["created_at"]) == _instant("2026-09-01T10:00:00.000Z")
    assert _instant(row["last_modified"]) == _instant("2026-09-01T10:04:00.000Z")
    assert (row["root_id"], row["scope_type"], row["workspace_id"]) == (CHAT_ID, "global", None)
    assert not row["deleted"]


def test_import_keeps_ids_parents_senders_timestamps_and_versions(db: CharactersRAGDB) -> None:
    _import(db)
    rows = db.get_messages_for_conversation(CHAT_ID, limit=50)
    assert [(row["id"], row["parent_message_id"], row["sender"], _instant(row["timestamp"])) for row in rows] == [
        ("m1", None, "user", _instant("2026-09-01T10:00:00.000Z")),
        ("m2", "m1", "assistant", _instant("2026-09-01T10:01:00.000Z")),
        ("m3", "m1", "assistant", _instant("2026-09-01T10:02:00.000Z")),
        ("m4", "m3", "user", _instant("2026-09-01T10:03:00.000Z")),
    ]
    assert {row["version"] for row in rows} == {1}
    assert {row["client_id"] for row in rows} == {"1"}
    assert [row["content"] for row in rows] == [f"text of m{index}" for index in range(1, 5)]


def test_import_stores_images_and_metadata(db: CharactersRAGDB) -> None:
    _import(db)
    first = db.get_message_by_id("m1")
    assert (bytes(first["image_data"]), first["image_mime_type"]) == (PNG, "image/png")
    assert [(image["position"], bytes(image["image_data"])) for image in db.get_message_images("m1")] == [(0, PNG)]
    assert db.get_message_metadata("m2")["extra"] == {"sender_role": "assistant", "model_id": "gpt-4o"}
    assert db.get_message_metadata("m1")["extra"] == {"sender_role": "user"}


def test_equal_timestamps_keep_the_order_the_messages_were_given_in(db: CharactersRAGDB) -> None:
    messages = [_message(mid, None, minute=7) for mid in ("zeta", "alpha", "mid")]
    _import(db, messages)
    assert [row["id"] for row in db.get_messages_for_conversation(CHAT_ID, limit=50)] == ["zeta", "alpha", "mid"]


def test_imported_branches_read_as_a_parent_graph_not_as_legacy_history(db: CharactersRAGDB) -> None:
    _import(db)
    snapshot = db.get_conversation_history_snapshot(CHAT_ID, owner_client_id="1")
    assert snapshot.interpretation_status == {"kind": "parent_graph_v1"}
    assert [(node["id"], node["parent_id"], node["settled"]) for node in snapshot.nodes] == [
        ("m1", None, True), ("m2", "m1", True), ("m3", "m1", True), ("m4", "m3", True),
    ]


def test_import_advances_the_history_version_once_per_message(db: CharactersRAGDB) -> None:
    _import(db)
    assert db.get_roleplay_resume_state(CHAT_ID)["history_version"] == 5
    assert db.get_roleplay_resume_state(CHAT_ID)["message_count"] == 4


def test_imported_text_is_searchable(db: CharactersRAGDB) -> None:
    _import(db, [_message("m1", None, content="the heliotrope itinerary")])
    found = db.search_messages_by_content("heliotrope", conversation_id=CHAT_ID)
    assert [row["id"] for row in found] == ["m1"]


def test_import_state_returns_the_fingerprint_stored_with_the_messages(db: CharactersRAGDB) -> None:
    assert db.chat_imports.get_import_state(CHAT_ID) == (None, None)
    _import(db)
    row, fingerprint = db.chat_imports.get_import_state(CHAT_ID)
    assert (row["id"], row["client_id"], fingerprint) == (CHAT_ID, "1", FINGERPRINT)
    stored = db.execute_query(
        "SELECT history_admission_json FROM messages WHERE conversation_id = ?", (CHAT_ID,), read_only=True
    ).fetchall()
    assert {import_fingerprint_from_authority(item["history_admission_json"]) for item in stored} == {FINGERPRINT}


def test_import_state_survives_later_edits_and_trash(db: CharactersRAGDB) -> None:
    _import(db)
    db.update_message("m2", {"content": "edited later"}, expected_version=1)
    db.add_message({"id": "later", "conversation_id": CHAT_ID, "sender": "user", "content": "new", "parent_message_id": "m4"})
    assert db.chat_imports.get_import_state(CHAT_ID)[1] == FINGERPRINT
    db.soft_delete_conversation(CHAT_ID, expected_version=db.get_conversation_by_id(CHAT_ID)["version"])
    row, fingerprint = db.chat_imports.get_import_state(CHAT_ID)
    assert (bool(row["deleted"]), fingerprint) == (True, FINGERPRINT)


def test_chat_that_was_not_imported_has_no_import_fingerprint(db: CharactersRAGDB) -> None:
    cid = db.add_conversation({"title": "Ordinary", "client_id": "1"})
    # A message whose text merely looks like the marker is not provenance.
    db.add_message({"conversation_id": cid, "sender": "user", "content": '"import":{"request_fingerprint":"' + FINGERPRINT + '"}'})
    row, fingerprint = db.chat_imports.get_import_state(cid)
    assert (row["id"], fingerprint) == (cid, None)


def test_taken_conversation_id_is_a_conflict_and_changes_nothing(db: CharactersRAGDB) -> None:
    _import(db)
    before = _counts(db)
    with pytest.raises(ConflictError) as raised:
        _import(db, [_message("other", None)], conversation=_conversation(title="Second"), request_fingerprint="b" * 64)
    assert (raised.value.entity, raised.value.entity_id) == ("conversations", CHAT_ID)
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID)["title"] == "Trip planning"
    assert db.chat_imports.get_import_state(CHAT_ID)[1] == FINGERPRINT


def test_taken_message_id_is_a_conflict_and_rolls_the_whole_import_back(db: CharactersRAGDB) -> None:
    other = db.add_conversation({"title": "Existing", "client_id": "1"})
    db.add_message({"id": "m3", "conversation_id": other, "sender": "user", "content": "already here"})
    before = _counts(db)
    with pytest.raises(ConflictError) as raised:
        _import(db)
    assert (raised.value.entity, raised.value.entity_id) == ("messages", "m3")
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    assert db.get_message_by_id("m1") is None
    assert db.get_message_by_id("m3")["conversation_id"] == other
    assert db.search_messages_by_content("text of m1") == []


def test_conversation_id_held_by_a_row_the_caller_cannot_see_is_a_conflict(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Under PostgreSQL row-level security another owner's chat is invisible; only the primary key refuses."""

    def hidden_row_holds_the_id(*_args: Any, **_kwargs: Any) -> str:
        raise UniqueConstraintError("PostgreSQL query execution failed")

    monkeypatch.setattr(db.conversation_store, "add_conversation", hidden_row_holds_the_id)
    before = _counts(db)
    with pytest.raises(ConflictError) as raised:
        _import(db)
    assert (raised.value.entity, raised.value.entity_id) == ("conversations", CHAT_ID)
    assert _counts(db) == before


def test_message_id_held_by_a_row_the_caller_cannot_see_is_a_conflict(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PostgreSQL reports the taken key without detail, wrapped by the message store."""
    real = db.message_store.add_message

    def hidden_row_holds_m3(msg_data: dict[str, Any], **kwargs: Any) -> str:
        if msg_data["id"] == "m3":
            try:
                raise UniqueConstraintError("PostgreSQL query execution failed")
            except UniqueConstraintError as error:
                raise CharactersRAGDBError("Database error adding message") from error
        return real(msg_data, **kwargs)

    monkeypatch.setattr(db.message_store, "add_message", hidden_row_holds_m3)
    before = _counts(db)
    with pytest.raises(ConflictError) as raised:
        _import(db)
    assert (raised.value.entity, raised.value.entity_id) == ("messages", "m3")
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None


def test_other_database_errors_are_not_reported_as_id_conflicts(
    db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*_args: Any, **_kwargs: Any) -> str:
        raise CharactersRAGDBError("Database error adding message: connection lost")

    monkeypatch.setattr(db.message_store, "add_message", broken)
    before = _counts(db)
    with pytest.raises(CharactersRAGDBError) as raised:
        _import(db)
    assert not isinstance(raised.value, ConflictError)
    assert _counts(db) == before


def test_default_state_matches_the_store(db: CharactersRAGDB) -> None:
    """An omitted state is fingerprinted as the default, so the two must not drift apart."""
    assert DEFAULT_CONVERSATION_STATE == db._DEFAULT_CONVERSATION_STATE
    _import(db, conversation=_conversation(state=None))
    assert db.get_conversation_by_id(CHAT_ID)["state"] == DEFAULT_CONVERSATION_STATE


def test_failure_part_way_through_leaves_nothing_behind(db: CharactersRAGDB, monkeypatch: pytest.MonkeyPatch) -> None:
    real = db.message_store._add_message_metadata_with_conn
    calls = {"n": 0}

    def fail_on_third(*args: Any, **kwargs: Any) -> bool:
        calls["n"] += 1
        if calls["n"] == 3:
            raise CharactersRAGDBError("disk full")
        return real(*args, **kwargs)

    monkeypatch.setattr(db.message_store, "_add_message_metadata_with_conn", fail_on_third)
    before = _counts(db)
    with pytest.raises(CharactersRAGDBError, match="disk full"):
        _import(db)
    assert _counts(db) == before
    assert db.get_conversation_by_id(CHAT_ID, include_deleted=True) is None
    assert db.search_messages_by_content("text of m1") == []
    assert db.search_conversations_by_title("Trip planning") == []
    monkeypatch.undo()
    assert _import(db) == CHAT_ID
    assert _counts(db)["messages"] == 4


def test_import_refuses_to_join_a_transaction_it_does_not_own(db: CharactersRAGDB) -> None:
    """Joined to an open transaction, the import would be committed or rolled back by someone else."""
    before = _counts(db)
    with db.transaction():
        with pytest.raises(CharactersRAGDBError, match="own transaction"):
            _import(db)
    assert _counts(db) == before
    assert _import(db) == CHAT_ID


def test_child_listed_before_its_parent_is_refused_before_any_write(db: CharactersRAGDB) -> None:
    before = _counts(db)
    with pytest.raises(InputError):
        _import(db, [_message("child", "parent"), _message("parent", None)])
    assert _counts(db) == before


@pytest.mark.parametrize("fingerprint", ["", "A" * 64, "a" * 63, "g" * 64, None])
def test_malformed_fingerprint_is_refused_before_any_write(db: CharactersRAGDB, fingerprint: Any) -> None:
    before = _counts(db)
    with pytest.raises(InputError):
        _import(db, request_fingerprint=fingerprint)
    assert _counts(db) == before


def test_owner_is_the_argument_never_a_field_of_the_payload(db: CharactersRAGDB) -> None:
    conversation = _conversation(client_id="999", scope_type="workspace", workspace_id="ws", deleted=1, version=7)
    messages = [_message("m1", None, client_id="999", deleted=1, version=9, conversation_id="someone-elses")]
    _import(db, messages, conversation=conversation, owner_client_id="42")
    row = db.get_conversation_by_id(CHAT_ID)
    assert (row["client_id"], row["scope_type"], row["workspace_id"], row["version"]) == ("42", "global", None, 1)
    message = db.get_message_by_id("m1")
    assert (message["client_id"], message["conversation_id"], message["version"]) == ("42", CHAT_ID, 1)
    assert not message["deleted"]


def test_fork_lineage_is_stored(db: CharactersRAGDB) -> None:
    parent = db.add_conversation({"title": "Parent", "client_id": "1"})
    source = db.add_message({"conversation_id": parent, "sender": "user", "content": "fork here"})
    _import(db, conversation=_conversation(root_id=parent, parent_conversation_id=parent, forked_from_message_id=source))
    row = db.get_conversation_by_id(CHAT_ID)
    assert (row["root_id"], row["parent_conversation_id"], row["forked_from_message_id"]) == (parent, parent, source)


def test_authority_is_written_compactly_so_the_lookup_can_find_it(db: CharactersRAGDB) -> None:
    _import(db, [_message("m1", None)])
    raw = db.execute_query(
        "SELECT history_admission_json FROM messages WHERE id = ?", ("m1",), read_only=True
    ).fetchone()["history_admission_json"]
    assert '"import":{' in raw
    assert json.loads(raw)["import"] == {"request_fingerprint": FINGERPRINT, "version": 1}
