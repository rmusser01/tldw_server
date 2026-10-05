"""Deleting a chat row Sync v2 never saw, with an active profile (#3181).

A row is *enrolled* when the owner's dataset holds Sync history for it. Rows that
predate the profile, rows written by a path that publishes nothing, and rows whose
capture was lost have none. Deleting such a row used to append a tombstone the
chat materializer rejected (``message_base_conflict`` / ``missing_server_message``),
which left an unresolved projection conflict and made every later Sync-routed
write for the user fail with 503.

The rule under test:

* a row with no Sync history is deleted directly, exactly as without a profile,
  and nothing is appended to the log;
* a row with Sync history is still tombstoned through the log and reaches the
  other devices;
* a dataset already blocked by the old behaviour recovers on the next
  server-origin write.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import server_origin
from tldw_Server_API.app.core.Sync.v2.errors import SyncMaterializationPredecessorError, SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncConflictCreate,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    SERVER_ORIGIN_DEVICE_ID,
    SyncServerOriginMutationNotSupportedError,
    canonical_payload_hash,
    capture_server_origin_mutation,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import (
    CHAT_ID,
    INPUT_ID,
    REPLY_ID,
    USER,
    _admit,
    _block_dataset,
    _client,
    _create_chat,
    _dataset_id,
    _log,
    _pull,
    _selection,
    _send_turn,
)
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import chacha_db as chacha_db
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import client as client
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import projection_db as projection_db
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import sync_service as sync_service

pytestmark = pytest.mark.integration

LEGACY_CHAT = "legacy-chat"
LEGACY_QUESTION = "legacy-question"
LEGACY_ANSWER = "legacy-answer"


def _legacy_chat(db: CharactersRAGDB, chat_id: str = LEGACY_CHAT, messages: tuple[str, ...] = ()) -> None:
    """Write a chat the way a path that publishes nothing does: owner rows, no envelope, no object state."""
    db.add_conversation({"id": chat_id, "character_id": None, "title": "Before the profile", "client_id": USER})
    for message_id in messages:
        _legacy_message(db, message_id, chat_id)


def _legacy_message(db: CharactersRAGDB, message_id: str, chat_id: str, sender: str = "user") -> None:
    db.add_message({"id": message_id, "conversation_id": chat_id, "sender": sender, "content": f"{message_id} text"})


def _is_deleted(db: CharactersRAGDB, message_id: str) -> bool:
    return bool(db.get_message_by_id(message_id, include_deleted=True)["deleted"])


def _delete_message(client: TestClient, db: CharactersRAGDB, message_id: str):
    version = db.get_message_by_id(message_id)["version"]
    return client.delete(f"/api/v1/messages/{message_id}", params={"expected_version": version})


def _delete_chat(client: TestClient, db: CharactersRAGDB, chat_id: str):
    version = db.get_conversation_by_id(chat_id)["version"]
    return client.delete(f"/api/v1/chats/{chat_id}", params={"expected_version": version})


def _blocker(service: SyncV2Service):
    return service.store.get_unresolved_materialization_conflict(_dataset_id(service))


def _has_sync_history(service: SyncV2Service, domain: str, object_id: str) -> bool:
    dataset_id = _dataset_id(service)
    return (
        service.store.get_current_head(dataset_id, domain, object_id) is not None
        or service.store.get_object_state(dataset_id, domain, object_id) is not None
    )


def _apply_to_replica(replica: CharactersRAGDB, envelopes: list[SyncEnvelope]) -> None:
    """Project pulled chat envelopes, tombstones included, the way another device would."""
    for envelope in envelopes:
        payload = envelope.payload
        key = (envelope.domain, envelope.operation)
        revision = envelope.object_revision or 1
        if key == ("chat.conversation", "upsert"):
            replica.upsert_conversation_from_sync(
                conversation_id=envelope.object_id,
                title=payload.get("title"),
                sync_client_id=str(replica.client_id),
                object_revision=revision,
                object_hash=envelope.payload_hash or "",
                state=payload.get("state"),
                source=payload.get("source"),
            )
        elif key == ("chat.message", "append"):
            replica.append_message_from_sync(
                stable_message_id=envelope.object_id,
                conversation_id=payload["conversation_id"],
                sender=payload["sender"],
                content=payload.get("content"),
                timestamp=payload.get("timestamp"),
                sync_client_id=str(replica.client_id),
                object_revision=revision,
                payload_hash=envelope.payload_hash or "",
                parent_message_id=payload.get("parent_message_id"),
            )
        elif key == ("chat.message", "tombstone"):
            replica.tombstone_message_from_sync(
                stable_message_id=envelope.object_id,
                sync_client_id=str(replica.client_id),
                object_revision=revision,
                object_hash=envelope.base_object_hash or "",
            )
        elif key == ("chat.conversation", "tombstone"):
            replica.tombstone_conversation_from_sync(
                conversation_id=envelope.object_id,
                sync_client_id=str(replica.client_id),
                object_revision=revision,
                object_hash=envelope.payload_hash or "",
            )
        else:
            raise AssertionError(f"unexpected envelope {envelope.domain}/{envelope.operation}")


def _leave_the_old_bug_behind(service: SyncV2Service, message_id: str, chat_id: str) -> SyncEnvelope:
    """Leave what the delete routes left before the fix: the blocked dataset of #3181.

    That is the server's own tombstone of a row Sync never saw, accepted, with
    ``apply_status = conflict`` and an unresolved projection conflict. It is
    stored directly, because projecting such a tombstone no longer rejects it.
    """
    dataset_id = _dataset_id(service)
    payload: dict[str, object] = {
        "id": message_id,
        "deleted": True,
        "conversation_id": chat_id,
        "client_id": USER,
        "owner_user_id": USER,
    }
    payload_hash, payload_size = canonical_payload_hash(payload)
    stranded = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id=f"server-origin-stranded-{message_id}",
            domain="chat.message",
            operation="tombstone",
            object_id=message_id,
            device_id=SERVER_ORIGIN_DEVICE_ID,
            client_sequence=None,
            object_revision=1,
            parent_id=chat_id,
            payload=payload,
            payload_hash=payload_hash,
            payload_size_bytes=payload_size,
            deleted=True,
            encryption_metadata={"policy": "server_trusted_v1"},
            routing_metadata={"source": "server_api", "origin": "server", "server_device_id": SERVER_ORIGIN_DEVICE_ID},
            status="accepted",
        )
    )
    stranded = service.store.mark_envelope_apply_status(
        stranded.server_cursor,
        apply_status="conflict",
        apply_error_code="message_base_conflict",
        apply_error_message="chat.message tombstone requires an existing server message base state",
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id=f"conflict-stranded-{message_id}",
            dataset_id=dataset_id,
            domain="chat.message",
            entity_id=message_id,
            conflict_type="message_base_conflict",
            local_envelope_id=stranded.client_envelope_id,
            server_sequence=stranded.server_cursor,
            metadata={"reason": "missing_server_message"},
        )
    )
    blocker = _blocker(service)
    assert blocker is not None and blocker.metadata["reason"] == "missing_server_message"
    return stranded


# ---------------------------------------------------------------------------
# A row Sync never saw is deleted directly
# ---------------------------------------------------------------------------


def test_deleting_a_message_sync_never_saw_succeeds_and_appends_nothing(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))

    deleted = _delete_message(client, chacha_db, LEGACY_QUESTION)

    assert deleted.status_code == 204, deleted.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert not _is_deleted(chacha_db, LEGACY_ANSWER)
    assert _log(sync_service) == []
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    assert not _has_sync_history(sync_service, "chat.message", LEGACY_QUESTION)
    assert _pull(sync_service) == []


def test_the_dataset_still_takes_writes_after_deleting_a_message_sync_never_saw(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The dataset-blocking half of #3181: every later Sync-routed write must still succeed."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    assert _delete_message(client, chacha_db, LEGACY_QUESTION).status_code == 204

    created = client.post("/api/v1/chats/", json={"title": "After the delete"})
    assert created.status_code == 201, created.text
    sent = client.post(f"/api/v1/chats/{created.json()['id']}/messages", json={"role": "user", "content": "still works"})
    assert sent.status_code == 201, sent.text
    reference = _send_turn(client)

    assert reference["input_message_id"] == INPUT_ID
    assert _blocker(sync_service) is None
    assert {item[3] for item in _log(sync_service)} == {"applied"}


def test_deleting_a_chat_sync_never_saw_succeeds_and_appends_nothing(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Neither the messages nor the chat reach the log: no device ever held them."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))

    deleted = _delete_chat(client, chacha_db, LEGACY_CHAT)

    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT, include_deleted=True)["deleted"] in (1, True)
    assert _is_deleted(chacha_db, LEGACY_QUESTION) and _is_deleted(chacha_db, LEGACY_ANSWER)
    assert _log(sync_service) == []
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    assert not _has_sync_history(sync_service, "chat.conversation", LEGACY_CHAT)
    # The dataset is still healthy.
    assert client.post("/api/v1/chats/", json={"title": "After the delete"}).status_code == 201


def test_deleting_an_empty_chat_sync_never_saw_appends_nothing(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _legacy_chat(chacha_db)

    deleted = _delete_chat(client, chacha_db, LEGACY_CHAT)

    assert deleted.status_code == 204, deleted.text
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert _log(sync_service) == []


def test_a_message_whose_capture_was_lost_can_be_deleted(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The window between the owner commit and the Sync commit leaves a row with no Sync history."""
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    original_insert = SyncDatabase.insert_envelope

    def failing_insert(self, envelope, **kwargs):
        raise SyncStoreError("append is unavailable")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", failing_insert)
    assert _admit(client, CHAT_ID, selection).status_code == 503
    monkeypatch.setattr(SyncDatabase, "insert_envelope", original_insert)
    assert chacha_db.get_message_by_id(INPUT_ID) is not None
    before = _log(sync_service)

    deleted = _delete_message(client, chacha_db, INPUT_ID)

    assert deleted.status_code == 204, deleted.text
    assert _is_deleted(chacha_db, INPUT_ID)
    assert _log(sync_service) == before
    assert _blocker(sync_service) is None


# ---------------------------------------------------------------------------
# A chat with both kinds of row
# ---------------------------------------------------------------------------


def test_deleting_a_published_chat_with_rows_sync_never_saw_tombstones_only_the_published_rows(
    tmp_path: Path,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The chat and one turn are in the log; a later unpublished save is not."""
    _send_turn(client)
    _legacy_message(chacha_db, "unpublished-save", CHAT_ID, sender="assistant")
    replica = CharactersRAGDB(db_path=str(tmp_path / "replica" / "ChaChaNotes.db"), client_id="offline-device")
    try:
        _apply_to_replica(replica, _pull(sync_service))
        assert [row["id"] for row in replica.get_messages_for_conversation(CHAT_ID)] == [INPUT_ID, REPLY_ID]

        deleted = _delete_chat(client, chacha_db, CHAT_ID)

        assert deleted.status_code == 204, deleted.text
        assert _log(sync_service)[3:] == [
            ("chat.message", "tombstone", INPUT_ID, "applied"),
            ("chat.message", "tombstone", REPLY_ID, "applied"),
            ("chat.conversation", "tombstone", CHAT_ID, "applied"),
        ]
        assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
        for message_id in (INPUT_ID, REPLY_ID, "unpublished-save"):
            assert _is_deleted(chacha_db, message_id)
        assert not _has_sync_history(sync_service, "chat.message", "unpublished-save")
        assert chacha_db.get_conversation_by_id(CHAT_ID) is None

        # The other device receives exactly the deletes of what it holds.
        _apply_to_replica(replica, _pull(sync_service)[3:])
        assert replica.get_conversation_by_id(CHAT_ID) is None
        assert replica.get_messages_for_conversation(CHAT_ID) == []
    finally:
        replica.close_all_connections()
    assert client.post("/api/v1/chats/", json={"title": "After the delete"}).status_code == 201


def test_deleting_a_chat_sync_never_saw_with_a_published_message_tombstones_only_that_message(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A chat that predates the profile, continued under it: only the new message is in the log."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    sent = client.post(f"/api/v1/chats/{LEGACY_CHAT}/messages", json={"role": "user", "content": "sent under the profile"})
    assert sent.status_code == 201, sent.text
    published_id = sent.json()["id"]
    assert _log(sync_service) == [("chat.message", "append", published_id, "applied")]

    deleted = _delete_chat(client, chacha_db, LEGACY_CHAT)

    assert deleted.status_code == 204, deleted.text
    assert _log(sync_service) == [
        ("chat.message", "append", published_id, "applied"),
        ("chat.message", "tombstone", published_id, "applied"),
    ]
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    assert _is_deleted(chacha_db, LEGACY_QUESTION) and _is_deleted(chacha_db, published_id)
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert client.post("/api/v1/chats/", json={"title": "After the delete"}).status_code == 201


def test_deleting_one_unpublished_message_leaves_its_published_neighbours_alone(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)
    _legacy_message(chacha_db, "unpublished-save", CHAT_ID, sender="assistant")
    before = _log(sync_service)

    deleted = _delete_message(client, chacha_db, "unpublished-save")

    assert deleted.status_code == 204, deleted.text
    assert _is_deleted(chacha_db, "unpublished-save")
    assert not _is_deleted(chacha_db, INPUT_ID) and not _is_deleted(chacha_db, REPLY_ID)
    assert _log(sync_service) == before
    # The chat is enrolled, so its row is not touched outside the log.
    state = sync_service.store.get_object_state(_dataset_id(sync_service), "chat.conversation", CHAT_ID)
    assert chacha_db.get_conversation_by_id(CHAT_ID)["version"] == state.object_revision


# ---------------------------------------------------------------------------
# A published row is still tombstoned through the log
# ---------------------------------------------------------------------------


def test_deleting_a_published_message_still_tombstones_it_and_reaches_a_second_device(
    tmp_path: Path,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)
    replica = CharactersRAGDB(db_path=str(tmp_path / "replica" / "ChaChaNotes.db"), client_id="offline-device")
    try:
        _apply_to_replica(replica, _pull(sync_service))

        deleted = _delete_message(client, chacha_db, REPLY_ID)

        assert deleted.status_code == 204, deleted.text
        assert _log(sync_service)[3:] == [("chat.message", "tombstone", REPLY_ID, "applied")]
        tombstone = _pull(sync_service)[-1]
        assert (tombstone.operation, tombstone.object_id, tombstone.device_id) == (
            "tombstone", REPLY_ID, SERVER_ORIGIN_DEVICE_ID,
        )
        assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", REPLY_ID).deleted
        assert _is_deleted(chacha_db, REPLY_ID)

        _apply_to_replica(replica, [tombstone])
        assert [row["id"] for row in replica.get_messages_for_conversation(CHAT_ID)] == [INPUT_ID]
    finally:
        replica.close_all_connections()


def test_a_message_whose_envelope_is_not_applied_yet_is_not_deleted_directly(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A device may already hold a row whose append is accepted but not projected: it is not "never seen"."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    dataset_id = _dataset_id(sync_service)
    sync_service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-pending-append",
            domain="chat.message",
            operation="append",
            object_id=LEGACY_QUESTION,
            device_id="offline-device",
            client_sequence=1,
            object_revision=1,
            parent_id=LEGACY_CHAT,
            payload={"conversation_id": LEGACY_CHAT, "sender": "user", "content": "pending"},
            payload_hash="sha256:pending-append",
            encryption_metadata={"policy": "server_trusted_v1"},
            status="accepted",
        )
    )
    before = _log(sync_service)

    refused = _delete_message(client, chacha_db, LEGACY_QUESTION)

    assert refused.status_code == 503, refused.text
    assert not _is_deleted(chacha_db, LEGACY_QUESTION)
    assert _log(sync_service) == before
    assert _blocker(sync_service) is None


def test_a_row_sync_never_saw_can_be_deleted_while_another_conflict_blocks_the_dataset(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A direct delete appends nothing, so an unrelated unresolved conflict does not stop it."""
    _send_turn(client)
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _block_dataset(sync_service)
    before = _log(sync_service)

    direct = _delete_message(client, chacha_db, LEGACY_QUESTION)
    published = _delete_message(client, chacha_db, REPLY_ID)

    assert direct.status_code == 204, direct.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    # A published row needs a tombstone, which the blocked dataset cannot take.
    assert published.status_code == 503, published.text
    assert published.json()["detail"]["error_code"] == "sync_server_origin_append_failed"
    assert not _is_deleted(chacha_db, REPLY_ID)
    assert _log(sync_service) == before
    assert _blocker(sync_service).conflict_id == "conflict-unresolved"


def test_a_blocked_dataset_refuses_a_chat_delete_that_needs_tombstones_before_deleting_anything(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The unpublished rows are not deleted on their own when the published ones cannot be."""
    _send_turn(client)
    _legacy_message(chacha_db, "unpublished-save", CHAT_ID, sender="assistant")
    _block_dataset(sync_service)
    before = _log(sync_service)

    refused = _delete_chat(client, chacha_db, CHAT_ID)

    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"]["error_code"] == "sync_server_origin_append_failed"
    for message_id in (INPUT_ID, REPLY_ID, "unpublished-save"):
        assert not _is_deleted(chacha_db, message_id)
    assert chacha_db.get_conversation_by_id(CHAT_ID) is not None
    assert _log(sync_service) == before


def test_a_stale_message_version_deletes_none_of_a_chat_sync_never_saw(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The direct message deletes of one chat delete are one transaction."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))
    original = chacha_db.get_messages_for_conversation

    def stale_second_message(*args, **kwargs):
        rows = original(*args, **kwargs)
        for row in rows:
            if row["id"] == LEGACY_ANSWER:
                row["version"] = 7
        return rows

    monkeypatch.setattr(chacha_db, "get_messages_for_conversation", stale_second_message)

    refused = _delete_chat(client, chacha_db, LEGACY_CHAT)

    assert refused.status_code == 409, refused.text
    assert not _is_deleted(chacha_db, LEGACY_QUESTION) and not _is_deleted(chacha_db, LEGACY_ANSWER)
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is not None
    assert _log(sync_service) == []
    assert _blocker(sync_service) is None


def test_version_mismatch_still_refuses_before_any_delete(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))

    stale_message = client.delete(f"/api/v1/messages/{LEGACY_QUESTION}", params={"expected_version": 7})
    stale_chat = client.delete(f"/api/v1/chats/{LEGACY_CHAT}", params={"expected_version": 7})

    assert stale_message.status_code == 409, stale_message.text
    assert stale_chat.status_code == 409, stale_chat.text
    assert not _is_deleted(chacha_db, LEGACY_QUESTION)
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is not None
    assert _log(sync_service) == []


# ---------------------------------------------------------------------------
# After a direct delete
# ---------------------------------------------------------------------------


def test_retrying_the_lost_admission_after_its_row_was_deleted_is_refused_without_blocking(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The retry that would have published the row finds it deleted: the owner refuses, nothing is recorded."""
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    original_insert = SyncDatabase.insert_envelope

    def failing_insert(self, envelope, **kwargs):
        raise SyncStoreError("append is unavailable")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", failing_insert)
    assert _admit(client, CHAT_ID, selection).status_code == 503
    monkeypatch.setattr(SyncDatabase, "insert_envelope", original_insert)
    assert _delete_message(client, chacha_db, INPUT_ID).status_code == 204
    before = _log(sync_service)

    retried = _admit(client, CHAT_ID, selection)

    assert retried.status_code == 409, retried.text
    assert retried.json()["detail"]["status"] == "stale_selection"
    assert _log(sync_service) == before
    assert _blocker(sync_service) is None
    assert client.post("/api/v1/chats/", json={"title": "After the retry"}).status_code == 201


def test_a_device_that_publishes_its_own_copy_of_a_directly_deleted_row_does_not_block_the_dataset(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Known limit of a direct delete, pinned so that a change to it is deliberate.

    A device can hold a row under the same id without Sync (the append
    materializer adopts a matching row that has no Sync metadata). The server's
    delete of such a row is not in the log, so the device's later append
    enrolls the message for the other devices while the server row stays
    deleted. The same happens to a row deleted before the profile existed.
    Nothing conflicts and the dataset stays healthy. See §5.2 of
    Docs/Design/2026-10-04-d7-sync-v2-native-history.md.
    """
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    assert _delete_message(client, chacha_db, LEGACY_QUESTION).status_code == 204
    dataset_id = _dataset_id(sync_service)

    pushed = sync_service.push(
        user_id=USER,
        dataset_id=dataset_id,
        device_id="offline-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-device-own-copy",
                domain="chat.message",
                operation="append",
                object_id=LEGACY_QUESTION,
                device_id="offline-device",
                client_sequence=1,
                object_revision=1,
                parent_id=LEGACY_CHAT,
                payload={"conversation_id": LEGACY_CHAT, "sender": "user", "content": f"{LEGACY_QUESTION} text"},
                payload_hash="sha256:device-own-copy",
                encryption_metadata={"policy": "server_trusted_v1"},
            )
        ],
    )

    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert pushed.conflicts == [] and pushed.rejected == []
    assert _blocker(sync_service) is None
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert sync_service.store.get_object_state(dataset_id, "chat.message", LEGACY_QUESTION).deleted is False
    assert client.post("/api/v1/chats/", json={"title": "After the push"}).status_code == 201


# ---------------------------------------------------------------------------
# A dataset already blocked by the old behaviour
# ---------------------------------------------------------------------------


def test_a_dataset_blocked_by_the_old_delete_recovers_on_the_next_write(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    stranded = _leave_the_old_bug_behind(sync_service, LEGACY_QUESTION, LEGACY_CHAT)
    conflict_id = _blocker(sync_service).conflict_id

    created = client.post("/api/v1/chats/", json={"title": "After the block"})

    assert created.status_code == 201, created.text
    assert _blocker(sync_service) is None
    conflict = sync_service.store.get_conflict(conflict_id)
    assert (conflict.status, conflict.resolution_action, conflict.resolution_notes) == (
        "dismissed", "skip", server_origin.STRANDED_TOMBSTONE_RESOLUTION_NOTE,
    )
    superseded = sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor)
    assert (superseded.apply_status, superseded.apply_error_code) == ("superseded", "sync_conflict_skipped")
    # The stranded tombstone was never delivered, and the row it named is untouched.
    assert [item.object_id for item in _pull(sync_service)] == [created.json()["id"]]
    assert not _is_deleted(chacha_db, LEGACY_QUESTION)
    assert not _has_sync_history(sync_service, "chat.message", LEGACY_QUESTION)

    # The delete that caused the block now works.
    assert _delete_message(client, chacha_db, LEGACY_QUESTION).status_code == 204
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert _blocker(sync_service) is None


def test_a_blocked_dataset_recovers_when_the_stranded_delete_is_retried(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)
    _legacy_message(chacha_db, "unpublished-save", CHAT_ID, sender="assistant")
    _leave_the_old_bug_behind(sync_service, "unpublished-save", CHAT_ID)

    retried = _delete_message(client, chacha_db, "unpublished-save")

    assert retried.status_code == 204, retried.text
    assert _is_deleted(chacha_db, "unpublished-save")
    assert _blocker(sync_service) is None
    # A published neighbour can be tombstoned again.
    assert _delete_message(client, chacha_db, REPLY_ID).status_code == 204
    assert _log(sync_service)[-1] == ("chat.message", "tombstone", REPLY_ID, "applied")


def test_a_blocked_dataset_recovers_for_a_versioned_history_write(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _leave_the_old_bug_behind(sync_service, LEGACY_QUESTION, LEGACY_CHAT)

    admitted = _admit(client, CHAT_ID, selection)

    assert admitted.status_code == 201, admitted.text
    assert _blocker(sync_service) is None
    assert _log(sync_service)[-1] == ("chat.message", "append", INPUT_ID, "applied")


def test_recovery_leaves_every_other_conflict_for_review(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Only a tombstone of a message with no Sync state is settled.

    A device's stranded tombstone is applied instead of dismissed; that is in
    test_sync_v2_satisfied_tombstone.py.
    """
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _block_dataset(sync_service)

    refused = client.post("/api/v1/chats/", json={"title": "Still blocked"})

    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"]["error_code"] == "sync_server_origin_append_failed"
    blocker = _blocker(sync_service)
    assert (blocker.conflict_id, blocker.status) == ("conflict-unresolved", "unresolved")


# ---------------------------------------------------------------------------
# The helper
# ---------------------------------------------------------------------------


def _objects(*ids: str) -> list[tuple[Any, str]]:
    return [("chat.message", message_id) for message_id in ids]


def test_helper_deletes_what_has_no_history_and_returns_what_has(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)
    deleted: list[list[tuple[Any, str]]] = []

    enrolled = server_origin.delete_unenrolled_server_origin_objects(
        sync_service,
        user_id=USER,
        objects=_objects("never-seen-one", INPUT_ID, "never-seen-two", REPLY_ID),
        delete=lambda objects: deleted.append(list(objects)),
    )

    assert enrolled == _objects(INPUT_ID, REPLY_ID)
    assert deleted == [_objects("never-seen-one", "never-seen-two")]


def test_helper_does_not_call_delete_when_everything_is_published(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _send_turn(client)

    def no_delete(_objects):
        raise AssertionError("nothing here is unpublished")

    assert server_origin.delete_unenrolled_server_origin_objects(
        sync_service, user_id=USER, objects=_objects(INPUT_ID), delete=no_delete,
    ) == _objects(INPUT_ID)


def test_helper_refuses_before_any_delete_when_a_needed_tombstone_is_blocked(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _send_turn(client)
    _block_dataset(sync_service)

    def no_delete(_objects):
        raise AssertionError("the request is refused as a whole")

    with pytest.raises(SyncMaterializationPredecessorError) as raised:
        server_origin.delete_unenrolled_server_origin_objects(
            sync_service, user_id=USER, objects=_objects("never-seen", INPUT_ID), delete=no_delete,
        )
    assert raised.value.conflict_id == "conflict-unresolved"


def test_helper_passes_the_owner_delete_error_through(sync_service: SyncV2Service) -> None:
    class OwnerRefusal(Exception):
        pass

    def refuse(_objects):
        raise OwnerRefusal("stale version")

    with pytest.raises(OwnerRefusal):
        server_origin.delete_unenrolled_server_origin_objects(
            sync_service, user_id=USER, objects=_objects("never-seen"), delete=refuse,
        )
    assert _log(sync_service) == []


def test_helper_reports_an_unavailable_fence_before_any_delete(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
) -> None:
    def no_delete(_objects):
        raise AssertionError("no delete without the fence")

    def busy(self, keys, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(SyncDatabase, "materialization_transaction", busy)

    with pytest.raises(SyncStoreError):
        server_origin.delete_unenrolled_server_origin_objects(
            sync_service, user_id=USER, objects=_objects("never-seen"), delete=no_delete,
        )


def test_helper_keeps_a_committed_delete_when_the_fence_fails_to_commit(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    """The fence recorded nothing, so a delete that is already committed is not reported as failed."""
    _send_turn(client)
    deleted: list[list[tuple[Any, str]]] = []
    original = SyncDatabase.materialization_transaction

    @contextmanager
    def failing_commit(self, keys, **kwargs):
        with original(self, keys, **kwargs) as connection:
            yield connection
        raise RuntimeError("commit failed")

    monkeypatch.setattr(SyncDatabase, "materialization_transaction", failing_commit)

    enrolled = server_origin.delete_unenrolled_server_origin_objects(
        sync_service,
        user_id=USER,
        objects=_objects("never-seen", INPUT_ID),
        delete=lambda objects: deleted.append(list(objects)),
    )

    assert deleted == [_objects("never-seen")]
    assert enrolled == _objects(INPUT_ID)


def test_a_failed_recovery_reports_the_original_refusal(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Recovery is best effort: when it cannot run, the write is refused as a blocked dataset, not as a crash."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _leave_the_old_bug_behind(sync_service, LEGACY_QUESTION, LEGACY_CHAT)

    def unavailable(self, dataset_id):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(SyncDatabase, "conflict_resolution_transaction", unavailable)

    with pytest.raises(SyncMaterializationPredecessorError):
        capture_server_origin_mutation(
            sync_service,
            user_id=USER,
            domain="chat.conversation",
            operation="upsert",
            object_id="chat-after-the-block",
            payload={"title": "Still blocked"},
            source="server_api",
        )
    assert _blocker(sync_service) is not None


def test_helper_refuses_a_client_private_dataset_before_any_delete(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
) -> None:
    """A dataset the server front end may not write is refused as for every other server-origin write."""

    def no_delete(_objects):
        raise AssertionError("no delete for a dataset the server may not write")

    monkeypatch.setattr(server_origin, "server_frontend_mutation_enabled_for_policy", lambda _policy: False)

    with pytest.raises(SyncServerOriginMutationNotSupportedError):
        server_origin.delete_unenrolled_server_origin_objects(
            sync_service, user_id=USER, objects=_objects("never-seen"), delete=no_delete,
        )


# ---------------------------------------------------------------------------
# Without Sync v2
# ---------------------------------------------------------------------------


def test_without_sync_v2_deletes_behave_as_before(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
) -> None:
    """No profile: the same direct deletes as ever, and the Sync store is never opened."""
    inactive = _client(monkeypatch, chacha_db, None)

    def no_sync_access(*_args, **_kwargs):
        raise AssertionError("an inactive Sync profile must not be touched")

    for name in ("insert_envelope", "upsert_object_state", "materialization_transaction", "get_current_head"):
        monkeypatch.setattr(SyncDatabase, name, no_sync_access)
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, LEGACY_ANSWER))
    chat_version = chacha_db.get_conversation_by_id(LEGACY_CHAT)["version"]

    deleted_message = _delete_message(inactive, chacha_db, LEGACY_QUESTION)

    assert deleted_message.status_code == 204, deleted_message.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    # Without a profile the message delete also bumps its conversation.
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT)["version"] == chat_version + 1

    deleted_chat = _delete_chat(inactive, chacha_db, LEGACY_CHAT)

    assert deleted_chat.status_code == 204, deleted_chat.text
    assert chacha_db.get_conversation_by_id(LEGACY_CHAT) is None
    assert _is_deleted(chacha_db, LEGACY_ANSWER)
