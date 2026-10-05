"""A device's delete of a chat message the server does not have is already satisfied.

A device can push a tombstone for a message the server never received through
Sync: it deleted a message it had from before the profile, or one it created
and deleted offline. The server used to reject that tombstone as
``message_base_conflict`` / ``missing_server_message``. The conflict blocked the
whole dataset and no other device ever received the delete.

The rule under test: the end state the device asked for (the message is gone)
already holds, so the tombstone is applied as a no-op.

* It is accepted and applied; no conflict is left and the dataset is not blocked.
* Object state records the message as tombstoned, so a later append of the same
  id is refused by the ordinary head check.
* The tombstone is delivered to the other devices, which converge on "gone".
  A copy the server itself holds outside Sync is deleted too.
* A tombstone that disagrees with a message the server does hold is still a
  conflict.
* A dataset already blocked by such a tombstone recovers on the next write.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.models import SyncConflictCreate, SyncEnvelope, SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.service import SyncPushResult, SyncV2Service
from tldw_Server_API.tests.Sync.test_sync_v2_native_history_capture import (
    CHAT_ID,
    INPUT_ID,
    REPLY_ID,
    USER,
    _active_profile,
    _admit,
    _block_dataset,
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
from tldw_Server_API.tests.Sync.test_sync_v2_unenrolled_delete import (
    LEGACY_CHAT,
    LEGACY_QUESTION,
    _blocker,
    _is_deleted,
    _legacy_chat,
)

pytestmark = pytest.mark.integration

GONE = "message-the-server-never-had"
PUSHER = "offline-device"
THIRD = "third-device"


@pytest.fixture()
def third_device(sync_service: SyncV2Service) -> str:
    sync_service.register_device(user_id=USER, display_name="Phone", client_type="chatbook", device_id=THIRD)
    return THIRD


def _envelope(
    service: SyncV2Service,
    operation: str,
    object_id: str,
    *,
    device: str = PUSHER,
    envelope_id: str | None = None,
    chat_id: str = LEGACY_CHAT,
    content: str | None = None,
    **overrides: Any,
) -> SyncEnvelopeCreate:
    deleted = operation == "tombstone"
    payload: dict[str, Any] = (
        {"id": object_id, "deleted": True}
        if deleted
        else {"conversation_id": chat_id, "sender": "user", "content": content or f"{object_id} text"}
    )
    values: dict[str, Any] = {
        "dataset_id": _dataset_id(service),
        "client_envelope_id": envelope_id or f"env-{device}-{operation}-{object_id}",
        "domain": "chat.message",
        "operation": operation,
        "object_id": object_id,
        "device_id": device,
        "client_sequence": 1,
        "object_revision": 1,
        "parent_id": chat_id,
        "payload": payload,
        "payload_hash": f"sha256:{device}-{operation}-{object_id}",
        "deleted": deleted,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    values.update(overrides)
    return SyncEnvelopeCreate(**values)


def _push(service: SyncV2Service, envelope: SyncEnvelopeCreate) -> SyncPushResult:
    return service.push(
        user_id=USER,
        dataset_id=envelope.dataset_id,
        device_id=envelope.device_id or PUSHER,
        envelopes=[envelope],
    )


def _push_tombstone(service: SyncV2Service, object_id: str = GONE, **overrides: Any) -> SyncPushResult:
    return _push(service, _envelope(service, "tombstone", object_id, **overrides))


def _state(service: SyncV2Service, object_id: str = GONE):
    return service.store.get_object_state(_dataset_id(service), "chat.message", object_id)


def _strand_device_tombstone(service: SyncV2Service, object_id: str = GONE) -> SyncEnvelope:
    """Leave what the old behaviour left: the device's tombstone in conflict, blocking the dataset."""
    dataset_id = _dataset_id(service)
    stranded = service.store.insert_envelope(replace(_envelope(service, "tombstone", object_id), status="accepted"))
    stranded = service.store.mark_envelope_apply_status(
        stranded.server_cursor,
        apply_status="conflict",
        apply_error_code="message_base_conflict",
        apply_error_message="chat.message tombstone requires an existing server message base state",
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-device-tombstone",
            dataset_id=dataset_id,
            domain="chat.message",
            entity_id=object_id,
            conflict_type="message_base_conflict",
            local_envelope_id=stranded.client_envelope_id,
            server_sequence=stranded.server_cursor,
            metadata={"reason": "missing_server_message"},
        )
    )
    assert _blocker(service).conflict_id == "conflict-device-tombstone"
    return stranded


@pytest.fixture()
def replica(tmp_path: Path) -> Iterator[tuple[SyncV2Service, CharactersRAGDB]]:
    """Another replica that runs the same projection: its own Sync store and its own chat database."""
    database = CharactersRAGDB(db_path=str(tmp_path / "replica" / "ChaChaNotes.db"), client_id=USER)
    projection = CharactersRAGDB(db_path=database.db_path_str, client_id=USER)
    try:
        service = _active_profile(SyncDatabase(sqlite_path=tmp_path / "replica" / "Sync_v2.db"), projection)
        yield service, database
    finally:
        projection.close_all_connections()
        database.close_all_connections()


def _deliver(replica_service: SyncV2Service, envelopes: list[SyncEnvelope]) -> SyncPushResult:
    """Hand pulled envelopes to the replica, which projects them under its own rules."""
    return replica_service.push(
        user_id=USER,
        dataset_id=_dataset_id(replica_service),
        device_id=PUSHER,
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=_dataset_id(replica_service),
                client_envelope_id=f"replica-{item.client_envelope_id}",
                domain=item.domain,
                operation=item.operation,
                object_id=item.object_id,
                device_id=PUSHER,
                client_sequence=index,
                object_revision=item.object_revision,
                parent_id=item.parent_id,
                payload=dict(item.payload),
                payload_hash=item.payload_hash,
                deleted=item.deleted,
                encryption_metadata={"policy": "server_trusted_v1"},
            )
            for index, item in enumerate(envelopes, start=1)
        ],
    )


# ---------------------------------------------------------------------------
# The tombstone is applied as a no-op
# ---------------------------------------------------------------------------


def test_device_tombstone_of_a_message_the_server_never_had_is_applied(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    pushed = _push_tombstone(sync_service)

    assert pushed.conflicts == [] and pushed.rejected == []
    assert [(item.entity_id, item.apply_status) for item in pushed.accepted] == [(GONE, "applied")]
    assert _log(sync_service) == [("chat.message", "tombstone", GONE, "applied")]
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    assert _blocker(sync_service) is None
    state = _state(sync_service)
    assert (state.deleted, state.object_revision, state.latest_server_cursor) == (
        True, 1, pushed.accepted[0].server_sequence,
    )
    # Nothing was invented to delete.
    assert chacha_db.get_message_by_id(GONE, include_deleted=True) is None


def test_the_dataset_still_takes_writes_after_a_satisfied_tombstone(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    assert _push_tombstone(sync_service).conflicts == []

    created = client.post("/api/v1/chats/", json={"title": "After the tombstone"})
    assert created.status_code == 201, created.text
    reference = _send_turn(client)
    appended = _push(
        sync_service, _envelope(sync_service, "append", "device-follow-up", chat_id=CHAT_ID, client_sequence=2)
    )

    assert reference["input_message_id"] == INPUT_ID
    assert [item.apply_status for item in appended.accepted] == ["applied"], appended.rejected
    assert _blocker(sync_service) is None
    assert {item[3] for item in _log(sync_service)} == {"applied"}


def test_pushing_the_same_tombstone_again_is_an_idempotent_replay(sync_service: SyncV2Service) -> None:
    first = _push_tombstone(sync_service)
    again = _push_tombstone(sync_service)

    assert [item.apply_status for item in again.accepted] == ["applied"]
    assert again.conflicts == []
    assert again.accepted[0].server_sequence == first.accepted[0].server_sequence
    assert len(_log(sync_service)) == 1


def test_a_copy_the_server_holds_outside_sync_is_deleted_with_the_tombstone(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The device deleted a message both sides had before the profile: it ends up deleted on both."""
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION, "legacy-neighbour"))

    pushed = _push_tombstone(sync_service, LEGACY_QUESTION)

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert not _is_deleted(chacha_db, "legacy-neighbour")
    assert _state(sync_service, LEGACY_QUESTION).deleted is True
    assert _blocker(sync_service) is None


# ---------------------------------------------------------------------------
# The other devices converge
# ---------------------------------------------------------------------------


def test_a_third_device_receives_the_tombstone(sync_service: SyncV2Service, third_device: str) -> None:
    _push_tombstone(sync_service)

    pulled = _pull(sync_service, third_device)

    assert [(item.operation, item.object_id, item.device_id) for item in pulled] == [("tombstone", GONE, PUSHER)]


def test_a_replica_that_never_had_the_message_applies_the_tombstone_as_a_no_op(
    sync_service: SyncV2Service,
    third_device: str,
    replica: tuple[SyncV2Service, CharactersRAGDB],
) -> None:
    replica_service, replica_db = replica
    _push_tombstone(sync_service)

    delivered = _deliver(replica_service, _pull(sync_service, third_device))

    assert delivered.conflicts == []
    assert [item.apply_status for item in delivered.accepted] == ["applied"]
    assert replica_db.get_message_by_id(GONE, include_deleted=True) is None
    assert _state(replica_service).deleted is True
    assert _blocker(replica_service) is None


def test_a_replica_that_still_has_the_message_deletes_it(
    sync_service: SyncV2Service,
    third_device: str,
    replica: tuple[SyncV2Service, CharactersRAGDB],
) -> None:
    """The copy a third device kept outside Sync goes when the tombstone arrives: every replica ends at "gone"."""
    replica_service, replica_db = replica
    _legacy_chat(replica_db, messages=(GONE, "replica-neighbour"))
    _push_tombstone(sync_service)

    delivered = _deliver(replica_service, _pull(sync_service, third_device))

    assert delivered.conflicts == []
    assert [item.apply_status for item in delivered.accepted] == ["applied"]
    assert _is_deleted(replica_db, GONE)
    assert not _is_deleted(replica_db, "replica-neighbour")
    assert _blocker(replica_service) is None


def test_a_later_append_of_the_same_id_is_refused_by_the_head_check_without_blocking(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    third_device: str,
) -> None:
    """A third device that created the message offline learns it was deleted; nothing is resurrected."""
    _legacy_chat(chacha_db)
    _push_tombstone(sync_service)

    late = _push(sync_service, _envelope(sync_service, "append", GONE, device=third_device))

    assert late.accepted == []
    assert [item.client_envelope_id for item in late.conflicts] == [f"env-{third_device}-append-{GONE}"]
    assert chacha_db.get_message_by_id(GONE, include_deleted=True) is None
    assert _state(sync_service).deleted is True
    # A push-time conflict is not a projection conflict: the dataset is healthy and the device can pull the delete.
    assert _blocker(sync_service) is None
    assert [(item.operation, item.object_id) for item in _pull(sync_service, third_device)] == [("tombstone", GONE)]
    assert client.post("/api/v1/chats/", json={"title": "Still healthy"}).status_code == 201


# ---------------------------------------------------------------------------
# What is still a conflict
# ---------------------------------------------------------------------------


def test_a_tombstone_whose_base_disagrees_with_a_message_the_server_holds_is_still_a_conflict(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)
    held = _state(sync_service, REPLY_ID)

    stale = _push_tombstone(
        sync_service,
        REPLY_ID,
        chat_id=CHAT_ID,
        object_revision=2,
        base_server_cursor=held.latest_server_cursor,
        base_object_revision=held.object_revision,
        base_object_hash="sha256:not-what-the-server-holds",
    )

    assert stale.accepted == []
    assert [item.entity_id for item in stale.conflicts] == [REPLY_ID]
    assert not _is_deleted(chacha_db, REPLY_ID)
    assert _state(sync_service, REPLY_ID).deleted is False


def test_a_baseless_tombstone_of_a_message_the_server_holds_is_still_a_conflict(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Only a message the dataset has no history for is "already gone"; this one is published and live."""
    _send_turn(client)

    baseless = _push_tombstone(sync_service, INPUT_ID, chat_id=CHAT_ID)

    assert baseless.accepted == []
    assert [item.entity_id for item in baseless.conflicts] == [INPUT_ID]
    assert not _is_deleted(chacha_db, INPUT_ID)
    assert _state(sync_service, INPUT_ID).deleted is False


def test_a_tombstone_that_names_a_base_the_dataset_never_issued_is_a_push_conflict_that_blocks_nothing(
    client: TestClient,
    sync_service: SyncV2Service,
    third_device: str,
) -> None:
    """The device claims server history that does not exist. It is told so, and nothing is held up.

    Unchanged by the satisfied rule: the head check refuses it before any projection.
    """
    claimed = _push_tombstone(
        sync_service,
        object_revision=2,
        base_server_cursor=7,
        base_object_revision=1,
        base_object_hash="sha256:a-base-this-dataset-never-issued",
    )

    assert claimed.accepted == []
    assert [item.entity_id for item in claimed.conflicts] == [GONE]
    assert _state(sync_service) is None
    assert _blocker(sync_service) is None
    assert _pull(sync_service, third_device) == []
    assert client.post("/api/v1/chats/", json={"title": "Still healthy"}).status_code == 201


def test_a_tombstone_never_deletes_a_message_in_another_owners_chat(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The message is found by id alone, and one PostgreSQL database holds every owner's chats.

    With no Sync state tying the id to this owner, only a chat this owner holds
    can be reached. For this owner the message does not exist, so the tombstone
    is satisfied and the other owner's row is untouched.
    """
    with chacha_db.transaction() as conn:
        conn.execute(
            "INSERT INTO conversations (id, root_id, title, client_id) VALUES (?, ?, ?, ?)",
            ("their-chat", "their-chat", "Theirs", "someone-else"),
        )
        conn.execute(
            "INSERT INTO messages (id, conversation_id, sender, content, client_id) VALUES (?, ?, ?, ?, ?)",
            ("their-message", "their-chat", "user", "not yours to delete", "someone-else"),
        )

    pushed = _push_tombstone(sync_service, "their-message")

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert not _is_deleted(chacha_db, "their-message")
    assert _state(sync_service, "their-message").deleted is True
    assert _blocker(sync_service) is None


def test_rows_that_belong_to_other_sync_history_are_not_deleted_by_a_baseless_tombstone(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    projection_db: CharactersRAGDB,
) -> None:
    """A row another Sync history projected, with no state in this dataset, is not this tombstone's to delete."""
    _legacy_chat(chacha_db)
    projection_db.append_message_from_sync(
        stable_message_id="projected-elsewhere",
        conversation_id=LEGACY_CHAT,
        sender="user",
        content="from another dataset",
        timestamp=None,
        sync_client_id=USER,
        object_revision=1,
        payload_hash="sha256:another-dataset",
    )

    pushed = _push_tombstone(sync_service, "projected-elsewhere")

    assert pushed.accepted == []
    assert [item.entity_id for item in pushed.conflicts] == ["projected-elsewhere"]
    assert not _is_deleted(chacha_db, "projected-elsewhere")
    blocker = _blocker(sync_service)
    assert (blocker.conflict_type, blocker.metadata["reason"]) == ("message_base_conflict", "missing_server_message")


# ---------------------------------------------------------------------------
# A dataset already blocked by a device's tombstone
# ---------------------------------------------------------------------------


def test_a_dataset_blocked_by_a_device_tombstone_recovers_on_the_next_server_write(
    client: TestClient,
    sync_service: SyncV2Service,
    third_device: str,
) -> None:
    stranded = _strand_device_tombstone(sync_service)

    created = client.post("/api/v1/chats/", json={"title": "After the block"})

    assert created.status_code == 201, created.text
    assert _blocker(sync_service) is None
    # The device's delete is honoured, not discarded: applied, recorded, and delivered.
    settled = sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor)
    assert (settled.apply_status, settled.apply_error_code) == ("applied", None)
    assert _state(sync_service).deleted is True
    conflict = sync_service.store.get_conflict("conflict-device-tombstone")
    assert conflict.status == "resolved"
    assert [(item.operation, item.object_id) for item in _pull(sync_service, third_device)] == [
        ("tombstone", GONE),
        ("upsert", created.json()["id"]),
    ]


def test_a_dataset_blocked_by_a_device_tombstone_recovers_on_the_next_push(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    third_device: str,
) -> None:
    """A user who only writes from devices is not left waiting for a server write."""
    _legacy_chat(chacha_db)
    stranded = _strand_device_tombstone(sync_service)

    pushed = _push(sync_service, _envelope(sync_service, "append", "from-the-phone", device=third_device))

    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert _blocker(sync_service) is None
    assert sync_service.store.get_envelope_by_server_cursor(stranded.server_cursor).apply_status == "applied"
    assert chacha_db.get_message_by_id("from-the-phone") is not None


def test_a_device_that_pushes_its_stranded_tombstone_again_gets_it_applied(sync_service: SyncV2Service) -> None:
    stranded = _strand_device_tombstone(sync_service)

    again = _push_tombstone(sync_service)

    assert again.conflicts == []
    assert [(item.server_sequence, item.apply_status) for item in again.accepted] == [
        (stranded.server_cursor, "applied"),
    ]
    assert _blocker(sync_service) is None
    assert sync_service.store.get_conflict("conflict-device-tombstone").status == "resolved"


def test_recovery_deletes_the_copy_the_server_holds_outside_sync(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _legacy_chat(chacha_db, messages=(LEGACY_QUESTION,))
    _strand_device_tombstone(sync_service, LEGACY_QUESTION)

    created = client.post("/api/v1/chats/", json={"title": "After the block"})

    assert created.status_code == 201, created.text
    assert _is_deleted(chacha_db, LEGACY_QUESTION)
    assert _state(sync_service, LEGACY_QUESTION).deleted is True
    assert _blocker(sync_service) is None


def test_a_dataset_blocked_by_a_device_tombstone_recovers_for_a_versioned_history_write(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    _strand_device_tombstone(sync_service)

    admitted = _admit(client, CHAT_ID, selection)

    assert admitted.status_code == 201, admitted.text
    assert _blocker(sync_service) is None
    assert _log(sync_service)[-1] == ("chat.message", "append", INPUT_ID, "applied")


def test_recovery_still_leaves_every_other_conflict_for_review(
    client: TestClient,
    sync_service: SyncV2Service,
    third_device: str,
) -> None:
    _block_dataset(sync_service)

    refused = client.post("/api/v1/chats/", json={"title": "Still blocked"})
    pushed = _push(sync_service, _envelope(sync_service, "append", "from-the-phone", device=third_device))

    assert refused.status_code == 503, refused.text
    assert pushed.accepted == []
    assert [item.conflict_id for item in pushed.conflicts] == ["conflict-unresolved"]
    assert _blocker(sync_service).conflict_id == "conflict-unresolved"
