"""Applied server-origin capture: publish a product write the caller already committed.

Versioned chat history writes validate and append under the conversation owner's
own transaction, so they cannot be replayed through a Sync materializer. They
run inside the Sync projection fence instead, and what they wrote is recorded as
an accepted envelope that is already applied.

Contract:

* The write runs once, inside the fence. If it raises, nothing is recorded.
* Each object it reports gets one accepted, applied, server-origin envelope and
  matching object state, in the order reported.
* An object that already has Sync history is left alone, so a replayed write
  publishes nothing twice and a write whose capture was lost is published by
  the retry.
* A dataset that cannot take the envelope refuses before the write runs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterRejected,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncMaterializationPredecessorError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.materializers import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    M1_SYNC_OPERATIONS,
    SyncConflictCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    SERVER_ORIGIN_DEVICE_ID,
    AppliedServerOriginObject,
    SyncServerOriginMaterializationError,
    SyncServerOriginMutationNotSupportedError,
    canonical_payload_hash,
    capture_applied_server_origin_write,
    capture_server_origin_mutation,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

USER = "user-1"
CHAT = "chat-native-1"
MESSAGE = "message-native-1"


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(db_path=str(tmp_path / "ChaChaNotes.db"), client_id=USER)


def _service(tmp_path: Path, chacha_db: CharactersRAGDB, registry: SyncAdapterRegistry) -> SyncV2Service:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=registry,
        materializers={
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
        },
        clock=lambda: "2026-05-23T18:12:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            supported_domains=list(M1_SYNC_DOMAINS),
            operations={domain: list(operations) for domain, operations in M1_SYNC_OPERATIONS.items()},
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.bootstrap_profile(
        user_id=USER,
        mode="server_frontend",
        device_id="frontend-device",
        device_name="Server frontend",
    )
    service.register_device(
        user_id=USER,
        display_name="Offline laptop",
        client_type="chatbook",
        device_id="offline-device",
    )
    return service


@pytest.fixture()
def sync_service(tmp_path: Path, chacha_db: CharactersRAGDB) -> SyncV2Service:
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
    )
    return _service(tmp_path, chacha_db, registry)


def _dataset_id(service: SyncV2Service) -> str:
    return service.profile(user_id=USER).active_dataset_id or ""


def _chat_envelopes(service: SyncV2Service):
    return service.store.list_envelopes_after(
        _dataset_id(service),
        0,
        domains=["chat.conversation", "chat.message"],
        limit=50,
    )


def _conversation_object(chat_id: str = CHAT, title: str = "Native chat") -> AppliedServerOriginObject:
    return AppliedServerOriginObject(
        domain="chat.conversation",
        operation="upsert",
        object_id=chat_id,
        payload={"title": title, "root_id": chat_id, "scope_type": "global", "client_id": USER},
    )


def _message_object(
    message_id: str = MESSAGE,
    *,
    chat_id: str = CHAT,
    parent: str | None = None,
    content: str = "Hello from a versioned write.",
) -> AppliedServerOriginObject:
    return AppliedServerOriginObject(
        domain="chat.message",
        operation="append",
        object_id=message_id,
        parent_id=chat_id,
        payload={
            "conversation_id": chat_id,
            "parent_message_id": parent,
            "sender": "user",
            "content": content,
            "timestamp": "2026-10-04T12:00:00.000Z",
            "client_id": USER,
        },
    )


def _block_dataset(service: SyncV2Service) -> None:
    """Leave an accepted envelope whose projection conflicted and is still unresolved."""
    dataset_id = _dataset_id(service)
    blocked = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-unresolved",
            domain="notes.note",
            operation="upsert",
            object_id="note-unresolved",
            device_id="offline-device",
            client_sequence=1,
            object_revision=1,
            payload={"title": "Conflict", "content": "Never projected"},
            payload_hash="sha256:unresolved",
            encryption_metadata={"policy": "server_trusted_v1"},
            status="accepted",
        )
    )
    blocked = service.store.mark_envelope_apply_status(
        blocked.server_cursor,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-unresolved",
            dataset_id=dataset_id,
            domain=blocked.domain,
            entity_id=blocked.object_id,
            conflict_type="projection_conflict",
            local_envelope_id=blocked.client_envelope_id,
            server_sequence=blocked.server_cursor,
        )
    )


def _native_message(db: CharactersRAGDB, message_id: str = MESSAGE, *, chat_id: str = CHAT) -> str:
    if db.get_conversation_by_id(chat_id) is None:
        db.add_conversation({"id": chat_id, "character_id": None, "title": "Native chat", "client_id": USER})
    return db.add_message(
        {"id": message_id, "conversation_id": chat_id, "sender": "user", "content": "Hello from a versioned write."}
    )


def test_write_is_recorded_as_one_applied_server_origin_envelope_with_object_state(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    result = capture_applied_server_origin_write(
        sync_service,
        user_id=USER,
        source="server_api",
        claims=[("chat.message", MESSAGE)],
        write=lambda: _native_message(chacha_db),
        describe=lambda message_id: [_message_object(message_id)],
    )

    assert result == MESSAGE
    (envelope,) = [item for item in _chat_envelopes(sync_service) if item.domain == "chat.message"]
    expected = _message_object()
    payload_hash, payload_size = canonical_payload_hash(expected.payload)
    assert (envelope.operation, envelope.object_id, envelope.parent_id) == ("append", MESSAGE, CHAT)
    assert (envelope.status, envelope.apply_status) == ("accepted", "applied")
    assert envelope.applied_at is not None
    assert envelope.device_id == SERVER_ORIGIN_DEVICE_ID
    assert envelope.payload == expected.payload
    assert (envelope.payload_hash, envelope.payload_size_bytes) == (payload_hash, payload_size)
    assert (envelope.object_revision, envelope.base_server_cursor, envelope.base_object_revision) == (1, None, None)
    assert envelope.routing_metadata["origin"] == "server"
    assert envelope.routing_metadata["source"] == "server_api"
    assert envelope.routing_metadata["server_owner_user_id"] == USER
    assert "Hello from a versioned write." not in str(envelope.routing_metadata)
    assert envelope.encryption_metadata["policy"] == "server_trusted_v1"

    state = sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", MESSAGE)
    assert state is not None
    assert (state.object_revision, state.object_hash, state.latest_server_cursor, state.deleted) == (
        1,
        payload_hash,
        envelope.server_cursor,
        False,
    )
    # The projection is the caller's row, untouched: no Sync metadata was stamped on it.
    assert chacha_db.get_message_metadata(MESSAGE) is None


def test_objects_are_recorded_in_the_order_the_write_reports_them(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    def write() -> list[str]:
        _native_message(chacha_db, "input-1")
        chacha_db.add_message(
            {
                "id": "reply-1",
                "conversation_id": CHAT,
                "sender": "assistant",
                "content": "Reply",
                "parent_message_id": "input-1",
            }
        )
        return ["input-1", "reply-1"]

    capture_applied_server_origin_write(
        sync_service,
        user_id=USER,
        source="server_api",
        claims=[("chat.conversation", CHAT), ("chat.message", "input-1"), ("chat.message", "reply-1")],
        write=write,
        describe=lambda ids: [
            _conversation_object(),
            _message_object(ids[0]),
            _message_object(ids[1], parent=ids[0], content="Reply"),
        ],
    )

    envelopes = _chat_envelopes(sync_service)
    assert [(item.domain, item.object_id, item.apply_status) for item in envelopes] == [
        ("chat.conversation", CHAT, "applied"),
        ("chat.message", "input-1", "applied"),
        ("chat.message", "reply-1", "applied"),
    ]
    assert [item.server_cursor for item in envelopes] == sorted(item.server_cursor for item in envelopes)
    assert envelopes[2].payload["parent_message_id"] == "input-1"


def test_failed_write_records_nothing(sync_service: SyncV2Service) -> None:
    class Refused(Exception):
        pass

    def write() -> str:
        raise Refused("stale_selection")

    def describe(_result: str) -> list[AppliedServerOriginObject]:
        raise AssertionError("a refused write has nothing to describe")

    with pytest.raises(Refused):
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=describe,
        )

    assert _chat_envelopes(sync_service) == []
    assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", MESSAGE) is None


def test_replayed_write_publishes_nothing_twice(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    writes: list[str] = []

    def write() -> str:
        writes.append("write")
        if chacha_db.get_message_by_id(MESSAGE) is None:
            _native_message(chacha_db)
        return MESSAGE

    for _ in range(2):
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert writes == ["write", "write"]
    assert [(item.domain, item.object_id) for item in _chat_envelopes(sync_service)] == [("chat.message", MESSAGE)]


def test_retry_publishes_a_write_whose_capture_was_lost(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The product row commits first; a lost capture is healed by the same idempotent request."""

    def write() -> str:
        if chacha_db.get_message_by_id(MESSAGE) is None:
            _native_message(chacha_db)
        return MESSAGE

    def capture() -> str:
        return capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    original_insert = SyncDatabase.insert_envelope

    def failing_insert(self, envelope, **kwargs):
        raise SyncStoreError("append is unavailable")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", failing_insert)
    with pytest.raises(SyncStoreError):
        capture()
    assert chacha_db.get_message_by_id(MESSAGE) is not None
    assert _chat_envelopes(sync_service) == []
    assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", MESSAGE) is None

    monkeypatch.setattr(SyncDatabase, "insert_envelope", original_insert)
    assert capture() == MESSAGE
    assert [(item.object_id, item.apply_status) for item in _chat_envelopes(sync_service)] == [(MESSAGE, "applied")]


def test_failure_after_the_write_is_reported_as_a_sync_failure_and_records_nothing(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Callers map SyncStoreError to a retryable response; a raw error would look like a failed write."""

    def write() -> list[str]:
        _native_message(chacha_db, "input-1")
        return ["input-1", "reply-missing"]

    def describe(ids: list[str]) -> list[AppliedServerOriginObject]:
        published = [_message_object(ids[0])]
        if chacha_db.get_message_by_id(ids[1]) is None:
            raise RuntimeError("row unavailable")
        return published

    with pytest.raises(SyncStoreError) as exc_info:
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", "input-1"), ("chat.message", "reply-missing")],
            write=write,
            describe=describe,
        )

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert chacha_db.get_message_by_id("input-1") is not None
    assert _chat_envelopes(sync_service) == []


def test_unavailable_fence_is_a_sync_failure_before_the_write(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
) -> None:
    """A backend that cannot open the fence transaction (a lock timeout) must not look like a failed write."""
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    @contextmanager
    def unavailable(self, keys, **kwargs):
        raise DatabaseError("SQLite transaction begin failed")
        yield  # pragma: no cover

    monkeypatch.setattr(SyncDatabase, "materialization_transaction", unavailable)

    def write() -> str:
        raise AssertionError("no fence, no product write")

    with pytest.raises(SyncStoreError) as exc_info:
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert isinstance(exc_info.value.__cause__, DatabaseError)


def test_failed_sync_commit_is_a_sync_failure_and_the_retry_publishes(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The window between the two commits: the product row stands, the log does not have it yet."""
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    def write() -> str:
        if chacha_db.get_message_by_id(MESSAGE) is None:
            _native_message(chacha_db)
        return MESSAGE

    def capture() -> str:
        return capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    original = SyncDatabase.materialization_transaction

    @contextmanager
    def failing_commit(self, keys, **kwargs):
        with original(self, keys, **kwargs) as connection:
            yield connection
            # Raised inside the transaction, so it rolls back as a failed commit would.
            raise DatabaseError("SQLite transaction commit failed")

    monkeypatch.setattr(SyncDatabase, "materialization_transaction", failing_commit)
    with pytest.raises(SyncStoreError) as exc_info:
        capture()
    assert isinstance(exc_info.value.__cause__, DatabaseError)
    assert chacha_db.get_message_by_id(MESSAGE) is not None
    assert _chat_envelopes(sync_service) == []
    assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", MESSAGE) is None

    monkeypatch.setattr(SyncDatabase, "materialization_transaction", original)
    assert capture() == MESSAGE
    assert [(item.object_id, item.apply_status) for item in _chat_envelopes(sync_service)] == [(MESSAGE, "applied")]


def test_error_from_the_write_itself_is_not_turned_into_a_sync_failure(sync_service: SyncV2Service) -> None:
    """A database error raised by the product write is the caller's to map, not a Sync error."""
    from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError

    def write() -> str:
        raise DatabaseError("product write failed")

    with pytest.raises(DatabaseError, match="product write failed"):
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert _chat_envelopes(sync_service) == []


def test_partly_recorded_capture_rolls_back_as_a_whole(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A reply is never left published without the input recorded in the same capture."""

    def write() -> list[str]:
        if chacha_db.get_message_by_id("input-1") is None:
            _native_message(chacha_db, "input-1")
            chacha_db.add_message(
                {
                    "id": "reply-1",
                    "conversation_id": CHAT,
                    "sender": "assistant",
                    "content": "Reply",
                    "parent_message_id": "input-1",
                }
            )
        return ["input-1", "reply-1"]

    def capture() -> None:
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", "input-1"), ("chat.message", "reply-1")],
            write=write,
            describe=lambda ids: [_message_object(ids[0]), _message_object(ids[1], parent=ids[0], content="Reply")],
        )

    original_upsert = SyncDatabase.upsert_object_state

    def fail_on_reply(self, state, **kwargs):
        if state.object_id == "reply-1":
            raise SyncStoreError("object state is unavailable")
        return original_upsert(self, state, **kwargs)

    monkeypatch.setattr(SyncDatabase, "upsert_object_state", fail_on_reply)
    with pytest.raises(SyncStoreError):
        capture()
    assert _chat_envelopes(sync_service) == []
    for message_id in ("input-1", "reply-1"):
        assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", message_id) is None

    monkeypatch.setattr(SyncDatabase, "upsert_object_state", original_upsert)
    capture()
    assert [(item.object_id, item.apply_status) for item in _chat_envelopes(sync_service)] == [
        ("input-1", "applied"),
        ("reply-1", "applied"),
    ]


def test_replay_of_a_published_object_still_works_while_the_dataset_is_blocked(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A blocked dataset takes no new history, but an idempotent replay adds none."""

    def capture() -> str:
        return capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=lambda: MESSAGE if chacha_db.get_message_by_id(MESSAGE) else _native_message(chacha_db),
            describe=lambda message_id: [_message_object(message_id)],
        )

    capture()
    _block_dataset(sync_service)
    before = _chat_envelopes(sync_service)

    assert capture() == MESSAGE
    assert _chat_envelopes(sync_service) == before


def test_object_with_existing_sync_history_is_left_alone(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    capture_server_origin_mutation(
        sync_service,
        user_id=USER,
        domain="chat.conversation",
        operation="upsert",
        object_id=CHAT,
        payload={"title": "Created through Sync", "scope_type": "global"},
        source="server_api",
    )
    before = _chat_envelopes(sync_service)

    capture_applied_server_origin_write(
        sync_service,
        user_id=USER,
        source="server_api",
        claims=[("chat.conversation", CHAT)],
        write=lambda: CHAT,
        describe=lambda _chat_id: [_conversation_object(title="A different title")],
    )

    assert _chat_envelopes(sync_service) == before
    assert chacha_db.get_conversation_by_id(CHAT)["title"] == "Created through Sync"


def test_unresolved_projection_conflict_refuses_before_the_write(sync_service: SyncV2Service) -> None:
    _block_dataset(sync_service)

    def write() -> str:
        raise AssertionError("a blocked dataset must refuse before the product write")

    with pytest.raises(SyncMaterializationPredecessorError):
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert _chat_envelopes(sync_service) == []


def test_claimed_id_with_unapplied_sync_history_refuses_before_the_write(sync_service: SyncV2Service) -> None:
    pending = sync_service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=_dataset_id(sync_service),
            client_envelope_id="env-device-pending",
            domain="chat.message",
            operation="append",
            object_id=MESSAGE,
            device_id="offline-device",
            client_sequence=1,
            object_revision=1,
            payload={"conversation_id": CHAT, "sender": "user", "content": "From the laptop"},
            payload_hash="sha256:device-pending",
            encryption_metadata={"policy": "server_trusted_v1"},
            status="accepted",
        )
    )
    assert pending.apply_status == "pending"

    def write() -> str:
        raise AssertionError("an id with unapplied Sync history must refuse before the product write")

    with pytest.raises(SyncServerOriginMaterializationError) as exc_info:
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert exc_info.value.envelope.client_envelope_id == "env-device-pending"
    assert [item.client_envelope_id for item in _chat_envelopes(sync_service)] == ["env-device-pending"]


def test_client_private_dataset_refuses_before_the_write(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
) -> None:
    dataset = sync_service.store.list_datasets_for_user(USER)[0]
    private_dataset = replace(dataset, encryption_policy="client_private_v1")
    monkeypatch.setattr(
        sync_service.store,
        "list_datasets_for_user",
        lambda user_id: [private_dataset] if user_id == USER else [],
    )

    def write() -> str:
        raise AssertionError("a client-private dataset must refuse before the product write")

    with pytest.raises(SyncServerOriginMutationNotSupportedError) as exc_info:
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert exc_info.value.error_code == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE


def test_user_without_a_default_dataset_refuses_before_the_write(sync_service: SyncV2Service) -> None:
    def write() -> str:
        raise AssertionError("a user without a Sync dataset must refuse before the product write")

    with pytest.raises(SyncStoreError):
        capture_applied_server_origin_write(
            sync_service,
            user_id="user-without-sync",
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )


def test_adapter_that_rejects_the_domain_version_refuses_before_the_write(
    tmp_path: Path,
    chacha_db: CharactersRAGDB,
) -> None:
    registry = SyncAdapterRegistry(
        [
            StaticSyncAdapter(
                domain=domain,
                supported_adapter_versions={2} if domain == "chat.message" else {1},
            )
            for domain in M1_SYNC_DOMAINS
        ]
    )
    service = _service(tmp_path, chacha_db, registry)

    def write() -> str:
        raise AssertionError("an unsupported adapter version must refuse before the product write")

    with pytest.raises(SyncStoreError):
        capture_applied_server_origin_write(
            service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=write,
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert _chat_envelopes(service) == []


def test_adapter_rejection_of_the_final_envelope_is_an_error_not_a_silent_skip(
    tmp_path: Path,
    chacha_db: CharactersRAGDB,
) -> None:
    """A rejecting adapter cannot be skipped: the caller must learn the write was not published."""
    from tldw_Server_API.app.core.Sync.v2.server_origin import applied_server_origin_envelope_id

    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
    )
    service = _service(tmp_path, chacha_db, registry)
    envelope_id = applied_server_origin_envelope_id(
        _dataset_id(service), source="server_api", domain="chat.message", operation="append", object_id=MESSAGE
    )
    registry.get("chat.message").outcomes = {
        envelope_id: AdapterRejected(
            client_envelope_id=envelope_id,
            error_code="policy_rejected",
            message="chat.message is not accepted here",
        )
    }

    with pytest.raises(SyncStoreError, match="chat.message is not accepted here"):
        capture_applied_server_origin_write(
            service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=lambda: _native_message(chacha_db),
            describe=lambda message_id: [_message_object(message_id)],
        )

    assert _chat_envelopes(service) == []


def test_only_first_revision_creates_can_be_recorded(sync_service: SyncV2Service) -> None:
    with pytest.raises(SyncStoreError):
        capture_applied_server_origin_write(
            sync_service,
            user_id=USER,
            source="server_api",
            claims=[("chat.message", MESSAGE)],
            write=lambda: MESSAGE,
            describe=lambda message_id: [
                AppliedServerOriginObject(
                    domain="chat.message",
                    operation="tombstone",
                    object_id=message_id,
                    payload={"id": message_id, "deleted": True},
                )
            ],
        )

    assert _chat_envelopes(sync_service) == []


def test_recorded_message_can_be_tombstoned_and_pulled_like_any_synced_message(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Object state is what lets a later delete tombstone the message instead of conflicting."""
    capture_applied_server_origin_write(
        sync_service,
        user_id=USER,
        source="server_api",
        claims=[("chat.message", MESSAGE)],
        write=lambda: _native_message(chacha_db),
        describe=lambda message_id: [_message_object(message_id)],
    )

    tombstone = capture_server_origin_mutation(
        sync_service,
        user_id=USER,
        domain="chat.message",
        operation="tombstone",
        object_id=MESSAGE,
        parent_id=CHAT,
        payload={"id": MESSAGE, "deleted": True, "conversation_id": CHAT},
        source="server_api",
    )

    assert tombstone.envelope.apply_status == "applied"
    assert chacha_db.get_message_by_id(MESSAGE, include_deleted=True)["deleted"] in (1, True)
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    pulled = sync_service.pull(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id="offline-device",
        cursor="0",
        domains=["chat.message"],
    )
    assert [(item.operation, item.object_id) for item in pulled.envelopes] == [
        ("append", MESSAGE),
        ("tombstone", MESSAGE),
    ]


def test_full_replay_treats_a_recorded_envelope_as_already_projected(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Repair re-runs applied envelopes too; it must not rewrite or duplicate the caller's row."""
    capture_applied_server_origin_write(
        sync_service,
        user_id=USER,
        source="server_api",
        claims=[("chat.message", MESSAGE)],
        write=lambda: _native_message(chacha_db),
        describe=lambda message_id: [_message_object(message_id)],
    )
    before = chacha_db.get_message_by_id(MESSAGE)

    report = sync_service.repair(user_id=USER, dataset_id=_dataset_id(sync_service), domains=["chat.message"])

    assert (report.applied_count, report.failed_count, report.conflict_count) == (1, 0, 0)
    assert chacha_db.get_message_by_id(MESSAGE) == before
    assert chacha_db.get_message_metadata(MESSAGE) is None
    assert chacha_db.count_messages_for_conversation(CHAT) == 1
