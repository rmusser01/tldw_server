"""Versioned chat history and client-id chat creation with an active Sync v2 profile.

A user who also runs a Sync v2 device (the Chatbook desktop client) must be able
to use the WebUI's native history owner:

* capture a history selection;
* admit a user turn with ``tldw_history_selection_v1``;
* settle the assistant reply with ``tldw_history_admission_v1``;
* create the chat with a client-supplied id (D7 P3).

Each write must reach the other Sync v2 devices as the same envelopes an
ordinary Sync-routed write produces (``chat.conversation`` upsert,
``chat.message`` append), exactly once, and must leave the dataset healthy for
later deletes and device pushes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
from tldw_Server_API.app.api.v1.endpoints import character_chat_sessions as chat_sessions_endpoint
from tldw_Server_API.app.api.v1.endpoints import character_messages as messages_endpoint
from tldw_Server_API.app.api.v1.endpoints import chat as chat_endpoint
from tldw_Server_API.app.core.Chat.history_selection import resolve_history_selection
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    M1_SYNC_OPERATIONS,
    SyncConflictCreate,
    SyncEnvelope,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin import SERVER_ORIGIN_DEVICE_ID
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration

USER = "user-1"
CHAT_ID = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
CHAT_BODY = {"title": "Planning notes", "state": "in-progress", "source": "webui-chat"}
INPUT_ID = "0b6a1c9e-7d15-4f0e-8a3c-2f4d5e6a7b80"
REPLY_ID = "1c7b2d0f-8e26-4a1f-9b4d-3a5e6f7b8c91"
CHAT_DOMAINS = ["chat.conversation", "chat.message"]


class _NoopCharacterRateLimiter:
    async def check_rate_limit(self, user_id, operation):
        return None

    async def check_chat_limit(self, user_id, current_chat_count):
        return None

    async def check_message_send_rate(self, user_id):
        return None

    async def check_message_limit(self, chat_id, message_count):
        return None


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@pytest.fixture()
def chacha_db(tmp_path: Path) -> Iterator[CharactersRAGDB]:
    database = CharactersRAGDB(db_path=str(tmp_path / "ChaChaNotes.db"), client_id=USER)
    try:
        yield database
    finally:
        database.close_all_connections()


@pytest.fixture()
def projection_db(chacha_db: CharactersRAGDB) -> Iterator[CharactersRAGDB]:
    """The Sync materializers' own handle on the owner's database, as the service factory wires it."""
    database = CharactersRAGDB(db_path=chacha_db.db_path_str, client_id=USER)
    try:
        yield database
    finally:
        database.close_all_connections()


@pytest.fixture()
def sync_service(tmp_path: Path, projection_db: CharactersRAGDB) -> SyncV2Service:
    """An active Sync v2 profile: the server front end plus one registered Chatbook device."""
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
    )
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db")),
        adapters=registry,
        materializers={
            "chat.conversation": ChatConversationMaterializer(projection_db),
            "chat.message": ChatMessageMaterializer(projection_db),
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


def _client(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service | None,
) -> TestClient:
    app = FastAPI()
    app.include_router(chat_sessions_endpoint.router, prefix="/api/v1/chats")
    app.include_router(messages_endpoint.router, prefix="/api/v1")
    app.include_router(chat_endpoint.router, prefix="/api/v1/chat")

    async def _db_override():
        return chacha_db

    async def _user_override():
        return User(id=USER, username=USER, is_admin=True)

    async def _no_expected_user():
        return None

    for module in (chat_sessions_endpoint, messages_endpoint, chat_endpoint):
        app.dependency_overrides[module.get_chacha_db_for_user] = _db_override
        app.dependency_overrides[module.get_request_user] = _user_override
        app.dependency_overrides[module.require_expected_user] = _no_expected_user
    for module in (chat_sessions_endpoint, messages_endpoint):
        monkeypatch.setattr(
            module,
            "get_active_server_origin_sync_service_for_user",
            lambda user_id: sync_service,
            raising=False,
        )
        monkeypatch.setattr(module, "get_character_rate_limiter", lambda: _NoopCharacterRateLimiter())
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    chacha_db: CharactersRAGDB,
    sync_service: SyncV2Service,
) -> TestClient:
    return _client(monkeypatch, chacha_db, sync_service)


def _dataset_id(service: SyncV2Service) -> str:
    return service.profile(user_id=USER).active_dataset_id or ""


def _envelopes(service: SyncV2Service) -> list[SyncEnvelope]:
    return service.store.list_envelopes_after(_dataset_id(service), 0, domains=CHAT_DOMAINS, limit=100)


def _log(service: SyncV2Service) -> list[tuple[str, str, str, str]]:
    return [(item.domain, item.operation, item.object_id, item.apply_status) for item in _envelopes(service)]


def _create_chat(client: TestClient, body: dict[str, Any] | None = None, chat_id: str = CHAT_ID):
    return client.post("/api/v1/chats/", json={"id": chat_id, **(CHAT_BODY if body is None else body)})


def _capture(client: TestClient, chat_id: str, cursor: dict[str, Any] | None = None, **view: Any):
    return client.post(
        f"/api/v1/chat/conversations/{chat_id}/history/selection",
        json={
            "purpose": "send",
            "view": {
                "view_session_id": "view-one",
                "conversation_id": chat_id,
                "interpretation": {"kind": "parent_graph_v1"},
                "cursor": cursor or {"kind": "empty"},
                "selection_revision": 1,
                **view,
            },
        },
    )


def _selection(client: TestClient, chat_id: str, cursor: dict[str, Any] | None = None) -> dict[str, Any]:
    response = _capture(client, chat_id, cursor)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "captured", body
    return resolve_history_selection(body["snapshot"], body["view"], "send", "client-context")["selection"]


def _admit(client: TestClient, chat_id: str, selection: dict[str, Any], *, message_id: str = INPUT_ID, content: str = "hello"):
    return client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={"id": message_id, "role": "user", "content": content, "tldw_history_selection_v1": selection},
    )


def _reference(admission: dict[str, Any]) -> dict[str, Any]:
    keys = ("version", "owner_key", "conversation_id", "input_message_id", "input_message_revision", "selection_digest")
    return {key: admission[key] for key in keys}


def _settle(client: TestClient, chat_id: str, reference: dict[str, Any], *, message_id: str = REPLY_ID, content: str = "hi"):
    return client.post(
        f"/api/v1/chats/{chat_id}/messages",
        json={"id": message_id, "role": "assistant", "content": content, "tldw_history_admission_v1": reference},
    )


def _send_turn(client: TestClient, chat_id: str = CHAT_ID) -> dict[str, Any]:
    """Create the chat with a client id, then capture, admit and settle one turn."""
    created = _create_chat(client, chat_id=chat_id)
    assert created.status_code == 201, created.text
    admitted = _admit(client, chat_id, _selection(client, chat_id))
    assert admitted.status_code == 201, admitted.text
    reference = _reference(admitted.json()["tldw_history_admission_v1"])
    settled = _settle(client, chat_id, reference)
    assert settled.status_code == 201, settled.text
    return reference


def _pull(service: SyncV2Service, device_id: str = "offline-device") -> list[SyncEnvelope]:
    return service.pull(
        user_id=USER,
        dataset_id=_dataset_id(service),
        device_id=device_id,
        cursor="0",
        domains=CHAT_DOMAINS,
    ).envelopes


def _apply_to_replica(replica: CharactersRAGDB, envelopes: list[SyncEnvelope]) -> None:
    """Project pulled envelopes the way another device would, in cursor order."""
    for envelope in envelopes:
        payload = envelope.payload
        if (envelope.domain, envelope.operation) == ("chat.conversation", "upsert"):
            replica.upsert_conversation_from_sync(
                conversation_id=envelope.object_id,
                title=payload.get("title"),
                sync_client_id=str(replica.client_id),
                object_revision=envelope.object_revision or 1,
                object_hash=envelope.payload_hash or "",
                root_id=payload.get("root_id"),
                assistant_kind=payload.get("assistant_kind"),
                assistant_id=payload.get("assistant_id"),
                character_id=payload.get("character_id"),
                state=payload.get("state"),
                source=payload.get("source"),
                scope_type=payload.get("scope_type"),
                workspace_id=payload.get("workspace_id"),
            )
        elif (envelope.domain, envelope.operation) == ("chat.message", "append"):
            replica.append_message_from_sync(
                stable_message_id=envelope.object_id,
                conversation_id=payload["conversation_id"],
                sender=payload["sender"],
                content=payload.get("content"),
                timestamp=payload.get("timestamp"),
                sync_client_id=str(replica.client_id),
                object_revision=envelope.object_revision or 1,
                payload_hash=envelope.payload_hash or "",
                parent_message_id=payload.get("parent_message_id"),
            )
        else:
            raise AssertionError(f"unexpected envelope {envelope.domain}/{envelope.operation}")


# ---------------------------------------------------------------------------
# Capture, admission and settlement
# ---------------------------------------------------------------------------


def test_capture_succeeds_and_records_nothing(client: TestClient, sync_service: SyncV2Service) -> None:
    assert _create_chat(client).status_code == 201
    before = _log(sync_service)

    response = _capture(client, CHAT_ID)

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "captured"
    assert _log(sync_service) == before


def test_capture_of_a_character_chat_is_still_refused(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A character chat takes server-settled turns, which a Sync v2 owner cannot use yet.

    Capture is the handshake, so it refuses as before and the client dispatches
    nothing. Without a profile the same chat captures.
    """
    character_id = chacha_db.add_character_card({"name": "Guide"})
    created = client.post(
        "/api/v1/chats/", json={"id": CHAT_ID, "character_id": character_id, "title": "With a character"}
    )
    assert created.status_code == 201, created.text
    before = _log(sync_service)

    refused = _capture(client, CHAT_ID)

    assert refused.status_code == 409, refused.text
    assert refused.json()["detail"] == {"status": "unsupported_history_capability", "code": "sync_owner_unsupported"}
    assert _log(sync_service) == before
    without_profile = _capture(_client(monkeypatch, chacha_db, None), CHAT_ID)
    assert without_profile.status_code == 200, without_profile.text
    assert without_profile.json()["status"] == "captured"


def test_capture_admit_and_settle_emit_one_applied_envelope_each(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)

    assert _log(sync_service) == [
        ("chat.conversation", "upsert", CHAT_ID, "applied"),
        ("chat.message", "append", INPUT_ID, "applied"),
        ("chat.message", "append", REPLY_ID, "applied"),
    ]
    conversation, user_message, assistant_message = _envelopes(sync_service)
    assert {item.device_id for item in (conversation, user_message, assistant_message)} == {SERVER_ORIGIN_DEVICE_ID}
    assert conversation.payload["title"] == CHAT_BODY["title"]
    assert conversation.payload["scope_type"] == "global"
    stored_input = chacha_db.get_message_by_id(INPUT_ID)
    assert user_message.parent_id == CHAT_ID
    assert user_message.payload == {
        "conversation_id": CHAT_ID,
        "parent_message_id": None,
        "sender": "user",
        "content": "hello",
        "timestamp": stored_input["timestamp"],
        "client_id": USER,
    }
    assert assistant_message.parent_id == CHAT_ID
    assert assistant_message.payload["parent_message_id"] == INPUT_ID
    assert assistant_message.payload["sender"] == "assistant"
    assert assistant_message.payload["content"] == "hi"
    # The native owner provenance stays on the server row and out of the envelope.
    assert chacha_db.get_message_by_id(REPLY_ID)["parent_message_id"] == INPUT_ID
    for envelope in (user_message, assistant_message):
        assert "history_admission" not in str(envelope.payload)
        assert "selection_digest" not in str(envelope.payload) + str(envelope.routing_metadata)
        state = sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", envelope.object_id)
        assert state is not None
        assert (state.object_revision, state.object_hash, state.latest_server_cursor) == (
            1,
            envelope.payload_hash,
            envelope.server_cursor,
        )


def test_second_turn_extends_the_first_and_is_published_in_order(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _send_turn(client)

    selection = _selection(client, CHAT_ID, {"kind": "after_message", "message_id": REPLY_ID})
    admitted = _admit(client, CHAT_ID, selection, message_id="second-input", content="and then?")
    assert admitted.status_code == 201, admitted.text
    settled = _settle(
        client,
        CHAT_ID,
        _reference(admitted.json()["tldw_history_admission_v1"]),
        message_id="second-reply",
        content="then this",
    )
    assert settled.status_code == 201, settled.text

    messages = [item for item in _envelopes(sync_service) if item.domain == "chat.message"]
    assert [(item.object_id, item.payload["parent_message_id"]) for item in messages] == [
        (INPUT_ID, None),
        (REPLY_ID, INPUT_ID),
        ("second-input", REPLY_ID),
        ("second-reply", "second-input"),
    ]


def test_second_device_sees_the_conversation_and_messages(
    tmp_path: Path,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)

    pulled = _pull(sync_service)

    assert [(item.domain, item.operation, item.object_id) for item in pulled] == [
        ("chat.conversation", "upsert", CHAT_ID),
        ("chat.message", "append", INPUT_ID),
        ("chat.message", "append", REPLY_ID),
    ]
    replica = CharactersRAGDB(db_path=str(tmp_path / "replica" / "ChaChaNotes.db"), client_id="offline-device")
    try:
        _apply_to_replica(replica, pulled)
        conversation = replica.get_conversation_by_id(CHAT_ID)
        assert conversation is not None
        assert conversation["title"] == CHAT_BODY["title"]
        replicated = replica.get_messages_for_conversation(CHAT_ID)
        original = chacha_db.get_messages_for_conversation(CHAT_ID)
        fields = ("id", "conversation_id", "parent_message_id", "sender", "content", "timestamp")
        assert [{key: row[key] for key in fields} for row in replicated] == [
            {key: row[key] for key in fields} for row in original
        ]
        assert [row["id"] for row in replicated] == [INPUT_ID, REPLY_ID]
    finally:
        replica.close_all_connections()


def test_repeated_admission_and_settlement_replay_without_new_envelopes(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    first = _admit(client, CHAT_ID, selection)
    replayed = _admit(client, CHAT_ID, selection)
    assert first.status_code == replayed.status_code == 201, replayed.text
    assert replayed.json()["tldw_history_admission_v1"] == first.json()["tldw_history_admission_v1"]
    reference = _reference(first.json()["tldw_history_admission_v1"])
    for _ in range(2):
        settled = _settle(client, CHAT_ID, reference)
        assert settled.status_code == 201, settled.text

    assert _log(sync_service) == [
        ("chat.conversation", "upsert", CHAT_ID, "applied"),
        ("chat.message", "append", INPUT_ID, "applied"),
        ("chat.message", "append", REPLY_ID, "applied"),
    ]
    assert chacha_db.count_messages_for_conversation(CHAT_ID) == 2


def test_refused_admission_writes_no_message_and_no_envelope(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    assert _admit(client, CHAT_ID, selection).status_code == 201
    before = _log(sync_service)

    # The same message id with different content is a conflict, not a second message.
    conflict = _admit(client, CHAT_ID, selection, content="something else")
    # A forged admission reference cannot settle.
    forged = _settle(client, CHAT_ID, {
        "version": 1,
        "owner_key": selection["owner_key"],
        "conversation_id": CHAT_ID,
        "input_message_id": INPUT_ID,
        "input_message_revision": "1",
        "selection_digest": "forged",
    })

    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"] == {"status": "stale_selection", "code": "message_id_conflict"}
    assert forged.status_code == 409, forged.text
    assert _log(sync_service) == before
    assert chacha_db.count_messages_for_conversation(CHAT_ID) == 1


def test_settlement_publishes_an_input_whose_capture_was_lost(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A lost capture heals on the retried request, and settlement never outruns its input."""
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    original_insert = SyncDatabase.insert_envelope

    def failing_insert(self, envelope, **kwargs):
        raise SyncStoreError("append is unavailable")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", failing_insert)
    lost = _admit(client, CHAT_ID, selection)
    assert lost.status_code == 503, lost.text
    assert lost.json()["detail"]["error_code"] == "sync_server_origin_append_failed"
    # The owner transaction committed before the capture failed.
    assert chacha_db.get_message_by_id(INPUT_ID) is not None
    assert [item[2] for item in _log(sync_service)] == [CHAT_ID]

    monkeypatch.setattr(SyncDatabase, "insert_envelope", original_insert)
    # The client could not read the admission, so it builds the reference it would have received.
    reference = {
        "version": 1,
        "owner_key": selection["owner_key"],
        "conversation_id": CHAT_ID,
        "input_message_id": INPUT_ID,
        "input_message_revision": "1",
        "selection_digest": selection["selection_digest"],
    }
    settled = _settle(client, CHAT_ID, reference)

    assert settled.status_code == 201, settled.text
    assert _log(sync_service) == [
        ("chat.conversation", "upsert", CHAT_ID, "applied"),
        ("chat.message", "append", INPUT_ID, "applied"),
        ("chat.message", "append", REPLY_ID, "applied"),
    ]


def test_retried_admission_publishes_a_message_whose_capture_was_lost(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    original_insert = SyncDatabase.insert_envelope

    def failing_insert(self, envelope, **kwargs):
        raise SyncStoreError("append is unavailable")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", failing_insert)
    assert _admit(client, CHAT_ID, selection).status_code == 503
    monkeypatch.setattr(SyncDatabase, "insert_envelope", original_insert)

    retried = _admit(client, CHAT_ID, selection)

    assert retried.status_code == 201, retried.text
    assert retried.json()["tldw_history_admission_v1"]["input_message_id"] == INPUT_ID
    assert _log(sync_service) == [
        ("chat.conversation", "upsert", CHAT_ID, "applied"),
        ("chat.message", "append", INPUT_ID, "applied"),
    ]


def _block_dataset(service: SyncV2Service) -> None:
    """Leave an accepted device envelope whose projection conflicted and is still unresolved."""
    dataset_id = _dataset_id(service)
    blocked = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-unresolved",
            domain="chat.message",
            operation="append",
            object_id="device-message",
            device_id="offline-device",
            client_sequence=1,
            object_revision=1,
            payload={"conversation_id": CHAT_ID, "sender": "user", "content": "Never projected"},
            payload_hash="sha256:unresolved",
            encryption_metadata={"policy": "server_trusted_v1"},
            status="accepted",
        )
    )
    blocked = service.store.mark_envelope_apply_status(
        blocked.server_cursor,
        apply_status="conflict",
        apply_error_code="message_stable_id_conflict",
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-unresolved",
            dataset_id=dataset_id,
            domain=blocked.domain,
            entity_id=blocked.object_id,
            conflict_type="message_stable_id_conflict",
            local_envelope_id=blocked.client_envelope_id,
            server_sequence=blocked.server_cursor,
        )
    )


def test_blocked_dataset_refuses_admission_before_any_write(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    _block_dataset(sync_service)

    refused = _admit(client, CHAT_ID, selection)

    assert refused.status_code == 503, refused.text
    assert refused.json()["detail"]["error_code"] == "sync_server_origin_append_failed"
    assert chacha_db.get_message_by_id(INPUT_ID) is None
    assert chacha_db.count_messages_for_conversation(CHAT_ID) == 0


def test_blocked_dataset_still_replays_what_is_already_published(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    """A blocked dataset takes no new history; idempotent retries add none and keep answering."""
    created = _create_chat(client)
    selection = _selection(client, CHAT_ID)
    admitted = _admit(client, CHAT_ID, selection)
    assert admitted.status_code == 201, admitted.text
    _block_dataset(sync_service)
    before = _log(sync_service)

    replayed_admission = _admit(client, CHAT_ID, selection)
    replayed_create = _create_chat(client)
    new_reply = _settle(client, CHAT_ID, _reference(admitted.json()["tldw_history_admission_v1"]))

    assert replayed_admission.status_code == 201, replayed_admission.text
    assert replayed_admission.json()["tldw_history_admission_v1"] == admitted.json()["tldw_history_admission_v1"]
    assert replayed_create.status_code == 200, replayed_create.text
    assert replayed_create.json()["id"] == created.json()["id"]
    # The reply is new history, so it waits for the conflict to be resolved.
    assert new_reply.status_code == 503, new_reply.text
    assert _log(sync_service) == before


def test_versioned_image_message_is_refused_before_any_write(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    import base64

    assert _create_chat(client).status_code == 201
    selection = _selection(client, CHAT_ID)
    png = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32).decode("ascii")
    before = _log(sync_service)

    response = client.post(
        f"/api/v1/chats/{CHAT_ID}/messages",
        json={
            "id": INPUT_ID,
            "role": "user",
            "content": "with an image",
            "image_base64": png,
            "tldw_history_selection_v1": selection,
        },
    )

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["error_code"] == "sync_v2_binary_message_unsupported"
    assert chacha_db.count_messages_for_conversation(CHAT_ID) == 0
    assert _log(sync_service) == before


# ---------------------------------------------------------------------------
# The dataset stays healthy for other Sync v2 writers
# ---------------------------------------------------------------------------


def test_chat_delete_after_versioned_turn_tombstones_every_message(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)
    version = client.get(f"/api/v1/chats/{CHAT_ID}").json()["version"]

    deleted = client.delete(f"/api/v1/chats/{CHAT_ID}", params={"expected_version": version})

    assert deleted.status_code == 204, deleted.text
    assert _log(sync_service)[3:] == [
        ("chat.message", "tombstone", INPUT_ID, "applied"),
        ("chat.message", "tombstone", REPLY_ID, "applied"),
        ("chat.conversation", "tombstone", CHAT_ID, "applied"),
    ]
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    for message_id in (INPUT_ID, REPLY_ID):
        assert chacha_db.get_message_by_id(message_id, include_deleted=True)["deleted"] in (1, True)
        assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.message", message_id).deleted


def test_device_push_into_the_conversation_still_applies_after_versioned_turn(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _send_turn(client)

    pushed = sync_service.push(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id="offline-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=_dataset_id(sync_service),
                client_envelope_id="env-device-follow-up",
                domain="chat.message",
                operation="append",
                object_id="device-follow-up",
                device_id="offline-device",
                client_sequence=1,
                object_revision=1,
                parent_id=CHAT_ID,
                payload={
                    "conversation_id": CHAT_ID,
                    "parent_message_id": REPLY_ID,
                    "sender": "user",
                    "content": "Sent from the laptop",
                    "timestamp": "2026-10-04T12:30:00+00:00",
                },
                payload_hash="sha256:device-follow-up",
                encryption_metadata={"policy": "server_trusted_v1"},
            )
        ],
    )

    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    assert pushed.rejected == [] and pushed.conflicts == []
    assert chacha_db.get_message_by_id("device-follow-up")["parent_message_id"] == REPLY_ID
    assert sync_service.store.list_conflicts(_dataset_id(sync_service)) == []
    # The device's message has no owner provenance, so the next WebUI send starts with a
    # history review. That is the existing rule for unversioned rows, not a Sync failure.
    after_push = _capture(client, CHAT_ID)
    assert after_push.status_code == 200, after_push.text
    assert after_push.json()["status"] == "legacy_review_required"


def test_webui_continues_a_chat_the_device_wrote_as_one_parent_chain(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A device chat whose messages name their parents is a graph: no review, and the new turn extends it."""
    dataset_id = _dataset_id(sync_service)
    device_chat = "device-chat"

    def device_envelope(sequence: int, domain: str, operation: str, object_id: str, payload: dict[str, Any]):
        return SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id=f"env-{object_id}",
            domain=domain,
            operation=operation,
            object_id=object_id,
            device_id="offline-device",
            client_sequence=sequence,
            object_revision=1,
            parent_id=None if domain == "chat.conversation" else device_chat,
            payload=payload,
            payload_hash=f"sha256:{object_id}",
            encryption_metadata={"policy": "server_trusted_v1"},
        )

    pushed = sync_service.push(
        user_id=USER,
        dataset_id=dataset_id,
        device_id="offline-device",
        envelopes=[
            device_envelope(1, "chat.conversation", "upsert", device_chat, {"title": "Started on the laptop"}),
            device_envelope(2, "chat.message", "append", "device-question", {
                "conversation_id": device_chat, "parent_message_id": None, "sender": "user",
                "content": "question", "timestamp": "2026-10-04T12:00:00+00:00"}),
            device_envelope(3, "chat.message", "append", "device-answer", {
                "conversation_id": device_chat, "parent_message_id": "device-question", "sender": "assistant",
                "content": "answer", "timestamp": "2026-10-04T12:00:05+00:00"}),
        ],
    )
    assert [item.apply_status for item in pushed.accepted] == ["applied"] * 3, pushed

    selection = _selection(client, device_chat, {"kind": "after_message", "message_id": "device-answer"})
    assert [row["id"] for row in selection["messages"]] == ["device-question", "device-answer"]
    admitted = _admit(client, device_chat, selection)
    assert admitted.status_code == 201, admitted.text
    settled = _settle(client, device_chat, _reference(admitted.json()["tldw_history_admission_v1"]))
    assert settled.status_code == 201, settled.text

    published = [item for item in _envelopes(sync_service) if item.device_id == SERVER_ORIGIN_DEVICE_ID]
    assert [(item.object_id, item.payload["parent_message_id"], item.parent_id) for item in published] == [
        (INPUT_ID, "device-answer", device_chat),
        (REPLY_ID, INPUT_ID, device_chat),
    ]
    assert chacha_db.count_messages_for_conversation(device_chat) == 4


def test_device_push_reusing_a_published_message_id_conflicts_without_blocking_the_dataset(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The published message is the object's head, so a reused id is a push conflict, not a second row."""
    _send_turn(client)

    pushed = sync_service.push(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id="offline-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=_dataset_id(sync_service),
                client_envelope_id="env-device-reused-id",
                domain="chat.message",
                operation="append",
                object_id=INPUT_ID,
                device_id="offline-device",
                client_sequence=1,
                object_revision=1,
                parent_id=CHAT_ID,
                payload={"conversation_id": CHAT_ID, "sender": "user", "content": "A different message"},
                payload_hash="sha256:device-reused-id",
                encryption_metadata={"policy": "server_trusted_v1"},
            )
        ],
    )

    assert pushed.accepted == []
    assert [item.client_envelope_id for item in pushed.conflicts] == ["env-device-reused-id"]
    assert chacha_db.get_message_by_id(INPUT_ID)["content"] == "hello"
    assert chacha_db.count_messages_for_conversation(CHAT_ID) == 2
    # A push-time conflict is not a projection conflict: the dataset still takes new history.
    assert sync_service.store.get_unresolved_materialization_conflict(_dataset_id(sync_service)) is None
    selection = _selection(client, CHAT_ID, {"kind": "after_message", "message_id": REPLY_ID})
    assert _admit(client, CHAT_ID, selection, message_id="after-collision").status_code == 201


def test_legacy_projection_can_be_confirmed_and_used(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Messages another device appended without parents need a reviewed path before the next send."""
    assert _create_chat(client).status_code == 201
    for message_id in ("device-one", "device-two"):
        sync_service.push(
            user_id=USER,
            dataset_id=_dataset_id(sync_service),
            device_id="offline-device",
            envelopes=[
                SyncEnvelopeCreate(
                    dataset_id=_dataset_id(sync_service),
                    client_envelope_id=f"env-{message_id}",
                    domain="chat.message",
                    operation="append",
                    object_id=message_id,
                    device_id="offline-device",
                    object_revision=1,
                    parent_id=CHAT_ID,
                    payload={"conversation_id": CHAT_ID, "sender": "user", "content": message_id},
                    payload_hash=f"sha256:{message_id}",
                    encryption_metadata={"policy": "server_trusted_v1"},
                )
            ],
        )
    review = _capture(client, CHAT_ID)
    assert review.status_code == 200, review.text
    assert review.json()["status"] == "legacy_review_required"
    snapshot = review.json()["snapshot"]

    confirmed = client.post(
        f"/api/v1/chat/conversations/{CHAT_ID}/history/legacy-projection",
        json={
            "confirmation": {
                "version": 1,
                "projection_id": "reviewed-under-sync",
                "owner_key": snapshot["owner_key"],
                "conversation_id": CHAT_ID,
                "source_digest": snapshot["source_digest"],
                "fences": snapshot["fences"],
                "source_members": [{"id": row["id"], "revision": row["revision"]} for row in snapshot["nodes"]],
                "ordered_path_ids": ["device-one", "device-two"],
                "cursor": {"kind": "after_message", "message_id": "device-two"},
                "selection_revision": 2,
            }
        },
    )

    assert confirmed.status_code == 200, confirmed.text
    reviewed = _capture(
        client,
        CHAT_ID,
        {"kind": "after_message", "message_id": "device-two"},
        interpretation={"kind": "legacy_linear_v1", "projection_id": "reviewed-under-sync"},
    )
    assert reviewed.status_code == 200, reviewed.text
    selection = resolve_history_selection(
        reviewed.json()["snapshot"], reviewed.json()["view"], "send", "client-context"
    )["selection"]
    admitted = _admit(client, CHAT_ID, selection)
    assert admitted.status_code == 201, admitted.text
    appended = _envelopes(sync_service)[-1]
    assert (appended.object_id, appended.payload["parent_message_id"]) == (INPUT_ID, "device-two")
    # The reviewed projection is this owner's interpretation; it is not a Sync object.
    assert {item.object_id for item in _envelopes(sync_service)} == {CHAT_ID, "device-one", "device-two", INPUT_ID}


# ---------------------------------------------------------------------------
# Client-id chat creation (D7 P3)
# ---------------------------------------------------------------------------


def test_client_id_create_publishes_the_conversation_and_stores_the_fingerprint(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    response = _create_chat(client)

    assert response.status_code == 201, response.text
    assert response.json()["id"] == CHAT_ID
    assert "Idempotency-Replayed" not in response.headers
    row = chacha_db.get_conversation_by_id(CHAT_ID)
    assert row["client_id"] == USER
    assert isinstance(row["create_request_fingerprint"], str) and len(row["create_request_fingerprint"]) == 64
    (envelope,) = _envelopes(sync_service)
    assert (envelope.domain, envelope.operation, envelope.object_id, envelope.apply_status) == (
        "chat.conversation",
        "upsert",
        CHAT_ID,
        "applied",
    )
    assert envelope.payload["title"] == CHAT_BODY["title"]
    assert envelope.payload["root_id"] == CHAT_ID
    # The fingerprint is server-internal: it never reaches another device.
    assert row["create_request_fingerprint"] not in str(envelope.payload) + str(envelope.routing_metadata)
    assert "create_request_fingerprint" not in str(envelope.payload) + str(envelope.routing_metadata)
    assert sync_service.store.get_object_state(_dataset_id(sync_service), "chat.conversation", CHAT_ID) is not None


def test_client_id_create_replays_with_200_and_one_envelope(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    first = _create_chat(client)
    second = _create_chat(client)

    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert second.headers["Idempotency-Replayed"] == "true"
    assert second.json()["id"] == first.json()["id"] == CHAT_ID
    assert second.json()["created_at"] == first.json()["created_at"]
    assert _log(sync_service) == [("chat.conversation", "upsert", CHAT_ID, "applied")]
    count = chacha_db.execute_query("SELECT COUNT(*) AS n FROM conversations", read_only=True).fetchone()["n"]
    assert int(count) == 1


def test_client_id_replay_survives_a_rename_through_sync(client: TestClient, sync_service: SyncV2Service) -> None:
    created = _create_chat(client)
    renamed = client.put(
        f"/api/v1/chats/{CHAT_ID}",
        params={"expected_version": created.json()["version"]},
        json={"title": "Renamed"},
    )
    assert renamed.status_code == 200, renamed.text

    replay = _create_chat(client)

    assert replay.status_code == 200, replay.text
    assert replay.json()["title"] == "Renamed"
    assert _log(sync_service) == [
        ("chat.conversation", "upsert", CHAT_ID, "applied"),
        ("chat.conversation", "upsert", CHAT_ID, "applied"),
    ]


def test_rename_through_sync_keeps_a_plain_chat_plain(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """An upsert that names no assistant must not turn the chat into a persona chat.

    The WebUI reads the stored identity to choose how to send, and the history
    context digest covers it. A chat created without an assistant stays that way
    when it is renamed, and the turn captured before the rename still settles.
    """
    created = _create_chat(client)
    admitted = _admit(client, CHAT_ID, _selection(client, CHAT_ID))
    assert admitted.status_code == 201, admitted.text
    before = chacha_db.get_conversation_by_id(CHAT_ID)
    assert (before["assistant_kind"], before["assistant_id"], before["character_id"]) == (None, None, None)

    renamed = client.put(
        f"/api/v1/chats/{CHAT_ID}",
        params={"expected_version": created.json()["version"]},
        json={"title": "Renamed"},
    )

    assert renamed.status_code == 200, renamed.text
    assert (renamed.json()["assistant_kind"], renamed.json()["assistant_id"]) == (None, None)
    after = chacha_db.get_conversation_by_id(CHAT_ID)
    assert after["title"] == "Renamed"
    assert (after["assistant_kind"], after["assistant_id"], after["character_id"]) == (None, None, None)
    settled = _settle(client, CHAT_ID, _reference(admitted.json()["tldw_history_admission_v1"]))
    assert settled.status_code == 201, settled.text
    next_selection = _selection(client, CHAT_ID, {"kind": "after_message", "message_id": REPLY_ID})
    assert [row["id"] for row in next_selection["messages"]] == [INPUT_ID, REPLY_ID]


def test_device_upsert_without_an_assistant_still_gets_the_sync_placeholder_on_a_new_chat(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Only an existing plain chat is kept plain; a chat first seen through Sync is projected as before."""
    pushed = sync_service.push(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id="offline-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=_dataset_id(sync_service),
                client_envelope_id="env-device-plain-chat",
                domain="chat.conversation",
                operation="upsert",
                object_id="device-plain-chat",
                device_id="offline-device",
                client_sequence=1,
                object_revision=1,
                payload={"title": "From the laptop"},
                payload_hash="sha256:device-plain-chat",
                encryption_metadata={"policy": "server_trusted_v1"},
            )
        ],
    )

    assert [item.apply_status for item in pushed.accepted] == ["applied"]
    row = chacha_db.get_conversation_by_id("device-plain-chat")
    assert (row["assistant_kind"], row["assistant_id"]) == ("persona", "sync-v2")


def test_client_id_with_a_different_request_is_409_and_changes_nothing(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    assert _create_chat(client).status_code == 201
    before = _log(sync_service)

    conflict = _create_chat(client, {**CHAT_BODY, "title": "A different chat"})

    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["error_code"] == "chat_id_conflict"
    assert CHAT_BODY["title"] not in conflict.text
    assert chacha_db.get_conversation_by_id(CHAT_ID)["title"] == CHAT_BODY["title"]
    assert _log(sync_service) == before


def test_client_id_of_a_chat_created_without_one_is_409(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """A chat another device (or an id-less create) made is never replayed as a client-id create."""
    sync_service.push(
        user_id=USER,
        dataset_id=_dataset_id(sync_service),
        device_id="offline-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=_dataset_id(sync_service),
                client_envelope_id="env-device-chat",
                domain="chat.conversation",
                operation="upsert",
                object_id=CHAT_ID,
                device_id="offline-device",
                client_sequence=1,
                object_revision=1,
                payload={"title": CHAT_BODY["title"], "state": "in-progress", "source": "webui-chat"},
                payload_hash="sha256:device-chat",
                encryption_metadata={"policy": "server_trusted_v1"},
            )
        ],
    )
    assert chacha_db.get_conversation_by_id(CHAT_ID) is not None
    before = _log(sync_service)

    conflict = _create_chat(client)

    assert conflict.status_code == 409, conflict.text
    assert conflict.json()["detail"]["error_code"] == "chat_id_conflict"
    assert _log(sync_service) == before


def test_client_id_create_after_trash_is_410(client: TestClient, sync_service: SyncV2Service) -> None:
    created = _create_chat(client)
    assert created.status_code == 201, created.text
    trashed = client.delete(f"/api/v1/chats/{CHAT_ID}", params={"expected_version": created.json()["version"]})
    assert trashed.status_code == 204, trashed.text
    before = _log(sync_service)
    assert before[-1] == ("chat.conversation", "tombstone", CHAT_ID, "applied")

    gone = _create_chat(client)

    assert gone.status_code == 410, gone.text
    assert gone.json()["detail"]["error_code"] == "chat_deleted"
    assert _log(sync_service) == before


def test_replayed_create_publishes_a_chat_whose_capture_was_lost(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    original_insert = SyncDatabase.insert_envelope

    def failing_insert(self, envelope, **kwargs):
        raise SyncStoreError("append is unavailable")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", failing_insert)
    lost = _create_chat(client)
    assert lost.status_code == 503, lost.text
    assert lost.json()["detail"]["error_code"] == "sync_server_origin_append_failed"
    assert chacha_db.get_conversation_by_id(CHAT_ID) is not None
    assert _log(sync_service) == []

    monkeypatch.setattr(SyncDatabase, "insert_envelope", original_insert)
    replay = _create_chat(client)

    assert replay.status_code == 200, replay.text
    assert replay.headers["Idempotency-Replayed"] == "true"
    assert _log(sync_service) == [("chat.conversation", "upsert", CHAT_ID, "applied")]


def test_concurrent_duplicate_creates_produce_one_chat_and_one_envelope(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Every request passes the pre-insert read first; the primary key and the fence pick one winner."""
    import threading
    import time

    workers = 6
    barrier = threading.Barrier(workers, timeout=30)
    real = chacha_db.get_conversation_by_id
    reads = {"count": 0}
    reads_lock = threading.Lock()

    def gated(conversation_id: str, include_deleted: bool = False) -> dict | None:
        row = real(conversation_id, include_deleted=include_deleted)
        if conversation_id == CHAT_ID:
            with reads_lock:
                reads["count"] += 1
                is_pre_insert_read = reads["count"] <= workers
            # The first read of each request is the replay lookup. Hold them all there.
            if is_pre_insert_read:
                barrier.wait()
        return row

    monkeypatch.setattr(chacha_db, "get_conversation_by_id", gated)
    results: list[Any] = [None] * workers

    def worker(index: int) -> None:
        # A request that could not take the Sync fence in time is refused before it writes
        # (503, retryable), so a client retries it.
        for _attempt in range(40):
            results[index] = _create_chat(client)
            if results[index].status_code != 503:
                return
            time.sleep(0.05)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    statuses = sorted(response.status_code for response in results)
    assert statuses == [200] * (workers - 1) + [201], [response.text for response in results]
    assert {response.json()["id"] for response in results} == {CHAT_ID}
    count = chacha_db.execute_query("SELECT COUNT(*) AS n FROM conversations", read_only=True).fetchone()["n"]
    assert int(count) == 1
    assert _log(sync_service) == [("chat.conversation", "upsert", CHAT_ID, "applied")]


def test_client_id_character_chat_is_published_without_a_seeded_greeting(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """As for an id-less create under Sync v2: the chat is created bare, so nothing unpublished is written."""
    character_id = chacha_db.add_character_card({"name": "Guide", "first_message": "Welcome."})

    created = client.post(
        "/api/v1/chats/",
        params={"seed_first_message": True},
        json={"id": CHAT_ID, "character_id": character_id, "title": "With a character"},
    )
    replayed = client.post(
        "/api/v1/chats/",
        params={"seed_first_message": True},
        json={"id": CHAT_ID, "character_id": character_id, "title": "With a character"},
    )

    assert created.status_code == 201, created.text
    assert created.headers["X-Chat-Seed-Status"] == "no_greeting"
    assert replayed.status_code == 200, replayed.text
    row = chacha_db.get_conversation_by_id(CHAT_ID)
    assert (row["character_id"], row["assistant_kind"]) == (character_id, "character")
    assert chacha_db.count_messages_for_conversation(CHAT_ID) == 0
    (envelope,) = _envelopes(sync_service)
    assert (envelope.object_id, envelope.apply_status) == (CHAT_ID, "applied")
    assert envelope.payload["character_id"] == character_id
    assert envelope.payload["assistant_kind"] == "character"


def test_workspace_client_id_create_stays_direct(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """Workspace chats are not part of the personal dataset, with or without a client id."""
    chacha_db.upsert_workspace("workspace-1", "Workspace")
    body = {"id": CHAT_ID, "title": "Workspace chat", "scope_type": "workspace", "workspace_id": "workspace-1"}

    created = client.post("/api/v1/chats/", json=body)
    replayed = client.post("/api/v1/chats/", json=body)

    assert created.status_code == 201, created.text
    assert replayed.status_code == 200, replayed.text
    assert chacha_db.get_conversation_by_id(CHAT_ID)["scope_type"] == "workspace"
    assert _log(sync_service) == []


def test_create_without_a_client_id_still_goes_through_the_sync_materializer(
    client: TestClient,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    response = client.post("/api/v1/chats/", json=CHAT_BODY)

    assert response.status_code == 201, response.text
    chat_id = response.json()["id"]
    assert chat_id != CHAT_ID
    assert chacha_db.get_conversation_by_id(chat_id)["create_request_fingerprint"] is None
    assert _log(sync_service) == [("chat.conversation", "upsert", chat_id, "applied")]


# ---------------------------------------------------------------------------
# Without Sync v2
# ---------------------------------------------------------------------------


def test_without_sync_v2_the_same_turn_writes_no_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    """The same requests with no active profile behave as before and never touch the Sync store."""
    inactive = _client(monkeypatch, chacha_db, None)

    def no_sync_writes(*_args, **_kwargs):
        raise AssertionError("an inactive Sync profile must not be written to")

    monkeypatch.setattr(SyncDatabase, "insert_envelope", no_sync_writes)
    monkeypatch.setattr(SyncDatabase, "upsert_object_state", no_sync_writes)

    reference = _send_turn(inactive)
    replay = _create_chat(inactive)

    assert replay.status_code == 200, replay.text
    assert reference["input_message_id"] == INPUT_ID
    assert chacha_db.get_message_by_id(REPLY_ID)["parent_message_id"] == INPUT_ID
    assert _log(sync_service) == []
