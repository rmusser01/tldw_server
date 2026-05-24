from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.materializers.chat import (
    ChatConversationMaterializer,
    ChatMessageMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@pytest.fixture()
def chacha_db(tmp_path: Path) -> CharactersRAGDB:
    return CharactersRAGDB(
        db_path=str(tmp_path / "ChaChaNotes.db"),
        client_id="server-user-1",
    )


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "Sync_v2.db"))


@pytest.fixture()
def sync_service(
    sync_store: SyncV2Store,
    chacha_db: CharactersRAGDB,
) -> SyncV2Service:
    registry = SyncAdapterRegistry(
        [
            StaticSyncAdapter(domain="chat.conversation", supported_adapter_versions={1}),
            StaticSyncAdapter(domain="chat.message", supported_adapter_versions={1}),
        ]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={
            "chat.conversation": ChatConversationMaterializer(chacha_db),
            "chat.message": ChatMessageMaterializer(chacha_db),
        },
        clock=lambda: "2026-05-23T18:12:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            supported_domains=["chat.conversation", "chat.message"],
            operations={
                "chat.conversation": ["upsert", "tombstone"],
                "chat.message": ["append", "tombstone"],
            },
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["chat.conversation", "chat.message"],
    )
    return service


def _conversation_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-conv-create",
        "domain": "chat.conversation",
        "operation": "upsert",
        "object_id": "conv-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "object_revision": 1,
        "payload": {
            "title": "Planning chat",
            "assistant_kind": "persona",
            "assistant_id": "sync-assistant",
            "state": "active",
        },
        "payload_hash": "sha256:conv-v1",
        "created_at_client": "2026-05-23T18:12:44+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _message_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-msg-create",
        "domain": "chat.message",
        "operation": "append",
        "object_id": "msg-1",
        "device_id": "device-1",
        "client_sequence": 2,
        "schema_version": 1,
        "object_revision": 1,
        "payload": {
            "conversation_id": "conv-1",
            "sender": "user",
            "content": "First synced message",
            "timestamp": "2026-05-23T18:13:00+00:00",
        },
        "payload_hash": "sha256:msg-v1",
        "created_at_client": "2026-05-23T18:13:00+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _push_one(service: SyncV2Service, envelope: SyncEnvelopeCreate):
    return service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[envelope],
    )


def test_chat_conversation_upsert_update_and_tombstone_project_to_chacha(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    create = _push_one(sync_service, _conversation_envelope())

    assert [item.client_envelope_id for item in create.accepted] == ["env-conv-create"]
    created = chacha_db.get_conversation_by_id("conv-1")
    assert created is not None
    assert created["title"] == "Planning chat"
    assert created["assistant_kind"] == "persona"
    assert created["assistant_id"] == "sync-assistant"

    base = sync_service.store.get_object_state("dataset-1", "chat.conversation", "conv-1")
    assert base is not None
    update = _push_one(
        sync_service,
        _conversation_envelope(
            client_envelope_id="env-conv-update",
            client_sequence=2,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={
                "title": "Planning chat revised",
                "assistant_kind": "persona",
                "assistant_id": "sync-assistant",
                "state": "archived",
            },
            payload_hash="sha256:conv-v2",
        ),
    )

    assert [item.client_envelope_id for item in update.accepted] == ["env-conv-update"]
    updated = chacha_db.get_conversation_by_id("conv-1")
    assert updated is not None
    assert updated["title"] == "Planning chat revised"
    assert updated["state"] == "resolved"

    current = sync_service.store.get_object_state("dataset-1", "chat.conversation", "conv-1")
    assert current is not None
    tombstone = _push_one(
        sync_service,
        _conversation_envelope(
            client_envelope_id="env-conv-delete",
            client_sequence=3,
            operation="tombstone",
            base_server_cursor=current.latest_server_cursor,
            base_object_revision=current.object_revision,
            base_object_hash=current.object_hash,
            object_revision=3,
            payload={"deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:conv-delete",
            deleted=True,
        ),
    )

    assert [item.client_envelope_id for item in tombstone.accepted] == ["env-conv-delete"]
    assert chacha_db.get_conversation_by_id("conv-1") is None
    deleted = chacha_db.get_conversation_by_id("conv-1", include_deleted=True)
    assert deleted is not None
    assert bool(deleted["deleted"]) is True
    state = sync_service.store.get_object_state("dataset-1", "chat.conversation", "conv-1")
    assert state is not None
    assert state.deleted is True
    assert state.object_revision == 3


def test_stale_conversation_base_conflicts_without_overwriting_projection(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    base = sync_service.store.get_object_state("dataset-1", "chat.conversation", "conv-1")
    assert base is not None
    _push_one(
        sync_service,
        _conversation_envelope(
            client_envelope_id="env-conv-current",
            client_sequence=2,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"title": "Server winner", "assistant_kind": "persona", "assistant_id": "sync-assistant"},
            payload_hash="sha256:conv-current",
        ),
    )

    stale = _push_one(
        sync_service,
        _conversation_envelope(
            client_envelope_id="env-conv-stale",
            client_sequence=3,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=3,
            payload={"title": "Stale edit", "assistant_kind": "persona", "assistant_id": "sync-assistant"},
            payload_hash="sha256:conv-stale",
        ),
    )

    assert stale.accepted == []
    assert [item.client_envelope_id for item in stale.conflicts] == ["env-conv-stale"]
    conversation = chacha_db.get_conversation_by_id("conv-1")
    assert conversation is not None
    assert conversation["title"] == "Server winner"
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "whole_object_conflict"


def test_chat_message_append_dedupes_same_payload_and_conflicts_divergent_stable_id(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    first = _push_one(sync_service, _message_envelope())

    assert [item.client_envelope_id for item in first.accepted] == ["env-msg-create"]
    stored = chacha_db.get_message_by_id("msg-1")
    assert stored is not None
    assert stored["conversation_id"] == "conv-1"
    assert stored["content"] == "First synced message"
    metadata = chacha_db.get_message_metadata("msg-1")
    assert metadata is not None
    assert metadata["extra"]["sync_v2"]["stable_message_id"] == "msg-1"
    assert metadata["extra"]["sync_v2"]["payload_hash"] == "sha256:msg-v1"

    duplicate = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-duplicate",
            client_sequence=3,
        ),
    )

    assert [item.client_envelope_id for item in duplicate.accepted] == ["env-msg-duplicate"]
    assert chacha_db.count_messages_for_conversation("conv-1", include_deleted=True) == 1

    divergent = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-divergent",
            client_sequence=4,
            payload={
                "conversation_id": "conv-1",
                "sender": "assistant",
                "content": "Conflicting synced message",
                "timestamp": "2026-05-23T18:14:00+00:00",
            },
            payload_hash="sha256:msg-v2",
        ),
    )

    assert divergent.accepted == []
    assert [item.client_envelope_id for item in divergent.conflicts] == ["env-msg-divergent"]
    versions = chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=True)
    assert len(versions) == 2
    assert {item["content"] for item in versions} == {
        "First synced message",
        "Conflicting synced message",
    }
    assert any(item["id"] != "msg-1" for item in versions)
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "message_stable_id_conflict"
    assert conflicts[0].metadata["stable_message_id"] == "msg-1"


def test_divergent_message_conflicts_even_when_existing_metadata_is_missing(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    _push_one(sync_service, _message_envelope())
    base = sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1")
    assert base is not None
    chacha_db.execute_query("DELETE FROM message_metadata WHERE message_id = ?", ("msg-1",), commit=True)

    divergent = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-divergent-missing-meta",
            client_sequence=4,
            payload={
                "conversation_id": "conv-1",
                "sender": "assistant",
                "content": "Conflicting synced message",
                "timestamp": "2026-05-23T18:14:00+00:00",
            },
            payload_hash="sha256:msg-v2",
        ),
    )

    assert divergent.accepted == []
    assert [item.client_envelope_id for item in divergent.conflicts] == [
        "env-msg-divergent-missing-meta"
    ]
    assert sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1") == base
    versions = chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=True)
    assert len(versions) == 2
    assert any(item["id"] == "msg-1" for item in versions)
    assert any(item["id"] != "msg-1" for item in versions)


def test_message_metadata_write_failure_is_replayable_without_duplicate_rows(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _push_one(sync_service, _conversation_envelope())

    monkeypatch.setattr(chacha_db.message_store, "set_message_metadata_extra", lambda *args, **kwargs: False)
    first_attempt = _push_one(sync_service, _message_envelope())
    monkeypatch.undo()

    assert [item.client_envelope_id for item in first_attempt.accepted] == ["env-msg-create"]
    stored_envelope = next(
        item
        for item in sync_service.store.list_envelopes_for_entity(
            "dataset-1",
            "chat.message",
            entity_id="msg-1",
            limit=10,
        )
        if item.client_envelope_id == "env-msg-create"
    )
    assert stored_envelope.apply_status == "failed"
    assert sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1") is None
    assert chacha_db.get_message_by_id("msg-1") is not None
    assert chacha_db.get_message_metadata("msg-1") is None

    retry = _push_one(sync_service, _message_envelope())

    assert [item.client_envelope_id for item in retry.accepted] == ["env-msg-create"]
    assert sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1") is not None
    assert chacha_db.count_messages_for_conversation("conv-1", include_deleted=True) == 1
    metadata = chacha_db.get_message_metadata("msg-1")
    assert metadata is not None
    assert metadata["extra"]["sync_v2"]["payload_hash"] == "sha256:msg-v1"


def test_metadata_missing_physical_id_fallback_does_not_adopt_unrelated_local_message(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    chacha_db.add_message(
        {
            "id": "msg-1",
            "conversation_id": "conv-1",
            "sender": "assistant",
            "content": "Unrelated local message with the same physical ID",
            "timestamp": "2026-05-23T18:12:00+00:00",
        }
    )

    result = _push_one(sync_service, _message_envelope())

    assert result.accepted == []
    assert [item.client_envelope_id for item in result.conflicts] == ["env-msg-create"]
    assert sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1") is None
    assert chacha_db.get_message_metadata("msg-1") is None
    versions = chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=True)
    assert len(versions) == 2
    assert any(item["content"] == "Unrelated local message with the same physical ID" for item in versions)
    assert any(item["content"] == "First synced message" for item in versions)


def test_retry_after_failed_message_conflict_status_keeps_conflict(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    _push_one(sync_service, _message_envelope())
    base = sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1")
    assert base is not None

    original_mark = sync_service.store.mark_envelope_apply_status
    failed_once = False

    def _fail_first_conflict_mark(server_cursor: int, **kwargs):
        nonlocal failed_once
        if kwargs.get("apply_status") == "conflict" and not failed_once:
            failed_once = True
            raise RuntimeError("simulated conflict status failure")
        return original_mark(server_cursor, **kwargs)

    divergent = _message_envelope(
        client_envelope_id="env-msg-divergent",
        client_sequence=4,
        payload={
            "conversation_id": "conv-1",
            "sender": "assistant",
            "content": "Conflicting synced message",
            "timestamp": "2026-05-23T18:14:00+00:00",
        },
        payload_hash="sha256:msg-v2",
    )

    monkeypatch.setattr(sync_service.store, "mark_envelope_apply_status", _fail_first_conflict_mark)
    first_attempt = _push_one(sync_service, divergent)
    monkeypatch.undo()

    assert [item.client_envelope_id for item in first_attempt.accepted] == ["env-msg-divergent"]
    stored_envelope = next(
        item
        for item in sync_service.store.list_envelopes_for_entity(
            "dataset-1",
            "chat.message",
            entity_id="msg-1",
            limit=10,
        )
        if item.client_envelope_id == "env-msg-divergent"
    )
    assert stored_envelope.apply_status == "failed"
    assert len(chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=True)) == 2

    retry = _push_one(sync_service, divergent)

    assert retry.accepted == []
    assert [item.client_envelope_id for item in retry.conflicts] == ["env-msg-divergent"]
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "message_stable_id_conflict"
    assert sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1") == base


def test_message_tombstone_soft_deletes_message_without_deleting_conversation(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    _push_one(sync_service, _message_envelope())
    base = sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1")
    assert base is not None

    result = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-delete",
            client_sequence=3,
            operation="tombstone",
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"conversation_id": "conv-1", "deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:msg-delete",
            deleted=True,
        ),
    )

    assert [item.client_envelope_id for item in result.accepted] == ["env-msg-delete"]
    assert chacha_db.get_message_by_id("msg-1") is None
    deleted_message = chacha_db.get_message_by_id("msg-1", include_deleted=True)
    assert deleted_message is not None
    assert bool(deleted_message["deleted"]) is True
    conversation = chacha_db.get_conversation_by_id("conv-1")
    assert conversation is not None
    assert bool(conversation["deleted"]) is False


def test_message_tombstone_requires_matching_base_state(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    _push_one(sync_service, _message_envelope())

    result = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-delete-missing-base",
            client_sequence=3,
            operation="tombstone",
            object_revision=2,
            payload={"conversation_id": "conv-1", "deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:msg-delete",
            deleted=True,
        ),
    )

    assert result.accepted == []
    assert [item.client_envelope_id for item in result.conflicts] == [
        "env-msg-delete-missing-base"
    ]
    assert chacha_db.get_message_by_id("msg-1") is not None
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "message_base_conflict"


def test_message_tombstone_deletes_canonical_projection_when_conflict_sorts_first(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    _push_one(
        sync_service,
        _message_envelope(
            payload={
                "conversation_id": "conv-1",
                "sender": "user",
                "content": "Canonical synced message",
                "timestamp": "2026-05-23T10:00:00+00:00",
            },
        ),
    )
    base = sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1")
    assert base is not None
    _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-divergent-earlier",
            client_sequence=3,
            payload={
                "conversation_id": "conv-1",
                "sender": "assistant",
                "content": "Earlier conflicting message",
                "timestamp": "2026-05-23T09:00:00+00:00",
            },
            payload_hash="sha256:msg-v2",
        ),
    )
    versions_before = chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=True)
    assert len(versions_before) == 2
    assert versions_before[0]["content"] == "Earlier conflicting message"

    tombstone = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-delete-after-conflict",
            client_sequence=4,
            operation="tombstone",
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"conversation_id": "conv-1", "deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:msg-delete",
            deleted=True,
        ),
    )

    assert [item.client_envelope_id for item in tombstone.accepted] == [
        "env-msg-delete-after-conflict"
    ]
    assert chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=False) == []
    versions_after = chacha_db.get_messages_by_sync_stable_id("msg-1", include_deleted=True)
    assert len(versions_after) == 2
    assert all(bool(item["deleted"]) is True for item in versions_after)


def test_message_append_after_tombstone_conflicts_without_resurrecting_state(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _conversation_envelope())
    _push_one(sync_service, _message_envelope())
    base = sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1")
    assert base is not None
    _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-delete",
            client_sequence=3,
            operation="tombstone",
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"conversation_id": "conv-1", "deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:msg-delete",
            deleted=True,
        ),
    )
    deleted_state = sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1")
    assert deleted_state is not None
    assert deleted_state.deleted is True

    duplicate_old_append = _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-stale-append",
            client_sequence=4,
        ),
    )

    assert duplicate_old_append.accepted == []
    assert [item.client_envelope_id for item in duplicate_old_append.conflicts] == [
        "env-msg-stale-append"
    ]
    assert chacha_db.get_message_by_id("msg-1") is None
    assert sync_service.store.get_object_state("dataset-1", "chat.message", "msg-1") == deleted_state


def test_chat_materialization_conflict_is_hidden_from_normal_pull(
    sync_service: SyncV2Service,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Tablet",
        client_type="chatbook",
        device_id="device-2",
    )
    _push_one(sync_service, _conversation_envelope())
    _push_one(sync_service, _message_envelope())
    _push_one(
        sync_service,
        _message_envelope(
            client_envelope_id="env-msg-divergent",
            client_sequence=3,
            payload={
                "conversation_id": "conv-1",
                "sender": "assistant",
                "content": "Conflicting synced message",
                "timestamp": "2026-05-23T18:14:00+00:00",
            },
            payload_hash="sha256:msg-v2",
        ),
    )

    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=0,
        domains=["chat.conversation", "chat.message"],
    )

    assert [item.client_envelope_id for item in pulled.envelopes] == [
        "env-conv-create",
        "env-msg-create",
    ]
