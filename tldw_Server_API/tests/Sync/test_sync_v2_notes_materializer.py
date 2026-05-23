from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
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
        [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": NotesMaterializer(chacha_db)},
        clock=lambda: "2026-05-23T18:12:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            supported_domains=["notes.note"],
            operations={"notes.note": ["upsert", "tombstone"]},
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
        domains=["notes.note"],
    )
    return service


def _note_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-create",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "object_revision": 1,
        "payload": {
            "title": "Trip notes",
            "body": "Packed outline and research links.",
        },
        "payload_hash": "sha256:note-v1",
        "created_at_client": "2026-05-23T18:12:44+00:00",
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _push_one(
    service: SyncV2Service,
    envelope: SyncEnvelopeCreate,
):
    return service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[envelope],
    )


def test_clean_notes_note_upsert_creates_normal_chacha_note(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    result = _push_one(sync_service, _note_envelope())

    assert [item.client_envelope_id for item in result.accepted] == ["env-create"]
    assert result.conflicts == []
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Trip notes"
    assert note["content"] == "Packed outline and research links."
    state = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert state is not None
    assert state.object_revision == 1
    assert state.object_hash == "sha256:note-v1"
    assert state.deleted is False


def test_exact_retry_of_applied_create_does_not_rematerialize_or_conflict(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    envelope = _note_envelope()
    _push_one(sync_service, envelope)
    before = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert before is not None

    result = _push_one(sync_service, envelope)

    assert [item.client_envelope_id for item in result.accepted] == ["env-create"]
    assert result.conflicts == []
    assert sync_service.store.list_conflicts("dataset-1") == []
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "notes.note",
        entity_id="note-1",
        limit=10,
    )
    assert [(item.client_envelope_id, item.apply_status) for item in stored] == [
        ("env-create", "applied")
    ]
    assert sync_service.store.get_object_state("dataset-1", "notes.note", "note-1") == before
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Trip notes"
    assert note["content"] == "Packed outline and research links."


def test_update_with_matching_base_updates_projection_and_object_state(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None

    result = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-update",
            client_sequence=2,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={
                "title": "Trip notes revised",
                "content": "Updated outline.",
            },
            payload_hash="sha256:note-v2",
        ),
    )

    assert [item.client_envelope_id for item in result.accepted] == ["env-update"]
    assert result.conflicts == []
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Trip notes revised"
    assert note["content"] == "Updated outline."
    state = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert state is not None
    assert state.object_revision == 2
    assert state.object_hash == "sha256:note-v2"


def test_exact_retry_of_applied_update_does_not_rematerialize_or_conflict(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    update = _note_envelope(
        client_envelope_id="env-update",
        client_sequence=2,
        base_server_cursor=base.latest_server_cursor,
        base_object_revision=base.object_revision,
        base_object_hash=base.object_hash,
        object_revision=2,
        payload={
            "title": "Trip notes revised",
            "content": "Updated outline.",
        },
        payload_hash="sha256:note-v2",
    )
    _push_one(sync_service, update)
    before = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert before is not None

    result = _push_one(sync_service, update)

    assert [item.client_envelope_id for item in result.accepted] == ["env-update"]
    assert result.conflicts == []
    assert sync_service.store.list_conflicts("dataset-1") == []
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "notes.note",
        entity_id="note-1",
        limit=10,
    )
    assert sorted((item.client_envelope_id, item.apply_status) for item in stored) == [
        ("env-create", "applied"),
        ("env-update", "applied"),
    ]
    assert sync_service.store.get_object_state("dataset-1", "notes.note", "note-1") == before
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Trip notes revised"
    assert note["content"] == "Updated outline."


@pytest.mark.parametrize(
    ("base_revision", "base_hash"),
    [(1, "sha256:note-v1"), (2, "sha256:stale-hash")],
)
def test_stale_base_creates_whole_object_conflict_without_overwriting_projection(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    base_revision: int,
    base_hash: str,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-update-current",
            client_sequence=2,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"title": "Server winner", "body": "Keep this body."},
            payload_hash="sha256:note-v2",
        ),
    )
    current = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert current is not None

    result = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id=f"env-stale-{base_revision}",
            client_sequence=3,
            base_server_cursor=current.latest_server_cursor,
            base_object_revision=base_revision,
            base_object_hash=base_hash,
            object_revision=3,
            payload={"title": "Stale edit", "body": "Do not apply."},
            payload_hash=f"sha256:stale-{base_revision}",
        ),
    )

    assert result.accepted == []
    assert len(result.conflicts) == 1
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Server winner"
    assert note["content"] == "Keep this body."
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].conflict_type == "whole_object_conflict"
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "notes.note",
        entity_id="note-1",
        limit=10,
    )
    conflicted = next(item for item in stored if item.client_envelope_id.startswith("env-stale"))
    assert conflicted.apply_status == "conflict"


def test_materialization_conflict_is_not_returned_by_normal_pull(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Tablet",
        client_type="chatbook",
        device_id="device-2",
    )
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-update-current",
            client_sequence=2,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"title": "Server winner", "body": "Keep this body."},
            payload_hash="sha256:note-v2",
        ),
    )

    stale = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-stale",
            client_sequence=3,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=3,
            payload={"title": "Stale edit", "body": "Do not apply."},
            payload_hash="sha256:stale-v3",
        ),
    )
    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=0,
        domains=["notes.note"],
    )

    assert stale.accepted == []
    assert [item.client_envelope_id for item in stale.conflicts] == ["env-stale"]
    assert [item.client_envelope_id for item in pulled.envelopes] == [
        "env-create",
        "env-update-current",
    ]
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Server winner"


def test_pull_paginates_across_hidden_materialization_conflict(
    sync_service: SyncV2Service,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Tablet",
        client_type="chatbook",
        device_id="device-2",
    )
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    stale = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-stale",
            client_sequence=2,
            base_server_cursor=base.latest_server_cursor + 100,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"title": "Stale edit", "body": "Do not apply."},
            payload_hash="sha256:stale-v2",
        ),
    )
    visible_update = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-update-current",
            client_sequence=3,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"title": "Server winner", "body": "Keep this body."},
            payload_hash="sha256:note-v2",
        ),
    )

    first_page = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=0,
        domains=["notes.note"],
        page_size=1,
    )
    second_page = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=first_page.next_cursor,
        domains=["notes.note"],
        page_size=1,
    )
    third_page = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=second_page.next_cursor,
        domains=["notes.note"],
        page_size=1,
    )

    assert [item.client_envelope_id for item in stale.conflicts] == ["env-stale"]
    assert [item.server_sequence for item in visible_update.accepted] == [3]
    assert [item.client_envelope_id for item in first_page.envelopes] == ["env-create"]
    assert first_page.has_more is True
    assert first_page.next_cursor == "2"
    assert [item.client_envelope_id for item in second_page.envelopes] == [
        "env-update-current"
    ]
    assert second_page.has_more is False
    assert second_page.next_cursor == "3"
    assert third_page.envelopes == []
    assert third_page.has_more is False
    assert third_page.next_cursor == "3"


def test_tombstone_soft_deletes_note_and_updates_object_state(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None

    result = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-delete",
            client_sequence=2,
            operation="tombstone",
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:note-delete",
            deleted=True,
        ),
    )

    assert [item.client_envelope_id for item in result.accepted] == ["env-delete"]
    assert chacha_db.get_note_by_id("note-1") is None
    deleted_note = chacha_db.get_note_by_id("note-1", include_deleted=True)
    assert deleted_note is not None
    assert bool(deleted_note["deleted"]) is True
    state = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert state is not None
    assert state.deleted is True
    assert state.object_revision == 2


def test_exact_retry_of_applied_tombstone_does_not_rematerialize_or_conflict(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    tombstone = _note_envelope(
        client_envelope_id="env-delete",
        client_sequence=2,
        operation="tombstone",
        base_server_cursor=base.latest_server_cursor,
        base_object_revision=base.object_revision,
        base_object_hash=base.object_hash,
        object_revision=2,
        payload={"deleted_at": "2026-05-23T18:35:00+00:00"},
        payload_hash="sha256:note-delete",
        deleted=True,
    )
    _push_one(sync_service, tombstone)
    before = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert before is not None

    result = _push_one(sync_service, tombstone)

    assert [item.client_envelope_id for item in result.accepted] == ["env-delete"]
    assert result.conflicts == []
    assert sync_service.store.list_conflicts("dataset-1") == []
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "notes.note",
        entity_id="note-1",
        limit=10,
    )
    assert sorted((item.client_envelope_id, item.apply_status) for item in stored) == [
        ("env-create", "applied"),
        ("env-delete", "applied"),
    ]
    assert sync_service.store.get_object_state("dataset-1", "notes.note", "note-1") == before
    assert chacha_db.get_note_by_id("note-1") is None
    deleted_note = chacha_db.get_note_by_id("note-1", include_deleted=True)
    assert deleted_note is not None
    assert bool(deleted_note["deleted"]) is True


def test_tombstoned_note_is_not_resurrected_by_stale_upsert(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-delete",
            client_sequence=2,
            operation="tombstone",
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"deleted_at": "2026-05-23T18:35:00+00:00"},
            payload_hash="sha256:note-delete",
            deleted=True,
        ),
    )

    result = _push_one(
        sync_service,
        _note_envelope(
            client_envelope_id="env-stale-resurrect",
            client_sequence=3,
            base_server_cursor=base.latest_server_cursor,
            base_object_revision=base.object_revision,
            base_object_hash=base.object_hash,
            object_revision=2,
            payload={"title": "Stale resurrect", "body": "Do not revive."},
            payload_hash="sha256:stale-resurrect",
        ),
    )

    assert result.accepted == []
    assert len(result.conflicts) == 1
    assert chacha_db.get_note_by_id("note-1") is None
    deleted_note = chacha_db.get_note_by_id("note-1", include_deleted=True)
    assert deleted_note is not None
    assert bool(deleted_note["deleted"]) is True


def test_apply_failure_marks_accepted_envelope_failed_and_replayable(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_projection(*args, **kwargs):
        raise RuntimeError("projection unavailable")

    monkeypatch.setattr(chacha_db, "upsert_note_from_sync", _fail_projection)

    result = _push_one(sync_service, _note_envelope())

    assert [item.client_envelope_id for item in result.accepted] == ["env-create"]
    failed = sync_service.store.list_failed_applies("dataset-1")
    assert [item.client_envelope_id for item in failed] == ["env-create"]
    assert failed[0].apply_error_code == "notes_projection_failed"
    replayable = sync_service.store.list_accepted_envelopes_for_replay("dataset-1")
    assert [item.client_envelope_id for item in replayable] == ["env-create"]


def test_exact_retry_of_failed_create_retries_materialization(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_upsert = chacha_db.upsert_note_from_sync
    attempts = 0

    def _fail_once_then_project(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("projection unavailable")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(chacha_db, "upsert_note_from_sync", _fail_once_then_project)
    envelope = _note_envelope()

    first = _push_one(sync_service, envelope)
    second = _push_one(sync_service, envelope)

    assert [item.client_envelope_id for item in first.accepted] == ["env-create"]
    assert [item.client_envelope_id for item in second.accepted] == ["env-create"]
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "notes.note",
        entity_id="note-1",
        limit=10,
    )
    assert [(item.client_envelope_id, item.apply_status) for item in stored] == [
        ("env-create", "applied")
    ]
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert attempts == 2


def test_exact_retry_after_object_state_update_and_failed_applied_mark_completes(
    sync_service: SyncV2Service,
    chacha_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _push_one(sync_service, _note_envelope())
    base = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert base is not None
    update = _note_envelope(
        client_envelope_id="env-update",
        client_sequence=2,
        base_server_cursor=base.latest_server_cursor,
        base_object_revision=base.object_revision,
        base_object_hash=base.object_hash,
        object_revision=2,
        payload={
            "title": "Trip notes revised",
            "content": "Updated outline.",
        },
        payload_hash="sha256:note-v2",
    )
    original_mark = sync_service.store.mark_envelope_apply_status
    failed_once = False

    def _fail_first_applied_mark(server_cursor, *, apply_status, **kwargs):
        nonlocal failed_once
        if apply_status == "applied" and not failed_once:
            failed_once = True
            raise RuntimeError("apply status unavailable")
        return original_mark(server_cursor, apply_status=apply_status, **kwargs)

    monkeypatch.setattr(
        sync_service.store,
        "mark_envelope_apply_status",
        _fail_first_applied_mark,
    )

    first = _push_one(sync_service, update)
    second = _push_one(sync_service, update)

    assert [item.client_envelope_id for item in first.accepted] == ["env-update"]
    assert [item.client_envelope_id for item in second.accepted] == ["env-update"]
    assert first.conflicts == []
    assert second.conflicts == []
    assert sync_service.store.list_conflicts("dataset-1") == []
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "notes.note",
        entity_id="note-1",
        limit=10,
    )
    assert sorted((item.client_envelope_id, item.apply_status) for item in stored) == [
        ("env-create", "applied"),
        ("env-update", "applied"),
    ]
    state = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")
    assert state is not None
    assert state.object_revision == 2
    assert state.object_hash == "sha256:note-v2"
    note = chacha_db.get_note_by_id("note-1")
    assert note is not None
    assert note["title"] == "Trip notes revised"
