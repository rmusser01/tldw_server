"""Stage26: Sync wikilink saves must not replace an intervening Notes edit."""

from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import notes as notes_endpoint
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes.wikilink_rename import (
    WikilinkRestoreTarget,
    WikilinkRewriteTarget,
    restore_title_links,
    rewrite_title_links,
)
from tldw_Server_API.app.core.Notes.wikilinks import WikilinkTokenReplacement
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import server_trusted_encryption_status_from_config
from tldw_Server_API.app.core.Sync.v2.server_origin import (
    SyncServerOriginMaterializationError,
    capture_server_origin_mutation,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.integration

NOTE_ID = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"


@pytest.fixture
def notes_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Use the real Notes and Sync stores, both exclusively in temporary SQLite."""

    monkeypatch.setenv("TLDW_USER_DB_BACKEND", "sqlite")
    monkeypatch.delenv("SYNC_V2_DATABASE_URL", raising=False)
    db = CharactersRAGDB(str(tmp_path / "notes.db"), client_id="1")
    sync_db = SyncDatabase(sqlite_path=tmp_path / "sync.db")
    service = SyncV2Service(
        store=SyncV2Store(sync_db),
        adapters=SyncAdapterRegistry([NotesDomainAdapter()]),
        materializers={"notes.note": NotesMaterializer(db)},
        clock=lambda: "2026-10-06T12:00:00+00:00",
        settings=SyncV2Settings(
            supported_domains=["notes.note"],
            operations={"notes.note": ["upsert", "tombstone"]},
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage", server_trusted_enabled=True, auth_mode="multi_user"
            ),
        ),
    )
    try:
        profile = service.bootstrap_profile(
            user_id="1",
            mode="server_frontend",
            device_id="frontend",
            device_name="Stage26",
            requested_domains=["notes.note"],
        )
        monkeypatch.setattr(notes_endpoint, "get_active_server_origin_sync_service_for_user", lambda user_id: service)
        yield db, service, profile.dataset.dataset_id
    finally:
        db.close_connection()
        sync_db.backend.get_pool().close_all()


@pytest.mark.parametrize("operation", ["rewrite", "undo"])
def test_sync_wikilink_save_skips_edit_between_read_and_capture(notes_sync, monkeypatch, operation: str) -> None:
    """A real concurrent Sync edit wins; the stale rewrite/undo records no mutation."""

    db, service, dataset_id = notes_sync
    original = "Before [[Old title]]." if operation == "rewrite" else "Before [[New title]]."
    capture_server_origin_mutation(
        service,
        user_id="1",
        domain="notes.note",
        operation="upsert",
        object_id=NOTE_ID,
        payload={"title": "Linker", "content": original},
        source="server_api",
    )
    concurrent = {"title": "Concurrent title", "content": "Concurrent edit [[Old title]]."}

    def edit_then_capture(*args, **kwargs):
        capture_server_origin_mutation(
            service,
            user_id="1",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            payload=concurrent,
            source="server_api",
        )
        return capture_server_origin_mutation(*args, **kwargs)

    # Finite interleaving at the actual capture boundary, after the core version read.
    monkeypatch.setattr(notes_endpoint, "capture_server_origin_mutation", edit_then_capture)
    save = notes_endpoint._wikilink_rename_saver(db, User(id=1, username="stage26", is_active=True))
    if operation == "rewrite":
        results = rewrite_title_links(
            db,
            old_title="Old title",
            replacement="[[New title]]",
            targets=[WikilinkRewriteTarget(NOTE_ID, 1)],
            save=save,
        )
    else:
        results = restore_title_links(
            db,
            old_title="Old title",
            replacement="[[New title]]",
            targets=[WikilinkRestoreTarget(NOTE_ID, 1, (WikilinkTokenReplacement(0, "[[Old title]]"),))],
            save=save,
        )

    current = db.get_note_by_id(NOTE_ID)
    assert (current["title"], current["content"], current["version"]) == (concurrent["title"], concurrent["content"], 2)
    assert (results[0].status, results[0].version, results[0].replacements) == ("skipped_conflict", 2, ())
    envelopes = service.store.list_envelopes_after(dataset_id, 0, domains=["notes.note"], limit=10)
    assert len(envelopes) == 2
    assert all(envelope.apply_status == "applied" for envelope in envelopes)


def _capture(service: SyncV2Service, **kwargs):
    return capture_server_origin_mutation(
        service,
        user_id="1",
        domain="notes.note",
        operation="upsert",
        object_id=NOTE_ID,
        payload=kwargs.pop("payload", {"title": "Linker", "content": "Before [[Old title]]."}),
        source=kwargs.pop("source", "notes-wikilink-rewrite"),
        **kwargs,
    )


@pytest.fixture
def seeded_note(notes_sync):
    db, service, dataset_id = notes_sync
    _capture(service, source="server_api")
    return db, service, dataset_id


def test_sync_wikilink_rewrite_and_undo_publish_applied_versions(seeded_note) -> None:
    db, service, dataset_id = seeded_note
    save = notes_endpoint._wikilink_rename_saver(db, User(id=1, username="stage26", is_active=True))
    rewritten = rewrite_title_links(
        db,
        old_title="Old title",
        replacement="[[New title]]",
        targets=[WikilinkRewriteTarget(NOTE_ID, 1)],
        save=save,
    )[0]
    restored = restore_title_links(
        db,
        old_title="Old title",
        replacement="[[New title]]",
        targets=[WikilinkRestoreTarget(NOTE_ID, 2, rewritten.replacements)],
        save=save,
    )[0]
    assert (rewritten.status, rewritten.version, restored.status, restored.version) == ("updated", 2, "restored", 3)
    assert db.get_note_by_id(NOTE_ID)["content"] == "Before [[Old title]]."
    assert [item.apply_status for item in service.store.list_envelopes_after(dataset_id, 0, limit=10)] == [
        "applied",
        "applied",
        "applied",
    ]


def test_product_race_at_atomic_write_is_recorded_as_conflict_and_replay_refuses(seeded_note, monkeypatch) -> None:
    db, service, dataset_id = seeded_note
    upsert = db.upsert_note_from_sync

    def edit_then_upsert(**kwargs):
        db.update_note(
            NOTE_ID, {"title": "Concurrent title", "content": "Authoritative local edit"}, expected_version=1
        )
        return upsert(**kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(db, "upsert_note_from_sync", edit_then_upsert)
        with pytest.raises(SyncServerOriginMaterializationError) as caught:
            _capture(service, expected_product_version=1)
    accepted = service.store.list_envelopes_after(dataset_id, 0, limit=10)[-1]
    assert (caught.value.envelope.status, accepted.apply_status) == ("accepted", "conflict")
    assert service._materialize_envelope(accepted).status == "conflict"
    assert (db.get_note_by_id(NOTE_ID)["content"], db.get_note_by_id(NOTE_ID)["version"]) == (
        "Authoritative local edit",
        2,
    )
    assert service.store.get_object_state(dataset_id, "notes.note", NOTE_ID).object_revision == 1
    with pytest.raises(SyncStoreError):
        _capture(service, source="server_api")
    assert len(service.store.list_envelopes_after(dataset_id, 0, limit=10)) == 2


def test_concurrent_sync_head_wins_between_snapshot_and_append(seeded_note, monkeypatch) -> None:
    db, service, dataset_id = seeded_note
    insert = service.store.insert_envelope

    def advance_then_insert(envelope):
        with monkeypatch.context() as scoped:
            scoped.setattr(service.store, "insert_envelope", insert)
            _capture(service, source="server_api", payload={"title": "Concurrent", "content": "Latest Sync edit"})
        return insert(envelope)

    monkeypatch.setattr(service.store, "insert_envelope", advance_then_insert)
    with pytest.raises(SyncStoreError):
        _capture(service, expected_product_version=1)
    assert (db.get_note_by_id(NOTE_ID)["content"], db.get_note_by_id(NOTE_ID)["version"]) == ("Latest Sync edit", 2)
    envelopes = service.store.list_envelopes_after(dataset_id, 0, limit=10)
    assert len(envelopes) == 2
    state = service.store.get_object_state(dataset_id, "notes.note", NOTE_ID)
    assert (state.latest_server_cursor, state.object_hash) == (envelopes[-1].server_cursor, envelopes[-1].payload_hash)


@pytest.mark.parametrize("version", [None, True, 0, -1, "1"])
def test_guarded_capture_requires_server_argument_not_routing_version(seeded_note, version) -> None:
    db, service, dataset_id = seeded_note
    with pytest.raises(SyncStoreError):
        _capture(
            service, expected_product_version=version, routing_metadata={"notes_ingestion_expected_product_version": 1}
        )
    assert db.get_note_by_id(NOTE_ID)["version"] == 1
    assert len(service.store.list_envelopes_after(dataset_id, 0, limit=10)) == 1


@pytest.mark.parametrize("version,owner", [(None, "1"), (True, "1"), (0, "1"), (-1, "1"), ("1", "1"), (1, "2")])
def test_required_guard_survives_invalid_or_missing_persisted_version(seeded_note, version, owner: str) -> None:
    db, service, dataset_id = seeded_note
    state = service.store.get_object_state(dataset_id, "notes.note", NOTE_ID)
    routing = {
        "source": "notes-wikilink-rewrite",
        "origin": "server",
        "server_device_id": "server-origin",
        "server_owner_user_id": owner,
    }
    if version is not None:
        routing["notes_ingestion_expected_product_version"] = version
    accepted = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="invalid-guard",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            device_id="server-origin",
            object_revision=2,
            base_server_cursor=state.latest_server_cursor,
            base_object_revision=state.object_revision,
            base_object_hash=state.object_hash,
            payload={"title": "Must not write", "content": "Invalid guard"},
            payload_hash="sha256:invalid-guard",
            routing_metadata=routing,
        )
    )
    assert service._materialize_envelope(accepted).status == "failed"
    assert service._materialize_envelope(service._envelope_snapshot(accepted)).status == "failed"
    assert (db.get_note_by_id(NOTE_ID)["title"], db.get_note_by_id(NOTE_ID)["version"]) == ("Linker", 1)


@pytest.mark.parametrize("source", ["notes-wikilink-rewrite", "notes-ingestion", {}])
def test_remote_sender_cannot_activate_guard_with_forged_routing(seeded_note, source) -> None:
    db, service, dataset_id = seeded_note
    state = service.store.get_object_state(dataset_id, "notes.note", NOTE_ID)
    result = service.push(
        user_id="1",
        dataset_id=dataset_id,
        device_id="frontend",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="remote-edit",
                domain="notes.note",
                operation="upsert",
                object_id=NOTE_ID,
                device_id="frontend",
                client_sequence=1,
                object_revision=2,
                base_server_cursor=state.latest_server_cursor,
                base_object_revision=state.object_revision,
                base_object_hash=state.object_hash,
                payload={"title": "Remote", "content": "Remote content"},
                payload_hash="sha256:remote-edit",
                encryption_metadata={"policy": "server_trusted_v1"},
                routing_metadata={
                    "source": source,
                    "origin": "server",
                    "server_device_id": "server-origin",
                    "server_owner_user_id": "1",
                    "notes_ingestion_expected_product_version": 999,
                },
            )
        ],
    )
    assert [item.apply_status for item in result.accepted] == ["applied"]
    assert db.get_note_by_id(NOTE_ID)["content"] == "Remote content"


@pytest.mark.parametrize("edited", [False, True])
def test_capture_retry_after_product_commit_only_accepts_exact_postcondition(seeded_note, monkeypatch, edited) -> None:
    db, service, dataset_id = seeded_note
    with monkeypatch.context() as scoped:

        def fail_state(*args, **kwargs):
            raise SyncStoreError("Injected state-recording failure")

        scoped.setattr(service.store, "upsert_object_state", fail_state)
        with pytest.raises(SyncServerOriginMaterializationError):
            _capture(service, expected_product_version=1, stable_key="rewrite-once")
    assert db.get_note_by_id(NOTE_ID)["version"] == 2
    if edited:
        db.update_note(NOTE_ID, {"content": "Later edit"}, expected_version=2)
        with pytest.raises(SyncServerOriginMaterializationError):
            _capture(service, expected_product_version=1, stable_key="rewrite-once")
        assert db.get_note_by_id(NOTE_ID)["content"] == "Later edit"
    else:
        result = _capture(service, expected_product_version=1, stable_key="rewrite-once")
        assert result.envelope.apply_status == "applied"
        assert service.store.get_object_state(dataset_id, "notes.note", NOTE_ID).object_revision == 2


def test_product_revision_can_advance_independently_without_ingestion_revision_rule(seeded_note) -> None:
    db, service, dataset_id = seeded_note
    db.update_note(NOTE_ID, {"content": "Local edit [[Old title]]."}, expected_version=1)
    save = notes_endpoint._wikilink_rename_saver(db, User(id=1, username="stage26", is_active=True))
    result = rewrite_title_links(
        db, old_title="Old title", replacement="[[New title]]", targets=[WikilinkRewriteTarget(NOTE_ID, 2)], save=save
    )[0]
    assert (result.status, result.version) == ("updated", 3)
    assert service.store.get_object_state(dataset_id, "notes.note", NOTE_ID).object_revision == 3


def test_unenrolled_note_refuses_without_resetting_version_or_accepting_debt(notes_sync) -> None:
    db, service, dataset_id = notes_sync
    note_id = db.add_note(title="Legacy", content="Before [[Old title]].")
    db.update_note(note_id, {"content": "Local edit [[Old title]]."}, expected_version=1)
    save = notes_endpoint._wikilink_rename_saver(db, User(id=1, username="stage26", is_active=True))
    result = rewrite_title_links(
        db, old_title="Old title", replacement="[[New title]]", targets=[WikilinkRewriteTarget(note_id, 2)], save=save
    )[0]
    assert result.status == "failed"
    assert (db.get_note_by_id(note_id)["content"], db.get_note_by_id(note_id)["version"]) == (
        "Local edit [[Old title]].",
        2,
    )
    assert service.store.list_envelopes_after(dataset_id, 0, limit=10) == []


def test_routing_cannot_override_server_expected_version_or_source(seeded_note) -> None:
    db, service, _dataset_id = seeded_note
    result = _capture(
        service,
        expected_product_version=1,
        routing_metadata={
            "source": "server_api",
            "notes_ingestion_expected_product_version": 999,
        },
    )
    assert result.envelope.apply_status == "applied"
    assert db.get_note_by_id(NOTE_ID)["version"] == 2


def test_ingestion_revision_equality_is_not_weakened(seeded_note) -> None:
    db, service, dataset_id = seeded_note
    state = service.store.get_object_state(dataset_id, "notes.note", NOTE_ID)
    accepted = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="invalid-ingestion-revision",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            device_id="server-origin",
            object_revision=3,
            base_server_cursor=state.latest_server_cursor,
            base_object_revision=state.object_revision,
            base_object_hash=state.object_hash,
            payload={"title": "Must not write", "content": "Inconsistent revision"},
            payload_hash="sha256:invalid-ingestion-revision",
            routing_metadata={
                "source": "notes-ingestion",
                "origin": "server",
                "server_device_id": "server-origin",
                "server_owner_user_id": "1",
                "notes_ingestion_expected_product_version": 1,
            },
        )
    )
    assert service._materialize_envelope(accepted).status == "failed"
    assert db.get_note_by_id(NOTE_ID)["version"] == 1
