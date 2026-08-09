from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
    NotesOrganizationSyncStore,
)
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
    NotesOrganizationDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.notes_organization_bootstrap import (
    NotesOrganizationBootstrapInterrupted,
    NotesOrganizationBootstrapper,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit

NOTE_ID = "11111111-1111-4111-8111-111111111111"
KEYWORD_ID = "22222222-2222-4222-8222-222222222222"
DELETED_KEYWORD_ID = "33333333-3333-4333-8333-333333333333"
COLLECTION_PARENT_ID = "44444444-4444-4444-8444-444444444444"
COLLECTION_CHILD_ID = "55555555-5555-4555-8555-555555555555"
FOLDER_PARENT_ID = "66666666-6666-4666-8666-666666666666"
FOLDER_CHILD_ID = "77777777-7777-4777-8777-777777777777"


def _service(tmp_path: Path) -> tuple[SyncV2Service, SyncV2Store, CharactersRAGDB]:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={"default_personal": True, "client_family": "chatbook"},
        )
    )
    dataset = store.begin_notes_organization_bootstrap(
        "dataset-1", owner_user_id="user-1", bootstrap_id="bootstrap-1"
    )
    assert dataset.metadata["notes_organization_v1"]["state"] == "initializing"
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
        + [NotesOrganizationDomainAdapter(domain) for domain in NOTES_ORGANIZATION_DOMAINS]
    )
    service = SyncV2Service(store=store, adapters=registry, clock=lambda: "2026-08-08T12:00:00Z")
    note_db = CharactersRAGDB(str(tmp_path / "notes.sqlite"), client_id="user-1")
    return service, store, note_db


def _seed_source(note_db: CharactersRAGDB, store: SyncV2Store) -> NotesOrganizationSyncStore:
    projection = NotesOrganizationSyncStore(note_db)
    note_db.add_note("Bootstrap note", "Body", note_id=NOTE_ID)
    store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="core-note",
            domain="notes.note",
            operation="upsert",
            object_id=NOTE_ID,
            object_revision=1,
            payload={"title": "Bootstrap note"},
            payload_hash="sha256:core-note",
            status="accepted",
        )
    )
    for object_id, name in (
        (KEYWORD_ID, "Active"),
        (DELETED_KEYWORD_ID, "Dormant"),
    ):
        projection.apply_resource(
            domain="notes.keyword",
            object_id=object_id,
            operation="upsert",
            payload={"keyword": name},
        )
    for domain, parent_id, child_id in (
        ("notes.keyword_collection", COLLECTION_PARENT_ID, COLLECTION_CHILD_ID),
        ("notes.folder", FOLDER_PARENT_ID, FOLDER_CHILD_ID),
    ):
        projection.apply_resource(
            domain=domain,
            object_id=parent_id,
            operation="upsert",
            payload={"name": "Parent", "parent_sync_id": None},
        )
        projection.apply_resource(
            domain=domain,
            object_id=child_id,
            operation="upsert",
            payload={"name": "Child", "parent_sync_id": parent_id},
        )
    projection.apply_relationship(
        domain="notes.keyword_link",
        object_id=organization_link_id(
            "notes.keyword_link", ["note", NOTE_ID, DELETED_KEYWORD_ID]
        ),
        operation="upsert",
        payload={
            "subject_type": "note",
            "subject_id": NOTE_ID,
            "keyword_sync_id": DELETED_KEYWORD_ID,
        },
        routing_metadata={},
    )
    # Deleting after the link exists makes the relationship dormant but still
    # part of the transactionally captured source state.
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="tombstone",
        payload={},
    )
    return projection


def _organization_envelopes(store: SyncV2Store):
    return store.list_envelopes_after(
        "dataset-1", 0, limit=1000, domains=NOTES_ORGANIZATION_DOMAINS
    )


def test_bootstrap_captures_in_dependency_order_without_replaying_product_state(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = _seed_source(note_db, store)
    before = projection.snapshot()

    result = NotesOrganizationBootstrapper(note_db, batch_size=2).bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    assert result.metadata["notes_organization_v1"] == {
        "bootstrap_id": "bootstrap-1",
        "state": "ready",
        "captured_count": len(before.resources) + len(before.relationships),
        "expected_count": len(before.resources) + len(before.relationships),
        "error_code": None,
    }
    assert projection.snapshot() == before
    envelopes = _organization_envelopes(store)
    assert envelopes
    assert all(envelope.apply_status == "applied" for envelope in envelopes)
    operations = [envelope.operation for envelope in envelopes]
    assert operations[-1] == "tombstone"
    child = next(item for item in envelopes if item.object_id == COLLECTION_CHILD_ID)
    parent = next(item for item in envelopes if item.object_id == COLLECTION_PARENT_ID)
    assert (parent.server_cursor or 0) < (child.server_cursor or 0)
    relationship = next(item for item in envelopes if item.domain == "notes.keyword_link")
    assert relationship.routing_metadata["bootstrap_capture"] is True
    note_db.close_connection()


def test_bootstrap_resumes_stable_groups_and_reconciles_a_changed_snapshot(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = _seed_source(note_db, store)
    interrupted = False

    def interrupt_once(_completed_groups: int) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise NotesOrganizationBootstrapInterrupted("injected interruption")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db, batch_size=2, after_group=interrupt_once
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    paused = store.get_dataset("dataset-1")
    assert paused.metadata["notes_organization_v1"]["state"] == "initializing"
    prior_ids = {item.client_envelope_id for item in _organization_envelopes(store)}

    projection.apply_resource(
        domain="notes.keyword",
        object_id="88888888-8888-4888-8888-888888888888",
        operation="upsert",
        payload={"keyword": "Added while paused"},
    )
    resumed = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=paused,
    )

    envelopes = _organization_envelopes(store)
    client_ids = [item.client_envelope_id for item in envelopes]
    assert resumed.metadata["notes_organization_v1"]["state"] == "ready"
    assert prior_ids.issubset(client_ids)
    assert len(client_ids) == len(set(client_ids))
    assert any(item.object_id == "88888888-8888-4888-8888-888888888888" for item in envelopes)
    note_db.close_connection()


def test_bootstrap_resume_tombstones_a_captured_relationship_removed_from_source(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = _seed_source(note_db, store)
    link_id = organization_link_id(
        "notes.keyword_link", ["note", NOTE_ID, DELETED_KEYWORD_ID]
    )
    link_payload = {
        "subject_type": "note",
        "subject_id": NOTE_ID,
        "keyword_sync_id": DELETED_KEYWORD_ID,
    }
    interrupted = False

    def interrupt_once(_completed_groups: int) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise NotesOrganizationBootstrapInterrupted("injected interruption")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db,
        batch_size=7,
        after_group=interrupt_once,
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    captured = store.get_current_head("dataset-1", "notes.keyword_link", link_id)
    assert captured is not None
    assert captured.operation == "upsert"
    assert captured.apply_status == "applied"

    projection.apply_relationship(
        domain="notes.keyword_link",
        object_id=link_id,
        operation="tombstone",
        payload=link_payload,
        routing_metadata={},
    )
    assert projection.snapshot().relationships == ()

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    removed = store.get_current_head("dataset-1", "notes.keyword_link", link_id)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert removed is not None
    assert removed.operation == "tombstone"
    assert removed.apply_status == "applied"
    assert removed.routing_metadata["bootstrap_removal"] is True
    assert projection.snapshot().relationships == ()
    note_db.close_connection()


def test_bootstrap_resume_of_unchanged_snapshot_reuses_history_without_duplicates(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = _seed_source(note_db, store)
    expected_steps = (
        len(projection.snapshot().resources)
        + len(projection.snapshot().relationships)
        + sum(item.deleted for item in projection.snapshot().resources)
    )
    interrupted = False

    def interrupt_once(_completed_groups: int) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise NotesOrganizationBootstrapInterrupted("injected interruption")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db, batch_size=2, after_group=interrupt_once
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    prior_ids = {item.client_envelope_id for item in _organization_envelopes(store)}

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )
    replay = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=ready,
    )

    envelopes = _organization_envelopes(store)
    client_ids = [item.client_envelope_id for item in envelopes]
    assert replay.metadata["notes_organization_v1"]["state"] == "ready"
    assert prior_ids.issubset(client_ids)
    assert len(envelopes) == expected_steps
    assert len(client_ids) == len(set(client_ids))
    note_db.close_connection()


def test_retryable_verification_failure_reuses_bootstrap_group_and_repairs_to_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, note_db = _service(tmp_path)
    _seed_source(note_db, store)
    bootstrapper = NotesOrganizationBootstrapper(note_db, batch_size=2)
    original_verifier = bootstrapper._step_matches_source
    failed_once = False

    def fail_once(envelope) -> bool:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            return False
        return original_verifier(envelope)

    monkeypatch.setattr(bootstrapper, "_step_matches_source", fail_once)
    paused = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    paused_metadata = paused.metadata["notes_organization_v1"]
    assert paused_metadata["bootstrap_id"] == "bootstrap-1"
    assert paused_metadata["state"] == "initializing"
    assert paused_metadata["error_code"] == "notes_organization_bootstrap_capture_failed"
    prior = _organization_envelopes(store)
    assert prior
    assert any(envelope.apply_status != "applied" for envelope in prior)
    prior_ids = {envelope.client_envelope_id for envelope in prior}
    projection = NotesOrganizationSyncStore(note_db)
    projection.apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "Renamed during repair"},
    )

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=paused,
    )

    envelopes = _organization_envelopes(store)
    client_ids = [envelope.client_envelope_id for envelope in envelopes]
    assert ready.metadata["notes_organization_v1"]["bootstrap_id"] == "bootstrap-1"
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert prior_ids.issubset(client_ids)
    assert len(client_ids) == len(set(client_ids))
    assert all(envelope.apply_status == "applied" for envelope in envelopes)
    keyword_head = store.get_current_head("dataset-1", "notes.keyword", KEYWORD_ID)
    assert keyword_head is not None
    assert keyword_head.payload == {"keyword": "Renamed during repair"}
    note_db.close_connection()


def test_bootstrap_fails_safe_on_corrupt_hierarchy_and_ready_requires_exact_counts(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = _seed_source(note_db, store)
    with note_db.transaction() as conn:
        first = conn.execute(
            "SELECT id FROM keyword_collections WHERE sync_id = ?", (COLLECTION_PARENT_ID,)
        ).fetchone()["id"]
        second = conn.execute(
            "SELECT id FROM keyword_collections WHERE sync_id = ?", (COLLECTION_CHILD_ID,)
        ).fetchone()["id"]
        conn.execute("UPDATE keyword_collections SET parent_id = ? WHERE id = ?", (second, first))
        conn.execute("UPDATE keyword_collections SET parent_id = ? WHERE id = ?", (first, second))

    failed = NotesOrganizationBootstrapper(note_db).bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    assert failed.metadata["notes_organization_v1"]["state"] == "failed"
    assert failed.metadata["notes_organization_v1"]["error_code"] == (
        "notes_organization_bootstrap_source_invalid"
    )
    assert "bootstrap_id" in failed.metadata["notes_organization_v1"]
    assert projection.snapshot().resources
    note_db.close_connection()
