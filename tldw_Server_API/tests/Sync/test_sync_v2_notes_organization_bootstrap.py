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
REPAIR_KEYWORD_ID = "88888888-8888-4888-8888-888888888888"
FOREIGN_KEYWORD_ID = "99999999-9999-4999-8999-999999999999"


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


def test_bootstrap_excludes_organization_rows_owned_by_another_tenant(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = _seed_source(note_db, store)
    with note_db.transaction() as connection:
        connection.execute(
            "INSERT INTO keywords(sync_id, keyword, client_id) VALUES (?, ?, ?)",
            (FOREIGN_KEYWORD_ID, "Other tenant private keyword", "user-2"),
        )

    result = NotesOrganizationBootstrapper(note_db).bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    assert result.metadata["notes_organization_v1"]["state"] == "ready"
    assert all(
        envelope.object_id != FOREIGN_KEYWORD_ID
        for envelope in _organization_envelopes(store)
    )
    assert projection.get_resource("notes.keyword", FOREIGN_KEYWORD_ID) is None
    with note_db.transaction() as connection:
        foreign = connection.execute(
            "SELECT deleted, client_id FROM keywords WHERE sync_id = ?",
            (FOREIGN_KEYWORD_ID,),
        ).fetchone()
    assert foreign is not None
    assert bool(foreign["deleted"]) is False
    assert foreign["client_id"] == "user-2"
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


def test_new_bootstrap_attempt_tombstones_relationship_captured_by_prior_attempt(
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

    def interrupt_after_capture(_completed_groups: int) -> None:
        raise NotesOrganizationBootstrapInterrupted("attempt one captured")

    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        NotesOrganizationBootstrapper(
            note_db,
            batch_size=100,
            after_group=interrupt_after_capture,
        ).bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    attempt_one = store.get_dataset("dataset-1")
    attempt_one_state = attempt_one.metadata["notes_organization_v1"]
    failed = store.transition_notes_organization_bootstrap(
        "dataset-1",
        bootstrap_id="bootstrap-1",
        expected_state="initializing",
        state="failed",
        captured_count=attempt_one_state["captured_count"],
        expected_count=attempt_one_state["expected_count"],
        error_code="notes_organization_bootstrap_capture_failed",
    )
    assert failed.metadata["notes_organization_v1"]["state"] == "failed"
    attempt_two = store.begin_notes_organization_bootstrap(
        "dataset-1",
        owner_user_id="user-1",
        bootstrap_id="bootstrap-2",
    )
    projection.apply_relationship(
        domain="notes.keyword_link",
        object_id=link_id,
        operation="tombstone",
        payload=link_payload,
        routing_metadata={},
    )

    ready = NotesOrganizationBootstrapper(note_db, batch_size=100).bootstrap(
        service=service,
        user_id="user-1",
        dataset=attempt_two,
    )

    removed = store.get_current_head("dataset-1", "notes.keyword_link", link_id)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert removed is not None
    assert removed.operation == "tombstone"
    assert removed.apply_status == "applied"
    assert removed.routing_metadata["bootstrap_id"] == "bootstrap-2"
    note_db.close_connection()


def test_bootstrap_restart_replays_same_applied_relationship_removal_group(
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
    captured_once = False

    def interrupt_after_capture(_completed_groups: int) -> None:
        nonlocal captured_once
        if not captured_once:
            captured_once = True
            raise NotesOrganizationBootstrapInterrupted("captured relationship")

    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        NotesOrganizationBootstrapper(
            note_db,
            batch_size=8,
            after_group=interrupt_after_capture,
        ).bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    projection.apply_relationship(
        domain="notes.keyword_link",
        object_id=link_id,
        operation="tombstone",
        payload=link_payload,
        routing_metadata={},
    )

    repair_boundaries: list[int] = []
    crashed_once = False

    def crash_after_removal_group(completed_groups: int) -> None:
        nonlocal crashed_once
        repair_boundaries.append(completed_groups)
        if not crashed_once:
            crashed_once = True
            raise NotesOrganizationBootstrapInterrupted("removal applied before ready")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db,
        batch_size=100,
        after_group=crash_after_removal_group,
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    before_restart = _organization_envelopes(store)
    removed = store.get_current_head("dataset-1", "notes.keyword_link", link_id)
    assert removed is not None
    assert removed.operation == "tombstone"
    assert removed.apply_status == "applied"
    removal_group_id = removed.mutation_group_id
    removal_ids = {
        item.client_envelope_id
        for item in before_restart
        if item.mutation_group_id == removal_group_id
    }

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    after_restart = _organization_envelopes(store)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert repair_boundaries == [1, 1]
    assert {
        item.client_envelope_id
        for item in after_restart
        if item.mutation_group_id == removal_group_id
    } == removal_ids
    assert len({item.client_envelope_id for item in after_restart}) == len(after_restart)
    note_db.close_connection()


def test_removal_repair_restart_reuses_manifest_with_new_resource(
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
    first_capture = True

    def interrupt_first_capture(_completed_groups: int) -> None:
        nonlocal first_capture
        if first_capture:
            first_capture = False
            raise NotesOrganizationBootstrapInterrupted("initial manifest applied")

    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        NotesOrganizationBootstrapper(
            note_db,
            batch_size=100,
            after_group=interrupt_first_capture,
        ).bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    projection.apply_relationship(
        domain="notes.keyword_link",
        object_id=link_id,
        operation="tombstone",
        payload=link_payload,
        routing_metadata={},
    )
    projection.apply_resource(
        domain="notes.keyword",
        object_id=REPAIR_KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "Added during removal repair"},
    )

    repair_crash = True

    def interrupt_repair(_completed_groups: int) -> None:
        nonlocal repair_crash
        if repair_crash:
            repair_crash = False
            raise NotesOrganizationBootstrapInterrupted("repair manifest applied")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db,
        batch_size=100,
        after_group=interrupt_repair,
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    before_restart = _organization_envelopes(store)
    repair_group = next(
        item.mutation_group_id
        for item in before_restart
        if item.object_id == REPAIR_KEYWORD_ID
    )
    repair_ids = {
        item.client_envelope_id
        for item in before_restart
        if item.mutation_group_id == repair_group
    }

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    after_restart = _organization_envelopes(store)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert {
        item.client_envelope_id
        for item in after_restart
        if item.mutation_group_id == repair_group
    } == repair_ids
    assert len({item.client_envelope_id for item in after_restart}) == len(after_restart)
    removed = store.get_current_head("dataset-1", "notes.keyword_link", link_id)
    added = store.get_current_head("dataset-1", "notes.keyword", REPAIR_KEYWORD_ID)
    assert removed is not None and removed.operation == "tombstone"
    assert added is not None and added.payload == {"keyword": "Added during removal repair"}
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


def test_retryable_deleted_resource_group_repairs_shadowed_steps_after_source_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = NotesOrganizationSyncStore(note_db)
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "Dormant"},
    )
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="tombstone",
        payload={},
    )
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

    prior = _organization_envelopes(store)
    assert paused.metadata["notes_organization_v1"]["state"] == "initializing"
    assert len(prior) == 2
    assert [item.operation for item in prior] == ["upsert", "tombstone"]
    assert all(item.apply_status == "pending" for item in prior)
    assert len({item.mutation_group_id for item in prior}) == 1
    assert store.get_current_head(
        "dataset-1", "notes.keyword", DELETED_KEYWORD_ID
    ) == prior[-1]

    projection.apply_resource(
        domain="notes.keyword",
        object_id="88888888-8888-4888-8888-888888888888",
        operation="upsert",
        payload={"keyword": "Added during repair"},
    )
    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=paused,
    )

    repaired = _organization_envelopes(store)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert all(item.apply_status == "applied" for item in repaired)
    assert len({item.client_envelope_id for item in repaired}) == len(repaired)
    assert {item.client_envelope_id for item in prior}.issubset(
        {item.client_envelope_id for item in repaired}
    )
    note_db.close_connection()


def test_exact_group_reconcile_never_rewrites_state_to_shadowed_pending_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = NotesOrganizationSyncStore(note_db)
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "Dormant"},
    )
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="tombstone",
        payload={},
    )
    bootstrapper = NotesOrganizationBootstrapper(note_db, batch_size=2)
    original_verifier = bootstrapper._step_matches_source
    monkeypatch.setattr(bootstrapper, "_step_matches_source", lambda _envelope: False)
    paused = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )
    monkeypatch.setattr(bootstrapper, "_step_matches_source", original_verifier)
    upsert, tombstone = _organization_envelopes(store)
    assert upsert.apply_status == "pending"
    assert tombstone.server_cursor is not None
    store.mark_bootstrap_envelope_verified(
        tombstone.server_cursor,
        bootstrap_id="bootstrap-1",
    )

    object_state_writes: list[int] = []
    original_execute = store.db.execute

    def trace_object_state_writes(sql: str, *args: object, **kwargs: object) -> object:
        normalized = " ".join(sql.split())
        if normalized.startswith("INSERT INTO sync_object_state"):
            params = args[0] if args else kwargs.get("params")
            assert isinstance(params, tuple)
            object_state_writes.append(int(params[5]))
        return original_execute(sql, *args, **kwargs)

    monkeypatch.setattr(store.db, "execute", trace_object_state_writes)
    bootstrapper._drain_preexisting_heads(service, paused)

    repaired_upsert = next(
        item
        for item in _organization_envelopes(store)
        if item.server_cursor == upsert.server_cursor
    )
    assert repaired_upsert.apply_status == "applied"
    assert upsert.server_cursor not in object_state_writes
    assert repaired_upsert.apply_error_code == "sync_bootstrap_superseded"
    note_db.close_connection()


def test_stale_pending_step_is_reconciled_only_after_verified_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = NotesOrganizationSyncStore(note_db)
    projection.apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "Before"},
    )
    bootstrapper = NotesOrganizationBootstrapper(note_db, batch_size=1)
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
    stale = _organization_envelopes(store)[0]
    assert paused.metadata["notes_organization_v1"]["state"] == "initializing"
    assert stale.apply_status == "pending"
    assert stale.server_cursor is not None

    projection.apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "After"},
    )
    original_mark_verified = store.mark_bootstrap_envelope_verified

    def reject_stale_preflight_mark(
        server_cursor: int,
        *,
        bootstrap_id: str,
        notes_task_bootstrap: bool = False,
    ):
        assert notes_task_bootstrap is False
        assert server_cursor != stale.server_cursor
        return original_mark_verified(
            server_cursor,
            bootstrap_id=bootstrap_id,
            notes_task_bootstrap=notes_task_bootstrap,
        )

    monkeypatch.setattr(
        store,
        "mark_bootstrap_envelope_verified",
        reject_stale_preflight_mark,
    )
    reconciliations: list[tuple[int, int]] = []
    original_reconcile = getattr(store, "reconcile_bootstrap_envelope_superseded", None)
    if original_reconcile is not None:

        def audit_reconciliation(
            server_cursor: int,
            *,
            bootstrap_id: str,
            superseded_by_cursor: int,
        ):
            correction = next(
                item
                for item in _organization_envelopes(store)
                if item.server_cursor == superseded_by_cursor
            )
            assert correction.apply_status == "applied"
            assert correction.payload == {"keyword": "After"}
            reconciliations.append((server_cursor, superseded_by_cursor))
            return original_reconcile(
                server_cursor,
                bootstrap_id=bootstrap_id,
                superseded_by_cursor=superseded_by_cursor,
            )

        monkeypatch.setattr(
            store,
            "reconcile_bootstrap_envelope_superseded",
            audit_reconciliation,
        )

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=paused,
    )

    envelopes = _organization_envelopes(store)
    repaired_stale = next(
        item for item in envelopes if item.server_cursor == stale.server_cursor
    )
    correction = store.get_current_head("dataset-1", "notes.keyword", KEYWORD_ID)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert correction is not None
    assert correction.payload == {"keyword": "After"}
    assert correction.apply_status == "applied"
    assert reconciliations == [(stale.server_cursor, correction.server_cursor)]
    assert repaired_stale.apply_status == "applied"
    assert repaired_stale.apply_error_code == "sync_bootstrap_superseded"
    note_db.close_connection()


def test_reverted_source_defers_stale_audit_until_matching_correction_is_applied(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = NotesOrganizationSyncStore(note_db)
    projection.apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "A"},
    )
    bootstrapper = NotesOrganizationBootstrapper(note_db, batch_size=1)
    original_verifier = bootstrapper._step_matches_source
    failed_once = False

    def fail_a_once(envelope) -> bool:
        nonlocal failed_once
        if not failed_once:
            failed_once = True
            return False
        return original_verifier(envelope)

    monkeypatch.setattr(bootstrapper, "_step_matches_source", fail_a_once)
    paused_a = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )
    old_a = _organization_envelopes(store)[0]
    assert paused_a.metadata["notes_organization_v1"]["state"] == "initializing"
    assert old_a.payload == {"keyword": "A"}
    assert old_a.apply_status == "pending"
    assert old_a.server_cursor is not None

    projection.apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "B"},
    )
    crashed = False

    def crash_after_b_group(_completed_groups: int) -> None:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise NotesOrganizationBootstrapInterrupted("B applied before stale audit")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db,
        batch_size=1,
        after_group=crash_after_b_group,
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=paused_a,
        )
    before_revert = _organization_envelopes(store)
    durable_b = store.get_current_head("dataset-1", "notes.keyword", KEYWORD_ID)
    assert durable_b is not None
    assert durable_b.payload == {"keyword": "B"}
    assert durable_b.apply_status == "applied"
    assert old_a.server_cursor < durable_b.server_cursor
    assert next(
        item for item in before_revert if item.server_cursor == old_a.server_cursor
    ).apply_status == "pending"

    projection.apply_resource(
        domain="notes.keyword",
        object_id=KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "A"},
    )
    chronology: list[tuple[str, int]] = []
    original_mark_verified = store.mark_bootstrap_envelope_verified
    original_reconcile = store.reconcile_bootstrap_envelope_superseded

    def record_verified(
        server_cursor: int,
        *,
        bootstrap_id: str,
        notes_task_bootstrap: bool = False,
    ):
        assert notes_task_bootstrap is False
        envelope = next(
            item
            for item in _organization_envelopes(store)
            if item.server_cursor == server_cursor
        )
        assert envelope.payload == {"keyword": "A"}
        chronology.append(("verified", server_cursor))
        return original_mark_verified(
            server_cursor,
            bootstrap_id=bootstrap_id,
            notes_task_bootstrap=notes_task_bootstrap,
        )

    def record_reconciled(
        server_cursor: int,
        *,
        bootstrap_id: str,
        superseded_by_cursor: int,
    ):
        correction = store.get_current_head("dataset-1", "notes.keyword", KEYWORD_ID)
        assert correction is not None
        assert correction.server_cursor == superseded_by_cursor
        assert correction.payload == {"keyword": "A"}
        chronology.append(("reconciled", superseded_by_cursor))
        return original_reconcile(
            server_cursor,
            bootstrap_id=bootstrap_id,
            superseded_by_cursor=superseded_by_cursor,
        )

    monkeypatch.setattr(store, "mark_bootstrap_envelope_verified", record_verified)
    monkeypatch.setattr(
        store,
        "reconcile_bootstrap_envelope_superseded",
        record_reconciled,
    )
    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    after_repair = _organization_envelopes(store)
    correction_a = store.get_current_head("dataset-1", "notes.keyword", KEYWORD_ID)
    old_a_after = next(
        item for item in after_repair if item.server_cursor == old_a.server_cursor
    )
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert correction_a is not None
    assert correction_a.payload == {"keyword": "A"}
    assert correction_a.apply_status == "applied"
    assert correction_a.server_cursor > durable_b.server_cursor
    assert chronology == [
        ("verified", correction_a.server_cursor),
        ("reconciled", correction_a.server_cursor),
    ]
    assert old_a_after.apply_status == "applied"
    assert old_a_after.apply_error_code == "sync_bootstrap_superseded"
    client_ids = [item.client_envelope_id for item in after_repair]
    assert len(client_ids) == len(set(client_ids))
    assert {item.client_envelope_id for item in before_revert}.issubset(client_ids)
    replay = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=ready,
    )
    assert replay.metadata["notes_organization_v1"]["state"] == "ready"
    assert [
        item.client_envelope_id for item in _organization_envelopes(store)
    ] == client_ids
    note_db.close_connection()


def test_deleted_resource_source_change_appends_explicit_correction_lineage(
    tmp_path: Path,
) -> None:
    service, store, note_db = _service(tmp_path)
    projection = NotesOrganizationSyncStore(note_db)
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "Before"},
    )
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="tombstone",
        payload={},
    )
    interrupted = False

    def interrupt_before_ready(_completed_groups: int) -> None:
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise NotesOrganizationBootstrapInterrupted("captured old deleted lineage")

    bootstrapper = NotesOrganizationBootstrapper(
        note_db,
        batch_size=2,
        after_group=interrupt_before_ready,
    )
    with pytest.raises(NotesOrganizationBootstrapInterrupted):
        bootstrapper.bootstrap(
            service=service,
            user_id="user-1",
            dataset=store.get_dataset("dataset-1"),
        )
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="upsert",
        payload={"keyword": "After"},
    )
    projection.apply_resource(
        domain="notes.keyword",
        object_id=DELETED_KEYWORD_ID,
        operation="tombstone",
        payload={},
    )

    ready = bootstrapper.bootstrap(
        service=service,
        user_id="user-1",
        dataset=store.get_dataset("dataset-1"),
    )

    history = [
        item
        for item in _organization_envelopes(store)
        if item.object_id == DELETED_KEYWORD_ID
    ]
    correction_upsert = next(
        item
        for item in history
        if item.operation == "upsert" and item.payload == {"keyword": "After"}
    )
    head = store.get_current_head("dataset-1", "notes.keyword", DELETED_KEYWORD_ID)
    resource = next(item for item in projection.snapshot().resources if item.sync_id == DELETED_KEYWORD_ID)
    assert ready.metadata["notes_organization_v1"]["state"] == "ready"
    assert correction_upsert.routing_metadata["restore_intent"] is True
    assert all(item.apply_status == "applied" for item in history)
    assert head is not None
    assert head.operation == "tombstone"
    assert resource.deleted is True
    assert resource.name == "After"
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
