from __future__ import annotations

import pytest


class FakeNotesDB:
    def __init__(self) -> None:
        self.updated_calls: list[dict[str, object]] = []
        self.added_calls: list[dict[str, object]] = []
        self.soft_deleted_calls: list[dict[str, object]] = []
        self.source_folder_calls: list[dict[str, object]] = []

    def update_note(self, note_id: str, update_data: dict[str, object], expected_version: int):
        self.updated_calls.append(
            {
                "note_id": note_id,
                "update_data": update_data,
                "expected_version": expected_version,
            }
        )
        return True

    def add_note(self, *, title: str, content: str):
        self.added_calls.append({"title": title, "content": content})
        return "n-2"

    def soft_delete_note(self, note_id: str, expected_version: int):
        self.soft_deleted_calls.append(
            {
                "note_id": note_id,
                "expected_version": expected_version,
            }
        )
        return True

    def sync_note_source_folders(self, note_id: str, source_id: int, folder_paths: list[str]):
        self.source_folder_calls.append(
            {
                "note_id": note_id,
                "source_id": source_id,
                "folder_paths": folder_paths,
            }
        )
        return [
            {
                "id": idx + 1,
                "name": path.split("/")[-1],
                "path": path,
                "parent_id": None if "/" not in path else idx,
            }
            for idx, path in enumerate(folder_paths)
        ]


@pytest.fixture
def fake_notes_db() -> FakeNotesDB:
    return FakeNotesDB()


@pytest.mark.unit
def test_notes_sink_does_not_overwrite_detached_note(fake_notes_db):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import apply_notes_change

    result = apply_notes_change(
        fake_notes_db,
        binding={"note_id": "n-1", "sync_status": "conflict_detached"},
        change={"event_type": "changed", "relative_path": "notes/a.md", "text": "# A\n\nNew body"},
        policy="canonical",
    )

    assert result["action"] == "skipped_detached"
    assert result["sync_status"] == "conflict_detached"
    assert fake_notes_db.updated_calls == []


@pytest.mark.unit
def test_notes_sink_creates_note_with_heading_title(fake_notes_db):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import apply_notes_change

    result = apply_notes_change(
        fake_notes_db,
        binding=None,
        change={"event_type": "created", "relative_path": "notes/a.md", "text": "# A\n\nBody"},
        policy="canonical",
    )

    assert result["action"] == "created"
    assert result["note_id"] == "n-2"
    assert fake_notes_db.added_calls[0]["title"] == "A"


@pytest.mark.unit
def test_notes_sink_soft_deletes_note_for_canonical_upstream_delete(fake_notes_db):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import apply_notes_change

    result = apply_notes_change(
        fake_notes_db,
        binding={"note_id": "n-1", "current_version": 3, "sync_status": "sync_managed"},
        change={"event_type": "deleted", "relative_path": "notes/a.md"},
        policy="canonical",
    )

    assert result["action"] == "archived"
    assert result["note_id"] == "n-1"
    assert result["sync_status"] == "archived_upstream_removed"
    assert fake_notes_db.soft_deleted_calls == [
        {
            "note_id": "n-1",
            "expected_version": 3,
        }
    ]


@pytest.mark.unit
def test_notes_sink_inactive_bound_update_preserves_direct_path(fake_notes_db):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import apply_notes_change

    result = apply_notes_change(
        fake_notes_db,
        binding={
            "note_id": "n-1",
            "current_version": 3,
            "sync_status": "sync_managed",
        },
        change={
            "event_type": "changed",
            "relative_path": "notes/a.md",
            "text": "# Updated\n\nBody",
        },
        policy="canonical",
    )

    assert result == {
        "action": "updated",
        "note_id": "n-1",
        "sync_status": "sync_managed",
    }
    assert fake_notes_db.updated_calls == [
        {
            "note_id": "n-1",
            "update_data": {"title": "Updated", "content": "# Updated\n\nBody"},
            "expected_version": 3,
        }
    ]


@pytest.mark.unit
def test_notes_sink_syncs_source_managed_folders_from_relative_path(fake_notes_db):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks.notes_sink import apply_notes_change

    result = apply_notes_change(
        fake_notes_db,
        binding=None,
        change={
            "event_type": "created",
            "relative_path": "docs/api/a.md",
            "text": "# A\n\nBody",
            "source_id": 91,
        },
        policy="canonical",
    )

    assert result["action"] == "created"
    assert fake_notes_db.source_folder_calls == [
        {
            "note_id": "n-2",
            "source_id": 91,
            "folder_paths": ["docs", "docs/api"],
        }
    ]


def _active_sync_stack(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
    from tldw_Server_API.app.core.Ingestion_Sources.sinks import notes_sink
    from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
    from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
    from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
        NotesOrganizationDomainAdapter,
    )
    from tldw_Server_API.app.core.Sync.v2.materializers.notes import NotesMaterializer
    from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
        NotesOrganizationMaterializer,
    )
    from tldw_Server_API.app.core.Sync.v2.models import (
        NOTES_ORGANIZATION_DOMAINS,
        SyncDatasetCreate,
    )
    from tldw_Server_API.app.core.Sync.v2.security import (
        server_trusted_encryption_status_from_config,
    )
    from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
    from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

    notes_db = CharactersRAGDB(
        db_path=str(tmp_path / "notes.sqlite"),
        client_id="user-1",
    )
    sync_store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.sqlite"))
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_organization_v1": {"state": "ready"},
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [
                NotesDomainAdapter(),
                *[
                    NotesOrganizationDomainAdapter(domain=domain)
                    for domain in NOTES_ORGANIZATION_DOMAINS
                ],
            ]
        ),
        materializers={
            "notes.note": NotesMaterializer(notes_db),
            **{
                domain: NotesOrganizationMaterializer(notes_db, domain)
                for domain in NOTES_ORGANIZATION_DOMAINS
            },
        },
        settings=SyncV2Settings(
            supported_domains=["notes.note", *NOTES_ORGANIZATION_DOMAINS],
            operations={
                "notes.note": ["upsert", "tombstone"],
                **{
                    domain: ["upsert", "tombstone"]
                    for domain in NOTES_ORGANIZATION_DOMAINS
                },
            },
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            ),
        ),
        clock=lambda: "2026-08-09T08:00:00+00:00",
    )
    monkeypatch.setattr(
        notes_sink,
        "get_active_server_origin_sync_service_for_user",
        lambda user_id: service,
        raising=False,
    )
    return notes_db, sync_store, service


@pytest.mark.unit
def test_notes_sink_active_ready_sync_captures_source_folder_provenance(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.DB_Management.chacha.organization_sync_store import (
        NotesOrganizationSyncStore,
    )
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
    from tldw_Server_API.app.core.Ingestion_Sources.sinks import notes_sink

    notes_db, sync_store, _service = _active_sync_stack(tmp_path, monkeypatch)

    def _direct_source_folder_write(*args, **kwargs):
        raise AssertionError("active-ready Sync must not use the legacy direct path")

    monkeypatch.setattr(notes_db, "sync_note_source_folders", _direct_source_folder_write)
    result = notes_sink.apply_notes_change(
        notes_db,
        binding=None,
        change={
            "event_type": "created",
            "relative_path": "docs/api/a.md",
            "text": "# A\n\nBody",
            "source_id": 91,
        },
        policy="canonical",
    )

    assert result["action"] == "created"
    note_id = str(result["note_id"])
    assert [row["path"] for row in notes_db.get_note_folders_for_note(note_id)] == [
        "docs",
        "docs/api",
    ]
    with notes_db.transaction() as conn:
        provenance = conn.execute(
            "SELECT source_id FROM note_folder_source_memberships "
            "WHERE note_id = ? ORDER BY folder_id",
            (note_id,),
        ).fetchall()
    assert [int(row["source_id"]) for row in provenance] == [91, 91]
    folder_links = [
        envelope
        for envelope in sync_store.list_envelopes_after("dataset-1", 0)
        if envelope.domain == "notes.folder_link"
    ]
    assert len(folder_links) == 2
    for envelope in folder_links:
        folder_provenance = envelope.routing_metadata[
            "notes_folder_origin_provenance"
        ]
        assert folder_provenance["operation"] == "source_upsert"
        assert folder_provenance["source_id"] == 91
        assert len(folder_provenance["pre_state_hash"]) == 64
        assert len(folder_provenance["post_state_hash"]) == 64
    assert len(NotesOrganizationSyncStore(notes_db).snapshot().relationships) == 2

    envelope_count = len(sync_store.list_envelopes_after("dataset-1", 0))
    with pytest.raises(ConflictError, match="version mismatch"):
        notes_sink.apply_notes_change(
            notes_db,
            binding={
                "note_id": note_id,
                "current_version": 99,
                "sync_status": "sync_managed",
            },
            change={
                "event_type": "changed",
                "relative_path": "docs/api/a.md",
                "text": "# Changed\n\nBody",
                "source_id": 91,
            },
            policy="canonical",
        )
    assert len(sync_store.list_envelopes_after("dataset-1", 0)) == envelope_count
    notes_db.close_connection()


@pytest.mark.unit
def test_notes_sink_active_ready_duplicate_unbound_delivery_replays_manifest(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks import notes_sink

    notes_db, sync_store, _service = _active_sync_stack(tmp_path, monkeypatch)
    change = {
        "event_type": "created",
        "relative_path": "a.md",
        "text": "# A\n\nBody",
        "source_id": 91,
    }

    first = notes_sink.apply_notes_change(
        notes_db,
        binding=None,
        change=change,
        policy="canonical",
    )
    envelope_count = len(sync_store.list_envelopes_after("dataset-1", 0))
    replay = notes_sink.apply_notes_change(
        notes_db,
        binding=None,
        change=change,
        policy="canonical",
    )

    assert replay == first
    assert len(sync_store.list_envelopes_after("dataset-1", 0)) == envelope_count
    assert notes_db.get_note_by_id(str(first["note_id"]))["version"] == 1
    notes_db.close_connection()


@pytest.mark.unit
def test_notes_sink_active_ready_bound_missing_note_conflicts_without_append(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
    from tldw_Server_API.app.core.Ingestion_Sources.sinks import notes_sink

    notes_db, sync_store, _service = _active_sync_stack(tmp_path, monkeypatch)

    with pytest.raises(ConflictError, match="missing"):
        notes_sink.apply_notes_change(
            notes_db,
            binding={
                "note_id": "missing-bound-note",
                "current_version": 4,
                "sync_status": "sync_managed",
            },
            change={
                "event_type": "changed",
                "relative_path": "missing.md",
                "text": "# Missing\n\nDo not recreate",
                "source_id": 91,
            },
            policy="canonical",
        )

    assert notes_db.get_note_by_id("missing-bound-note", include_deleted=True) is None
    assert sync_store.list_envelopes_after("dataset-1", 0) == []
    notes_db.close_connection()


@pytest.mark.unit
def test_notes_sink_active_ready_version_change_after_precheck_does_not_overwrite(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks import notes_sink
    from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
        SyncServerOriginBatchMaterializationError,
    )

    notes_db, _sync_store, service = _active_sync_stack(tmp_path, monkeypatch)
    created = notes_sink.apply_notes_change(
        notes_db,
        binding=None,
        change={
            "event_type": "created",
            "relative_path": "race.md",
            "text": "# Before\n\nBody",
            "source_id": 91,
        },
        policy="canonical",
    )
    note_id = str(created["note_id"])
    delegate = service.materializers["notes.note"]

    class _ConcurrentNoteMutation:
        def __init__(self) -> None:
            self.mutated = False

        def apply(self, envelope, *, store):
            if envelope.object_id == note_id and not self.mutated:
                self.mutated = True
                notes_db.update_note(
                    note_id,
                    {"title": "Concurrent", "content": "Preserve me"},
                    expected_version=1,
                )
            return delegate.apply(envelope, store=store)

    service.materializers["notes.note"] = _ConcurrentNoteMutation()

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        notes_sink.apply_notes_change(
            notes_db,
            binding={
                "note_id": note_id,
                "current_version": 1,
                "sync_status": "sync_managed",
            },
            change={
                "event_type": "changed",
                "relative_path": "race.md",
                "text": "# Upstream\n\nOverwrite",
                "source_id": 91,
            },
            policy="canonical",
        )

    current = notes_db.get_note_by_id(note_id)
    assert current["title"] == "Concurrent"
    assert current["content"] == "Preserve me"
    assert current["version"] == 2
    notes_db.close_connection()


@pytest.mark.unit
def test_notes_sink_active_ready_product_commit_retry_completes_bookkeeping(
    tmp_path,
    monkeypatch,
):
    from tldw_Server_API.app.core.Ingestion_Sources.sinks import notes_sink
    from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
        SyncServerOriginBatchMaterializationError,
    )

    notes_db, sync_store, _service = _active_sync_stack(tmp_path, monkeypatch)
    created = notes_sink.apply_notes_change(
        notes_db,
        binding=None,
        change={
            "event_type": "created",
            "relative_path": "retry.md",
            "text": "# Before\n\nBody",
            "source_id": 91,
        },
        policy="canonical",
    )
    note_id = str(created["note_id"])
    delegate_upsert_object_state = sync_store.upsert_object_state
    failed_once = False

    def _fail_bookkeeping_once(state):
        nonlocal failed_once
        if state.object_id == note_id and state.object_revision == 2 and not failed_once:
            failed_once = True
            raise RuntimeError("sync bookkeeping unavailable")
        return delegate_upsert_object_state(state)

    monkeypatch.setattr(sync_store, "upsert_object_state", _fail_bookkeeping_once)
    binding = {
        "note_id": note_id,
        "current_version": 1,
        "sync_status": "sync_managed",
    }
    change = {
        "event_type": "changed",
        "relative_path": "retry.md",
        "text": "# After\n\nExact intended body",
        "source_id": 91,
    }

    with pytest.raises(SyncServerOriginBatchMaterializationError):
        notes_sink.apply_notes_change(
            notes_db,
            binding=binding,
            change=change,
            policy="canonical",
        )

    after_product_commit = notes_db.get_note_by_id(note_id)
    assert after_product_commit["title"] == "After"
    assert after_product_commit["content"] == "# After\n\nExact intended body"
    assert after_product_commit["version"] == 2
    envelopes_after_failure = sync_store.list_envelopes_after("dataset-1", 0)
    update_envelope = next(
        envelope
        for envelope in envelopes_after_failure
        if envelope.domain == "notes.note" and envelope.object_revision == 2
    )
    assert update_envelope.apply_status == "failed"
    update_identity = (
        update_envelope.client_envelope_id,
        update_envelope.mutation_group_id,
    )

    retried = notes_sink.apply_notes_change(
        notes_db,
        binding=binding,
        change=change,
        policy="canonical",
    )

    after_retry = notes_db.get_note_by_id(note_id)
    assert retried == {
        "action": "updated",
        "note_id": note_id,
        "sync_status": "sync_managed",
    }
    assert after_retry == after_product_commit
    envelopes_after_retry = sync_store.list_envelopes_after("dataset-1", 0)
    assert len(envelopes_after_retry) == len(envelopes_after_failure)
    replayed_update = next(
        envelope
        for envelope in envelopes_after_retry
        if envelope.domain == "notes.note" and envelope.object_revision == 2
    )
    assert (
        replayed_update.client_envelope_id,
        replayed_update.mutation_group_id,
    ) == update_identity
    assert replayed_update.apply_status == "applied"
    object_state = sync_store.get_object_state("dataset-1", "notes.note", note_id)
    assert object_state is not None
    assert object_state.object_revision == 2
    assert object_state.latest_server_cursor == replayed_update.server_cursor
    notes_db.close_connection()
