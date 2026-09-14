from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import NotesLinkDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GuardedProductMutation,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import NotesLinkMaterializer
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, SyncEnvelopeCreate, SyncObjectState
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkCoordinator,
    NotesLinkDatasetConflictError,
    NotesLinkNotReadyError,
    NotesLinkPreflightError,
    NotesLinkSyncInactiveDatasetError,
    resolve_notes_link_coordinator,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.server_origin_batch import (
    SyncServerOriginBatchIdempotencyConflictError,
    SyncServerOriginBatchMaterializationError,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

OWNER_ID = "owner-1"
SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
NOW = "2026-08-10T12:00:00+00:00"


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _service(
    tmp_path: Path,
    note_db: CharactersRAGDB,
    *,
    materializer: object | None = None,
    clock=None,
) -> tuple[SyncV2Service, SyncV2Store, str]:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain) for domain in M1_SYNC_DOMAINS] + [NotesLinkDomainAdapter()]
    )
    service = SyncV2Service(
        store=store,
        adapters=registry,
        materializers={
            "notes.link": materializer or NotesLinkMaterializer(note_db),
        },
        clock=clock or (lambda: NOW),
        id_factory=lambda prefix: f"{prefix}-stable",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    profile = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
    )
    assert profile.dataset is not None
    dataset_id = profile.dataset.dataset_id
    _seed_note_heads(store, dataset_id)
    return service, store, dataset_id


def _ready_links(store: SyncV2Store, dataset_id: str) -> None:
    store.begin_notes_link_bootstrap(
        dataset_id,
        owner_user_id=OWNER_ID,
        bootstrap_id="links-ready",
    )
    store.transition_notes_link_bootstrap(
        dataset_id,
        bootstrap_id="links-ready",
        expected_state="initializing",
        state="ready",
        captured_count=0,
        expected_count=0,
        source_hash=None,
        ready_verifier=lambda: True,
    )


def _seed_note_heads(store: SyncV2Store, dataset_id: str) -> None:
    for index, note_id in enumerate((SOURCE_ID, TARGET_ID), start=1):
        stored = store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id=f"note-{index}",
                domain="notes.note",
                operation="upsert",
                object_id=note_id,
                device_id="device-1",
                object_revision=1,
                entity_version=1,
                payload={"title": note_id, "content": "body"},
                payload_hash=f"sha256:note-{index}",
                created_at_client=NOW,
                status="accepted",
            )
        )
        assert stored.server_cursor is not None
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=dataset_id,
                domain="notes.note",
                object_id=note_id,
                object_revision=1,
                object_hash=stored.payload_hash or "",
                latest_server_cursor=stored.server_cursor,
                deleted=False,
            )
        )
        store.mark_envelope_apply_status(stored.server_cursor, apply_status="applied")


@pytest.fixture()
def note_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "notes.db", client_id=OWNER_ID)
    for note_id in (SOURCE_ID, TARGET_ID):
        db.note_store.add_note(note_id, "body", note_id=note_id)
    try:
        yield db
    finally:
        db.close_connection()


def test_dataset_resolution_defaults_only_to_active_canonical_dataset(
    tmp_path: Path,
    note_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, store, dataset_id = _service(tmp_path, note_db)
    _ready_links(store, dataset_id)
    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.notes_link_coordinator.get_active_server_origin_sync_service_for_user",
        lambda _user_id: service,
    )

    implicit = resolve_notes_link_coordinator(
        user_id=OWNER_ID,
        note_db=note_db,
        dataset_id=None,
    )
    explicit = resolve_notes_link_coordinator(
        user_id=OWNER_ID,
        note_db=note_db,
        dataset_id=dataset_id,
    )
    assert implicit is not None and implicit.dataset.dataset_id == dataset_id
    assert explicit is not None and explicit.dataset.dataset_id == dataset_id

    with pytest.raises(NotesLinkDatasetConflictError):
        resolve_notes_link_coordinator(
            user_id=OWNER_ID,
            note_db=note_db,
            dataset_id="another-personal-dataset",
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.notes_link_coordinator.get_active_server_origin_sync_service_for_user",
        lambda _user_id: None,
    )
    assert (
        resolve_notes_link_coordinator(
            user_id=OWNER_ID,
            note_db=note_db,
            dataset_id=None,
        )
        is None
    )
    with pytest.raises(NotesLinkSyncInactiveDatasetError):
        resolve_notes_link_coordinator(
            user_id=OWNER_ID,
            note_db=note_db,
            dataset_id=dataset_id,
        )


def test_create_exact_replay_and_changed_request_conflict(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    service, store, dataset_id = _service(tmp_path, note_db)
    _ready_links(store, dataset_id)
    coordinator = NotesLinkCoordinator(service, note_db, OWNER_ID, store.get_dataset(dataset_id))

    first = coordinator.create(
        source_note_id=SOURCE_ID,
        target_note_id=TARGET_ID,
        directed=False,
        weight=1.0,
        label=None,
        properties={"kind": "reference"},
        idempotency_key="create-one",
    )
    replay_events: list[tuple[str, str | None]] = []
    replay = coordinator.create(
        source_note_id=TARGET_ID,
        target_note_id=SOURCE_ID,
        directed=False,
        weight=1.0,
        label=None,
        properties={"kind": "reference"},
        idempotency_key="create-one",
        guarded_mutation=GuardedProductMutation(
            expected_domain="notes.link",
            expected_object_id=first.edge_id,
            before=lambda _conn: replay_events.append(("before", None)),
            after=lambda _conn, identity: replay_events.append(("after", identity)),
        ),
    )

    assert replay == first
    assert replay_events == [("before", None), ("after", first.edge_id)]
    assert first.version == 1
    assert len(store.list_current_heads(dataset_id, "notes.link", limit=10, offset=0)) == 1

    with pytest.raises(NotesLinkPreflightError):
        coordinator.create(
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            directed=False,
            weight=1.0,
            label="same logical identity",
            properties={},
            idempotency_key="create-duplicate",
        )
    assert len(store.list_current_heads(dataset_id, "notes.link", limit=10, offset=0)) == 1

    assert note_db.soft_delete_note(SOURCE_ID, expected_version=1)
    assert (
        coordinator.create(
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            directed=False,
            weight=1.0,
            label=None,
            properties={"kind": "reference"},
            idempotency_key="create-one",
        )
        == first
    )
    with pytest.raises(SyncServerOriginBatchIdempotencyConflictError):
        coordinator.create(
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            directed=False,
            weight=2.0,
            label=None,
            properties={"kind": "reference"},
            idempotency_key="create-one",
        )


def test_update_tombstone_and_restore_are_versioned_and_normalize_label(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    service, store, dataset_id = _service(tmp_path, note_db)
    _ready_links(store, dataset_id)
    coordinator = NotesLinkCoordinator(service, note_db, OWNER_ID, store.get_dataset(dataset_id))
    created = coordinator.create(
        source_note_id=SOURCE_ID,
        target_note_id=TARGET_ID,
        directed=True,
        weight=1.0,
        label="related",
        properties={"origin": "manual"},
        idempotency_key="create-two",
    )
    updated = coordinator.update(
        edge_id=created.edge_id,
        expected_version=1,
        weight=2.0,
        label="strong",
        properties={"origin": "manual", "rank": 2},
        idempotency_key="update-two",
    )
    deleted = coordinator.tombstone(
        edge_id=created.edge_id,
        expected_version=2,
        reason="manual-delete",
        idempotency_key="delete-two",
    )
    restored = coordinator.restore(
        edge_id=created.edge_id,
        expected_version=3,
        idempotency_key="restore-two",
    )

    assert (updated.version, updated.label, updated.weight) == (2, "strong", 2.0)
    assert deleted.version == 3 and deleted.deleted is True
    assert restored.version == 4 and restored.deleted is False
    assert restored.created_at == created.created_at
    assert restored.created_by == created.created_by


def test_not_ready_and_projection_failure_never_report_success_or_write_product(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    service, store, dataset_id = _service(tmp_path, note_db)
    store.begin_notes_link_bootstrap(
        dataset_id,
        owner_user_id=OWNER_ID,
        bootstrap_id="links-paused",
    )
    dataset = store.get_dataset(dataset_id)
    assert dataset is not None
    coordinator = NotesLinkCoordinator(service, note_db, OWNER_ID, dataset)

    with pytest.raises(NotesLinkNotReadyError):
        coordinator.create(
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            directed=False,
            weight=1.0,
            label=None,
            properties={},
            idempotency_key="not-ready",
        )
    assert note_db.notes_link_store.snapshot() == ()
    assert store.list_current_heads(dataset_id, "notes.link", limit=10, offset=0) == []

    class _FailingMaterializer:
        domain = "notes.link"

        def apply(self, envelope, *, store):
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="notes_link_projection_failed",
                apply_error_message="safe failure",
            )
            from tldw_Server_API.app.core.Sync.v2.materializers.base import MaterializationResult

            return MaterializationResult(
                status="failed",
                error_code="notes_link_projection_failed",
                message="safe failure",
            )

    failing_service, failing_store, failing_dataset_id = _service(
        tmp_path / "failing",
        note_db,
        materializer=_FailingMaterializer(),
    )
    _ready_links(failing_store, failing_dataset_id)
    failing = NotesLinkCoordinator(
        failing_service,
        note_db,
        OWNER_ID,
        failing_store.get_dataset(failing_dataset_id),
    )
    with pytest.raises(SyncServerOriginBatchMaterializationError):
        failing.create(
            source_note_id=SOURCE_ID,
            target_note_id=TARGET_ID,
            directed=False,
            weight=1.0,
            label=None,
            properties={},
            idempotency_key="projection-fails",
        )
    assert note_db.notes_link_store.snapshot() == ()


def test_active_dataset_lookup_failures_are_not_misreported_as_inactive(
    note_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail(_user_id: str):
        raise SyncStoreError("lookup failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Sync.v2.notes_link_coordinator.get_active_server_origin_sync_service_for_user",
        _fail,
    )
    with pytest.raises(SyncStoreError, match="lookup failed"):
        resolve_notes_link_coordinator(
            user_id=OWNER_ID,
            note_db=note_db,
            dataset_id=None,
        )


def test_server_origin_payload_and_envelope_use_one_canonical_timestamp(
    tmp_path: Path,
    note_db: CharactersRAGDB,
) -> None:
    ticks = count()

    def ticking_clock() -> str:
        return (datetime(2026, 8, 10, 12, tzinfo=timezone.utc) + timedelta(seconds=next(ticks))).isoformat()

    service, store, dataset_id = _service(
        tmp_path,
        note_db,
        clock=ticking_clock,
    )
    _ready_links(store, dataset_id)
    coordinator = NotesLinkCoordinator(service, note_db, OWNER_ID, store.get_dataset(dataset_id))

    created = coordinator.create(
        source_note_id=SOURCE_ID,
        target_note_id=TARGET_ID,
        directed=False,
        weight=1.0,
        label=None,
        properties={},
        idempotency_key="one-clock",
    )
    head = store.get_current_head(dataset_id, "notes.link", created.edge_id)

    assert head is not None
    assert created.created_at == head.created_at_client
    assert created.last_modified == head.created_at_client
