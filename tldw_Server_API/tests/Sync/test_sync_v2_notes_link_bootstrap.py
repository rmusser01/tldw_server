from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLinkStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import notes_link_bootstrap as notes_link_bootstrap_module
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import NotesLinkDomainAdapter
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import NotesLinkMaterializer
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SyncEnvelopeCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.notes_link_bootstrap import (
    NotesLinkBootstrapInterrupted,
    NotesLinkBootstrapper,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

OWNER_ID = "owner-1"
SOURCE_ID = "11111111-1111-4111-8111-111111111111"
TARGET_ID = "22222222-2222-4222-8222-222222222222"
THIRD_ID = "33333333-3333-4333-8333-333333333333"
LIVE_EDGE_ID = "44444444-4444-4444-8444-444444444444"
DELETED_EDGE_ID = "55555555-5555-4555-8555-555555555555"
NOW = "2026-08-10T12:00:00+00:00"


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _registry() -> SyncAdapterRegistry:
    return SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain) for domain in M1_SYNC_DOMAINS] + [NotesLinkDomainAdapter()]
    )


def _service(
    tmp_path: Path,
    note_db: CharactersRAGDB,
    *,
    bootstrapper: NotesLinkBootstrapper | None = None,
) -> tuple[SyncV2Service, SyncV2Store]:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync.db"))
    service = SyncV2Service(
        store=store,
        adapters=_registry(),
        materializers={"notes.link": NotesLinkMaterializer(note_db)},
        clock=lambda: NOW,
        id_factory=lambda prefix: f"{prefix}-stable",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
        notes_link_bootstrapper=bootstrapper,
    )
    return service, store


def _seed_product(note_db: CharactersRAGDB) -> None:
    for note_id in (SOURCE_ID, TARGET_ID, THIRD_ID):
        note_db.note_store.add_note(note_id, "body", note_id=note_id)
    links = NotesLinkStore(note_db)
    links.upsert(
        edge_id=LIVE_EDGE_ID,
        payload=_payload(SOURCE_ID, TARGET_ID),
        expected_version=None,
    )
    links.upsert(
        edge_id=DELETED_EDGE_ID,
        payload=_payload(SOURCE_ID, THIRD_ID),
        expected_version=None,
    )
    links.tombstone(
        edge_id=DELETED_EDGE_ID,
        payload={
            **_payload(SOURCE_ID, THIRD_ID, modified="2026-08-10T12:00:01+00:00"),
            "deleted_at": "2026-08-10T12:00:01+00:00",
            "reason": "manual-delete",
        },
        expected_version=1,
    )


def _payload(source: str, target: str, *, modified: str = NOW) -> dict[str, object]:
    source, target = sorted((source, target))
    return {
        "source_note_id": source,
        "target_note_id": target,
        "type": "manual",
        "directed": False,
        "weight": 1.0,
        "label": None,
        "properties": {},
        "created_at": NOW,
        "last_modified": modified,
        "created_by": "device-1",
    }


def _seed_note_heads(store: SyncV2Store, dataset_id: str) -> None:
    for index, note_id in enumerate((SOURCE_ID, TARGET_ID, THIRD_ID), start=1):
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
def product_db(tmp_path: Path):
    db = CharactersRAGDB(tmp_path / "product.db", client_id=OWNER_ID)
    _seed_product(db)
    try:
        yield db
    finally:
        db.close_connection()


def test_existing_ready_dataset_enrolls_link_without_mutating_organization_state(
    tmp_path: Path,
    product_db: CharactersRAGDB,
) -> None:
    service, store = _service(tmp_path, product_db)
    first = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
    )
    assert first.dataset is not None
    dataset_id = first.dataset.dataset_id
    dataset = store.begin_notes_organization_bootstrap(
        dataset_id,
        owner_user_id=OWNER_ID,
        bootstrap_id="organization-stable",
    )
    dataset = store.transition_notes_organization_bootstrap(
        dataset_id,
        bootstrap_id="organization-stable",
        expected_state="initializing",
        state="ready",
        captured_count=0,
        expected_count=0,
        ready_verifier=lambda: True,
    )
    original_organization = dict(dataset.metadata["notes_organization_v1"])

    initializing = store.begin_notes_link_bootstrap(
        dataset_id,
        owner_user_id=OWNER_ID,
        bootstrap_id="links-stable",
    )

    assert "notes.link" in initializing.domains
    assert initializing.metadata["notes_link_v1"]["state"] == "initializing"
    assert initializing.metadata["notes_organization_v1"] == original_organization
    assert set(NOTES_ORGANIZATION_DOMAINS).issubset(initializing.domains)


def test_generic_enrollment_cannot_alias_notes_link_product_authority(
    tmp_path: Path,
    product_db: CharactersRAGDB,
) -> None:
    service, _store = _service(tmp_path, product_db)

    with pytest.raises(SyncStoreError, match="sync_reserved_dataset_enrollment"):
        service.enroll_dataset(
            user_id=OWNER_ID,
            dataset_id="other-personal",
            domains=["notes.note", "notes.link"],
        )


def test_link_bootstrap_captures_live_and_tombstoned_heads_without_product_reapply(
    tmp_path: Path,
    product_db: CharactersRAGDB,
) -> None:
    bootstrapper = NotesLinkBootstrapper(product_db, batch_size=1)
    service, store = _service(tmp_path, product_db, bootstrapper=bootstrapper)
    initial = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
    )
    assert initial.dataset is not None
    _seed_note_heads(store, initial.dataset.dataset_id)
    versions_before = {item.edge_id: item.version for item in product_db.notes_link_store.snapshot()}

    ready = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
        requested_domains=[*M1_SYNC_DOMAINS, "notes.link"],
    )

    assert ready.dataset is not None
    assert ready.dataset.notes_link == {
        "state": "ready",
        "captured_count": 2,
        "expected_count": 2,
        "error_code": None,
    }
    heads = store.list_current_heads(
        ready.dataset.dataset_id,
        "notes.link",
        limit=10,
        offset=0,
    )
    assert [(head.object_id, head.operation, head.apply_status) for head in heads] == [
        (LIVE_EDGE_ID, "upsert", "applied"),
        (DELETED_EDGE_ID, "tombstone", "applied"),
    ]
    assert [head.entity_version for head in heads] == [1, 1]
    assert {item.edge_id: item.version for item in product_db.notes_link_store.snapshot()} == versions_before


def test_link_bootstrap_scans_source_in_bounded_keyset_pages(
    tmp_path: Path,
    product_db: CharactersRAGDB,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page_calls: list[tuple[str | None, int]] = []
    original_list_page = NotesLinkStore.list_page

    def list_page(
        store: NotesLinkStore,
        *,
        after_edge_id: str | None,
        limit: int,
        include_deleted_links: bool = False,
        include_deleted_endpoints: bool = False,
    ):
        page_calls.append((after_edge_id, limit))
        return original_list_page(
            store,
            after_edge_id=after_edge_id,
            limit=limit,
            include_deleted_links=include_deleted_links,
            include_deleted_endpoints=include_deleted_endpoints,
        )

    monkeypatch.setattr(NotesLinkStore, "list_page", list_page)
    monkeypatch.setattr(notes_link_bootstrap_module, "_SOURCE_PAGE_SIZE", 1)
    monkeypatch.setattr(
        NotesLinkStore,
        "snapshot",
        lambda *_args, **_kwargs: pytest.fail("bootstrap source scan must be paged"),
    )
    service, store = _service(
        tmp_path,
        product_db,
        bootstrapper=NotesLinkBootstrapper(product_db, batch_size=1),
    )
    initial = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
    )
    assert initial.dataset is not None
    _seed_note_heads(store, initial.dataset.dataset_id)

    ready = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
        requested_domains=[*M1_SYNC_DOMAINS, "notes.link"],
    )

    assert ready.dataset is not None and ready.dataset.notes_link["state"] == "ready"
    assert page_calls
    assert all(1 <= limit <= 201 for _cursor, limit in page_calls)
    assert any(cursor is not None for cursor, _limit in page_calls)


def test_link_bootstrap_resumes_same_id_and_fails_closed_on_source_drift(
    tmp_path: Path,
    product_db: CharactersRAGDB,
) -> None:
    calls = 0

    def interrupt_once(_completed: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise NotesLinkBootstrapInterrupted

    interrupted = NotesLinkBootstrapper(
        product_db,
        batch_size=1,
        after_group=interrupt_once,
    )
    service, store = _service(tmp_path, product_db, bootstrapper=interrupted)
    initial = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
    )
    assert initial.dataset is not None
    _seed_note_heads(store, initial.dataset.dataset_id)

    with pytest.raises(NotesLinkBootstrapInterrupted):
        service.bootstrap_profile(
            user_id=OWNER_ID,
            mode="offline_sync",
            device_id="device-1",
            requested_domains=[*M1_SYNC_DOMAINS, "notes.link"],
        )
    paused = store.get_dataset(initial.dataset.dataset_id)
    assert paused is not None
    bootstrap_id = paused.metadata["notes_link_v1"]["bootstrap_id"]

    service.notes_link_bootstrapper = NotesLinkBootstrapper(product_db, batch_size=1)
    ready = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
        requested_domains=[*M1_SYNC_DOMAINS, "notes.link"],
    )
    assert ready.dataset is not None
    stored = store.get_dataset(ready.dataset.dataset_id)
    assert stored is not None
    assert stored.metadata["notes_link_v1"]["bootstrap_id"] == bootstrap_id
    assert stored.metadata["notes_link_v1"]["state"] == "ready"

    # A ready bootstrap is immutable and remains ready even if product state changes later.
    NotesLinkStore(product_db).upsert(
        edge_id=LIVE_EDGE_ID,
        payload=_payload(SOURCE_ID, TARGET_ID, modified="2026-08-10T12:00:03+00:00"),
        expected_version=1,
    )
    replay = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
        requested_domains=[*M1_SYNC_DOMAINS, "notes.link"],
    )
    assert replay.dataset is not None and replay.dataset.notes_link["state"] == "ready"


def test_link_bootstrap_detects_source_drift_before_ready(
    tmp_path: Path,
    product_db: CharactersRAGDB,
) -> None:
    changed = False

    def drift(_completed: int) -> None:
        nonlocal changed
        if changed:
            return
        changed = True
        NotesLinkStore(product_db).upsert(
            edge_id=LIVE_EDGE_ID,
            payload=_payload(
                SOURCE_ID,
                TARGET_ID,
                modified="2026-08-10T12:00:04+00:00",
            ),
            expected_version=1,
        )

    bootstrapper = NotesLinkBootstrapper(product_db, batch_size=1, after_group=drift)
    service, store = _service(tmp_path, product_db, bootstrapper=bootstrapper)
    initial = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
    )
    assert initial.dataset is not None
    _seed_note_heads(store, initial.dataset.dataset_id)

    failed = service.bootstrap_profile(
        user_id=OWNER_ID,
        mode="offline_sync",
        device_id="device-1",
        requested_domains=[*M1_SYNC_DOMAINS, "notes.link"],
    )

    assert failed.dataset is not None
    assert failed.dataset.notes_link == {
        "state": "failed",
        "captured_count": 2,
        "expected_count": 2,
        "error_code": "notes_link_bootstrap_source_invalid",
    }
