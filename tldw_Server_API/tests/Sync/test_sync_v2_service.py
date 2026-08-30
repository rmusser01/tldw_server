from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from threading import Barrier, Event, Lock, get_ident
from typing import cast

import pytest

from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    SyncCapabilitiesResponse,
    SyncProfileResponse,
)
from tldw_Server_API.app.core.DB_Management import Sync_DB as sync_db_module
from tldw_Server_API.app.core.DB_Management.chacha.note_link_store import NotesLinkStore
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2 import service as sync_v2_service_module
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    AttachmentRefAdapter,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.domain_adapters.media import MediaMetadataAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes import NotesDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_link import NotesLinkDomainAdapter
from tldw_Server_API.app.core.Sync.v2.domain_adapters.source_cache import SourceCacheAdapter
from tldw_Server_API.app.core.Sync.v2.errors import (
    SyncIdempotencyConflictError,
    SyncMaterializationBusyError,
    SyncStoreError,
)
from tldw_Server_API.app.core.Sync.v2.materializers import MaterializationResult, NotesMaterializer
from tldw_Server_API.app.core.Sync.v2.materializers.guarded_product_mutation import (
    GUARD_REQUIRED_ROUTING_KEY,
)
from tldw_Server_API.app.core.Sync.v2.materializers.notes_link import NotesLinkMaterializer
from tldw_Server_API.app.core.Sync.v2.materializers.notes_organization import (
    NotesOrganizationMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    M1_SYNC_DOMAINS,
    NOTES_MOODBOARD_STUDIO_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    SYNC_V2_SUPPORTED_OPERATIONS,
    EncryptionPolicy,
    SyncConflict,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncDeviceBlobAckCreate,
    SyncDeviceBlobIdAckCreate,
    SyncDeviceCursor,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncDomain,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.mutation_group_validation import (
    mutation_group_plan_hash,
)
from tldw_Server_API.app.core.Sync.v2.notes_organization import organization_link_id
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Sync.notes_organization_test_support import (
    build_ready_notes_sync_stack,
)


def _clock() -> str:
    return "2026-05-10T12:00:00+00:00"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_service.db"))


@pytest.fixture(autouse=True)
def _ready_sync_v2_encryption_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SYNC_V2_AT_REST_ENCRYPTION_MODE", "managed_storage")
    monkeypatch.setenv("SYNC_V2_SERVER_TRUSTED_ENABLED", "true")


@pytest.fixture()
def registry() -> SyncAdapterRegistry:
    registry = SyncAdapterRegistry()
    registry.register(StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}))
    registry.register(StaticSyncAdapter(domain="chat.conversation", supported_adapter_versions={1}))
    registry.register(StaticSyncAdapter(domain="chat.message", supported_adapter_versions={1}))
    registry.register(StaticSyncAdapter(domain="attachment.ref", supported_adapter_versions={1}))
    return registry


WORKSPACE_DOMAINS = [
    cast(SyncDomain, "workspaces.workspace"),
    cast(SyncDomain, "workspaces.source_ref"),
]


def _workspace_registry() -> SyncAdapterRegistry:
    registry = SyncAdapterRegistry()
    for domain in WORKSPACE_DOMAINS:
        registry.register(StaticSyncAdapter(domain=domain, supported_adapter_versions={1}))
    return registry


def _wire_personal_context_components(
    registry: SyncAdapterRegistry,
) -> dict[SyncDomain, _OutcomeMaterializer]:
    materializers: dict[SyncDomain, _OutcomeMaterializer] = {}
    for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
        registry.register(
            StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
        )
        materializer = _OutcomeMaterializer()
        materializer.domain = domain
        materializers[domain] = materializer
    return materializers


def _workspace_service(
    sync_store: SyncV2Store,
    allowed_workspace_permissions: set[tuple[str, str, str]],
    *,
    blob_store: LocalSyncBlobStore | None = None,
) -> SyncV2Service:
    def can_sync_workspace(user_id: str, workspace_id: str, permission: str) -> bool:
        return (user_id, workspace_id, permission) in allowed_workspace_permissions

    return SyncV2Service(
        store=sync_store,
        adapters=_workspace_registry(),
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=blob_store,
        workspace_access_checker=can_sync_workspace,
        settings=SyncV2Settings(
            supports_attachments=blob_store is not None,
            max_blob_bytes=4096,
            max_chunk_bytes=1024,
            server_trusted_encryption=_ready_encryption(),
        ),
    )


@pytest.fixture()
def sync_service(sync_store: SyncV2Store, registry: SyncAdapterRegistry) -> SyncV2Service:
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            max_batch_size=10,
            max_pull_page_size=2,
            max_envelope_payload_bytes=1024,
            max_attachment_bytes=4096,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    return service


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes.note",
        "entity_id": "note-1",
        "stable_key": "note:note-1",
        "operation": "upsert",
        "device_id": "device-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "entity_version": "v1",
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": "ciphertext:opaque",
        "payload_clear": {"status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 24,
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _m1_note_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "payload": {"title": "Research note"},
        "payload_hash": "sha256:note-1",
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _store_pulled_envelope_at_canonical_cursor(
    store: SyncV2Store,
    envelope: SyncEnvelope,
) -> SyncEnvelope:
    """Store a pulled envelope with its canonical cursor in a fresh SQLite store."""

    assert envelope.server_cursor is not None
    if envelope.server_cursor > 1:
        with store.db.backend.transaction() as connection:
            store.db.execute(
                "UPDATE sqlite_sequence SET seq = ? WHERE name = 'sync_envelopes'",
                (envelope.server_cursor - 1,),
                connection=connection,
            )
    stored = store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=envelope.dataset_id,
            client_envelope_id=envelope.client_envelope_id,
            domain=envelope.domain,
            operation=envelope.operation,
            object_id=envelope.object_id,
            device_id=envelope.device_id,
            client_sequence=envelope.client_sequence,
            base_server_cursor=envelope.base_server_cursor,
            base_object_revision=envelope.base_object_revision,
            base_object_hash=envelope.base_object_hash,
            object_revision=envelope.object_revision,
            schema_version=envelope.schema_version,
            routing_metadata=envelope.routing_metadata,
            payload=envelope.payload,
            payload_hash=envelope.payload_hash,
            created_at_client=envelope.created_at_client,
            deleted=envelope.deleted,
            encryption_metadata=envelope.encryption_metadata,
            adapter_version=envelope.adapter_version,
            status="accepted",
            apply_status="pending",
        )
    )
    assert stored.server_cursor == envelope.server_cursor
    return stored


def _workspace_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "workspace-dataset",
        "client_envelope_id": "workspace-env-1",
        "domain": "workspaces.workspace",
        "operation": "upsert",
        "object_id": "workspace-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "payload": {"name": "Shared research"},
        "payload_hash": "sha256:workspace-1",
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _source_cache_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "source-cache-env-1",
        "domain": "source_cache.entry",
        "operation": "upsert",
        "object_id": "source-1:sha256-source",
        "device_id": "device-1",
        "client_sequence": 1,
        "object_revision": 1,
        "schema_version": 1,
        "stable_key": "source_cache.entry:source-1:sha256-source",
        "routing_metadata": {"entity_kind": "source_cache_entry"},
        "payload": {
            "entity_kind": "source_cache_entry",
            "source_id": "source-1",
            "content_hash": "sha256:source",
            "provenance": {"kind": "url", "uri": "https://example.test/source"},
        },
        "payload_hash": "sha256:source-cache-entry",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _media_envelope(domain: SyncDomain = "media.item", **overrides) -> SyncEnvelopeCreate:
    object_id = {
        "media.item": "media-1",
        "media.keyword": "keyword-1",
        "media.keyword_link": "media-1:keyword-1",
    }[domain]
    payload_by_domain = {
        "media.item": {"media_id": "media-1", "media_type": "video", "title": "Lecture"},
        "media.keyword": {"keyword_id": "keyword-1", "name": "research"},
        "media.keyword_link": {"media_id": "media-1", "keyword_id": "keyword-1"},
    }
    sequence_by_domain = {
        "media.item": 1,
        "media.keyword": 2,
        "media.keyword_link": 3,
    }
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": f"{domain}-env-1",
        "domain": domain,
        "operation": "upsert",
        "object_id": object_id,
        "device_id": "device-1",
        "client_sequence": sequence_by_domain[domain],
        "object_revision": 1,
        "schema_version": 1,
        "stable_key": f"{domain}:{object_id}",
        "payload": payload_by_domain[domain],
        "payload_hash": f"sha256:{domain}-v1",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-10T00:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _register_devices(
    service: SyncV2Service,
    user_id: str,
    *device_ids: str,
) -> None:
    for device_id in device_ids:
        service.register_device(
            user_id=user_id,
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
        )


class _OutcomeMaterializer:
    domain: SyncDomain = "notes.note"

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
        if envelope.object_id == "note-conflict":
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="projection_conflict",
                apply_error_message="resolution projection conflicted",
            )
            return MaterializationResult(
                status="conflict",
                conflict_type="projection_conflict",
                error_code="projection_conflict",
                message="resolution projection conflicted",
            )
        if envelope.object_id == "note-fail":
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="projection_failed",
                apply_error_message="projection is replayable",
            )
            return MaterializationResult(
                status="failed",
                error_code="projection_failed",
                message="projection is replayable",
            )
        object_revision = envelope.object_revision or 1
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=envelope.dataset_id,
                domain=envelope.domain,
                object_id=envelope.object_id,
                object_revision=object_revision,
                object_hash=envelope.payload_hash or "",
                latest_server_cursor=envelope.server_cursor,
                deleted=envelope.operation == "tombstone",
            )
        )
        store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
        return MaterializationResult(status="applied")


class _CountingMaterializer(_OutcomeMaterializer):
    def __init__(self) -> None:
        self.calls = 0

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
        self.calls += 1
        return super().apply(envelope, store=store)


class _RaisingMaterializer:
    domain: SyncDomain = "notes.note"

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
        raise RuntimeError("projection exploded at /private/user-data")


class _AcceptedConflictThenApplyMaterializer(_OutcomeMaterializer):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.release_resolution: Event | None = None
        self.resolution_entered = Event()
        self.later_entered = Event()

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
        self.calls.append(envelope.client_envelope_id)
        if envelope.client_envelope_id == "env-materialization-conflict":
            assert envelope.server_cursor is not None
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="conflict",
                apply_error_code="projection_conflict",
                apply_error_message="review required",
            )
            return MaterializationResult(
                status="conflict",
                conflict_type="projection_conflict",
                error_code="projection_conflict",
            )
        if envelope.client_envelope_id == "env-resolution-copy":
            self.resolution_entered.set()
            if self.release_resolution is not None:
                assert self.release_resolution.wait(timeout=5)
        if envelope.client_envelope_id == "env-after-resolution":
            self.later_entered.set()
        return super().apply(envelope, store=store)


def _accepted_materialization_conflict_service(
    sync_store: SyncV2Store,
    *,
    real_notes_adapter: bool = False,
) -> tuple[SyncV2Service, _AcceptedConflictThenApplyMaterializer]:
    materializer = _AcceptedConflictThenApplyMaterializer()
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [
                NotesDomainAdapter()
                if real_notes_adapter
                else StaticSyncAdapter(
                    domain="notes.note",
                    supported_adapter_versions={1},
                )
            ]
        ),
        materializers={"notes.note": materializer},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    return service, materializer


def _accepted_conflict_after_applied_predecessor(
    sync_store: SyncV2Store,
    note_db: CharactersRAGDB,
) -> tuple[SyncV2Service, SyncEnvelope, SyncEnvelope, SyncConflict]:
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry([NotesDomainAdapter()]),
        materializers={"notes.note": NotesMaterializer(note_db)},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    baseline_result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-applied-predecessor",
                object_id="note-original",
                client_sequence=1,
                object_revision=1,
                payload={"title": "Applied", "content": "Projected baseline"},
                payload_hash="sha256:applied",
            )
        ],
    )
    baseline = sync_store.get_envelope_by_server_cursor(
        baseline_result.accepted[0].server_sequence
    )
    assert baseline is not None
    source = sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-materialization-conflict",
            object_id="note-original",
            client_sequence=2,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash="sha256:applied",
            object_revision=2,
            payload={"title": "Conflict", "content": "Never projected"},
            payload_hash="sha256:conflict",
            status="accepted",
        )
    )
    source = sync_store.mark_envelope_apply_status(
        source.server_cursor,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-materialization",
            dataset_id="dataset-1",
            domain="notes.note",
            entity_id="note-original",
            conflict_type="projection_conflict",
            local_envelope_id=source.client_envelope_id,
            server_sequence=source.server_cursor,
        )
    )
    return service, baseline, source, conflict


def _accepted_keyword_conflict_after_applied_predecessor(
    tmp_path: Path,
) -> tuple[CharactersRAGDB, SyncV2Store, SyncV2Service, str, SyncEnvelope, SyncEnvelope, SyncConflict]:
    note_db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    profile = service.profile(user_id="user-1", device_id="frontend-device")
    dataset_id = profile.active_dataset_id
    assert dataset_id is not None
    service.register_device(
        user_id="user-1",
        display_name="device-2",
        client_type="chatbook",
        device_id="device-2",
        capabilities={
            "requested_domains": [*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]
        },
    )
    keyword_id = "11111111-1111-4111-8111-111111111111"
    baseline_result = service.push(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="frontend-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-keyword-baseline",
                domain="notes.keyword",
                operation="upsert",
                object_id=keyword_id,
                device_id="frontend-device",
                client_sequence=1,
                object_revision=1,
                schema_version=1,
                payload={"keyword": "Baseline"},
                payload_hash="sha256:keyword-baseline",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
            )
        ],
    )
    baseline = sync_store.get_envelope_by_server_cursor(
        baseline_result.accepted[0].server_sequence
    )
    assert baseline is not None
    source = sync_store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-keyword-conflict-source",
            domain="notes.keyword",
            operation="upsert",
            object_id=keyword_id,
            device_id="frontend-device",
            client_sequence=2,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            schema_version=1,
            payload={"keyword": "Never projected"},
            payload_hash="sha256:keyword-conflict",
            encryption_metadata={"policy": "server_trusted_v1"},
            adapter_version=1,
            status="accepted",
        )
    )
    source = sync_store.mark_envelope_apply_status(
        source.server_cursor,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-keyword-materialization",
            dataset_id=dataset_id,
            domain="notes.keyword",
            entity_id=keyword_id,
            conflict_type="projection_conflict",
            local_envelope_id=source.client_envelope_id,
            server_sequence=source.server_cursor,
        )
    )
    return note_db, sync_store, service, dataset_id, baseline, source, conflict


def test_notes_link_concurrent_product_edit_records_safe_conflict_and_skip_is_idempotent(
    tmp_path: Path,
) -> None:
    note_db = CharactersRAGDB(tmp_path / "notes-link-conflict.db", client_id="user-1")
    source_note_id = "11111111-1111-4111-8111-111111111111"
    target_note_id = "22222222-2222-4222-8222-222222222222"
    edge_id = "33333333-3333-4333-8333-333333333333"
    created_at = "2026-08-10T12:00:00+00:00"

    def link_payload(*, weight: float, modified_at: str) -> dict[str, object]:
        return {
            "source_note_id": source_note_id,
            "target_note_id": target_note_id,
            "type": "manual",
            "directed": False,
            "weight": weight,
            "label": None,
            "properties": {},
            "created_at": created_at,
            "last_modified": modified_at,
            "created_by": "device-1",
        }

    try:
        for note_id in (source_note_id, target_note_id):
            note_db.note_store.add_note(note_id, "body", note_id=note_id)
        sync_store = SyncV2Store(
            SyncDatabase(sqlite_path=tmp_path / "notes-link-conflict-sync.db")
        )
        service = SyncV2Service(
            store=sync_store,
            adapters=SyncAdapterRegistry(
                [
                    StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}),
                    NotesLinkDomainAdapter(),
                ]
            ),
            materializers={"notes.link": NotesLinkMaterializer(note_db)},
            clock=_clock,
            id_factory=lambda prefix: f"{prefix}-notes-link",
            settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
        )
        _register_devices(service, "user-1", "device-1")
        service.enroll_dataset(
            user_id="user-1",
            dataset_id="dataset-1",
            domains=["notes.note"],
        )
        sync_store.begin_notes_link_bootstrap(
            "dataset-1",
            owner_user_id="user-1",
            bootstrap_id="notes-link-conflict-ready",
        )
        sync_store.transition_notes_link_bootstrap(
            "dataset-1",
            bootstrap_id="notes-link-conflict-ready",
            expected_state="initializing",
            state="ready",
            captured_count=0,
            expected_count=0,
            source_hash=None,
            ready_verifier=lambda: True,
        )
        for index, note_id in enumerate((source_note_id, target_note_id), start=1):
            note = sync_store.insert_envelope(
                SyncEnvelopeCreate(
                    dataset_id="dataset-1",
                    client_envelope_id=f"env-notes-link-note-{index}",
                    domain="notes.note",
                    operation="upsert",
                    object_id=note_id,
                    device_id="device-1",
                    object_revision=1,
                    payload={"title": note_id, "content": "body"},
                    payload_hash=f"sha256:notes-link-note-{index}",
                    created_at_client=created_at,
                    apply_status="applied",
                )
            )
            sync_store.upsert_object_state(
                SyncObjectState(
                    dataset_id="dataset-1",
                    domain="notes.note",
                    object_id=note_id,
                    object_revision=1,
                    object_hash=note.payload_hash or "",
                    latest_server_cursor=note.server_cursor or 0,
                    deleted=False,
                )
            )
        baseline_result = service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[
                SyncEnvelopeCreate(
                    dataset_id="dataset-1",
                    client_envelope_id="env-notes-link-baseline",
                    domain="notes.link",
                    operation="upsert",
                    object_id=edge_id,
                    device_id="device-1",
                    client_sequence=1,
                    object_revision=1,
                    entity_version=1,
                    payload=link_payload(weight=1.0, modified_at=created_at),
                    payload_hash="sha256:notes-link-baseline",
                    created_at_client=created_at,
                )
            ],
        )
        baseline = sync_store.get_envelope_by_server_cursor(
            baseline_result.accepted[0].server_sequence
        )
        assert baseline is not None and baseline.apply_status == "applied"
        NotesLinkStore(note_db).upsert(
            edge_id=edge_id,
            payload=link_payload(
                weight=9.0,
                modified_at="2026-08-10T12:00:01+00:00",
            ),
            expected_version=1,
        )

        pushed = service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[
                SyncEnvelopeCreate(
                    dataset_id="dataset-1",
                    client_envelope_id="env-notes-link-concurrent",
                    domain="notes.link",
                    operation="upsert",
                    object_id=edge_id,
                    device_id="device-1",
                    client_sequence=2,
                    base_server_cursor=baseline.server_cursor,
                    base_object_revision=baseline.object_revision,
                    base_object_hash=baseline.payload_hash,
                    base_version=1,
                    object_revision=2,
                    entity_version=2,
                    payload=link_payload(
                        weight=2.0,
                        modified_at="2026-08-10T12:00:02+00:00",
                    ),
                    payload_hash="sha256:notes-link-concurrent",
                    created_at_client="2026-08-10T12:00:02+00:00",
                )
            ],
        )
        stored_conflict = sync_store.get_conflict(pushed.conflicts[0].conflict_id)
        assert stored_conflict is not None
        assert stored_conflict.conflict_type == "notes_link_product_conflict"
        assert stored_conflict.metadata == {"reason": "product_state_conflict"}
        assert "source_note_id" not in str(stored_conflict)

        first = service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=stored_conflict.conflict_id,
            action="skip",
            resolved_by_device_id="device-1",
        )
        replayed = service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=stored_conflict.conflict_id,
            action="skip",
            resolved_by_device_id="device-1",
        )
        source = sync_store.get_envelope_by_server_cursor(stored_conflict.server_cursor)
        assert replayed == first
        assert source is not None and source.apply_status == "superseded"
    finally:
        note_db.close_connection()


def test_capabilities_returns_protocol_domains_limits_and_encryption_policies(
    sync_service: SyncV2Service,
):
    capabilities = sync_service.capabilities()

    assert capabilities.protocol_version == "sync-v2-m1"
    assert capabilities.min_supported_protocol_version == "sync-v2-m1"
    assert capabilities.supported_domains == [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
        "workspaces.workspace",
        "workspaces.source_ref",
        "source_cache.entry",
        "media.item",
        "media.keyword",
        "media.keyword_link",
        "notes.keyword",
        "notes.keyword_link",
        "notes.keyword_collection",
        "notes.keyword_collection_link",
        "notes.folder",
        "notes.folder_link",
        "notes.link",
        *PERSONAL_CONTEXT_SYNC_DOMAINS,
    ]
    assert {
        domain: capabilities.operations[domain]
        for domain in NOTES_ORGANIZATION_DOMAINS
    } == {
        "notes.keyword": ["upsert", "tombstone"],
        "notes.keyword_link": ["upsert", "tombstone"],
        "notes.keyword_collection": ["upsert", "tombstone"],
        "notes.keyword_collection_link": ["upsert", "tombstone"],
        "notes.folder": ["upsert", "tombstone"],
        "notes.folder_link": ["upsert", "tombstone"],
    }
    assert capabilities.operations["notes.link"] == ["upsert", "tombstone"]
    assert capabilities.max_batch_size == 10
    assert capabilities.max_envelope_payload_bytes == 1024
    assert capabilities.max_attachment_bytes == 4096
    assert capabilities.encryption_policies == ["server_trusted_v1"]
    assert capabilities.supports_attachments is False


def test_personal_context_capability_fails_closed_without_profile_key(
    monkeypatch: pytest.MonkeyPatch,
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    monkeypatch.delenv("TLDW_PERSONAL_CONTEXT_MASTER_KEY", raising=False)
    materializers = _wire_personal_context_components(registry)
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers=materializers,
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="personal-context-dataset",
            owner_user_id="user-1",
            domains=list(PERSONAL_CONTEXT_SYNC_DOMAINS),
        )
    )

    capabilities = service.capabilities(
        user_id="user-1",
        dataset_id="personal-context-dataset",
    )

    assert capabilities.personal_context.available is False
    assert capabilities.personal_context.blockers == (
        "personal_context_profile_key_unavailable",
    )
    assert capabilities.personal_context.authorization_policy == "server_trusted_v1"
    assert all(
        capabilities.writable_adapter_versions[domain] == []
        for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
    )


def test_personal_context_capability_waits_for_transport_components(
    monkeypatch: pytest.MonkeyPatch,
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=",
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )

    capability = service.capabilities().personal_context

    assert capability.available is False
    assert capability.blockers == ("personal_context_transport_unavailable",)
    assert capability.min_schema_version == 1
    assert capability.max_schema_version == 1
    assert capability.max_record_bytes == 16_384
    capabilities = service.capabilities()
    assert all(
        capabilities.supported_adapter_versions[domain] == []
        for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
    )


def test_personal_context_capability_is_available_when_fully_wired(
    monkeypatch: pytest.MonkeyPatch,
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=",
    )
    materializers = _wire_personal_context_components(registry)
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers=materializers,
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )

    capabilities = service.capabilities()

    assert capabilities.personal_context.available is True
    assert capabilities.personal_context.blockers == ()
    assert all(
        capabilities.supported_adapter_versions[domain] == [1]
        for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
    )


def test_personal_context_capability_requires_server_trusted_readiness(
    monkeypatch: pytest.MonkeyPatch,
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=",
    )
    unavailable_encryption = server_trusted_encryption_status_from_config(
        mode=None,
        server_trusted_enabled=False,
        auth_mode="multi_user",
    )
    materializers = _wire_personal_context_components(registry)
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers=materializers,
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=unavailable_encryption),
    )

    capability = service.capabilities().personal_context

    assert capability.available is False
    assert capability.blockers == ("personal_context_server_trusted_unavailable",)


def test_personal_context_capability_requires_supported_shared_schema(
    monkeypatch: pytest.MonkeyPatch,
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        "MDEyMzQ1Njc4OWFiY2RlZjAxMjM0NTY3ODlhYmNkZWY=",
    )
    monkeypatch.setattr(sync_v2_service_module, "SERIALIZED_SCHEMA_VERSION", 2)
    materializers = _wire_personal_context_components(registry)
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers=materializers,
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )

    capability = service.capabilities().personal_context

    assert capability.available is False
    assert capability.blockers == ("personal_context_schema_unsupported",)


def test_configured_moodboard_studio_domains_stay_private_in_capabilities(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    dormant = set(NOTES_MOODBOARD_STUDIO_DOMAINS)
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            supported_domains=[
                *SYNC_V2_SUPPORTED_DOMAINS,
                *NOTES_MOODBOARD_STUDIO_DOMAINS,
            ],
            operations={
                **{
                    domain: list(operations)
                    for domain, operations in SYNC_V2_SUPPORTED_OPERATIONS.items()
                },
                **{
                    domain: ["upsert", "tombstone"]
                    for domain in NOTES_MOODBOARD_STUDIO_DOMAINS
                },
            },
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-forged-moodboard-studio",
            owner_user_id="user-1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_moodboard_v1": {
                    "state": "ready",
                    "source_cursor": "00000000-0000-4000-8000-000000000101",
                    "source_count": 1,
                    "source_fingerprint": "a" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "notes_moodboard_note_v1": {
                    "state": "ready",
                    "source_cursor": (
                        "00000000-0000-4000-8000-000000000101|"
                        "00000000-0000-4000-8000-000000000201"
                    ),
                    "source_count": 1,
                    "source_fingerprint": "b" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "notes_studio_document_v1": {
                    "state": "ready",
                    "source_cursor": "00000000-0000-4000-8000-000000000301",
                    "source_count": 1,
                    "source_fingerprint": "c" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "moodboard_capture_enabled": True,
                "studio_document_capture_enabled": True,
            },
        )
    )

    core_capabilities = service.capabilities()
    scoped_capabilities = service.capabilities(
        user_id="user-1",
        dataset_id="dataset-forged-moodboard-studio",
    )
    enrollment = service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-forged-moodboard-studio",
    )
    profile = service.profile(user_id="user-1")
    api_capabilities = SyncCapabilitiesResponse(**asdict(core_capabilities))
    api_profile = SyncProfileResponse(**asdict(profile))

    for capabilities in (
        core_capabilities,
        scoped_capabilities,
        profile.capabilities,
        api_capabilities,
        api_profile.capabilities,
    ):
        domains = (
            capabilities.domains
            if isinstance(capabilities, SyncCapabilitiesResponse)
            else capabilities.supported_domains
        )
        assert dormant.isdisjoint(domains)
        assert dormant.isdisjoint(capabilities.operations)
        assert dormant.isdisjoint(capabilities.domain_schemas)
        assert dormant.isdisjoint(capabilities.supported_adapter_versions)
        assert dormant.isdisjoint(capabilities.writable_adapter_versions)

    assert dormant.isdisjoint(enrollment.dataset.domains)


def test_capabilities_warn_when_client_private_policy_is_advertised(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(
            encryption_policies=["server_trusted_v1", "client_private_v1"],
            server_trusted_encryption=_ready_encryption(),
        ),
    )

    capabilities = service.capabilities()

    assert capabilities.compatibility_flags["server_frontend_client_private_mutation"] is False
    assert any(
        warning.get("code") == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
        for warning in capabilities.warnings
    )


def test_capabilities_can_advertise_m2_resumable_blob_transfer(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(
            protocol_version="sync-v2-m2",
            min_supported_protocol_version="sync-v2-m1",
            supports_attachments=True,
            max_attachment_bytes=8192,
            max_blob_bytes=16384,
            max_chunk_bytes=1024,
            max_active_blob_uploads=3,
            user_blob_quota_bytes=65536,
            server_trusted_encryption=_ready_encryption(),
        ),
    )

    capabilities = service.capabilities()

    assert capabilities.protocol_version == "sync-v2-m2"
    assert capabilities.min_supported_protocol_version == "sync-v2-m1"
    assert capabilities.supports_attachments is True
    assert capabilities.blob_transfer == {
        "supported": True,
        "resumable_upload": True,
        "resumable_download": True,
        "chunk_checksums": True,
        "full_checksum": "sha256",
        "storage_backend": "local_fs",
    }
    assert capabilities.quota == {
        "max_blob_bytes": 16384,
        "max_chunk_bytes": 1024,
        "max_active_uploads": 3,
        "user_blob_quota_bytes": 65536,
        "reserved_blob_bytes": 0,
        "used_blob_bytes": 0,
    }


def test_default_attachment_ref_v2_rollout_gate_is_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry

    monkeypatch.delenv("SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC", raising=False)
    default_sync_v2_registry.cache_clear()

    adapter = default_sync_v2_registry().get("attachment.ref")
    assert adapter.supported_adapter_versions == {1, 2}
    assert adapter.v2_writes_enabled is False


def test_attachment_ref_v2_rollout_gate_can_be_enabled_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry

    monkeypatch.setenv("SYNC_V2_ENABLE_NOTES_ATTACHMENT_SYNC", "true")
    default_sync_v2_registry.cache_clear()

    adapter = default_sync_v2_registry().get("attachment.ref")
    assert adapter.v2_writes_enabled is True


def test_core_capabilities_without_dataset_are_conservative(
    sync_service: SyncV2Service,
) -> None:
    capabilities = sync_service.capabilities()

    assert capabilities.supported_adapter_versions["attachment.ref"] == [1, 2]
    assert all(not versions for versions in capabilities.writable_adapter_versions.values())


def test_core_capabilities_only_report_enrolled_domains_as_writable(
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    capabilities = sync_service.capabilities(
        user_id="user-1",
        dataset_id="dataset-1",
    )

    assert capabilities.writable_adapter_versions["notes.note"] == [1]
    assert capabilities.writable_adapter_versions["chat.conversation"] == []
    assert capabilities.writable_adapter_versions["media.item"] == []


@pytest.mark.parametrize(
    ("gate_enabled", "blob_enabled", "state", "domains", "policy", "expected"),
    [
        (False, True, "ready", ["notes.note", "attachment.ref"], "server_trusted_v1", []),
        (True, False, "ready", ["notes.note", "attachment.ref"], "server_trusted_v1", []),
        (True, True, "initializing", ["notes.note", "attachment.ref"], "server_trusted_v1", []),
        (True, True, "ready", ["attachment.ref"], "server_trusted_v1", []),
        (True, True, "ready", ["notes.note", "attachment.ref"], "server_trusted_v1", [2]),
    ],
)
def test_core_capabilities_bind_attachment_writability_to_selected_dataset(
    sync_service: SyncV2Service,
    gate_enabled: bool,
    blob_enabled: bool,
    state: str,
    domains: list[SyncDomain],
    policy: EncryptionPolicy,
    expected: list[int],
) -> None:
    sync_service.settings = replace(
        sync_service.settings,
        supports_attachments=blob_enabled,
    )
    sync_service.adapters.register(
        AttachmentRefAdapter(v2_writes_enabled=gate_enabled)
    )
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy=policy,
            domains=domains,
            metadata={"notes_attachment_v2": {"state": state}},
        )
    )

    capabilities = sync_service.capabilities(
        user_id="user-1",
        dataset_id="dataset-1",
    )

    assert capabilities.supported_adapter_versions["attachment.ref"] == [1, 2]
    assert capabilities.writable_adapter_versions["attachment.ref"] == expected


def test_device_registration_creates_and_refreshes_same_device(sync_service: SyncV2Service):
    first = sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        client_version="0.1.0",
        capabilities={"domains": ["notes.note"]},
        device_id="device-1",
    )
    refreshed = sync_service.register_device(
        user_id="user-1",
        display_name="Renamed Laptop",
        client_type="chatbook",
        client_version="0.1.1",
        capabilities={"domains": ["notes.note", "chat.conversation"]},
        device_id="device-1",
    )

    assert refreshed.device.device_id == first.device.device_id
    assert refreshed.device.registered_at == first.device.registered_at
    assert refreshed.device.display_name == "Renamed Laptop"
    assert refreshed.device.client_version == "0.1.1"
    assert refreshed.device.capabilities == {"domains": ["notes.note", "chat.conversation"]}


def test_device_registration_rejects_cross_user_device_takeover(sync_service: SyncV2Service):
    sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="shared-device",
    )

    with pytest.raises(SyncStoreError):
        sync_service.register_device(
            user_id="user-2",
            display_name="Other Laptop",
            client_type="chatbook",
            device_id="shared-device",
        )


def test_active_device_registration_cannot_remove_advertised_adapter_versions(
    sync_service: SyncV2Service,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-new",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
        },
    )

    with pytest.raises(SyncStoreError, match="adapter version"):
        sync_service.register_device(
            user_id="user-1",
            display_name="Laptop refreshed",
            client_type="chatbook",
            device_id="device-new",
            capabilities={
                "requested_domains": ["notes.note"],
                "supported_adapter_versions": {"notes.note": [2]},
            },
        )

    stored = sync_service.store.get_device("user-1", "device-new")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "notes.note": [1, 2]
    }


def test_device_capability_patch_merges_and_adds_adapter_versions_monotonically(
    sync_service: SyncV2Service,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-new",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1]},
            "theme": "dark",
        },
    )

    updated = sync_service.update_device(
        user_id="user-1",
        device_id="device-new",
        capabilities={
            "supported_adapter_versions": {
                "notes.note": [1, 2],
                "attachment.ref": [2],
            },
            "telemetry": True,
        },
    )

    assert updated.capabilities == {
        "requested_domains": ["notes.note", "attachment.ref"],
        "supported_adapter_versions": {
            "notes.note": [1, 2],
            "attachment.ref": [2],
        },
        "theme": "dark",
        "telemetry": True,
    }


def test_concurrent_device_adapter_version_additions_are_unioned_atomically(
    sync_service: SyncV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-new",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1]},
        },
    )
    original_get_device = sync_service.store.get_device
    reads_by_thread: dict[int, int] = {}
    reads_lock = Lock()
    prewrite_barrier = Barrier(2)

    def get_device_after_both_service_preflights(
        user_id: str,
        device_id: str,
    ):
        device = original_get_device(user_id, device_id)
        if device_id != "device-new":
            return device
        thread_id = get_ident()
        with reads_lock:
            reads_by_thread[thread_id] = reads_by_thread.get(thread_id, 0) + 1
            read_count = reads_by_thread[thread_id]
        if read_count == 2:
            prewrite_barrier.wait(timeout=5)
        return device

    monkeypatch.setattr(
        sync_service.store,
        "get_device",
        get_device_after_both_service_preflights,
    )

    def add_version(version: int):
        return sync_service.update_device(
            user_id="user-1",
            device_id="device-new",
            capabilities={
                "supported_adapter_versions": {"notes.note": [1, version]}
            },
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(add_version, [2, 3]))

    assert len(results) == 2
    stored = original_get_device("user-1", "device-new")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "notes.note": [1, 2, 3]
    }


def test_legacy_device_capability_patch_preserves_implicit_v1_versions(
    sync_service: SyncV2Service,
) -> None:
    with pytest.raises(SyncStoreError, match="adapter version"):
        sync_service.update_device(
            user_id="user-1",
            device_id="device-1",
            capabilities={
                "supported_adapter_versions": {"attachment.ref": [2]}
            },
        )

    updated = sync_service.update_device(
        user_id="user-1",
        device_id="device-1",
        capabilities={
            "supported_adapter_versions": {"attachment.ref": [1, 2]}
        },
    )

    assert updated.capabilities["requested_domains"] == list(M1_SYNC_DOMAINS)
    assert updated.capabilities["supported_adapter_versions"] == {
        **{domain: [1] for domain in M1_SYNC_DOMAINS},
        "attachment.ref": [1, 2],
    }


@pytest.mark.parametrize(
    "version_map",
    [
        {"notes.note": [2]},
        {"attachment.ref": [True]},
        None,
    ],
)
def test_device_capability_patch_rejects_removal_or_malformed_adapter_map(
    sync_service: SyncV2Service,
    version_map: dict[str, list[object]] | None,
) -> None:
    original = sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1]},
            "theme": "dark",
        },
    ).device

    with pytest.raises(SyncStoreError, match="adapter version"):
        sync_service.update_device(
            user_id="user-1",
            device_id="device-1",
            capabilities={"supported_adapter_versions": version_map},
        )

    stored = sync_service.store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities == original.capabilities


def test_pending_device_cannot_push_until_authorized(sync_service: SyncV2Service):
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    sync_service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-pending",
            user_id="user-1",
            display_name="New laptop",
            client_type="chatbook",
            status="pending_authorization",
        )
    )

    with pytest.raises(SyncStoreError, match="Sync device was not found or is not accessible"):
        sync_service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-pending",
            envelopes=[
                _m1_note_envelope(
                    client_envelope_id="env-pending",
                    device_id="device-pending",
                )
            ],
        )


def test_revoked_device_cannot_sync_or_start_device_scoped_blob_upload(
    tmp_path: Path,
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync-blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_attachment_bytes=8192,
            max_blob_bytes=8192,
            max_chunk_bytes=1024,
            user_blob_quota_bytes=65536,
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
        domains=["notes.note", "attachment.ref"],
    )
    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="notes.note",
        entity_id="note-before-revoke",
        attachment_id="attachment-before-revoke",
        content_type="application/octet-stream",
        size_bytes=1024,
        payload_hash=_sha256(b"blob-before-revoke"),
        chunk_size=1024,
        chunk_count=1,
    )
    service.store.revoke_device(
        user_id="user-1",
        device_id="device-1",
        reason="lost_device",
        revoke_key_records=True,
    )

    with pytest.raises(SyncStoreError, match="Sync device was not found or is not accessible"):
        service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            envelopes=[_m1_note_envelope()],
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found or is not accessible"):
        service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found or is not accessible"):
        service.store_key_recovery_bundle(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:opaque",
            kdf_metadata={"algorithm": "argon2id"},
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found or is not accessible"):
        service.upload_blob_chunk(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=session.upload_id,
            chunk_index=0,
            offset_bytes=0,
            chunk_payload=b"b" * 1024,
            chunk_hash=_sha256(b"b" * 1024),
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found or is not accessible"):
        service.create_blob_upload_session(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            domain="attachment.ref",
            entity_id="attachment-1",
            attachment_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=1024,
            payload_hash=_sha256(b"blob"),
            chunk_size=1024,
            chunk_count=1,
        )


def test_background_policy_lease_and_status_aggregation(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )
    sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-device-1",
            object_id="note-1",
            device_id="device-1",
            client_sequence=1,
            payload_hash="sha256:note-1",
            apply_status="applied",
        )
    )
    failed = sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-device-2",
            object_id="note-2",
            device_id="device-2",
            client_sequence=1,
            payload_hash="sha256:note-2",
            apply_status="failed",
        )
    )
    sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-1",
            dataset_id="dataset-1",
            domain="notes.note",
            object_id="note-2",
            conflict_type="version_divergence",
            local_envelope_id="env-device-2",
            server_cursor=failed.server_sequence,
        )
    )
    sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            last_pulled_sequence=1,
        )
    )

    default_policy = sync_service.get_background_policy(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
    )
    paused_policy = sync_service.update_background_policy(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        enabled=False,
        paused_reason="user_paused",
        pending_local_changes=True,
    )
    resumed_policy = sync_service.update_background_policy(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        enabled=True,
    )
    paused_policy = sync_service.update_background_policy(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        enabled=False,
        paused_reason="user_paused",
        pending_local_changes=True,
    )
    lease = sync_service.acquire_background_lease(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        lease_id="lease-1",
        ttl_seconds=120,
    )
    held = sync_service.acquire_background_lease(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        lease_id="lease-2",
        ttl_seconds=120,
    )
    status = sync_service.background_status(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
    )

    assert default_policy.enabled is True
    assert default_policy.max_batch_size == 10
    assert paused_policy.enabled is False
    assert paused_policy.paused_reason == "user_paused"
    assert paused_policy.pending_local_changes is True
    assert resumed_policy.enabled is True
    assert resumed_policy.paused_reason is None
    assert lease.acquired is True
    assert lease.status == "acquired"
    assert held.acquired is False
    assert held.status == "held_by_other"
    assert status.policy.enabled is False
    assert status.lease is not None
    assert status.lease.lease_id == "lease-1"
    assert status.conflict_count == 1
    assert status.replayable_failure_count == 1
    assert status.restore_completeness == "blocked_by_conflicts"
    notes_status = {item.domain: item for item in status.domains}["notes.note"]
    assert notes_status.last_server_sequence == failed.server_sequence
    assert notes_status.last_pulled_sequence == 1
    assert notes_status.cursor_lag_count == 1
    assert notes_status.unresolved_conflicts == 1
    assert notes_status.replayable_failures == 1
    assert notes_status.last_successful_push_at is not None
    assert notes_status.last_successful_pull_at is not None


def test_background_health_counts_pending_projection_as_replayable(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-pending-projection",
            object_id="note-pending",
            device_id="device-1",
            client_sequence=1,
            payload_hash="sha256:note-pending",
            apply_status="pending",
        )
    )

    status = sync_service.background_status(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
    )

    notes_status = {item.domain: item for item in status.domains}["notes.note"]
    assert notes_status.replayable_failures == 1
    assert status.replayable_failure_count == 1


def test_push_lock_timeout_stays_pending_and_surfaces_in_background_health(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
        ),
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    @contextmanager
    def busy_guard(*args, **kwargs):
        raise SyncMaterializationBusyError()
        yield  # pragma: no cover - required for the contextmanager protocol.

    monkeypatch.setattr(sync_store, "materialization_guard", busy_guard)

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_m1_note_envelope()],
    )
    status = service.background_status(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
    )

    assert result.rejected == []
    assert result.conflicts == []
    assert result.accepted[0].apply_status == "pending"
    assert status.replayable_failure_count == 1


def test_background_sync_rejects_revoked_devices(sync_service: SyncV2Service) -> None:
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1")
    sync_service.revoke_device(user_id="user-1", device_id="device-1", reason="lost")

    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        sync_service.get_background_policy(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
        )
    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        sync_service.acquire_background_lease(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            ttl_seconds=120,
        )
    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        sync_service.background_status(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
        )


def test_dataset_enrollment_creates_personal_dataset_by_default(sync_service: SyncV2Service):
    enrolled = sync_service.enroll_dataset(user_id="user-1")

    assert enrolled.dataset.dataset_id == "dataset-generated"
    assert enrolled.dataset.owner_user_id == "user-1"
    assert enrolled.dataset.scope_type == "personal"
    assert enrolled.dataset.encryption_policy == "server_trusted_v1"
    assert enrolled.dataset.domains == [
        "notes.note",
        "chat.conversation",
        "chat.message",
        "attachment.ref",
    ]
    assert enrolled.key_setup_required is False


def test_dataset_enrollment_rejects_cross_user_dataset_takeover(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="shared-dataset")

    with pytest.raises(SyncStoreError):
        sync_service.enroll_dataset(user_id="user-2", dataset_id="shared-dataset")


def test_public_dataset_enrollment_rejects_notes_organization_and_reserved_metadata(
    sync_service: SyncV2Service,
) -> None:
    forged_metadata = {
        "default_personal": True,
        "client_family": "chatbook",
        "notes_organization_v1": {
            "state": "ready",
            "bootstrap_id": "client-forged",
            "captured_count": 1,
            "expected_count": 1,
        },
    }

    with pytest.raises(SyncStoreError, match="sync_reserved_dataset_enrollment"):
        sync_service.enroll_dataset(
            user_id="user-1",
            dataset_id="forged-org",
            domains=[*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS],
            metadata=forged_metadata,
        )
    with pytest.raises(SyncStoreError, match="sync_reserved_dataset_enrollment"):
        sync_service.enroll_dataset(
            user_id="user-1",
            dataset_id="forged-markers",
            domains=list(M1_SYNC_DOMAINS),
            metadata=forged_metadata,
        )

    assert sync_service.store.list_datasets_for_user("user-1") == []


@pytest.mark.parametrize(
    ("metadata_key", "value"),
    [
        (
            "notes_task_v1",
            {
                "state": "ready",
                "source_cursor": "00000000-0000-4000-8000-000000000001",
                "source_count": 1,
                "source_fingerprint": "a" * 64,
                "reason_code": None,
                "resume_phase": None,
            },
        ),
        (
            "notes_task_activity_v1",
            {
                "state": "ready",
                "source_cursor": (
                    "2026-08-13T00:00:00+00:00|"
                    "00000000-0000-4000-8000-000000000011"
                ),
                "source_count": 1,
                "source_fingerprint": "b" * 64,
                "reason_code": None,
                "resume_phase": None,
            },
        ),
        ("task_activity_capture_enabled", True),
    ],
)
def test_public_dataset_enrollment_rejects_forged_task_readiness_metadata(
    sync_service: SyncV2Service,
    metadata_key: str,
    value: object,
) -> None:
    with pytest.raises(SyncStoreError, match="sync_reserved_dataset_enrollment"):
        sync_service.enroll_dataset(
            user_id="user-1",
            dataset_id=f"forged-{metadata_key}",
            metadata={metadata_key: value},
        )

    assert sync_service.store.list_datasets_for_user("user-1") == []


def test_public_enrollment_and_manifest_redact_internal_task_readiness(
    sync_service: SyncV2Service,
) -> None:
    internal_metadata = {
        "notes_task_v1": {
            "state": "ready",
            "source_cursor": "00000000-0000-4000-8000-000000000001",
            "source_count": 1,
            "source_fingerprint": "a" * 64,
            "reason_code": None,
            "resume_phase": None,
        },
        "notes_task_activity_v1": {
            "state": "blocked",
            "source_cursor": (
                "2026-08-13T00:00:00+00:00|"
                "00000000-0000-4000-8000-000000000011"
            ),
            "source_count": 1,
            "source_fingerprint": "b" * 64,
            "reason_code": "notes_task_activity_source_invalid",
            "resume_phase": "bootstrapping",
        },
        "task_activity_capture_enabled": True,
    }
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-internal-task-readiness",
            owner_user_id="user-1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={"label": "before", **internal_metadata},
        )
    )

    enrollment = sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-internal-task-readiness",
        metadata={"label": "after"},
    )
    manifest = sync_service.restore_manifest(user_id="user-1")
    stored = sync_service.store.get_dataset(
        "dataset-internal-task-readiness",
        owner_user_id="user-1",
    )

    assert enrollment.dataset.metadata == {"label": "after"}
    assert manifest.datasets[0].metadata == {"label": "after"}
    assert stored is not None
    assert stored.metadata == {"label": "after", **internal_metadata}


def test_adapter_registry_accepts_known_domains_and_rejects_unknown_domains(
    registry: SyncAdapterRegistry,
):
    assert registry.get("notes.note").domain == "notes.note"

    with pytest.raises(KeyError):
        registry.get("media")

    with pytest.raises(ValueError):
        registry.register(
            StaticSyncAdapter(domain=cast(SyncDomain, "bogus"), supported_adapter_versions={1})
        )


def test_adapter_registry_accepts_workspace_metadata_domains():
    registry = _workspace_registry()

    assert registry.get(cast(SyncDomain, "workspaces.workspace")).domain == "workspaces.workspace"
    assert registry.get(cast(SyncDomain, "workspaces.source_ref")).domain == "workspaces.source_ref"


def test_adapter_registry_accepts_source_cache_entry_without_legacy_source_cache():
    registry = SyncAdapterRegistry([SourceCacheAdapter()])

    assert registry.get(cast(SyncDomain, "source_cache.entry")).domain == "source_cache.entry"
    with pytest.raises(KeyError):
        registry.get(cast(SyncDomain, "source_cache"))


def test_adapter_registry_accepts_media_metadata_without_legacy_media():
    registry = SyncAdapterRegistry(
        [
            MediaMetadataAdapter(domain=cast(SyncDomain, "media.item")),
            MediaMetadataAdapter(domain=cast(SyncDomain, "media.keyword")),
            MediaMetadataAdapter(domain=cast(SyncDomain, "media.keyword_link")),
        ]
    )

    for domain in ("media.item", "media.keyword", "media.keyword_link"):
        assert registry.get(cast(SyncDomain, domain)).domain == domain
    with pytest.raises(KeyError):
        registry.get(cast(SyncDomain, "media"))


def test_workspace_dataset_enrollment_requires_workspace_sync_permission(
    sync_store: SyncV2Store,
):
    no_checker_service = SyncV2Service(
        store=sync_store,
        adapters=_workspace_registry(),
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    with pytest.raises(SyncStoreError, match="workspace.*not found or is not accessible"):
        no_checker_service.enroll_dataset(
            user_id="user-1",
            dataset_id="workspace-dataset-no-checker",
            scope_type="workspace",
            workspace_id="workspace-1",
            domains=WORKSPACE_DOMAINS,
        )

    allowed = {("user-1", "workspace-1", "sync")}
    service = _workspace_service(sync_store, allowed)
    enrollment = service.enroll_dataset(
        user_id="user-1",
        dataset_id="workspace-dataset",
        scope_type="workspace",
        workspace_id="workspace-1",
        domains=WORKSPACE_DOMAINS,
    )

    assert enrollment.dataset.scope_type == "workspace"
    assert enrollment.dataset.workspace_id == "workspace-1"
    assert enrollment.dataset.domains == WORKSPACE_DOMAINS

    with pytest.raises(SyncStoreError, match="workspace.*not found or is not accessible"):
        service.enroll_dataset(
            user_id="user-2",
            dataset_id="workspace-dataset-denied",
            scope_type="workspace",
            workspace_id="workspace-1",
            domains=WORKSPACE_DOMAINS,
        )


def test_workspace_dataset_member_access_is_not_tied_to_dataset_owner(
    sync_store: SyncV2Store,
):
    allowed = {
        ("user-1", "workspace-1", "sync"),
        ("user-2", "workspace-1", "sync"),
    }
    service = _workspace_service(sync_store, allowed)
    _register_devices(service, "user-1", "owner-device")
    _register_devices(service, "user-2", "member-device")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="workspace-dataset",
        scope_type="workspace",
        workspace_id="workspace-1",
        domains=WORKSPACE_DOMAINS,
    )

    policy = service.update_background_policy(
        user_id="user-2",
        dataset_id="workspace-dataset",
        device_id="member-device",
        enabled=False,
        pending_local_changes=True,
    )
    push_result = service.push(
        user_id="user-2",
        dataset_id="workspace-dataset",
        device_id="member-device",
        envelopes=[
            _workspace_envelope(
                device_id="member-device",
                client_envelope_id="workspace-member-env",
            )
        ],
    )

    assert policy.enabled is False
    assert policy.pending_local_changes is True
    assert push_result.rejected == []
    assert push_result.accepted[0].client_envelope_id == "workspace-member-env"


def test_workspace_dataset_access_is_rechecked_for_dataset_scoped_operations(
    sync_store: SyncV2Store,
    tmp_path: Path,
):
    allowed = {("user-1", "workspace-1", "sync")}
    service = _workspace_service(
        sync_store,
        allowed,
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="workspace-dataset",
        scope_type="workspace",
        workspace_id="workspace-1",
        domains=WORKSPACE_DOMAINS,
    )
    authorization = service.create_device_authorization(
        user_id="user-1",
        dataset_id="workspace-dataset",
        device_id="device-1",
        authorization_method="existing_device",
    )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="workspace-conflict-1",
            dataset_id="workspace-dataset",
            domain=cast(SyncDomain, "workspaces.workspace"),
            object_id="workspace-1",
            conflict_type="rename_conflict",
            metadata={"reason": "workspace renamed on two devices"},
        )
    )

    allowed.clear()

    push_result = service.push(
        user_id="user-1",
        dataset_id="workspace-dataset",
        device_id="device-1",
        envelopes=[_workspace_envelope()],
    )
    assert push_result.accepted == []
    assert push_result.rejected[0].error_code == "dataset_not_found_or_forbidden"

    denied_calls = [
        lambda: service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
        ),
        lambda: service.get_background_policy(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
        ),
        lambda: service.acquire_background_lease(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
        ),
        lambda: service.background_status(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
        ),
        lambda: service.create_device_authorization(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
            authorization_method="existing_device",
        ),
        lambda: service.approve_device_authorization(
            authorization.authorization_id,
            user_id="user-1",
            dataset_id="workspace-dataset",
            approving_device_id="device-1",
        ),
        lambda: service.pull(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
        ),
        lambda: service.restore_manifest(
            user_id="user-1",
            device_id="device-1",
            dataset_ids=["workspace-dataset"],
        ),
        lambda: service.restore_preview(
            user_id="user-1",
            device_id="device-1",
            dataset_ids=["workspace-dataset"],
        ),
        lambda: service.repair(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
        ),
        lambda: service.list_conflicts(
            user_id="user-1",
            dataset_id="workspace-dataset",
        ),
        lambda: service.resolve_conflict(
            user_id="user-1",
            conflict_id=conflict.conflict_id,
            dataset_id="workspace-dataset",
            action="skip",
            resolved_by_device_id="device-1",
        ),
        lambda: service.store_key_recovery_bundle(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:opaque",
            kdf_metadata={"algorithm": "argon2id"},
        ),
        lambda: service.list_key_recovery_bundles(
            user_id="user-1",
            dataset_id="workspace-dataset",
        ),
        lambda: service.create_blob_upload_session(
            user_id="user-1",
            dataset_id="workspace-dataset",
            device_id="device-1",
            domain=cast(SyncDomain, "workspaces.source_ref"),
            entity_id="source-ref-1",
            attachment_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=16,
            payload_hash=_sha256(b"0123456789abcdef"),
            chunk_size=16,
            chunk_count=1,
        ),
    ]
    for call in denied_calls:
        with pytest.raises(SyncStoreError, match="not found or is not accessible"):
            call()


def test_source_cache_envelopes_materialize_and_repair_object_state(
    sync_store: SyncV2Store,
):
    from tldw_Server_API.app.core.Sync.v2.materializers.source_cache import SourceCacheMaterializer

    registry = SyncAdapterRegistry([SourceCacheAdapter()])
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"source_cache.entry": SourceCacheMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=[cast(SyncDomain, "source_cache.entry")],
    )

    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_source_cache_envelope()],
    )
    state = sync_store.get_object_state(
        "dataset-1",
        cast(SyncDomain, "source_cache.entry"),
        "source-1:sha256-source",
    )

    assert pushed.rejected == []
    assert pushed.conflicts == []
    assert pushed.accepted[0].apply_status == "applied"
    assert state is not None
    assert state.object_hash == "sha256:source-cache-entry"
    assert state.deleted is False

    sync_store.mark_envelope_apply_status(
        pushed.accepted[0].server_sequence,
        apply_status="failed",
        apply_error_code="projection_failed",
        apply_error_message="retry source cache projection",
    )
    repaired = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domains=[cast(SyncDomain, "source_cache.entry")],
        failed_only=True,
    )

    assert repaired.applied_count == 1
    assert repaired.failed_count == 0

    tombstoned = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _source_cache_envelope(
                client_envelope_id="source-cache-tombstone",
                operation="tombstone",
                client_sequence=2,
                object_revision=2,
                payload={
                    "entity_kind": "source_cache_entry",
                    "source_id": "source-1",
                    "content_hash": "sha256:source",
                    "provenance": {"kind": "url", "uri": "https://example.test/source"},
                    "tombstone": True,
                },
                payload_hash="sha256:source-cache-tombstone",
                base_server_cursor=pushed.accepted[0].server_sequence,
                base_object_revision=1,
                base_object_hash="sha256:source-cache-entry",
            )
        ],
    )
    deleted_state = sync_store.get_object_state(
        "dataset-1",
        cast(SyncDomain, "source_cache.entry"),
        "source-1:sha256-source",
    )

    assert tombstoned.rejected == []
    assert tombstoned.conflicts == []
    assert deleted_state is not None
    assert deleted_state.deleted is True
    assert deleted_state.object_revision == 2


def test_media_metadata_envelopes_materialize_and_repair_object_state(
    sync_store: SyncV2Store,
):
    from tldw_Server_API.app.core.Sync.v2.materializers.media_metadata import (
        MediaMetadataMaterializer,
    )

    media_domains = [
        cast(SyncDomain, "media.item"),
        cast(SyncDomain, "media.keyword"),
        cast(SyncDomain, "media.keyword_link"),
    ]
    registry = SyncAdapterRegistry(
        [MediaMetadataAdapter(domain=domain) for domain in media_domains]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={domain: MediaMetadataMaterializer(domain=domain) for domain in media_domains},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=media_domains,
    )

    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_media_envelope(domain) for domain in media_domains],
    )

    assert pushed.rejected == []
    assert pushed.conflicts == []
    assert [item.apply_status for item in pushed.accepted] == ["applied", "applied", "applied"]
    for domain in media_domains:
        state = sync_store.get_object_state(
            "dataset-1",
            domain,
            {
                "media.item": "media-1",
                "media.keyword": "keyword-1",
                "media.keyword_link": "media-1:keyword-1",
            }[domain],
        )
        assert state is not None
        assert state.object_hash == f"sha256:{domain}-v1"
        assert state.deleted is False

    sync_store.mark_envelope_apply_status(
        pushed.accepted[0].server_sequence,
        apply_status="failed",
        apply_error_code="projection_failed",
        apply_error_message="retry media metadata projection",
    )
    repaired = service.repair(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domains=[cast(SyncDomain, "media.item")],
        failed_only=True,
    )

    assert repaired.applied_count == 1
    assert repaired.failed_count == 0

    tombstoned = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _media_envelope(
                cast(SyncDomain, "media.item"),
                client_envelope_id="media-item-tombstone",
                operation="tombstone",
                client_sequence=4,
                object_revision=2,
                payload={"media_id": "media-1", "media_type": "video", "tombstone": True},
                payload_hash="sha256:media-item-tombstone",
                base_server_cursor=pushed.accepted[0].server_sequence,
                base_object_revision=1,
                base_object_hash="sha256:media.item-v1",
            )
        ],
    )
    deleted_state = sync_store.get_object_state(
        "dataset-1",
        cast(SyncDomain, "media.item"),
        "media-1",
    )

    assert tombstoned.rejected == []
    assert tombstoned.conflicts == []
    assert deleted_state is not None
    assert deleted_state.deleted is True
    assert deleted_state.object_revision == 2


def test_push_supports_legacy_adapter_without_context_keyword(sync_store: SyncV2Store):
    class LegacyNotesAdapter:
        domain: SyncDomain = "notes.note"
        supported_adapter_versions = {1}

        def __init__(self) -> None:
            self.dataset_id: str | None = None

        def evaluate_envelope(
            self,
            envelope: SyncEnvelopeCreate,
            *,
            dataset: SyncDataset,
        ) -> AdapterAccepted:
            self.dataset_id = dataset.dataset_id
            return AdapterAccepted(client_envelope_id=envelope.client_envelope_id)

    adapter = LegacyNotesAdapter()
    registry = SyncAdapterRegistry([adapter])
    service = SyncV2Service(store=sync_store, adapters=registry, clock=_clock)
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope()],
    )

    assert result.accepted[0].client_envelope_id == "env-1"
    assert adapter.dataset_id == "dataset-1"


def test_push_rejects_envelopes_for_datasets_user_cannot_access(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    _register_devices(sync_service, "user-2", "user-2-device")

    result = sync_service.push(
        user_id="user-2",
        dataset_id="dataset-1",
        device_id="user-2-device",
        envelopes=[_envelope()],
    )

    assert result.accepted == []
    assert result.conflicts == []
    assert result.rejected[0].client_envelope_id == "env-1"
    assert result.rejected[0].error_code == "dataset_not_found_or_forbidden"

    with pytest.raises(SyncStoreError, match="dataset was not found"):
        sync_service.pull(
            user_id="user-2",
            dataset_id="dataset-1",
            device_id="user-2-device",
        )


@pytest.mark.parametrize("marker_value", [True, False, "spoofed"])
def test_push_rejects_reserved_guard_routing_before_persistence(
    sync_service: SyncV2Service,
    marker_value: object,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                routing_metadata={
                    "entity_kind": "note",
                    GUARD_REQUIRED_ROUTING_KEY: marker_value,
                }
            )
        ],
    )

    assert result.accepted == []
    assert result.conflicts == []
    assert [item.error_code for item in result.rejected] == [
        "reserved_routing_metadata"
    ]
    assert sync_service.store.list_envelopes_after("dataset-1", 0) == []


def test_push_returns_per_envelope_accepted_rejected_and_conflict_outcomes(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}))
    registry.register(
        StaticSyncAdapter(
            domain="chat.conversation",
            supported_adapter_versions={1},
            outcomes={
                "env-rejected": AdapterRejected(
                    client_envelope_id="env-rejected",
                    error_code="domain_validation_failed",
                    message="invalid chat shape",
                ),
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="chat.conversation",
                    entity_id="conversation-1",
                    conflict_type="version_divergence",
                    message="chat conflict",
                ),
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "chat.conversation"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="env-accepted"),
            _envelope(
                client_envelope_id="env-rejected",
                domain="chat.conversation",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:chat-rejected",
            ),
            _envelope(
                client_envelope_id="env-conflict",
                domain="chat.conversation",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:chat-conflict",
            ),
        ],
    )

    assert result.accepted[0].client_envelope_id == "env-accepted"
    assert result.rejected[0].client_envelope_id == "env-rejected"
    assert result.rejected[0].error_code == "domain_validation_failed"
    assert result.conflicts[0].client_envelope_id == "env-conflict"
    assert result.conflicts[0].conflict_id == "conflict-generated"
    assert result.conflicts[0].server_sequence is not None
    assert result.next_cursor == str(result.conflicts[0].server_sequence)


def test_concurrent_client_pushes_from_same_head_create_one_reviewable_conflict(
    sync_service: SyncV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    seed = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                payload={"title": "Original", "content": "Body"},
                payload_hash="sha256:original",
                object_revision=1,
            )
        ],
    )
    seed_cursor = seed.accepted[0].server_sequence
    append_barrier = Barrier(2)
    original_insert = sync_service.store.insert_envelope

    def insert_after_both_preflights(envelope):
        if envelope.status == "accepted":
            append_barrier.wait()
        return original_insert(envelope)

    monkeypatch.setattr(sync_service.store, "insert_envelope", insert_after_both_preflights)

    def push(device_id: str, envelope_id: str, title: str):
        return sync_service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id=device_id,
            envelopes=[
                _m1_note_envelope(
                    client_envelope_id=envelope_id,
                    device_id=device_id,
                    client_sequence=2,
                    payload={"title": title, "content": "Body"},
                    payload_hash=f"sha256:{title.lower()}",
                    object_revision=2,
                    base_server_cursor=seed_cursor,
                    base_object_revision=1,
                    base_object_hash="sha256:original",
                )
            ],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda item: push(*item),
                [
                    ("device-1", "env-concurrent-1", "First"),
                    ("device-2", "env-concurrent-2", "Second"),
                ],
            )
        )

    assert sum(len(result.accepted) for result in results) == 1
    assert sum(len(result.conflicts) for result in results) == 1
    assert sum(len(result.rejected) for result in results) == 0


def test_m1_push_reports_apply_outcomes_and_failed_projection_is_replayable(
    sync_store: SyncV2Store,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
        ),
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-applied",
                object_id="note-applied",
                object_revision=1,
                payload_hash="sha256:applied",
            ),
            _m1_note_envelope(
                client_envelope_id="env-failed",
                object_id="note-fail",
                client_sequence=2,
                object_revision=1,
                payload_hash="sha256:failed",
            ),
        ],
    )

    assert [(item.client_envelope_id, item.server_sequence) for item in result.accepted] == [
        ("env-applied", 1),
        ("env-failed", 2),
    ]
    assert [
        (item.client_envelope_id, item.object_revision, item.apply_status)
        for item in result.accepted
    ] == [
        ("env-applied", 1, "applied"),
        ("env-failed", 1, "failed"),
    ]
    assert result.accepted[1].apply_error_code == "projection_failed"
    assert "replayable" in result.accepted[1].apply_error_message
    failed = sync_store.list_failed_applies("dataset-1")
    assert [item.client_envelope_id for item in failed] == ["env-failed"]
    assert failed[0].status == "accepted"
    assert failed[0].apply_status == "failed"


def test_m1_push_catches_materializer_exceptions_after_acceptance(
    sync_store: SyncV2Store,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
        ),
        materializers={"notes.note": _RaisingMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-raises",
                object_id="note-raises",
                payload_hash="sha256:raises",
            )
        ],
    )
    failed = sync_store.list_failed_applies("dataset-1")
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
        domains=["notes.note"],
    )

    assert result.rejected == []
    assert result.conflicts == []
    assert len(result.accepted) == 1
    assert result.accepted[0].client_envelope_id == "env-raises"
    assert result.accepted[0].server_sequence == 1
    assert result.accepted[0].apply_status == "failed"
    assert result.accepted[0].apply_error_code == "sync_projection_failed"
    assert result.accepted[0].apply_error_message == "Projection failed: RuntimeError"
    assert "/private" not in result.accepted[0].apply_error_message
    assert [(item.client_envelope_id, item.status, item.apply_status) for item in failed] == [
        ("env-raises", "accepted", "failed")
    ]
    assert pulled.envelopes[0].client_envelope_id == "env-raises"
    assert pulled.envelopes[0].apply_status == "failed"
    assert pulled.envelopes[0].apply_error_code == "sync_projection_failed"
    assert pulled.envelopes[0].apply_error_message == "Projection failed: RuntimeError"


def test_push_rejects_unsupported_adapter_versions_per_envelope(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(adapter_version=2)],
    )

    assert result.accepted == []
    assert result.rejected[0].client_envelope_id == "env-1"
    assert result.rejected[0].error_code == "unsupported_adapter_version"


def test_push_treats_omitted_device_adapter_map_as_v1_only(
    sync_service: SyncV2Service,
) -> None:
    sync_service.adapters.register(
        StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(adapter_version=2)],
    )

    assert result.accepted == []
    assert result.rejected[0].error_code == "device_adapter_version_not_advertised"
    assert sync_service.store.list_envelopes_after("dataset-1", 0) == []


def test_push_accepts_adapter_version_advertised_by_registered_device(
    sync_service: SyncV2Service,
) -> None:
    sync_service.adapters.register(
        StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})
    )
    sync_service.register_device(
        user_id="user-1",
        display_name="device-1",
        client_type="chatbook",
        device_id="device-1",
        capabilities={"supported_adapter_versions": {"notes.note": [1, 2]}},
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(adapter_version=2)],
    )

    assert len(result.accepted) == 1
    assert result.rejected == []


def test_push_rejects_version_not_requested_for_envelope_domain(
    sync_service: SyncV2Service,
) -> None:
    sync_service.adapters.register(
        StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})
    )
    sync_service.register_device(
        user_id="user-1",
        display_name="device-new",
        client_type="chatbook",
        device_id="device-new",
        capabilities={"supported_adapter_versions": {"attachment.ref": [2]}},
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-new",
        envelopes=[_envelope(adapter_version=2, device_id="device-new")],
    )

    assert result.accepted == []
    assert result.rejected[0].error_code == "device_adapter_version_not_advertised"
    assert sync_service.store.list_envelopes_after("dataset-1", 0) == []


def test_push_rejects_mismatched_device_id_and_fills_missing_device_id(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    spoofed = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(device_id="device-2")],
    )
    accepted = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="env-no-device",
                device_id=None,
                entity_id="note-no-device",
                stable_key="note:no-device",
                payload_hash="sha256:no-device",
            )
        ],
    )
    same_device_pull = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor="0",
        domains=["notes.note"],
        include_own_changes=False,
    )

    assert spoofed.accepted == []
    assert spoofed.rejected[0].error_code == "device_mismatch"
    assert [item.client_envelope_id for item in accepted.accepted] == ["env-no-device"]
    assert same_device_pull.envelopes == []


def test_push_rejects_envelope_dataset_mismatch_before_persistence(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    _register_devices(sync_service, "user-2", "user-2-device")
    sync_service.enroll_dataset(user_id="user-2", dataset_id="dataset-2", domains=["notes.note"])

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="cross-dataset",
                dataset_id="dataset-2",
                payload_hash="sha256:cross-dataset",
            )
        ],
    )
    leaked = sync_service.pull(
        user_id="user-2",
        dataset_id="dataset-2",
        device_id="user-2-device",
        cursor="0",
        domains=["notes.note"],
        include_own_changes=True,
    )

    assert result.accepted == []
    assert result.conflicts == []
    assert result.rejected[0].client_envelope_id == "cross-dataset"
    assert result.rejected[0].error_code == "dataset_mismatch"
    assert leaked.envelopes == []


def test_push_reports_dataset_mismatch_per_envelope_in_mixed_batch(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="env-valid"),
            _envelope(
                client_envelope_id="env-wrong-dataset",
                dataset_id="dataset-other",
                entity_id="note-other",
                stable_key="note:other",
                payload_hash="sha256:wrong-dataset",
            ),
        ],
    )
    stored = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
        domains=["notes.note"],
    )

    assert [item.client_envelope_id for item in result.accepted] == ["env-valid"]
    assert result.rejected[0].client_envelope_id == "env-wrong-dataset"
    assert result.rejected[0].error_code == "dataset_mismatch"
    assert [item.client_envelope_id for item in stored.envelopes] == ["env-valid"]


def test_push_rejects_envelopes_beyond_batch_limit(sync_store: SyncV2Store, registry: SyncAdapterRegistry):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(max_batch_size=2),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="env-1", payload_hash="sha256:1"),
            _envelope(
                client_envelope_id="env-2",
                entity_id="note-2",
                stable_key="note:2",
                payload_hash="sha256:2",
            ),
            _envelope(
                client_envelope_id="env-3",
                entity_id="note-3",
                stable_key="note:3",
                payload_hash="sha256:3",
            ),
        ],
    )

    assert [item.client_envelope_id for item in result.accepted] == ["env-1", "env-2"]
    assert [item.client_envelope_id for item in result.rejected] == ["env-3"]
    assert result.rejected[0].error_code == "batch_limit_exceeded"
    assert result.rejected[0].retryable is False


def test_push_rejects_unenrolled_domain_before_adapter_conflict_path(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}))
    registry.register(
        StaticSyncAdapter(
            domain="chat.conversation",
            supported_adapter_versions={1},
            outcomes={
                "unenrolled-conflict": AdapterConflict(
                    client_envelope_id="unenrolled-conflict",
                    domain="chat.conversation",
                    entity_id="conversation-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(store=sync_store, adapters=registry, clock=_clock)
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="unenrolled-conflict",
                domain="chat.conversation",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:unenrolled-conflict",
            )
        ],
    )

    assert result.accepted == []
    assert result.conflicts == []
    assert result.rejected[0].error_code == "domain_not_enrolled"


def test_push_rejects_payloads_over_advertised_size_limit(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(max_envelope_payload_bytes=10),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="declared-too-large", payload_size_bytes=11),
            _envelope(
                client_envelope_id="ciphertext-too-large",
                entity_id="note-2",
                stable_key="note:2",
                payload_hash="sha256:large-ciphertext",
                payload_size_bytes=1,
                payload_ciphertext="x" * 11,
            ),
        ],
    )

    assert result.accepted == []
    assert [item.error_code for item in result.rejected] == [
        "payload_too_large",
        "payload_too_large",
    ]


def test_push_rejects_clear_payloads_over_actual_serialized_size_limit(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(max_envelope_payload_bytes=40),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="clear-too-large-missing-size",
                payload_ciphertext=None,
                payload_clear={"body": "x" * 80},
                payload_size_bytes=None,
            ),
            _envelope(
                client_envelope_id="clear-too-large-underreported",
                entity_id="note-2",
                stable_key="note:2",
                payload_hash="sha256:clear-underreported",
                payload_ciphertext=None,
                payload_clear={"status": "active"},
                routing_metadata={"entity_kind": "note", "summary": "y" * 80},
                dependencies=[{"entity_id": "source-1", "label": "z" * 80}],
                payload_size_bytes=1,
            ),
        ],
    )

    assert result.accepted == []
    assert [item.error_code for item in result.rejected] == [
        "payload_too_large",
        "payload_too_large",
    ]


def test_push_rejects_divergent_legacy_payload_over_actual_serialized_size_limit(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
) -> None:
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(max_envelope_payload_bytes=40),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    envelope = _envelope(
        client_envelope_id="legacy-payload-too-large",
        payload_ciphertext=None,
        payload_clear={},
        routing_metadata={},
        payload_size_bytes=1,
    )
    object.__setattr__(envelope, "payload", {"body": "x" * 80})

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[envelope],
    )

    assert result.accepted == []
    assert [item.error_code for item in result.rejected] == ["payload_too_large"]


def test_conflict_push_retry_reuses_existing_unresolved_conflict(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    conflict_ids = iter(["conflict-first", "conflict-second"])
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: next(conflict_ids) if prefix == "conflict" else f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    first = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    retried = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    manifest = service.restore_manifest(user_id="user-1")

    assert first.conflicts[0].conflict_id == "conflict-first"
    assert first.next_cursor == str(first.conflicts[0].server_sequence)
    assert retried.conflicts[0].conflict_id == first.conflicts[0].conflict_id
    assert retried.conflicts[0].server_sequence == first.conflicts[0].server_sequence
    assert retried.next_cursor == str(retried.conflicts[0].server_sequence)
    assert len(sync_store.list_conflicts("dataset-1", status="unresolved")) == 1
    assert manifest.datasets[0].unresolved_conflicts == 1


def test_conflict_push_rejects_idempotency_drift_without_aborting_batch(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    first = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    drift = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="env-conflict",
                payload_hash="sha256:drifted-conflict",
                payload_ciphertext="ciphertext:drifted-conflict",
            ),
            _envelope(
                client_envelope_id="env-after-drift",
                entity_id="note-after-drift",
                stable_key="note:after-drift",
                payload_hash="sha256:after-drift",
            ),
        ],
    )

    assert [conflict.client_envelope_id for conflict in first.conflicts] == ["env-conflict"]
    assert drift.conflicts == []
    assert [item.client_envelope_id for item in drift.rejected] == ["env-conflict"]
    assert drift.rejected[0].error_code == "idempotency_conflict"
    assert [item.client_envelope_id for item in drift.accepted] == ["env-after-drift"]
    assert len(sync_store.list_conflicts("dataset-1", status="unresolved")) == 1


def test_resolve_conflict_stores_resolution_envelope(sync_store: SyncV2Store):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    original_conflict_cursor = pushed.conflicts[0].server_sequence

    resolved = service.resolve_conflict(
        user_id="user-1",
        conflict_id=conflict_id,
        action="overwrite",
        resolved_by_device_id="device-1",
        resolution_envelope=_envelope(
            client_envelope_id="env-resolution",
            operation="upsert",
            payload_hash="sha256:resolution",
        ),
    )
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
    )

    assert resolved.status == "resolved"
    assert resolved.resolved_by_envelope_id == pulled.envelopes[0].envelope_id
    assert resolved.server_cursor == original_conflict_cursor
    assert resolved.server_cursor != pulled.envelopes[0].server_cursor
    assert resolved.resolution_action == "overwrite"
    assert [envelope.client_envelope_id for envelope in pulled.envelopes] == ["env-resolution"]
    assert pulled.envelopes[0].status == "accepted"


def test_resolve_conflict_rejects_reserved_guard_routing_before_claim(
    sync_store: SyncV2Store,
) -> None:
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="reserved routing metadata"):
        service.resolve_conflict(
            user_id="user-1",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
            resolution_envelope=_envelope(
                client_envelope_id="env-resolution-reserved-routing",
                payload_hash="sha256:reserved-routing",
                routing_metadata={
                    "entity_kind": "note",
                    GUARD_REQUIRED_ROUTING_KEY: True,
                },
            ),
        )

    conflict = sync_store.get_conflict(conflict_id)
    assert conflict is not None
    assert conflict.status == "unresolved"
    assert all(
        envelope.client_envelope_id != "env-resolution-reserved-routing"
        for envelope in sync_store.list_envelopes_after("dataset-1", 0)
    )


def test_resolve_conflict_duplicate_rename_accepts_distinct_object_id(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-original",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-original",
                payload_hash="sha256:original",
            )
        ],
    )
    original_conflict_cursor = pushed.conflicts[0].server_sequence

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=pushed.conflicts[0].conflict_id,
        action="duplicate_rename",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-resolution-copy",
            object_id="note-copy",
            client_sequence=2,
            payload={"title": "Research note copy"},
            payload_hash="sha256:copy",
        ),
    )
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
    )

    assert resolved.status == "resolved"
    assert resolved.resolution_action == "duplicate_rename"
    assert [envelope.object_id for envelope in pulled.envelopes] == ["note-copy"]
    assert pulled.envelopes[0].client_envelope_id == "env-resolution-copy"
    assert resolved.resolved_by_envelope_id == pulled.envelopes[0].envelope_id
    assert resolved.resolved_by_envelope_id.startswith("srv_env_")
    assert resolved.resolved_by_envelope_id != "env-resolution-copy"
    assert resolved.server_cursor == original_conflict_cursor
    assert resolved.server_cursor != pulled.envelopes[0].server_cursor


def test_resolve_conflict_overwrite_materializes_resolution_envelope(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=pushed.conflicts[0].conflict_id,
        action="overwrite",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-overwrite",
            object_id="note-1",
            client_sequence=2,
            object_revision=1,
            payload_hash="sha256:overwrite",
        ),
    )
    state = sync_store.get_object_state("dataset-1", "notes.note", "note-1")
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
        include_own_changes=True,
    )

    assert resolved.status == "resolved"
    assert resolved.resolution_action == "overwrite"
    assert state is not None
    assert state.object_revision == 1
    assert state.object_hash == "sha256:overwrite"
    assert [(item.client_envelope_id, item.apply_status) for item in pulled.envelopes] == [
        ("env-overwrite", "applied")
    ]


def test_accepted_materialization_conflict_duplicate_resolution_terminalizes_predecessor(
    sync_store: SyncV2Store,
) -> None:
    service, materializer = _accepted_materialization_conflict_service(sync_store)
    source_create = _m1_note_envelope(
        client_envelope_id="env-materialization-conflict",
        object_id="note-original",
        object_revision=1,
        payload_hash="sha256:original",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[source_create],
    )
    conflict = pushed.conflicts[0]

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action="duplicate_rename",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-resolution-copy",
            object_id="note-copy",
            client_sequence=2,
            object_revision=1,
            payload_hash="sha256:copy",
        ),
    )
    envelopes = sync_store.list_envelopes_after("dataset-1", 0, status=None)
    source = next(
        item
        for item in envelopes
        if item.client_envelope_id == "env-materialization-conflict"
    )
    resolution = next(
        item for item in envelopes if item.client_envelope_id == "env-resolution-copy"
    )
    replayed_source = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[source_create],
    )

    assert resolved.status == "resolved"
    assert source.apply_status == "superseded"
    assert source.apply_error_code == "sync_conflict_superseded"
    assert resolution.apply_status == "applied"
    assert replayed_source.accepted[0].apply_status == "superseded"
    assert materializer.calls.count("env-materialization-conflict") == 1


def test_accepted_conflict_overwrite_materializes_from_last_applied_predecessor(
    sync_store: SyncV2Store,
    tmp_path: Path,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / "conflict-overwrite-notes.db"),
        client_id="user-1",
    )
    service, baseline, source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-overwrite",
            object_id="note-original",
            client_sequence=3,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            payload={"title": "Resolved", "content": "Projected replacement"},
            payload_hash="sha256:resolved",
        ),
    )

    head = sync_store.get_current_head("dataset-1", "notes.note", "note-original")
    state = sync_store.get_object_state("dataset-1", "notes.note", "note-original")
    note = note_db.get_note_by_id("note-original")
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=0,
        domains=["notes.note"],
    )
    client_store = SyncV2Store(
        SyncDatabase(sqlite_path=tmp_path / "fresh-client-sync.db")
    )
    client_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
        )
    )
    client_db = CharactersRAGDB(
        db_path=str(tmp_path / "fresh-client-notes.db"),
        client_id="user-1",
    )
    client_materializer = NotesMaterializer(client_db)
    client_results = []
    for envelope in pulled.envelopes:
        stored = client_store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id=envelope.dataset_id,
                client_envelope_id=envelope.client_envelope_id,
                domain=envelope.domain,
                operation=envelope.operation,
                object_id=envelope.object_id,
                device_id=envelope.device_id,
                client_sequence=envelope.client_sequence,
                base_server_cursor=envelope.base_server_cursor,
                base_object_revision=envelope.base_object_revision,
                base_object_hash=envelope.base_object_hash,
                object_revision=envelope.object_revision,
                payload=envelope.payload,
                payload_hash=envelope.payload_hash,
                created_at_client=envelope.created_at_client,
                deleted=envelope.deleted,
                encryption_metadata=envelope.encryption_metadata,
                status="accepted",
                apply_status="pending",
            )
        )
        client_results.append(client_materializer.apply(stored, store=client_store))
    client_note = client_db.get_note_by_id("note-original")
    assert resolved.status == "resolved"
    assert head is not None and head.client_envelope_id == "env-overwrite"
    assert state is not None and state.latest_server_cursor == head.server_cursor
    assert state.object_revision == 2
    assert baseline.server_cursor < source.server_cursor < head.server_cursor
    assert note is not None and note["title"] == "Resolved"
    assert [item.client_envelope_id for item in pulled.envelopes] == [
        "env-applied-predecessor",
        "env-overwrite",
    ]
    assert pulled.envelopes[1].base_server_cursor == baseline.server_cursor
    assert pulled.envelopes[1].base_object_revision == baseline.object_revision
    assert pulled.envelopes[1].base_object_hash == baseline.payload_hash
    assert [result.status for result in client_results] == ["applied", "applied"]
    assert client_note is not None and client_note["title"] == "Resolved"
    assert "Never projected" not in str(pulled)


def test_accepted_keyword_conflict_overwrite_evaluates_against_projected_predecessor(
    tmp_path: Path,
) -> None:
    (
        _note_db,
        sync_store,
        service,
        dataset_id,
        baseline,
        source,
        conflict,
    ) = _accepted_keyword_conflict_after_applied_predecessor(tmp_path)
    with sync_store.db.backend.transaction() as connection:
        later_duplicate = sync_store.db._insert_envelope_in_transaction(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-later-duplicate-keyword",
                domain="notes.keyword",
                operation="upsert",
                object_id="99999999-9999-4999-8999-999999999999",
                device_id="frontend-device",
                client_sequence=3,
                object_revision=1,
                schema_version=1,
                payload={"keyword": "Resolved"},
                payload_hash="sha256:later-duplicate-keyword",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
                status="accepted",
            ),
            connection=connection,
        )

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-keyword-resolution",
            domain="notes.keyword",
            operation="upsert",
            object_id=source.object_id,
            device_id="frontend-device",
            client_sequence=4,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            schema_version=1,
            payload={"keyword": "Resolved"},
            payload_hash="sha256:keyword-resolution",
            encryption_metadata={"policy": "server_trusted_v1"},
            adapter_version=1,
        ),
    )

    head = sync_store.get_current_head(dataset_id, "notes.keyword", source.object_id)
    later = sync_store.get_envelope_by_server_cursor(later_duplicate.server_cursor)
    assert resolved.status == "resolved"
    assert head is not None and head.client_envelope_id == "env-keyword-resolution"
    assert head.apply_status == "applied"
    assert later is not None and later.apply_error_code == (
        "sync_rebase_required_after_conflict_resolution"
    )


def test_original_conflict_context_is_one_complete_source_cursor_snapshot(
    sync_store: SyncV2Store,
) -> None:
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
        ),
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
        )
    )
    source_baseline = sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-snapshot-source-baseline",
            object_id="note-source",
            client_sequence=1,
            object_revision=1,
            payload_hash="sha256:source-baseline",
        )
    )
    source_baseline = sync_store.mark_envelope_apply_status(
        source_baseline.server_cursor,
        apply_status="applied",
    )
    dependency_baseline = sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-snapshot-dependency-baseline",
            object_id="note-dependency",
            client_sequence=2,
            object_revision=1,
            payload_hash="sha256:dependency-baseline",
        )
    )
    dependency_baseline = sync_store.mark_envelope_apply_status(
        dependency_baseline.server_cursor,
        apply_status="applied",
    )
    source = sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-snapshot-source-conflict",
            object_id="note-source",
            client_sequence=3,
            base_server_cursor=source_baseline.server_cursor,
            base_object_revision=source_baseline.object_revision,
            base_object_hash=source_baseline.payload_hash,
            object_revision=2,
            payload_hash="sha256:source-conflict",
            status="accepted",
        )
    )
    source = sync_store.mark_envelope_apply_status(
        source.server_cursor,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-complete-snapshot",
            dataset_id="dataset-1",
            domain=source.domain,
            entity_id=source.object_id,
            conflict_type="projection_conflict",
            local_envelope_id=source.client_envelope_id,
            server_sequence=source.server_cursor,
        )
    )
    with sync_store.db.backend.transaction() as connection:
        later_dependency = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                client_envelope_id="env-snapshot-later-dependency",
                object_id="note-dependency",
                client_sequence=4,
                base_server_cursor=dependency_baseline.server_cursor,
                base_object_revision=dependency_baseline.object_revision,
                base_object_hash=dependency_baseline.payload_hash,
                object_revision=2,
                payload_hash="sha256:later-dependency",
                status="accepted",
            ),
            connection=connection,
        )
    dataset = sync_store.get_dataset("dataset-1")
    assert dataset is not None

    with sync_store.materialization_guard(
        [source], require_predecessors=False
    ) as guarded_store:
        context = service._conflict_resolution_adapter_context(
            dataset,
            conflict=conflict,
            source=source,
            resolution_envelope=_m1_note_envelope(
                client_envelope_id="env-snapshot-resolution",
                object_id=source.object_id,
                client_sequence=5,
                base_server_cursor=source_baseline.server_cursor,
                base_object_revision=source_baseline.object_revision,
                base_object_hash=source_baseline.payload_hash,
                object_revision=2,
            ),
            action="overwrite",
            store=guarded_store,
        )
        dependency = context.get_head("notes.note", "note-dependency")
        listed = {
            item.object_id: item for item in context.list_heads("notes.note")
        }

    assert later_dependency.server_cursor > source.server_cursor
    assert dependency is not None
    assert dependency.server_cursor == dependency_baseline.server_cursor
    assert listed["note-dependency"].server_cursor == (
        dependency_baseline.server_cursor
    )
    assert listed["note-source"].server_cursor == source_baseline.server_cursor


def test_rebase_conflict_append_uses_current_applied_head_not_physical_predecessor(
    sync_store: SyncV2Store,
) -> None:
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
        )
    )
    baseline = sync_store.insert_envelope(
        _m1_note_envelope(
            client_envelope_id="env-rebase-c1",
            object_id="note-rebase",
            client_sequence=1,
            object_revision=1,
            payload_hash="sha256:rebase-c1",
        )
    )
    baseline = sync_store.mark_envelope_apply_status(
        baseline.server_cursor,
        apply_status="applied",
    )
    with sync_store.db.backend.transaction() as connection:
        queued = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                client_envelope_id="env-rebase-c3",
                object_id="note-rebase",
                client_sequence=3,
                base_server_cursor=baseline.server_cursor,
                base_object_revision=baseline.object_revision,
                base_object_hash=baseline.payload_hash,
                object_revision=2,
                payload_hash="sha256:rebase-c3",
                status="accepted",
                apply_status="conflict",
                apply_error_code="sync_rebase_required_after_conflict_resolution",
            ),
            connection=connection,
        )
        replacement = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                client_envelope_id="env-rebase-c4",
                object_id="note-rebase",
                client_sequence=4,
                base_server_cursor=baseline.server_cursor,
                base_object_revision=baseline.object_revision,
                base_object_hash=baseline.payload_hash,
                object_revision=2,
                payload_hash="sha256:rebase-c4",
                status="accepted",
                apply_status="applied",
            ),
            connection=connection,
        )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-rebase-current-head",
            dataset_id="dataset-1",
            domain="notes.note",
            entity_id=queued.object_id,
            conflict_type="sync_rebase_required_after_conflict_resolution",
            local_envelope_id=queued.client_envelope_id,
            server_sequence=queued.server_cursor,
        )
    )
    resolution = _m1_note_envelope(
        client_envelope_id="env-rebase-c5",
        object_id="note-rebase",
        client_sequence=5,
        base_server_cursor=replacement.server_cursor,
        base_object_revision=replacement.object_revision,
        base_object_hash=replacement.payload_hash,
        object_revision=3,
        payload_hash="sha256:rebase-c5",
        status="accepted",
    )

    with sync_store.materialization_guard(
        [queued], require_predecessors=False
    ) as guarded_store:
        guarded_store.claim_conflict_resolution(
            conflict.conflict_id,
            dataset_id="dataset-1",
            resolved_by_device_id="device-1",
            resolution_action="overwrite",
            resolution_notes=None,
        )
        inserted = guarded_store.insert_claimed_conflict_resolution_envelope(
            resolution,
            conflict_id=conflict.conflict_id,
            dataset_id="dataset-1",
            resolved_by_device_id="device-1",
            resolution_action="overwrite",
            resolution_notes=None,
        )

    assert inserted.base_server_cursor == replacement.server_cursor
    assert inserted.server_cursor > replacement.server_cursor


def test_rebase_conflict_overwrite_uses_current_note_replacement_chain_on_fresh_pull(
    sync_store: SyncV2Store,
    tmp_path: Path,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / "rebase-current-notes.db"),
        client_id="user-1",
    )
    service, baseline, source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )
    with sync_store.db.backend.transaction() as connection:
        queued = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                client_envelope_id="env-note-c3-queued",
                object_id=source.object_id,
                client_sequence=3,
                base_server_cursor=baseline.server_cursor,
                base_object_revision=baseline.object_revision,
                base_object_hash=baseline.payload_hash,
                object_revision=2,
                payload={"title": "Queued", "content": "Must rebase"},
                payload_hash="sha256:note-c3",
                status="accepted",
            ),
            connection=connection,
        )
    service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-note-c4-replacement",
            object_id=source.object_id,
            client_sequence=4,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            payload={"title": "Replacement", "content": "Applied c4"},
            payload_hash="sha256:note-c4",
        ),
    )
    replacement = sync_store.get_current_head(
        "dataset-1", "notes.note", source.object_id
    )
    rebase_conflict = sync_store.get_unresolved_conflict_for_envelope(
        "dataset-1",
        local_envelope_id=queued.client_envelope_id,
        server_sequence=queued.server_cursor,
    )
    assert replacement is not None
    assert rebase_conflict is not None

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=rebase_conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-note-c5-rebased",
            object_id=source.object_id,
            client_sequence=5,
            base_server_cursor=replacement.server_cursor,
            base_object_revision=replacement.object_revision,
            base_object_hash=replacement.payload_hash,
            object_revision=3,
            payload={"title": "Rebased", "content": "Applied c5"},
            payload_hash="sha256:note-c5",
        ),
    )

    head = sync_store.get_current_head("dataset-1", "notes.note", source.object_id)
    state = sync_store.get_object_state("dataset-1", "notes.note", source.object_id)
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=0,
        domains=["notes.note"],
    )
    assert resolved.status == "resolved"
    assert [item.client_envelope_id for item in pulled.envelopes] == [
        "env-applied-predecessor",
        "env-note-c4-replacement",
        "env-note-c5-rebased",
    ]
    assert pulled.envelopes[1].base_server_cursor == baseline.server_cursor
    assert pulled.envelopes[2].base_server_cursor == replacement.server_cursor
    assert head is not None and head.client_envelope_id == "env-note-c5-rebased"
    assert state is not None and state.latest_server_cursor == head.server_cursor
    assert state.object_revision == 3

    client_store = SyncV2Store(
        SyncDatabase(sqlite_path=tmp_path / "rebase-fresh-client-sync.db")
    )
    client_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
        )
    )
    client_db = CharactersRAGDB(
        db_path=str(tmp_path / "rebase-fresh-client-notes.db"),
        client_id="user-1",
    )
    client_materializer = NotesMaterializer(client_db)
    results = [
        client_materializer.apply(
            _store_pulled_envelope_at_canonical_cursor(client_store, envelope),
            store=client_store,
        )
        for envelope in pulled.envelopes
    ]
    client_note = client_db.get_note_by_id(source.object_id)
    assert [item.status for item in results] == ["applied", "applied", "applied"]
    assert client_note is not None and client_note["title"] == "Rebased"


def test_rebase_conflict_overwrite_uses_current_keyword_replacement(
    tmp_path: Path,
) -> None:
    (
        _note_db,
        sync_store,
        service,
        dataset_id,
        baseline,
        source,
        conflict,
    ) = _accepted_keyword_conflict_after_applied_predecessor(tmp_path)
    with sync_store.db.backend.transaction() as connection:
        queued = sync_store.db._insert_envelope_in_transaction(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-keyword-c3-queued",
                domain="notes.keyword",
                operation="upsert",
                object_id=source.object_id,
                device_id="frontend-device",
                client_sequence=3,
                base_server_cursor=baseline.server_cursor,
                base_object_revision=baseline.object_revision,
                base_object_hash=baseline.payload_hash,
                object_revision=2,
                schema_version=1,
                payload={"keyword": "Queued"},
                payload_hash="sha256:keyword-c3",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
                status="accepted",
            ),
            connection=connection,
        )
    service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-keyword-c4-replacement",
            domain="notes.keyword",
            operation="upsert",
            object_id=source.object_id,
            device_id="frontend-device",
            client_sequence=4,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            schema_version=1,
            payload={"keyword": "Replacement"},
            payload_hash="sha256:keyword-c4",
            encryption_metadata={"policy": "server_trusted_v1"},
            adapter_version=1,
        ),
    )
    replacement = sync_store.get_current_head(
        dataset_id, "notes.keyword", source.object_id
    )
    rebase_conflict = sync_store.get_unresolved_conflict_for_envelope(
        dataset_id,
        local_envelope_id=queued.client_envelope_id,
        server_sequence=queued.server_cursor,
    )
    assert replacement is not None
    assert rebase_conflict is not None

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=rebase_conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-keyword-c5-rebased",
            domain="notes.keyword",
            operation="upsert",
            object_id=source.object_id,
            device_id="frontend-device",
            client_sequence=5,
            base_server_cursor=replacement.server_cursor,
            base_object_revision=replacement.object_revision,
            base_object_hash=replacement.payload_hash,
            object_revision=3,
            schema_version=1,
            payload={"keyword": "Rebased"},
            payload_hash="sha256:keyword-c5",
            encryption_metadata={"policy": "server_trusted_v1"},
            adapter_version=1,
        ),
    )
    head = sync_store.get_current_head(dataset_id, "notes.keyword", source.object_id)
    state = sync_store.get_object_state(dataset_id, "notes.keyword", source.object_id)
    pulled = service.pull(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="device-2",
        cursor=0,
        domains=["notes.keyword"],
    )
    assert resolved.status == "resolved"
    assert head is not None and head.client_envelope_id == "env-keyword-c5-rebased"
    assert state is not None and state.latest_server_cursor == head.server_cursor
    assert state.object_revision == 3
    assert [item.client_envelope_id for item in pulled.envelopes] == [
        "env-keyword-baseline",
        "env-keyword-c4-replacement",
        "env-keyword-c5-rebased",
    ]
    assert pulled.envelopes[2].base_server_cursor == replacement.server_cursor

    client_store = SyncV2Store(
        SyncDatabase(sqlite_path=tmp_path / "rebase-fresh-keyword-sync.db")
    )
    client_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset_id,
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=list(NOTES_ORGANIZATION_DOMAINS),
            metadata={"notes_organization_v1": {"state": "ready"}},
        )
    )
    client_db = CharactersRAGDB(
        db_path=str(tmp_path / "rebase-fresh-keyword-notes.db"),
        client_id="user-1",
    )
    client_materializer = NotesOrganizationMaterializer(client_db, "notes.keyword")
    results = [
        client_materializer.apply(
            _store_pulled_envelope_at_canonical_cursor(client_store, envelope),
            store=client_store,
        )
        for envelope in pulled.envelopes
    ]
    client_state = client_store.get_object_state(
        dataset_id, "notes.keyword", source.object_id
    )
    assert [item.status for item in results] == ["applied", "applied", "applied"]
    assert client_state is not None
    assert client_state.latest_server_cursor == pulled.envelopes[-1].server_cursor
    assert client_state.object_revision == 3


def test_original_conflict_snapshot_excludes_later_queued_dependency(
    tmp_path: Path,
) -> None:
    _note_db, sync_store, service = build_ready_notes_sync_stack(tmp_path)
    dataset_id = service.profile(
        user_id="user-1", device_id="frontend-device"
    ).active_dataset_id
    assert dataset_id is not None
    note_id = "55555555-5555-4555-8555-555555555555"
    keyword_id = "66666666-6666-4666-8666-666666666666"
    note_result = service.push(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="frontend-device",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-snapshot-note",
                domain="notes.note",
                operation="upsert",
                object_id=note_id,
                device_id="frontend-device",
                client_sequence=1,
                object_revision=1,
                schema_version=1,
                payload={"title": "Snapshot", "content": "Applied note"},
                payload_hash="sha256:snapshot-note",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
            )
        ],
    )
    assert note_result.accepted
    link_id = organization_link_id(
        "notes.keyword_link", ["note", note_id, keyword_id]
    )
    source_create = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id="env-snapshot-link-conflict",
        domain="notes.keyword_link",
        operation="upsert",
        object_id=link_id,
        device_id="frontend-device",
        client_sequence=2,
        object_revision=1,
        schema_version=1,
        payload={
            "subject_type": "note",
            "subject_id": note_id,
            "keyword_sync_id": keyword_id,
        },
        payload_hash="sha256:snapshot-link-conflict",
        encryption_metadata={"policy": "server_trusted_v1"},
        adapter_version=1,
        status="accepted",
    )
    with sync_store.db.backend.transaction() as connection:
        source = sync_store.db._insert_envelope_in_transaction(
            source_create,
            connection=connection,
        )
    source = sync_store.mark_envelope_apply_status(
        source.server_cursor,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-snapshot-link",
            dataset_id=dataset_id,
            domain=source.domain,
            entity_id=source.object_id,
            conflict_type="projection_conflict",
            local_envelope_id=source.client_envelope_id,
            server_sequence=source.server_cursor,
        )
    )
    with sync_store.db.backend.transaction() as connection:
        later_keyword = sync_store.db._insert_envelope_in_transaction(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-snapshot-later-keyword",
                domain="notes.keyword",
                operation="upsert",
                object_id=keyword_id,
                device_id="frontend-device",
                client_sequence=3,
                object_revision=1,
                schema_version=1,
                payload={"keyword": "Later dependency"},
                payload_hash="sha256:snapshot-later-keyword",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
                status="accepted",
            ),
            connection=connection,
        )

    with pytest.raises(SyncStoreError, match="resolution envelope was not accepted"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id=dataset_id,
            conflict_id=conflict.conflict_id,
            action="overwrite",
            resolved_by_device_id="frontend-device",
            resolution_envelope=replace(
                source_create,
                client_envelope_id="env-snapshot-link-resolution",
                client_sequence=4,
                status="pending",
            ),
        )
    stored_later = sync_store.get_envelope_by_server_cursor(later_keyword.server_cursor)
    stored_conflict = sync_store.get_conflict(conflict.conflict_id)
    assert stored_later is not None and stored_later.apply_status == "pending"
    assert stored_conflict is not None and stored_conflict.status == "unresolved"
    assert stored_conflict.resolution_action is None


@pytest.mark.parametrize("action", ["skip", "duplicate_rename"])
def test_terminalized_conflict_repoints_head_for_same_object_successor(
    sync_store: SyncV2Store,
    tmp_path: Path,
    action: str,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / f"conflict-{action}-notes.db"),
        client_id="user-1",
    )
    service, baseline, _source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )
    resolution = (
        _m1_note_envelope(
            client_envelope_id="env-resolution-copy",
            object_id="note-copy",
            client_sequence=3,
            object_revision=1,
            payload={"title": "Copy", "content": "Projected copy"},
            payload_hash="sha256:copy",
        )
        if action == "duplicate_rename"
        else None
    )

    service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action=action,
        resolved_by_device_id="device-1",
        resolution_envelope=resolution,
    )
    head = sync_store.get_current_head("dataset-1", "notes.note", "note-original")
    preview = service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note"],
    )
    successor = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id=f"env-after-{action}",
                object_id="note-original",
                device_id="device-2",
                client_sequence=1,
                base_server_cursor=baseline.server_cursor,
                base_object_revision=baseline.object_revision,
                base_object_hash=baseline.payload_hash,
                object_revision=2,
                payload={"title": "Successor", "content": "After resolution"},
                payload_hash="sha256:successor",
            )
        ],
    )

    assert head is not None and head.server_cursor == baseline.server_cursor
    assert baseline.server_cursor in [item.server_cursor for item in preview.ordered_actions]
    assert all(
        item.server_cursor != _source.server_cursor
        for item in preview.ordered_actions
    )
    if action == "duplicate_rename":
        assert {item.object_id for item in preview.ordered_actions} == {
            "note-original",
            "note-copy",
        }
    else:
        assert {item.object_id for item in preview.ordered_actions} == {
            "note-original"
        }
    assert successor.accepted[0].apply_status == "applied"


def test_restore_group_expansion_never_emits_superseded_sibling(
    sync_store: SyncV2Store,
) -> None:
    service = SyncV2Service(
        store=sync_store,
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1})]
        ),
        materializers={},
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    draft_group = [
            _m1_note_envelope(
                client_envelope_id="env-group-superseded",
                object_id="note-superseded",
                client_sequence=1,
                object_revision=1,
                payload_hash="sha256:group-superseded",
                mutation_group_id="server-origin-restore-superseded",
                mutation_step=0,
                mutation_step_count=2,
                mutation_plan_hash="a" * 64,
            ),
            _m1_note_envelope(
                client_envelope_id="env-group-active",
                object_id="note-active",
                client_sequence=2,
                object_revision=1,
                payload_hash="sha256:group-active",
                mutation_group_id="server-origin-restore-superseded",
                mutation_step=1,
                mutation_step_count=2,
                mutation_plan_hash="a" * 64,
            ),
        ]
    plan_hash = mutation_group_plan_hash(draft_group)
    group = sync_store.insert_envelopes_atomic(
        [replace(envelope, mutation_plan_hash=plan_hash) for envelope in draft_group]
    )
    sync_store.mark_envelope_apply_status(
        group[0].server_cursor,
        apply_status="superseded",
        apply_error_code="sync_conflict_skipped",
    )
    sync_store.mark_envelope_apply_status(
        group[1].server_cursor,
        apply_status="applied",
    )

    preview = service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        domains=["notes.note"],
    )

    assert [item.server_cursor for item in preview.ordered_actions] == [
        group[1].server_cursor
    ]


@pytest.mark.parametrize("action", ["skip", "overwrite"])
def test_conflict_resolution_converts_later_legacy_pending_cursor_to_rebase_conflict(
    sync_store: SyncV2Store,
    tmp_path: Path,
    action: str,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / f"conflict-order-{action}.db"),
        client_id="user-1",
    )
    service, baseline, source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )
    intervening_create = _m1_note_envelope(
            client_envelope_id="env-intervening-pending",
            object_id="note-other",
            client_sequence=3,
            object_revision=1,
            payload={"title": "Pending", "content": "Must project first"},
            payload_hash="sha256:pending",
            status="accepted",
        )
    # Explicitly model a row queued by an older server before the append gate.
    with sync_store.db.backend.transaction() as connection:
        intervening = sync_store.db._insert_envelope_in_transaction(
            intervening_create,
            connection=connection,
        )
    resolution = (
        _m1_note_envelope(
            client_envelope_id="env-ordered-resolution",
            object_id="note-original",
            client_sequence=4,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            payload={"title": "Resolved", "content": "Must wait"},
            payload_hash="sha256:ordered-resolution",
        )
        if action == "overwrite"
        else None
    )

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action=action,
        resolved_by_device_id="device-1",
        resolution_envelope=resolution,
    )
    rebased = sync_store.get_envelope_by_server_cursor(intervening.server_cursor)
    rebase_conflict = sync_store.get_unresolved_conflict_for_envelope(
        "dataset-1",
        local_envelope_id=intervening.client_envelope_id,
        server_sequence=intervening.server_cursor,
    )
    stored = sync_store.list_envelopes_after("dataset-1", 0, status=None)
    assert intervening.server_cursor > source.server_cursor
    assert resolved.status in {"dismissed", "resolved"}
    assert rebased is not None and rebased.apply_status == "conflict"
    assert rebased.apply_error_code == "sync_rebase_required_after_conflict_resolution"
    assert rebase_conflict is not None
    assert rebase_conflict.conflict_type == "sync_rebase_required_after_conflict_resolution"
    assert sync_store.get_conflict(conflict.conflict_id).status != "unresolved"
    assert (action == "skip") == all(
        item.client_envelope_id != "env-ordered-resolution" for item in stored
    )


def test_conflict_resolution_reuses_later_legacy_conflict_record_idempotently(
    sync_store: SyncV2Store,
    tmp_path: Path,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / "conflict-existing-rebase.db"),
        client_id="user-1",
    )
    service, _baseline, source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )
    with sync_store.db.backend.transaction() as connection:
        later = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                client_envelope_id="env-existing-later-conflict",
                object_id="note-existing-later-conflict",
                client_sequence=3,
                object_revision=1,
                payload_hash="sha256:existing-later-conflict",
                status="accepted",
            ),
            connection=connection,
        )
    later = sync_store.mark_envelope_apply_status(
        later.server_cursor,
        apply_status="conflict",
        apply_error_code="legacy_projection_conflict",
    )
    existing = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-existing-later",
            dataset_id="dataset-1",
            domain=later.domain,
            entity_id=later.object_id,
            conflict_type="legacy_projection_conflict",
            base_envelope_id="base-envelope-audit",
            local_envelope_id=later.client_envelope_id,
            remote_envelope_id="remote-envelope-audit",
            server_sequence=later.server_cursor,
            metadata={"legacy_reason": "projection"},
        )
    )

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
    )
    replayed = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict.conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
    )
    rebased = sync_store.get_conflict(existing.conflict_id)

    assert resolved == replayed
    assert source.server_cursor < later.server_cursor
    assert rebased is not None
    assert rebased.conflict_id == existing.conflict_id
    assert rebased.conflict_type == "sync_rebase_required_after_conflict_resolution"
    assert rebased.base_envelope_id == "base-envelope-audit"
    assert rebased.remote_envelope_id == "remote-envelope-audit"
    assert rebased.metadata["previous_conflict_type"] == "legacy_projection_conflict"
    assert rebased.metadata["previous_conflict_metadata"] == {
        "legacy_reason": "projection"
    }
    assert len(sync_store.list_conflicts("dataset-1")) == 2


def test_conflict_resolution_rebases_later_dependency_and_paginates_without_queued_history(
    tmp_path: Path,
) -> None:
    (
        _note_db,
        sync_store,
        service,
        dataset_id,
        baseline,
        source,
        conflict,
    ) = _accepted_keyword_conflict_after_applied_predecessor(tmp_path)
    note_id = "22222222-2222-4222-8222-222222222222"
    relationship_id = organization_link_id(
        "notes.keyword_link",
        ["note", note_id, source.object_id],
    )
    relationship_create = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id="env-legacy-dependent-link",
        domain="notes.keyword_link",
        operation="upsert",
        object_id=relationship_id,
        device_id="frontend-device",
        client_sequence=3,
        object_revision=1,
        schema_version=1,
        payload={
            "subject_type": "note",
            "subject_id": note_id,
            "keyword_sync_id": source.object_id,
        },
        payload_hash="sha256:legacy-dependent-link",
        encryption_metadata={"policy": "server_trusted_v1"},
        adapter_version=1,
        status="accepted",
    )
    unrelated_id = "33333333-3333-4333-8333-333333333333"
    unrelated_create = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id="env-legacy-unrelated-keyword",
        domain="notes.keyword",
        operation="upsert",
        object_id=unrelated_id,
        device_id="frontend-device",
        client_sequence=4,
        object_revision=1,
        schema_version=1,
        payload={"keyword": "Unrelated queued change"},
        payload_hash="sha256:legacy-unrelated-keyword",
        encryption_metadata={"policy": "server_trusted_v1"},
        adapter_version=1,
        status="accepted",
    )
    with sync_store.db.backend.transaction() as connection:
        relationship = sync_store.db._insert_envelope_in_transaction(
            relationship_create,
            connection=connection,
        )
        unrelated = sync_store.db._insert_envelope_in_transaction(
            unrelated_create,
            connection=connection,
        )

    cursor = "0"
    pulled_ids: list[str] = []
    while True:
        page = service.pull(
            user_id="user-1",
            dataset_id=dataset_id,
            device_id="device-2",
            cursor=cursor,
            domains=["notes.keyword", "notes.keyword_link"],
            page_size=1,
            include_own_changes=True,
        )
        pulled_ids.extend(item.client_envelope_id for item in page.envelopes)
        assert page.next_cursor != cursor or not page.has_more
        cursor = page.next_cursor or cursor
        if not page.has_more:
            break

    assert relationship.client_envelope_id not in pulled_ids
    assert unrelated.client_envelope_id not in pulled_ids

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="env-keyword-replacement-before-rebase",
            domain="notes.keyword",
            operation="upsert",
            object_id=source.object_id,
            device_id="frontend-device",
            client_sequence=5,
            base_server_cursor=baseline.server_cursor,
            base_object_revision=baseline.object_revision,
            base_object_hash=baseline.payload_hash,
            object_revision=2,
            schema_version=1,
            payload={"keyword": "Replacement"},
            payload_hash="sha256:keyword-replacement-before-rebase",
            encryption_metadata={"policy": "server_trusted_v1"},
            adapter_version=1,
        ),
    )

    assert resolved.status == "resolved"
    rebase_conflicts = []
    for queued in (relationship, unrelated):
        stored = sync_store.get_envelope_by_server_cursor(queued.server_cursor)
        assert stored is not None and stored.apply_status == "conflict"
        assert stored.apply_error_code == "sync_rebase_required_after_conflict_resolution"
        recorded = sync_store.get_unresolved_conflict_for_envelope(
            dataset_id,
            local_envelope_id=queued.client_envelope_id,
            server_sequence=queued.server_cursor,
        )
        assert recorded is not None
        assert recorded.conflict_type == "sync_rebase_required_after_conflict_resolution"
        rebase_conflicts.append(recorded)
        assert sync_store.get_current_head(
            dataset_id,
            queued.domain,
            queued.object_id,
        ) is None

    cursor = "0"
    after_resolution_ids: list[str] = []
    while True:
        page = service.pull(
            user_id="user-1",
            dataset_id=dataset_id,
            device_id="device-2",
            cursor=cursor,
            domains=["notes.keyword", "notes.keyword_link"],
            page_size=1,
            include_own_changes=True,
        )
        after_resolution_ids.extend(
            item.client_envelope_id for item in page.envelopes
        )
        cursor = page.next_cursor or cursor
        if not page.has_more:
            break
    assert "env-keyword-replacement-before-rebase" in after_resolution_ids
    assert relationship.client_envelope_id not in after_resolution_ids
    assert unrelated.client_envelope_id not in after_resolution_ids

    blocked = service.push(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="device-2",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-after-rebase-required",
                domain="notes.keyword",
                operation="upsert",
                object_id="44444444-4444-4444-8444-444444444444",
                device_id="device-2",
                client_sequence=1,
                object_revision=1,
                schema_version=1,
                payload={"keyword": "Must wait"},
                payload_hash="sha256:after-rebase-required",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
            )
        ],
    )
    assert blocked.accepted == []
    assert [item.conflict_id for item in blocked.conflicts] == [
        rebase_conflicts[0].conflict_id
    ]


def test_generated_rebase_conflicts_preserve_original_provenance_when_resolved_in_order(
    tmp_path: Path,
) -> None:
    (
        note_db,
        sync_store,
        service,
        dataset_id,
        baseline,
        source,
        original_conflict,
    ) = _accepted_keyword_conflict_after_applied_predecessor(tmp_path)
    note_id = "22222222-2222-4222-8222-222222222222"
    relationship_id = organization_link_id(
        "notes.keyword_link",
        ["note", note_id, source.object_id],
    )
    with sync_store.db.backend.transaction() as connection:
        note = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                dataset_id=dataset_id,
                client_envelope_id="env-rebase-chain-note",
                object_id=note_id,
                device_id="frontend-device",
                client_sequence=3,
                object_revision=1,
                payload={"title": "Dependency", "content": "Applied directly"},
                payload_hash="sha256:rebase-chain-note",
                status="accepted",
            ),
            connection=connection,
        )
    with sync_store.materialization_guard(
        [note],
        require_predecessors=False,
    ) as guarded_store:
        assert NotesMaterializer(note_db).apply(note, store=guarded_store).status == "applied"

    with sync_store.db.backend.transaction() as connection:
        queued_keyword = sync_store.db._insert_envelope_in_transaction(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-rebase-chain-keyword",
                domain="notes.keyword",
                operation="upsert",
                object_id=source.object_id,
                device_id="frontend-device",
                client_sequence=4,
                base_server_cursor=source.server_cursor,
                base_object_revision=source.object_revision,
                base_object_hash=source.payload_hash,
                object_revision=3,
                schema_version=1,
                payload={"keyword": "Queued same identity"},
                payload_hash="sha256:rebase-chain-keyword",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
                status="accepted",
            ),
            connection=connection,
        )
        queued_relationship = sync_store.db._insert_envelope_in_transaction(
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-rebase-chain-link",
                domain="notes.keyword_link",
                operation="upsert",
                object_id=relationship_id,
                device_id="frontend-device",
                client_sequence=5,
                object_revision=1,
                schema_version=1,
                payload={
                    "subject_type": "note",
                    "subject_id": note_id,
                    "keyword_sync_id": source.object_id,
                },
                payload_hash="sha256:rebase-chain-link",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
                status="accepted",
            ),
            connection=connection,
        )

    original_resolution_request = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id="env-rebase-chain-original-resolution",
        domain="notes.keyword",
        operation="upsert",
        object_id=source.object_id,
        device_id="frontend-device",
        client_sequence=6,
        base_server_cursor=baseline.server_cursor,
        base_object_revision=baseline.object_revision,
        base_object_hash=baseline.payload_hash,
        object_revision=2,
        schema_version=1,
        payload={"keyword": "Original replacement"},
        payload_hash="sha256:rebase-chain-original-resolution",
        encryption_metadata={"policy": "server_trusted_v1"},
        adapter_version=1,
    )
    original_resolution = service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=original_conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=original_resolution_request,
    )
    assert original_resolution == service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=original_conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=original_resolution_request,
    )
    keyword_conflict = sync_store.get_unresolved_conflict_for_envelope(
        dataset_id,
        local_envelope_id=queued_keyword.client_envelope_id,
        server_sequence=queued_keyword.server_cursor,
    )
    relationship_conflict = sync_store.get_unresolved_conflict_for_envelope(
        dataset_id,
        local_envelope_id=queued_relationship.client_envelope_id,
        server_sequence=queued_relationship.server_cursor,
    )
    assert keyword_conflict is not None and relationship_conflict is not None
    relationship_audit = (
        relationship_conflict.conflict_id,
        dict(relationship_conflict.metadata),
    )
    assert relationship_conflict.metadata["source_conflict_id"] == (
        original_conflict.conflict_id
    )

    skipped = service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=keyword_conflict.conflict_id,
        action="skip",
        resolved_by_device_id="frontend-device",
    )
    assert skipped == service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=keyword_conflict.conflict_id,
        action="skip",
        resolved_by_device_id="frontend-device",
    )
    preserved_relationship_conflict = sync_store.get_conflict(
        relationship_conflict.conflict_id
    )
    assert preserved_relationship_conflict is not None
    assert preserved_relationship_conflict.status == "unresolved"
    assert (
        preserved_relationship_conflict.conflict_id,
        dict(preserved_relationship_conflict.metadata),
    ) == relationship_audit

    relationship_resolution_request = SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id="env-rebase-chain-link-resolution",
        domain="notes.keyword_link",
        operation="upsert",
        object_id=relationship_id,
        device_id="frontend-device",
        client_sequence=7,
        object_revision=1,
        schema_version=1,
        payload={
            "subject_type": "note",
            "subject_id": note_id,
            "keyword_sync_id": source.object_id,
        },
        payload_hash="sha256:rebase-chain-link-resolution",
        encryption_metadata={"policy": "server_trusted_v1"},
        adapter_version=1,
    )
    relationship_resolution = service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=relationship_conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=relationship_resolution_request,
    )
    assert relationship_resolution == service.resolve_conflict(
        user_id="user-1",
        dataset_id=dataset_id,
        conflict_id=relationship_conflict.conflict_id,
        action="overwrite",
        resolved_by_device_id="frontend-device",
        resolution_envelope=relationship_resolution_request,
    )

    unblocked = service.push(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="device-2",
        envelopes=[
            SyncEnvelopeCreate(
                dataset_id=dataset_id,
                client_envelope_id="env-rebase-chain-after",
                domain="notes.keyword",
                operation="upsert",
                object_id="44444444-4444-4444-8444-444444444444",
                device_id="device-2",
                client_sequence=1,
                object_revision=1,
                schema_version=1,
                payload={"keyword": "After chain"},
                payload_hash="sha256:rebase-chain-after",
                encryption_metadata={"policy": "server_trusted_v1"},
                adapter_version=1,
            )
        ],
    )
    assert len(unblocked.accepted) == 1

    keyword_head = sync_store.get_current_head(
        dataset_id,
        "notes.keyword",
        source.object_id,
    )
    relationship_head = sync_store.get_current_head(
        dataset_id,
        "notes.keyword_link",
        relationship_id,
    )
    keyword_state = sync_store.get_object_state(
        dataset_id,
        "notes.keyword",
        source.object_id,
    )
    relationship_state = sync_store.get_object_state(
        dataset_id,
        "notes.keyword_link",
        relationship_id,
    )
    assert keyword_head is not None
    assert keyword_head.client_envelope_id == original_resolution_request.client_envelope_id
    assert keyword_state is not None
    assert keyword_state.latest_server_cursor == keyword_head.server_cursor
    assert relationship_head is not None
    assert relationship_head.client_envelope_id == (
        relationship_resolution_request.client_envelope_id
    )
    assert relationship_state is not None
    assert relationship_state.latest_server_cursor == relationship_head.server_cursor

    cursor = "0"
    pulled_ids: list[str] = []
    while True:
        page = service.pull(
            user_id="user-1",
            dataset_id=dataset_id,
            device_id="device-2",
            cursor=cursor,
            domains=["notes.note", "notes.keyword", "notes.keyword_link"],
            page_size=1,
            include_own_changes=True,
        )
        pulled_ids.extend(item.client_envelope_id for item in page.envelopes)
        cursor = page.next_cursor or cursor
        if not page.has_more:
            break
    assert queued_keyword.client_envelope_id not in pulled_ids
    assert queued_relationship.client_envelope_id not in pulled_ids
    assert source.client_envelope_id not in pulled_ids
    assert original_resolution_request.client_envelope_id in pulled_ids
    assert relationship_resolution_request.client_envelope_id in pulled_ids
    assert "env-rebase-chain-after" in pulled_ids


def test_conflict_resolution_fails_safe_when_later_rebase_scan_exceeds_group_cap(
    sync_store: SyncV2Store,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / "conflict-rebase-limit.db"),
        client_id="user-1",
    )
    service, _baseline, source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )
    queued: list[SyncEnvelope] = []
    with sync_store.db.backend.transaction() as connection:
        for sequence in (3, 4):
            queued.append(
                sync_store.db._insert_envelope_in_transaction(
                    _m1_note_envelope(
                        client_envelope_id=f"env-rebase-limit-{sequence}",
                        object_id=f"note-rebase-limit-{sequence}",
                        client_sequence=sequence,
                        object_revision=1,
                        payload_hash=f"sha256:rebase-limit-{sequence}",
                        status="accepted",
                    ),
                    connection=connection,
                )
            )
    monkeypatch.setattr(sync_db_module, "SYNC_MUTATION_GROUP_MAX_SIZE", 1)
    resolution = _m1_note_envelope(
        client_envelope_id="env-rebase-limit-resolution",
        object_id="note-original",
        client_sequence=5,
        base_server_cursor=_baseline.server_cursor,
        base_object_revision=_baseline.object_revision,
        base_object_hash=_baseline.payload_hash,
        object_revision=2,
        payload={"title": "Must not project", "content": "Limit validation failed"},
        payload_hash="sha256:rebase-limit-resolution",
    )

    with pytest.raises(
        SyncStoreError,
        match="sync_conflict_resolution_rebase_limit_exceeded",
    ):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict.conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
            resolution_envelope=resolution,
        )

    stored_source = sync_store.get_envelope_by_server_cursor(source.server_cursor)
    assert stored_source is not None and stored_source.apply_status == "conflict"
    assert sync_store.get_conflict(conflict.conflict_id).status == "unresolved"
    assert all(
        sync_store.get_envelope_by_server_cursor(item.server_cursor).apply_status == "pending"
        for item in queued
    )
    note = note_db.get_note_by_id("note-original")
    assert note is not None
    assert note["title"] == "Applied"
    assert note["content"] == "Projected baseline"


def test_conflict_resolution_validates_incompatible_rebase_record_before_product_write(
    sync_store: SyncV2Store,
    tmp_path: Path,
) -> None:
    note_db = CharactersRAGDB(
        db_path=str(tmp_path / "conflict-rebase-incompatible.db"),
        client_id="user-1",
    )
    service, baseline, source, conflict = _accepted_conflict_after_applied_predecessor(
        sync_store,
        note_db,
    )
    with sync_store.db.backend.transaction() as connection:
        later = sync_store.db._insert_envelope_in_transaction(
            _m1_note_envelope(
                client_envelope_id="env-incompatible-later-conflict",
                object_id="note-incompatible-later-conflict",
                client_sequence=3,
                object_revision=1,
                payload_hash="sha256:incompatible-later-conflict",
                status="accepted",
            ),
            connection=connection,
        )
    existing = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-incompatible-later",
            dataset_id="dataset-1",
            domain=later.domain,
            entity_id=later.object_id,
            conflict_type="legacy_projection_conflict",
            local_envelope_id=later.client_envelope_id,
            server_sequence=later.server_cursor,
        )
    )
    sync_store.resolve_conflict(
        existing.conflict_id,
        dataset_id="dataset-1",
        server_cursor=later.server_cursor,
        status="dismissed",
        resolution_action="skip",
    )

    with pytest.raises(
        SyncStoreError,
        match="sync_conflict_resolution_rebase_record_incompatible",
    ):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict.conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id="env-incompatible-resolution",
                object_id="note-original",
                client_sequence=4,
                base_server_cursor=baseline.server_cursor,
                base_object_revision=baseline.object_revision,
                base_object_hash=baseline.payload_hash,
                object_revision=2,
                payload={"title": "Must not project", "content": "Bad plan"},
                payload_hash="sha256:incompatible-resolution",
            ),
        )

    note = note_db.get_note_by_id("note-original")
    assert note is not None
    assert note["title"] == "Applied"
    assert note["content"] == "Projected baseline"
    assert all(
        envelope.client_envelope_id != "env-incompatible-resolution"
        for envelope in sync_store.list_envelopes_after(
            "dataset-1",
            0,
            status=None,
        )
    )
    stored_source = sync_store.get_envelope_by_server_cursor(source.server_cursor)
    assert stored_source is not None and stored_source.apply_status == "conflict"


def test_unresolved_accepted_materialization_conflict_blocks_new_client_append(
    sync_store: SyncV2Store,
) -> None:
    service, _materializer = _accepted_materialization_conflict_service(sync_store)
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-materialization-conflict",
                object_id="note-original",
                object_revision=1,
                payload_hash="sha256:original",
            )
        ],
    )
    later = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-must-not-queue",
                object_id="note-later",
                device_id="device-2",
                client_sequence=1,
                object_revision=1,
                payload_hash="sha256:later",
            )
        ],
    )
    accepted = sync_store.list_envelopes_after(
        "dataset-1",
        0,
        status="accepted",
    )

    assert len(pushed.conflicts) == 1
    assert later.accepted == []
    assert len(later.conflicts) == 1
    assert later.conflicts[0].conflict_id == pushed.conflicts[0].conflict_id
    assert [item.client_envelope_id for item in accepted] == [
        "env-materialization-conflict"
    ]
    assert len(sync_store.list_envelopes_after("dataset-1", 0, status=None)) == 1
    assert len(sync_store.list_conflicts("dataset-1")) == 1


def test_preflight_note_conflicts_reuse_accepted_materialization_blocker_without_writes(
    sync_store: SyncV2Store,
) -> None:
    service, _materializer = _accepted_materialization_conflict_service(
        sync_store,
        real_notes_adapter=True,
    )
    blocked = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-materialization-conflict",
                object_id="note-original",
                object_revision=1,
                payload={"title": "Blocked", "content": "Never projected"},
                payload_hash="sha256:blocked",
            )
        ],
    )
    blocker = blocked.conflicts[0]
    history_count = len(sync_store.list_envelopes_after("dataset-1", 0, status=None))
    conflict_count = len(sync_store.list_conflicts("dataset-1"))

    results = [
        service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            envelopes=[
                _m1_note_envelope(
                    client_envelope_id=f"env-unique-preflight-{sequence}",
                    object_id="note-original",
                    device_id="device-2",
                    client_sequence=sequence,
                    base_server_cursor=0,
                    base_object_revision=0,
                    base_object_hash="sha256:stale",
                    object_revision=2,
                    payload={"title": f"Changed {sequence}", "content": "Unique"},
                    payload_hash=f"sha256:unique-{sequence}",
                )
            ],
        )
        for sequence in (1, 2)
    ]

    assert [result.conflicts[0].conflict_id for result in results] == [
        blocker.conflict_id,
        blocker.conflict_id,
    ]
    assert all(result.accepted == [] and result.rejected == [] for result in results)
    assert len(sync_store.list_envelopes_after("dataset-1", 0, status=None)) == history_count
    assert len(sync_store.list_conflicts("dataset-1")) == conflict_count


def test_preflight_conflict_rechecks_materialization_blocker_after_evaluation_race(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _materializer = _accepted_materialization_conflict_service(
        sync_store,
        real_notes_adapter=True,
    )
    baseline_result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-race-baseline",
                object_id="note-race",
                object_revision=1,
                payload={"title": "Baseline", "content": "Applied"},
                payload_hash="sha256:race-baseline",
            )
        ],
    )
    baseline = sync_store.get_envelope_by_server_cursor(
        baseline_result.accepted[0].server_sequence
    )
    assert baseline is not None
    evaluated = Event()
    release = Event()
    original_evaluate = service._evaluate_envelope

    def blocking_evaluate(dataset, envelope, *, context=None):
        outcome = original_evaluate(dataset, envelope, context=context)
        if envelope.client_envelope_id == "env-racing-preflight":
            assert isinstance(outcome, AdapterConflict)
            evaluated.set()
            assert release.wait(timeout=5)
        return outcome

    monkeypatch.setattr(service, "_evaluate_envelope", blocking_evaluate)
    racing = _m1_note_envelope(
        client_envelope_id="env-racing-preflight",
        object_id="note-race",
        device_id="device-2",
        client_sequence=1,
        base_server_cursor=0,
        base_object_revision=0,
        base_object_hash="sha256:stale",
        object_revision=2,
        payload={"title": "Racing", "content": "Changed"},
        payload_hash="sha256:racing",
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            service.push,
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            envelopes=[racing],
        )
        assert evaluated.wait(timeout=5)
        source = sync_store.insert_envelope(
            _m1_note_envelope(
                client_envelope_id="env-race-blocker",
                object_id="note-blocker",
                client_sequence=2,
                object_revision=1,
                payload={"title": "Blocker", "content": "Never projected"},
                payload_hash="sha256:race-blocker",
                status="accepted",
            )
        )
        source = sync_store.mark_envelope_apply_status(
            source.server_cursor,
            apply_status="conflict",
            apply_error_code="projection_conflict",
        )
        blocker = sync_store.insert_conflict(
            SyncConflictCreate(
                conflict_id="conflict-race-blocker",
                dataset_id="dataset-1",
                domain="notes.note",
                entity_id=source.object_id,
                conflict_type="projection_conflict",
                local_envelope_id=source.client_envelope_id,
                server_sequence=source.server_cursor,
            )
        )
        release.set()
        result = future.result(timeout=5)

    assert result.accepted == [] and result.rejected == []
    assert [item.conflict_id for item in result.conflicts] == [blocker.conflict_id]
    assert all(
        item.client_envelope_id != racing.client_envelope_id
        for item in sync_store.list_envelopes_after("dataset-1", 0, status=None)
    )


def test_push_records_materialization_conflict_in_bound_sync_transaction(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _materializer = _accepted_materialization_conflict_service(sync_store)
    observed_connections: list[object | None] = []
    original_insert = sync_store.db.insert_conflict

    def record_insert(conflict, *, connection=None):
        observed_connections.append(connection)
        return original_insert(conflict, connection=connection)

    monkeypatch.setattr(sync_store.db, "insert_conflict", record_insert)

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-materialization-conflict",
                object_id="note-original",
                object_revision=1,
                payload_hash="sha256:original",
            )
        ],
    )

    assert len(result.conflicts) == 1
    assert len(observed_connections) == 1
    assert observed_connections[0] is not None


def test_skip_terminalizes_accepted_conflict_and_unblocks_later_projection(
    sync_store: SyncV2Store,
) -> None:
    service, _materializer = _accepted_materialization_conflict_service(sync_store)
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-materialization-conflict",
                object_id="note-original",
                object_revision=1,
                payload_hash="sha256:original",
            )
        ],
    )

    dismissed = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=pushed.conflicts[0].conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
    )
    later = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-after-resolution",
                object_id="note-later",
                device_id="device-2",
                client_sequence=1,
                object_revision=1,
                payload_hash="sha256:later",
            )
        ],
    )
    source = next(
        item
        for item in sync_store.list_envelopes_after("dataset-1", 0, status=None)
        if item.client_envelope_id == "env-materialization-conflict"
    )

    assert dismissed.status == "dismissed"
    assert source.apply_status == "superseded"
    assert source.apply_error_code == "sync_conflict_skipped"
    assert later.accepted[0].apply_status == "applied"


def test_skip_holds_dataset_guard_until_terminalization_and_resolution_commit(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, materializer = _accepted_materialization_conflict_service(sync_store)
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-materialization-conflict",
                object_id="note-original",
                object_revision=1,
                payload_hash="sha256:original",
            )
        ],
    )
    resolve_entered = Event()
    release_resolve = Event()
    original_resolve = sync_store.db.resolve_conflict

    def blocking_resolve(*args, **kwargs):
        resolve_entered.set()
        assert release_resolve.wait(timeout=5)
        return original_resolve(*args, **kwargs)

    monkeypatch.setattr(sync_store.db, "resolve_conflict", blocking_resolve)

    def skip():
        return service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=pushed.conflicts[0].conflict_id,
            action="skip",
            resolved_by_device_id="device-1",
        )

    def push_later():
        return service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            envelopes=[
                _m1_note_envelope(
                    client_envelope_id="env-after-resolution",
                    object_id="note-later",
                    device_id="device-2",
                    client_sequence=1,
                    object_revision=1,
                    payload_hash="sha256:later",
                )
            ],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        skipping = pool.submit(skip)
        assert resolve_entered.wait(timeout=5)
        later = pool.submit(push_later)
        assert not later.done()
        assert not materializer.later_entered.wait(timeout=0.2)
        assert not later.done()
        release_resolve.set()
        assert skipping.result(timeout=5).status == "dismissed"
        assert later.result(timeout=5).accepted[0].apply_status == "applied"

    assert materializer.later_entered.is_set()


def test_resolution_guard_serializes_later_projection_until_conflict_is_terminal(
    sync_store: SyncV2Store,
) -> None:
    service, materializer = _accepted_materialization_conflict_service(sync_store)
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-materialization-conflict",
                object_id="note-original",
                object_revision=1,
                payload_hash="sha256:original",
            )
        ],
    )
    release_resolution = Event()
    materializer.release_resolution = release_resolution

    def resolve():
        return service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=pushed.conflicts[0].conflict_id,
            action="duplicate_rename",
            resolved_by_device_id="device-1",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id="env-resolution-copy",
                object_id="note-copy",
                client_sequence=2,
                object_revision=1,
                payload_hash="sha256:copy",
            ),
        )

    def push_later():
        return service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            envelopes=[
                _m1_note_envelope(
                    client_envelope_id="env-after-resolution",
                    object_id="note-later",
                    device_id="device-2",
                    client_sequence=1,
                    object_revision=1,
                    payload_hash="sha256:later",
                )
            ],
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        resolving = pool.submit(resolve)
        assert materializer.resolution_entered.wait(timeout=5)
        later = pool.submit(push_later)
        assert not materializer.later_entered.wait(timeout=0.2)
        release_resolution.set()
        assert resolving.result(timeout=5).status == "resolved"
        assert later.result(timeout=5).accepted[0].apply_status == "applied"

    assert materializer.later_entered.is_set()


@pytest.mark.parametrize(
    ("object_id", "expected_apply_status"),
    [
        ("note-fail", "failed"),
        ("note-conflict", "conflict"),
    ],
)
def test_resolve_conflict_rejects_unapplied_resolution_envelope_without_closing_original(
    sync_store: SyncV2Store,
    object_id: str,
    expected_apply_status: str,
):
    id_counter = {"value": 0}

    def id_factory(prefix: str) -> str:
        id_counter["value"] += 1
        return f"{prefix}-{id_counter['value']}"

    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id=object_id,
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=id_factory,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id=object_id,
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="resolution envelope was not applied"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id=f"env-resolution-{expected_apply_status}",
                object_id=object_id,
                client_sequence=2,
                object_revision=1,
                payload_hash=f"sha256:{expected_apply_status}",
            ),
        )
    conflict = sync_store.get_conflict(conflict_id)
    envelopes = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    assert conflict.status == "unresolved"
    assert conflict.resolution_action is None
    assert all(
        item.client_envelope_id != f"env-resolution-{expected_apply_status}"
        for item in envelopes
    )


@pytest.mark.parametrize(
    ("action", "resolution_object_id"),
    [
        ("overwrite", "note-1"),
        ("duplicate_rename", "note-copy"),
    ],
)
def test_resolve_conflict_rejects_preclaimed_resolution_before_materialization(
    sync_store: SyncV2Store,
    action: str,
    resolution_object_id: str,
):
    materializer = _CountingMaterializer()
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": materializer},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    before_state = sync_store.get_object_state("dataset-1", "notes.note", resolution_object_id)
    sync_store.db.execute(
        """
        UPDATE sync_conflicts
           SET resolution_action = ?,
               resolved_by_device_id = ?,
               resolution_notes = ?
         WHERE conflict_id = ?
        """,
        (action, "device-2", "other resolution", conflict_id),
    )

    with pytest.raises(SyncStoreError, match="already claimed"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action=action,
            resolved_by_device_id="device-1",
            notes="losing resolution",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id="env-losing-resolution",
                object_id=resolution_object_id,
                client_sequence=2,
                object_revision=1,
                payload_hash="sha256:losing",
            ),
        )
    conflict = sync_store.get_conflict(conflict_id)
    envelopes = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    assert materializer.calls == 0
    assert sync_store.get_object_state("dataset-1", "notes.note", resolution_object_id) == before_state
    assert conflict.status == "unresolved"
    assert conflict.resolution_action == action
    assert conflict.resolved_by_device_id == "device-2"
    assert conflict.resolution_notes == "other resolution"
    assert all(item.client_envelope_id != "env-losing-resolution" for item in envelopes)


@pytest.mark.parametrize(
    ("object_id", "expected_apply_status"),
    [
        ("note-fail", "failed"),
        ("note-conflict", "conflict"),
    ],
)
def test_resolve_conflict_releases_claim_when_resolution_envelope_does_not_apply(
    sync_store: SyncV2Store,
    object_id: str,
    expected_apply_status: str,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id=object_id,
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id=object_id,
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    with pytest.raises(SyncStoreError, match="resolution envelope was not applied"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
            notes="cleanup claim",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id=f"env-cleanup-{expected_apply_status}",
                object_id=object_id,
                client_sequence=2,
                object_revision=1,
                payload_hash=f"sha256:cleanup-{expected_apply_status}",
            ),
        )
    conflict = sync_store.get_conflict(conflict_id)
    envelopes = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    assert conflict.status == "unresolved"
    assert conflict.resolution_action is None
    assert conflict.resolved_by_device_id is None
    assert conflict.resolution_notes is None
    assert conflict.resolved_by_envelope_id is None
    assert all(
        item.client_envelope_id != f"env-cleanup-{expected_apply_status}"
        for item in envelopes
    )


def test_resolve_conflict_skip_persists_m1_action_without_mutating_envelopes(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    before = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=pushed.conflicts[0].conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
    )
    after = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    assert resolved.status == "dismissed"
    assert resolved.resolution_action == "skip"
    assert resolved.resolved_by_envelope_id is None
    assert [item.envelope_id for item in after] == [item.envelope_id for item in before]
    assert after[0].apply_status == "superseded"
    assert after[0].apply_error_code == "sync_conflict_skipped"


def test_resolve_conflict_replay_same_resolved_decision_is_idempotent(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    first = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
        notes="same decision",
    )
    monkeypatch.setattr(sync_db_module, "utcnow_iso", lambda: "2099-01-01T00:00:00+00:00")

    replayed = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
        notes="same decision",
    )

    assert replayed == first
    assert sync_store.get_conflict(conflict_id) == first


def test_resolve_conflict_rejects_conflicting_second_resolution_without_mutation(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict_id,
        action="skip",
        resolved_by_device_id="device-1",
        notes="original decision",
    )

    with pytest.raises(SyncStoreError, match="already resolved"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="skip",
            resolved_by_device_id="device-1",
            notes="different decision",
        )

    assert sync_store.get_conflict(conflict_id) == resolved


def test_resolve_conflict_rejects_replay_with_changed_resolution_envelope_fingerprint(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        materializers={"notes.note": _OutcomeMaterializer()},
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                object_revision=1,
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    resolved = service.resolve_conflict(
        user_id="user-1",
        dataset_id="dataset-1",
        conflict_id=conflict_id,
        action="overwrite",
        resolved_by_device_id="device-1",
        resolution_envelope=_m1_note_envelope(
            client_envelope_id="env-resolution",
            object_id="note-1",
            client_sequence=2,
            object_revision=1,
            payload={"title": "Accepted resolution"},
            payload_hash="sha256:resolution",
            routing_metadata={"route": "original"},
        ),
    )

    with pytest.raises(SyncStoreError, match="already resolved"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id="env-resolution",
                object_id="note-1",
                client_sequence=2,
                object_revision=1,
                payload={"title": "Changed resolution"},
                payload_hash="sha256:resolution",
                routing_metadata={"route": "changed"},
            ),
        )

    assert sync_store.get_conflict(conflict_id) == resolved


def test_resolve_conflict_duplicate_rename_requires_resolution_envelope(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-original",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-original",
                payload_hash="sha256:original",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="requires a resolution envelope"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="duplicate_rename",
            resolved_by_device_id="device-1",
        )

    assert sync_store.get_conflict(conflict_id).status == "unresolved"


def test_resolve_conflict_overwrite_requires_resolution_envelope_without_mutation(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-original",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-original",
                payload_hash="sha256:original",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id
    before = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    with pytest.raises(SyncStoreError, match="overwrite requires a resolution envelope"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="overwrite",
            resolved_by_device_id="device-1",
        )
    after = sync_store.list_envelopes_after("dataset-1", 0, status=None)

    assert sync_store.get_conflict(conflict_id).status == "unresolved"
    assert [(item.envelope_id, item.status, item.apply_status) for item in after] == [
        (item.envelope_id, item.status, item.apply_status) for item in before
    ]


def test_resolve_conflict_skip_rejects_resolution_envelope_without_mutation(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-original",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-original",
                payload_hash="sha256:original",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="skip.*resolution envelope"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="skip",
            resolved_by_device_id="device-1",
            resolution_envelope=_m1_note_envelope(
                client_envelope_id="env-skip-resolution",
                object_id="note-original",
                client_sequence=2,
                payload={"title": "Should not be applied"},
                payload_hash="sha256:skip-resolution",
            ),
        )

    conflict = sync_store.get_conflict(conflict_id)
    assert conflict.status == "unresolved"
    assert conflict.resolved_by_envelope_id is None
    stored_envelope_ids = {
        envelope.client_envelope_id
        for envelope in sync_store.list_envelopes_after("dataset-1", 0)
    }
    assert "env-skip-resolution" not in stored_envelope_ids


def test_resolve_conflict_rejects_non_m1_dismiss_action_without_mutation(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-conflict",
                object_id="note-1",
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="resolution action is not supported"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-1",
            conflict_id=conflict_id,
            action="dismiss",
            resolved_by_device_id="device-1",
        )

    assert sync_store.get_conflict(conflict_id).status == "unresolved"


def test_resolve_conflict_rejects_expected_dataset_mismatch_without_mutation(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict-b": AdapterConflict(
                    client_envelope_id="env-conflict-b",
                    domain="notes.note",
                    entity_id="note-b",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-a",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-b",
        domains=["notes.note"],
        encryption_policy="server_trusted_v1",
    )
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-b",
        device_id="device-1",
        envelopes=[
            _m1_note_envelope(
                dataset_id="dataset-b",
                client_envelope_id="env-conflict-b",
                object_id="note-b",
                payload_hash="sha256:note-b",
            )
        ],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        service.resolve_conflict(
            user_id="user-1",
            dataset_id="dataset-a",
            conflict_id=conflict_id,
            action="skip",
        )

    assert sync_store.get_conflict(conflict_id).status == "unresolved"


def test_resolve_conflict_uses_direct_lookup_without_dataset_conflict_scan(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )

    def fail_scan(*_args, **_kwargs):
        raise AssertionError("resolve_conflict should not scan dataset conflict lists")

    monkeypatch.setattr(sync_store, "list_datasets_for_user", fail_scan)
    monkeypatch.setattr(sync_store, "list_conflicts", fail_scan)

    resolved = service.resolve_conflict(
        user_id="user-1",
        conflict_id=pushed.conflicts[0].conflict_id,
        action="skip",
    )

    assert resolved.status == "dismissed"
    assert resolved.resolution_action == "skip"


def test_resolve_conflict_rejects_conflicts_owned_by_another_user(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    _register_devices(service, "user-2", "device-2")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )

    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        service.resolve_conflict(
            user_id="user-2",
            conflict_id=pushed.conflicts[0].conflict_id,
            action="skip",
        )


def test_device_scoped_operations_require_registered_user_device(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    with pytest.raises(SyncStoreError, match="Sync device was not found"):
        service.push(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="unregistered-device",
            envelopes=[_envelope(client_envelope_id="env-unregistered")],
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found"):
        service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="unregistered-device",
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found"):
        service.resolve_conflict(
            user_id="user-1",
            conflict_id=conflict_id,
            action="skip",
            resolved_by_device_id="unregistered-device",
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found"):
        service.resolve_conflict(
            user_id="user-1",
            conflict_id=conflict_id,
            action="overwrite",
            resolution_envelope=_envelope(
                client_envelope_id="env-unregistered-resolution",
                operation="upsert",
                device_id="unregistered-device",
                payload_hash="sha256:unregistered-resolution",
            ),
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found"):
        service.store_key_recovery_bundle(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="unregistered-device",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:key",
        )


def test_list_key_recovery_bundles_returns_filtered_opaque_material(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:opaque-dataset-key",
        kdf_metadata={"algorithm": "scrypt", "salt": "opaque-salt"},
        recovery_hint="laptop",
    )

    records = sync_service.list_key_recovery_bundles(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
    )
    wrong_device = sync_service.list_key_recovery_bundles(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        key_purpose="dataset_recovery",
    )
    wrong_purpose = sync_service.list_key_recovery_bundles(
        user_id="user-1",
        dataset_id="dataset-1",
        key_purpose="workspace_share",
    )
    manifest = sync_service.restore_manifest(user_id="user-1")

    assert len(records) == 1
    assert records[0].wrapped_key_blob == "wrapped:opaque-dataset-key"
    assert records[0].kdf_metadata == {"algorithm": "scrypt", "salt": "opaque-salt"}
    assert records[0].recovery_hint == "laptop"
    assert records[0].revoked_at is None
    assert wrong_device == []
    assert wrong_purpose == []
    assert manifest.datasets[0].key_recovery_available is True
    assert "wrapped:opaque-dataset-key" not in str(manifest)
    assert "opaque-salt" not in str(manifest)


def test_store_key_recovery_bundle_persists_epoch_rotation_metadata(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    stored = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:passphrase-key",
        kdf_metadata={"algorithm": "scrypt", "salt": "opaque-salt"},
        recovery_hint="vault passphrase",
        encryption_policy="passphrase_wrapped_v1",
        key_epoch=3,
        active_from_server_sequence=12,
        wrapped_for="passphrase",
        rewrap_status="pending",
    )
    records = sync_service.list_key_recovery_bundles(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
    )

    assert records == [stored]
    assert stored.encryption_policy == "passphrase_wrapped_v1"
    assert stored.key_epoch == 3
    assert stored.active_from_server_sequence == 12
    assert stored.superseded_at is None
    assert stored.wrapped_for == "passphrase"
    assert stored.rewrap_status == "pending"
    assert stored.wrapped_key_blob == "wrapped:passphrase-key"


def test_store_key_recovery_bundle_validates_purpose_wrapping_metadata_and_rotation(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
):
    issued_ids: list[str] = []

    def _unique_id(prefix: str) -> str:
        issued_ids.append(prefix)
        return f"{prefix}-{len(issued_ids)}"

    sync_service.id_factory = _unique_id
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    original = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:original-key",
        kdf_metadata={"algorithm": "scrypt", "salt": "opaque-salt"},
    )
    sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-revoked",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:revoked-key",
            kdf_metadata={"algorithm": "scrypt", "salt": "revoked-salt"},
            revoked_at="2026-05-10T12:30:00+00:00",
        )
    )

    invalid_cases = [
        {"key_purpose": "workspace_share"},
        {"kdf_metadata": {}},
        {"kdf_metadata": {"algorithm": "scrypt"}},
        {"rotation_of_key_record_id": "missing-key"},
        {"rotation_of_key_record_id": "key-revoked"},
    ]
    for overrides in invalid_cases:
        kwargs = {
            "user_id": "user-1",
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped:super-secret-key-material",
            "kdf_metadata": {"algorithm": "scrypt", "salt": "opaque-salt"},
        }
        kwargs.update(overrides)

        with pytest.raises(SyncStoreError, match="Sync key recovery bundle is invalid") as exc_info:
            sync_service.store_key_recovery_bundle(**kwargs)

        assert "wrapped:super-secret-key-material" not in str(exc_info.value)
        assert "opaque-salt" not in str(exc_info.value)

    rotated = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:rotated-key",
        kdf_metadata={"algorithm": "scrypt", "salt": "rotated-salt"},
        rotation_of_key_record_id=original.key_record_id,
    )

    assert rotated.rotation_of_key_record_id == original.key_record_id


def test_key_rotation_preview_reports_redacted_next_epoch_and_retained_range(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_service.register_device(
        user_id="user-1",
        display_name="Tablet",
        client_type="chatbook",
        device_id="device-3",
    )
    recovery_key = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:current-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "secret-salt"},
        recovery_hint="laptop",
    )
    sync_service.push(
        user_id="user-1",
        device_id="device-1",
        dataset_id="dataset-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-1",
                object_id="note-1",
                client_sequence=1,
                payload_hash="sha256:note-1",
            ),
            _m1_note_envelope(
                client_envelope_id="env-2",
                object_id="note-2",
                client_sequence=2,
                payload_hash="sha256:note-2",
            ),
        ],
    )

    preview = sync_service.preview_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        target_encryption_policy="passphrase_wrapped_v1",
        source_key_record_ids=[recovery_key.key_record_id],
    )

    assert preview.dataset_id == "dataset-1"
    assert preview.target_encryption_policy == "passphrase_wrapped_v1"
    assert preview.next_key_epoch == 2
    assert preview.active_from_server_sequence == 3
    assert preview.can_commit is True
    assert preview.committed is False
    assert preview.retained_envelope_range.from_server_sequence == 1
    assert preview.retained_envelope_range.through_server_sequence == 2
    assert [record.key_record_id for record in preview.affected_key_records] == [
        recovery_key.key_record_id
    ]
    assert preview.device_ids == ["device-1"]
    assert preview.recovery_target_count == 1
    assert preview.blockers == []
    assert "wrapped:current-secret" not in str(preview)
    assert "secret-salt" not in str(preview)


def test_key_rotation_commit_is_idempotent_and_supersedes_source_records(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    recovery_key = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:current-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "secret-salt"},
        recovery_hint="laptop",
    )
    sync_service.push(
        user_id="user-1",
        device_id="device-1",
        dataset_id="dataset-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-1",
                object_id="note-1",
                client_sequence=1,
                payload_hash="sha256:note-1",
            )
        ],
    )

    committed = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        rotation_id="rotation-1",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
        source_key_record_ids=[recovery_key.key_record_id],
        wrapped_for="passphrase",
    )
    repeated = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        rotation_id="rotation-1",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
        source_key_record_ids=[recovery_key.key_record_id],
        wrapped_for="passphrase",
    )
    records = sync_service.list_key_recovery_bundles(
        user_id="user-1",
        dataset_id="dataset-1",
        key_purpose="dataset_recovery",
    )
    by_id = {record.key_record_id: record for record in records}

    assert committed == repeated
    assert committed.committed is True
    assert committed.next_key_epoch == 2
    assert committed.active_from_server_sequence == 2
    assert committed.new_key_record is not None
    assert committed.new_key_record.key_epoch == 2
    assert committed.new_key_record.rotation_of_key_record_id == recovery_key.key_record_id
    assert committed.new_key_record.wrapped_for == "passphrase"
    assert committed.new_key_record.rewrap_status == "complete"
    assert committed.affected_key_records[0].superseded_at == _clock()
    assert by_id[recovery_key.key_record_id].superseded_at == _clock()
    assert by_id[committed.new_key_record.key_record_id].wrapped_key_blob == "wrapped:new-secret"
    assert "wrapped:new-secret" not in str(committed)
    assert "new-secret-salt" not in str(committed)


def test_key_rotation_commit_recomputes_epoch_boundary_inside_commit(
    sync_service: SyncV2Service,
    monkeypatch: pytest.MonkeyPatch,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    recovery_key = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:current-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "secret-salt"},
    )
    sync_service.push(
        user_id="user-1",
        device_id="device-1",
        dataset_id="dataset-1",
        envelopes=[
            _m1_note_envelope(
                client_envelope_id="env-1",
                object_id="note-1",
                client_sequence=1,
                payload_hash="sha256:note-1",
            )
        ],
    )
    original_commit_key_rotation = sync_service.store.commit_key_rotation

    def push_between_preview_and_store_commit(*args, **kwargs):
        sync_service.push(
            user_id="user-1",
            device_id="device-1",
            dataset_id="dataset-1",
            envelopes=[
                _m1_note_envelope(
                    client_envelope_id="env-2",
                    object_id="note-2",
                    client_sequence=2,
                    payload_hash="sha256:note-2",
                )
            ],
        )
        return original_commit_key_rotation(*args, **kwargs)

    monkeypatch.setattr(
        sync_service.store,
        "commit_key_rotation",
        push_between_preview_and_store_commit,
    )

    committed = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        rotation_id="rotation-1",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
        source_key_record_ids=[recovery_key.key_record_id],
        wrapped_for="passphrase",
    )

    assert committed.active_from_server_sequence == 3
    assert committed.retained_envelope_range.through_server_sequence == 2
    assert committed.retained_envelope_range.envelope_count == 2


def test_key_rotation_commit_replay_requires_same_source_manifest(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    source_a = sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-source-a",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:current-secret-a",
            kdf_metadata={"algorithm": "scrypt", "salt": "secret-salt-a"},
        )
    )
    source_b = sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-source-b",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-2",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:current-secret-b",
            kdf_metadata={"algorithm": "scrypt", "salt": "secret-salt-b"},
        )
    )

    committed = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        rotation_id="rotation-1",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
        source_key_record_ids=[source_a.key_record_id, source_b.key_record_id],
        wrapped_for="passphrase",
    )
    repeated = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        rotation_id="rotation-1",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
        source_key_record_ids=[source_b.key_record_id, source_a.key_record_id],
        wrapped_for="passphrase",
    )

    assert {
        record.key_record_id for record in repeated.affected_key_records
    } == {source_a.key_record_id, source_b.key_record_id}
    assert repeated == committed

    with pytest.raises(SyncIdempotencyConflictError):
        sync_service.commit_key_rotation(
            user_id="user-1",
            dataset_id="dataset-1",
            rotation_id="rotation-1",
            target_encryption_policy="passphrase_wrapped_v1",
            wrapped_key_blob="wrapped:new-secret",
            kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
            source_key_record_ids=[source_a.key_record_id],
            wrapped_for="passphrase",
        )

    with pytest.raises(SyncStoreError, match="Sync key rotation is invalid"):
        sync_service.commit_key_rotation(
            user_id="user-1",
            dataset_id="dataset-1",
            rotation_id="rotation-1",
            target_encryption_policy="passphrase_wrapped_v1",
            wrapped_key_blob="wrapped:new-secret",
            kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
            source_key_record_ids=["missing-key"],
            wrapped_for="passphrase",
        )


def test_key_rotation_ids_are_scoped_by_user_dataset_and_rotation_id(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-2", domains=["notes.note"])
    source_1 = sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-dataset-1-source",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:dataset-1-secret",
            kdf_metadata={"algorithm": "scrypt", "salt": "dataset-1-salt"},
        )
    )
    source_2 = sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-dataset-2-source",
            dataset_id="dataset-2",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:dataset-2-secret",
            kdf_metadata={"algorithm": "scrypt", "salt": "dataset-2-salt"},
        )
    )

    dataset_1_rotation = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-1",
        rotation_id="same-client-operation-id",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-dataset-1-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-dataset-1-salt"},
        source_key_record_ids=[source_1.key_record_id],
        wrapped_for="passphrase",
    )
    dataset_2_rotation = sync_service.commit_key_rotation(
        user_id="user-1",
        dataset_id="dataset-2",
        rotation_id="same-client-operation-id",
        target_encryption_policy="passphrase_wrapped_v1",
        wrapped_key_blob="wrapped:new-dataset-2-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "new-dataset-2-salt"},
        source_key_record_ids=[source_2.key_record_id],
        wrapped_for="passphrase",
    )

    assert dataset_1_rotation.new_key_record is not None
    assert dataset_2_rotation.new_key_record is not None
    assert dataset_1_rotation.new_key_record.key_record_id != dataset_2_rotation.new_key_record.key_record_id
    assert dataset_1_rotation.dataset_id == "dataset-1"
    assert dataset_2_rotation.dataset_id == "dataset-2"


def test_key_rotation_commit_rejects_revoked_or_missing_source_without_secret_leak(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-revoked",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:revoked-secret",
            kdf_metadata={"algorithm": "scrypt", "salt": "revoked-salt"},
            revoked_at="2026-05-10T12:30:00+00:00",
        )
    )

    for source_key_record_ids in (["missing-key"], ["key-revoked"]):
        with pytest.raises(SyncStoreError, match="Sync key rotation is invalid") as exc_info:
            sync_service.commit_key_rotation(
                user_id="user-1",
                dataset_id="dataset-1",
                rotation_id="rotation-invalid",
                target_encryption_policy="passphrase_wrapped_v1",
                wrapped_key_blob="wrapped:new-secret",
                kdf_metadata={"algorithm": "scrypt", "salt": "new-secret-salt"},
                source_key_record_ids=source_key_record_ids,
                wrapped_for="passphrase",
            )

        assert "wrapped:new-secret" not in str(exc_info.value)
        assert "new-secret-salt" not in str(exc_info.value)


def test_restore_preview_warns_when_selected_dataset_lacks_active_key_recovery(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-revoked",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:revoked-key",
            kdf_metadata={"algorithm": "scrypt", "salt": "revoked-salt"},
            revoked_at="2026-05-10T12:30:00+00:00",
        )
    )

    manifest = sync_service.restore_manifest(user_id="user-1", dataset_ids=["dataset-1"])
    preview = sync_service.restore_preview(user_id="user-1", dataset_ids=["dataset-1"])
    records = sync_service.list_key_recovery_bundles(
        user_id="user-1",
        dataset_id="dataset-1",
        key_purpose="dataset_recovery",
    )

    assert manifest.datasets[0].key_recovery_available is False
    assert preview.key_status == {"dataset-1": {"key_recovery_available": False}}
    assert preview.datasets[0].key_recovery_available is False
    assert [warning.code for warning in preview.warnings] == ["sync_key_recovery_missing"]
    assert preview.warnings[0].dataset_id == "dataset-1"
    assert "revoked-key" not in repr(preview)
    assert records == []


def test_list_key_recovery_bundles_rejects_inaccessible_dataset(
    sync_service: SyncV2Service,
):
    with pytest.raises(SyncStoreError, match="Sync dataset was not found"):
        sync_service.list_key_recovery_bundles(
            user_id="user-1",
            dataset_id="missing-dataset",
            key_purpose="dataset_recovery",
        )


def test_resolve_conflict_rejects_unsupported_action_without_mutation(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )

    with pytest.raises(SyncStoreError, match="resolution action is not supported"):
        service.resolve_conflict(
            user_id="user-1",
            conflict_id=pushed.conflicts[0].conflict_id,
            action="merge",
            resolved_by_device_id="device-1",
        )


def test_pull_uses_stable_server_cursor(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-1")],
    )

    first_pull = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
        include_own_changes=True,
    )
    second_pull = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=first_pull.next_cursor,
        include_own_changes=True,
    )

    assert [envelope.client_envelope_id for envelope in first_pull.envelopes] == ["env-1"]
    assert first_pull.next_cursor == "1"
    assert second_pull.envelopes == []
    assert second_pull.next_cursor == "1"


def test_adapter_cursor_version_upgrade_replays_v2_without_rewinding_v1(
    sync_store: SyncV2Store,
) -> None:
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(
            max_pull_page_size=10,
            pull_token_signing_secret="test-only-pull-secret",
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1]},
        },
    )
    service.enroll_dataset(
        user_id="user-1", dataset_id="dataset-1", domains=["notes.note"]
    )
    v1 = sync_store.insert_envelope(
        _envelope(client_envelope_id="v1-history", adapter_version=1)
    )
    v2 = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="v2-history",
            entity_id="note-v2",
            stable_key="note:v2",
            payload_hash="sha256:v2-history",
            adapter_version=2,
            schema_version=2,
        )
    )
    sync_store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            adapter_version=1,
            last_pulled_sequence=v1.server_sequence,
            max_delivered_sequence=v1.server_sequence,
        )
    )
    service.update_device(
        user_id="user-1",
        device_id="device-1",
        capabilities={
            "supported_adapter_versions": {"notes.note": [1, 2]},
        },
    )

    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        include_own_changes=True,
    )

    assert [item.client_envelope_id for item in pulled.envelopes] == ["v2-history"]
    assert pulled.next_cursor is not None and not pulled.next_cursor.isdigit()
    stored_v1 = sync_store.get_device_cursor(
        "dataset-1", "device-1", "notes.note", adapter_version=1
    )
    stored_v2 = sync_store.get_device_cursor(
        "dataset-1", "device-1", "notes.note", adapter_version=2
    )
    assert stored_v1 is not None and stored_v1.max_delivered_sequence == v1.server_sequence
    assert stored_v2 is not None and stored_v2.max_delivered_sequence == v2.server_sequence


@pytest.mark.unit
def test_versioned_pull_does_not_advance_past_unresolved_conflict(
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(
            max_pull_page_size=10,
            pull_token_signing_secret="test-only-pull-secret",
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
        },
    )
    service.enroll_dataset(
        user_id="user-1", dataset_id="dataset-1", domains=["notes.note"]
    )
    blocked = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="blocked-v2",
            adapter_version=2,
            schema_version=2,
            status="accepted",
            apply_status="applied",
        )
    )
    later = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="later-v2",
            entity_id="note-later-v2",
            stable_key="note:later-v2",
            payload_hash="sha256:later-v2",
            adapter_version=2,
            schema_version=2,
            status="accepted",
            apply_status="applied",
        )
    )
    blocked = sync_store.mark_envelope_apply_status(
        blocked.server_sequence,
        apply_status="conflict",
        apply_error_code="projection_conflict",
    )
    conflict = sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-blocked-v2",
            dataset_id="dataset-1",
            domain="notes.note",
            object_id=blocked.object_id,
            conflict_type="projection_conflict",
            local_envelope_id=blocked.client_envelope_id,
            server_cursor=blocked.server_sequence,
        )
    )

    first = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        include_own_changes=True,
    )
    monkeypatch.setattr(
        sync_store,
        "get_unresolved_materialization_conflict",
        lambda _dataset_id: None,
    )
    second = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor=first.next_cursor,
        include_own_changes=True,
    )

    assert conflict.server_sequence == blocked.server_sequence
    assert first.envelopes == []
    assert [item.client_envelope_id for item in second.envelopes] == [
        later.client_envelope_id
    ]


def test_pull_token_rejects_tampering_oversize_and_negotiated_version_set_change(
    sync_store: SyncV2Store,
) -> None:
    registry = SyncAdapterRegistry(
        [
            StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2}),
            StaticSyncAdapter(domain="chat.conversation", supported_adapter_versions={1}),
        ]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(
            max_pull_page_size=1,
            pull_token_signing_secret="test-only-pull-secret",
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
        },
    )
    service.enroll_dataset(
        user_id="user-1", dataset_id="dataset-1", domains=["notes.note"]
    )
    sync_store.insert_envelope(_envelope(client_envelope_id="token-v1"))
    sync_store.insert_envelope(
        _envelope(
            client_envelope_id="token-v2",
            entity_id="note-v2",
            stable_key="note:v2",
            payload_hash="sha256:token-v2",
            adapter_version=2,
            schema_version=2,
        )
    )
    first = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        page_size=1,
        include_own_changes=True,
    )
    assert first.next_cursor is not None

    with pytest.raises(SyncStoreError, match="sync_pull_token_invalid"):
        service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            cursor=first.next_cursor[:-1] + ("A" if first.next_cursor[-1] != "A" else "B"),
            include_own_changes=True,
        )
    with pytest.raises(SyncStoreError, match="sync_pull_token_too_large"):
        service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            cursor="x" * 32_769,
            include_own_changes=True,
        )

    second = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor=first.next_cursor,
        page_size=1,
        include_own_changes=True,
    )
    delivered_v2 = next(
        envelope for envelope in second.envelopes if envelope.adapter_version == 2
    )
    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain_acks=[
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=2,
                through_server_sequence=delivered_v2.server_sequence,
                applied_at=_clock(),
            )
        ],
    )

    service.update_device(
        user_id="user-1",
        device_id="device-1",
        capabilities={
            "requested_domains": ["chat.conversation"],
            "supported_adapter_versions": {"chat.conversation": [1]},
        },
    )
    with pytest.raises(SyncStoreError, match="sync_pull_restart_required"):
        service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            cursor=first.next_cursor,
            include_own_changes=True,
        )


def test_version_ack_accepts_only_exact_adapter_stream_delivered_by_pull(
    sync_store: SyncV2Store,
) -> None:
    registry = SyncAdapterRegistry(
        [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        settings=SyncV2Settings(
            pull_token_signing_secret="test-only-pull-secret",
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
        },
    )
    service.enroll_dataset(
        user_id="user-1", dataset_id="dataset-1", domains=["notes.note"]
    )
    delivered = sync_store.insert_envelope(
        _envelope(
            client_envelope_id="ack-v2",
            adapter_version=2,
            schema_version=2,
        )
    )
    service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        include_own_changes=True,
    )

    summary = service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain_acks=[
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-1",
                device_id="device-1",
                domain="notes.note",
                adapter_version=2,
                through_server_sequence=delivered.server_sequence,
                applied_at=_clock(),
            )
        ],
    )
    with pytest.raises(SyncStoreError, match="delivered watermark"):
        service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            domain_acks=[
                SyncDeviceDomainAckCreate(
                    dataset_id="dataset-1",
                    device_id="device-1",
                    domain="notes.note",
                    adapter_version=1,
                    through_server_sequence=delivered.server_sequence,
                    applied_at=_clock(),
                )
            ],
        )

    service.register_device(
        user_id="user-1",
        display_name="Legacy client",
        client_type="chatbook",
        device_id="legacy-device",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1]},
        },
    )
    unacknowledged = service._retention_unacknowledged_devices(
        dataset_id="dataset-1",
        domain="notes.note",
        adapter_version=2,
        server_sequence=delivered.server_sequence,
        active_devices=sync_store.list_devices_for_user("user-1"),
    )

    assert summary.version_acks[0].adapter_version == 2
    assert unacknowledged == []


@pytest.mark.parametrize("later_failure", ["domain", "blob", "blob_id"])
def test_mixed_ack_batch_rolls_back_every_earlier_ack_type(
    sync_service: SyncV2Service,
    later_failure: str,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    delivered = sync_service.store.insert_envelope(_envelope())
    sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        include_own_changes=True,
    )
    valid_domain = SyncDeviceDomainAckCreate(
        dataset_id="dataset-1",
        device_id="device-2",
        domain="notes.note",
        through_server_sequence=delivered.server_sequence,
        applied_at=_clock(),
    )
    valid_blob = SyncDeviceBlobAckCreate(
        dataset_id="dataset-1",
        device_id="device-2",
        attachment_id="attachment-1",
        payload_hash="sha256:" + "a" * 64,
        verified_at=_clock(),
    )
    domain_acks = [valid_domain]
    blob_acks: list[SyncDeviceBlobAckCreate] = []
    blob_id_acks: list[SyncDeviceBlobIdAckCreate] = []
    if later_failure == "domain":
        domain_acks.append(
            replace(
                valid_domain,
                through_server_sequence=delivered.server_sequence + 1,
            )
        )
    elif later_failure == "blob":
        blob_acks.extend(
            [valid_blob, replace(valid_blob, dataset_id="other-dataset")]
        )
    else:
        blob_acks.append(valid_blob)
        blob_id_acks.append(
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-1",
                device_id="device-2",
                blob_id="missing-blob",
                payload_hash="sha256:" + "b" * 64,
                verified_at=_clock(),
            )
        )

    with pytest.raises(SyncStoreError):
        sync_service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            domain_acks=domain_acks,
            blob_acks=blob_acks,
            blob_id_acks=blob_id_acks,
        )

    summary = sync_service.store.list_device_acknowledgments(
        "dataset-1",
        "device-2",
    )
    assert summary.version_acks == []
    assert summary.blob_acks == []
    assert summary.blob_id_acks == []


def test_implicit_pull_isolates_legacy_device_and_rejects_explicit_unsupported_domain(
    sync_store: SyncV2Store,
) -> None:
    registry = SyncAdapterRegistry(
        [
            StaticSyncAdapter(domain=domain, supported_adapter_versions={1})
            for domain in [*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]
        ]
    )
    service = SyncV2Service(store=sync_store, adapters=registry, clock=_clock)
    legacy_domains = list(M1_SYNC_DOMAINS)
    upgraded_domains = [*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]
    service.register_device(
        user_id="user-1",
        display_name="Legacy",
        client_type="chatbook",
        device_id="legacy-device",
        capabilities={"requested_domains": legacy_domains},
    )
    service.register_device(
        user_id="user-1",
        display_name="Upgraded",
        client_type="chatbook",
        device_id="upgraded-device",
        capabilities={"requested_domains": upgraded_domains},
    )
    sync_store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            domains=upgraded_domains,
            metadata={
                "notes_organization_v1": {
                    "bootstrap_id": "internal-bootstrap-id",
                    "state": "ready",
                    "captured_count": 1,
                    "expected_count": 1,
                    "error_code": None,
                }
            },
        )
    )
    sync_store.insert_envelope(
        _m1_note_envelope(
            dataset_id="dataset-1",
            client_envelope_id="core-note",
            object_id="note-core",
            device_id="upgraded-device",
        )
    )
    sync_store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="organization-keyword",
            domain="notes.keyword",
            operation="upsert",
            object_id="11111111-1111-4111-8111-111111111111",
            device_id="upgraded-device",
            object_revision=1,
            payload={"keyword": "Research"},
            payload_hash="sha256:organization-keyword",
        )
    )

    legacy = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="legacy-device",
        cursor=0,
        include_own_changes=True,
    )
    upgraded = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="upgraded-device",
        cursor=0,
        include_own_changes=True,
    )

    assert [item.domain for item in legacy.envelopes] == ["notes.note"]
    assert {item.domain for item in upgraded.envelopes} == {
        "notes.note",
        "notes.keyword",
    }
    with pytest.raises(SyncStoreError, match="sync_device_domain_not_supported"):
        service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="legacy-device",
            cursor=0,
            domains=["notes.keyword"],
            include_own_changes=True,
        )


def test_pull_does_not_persist_empty_explicit_high_cursor(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-1")],
    )

    poisoned = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="999",
        include_own_changes=True,
    )
    later = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        include_own_changes=True,
    )

    assert poisoned.envelopes == []
    assert [envelope.client_envelope_id for envelope in later.envelopes] == ["env-1"]


def test_v1_explicit_cursor_pages_record_only_delivered_watermarks(
    sync_service: SyncV2Service,
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    for sequence in range(1, 4):
        sync_service.store.insert_envelope(
            _envelope(
                client_envelope_id=f"explicit-page-{sequence}",
                entity_id=f"note-{sequence}",
                stable_key=f"note:{sequence}",
                payload_hash=f"sha256:explicit-page-{sequence}",
                client_sequence=sequence,
            )
        )

    cursor: str | int = 0
    delivered: list[int] = []
    for _page_number in range(3):
        page = sync_service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            cursor=cursor,
            page_size=1,
            include_own_changes=True,
        )
        assert len(page.envelopes) == 1
        sequence = page.envelopes[0].server_sequence
        delivered.append(sequence)
        sync_service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            domain_acks=[
                SyncDeviceDomainAckCreate(
                    dataset_id="dataset-1",
                    device_id="device-2",
                    domain="notes.note",
                    through_server_sequence=sequence,
                    applied_at=_clock(),
                )
            ],
        )
        cursor = page.next_cursor or cursor

    stored = sync_service.store.get_device_cursor(
        "dataset-1",
        "device-2",
        "notes.note",
    )
    assert delivered == sorted(delivered)
    assert stored is not None
    assert stored.last_pulled_sequence == delivered[-1]
    assert stored.max_delivered_sequence == delivered[-1]


def test_pull_does_not_persist_visible_empty_explicit_cursor_over_echoes(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="remote-before-echo")],
    )
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        envelopes=[
            _envelope(
                client_envelope_id="own-echo",
                entity_id="note-own-echo",
                stable_key="note:own-echo",
                device_id="device-2",
                payload_hash="sha256:own-echo",
            )
        ],
    )

    poisoned = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor=str(first.next_cursor),
        include_own_changes=False,
    )
    later = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        include_own_changes=False,
    )

    assert poisoned.envelopes == []
    assert [envelope.client_envelope_id for envelope in later.envelopes] == [
        "remote-before-echo"
    ]


def test_pull_rejects_invalid_cursor(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    with pytest.raises(SyncStoreError, match="Invalid sync cursor"):
        sync_service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            cursor="not-a-cursor",
        )


def test_pull_rejects_non_positive_page_size(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    with pytest.raises(SyncStoreError, match="page_size must be greater than zero"):
        sync_service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            page_size=0,
        )


def test_default_clock_and_id_factory_are_not_repeatable(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
):
    service = SyncV2Service(store=sync_store, adapters=registry)

    first_device = service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
    )
    second_device = service.register_device(
        user_id="user-1",
        display_name="Phone",
        client_type="chatbook",
    )
    first_dataset = service.enroll_dataset(user_id="user-1")
    second_dataset = service.enroll_dataset(user_id="user-1")

    assert first_device.device.device_id.startswith("device-")
    assert first_device.device.device_id != "device-"
    assert second_device.device.device_id != first_device.device.device_id
    assert first_dataset.dataset.dataset_id.startswith("dataset-")
    assert second_dataset.dataset.dataset_id != first_dataset.dataset.dataset_id
    assert service.capabilities().server_time is not None


def test_pull_propagates_cursor_persistence_errors(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-1")],
    )

    def fail_cursor_update(*_args, **_kwargs):
        raise SyncStoreError("cursor write failed")

    monkeypatch.setattr(sync_store, "update_device_cursor", fail_cursor_update)

    with pytest.raises(SyncStoreError, match="cursor write failed"):
        sync_service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-2",
            include_own_changes=True,
        )


def test_pull_treats_missing_domain_cursors_as_zero(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "chat.conversation"])
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="chat-before-notes",
                domain="chat.conversation",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:chat-before-notes",
            ),
            _envelope(
                client_envelope_id="note-pulled-first",
                entity_id="note-1",
                stable_key="note:1",
                payload_hash="sha256:note-pulled-first",
            ),
        ],
    )
    notes_only = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        domains=["notes.note"],
        include_own_changes=True,
    )

    multi_domain = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        domains=["notes.note", "chat.conversation"],
        include_own_changes=True,
    )

    assert [envelope.client_envelope_id for envelope in notes_only.envelopes] == [
        "note-pulled-first"
    ]
    assert "chat-before-notes" in [
        envelope.client_envelope_id for envelope in multi_domain.envelopes
    ]


def test_pull_honors_filters_echo_policy_page_size_and_has_more(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "chat.conversation"])
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="own-note", entity_id="note-own", payload_hash="sha256:own"),
        ],
    )
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        envelopes=[
            _envelope(
                client_envelope_id="remote-chat",
                domain="chat.conversation",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                device_id="device-2",
                payload_hash="sha256:chat",
            ),
            _envelope(
                client_envelope_id="remote-note-1",
                entity_id="note-remote-1",
                stable_key="note:remote-1",
                device_id="device-2",
                payload_hash="sha256:remote-1",
            ),
            _envelope(
                client_envelope_id="remote-note-2",
                entity_id="note-remote-2",
                stable_key="note:remote-2",
                device_id="device-2",
                payload_hash="sha256:remote-2",
            ),
        ],
    )

    page = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor="0",
        domains=["notes.note"],
        page_size=1,
        include_own_changes=False,
    )
    next_page = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor=page.next_cursor,
        domains=["notes.note"],
        page_size=1,
        include_own_changes=False,
    )

    assert [envelope.client_envelope_id for envelope in page.envelopes] == ["remote-note-1"]
    assert page.next_cursor == "3"
    assert page.has_more is True
    assert [envelope.client_envelope_id for envelope in next_page.envelopes] == ["remote-note-2"]
    assert next_page.has_more is False


def test_pull_excludes_conflict_envelopes_from_normal_results(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes.note",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes.note",
                    entity_id="note-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    push_result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )

    pull_result = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        cursor="0",
        domains=["notes.note"],
        include_own_changes=True,
    )

    assert [conflict.client_envelope_id for conflict in push_result.conflicts] == ["env-conflict"]
    assert pull_result.envelopes == []


def test_pull_uses_server_side_filters_past_echo_filled_raw_window(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    monkeypatch: pytest.MonkeyPatch,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(max_batch_size=20, max_pull_page_size=1),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    own_envelopes = [
        _envelope(
            client_envelope_id=f"own-{index}",
            entity_id=f"note-own-{index}",
            stable_key=f"note:own:{index}",
            payload_hash=f"sha256:own-{index}",
        )
        for index in range(11)
    ]
    remote_envelope = _envelope(
        client_envelope_id="remote-after-echoes",
        entity_id="note-remote",
        stable_key="note:remote",
        device_id="device-2",
        payload_hash="sha256:remote",
    )
    service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=own_envelopes,
    )
    service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        envelopes=[remote_envelope],
    )
    original_list_envelopes_after = sync_store.list_envelopes_after
    list_calls: list[dict[str, object]] = []

    def tracked_list_envelopes_after(*args, **kwargs):
        list_calls.append(dict(kwargs))
        return original_list_envelopes_after(*args, **kwargs)

    monkeypatch.setattr(sync_store, "list_envelopes_after", tracked_list_envelopes_after)

    page = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor="0",
        domains=["notes.note"],
        page_size=1,
        include_own_changes=False,
    )

    assert [envelope.client_envelope_id for envelope in page.envelopes] == [
        "remote-after-echoes"
    ]
    assert list_calls == [
        {
            "limit": 2,
            "domains": ["notes.note"],
            "adapter_versions": [1],
            "status": "accepted",
            "exclude_device_id": "device-1",
        }
    ]
    assert page.next_cursor == "12"
    assert page.has_more is False


def test_restore_manifest_is_metadata_only_and_includes_inventory_status(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
    monkeypatch: pytest.MonkeyPatch,
):
    sync_service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Private Laptop",
        client_type="chatbook",
        client_version="0.1.0",
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
        metadata={"label": "known private label"},
    )
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="note-private",
                payload_ciphertext="ciphertext:known-private-note",
                payload_clear={"status": "active"},
                payload_size_bytes=128,
            ),
            _envelope(
                client_envelope_id="attachment-ref",
                domain="attachment.ref",
                entity_id="attachment-1",
                stable_key="attachment:1",
                payload_hash="sha256:attachment",
                payload={
                    "attachment_id": "attachment-1",
                    "parent_domain": "notes.note",
                    "parent_object_id": "note-1",
                    "content_type": "application/octet-stream",
                    "size_bytes": 512,
                    "payload_hash": "sha256:attachment",
                    "availability": "client_local",
                },
                payload_size_bytes=512,
            ),
        ],
    )
    sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-1",
            dataset_id="dataset-1",
            domain="notes.note",
            entity_id="note-1",
            conflict_type="version_divergence",
        )
    )
    sync_store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-1",
            dataset_id="dataset-1",
            user_id="user-1",
            device_id="device-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="wrapped:secret-key",
        )
    )

    def fail_scan(*_args, **_kwargs):
        raise AssertionError("restore_manifest should use aggregate store summaries")

    monkeypatch.setattr(sync_store, "list_envelopes_after", fail_scan)
    monkeypatch.setattr(sync_store, "list_conflicts", fail_scan)
    monkeypatch.setattr(sync_store, "list_key_records", fail_scan)

    manifest = sync_service.restore_manifest(user_id="user-1")

    assert manifest.generated_at == _clock()
    assert manifest.devices[0].device_id == "device-1"
    assert manifest.devices[0].last_seen_at is not None
    assert manifest.datasets[0].dataset_id == "dataset-1"
    assert manifest.datasets[0].encryption_policy == "server_trusted_v1"
    assert manifest.datasets[0].metadata == {"label": "known private label"}
    assert manifest.datasets[0].approximate_counts == {"notes.note": 1, "attachment.ref": 1}
    assert manifest.datasets[0].unresolved_conflicts == 1
    assert manifest.datasets[0].attachment_availability == {}
    assert manifest.datasets[0].attachment_size_classes == {}
    assert manifest.datasets[0].key_recovery_available is True
    assert "ciphertext:known-private-note" not in repr(manifest)
    assert "wrapped:secret-key" not in repr(manifest)


def test_restore_preview_reports_m2_restore_completeness_states(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_attachment_bytes=64,
            max_blob_bytes=128,
            max_chunk_bytes=8,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "chat.conversation", "attachment.ref"],
    )
    service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="note-restore"),
            _envelope(
                client_envelope_id="conversation-restore",
                domain="chat.conversation",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:conversation-1",
            ),
            _envelope(
                client_envelope_id="attachment-available-ref",
                domain="attachment.ref",
                entity_id="attachment-available",
                stable_key="attachment:available",
                payload_hash=_sha256(b"available payload"),
                payload={
                    "attachment_id": "attachment-available",
                    "parent_domain": "notes.note",
                    "parent_object_id": "note-1",
                    "content_type": "application/octet-stream",
                    "size_bytes": len(b"available payload"),
                    "payload_hash": _sha256(b"available payload"),
                    "availability": "client_local",
                },
            ),
            _envelope(
                client_envelope_id="attachment-missing-ref",
                domain="attachment.ref",
                entity_id="attachment-missing",
                stable_key="attachment:missing",
                payload_hash=_sha256(b"missing payload"),
                payload={
                    "attachment_id": "attachment-missing",
                    "parent_domain": "notes.note",
                    "parent_object_id": "note-1",
                    "content_type": "application/octet-stream",
                    "size_bytes": len(b"missing payload"),
                    "payload_hash": _sha256(b"missing payload"),
                    "availability": "client_local",
                },
            ),
        ],
    )
    available_payload = b"available payload"
    service.store_attachment(
        user_id="user-1",
        dataset_id="dataset-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-available",
        content_type="application/octet-stream",
        size_bytes=len(available_payload),
        payload_ciphertext=available_payload.decode("utf-8"),
        payload_hash=_sha256(available_payload),
    )

    blob_incomplete = service.restore_preview(user_id="user-1", dataset_ids=["dataset-1"])
    content_complete = service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        selected_attachment_ids=["attachment-available"],
    )
    verified_complete = service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        selected_attachment_ids=["attachment-available"],
        attachment_availability={"attachment-available": "verified"},
    )
    metadata_ready = service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        selected_attachment_ids=["attachment-missing"],
        metadata_only=True,
    )
    blocked_by_conflicts = service.restore_preview(
        user_id="user-1",
        dataset_ids=["dataset-1"],
        selected_object_ids=["conversation-1"],
        local_inventory=[
            {
                "domain": "chat.conversation",
                "object_id": "conversation-1",
                "object_revision": 1,
                "object_hash": "sha256:local-conversation",
                "deleted": False,
            }
        ],
    )

    assert blob_incomplete.restore_status == "blob_incomplete"
    assert [blob.attachment_id for blob in blob_incomplete.blob_details] == [
        "attachment-available",
        "attachment-missing",
    ]
    assert [blob.server_availability for blob in blob_incomplete.blob_details] == [
        "available",
        "metadata_only",
    ]
    assert content_complete.restore_status == "content_complete"
    content_attachment_detail = next(
        item for item in content_complete.domain_details if item.domain == "attachment.ref"
    )
    assert content_attachment_detail.status == "content_complete"
    assert verified_complete.restore_status == "verified_complete"
    verified_attachment_detail = next(
        item for item in verified_complete.domain_details if item.domain == "attachment.ref"
    )
    assert verified_attachment_detail.verified_blob_count == 1
    assert metadata_ready.restore_status == "metadata_ready"
    assert metadata_ready.blob_details[0].required_for_restore is False
    assert blocked_by_conflicts.restore_status == "blocked_by_conflicts"
    assert blocked_by_conflicts.object_conflicts[0].domain == "chat.conversation"


def test_blob_upload_session_chunk_and_complete_flow_commits_blob(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=64,
            max_chunk_bytes=8,
            user_blob_quota_bytes=128,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"hello world"

    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-1",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=6,
        chunk_count=2,
        idempotency_key="upload-key-1",
    )
    duplicate = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-1",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=6,
        chunk_count=2,
        idempotency_key="upload-key-1",
    )
    first_chunk = service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload[:6],
        chunk_hash=_sha256(payload[:6]),
    )
    second_chunk = service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        chunk_index=1,
        offset_bytes=6,
        chunk_payload=payload[6:],
        chunk_hash=_sha256(payload[6:]),
    )
    blob = service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
    )
    quota = sync_store.summarize_blob_quota("user-1", dataset_id="dataset-1")

    assert duplicate.upload_id == session.upload_id
    assert first_chunk.chunk_index == 0
    assert second_chunk.chunk_index == 1
    assert blob.attachment_id == "attachment-1"
    assert blob.status == "available"
    assert service.blob_store is not None
    assert service.blob_store.read_blob(blob.storage_key) == payload
    assert quota.reserved_blob_bytes == 0
    assert quota.used_blob_bytes == len(payload)
    assert quota.active_upload_count == 0


def test_blob_upload_completion_returns_committed_blob_when_cleanup_fails(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    blob_store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=blob_store,
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=64,
            max_chunk_bytes=16,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"cleanup failure"
    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-1",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=len(payload),
        chunk_count=1,
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )

    def fail_discard(_upload_id: str) -> None:
        raise OSError("cleanup blocked")

    monkeypatch.setattr(blob_store, "discard_upload", fail_discard)

    blob = service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
    )

    assert blob.status == "available"
    assert blob_store.read_blob(blob.storage_key) == payload


def test_blob_upload_conflicting_duplicate_chunk_does_not_overwrite_existing_chunk(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
) -> None:
    blob_store = LocalSyncBlobStore(tmp_path / "sync_blobs")
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=blob_store,
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=64,
            max_chunk_bytes=16,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )
    payload = b"original"
    conflicting_payload = b"rewritte"
    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-1",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_hash=_sha256(payload),
        chunk_size=len(payload),
        chunk_count=1,
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )

    with pytest.raises(SyncIdempotencyConflictError, match="different content"):
        service.upload_blob_chunk(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=session.upload_id,
            chunk_index=0,
            offset_bytes=0,
            chunk_payload=conflicting_payload,
            chunk_hash=_sha256(conflicting_payload),
        )

    blob = service.complete_blob_upload(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
    )

    assert blob_store.read_blob(blob.storage_key) == payload


def test_blob_upload_rejects_bad_hash_domain_and_quota(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=32,
            max_chunk_bytes=8,
            user_blob_quota_bytes=8,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    with pytest.raises(SyncStoreError, match="quota"):
        service.create_blob_upload_session(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            entity_id="note-1",
            attachment_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=9,
            payload_hash=_sha256(b"123456789"),
            chunk_size=8,
            chunk_count=2,
        )

    with pytest.raises(SyncStoreError, match="not enrolled"):
        service.create_blob_upload_session(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            domain="chat.conversation",
            entity_id="chat-1",
            attachment_id="attachment-2",
            content_type="application/octet-stream",
            size_bytes=4,
            payload_hash=_sha256(b"data"),
            chunk_size=4,
            chunk_count=1,
        )

    session = service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-3",
        content_type="application/octet-stream",
        size_bytes=4,
        payload_hash=_sha256(b"data"),
        chunk_size=4,
        chunk_count=1,
    )
    with pytest.raises(SyncStoreError, match="hash"):
        service.upload_blob_chunk(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=session.upload_id,
            chunk_index=0,
            offset_bytes=0,
            chunk_payload=b"data",
            chunk_hash="sha256:" + "0" * 64,
        )


def test_store_attachment_uses_blob_upload_commit_path_for_small_blobs(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_attachment_bytes=64,
            max_blob_bytes=64,
            max_chunk_bytes=64,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"small encrypted payload"

    attachment = service.store_attachment(
        user_id="user-1",
        dataset_id="dataset-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-small",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_ciphertext=payload.decode("utf-8"),
        payload_hash=_sha256(payload),
    )
    quota = sync_store.summarize_blob_quota("user-1", dataset_id="dataset-1")

    assert attachment.stored is True
    assert attachment.metadata["blob_id"] == "blob-generated"
    assert quota.used_blob_bytes == len(payload)


def test_blob_download_manifest_and_bytes_for_available_blob(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_attachment_bytes=64,
            max_blob_bytes=64,
            max_chunk_bytes=8,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"downloadable payload"
    service.store_attachment(
        user_id="user-1",
        dataset_id="dataset-1",
        domain="notes.note",
        entity_id="note-1",
        attachment_id="attachment-download",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_ciphertext=payload.decode("utf-8"),
        payload_hash=_sha256(payload),
    )

    manifest = service.blob_download_manifest(
        user_id="user-1",
        dataset_id="dataset-1",
        attachment_id="attachment-download",
        chunk_size=8,
    )
    body = service.read_blob_bytes(
        user_id="user-1",
        dataset_id="dataset-1",
        attachment_id="attachment-download",
        offset=5,
        size=8,
    )

    assert manifest.availability == "available"
    assert manifest.size_bytes == len(payload)
    assert manifest.payload_hash == _sha256(payload)
    assert [chunk.chunk_index for chunk in manifest.chunks] == [0, 1, 2]
    assert manifest.chunks[0].chunk_hash == _sha256(payload[:8])
    assert body == payload[5:13]


def test_blob_download_manifest_reports_metadata_only_attachment_ref(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="attachment-ref",
                domain="attachment.ref",
                entity_id="attachment-meta",
                stable_key="attachment:meta",
                payload_hash=_sha256(b"metadata-only"),
                payload={
                    "attachment_id": "attachment-meta",
                    "parent_domain": "notes.note",
                    "parent_object_id": "note-1",
                    "content_type": "text/plain",
                    "size_bytes": 13,
                    "payload_hash": _sha256(b"metadata-only"),
                    "availability": "metadata_only",
                },
            )
        ],
    )

    manifest = service.blob_download_manifest(
        user_id="user-1",
        dataset_id="dataset-1",
        attachment_id="attachment-meta",
    )

    assert manifest.availability == "metadata_only"
    assert manifest.content_type == "text/plain"
    assert manifest.chunks == []


def test_blob_download_rejects_missing_and_cross_user_blob(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
    tmp_path: Path,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_attachment_bytes=64,
            max_blob_bytes=64,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    _register_devices(service, "user-1", "device-1")
    service.register_device(
        user_id="user-2",
        device_id="device-2",
        display_name="Other",
        client_type="chatbook",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    service.enroll_dataset(user_id="user-2", dataset_id="dataset-2", domains=["notes.note", "attachment.ref"])
    payload = b"private payload"
    service.store_attachment(
        user_id="user-2",
        dataset_id="dataset-2",
        domain="notes.note",
        entity_id="note-2",
        attachment_id="attachment-private",
        content_type="application/octet-stream",
        size_bytes=len(payload),
        payload_ciphertext=payload.decode("utf-8"),
        payload_hash=_sha256(payload),
    )

    with pytest.raises(SyncStoreError, match="not found|not accessible"):
        service.blob_download_manifest(
            user_id="user-1",
            dataset_id="dataset-1",
            attachment_id="missing",
        )
    with pytest.raises(SyncStoreError, match="not found|not accessible"):
        service.read_blob_bytes(
            user_id="user-1",
            dataset_id="dataset-2",
            attachment_id="attachment-private",
        )


def test_store_attachment_rejects_blob_transfer_in_m1(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    with pytest.raises(SyncStoreError, match="sync_blob_transfer_not_supported"):
        sync_service.store_attachment(
            user_id="user-1",
            dataset_id="dataset-1",
            domain="notes.note",
            entity_id="note-1",
            attachment_id="attachment-1",
            content_type="application/octet-stream",
            size_bytes=512,
            payload_ciphertext="ciphertext:attachment-secret",
            payload_hash="sha256:attachment",
            encryption_policy="client_private_v1",
            metadata={"slot": "body-image"},
        )


def test_store_attachment_rejects_before_domain_size_and_policy_checks_in_m1(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )

    with pytest.raises(SyncStoreError, match="sync_blob_transfer_not_supported"):
        sync_service.store_attachment(
            user_id="user-2",
            dataset_id="dataset-1",
            domain="notes.note",
            entity_id="note-1",
            attachment_id="attachment-forbidden",
            content_type="application/octet-stream",
            size_bytes=128,
            payload_ciphertext="ciphertext:forbidden",
            payload_hash="sha256:forbidden",
            encryption_policy="client_private_v1",
        )
