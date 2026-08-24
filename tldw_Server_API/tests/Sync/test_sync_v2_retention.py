from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Notes_Tasks.projection_markers import (
    TaskMarker,
    task_marker_hash,
)
from tldw_Server_API.app.core.Sync.v2 import service as service_module
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.materializers import NotesTaskMaterializer
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncBlobObjectCreate,
    SyncDatasetCreate,
    SyncDeviceBlobAckCreate,
    SyncDeviceBlobIdAckCreate,
    SyncDeviceDomainAckCreate,
    SyncDeviceUpsert,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-24T00:00:00+00:00"


def _test_user() -> User:
    return User(id="user-1", username="user-1")


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
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_retention.db"))


@pytest.fixture()
def sync_service(sync_store: SyncV2Store, tmp_path: Path) -> SyncV2Service:
    registry = SyncAdapterRegistry(
        [
            StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}),
            StaticSyncAdapter(domain="attachment.ref", supported_adapter_versions={1}),
            StaticSyncAdapter(domain="workspaces.source_ref", supported_adapter_versions={1}),
        ]
    )
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=1024,
            max_chunk_bytes=128,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-1",
            user_id="user-1",
            display_name="Primary laptop",
            client_type="chatbook",
        )
    )
    service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-2",
            user_id="user-1",
            display_name="Phone",
            client_type="chatbook",
        )
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )
    return service


def _client_for_service(service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: service
    return TestClient(app)


def _note_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "note-env-1",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "object_revision": 1,
        "schema_version": 1,
        "payload": {"title": "Research note"},
        "payload_hash": "sha256:note-v1",
        "created_at_client": "2026-05-23T23:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _attachment_ref_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "attachment-env-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": "attachment-1",
        "device_id": "device-1",
        "client_sequence": 50,
        "schema_version": 1,
        "payload": {
            "attachment_id": "attachment-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "filename": "paper.pdf",
            "content_type": "application/pdf",
            "size_bytes": 13,
            "payload_hash": _sha256(b"paper payload"),
            "availability": "server",
        },
        "payload_hash": _sha256(b"paper payload"),
        "created_at_client": "2026-05-23T23:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


_ATTACHMENT_A = "a1111111-1111-4111-8111-111111111111"
_ATTACHMENT_B = "a2222222-2222-4222-8222-222222222222"
_ATTACHMENT_C = "a3333333-3333-4333-8333-333333333333"


def _v2_attachment_payload(
    blob_hash: str,
    *,
    attachment_id: str = _ATTACHMENT_A,
) -> dict[str, Any]:
    return {
        "attachment_id": attachment_id,
        "parent_domain": "notes.note",
        "parent_object_id": "b2222222-2222-4222-8222-222222222222",
        "file_name": "paper.bin",
        "original_file_name": "paper.bin",
        "content_type": "application/octet-stream",
        "size_bytes": 1,
        "blob_hash": blob_hash,
        "created_at": _clock(),
        "last_modified": _clock(),
        "created_by": "device-v2",
    }


def _v2_attachment_envelope(
    blob_hash: str,
    *,
    attachment_id: str = _ATTACHMENT_A,
    **overrides: Any,
) -> SyncEnvelopeCreate:
    payload = overrides.pop(
        "payload",
        _v2_attachment_payload(blob_hash, attachment_id=attachment_id),
    )
    values: dict[str, Any] = {
        "dataset_id": "dataset-v2",
        "client_envelope_id": "v2-attachment-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": attachment_id,
        "device_id": "device-v2",
        "client_sequence": 1,
        "schema_version": 2,
        "adapter_version": 2,
        "object_revision": 1,
        "payload": payload,
        "payload_hash": "sha256:" + "c" * 64,
        "created_at_client": _clock(),
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    values.update(overrides)
    return SyncEnvelopeCreate(**values)


def _v2_retention_service(tmp_path: Path) -> SyncV2Service:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "v2-retention.db")),
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="attachment.ref", supported_adapter_versions={1, 2})]
        ),
        clock=_clock,
        blob_store=LocalSyncBlobStore(tmp_path / "v2-retention-blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            server_trusted_encryption=_ready_encryption(),
            pull_token_signing_secret="retention-test-secret",
        ),
    )
    for device_id, versions in (
        ("device-v2", [2]),
        ("device-v1", [1]),
    ):
        service.register_device(
            user_id="user-1",
            device_id=device_id,
            display_name=device_id,
            client_type="chatbook",
            capabilities={
                "requested_domains": ["attachment.ref"],
                "supported_adapter_versions": {"attachment.ref": versions},
            },
        )
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-v2",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["attachment.ref"],
            metadata={"notes_attachment_v2": {"state": "ready"}},
        )
    )
    return service


def _notes_task_retention_service(
    tmp_path: Path,
) -> tuple[SyncV2Service, CharactersRAGDB, str, str]:
    owner_id = "task-retention-owner"
    dataset_id = "task-retention-dataset"
    note_id = "11111111-1111-4111-8111-111111111111"
    task_id = "22222222-2222-4222-8222-222222222222"
    product = CharactersRAGDB(tmp_path / "task-retention-product.db", client_id=owner_id)
    product.note_store.add_note("Tasks", "body", note_id=note_id)
    product.bind_local_task_graph_to_dataset(
        owner_user_id=owner_id,
        target_dataset_id=dataset_id,
    )
    product.task_store.create_task(
        owner_user_id=owner_id,
        dataset_id=dataset_id,
        note_id=note_id,
        task_id=task_id,
        text="Retain this task",
        projection_status="unlinked",
    )
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "task-retention-sync.db")),
        adapters=SyncAdapterRegistry(
            [
                StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}),
                StaticSyncAdapter(domain="notes.task", supported_adapter_versions={1}),
                StaticSyncAdapter(
                    domain="notes.task_activity",
                    supported_adapter_versions={1},
                ),
            ]
        ),
        materializers={"notes.task": NotesTaskMaterializer(product)},
        clock=_clock,
    )
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=dataset_id,
            owner_user_id=owner_id,
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note"],
        )
    )
    service.store.db.execute(
        "UPDATE sync_datasets SET domain_set_json = ? WHERE dataset_id = ?",
        ('["notes.note","notes.task","notes.task_activity"]', dataset_id),
    )
    return service, product, note_id, task_id


def _insert_task_retention_history(
    service: SyncV2Service,
    *,
    dataset_id: str,
    note_id: str,
    task_id: str,
) -> tuple[Any, Any]:
    first_hash = "sha256:" + "1" * 64
    second_hash = "sha256:" + "2" * 64
    first = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="task-retention-v1",
            domain="notes.task",
            operation="upsert",
            object_id=task_id,
            parent_id=note_id,
            device_id="server-origin",
            object_revision=1,
            entity_version=1,
            payload={"task_id": task_id, "note_id": note_id, "title": "First"},
            payload_hash=first_hash,
            created_at_client="2026-05-20T00:00:00+00:00",
            status="accepted",
            apply_status="applied",
            applied_at="2026-05-20T00:00:00+00:00",
        )
    )
    second = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id=dataset_id,
            client_envelope_id="task-retention-v2",
            domain="notes.task",
            operation="upsert",
            object_id=task_id,
            parent_id=note_id,
            device_id="server-origin",
            base_server_cursor=first.server_cursor,
            base_object_revision=1,
            base_object_hash=first_hash,
            object_revision=2,
            entity_version=2,
            payload={"task_id": task_id, "note_id": note_id, "title": "Second"},
            payload_hash=second_hash,
            created_at_client="2026-05-21T00:00:00+00:00",
            status="accepted",
            apply_status="applied",
            applied_at="2026-05-21T00:00:00+00:00",
        )
    )
    return first, second


def _namespaced_storage_key(
    service: SyncV2Service,
    *,
    dataset_id: str,
    owner_user_id: str,
    payload_hash: str,
    payload: bytes | None = None,
) -> str:
    assert service.blob_store is not None
    namespace = service.store.get_or_create_storage_namespace(
        dataset_id,
        owner_user_id=owner_user_id,
    )
    storage_key = service.blob_store.namespace_storage_key(
        namespace.storage_namespace_id,
        payload_hash,
    )
    target = service.blob_store.resolve_storage_key(storage_key)
    target.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        target.write_bytes(payload)
    return storage_key


def _store_v2_blob(
    service: SyncV2Service,
    *,
    blob_id: str,
    blob_hash: str,
) -> None:
    storage_key = _namespaced_storage_key(
        service,
        dataset_id="dataset-v2",
        owner_user_id="user-1",
        payload_hash=blob_hash,
    )
    service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id=blob_id,
            dataset_id="dataset-v2",
            owner_user_id="user-1",
            attachment_id="a1111111-1111-4111-8111-111111111111",
            payload_hash=blob_hash,
            content_type="application/octet-stream",
            size_bytes=1,
            storage_backend="local_fs",
            storage_key=storage_key,
        )
    )


def _tombstone_v2_attachment(
    service: SyncV2Service,
    *,
    blob_hash: str,
    head,
    revision: int,
    attachment_id: str = _ATTACHMENT_A,
    client_sequence: int | None = None,
):
    current = service.store.get_current_head(
        "dataset-v2",
        "attachment.ref",
        attachment_id,
    )
    assert current is not None and current.server_sequence == head.server_sequence
    payload = _v2_attachment_payload(blob_hash, attachment_id=attachment_id)
    payload["deleted_at"] = _clock()
    return service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[
            _v2_attachment_envelope(
                blob_hash,
                attachment_id=attachment_id,
                client_envelope_id=f"v2-tombstone-{attachment_id}-{revision}",
                operation="tombstone",
                client_sequence=client_sequence or revision,
                object_revision=revision,
                payload=payload,
                base_server_cursor=current.server_sequence,
                base_object_revision=current.object_revision,
                base_object_hash=current.payload_hash,
            )
        ],
    ).accepted[0]


def _ack_all_domains(service: SyncV2Service, through_sequence: int) -> None:
    for device_id in ("device-1", "device-2"):
        pulled = service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id=device_id,
            include_own_changes=True,
        )
        delivered_by_domain: dict[str, int] = {}
        for envelope in pulled.envelopes:
            if envelope.server_sequence <= through_sequence:
                delivered_by_domain[envelope.domain] = max(
                    envelope.server_sequence,
                    delivered_by_domain.get(envelope.domain, 0),
                )
        service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id=device_id,
            domain_acks=[
                SyncDeviceDomainAckCreate(
                    dataset_id="dataset-1",
                    device_id=device_id,
                    domain=domain,
                    through_server_sequence=sequence,
                    applied_at=_clock(),
                )
                for domain, sequence in sorted(delivered_by_domain.items())
            ],
        )


def _ack_v2_attachment_ref_through(
    service: SyncV2Service,
    through_sequence: int,
) -> None:
    pulled = service.pull(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        include_own_changes=True,
    )
    assert any(
        envelope.adapter_version == 2
        and envelope.server_sequence >= through_sequence
        for envelope in pulled.envelopes
    )
    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        domain_acks=[
            SyncDeviceDomainAckCreate(
                dataset_id="dataset-v2",
                device_id="device-v2",
                domain="attachment.ref",
                adapter_version=2,
                through_server_sequence=through_sequence,
                applied_at=_clock(),
            )
        ],
    )


def _ack_v2_blob(service: SyncV2Service, *, blob_id: str, blob_hash: str) -> None:
    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        blob_id_acks=[
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-v2",
                device_id="device-v2",
                blob_id=blob_id,
                payload_hash=blob_hash,
                verified_at=_clock(),
            )
        ],
    )


def _push_shared_v2_refs(
    service: SyncV2Service,
    *,
    blob_id: str,
    blob_hash: str,
    attachment_ids: tuple[str, ...] = (_ATTACHMENT_A, _ATTACHMENT_B),
):
    _store_v2_blob(service, blob_id=blob_id, blob_hash=blob_hash)
    return [
        service.push(
            user_id="user-1",
            dataset_id="dataset-v2",
            device_id="device-v2",
            envelopes=[
                _v2_attachment_envelope(
                    blob_hash,
                    attachment_id=attachment_id,
                    client_envelope_id=f"shared-{index}",
                    client_sequence=index,
                )
            ],
        ).accepted[0]
        for index, attachment_id in enumerate(attachment_ids, start=1)
    ]


def _shared_blob_candidate(service: SyncV2Service, blob_id: str):
    dry_run = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=0,
    )
    return next(
        item
        for item in dry_run.candidates
        if item.candidate_type == "blob_gc" and item.blob_id == blob_id
    )


def test_v2_shared_blob_live_sibling_binding_blocks_gc(tmp_path: Path) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    first, second = _push_shared_v2_refs(
        service,
        blob_id="blob-shared",
        blob_hash=digest,
    )
    tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=first,
        revision=2,
        attachment_id=_ATTACHMENT_A,
        client_sequence=101,
    )
    _ack_v2_attachment_ref_through(service, tombstone.server_sequence)
    _ack_v2_blob(service, blob_id="blob-shared", blob_hash=digest)

    candidate = _shared_blob_candidate(service, "blob-shared")

    assert second.entity_id == _ATTACHMENT_B
    assert "retention_active_blob_reference" in candidate.blockers


def test_v2_shared_blob_all_eligible_bindings_allow_exact_evidence(
    tmp_path: Path,
) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    first, second = _push_shared_v2_refs(
        service,
        blob_id="blob-shared",
        blob_hash=digest,
    )
    first_tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=first,
        revision=2,
        attachment_id=_ATTACHMENT_A,
        client_sequence=101,
    )
    second_tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=second,
        revision=2,
        attachment_id=_ATTACHMENT_B,
        client_sequence=102,
    )
    _ack_v2_attachment_ref_through(
        service,
        max(first_tombstone.server_sequence, second_tombstone.server_sequence),
    )
    _ack_v2_blob(service, blob_id="blob-shared", blob_hash=digest)

    candidate = _shared_blob_candidate(service, "blob-shared")

    assert candidate.blockers == []
    assert candidate.unacknowledged_device_ids == []


def test_v2_shared_blob_binding_pagination_still_finds_live_ref(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    first, second, third = _push_shared_v2_refs(
        service,
        blob_id="blob-shared",
        blob_hash=digest,
        attachment_ids=(_ATTACHMENT_A, _ATTACHMENT_B, _ATTACHMENT_C),
    )
    for client_sequence, attachment_id, head in (
        (101, _ATTACHMENT_A, first),
        (102, _ATTACHMENT_B, second),
    ):
        _tombstone_v2_attachment(
            service,
            blob_hash=digest,
            head=head,
            revision=2,
            attachment_id=attachment_id,
            client_sequence=client_sequence,
        )
    monkeypatch.setattr(service_module, "SYNC_RETENTION_BINDING_PAGE_SIZE", 1)
    _ack_v2_blob(service, blob_id="blob-shared", blob_hash=digest)

    candidate = _shared_blob_candidate(service, "blob-shared")

    assert third.entity_id == _ATTACHMENT_C
    assert "retention_active_blob_reference" in candidate.blockers


def test_v2_shared_blob_released_live_binding_stops_blocking(tmp_path: Path) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    first, second = _push_shared_v2_refs(
        service,
        blob_id="blob-shared",
        blob_hash=digest,
    )
    tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=first,
        revision=2,
        attachment_id=_ATTACHMENT_A,
        client_sequence=101,
    )
    service.store.release_attachment_revision_binding(
        "dataset-v2",
        _ATTACHMENT_B,
        1,
        released_at=_clock(),
        owner_user_id="user-1",
    )
    _ack_v2_attachment_ref_through(service, tombstone.server_sequence)
    _ack_v2_blob(service, blob_id="blob-shared", blob_hash=digest)

    candidate = _shared_blob_candidate(service, "blob-shared")

    assert second.entity_id == _ATTACHMENT_B
    assert "retention_active_blob_reference" not in candidate.blockers
    assert candidate.blockers == []


def test_retention_apply_revalidates_stale_blob_candidate_under_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    first = _push_shared_v2_refs(
        service,
        blob_id="blob-stale",
        blob_hash=digest,
        attachment_ids=(_ATTACHMENT_A,),
    )[0]
    tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=first,
        revision=2,
        attachment_id=_ATTACHMENT_A,
        client_sequence=2,
    )
    _ack_v2_attachment_ref_through(service, tombstone.server_sequence)
    _ack_v2_blob(service, blob_id="blob-stale", blob_hash=digest)
    original_dry_run = service.retention_dry_run
    injected = False

    def stale_dry_run(**kwargs):
        nonlocal injected
        result = original_dry_run(**kwargs)
        if not injected:
            injected = True
            pushed = service.push(
                user_id="user-1",
                dataset_id="dataset-v2",
                device_id="device-v2",
                envelopes=[
                    _v2_attachment_envelope(
                        digest,
                        attachment_id=_ATTACHMENT_B,
                        client_envelope_id="stale-live-binding",
                        client_sequence=3,
                    )
                ],
            )
            assert pushed.accepted
        return result

    monkeypatch.setattr(service, "retention_dry_run", stale_dry_run)

    applied = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-v2",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_binding_release=False,
        apply_blob_gc=True,
    )

    assert applied.mutation_performed is False
    assert applied.applied_count == 0
    blob = service.store.get_blob_object(
        "dataset-v2",
        blob_id="blob-stale",
        owner_user_id="user-1",
    )
    assert blob is not None and blob.status == "available"


def test_v2_blob_retention_uses_blob_id_ack_and_exact_version_devices(
    tmp_path: Path,
) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    _store_v2_blob(service, blob_id="blob-v2", blob_hash=digest)
    upsert = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[_v2_attachment_envelope(digest)],
    ).accepted[0]
    tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=upsert,
        revision=2,
    )
    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        blob_acks=[
            SyncDeviceBlobAckCreate(
                dataset_id="dataset-v2",
                device_id="device-v2",
                attachment_id="a1111111-1111-4111-8111-111111111111",
                payload_hash=digest,
                verified_at=_clock(),
            )
        ],
    )

    legacy_only = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=0,
    )
    candidate = next(item for item in legacy_only.candidates if item.blob_id == "blob-v2")
    assert candidate.required_device_ids == ["device-v2"]
    assert candidate.unacknowledged_device_ids == ["device-v2"]
    assert "retention_blob_unverified_by_device" in candidate.blockers

    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        blob_id_acks=[
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-v2",
                device_id="device-v2",
                blob_id="blob-v2",
                payload_hash=digest,
                verified_at=_clock(),
            )
        ],
    )
    blob_only = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=0,
    )
    candidate = next(item for item in blob_only.candidates if item.blob_id == "blob-v2")
    assert candidate.unacknowledged_device_ids == ["device-v2"]
    assert "retention_blob_ref_unacknowledged" in candidate.blockers
    assert "retention_blob_unverified_by_device" not in candidate.blockers

    _ack_v2_attachment_ref_through(service, tombstone.server_sequence)
    acknowledged = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=0,
    )
    candidate = next(item for item in acknowledged.candidates if item.blob_id == "blob-v2")
    assert candidate.unacknowledged_device_ids == []
    assert "retention_blob_ref_unacknowledged" not in candidate.blockers
    assert "retention_blob_unverified_by_device" not in candidate.blockers


def test_v2_blob_retention_keeps_replacement_evidence_separate(tmp_path: Path) -> None:
    service = _v2_retention_service(tmp_path)
    old_digest = "sha256:" + "a" * 64
    new_digest = "sha256:" + "b" * 64
    _store_v2_blob(service, blob_id="blob-old", blob_hash=old_digest)
    _store_v2_blob(service, blob_id="blob-new", blob_hash=new_digest)
    service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[_v2_attachment_envelope(old_digest)],
    ).accepted[0]
    first_head = service.store.get_current_head(
        "dataset-v2",
        "attachment.ref",
        "a1111111-1111-4111-8111-111111111111",
    )
    assert first_head is not None
    second = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[
            _v2_attachment_envelope(
                new_digest,
                client_envelope_id="v2-attachment-2",
                client_sequence=2,
                object_revision=2,
                payload_hash="sha256:" + "d" * 64,
                base_server_cursor=first_head.server_sequence,
                base_object_revision=first_head.object_revision,
                base_object_hash=first_head.payload_hash,
            )
        ],
    ).accepted[0]
    _tombstone_v2_attachment(
        service,
        blob_hash=new_digest,
        head=second,
        revision=3,
    )
    _ack_v2_attachment_ref_through(service, second.server_sequence)
    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        blob_id_acks=[
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-v2",
                device_id="device-v2",
                blob_id="blob-old",
                payload_hash=old_digest,
                verified_at=_clock(),
            )
        ],
    )

    dry_run = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=0,
    )
    by_blob = {
        item.blob_id: item
        for item in dry_run.candidates
        if item.candidate_type == "blob_gc"
    }
    assert by_blob["blob-old"].unacknowledged_device_ids == []
    assert by_blob["blob-new"].unacknowledged_device_ids == ["device-v2"]


def test_retention_candidate_binding_release_requires_historical_exact_evidence(
    tmp_path: Path,
) -> None:
    service = _v2_retention_service(tmp_path)
    old_digest = "sha256:" + "a" * 64
    new_digest = "sha256:" + "b" * 64
    _store_v2_blob(service, blob_id="blob-old", blob_hash=old_digest)
    _store_v2_blob(service, blob_id="blob-new", blob_hash=new_digest)
    first = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[_v2_attachment_envelope(old_digest)],
    ).accepted[0]
    first_head = service.store.get_current_head(
        "dataset-v2", "attachment.ref", _ATTACHMENT_A
    )
    assert first_head is not None
    second = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[
            _v2_attachment_envelope(
                new_digest,
                client_envelope_id="v2-attachment-replacement",
                client_sequence=2,
                object_revision=2,
                payload_hash="sha256:" + "d" * 64,
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash=first_head.payload_hash,
            )
        ],
    ).accepted[0]

    before_ack = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
    )
    releases = {
        candidate.attachment_revision: candidate
        for candidate in before_ack.candidates
        if candidate.candidate_type == "binding_release"
    }
    assert releases[1].blob_id == "blob-old"
    assert releases[1].required_device_ids == ["device-v2"]
    assert releases[1].unacknowledged_device_ids == ["device-v2"]
    assert "retention_blob_ref_unacknowledged" in releases[1].blockers
    assert "retention_blob_unverified_by_device" in releases[1].blockers
    assert 2 not in releases
    response = _client_for_service(service).post(
        "/api/v1/sync/retention/dry-run",
        json={"dataset_id": "dataset-v2", "audit_mode": False},
    )
    assert response.status_code == 200
    release_json = next(
        candidate
        for candidate in response.json()["candidates"]
        if candidate["candidate_type"] == "binding_release"
    )
    assert release_json["attachment_revision"] == 1

    _ack_v2_attachment_ref_through(service, second.server_sequence)
    _ack_v2_blob(service, blob_id="blob-old", blob_hash=old_digest)
    eligible = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
    )
    old_release = next(
        candidate
        for candidate in eligible.candidates
        if candidate.candidate_type == "binding_release"
        and candidate.attachment_revision == 1
    )
    assert old_release.blockers == []

    applied = service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-v2",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_blob_gc=False,
        apply_binding_release=True,
    )
    released = service.store.get_attachment_revision_binding(
        "dataset-v2",
        _ATTACHMENT_A,
        1,
        owner_user_id="user-1",
    )
    current = service.store.get_attachment_revision_binding(
        "dataset-v2",
        _ATTACHMENT_A,
        2,
        owner_user_id="user-1",
    )
    assert applied.binding_releases == [
        {
            "attachment_id": _ATTACHMENT_A,
            "attachment_revision": 1,
            "blob_id": "blob-old",
            "payload_hash": old_digest,
            "size_bytes": 1,
            "establishing_server_cursor": first.server_sequence,
        }
    ]
    assert released is not None and released.retention_released_at == _clock()
    assert released.blob_hash == old_digest
    assert released.resolved_blob_id == "blob-old"
    assert current is not None and current.retention_released_at is None


def test_binding_release_candidate_reports_audit_restore_and_repair_holds(
    tmp_path: Path,
) -> None:
    service = _v2_retention_service(tmp_path)
    old_digest = "sha256:" + "a" * 64
    new_digest = "sha256:" + "b" * 64
    _store_v2_blob(service, blob_id="blob-old", blob_hash=old_digest)
    _store_v2_blob(service, blob_id="blob-new", blob_hash=new_digest)
    first = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[_v2_attachment_envelope(old_digest)],
    ).accepted[0]
    first_head = service.store.get_current_head(
        "dataset-v2", "attachment.ref", _ATTACHMENT_A
    )
    assert first_head is not None
    service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[
            _v2_attachment_envelope(
                new_digest,
                client_envelope_id="v2-attachment-replacement",
                client_sequence=2,
                object_revision=2,
                payload_hash="sha256:" + "d" * 64,
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash=first_head.payload_hash,
            )
        ],
    )
    service.store.db.execute(
        "UPDATE sync_blob_objects SET status = 'quarantined' WHERE blob_id = ?",
        ("blob-old",),
    )

    dry_run = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=True,
        minimum_envelope_age_seconds=3600,
        offline_restore_window_seconds=3600,
    )
    candidate = next(
        candidate
        for candidate in dry_run.candidates
        if candidate.candidate_type == "binding_release"
        and candidate.attachment_revision == 1
    )
    assert "retention_audit_mode" in candidate.blockers
    assert "retention_envelope_window_active" in candidate.blockers
    assert "retention_restore_window_active" in candidate.blockers
    assert "retention_blob_quarantined" in candidate.blockers


def test_binding_release_tombstone_window_and_later_restore_remain_protected(
    tmp_path: Path,
) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    _store_v2_blob(service, blob_id="blob-v2", blob_hash=digest)
    first = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[_v2_attachment_envelope(digest)],
    ).accepted[0]
    tombstone = _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=first,
        revision=2,
    )

    dry_run = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=3600,
    )
    by_revision = {
        candidate.attachment_revision: candidate
        for candidate in dry_run.candidates
        if candidate.candidate_type == "binding_release"
    }
    assert "retention_tombstone_window_active" in by_revision[1].blockers
    assert "retention_tombstone_window_active" in by_revision[2].blockers
    tombstone_head = service.store.get_current_head(
        "dataset-v2", "attachment.ref", _ATTACHMENT_A
    )
    assert tombstone_head is not None

    restored = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[
            _v2_attachment_envelope(
                digest,
                client_envelope_id="v2-attachment-restore",
                client_sequence=3,
                object_revision=3,
                payload_hash="sha256:" + "e" * 64,
                base_server_cursor=tombstone.server_sequence,
                base_object_revision=tombstone.object_revision,
                base_object_hash=tombstone_head.payload_hash,
                routing_metadata={"restore_intent": True},
            )
        ],
    ).accepted[0]
    after_restore = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
    )
    release_revisions = {
        candidate.attachment_revision
        for candidate in after_restore.candidates
        if candidate.candidate_type == "binding_release"
    }
    assert restored.object_revision == 3
    assert 3 not in release_revisions


@pytest.mark.parametrize("mutation", ["missing", "mismatch"])
def test_v2_blob_retention_fails_closed_on_invalid_binding(
    tmp_path: Path,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _v2_retention_service(tmp_path)
    digest = "sha256:" + "a" * 64
    _store_v2_blob(service, blob_id="blob-v2", blob_hash=digest)
    upsert = service.push(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        envelopes=[_v2_attachment_envelope(digest)],
    ).accepted[0]
    _tombstone_v2_attachment(
        service,
        blob_hash=digest,
        head=upsert,
        revision=2,
    )
    if mutation == "missing":
        service.store.db.execute(
            "DELETE FROM sync_attachment_revision_bindings WHERE dataset_id = ?",
            ("dataset-v2",),
        )
        monkeypatch.setattr(
            service.store,
            "list_envelopes_for_entity",
            lambda *args, **kwargs: [],
        )
    else:
        service.store.db.execute(
            "UPDATE sync_attachment_revision_bindings SET blob_hash = ? "
            "WHERE dataset_id = ?",
            ("sha256:" + "f" * 64, "dataset-v2"),
        )
    service.acknowledge_device_state(
        user_id="user-1",
        dataset_id="dataset-v2",
        device_id="device-v2",
        blob_id_acks=[
            SyncDeviceBlobIdAckCreate(
                dataset_id="dataset-v2",
                device_id="device-v2",
                blob_id="blob-v2",
                payload_hash=digest,
                verified_at=_clock(),
            )
        ],
    )

    dry_run = service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-v2",
        audit_mode=False,
        minimum_tombstone_age_seconds=0,
    )
    candidate = next(item for item in dry_run.candidates if item.blob_id == "blob-v2")
    assert "retention_blob_binding_invalid" in candidate.blockers


def test_retention_dry_run_blocks_compaction_until_active_devices_ack(
    sync_service: SyncV2Service,
) -> None:
    assert hasattr(sync_service, "retention_dry_run")
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    )
    envelope_count_before = len(sync_service.store.list_envelopes_after("dataset-1", 0, limit=10))
    object_state_before = sync_service.store.get_object_state("dataset-1", "notes.note", "note-1")

    dry_run = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=False,
        minimum_envelope_age_seconds=0,
    )

    assert dry_run.mutation_performed is False
    assert dry_run.candidate_count == 1
    assert dry_run.blocked_count == 1
    assert dry_run.blocker_counts == {"retention_unacknowledged_device": 1}
    candidate = dry_run.candidates[0]
    assert candidate.candidate_type == "envelope_compaction"
    assert candidate.domain == "notes.note"
    assert candidate.object_id == "note-1"
    assert candidate.server_sequence == first.server_sequence
    assert candidate.blockers == ["retention_unacknowledged_device"]
    assert candidate.required_device_ids == ["device-1", "device-2"]
    assert len(sync_service.store.list_envelopes_after("dataset-1", 0, limit=10)) == envelope_count_before
    assert sync_service.store.get_object_state("dataset-1", "notes.note", "note-1") == object_state_before


def test_retention_limit_uses_one_bounded_blob_page_budget(
    sync_service: SyncV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(3):
        payload_hash = "sha256:" + str(index) * 64
        sync_service.store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=f"blob-budget-{index}",
                dataset_id="dataset-1",
                owner_user_id="user-1",
                attachment_id=f"attachment-budget-{index}",
                payload_hash=payload_hash,
                content_type="application/octet-stream",
                size_bytes=1,
                storage_backend="local_fs",
                storage_key=_namespaced_storage_key(
                    sync_service,
                    dataset_id="dataset-1",
                    owner_user_id="user-1",
                    payload_hash=payload_hash,
                ),
            )
        )
    original = sync_service.store.list_blob_objects_for_dataset
    page_limits: list[int] = []

    def bounded_page(dataset_id: str, *, limit: int, **_kwargs):
        page_limits.append(limit)
        return original(dataset_id)[:limit]

    def unbounded_list(*_args, **_kwargs):
        raise AssertionError("retention must not use the unbounded blob listing")

    monkeypatch.setattr(
        sync_service.store,
        "list_blob_objects_for_dataset_page",
        bounded_page,
        raising=False,
    )
    monkeypatch.setattr(
        sync_service.store,
        "list_blob_objects_for_dataset",
        unbounded_list,
    )

    result = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=False,
        limit=1,
    )

    assert len(result.candidates) == 1
    assert page_limits == [1]


def test_retention_dry_run_reports_eligible_candidate_after_all_devices_ack(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    second = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=second.server_sequence)

    dry_run = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=False,
        minimum_envelope_age_seconds=0,
    )

    assert dry_run.candidate_count == 1
    assert dry_run.blocked_count == 0
    assert dry_run.blocker_counts == {}
    assert dry_run.candidates[0].server_sequence == first.server_sequence
    assert dry_run.candidates[0].blockers == []


def test_workspace_retention_blocks_until_ack_scope_is_explicit(
    sync_service: SyncV2Service,
) -> None:
    sync_service.workspace_access_checker = lambda _user_id, workspace_id, permission: (
        workspace_id == "workspace-1" and permission == "sync"
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="workspace-dataset",
        scope_type="workspace",
        workspace_id="workspace-1",
        domains=["workspaces.source_ref"],
    )
    first = sync_service.push(
        user_id="user-1",
        dataset_id="workspace-dataset",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                dataset_id="workspace-dataset",
                domain="workspaces.source_ref",
                client_envelope_id="workspace-note-v1",
            )
        ],
    ).accepted[0]
    sync_service.push(
        user_id="user-1",
        dataset_id="workspace-dataset",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                dataset_id="workspace-dataset",
                domain="workspaces.source_ref",
                client_envelope_id="workspace-note-v2",
                client_sequence=2,
                object_revision=2,
                base_server_cursor=first.server_sequence,
                base_object_revision=1,
                base_object_hash="sha256:note-v1",
                payload_hash="sha256:note-v2",
            )
        ],
    )

    dry_run = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="workspace-dataset",
        audit_mode=False,
        minimum_envelope_age_seconds=0,
        minimum_tombstone_age_seconds=0,
        offline_restore_window_seconds=0,
    )

    assert dry_run.candidates
    assert "retention_workspace_ack_scope_unknown" in dry_run.candidates[0].blockers
    assert dry_run.blocker_counts["retention_workspace_ack_scope_unknown"] >= 1


def test_retention_dry_run_blocks_candidates_during_offline_restore_window(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    second = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=second.server_sequence)

    dry_run = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=False,
        minimum_envelope_age_seconds=0,
        offline_restore_window_seconds=86_400,
    )

    assert dry_run.candidates[0].server_sequence == first.server_sequence
    assert dry_run.candidates[0].blockers == ["retention_restore_window_active"]
    assert dry_run.blocker_counts == {"retention_restore_window_active": 1}


def test_retention_dry_run_blocks_tombstone_until_window_expires(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    tombstone = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-tombstone",
                operation="tombstone",
                client_sequence=2,
                object_revision=2,
                payload={},
                payload_hash="sha256:note-tombstone",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=tombstone.server_sequence)

    dry_run = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=False,
        minimum_tombstone_age_seconds=86_400,
    )

    assert dry_run.candidate_count == 1
    assert dry_run.candidates[0].candidate_type == "tombstone_prune"
    assert dry_run.candidates[0].server_sequence == tombstone.server_sequence
    assert dry_run.candidates[0].blockers == ["retention_tombstone_window_active"]
    assert dry_run.blocker_counts == {"retention_tombstone_window_active": 1}


def test_retention_dry_run_keeps_audit_restore_window_and_active_blob_refs_as_blockers(
    sync_service: SyncV2Service,
) -> None:
    pushed = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_attachment_ref_envelope()],
    ).accepted[0]
    payload_hash = _sha256(b"paper payload")
    sync_service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-1",
            payload_hash=payload_hash,
            content_type="application/pdf",
            size_bytes=13,
            storage_backend="local_fs",
            storage_key=_namespaced_storage_key(
                sync_service,
                dataset_id="dataset-1",
                owner_user_id="user-1",
                payload_hash=payload_hash,
                payload=b"paper payload",
            ),
        )
    )
    _ack_all_domains(sync_service, through_sequence=pushed.server_sequence)
    for device_id in ("device-1", "device-2"):
        sync_service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id=device_id,
            blob_acks=[
                SyncDeviceBlobAckCreate(
                    dataset_id="dataset-1",
                    device_id=device_id,
                    attachment_id="attachment-1",
                    payload_hash=payload_hash,
                    verified_at=_clock(),
                )
            ],
        )

    dry_run = sync_service.retention_dry_run(
        user_id="user-1",
        dataset_id="dataset-1",
        audit_mode=True,
        minimum_envelope_age_seconds=0,
        offline_restore_window_seconds=86_400,
    )

    blob_candidates = [
        candidate for candidate in dry_run.candidates if candidate.candidate_type == "blob_gc"
    ]
    assert len(blob_candidates) == 1
    assert blob_candidates[0].attachment_id == "attachment-1"
    assert blob_candidates[0].payload_hash == payload_hash
    assert blob_candidates[0].blockers == [
        "retention_audit_mode",
        "retention_restore_window_active",
        "retention_active_blob_reference",
    ]
    assert dry_run.blocker_counts["retention_audit_mode"] >= 1
    assert dry_run.blocker_counts["retention_restore_window_active"] >= 1
    assert dry_run.blocker_counts["retention_active_blob_reference"] == 1


def test_retention_dry_run_endpoint_returns_redacted_candidates(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    second = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=second.server_sequence)
    client = _client_for_service(sync_service)

    response = client.post(
        "/api/v1/sync/retention/dry-run",
        json={
            "dataset_id": "dataset-1",
            "audit_mode": False,
            "minimum_envelope_age_seconds": 0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "dataset-1"
    assert body["mutation_performed"] is False
    assert body["candidate_count"] == 1
    assert body["candidates"][0]["candidate_type"] == "envelope_compaction"
    assert body["candidates"][0]["server_sequence"] == first.server_sequence
    assert "payload" not in body["candidates"][0]
    assert "payload_ciphertext" not in body["candidates"][0]


def test_retention_compact_requires_confirmation_without_mutation(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    second = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=second.server_sequence)
    envelope_count_before = len(sync_service.store.list_envelopes_after("dataset-1", 0, limit=10))

    result = sync_service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=False,
        minimum_envelope_age_seconds=0,
    )

    assert result.dry_run is True
    assert result.mutation_performed is False
    assert result.confirmation_required is True
    assert result.blockers == ["retention_confirmation_required"]
    assert result.candidate_count == 1
    assert sync_service.store.get_domain_compaction_sequence("dataset-1", "notes.note") == 0
    assert len(sync_service.store.list_envelopes_after("dataset-1", 0, limit=10)) == envelope_count_before


def test_retention_compact_refuses_blocked_candidates_without_mutation(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    )

    result = sync_service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        minimum_envelope_age_seconds=0,
    )

    assert result.mutation_performed is False
    assert result.blockers == ["retention_blocked_candidates_present"]
    assert result.blocked_count == 1
    assert result.applied_count == 0
    assert sync_service.store.get_domain_compaction_sequence("dataset-1", "notes.note") == 0


def test_retention_compact_records_domain_checkpoint_without_deleting_envelopes(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    second = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=second.server_sequence)

    result = sync_service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        minimum_envelope_age_seconds=0,
        apply_blob_gc=False,
    )

    assert result.dry_run is False
    assert result.mutation_performed is True
    assert result.applied_count == 1
    assert result.domain_compactions == [
        {
            "domain": "notes.note",
            "through_server_sequence": first.server_sequence,
            "candidate_count": 1,
        }
    ]
    assert sync_service.store.get_domain_compaction_sequence("dataset-1", "notes.note") == first.server_sequence
    assert len(sync_service.store.list_envelopes_after("dataset-1", 0, limit=10)) == 2


def test_retention_compact_soft_deletes_eligible_blob_metadata(
    sync_service: SyncV2Service,
) -> None:
    upsert = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_attachment_ref_envelope(object_revision=1)],
    ).accepted[0]
    head = sync_service.store.get_current_head(
        "dataset-1",
        "attachment.ref",
        "attachment-1",
    )
    assert head is not None and head.server_sequence == upsert.server_sequence
    assert head.object_revision is not None
    tombstone = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _attachment_ref_envelope(
                client_envelope_id="attachment-env-tombstone",
                operation="tombstone",
                client_sequence=51,
                base_server_cursor=head.server_sequence,
                base_object_revision=head.object_revision,
                base_object_hash=head.payload_hash,
                object_revision=2,
            )
        ],
    ).accepted[0]
    payload_hash = _sha256(b"paper payload")
    sync_service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-1",
            payload_hash=payload_hash,
            content_type="application/pdf",
            size_bytes=13,
            storage_backend="local_fs",
            storage_key=_namespaced_storage_key(
                sync_service,
                dataset_id="dataset-1",
                owner_user_id="user-1",
                payload_hash=payload_hash,
                payload=b"paper payload",
            ),
        )
    )
    _ack_all_domains(sync_service, through_sequence=tombstone.server_sequence)
    for device_id in ("device-1", "device-2"):
        sync_service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id=device_id,
            blob_acks=[
                SyncDeviceBlobAckCreate(
                    dataset_id="dataset-1",
                    device_id=device_id,
                    attachment_id="attachment-1",
                    payload_hash=payload_hash,
                    verified_at=_clock(),
                )
            ],
        )

    result = sync_service.retention_compact(
        user_id="user-1",
        dataset_id="dataset-1",
        confirm=True,
        apply_envelope_compaction=False,
        apply_tombstone_prune=False,
        apply_blob_gc=True,
    )

    assert result.mutation_performed is True
    assert result.applied_count == 1
    assert result.blob_gc == [
        {
            "attachment_id": "attachment-1",
            "blob_id": "blob-1",
            "payload_hash": payload_hash,
            "size_bytes": 13,
        }
    ]
    assert sync_service.store.list_blob_objects_for_dataset("dataset-1") == []
    deleted_blobs = sync_service.store.list_blob_objects_for_dataset("dataset-1", status=None)
    assert len(deleted_blobs) == 1
    assert deleted_blobs[0].status == "deleted"


def test_retention_compact_endpoint_returns_redacted_apply_summary(
    sync_service: SyncV2Service,
) -> None:
    first = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    second = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated note"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    ).accepted[0]
    _ack_all_domains(sync_service, through_sequence=second.server_sequence)
    client = _client_for_service(sync_service)

    response = client.post(
        "/api/v1/sync/retention/compact",
        json={
            "dataset_id": "dataset-1",
            "confirm": True,
            "minimum_envelope_age_seconds": 0,
            "apply_blob_gc": False,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "dataset-1"
    assert body["mutation_performed"] is True
    assert body["applied_count"] == 1
    assert body["domain_compactions"][0]["domain"] == "notes.note"
    assert "payload" not in body
    assert "payload_ciphertext" not in body


def test_notes_task_open_drift_blocks_exact_envelope_until_resolved(
    tmp_path: Path,
) -> None:
    service, product, note_id, task_id = _notes_task_retention_service(tmp_path)
    try:
        first, _second = _insert_task_retention_history(
            service,
            dataset_id="task-retention-dataset",
            note_id=note_id,
            task_id=task_id,
        )
        assert first.server_cursor is not None
        drift = product.task_store.create_task_projection_drift(
            owner_user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            drift_id="task-retention-drift",
            note_id=note_id,
            task_id=task_id,
            marker_base_revision=1,
            marker_base_hash=str(first.payload_hash),
            note_head_cursor=None,
            note_head_hash=None,
            task_head_cursor=first.server_cursor,
            task_head_hash=str(first.payload_hash),
            reason_code="both_changed",
        )

        blocked = service.retention_dry_run(
            user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            domains=["notes.task"],
            audit_mode=False,
        )
        candidate = next(
            item for item in blocked.candidates if item.server_sequence == first.server_sequence
        )
        assert "retention_task_projection_drift" in candidate.blockers

        product.task_store.compare_and_set_task_projection_drift(
            owner_user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            note_id=note_id,
            task_id=task_id,
            drift_id=str(drift["id"]),
            expected_note_head_cursor=None,
            expected_note_head_hash=None,
            expected_task_head_cursor=first.server_cursor,
            expected_task_head_hash=str(first.payload_hash),
            status="resolved",
        )
        released = service.retention_dry_run(
            user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            domains=["notes.task"],
            audit_mode=False,
        )
        candidate = next(
            item for item in released.candidates if item.server_sequence == first.server_sequence
        )
        assert "retention_task_projection_drift" not in candidate.blockers
    finally:
        product.close_connection()


def test_notes_task_open_drift_blocks_standalone_note_envelope(
    tmp_path: Path,
) -> None:
    service, product, note_id, task_id = _notes_task_retention_service(tmp_path)
    try:
        first_hash = "sha256:" + "5" * 64
        first = service.store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id="task-retention-dataset",
                client_envelope_id="note-retention-v1",
                domain="notes.note",
                operation="upsert",
                object_id=note_id,
                device_id="server-origin",
                object_revision=1,
                payload={"title": "Tasks", "content": "First"},
                payload_hash=first_hash,
                created_at_client="2026-05-20T00:00:00+00:00",
                status="accepted",
                apply_status="applied",
                applied_at="2026-05-20T00:00:00+00:00",
            )
        )
        second_hash = "sha256:" + "6" * 64
        service.store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id="task-retention-dataset",
                client_envelope_id="note-retention-v2",
                domain="notes.note",
                operation="upsert",
                object_id=note_id,
                device_id="server-origin",
                base_server_cursor=first.server_cursor,
                base_object_revision=1,
                base_object_hash=first_hash,
                object_revision=2,
                payload={"title": "Tasks", "content": "Second"},
                payload_hash=second_hash,
                created_at_client="2026-05-21T00:00:00+00:00",
                status="accepted",
                apply_status="applied",
                applied_at="2026-05-21T00:00:00+00:00",
            )
        )
        assert first.server_cursor is not None
        drift = product.task_store.create_task_projection_drift(
            owner_user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            drift_id="note-retention-drift",
            note_id=note_id,
            task_id=task_id,
            marker_base_revision=1,
            marker_base_hash="sha256:" + "1" * 64,
            note_head_cursor=first.server_cursor,
            note_head_hash=first_hash,
            task_head_cursor=None,
            task_head_hash=None,
            reason_code="both_changed",
        )

        blocked = service.retention_dry_run(
            user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            domains=["notes.note"],
            audit_mode=False,
        )
        candidate = next(
            item for item in blocked.candidates if item.server_sequence == first.server_sequence
        )
        assert "retention_task_projection_drift" in candidate.blockers

        product.task_store.compare_and_set_task_projection_drift(
            owner_user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            note_id=note_id,
            task_id=task_id,
            drift_id=str(drift["id"]),
            expected_note_head_cursor=first.server_cursor,
            expected_note_head_hash=first_hash,
            expected_task_head_cursor=None,
            expected_task_head_hash=None,
            status="resolved",
        )
        released = service.retention_dry_run(
            user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            domains=["notes.note"],
            audit_mode=False,
        )
        candidate = next(
            item for item in released.candidates if item.server_sequence == first.server_sequence
        )
        assert "retention_task_projection_drift" not in candidate.blockers
    finally:
        product.close_connection()


def test_notes_task_linked_tombstone_retains_immutable_anchor_without_cache(
    tmp_path: Path,
) -> None:
    service, product, note_id, task_id = _notes_task_retention_service(tmp_path)
    try:
        _first, second = _insert_task_retention_history(
            service,
            dataset_id="task-retention-dataset",
            note_id=note_id,
            task_id=task_id,
        )
        assert second.server_cursor is not None
        tombstone_hash = "sha256:" + "3" * 64
        marker = TaskMarker(task_id=task_id, revision=3, object_hash=tombstone_hash)
        tombstone = service.store.insert_envelope(
            SyncEnvelopeCreate(
                dataset_id="task-retention-dataset",
                client_envelope_id="task-retention-v3",
                domain="notes.task",
                operation="tombstone",
                object_id=task_id,
                parent_id=note_id,
                device_id="server-origin",
                base_server_cursor=second.server_cursor,
                base_object_revision=2,
                base_object_hash=second.payload_hash,
                object_revision=3,
                entity_version=3,
                payload={"task_id": task_id, "note_id": note_id, "title": "Second"},
                payload_hash=tombstone_hash,
                created_at_client="2026-05-22T00:00:00+00:00",
                routing_metadata={
                    "task_projection": {
                        "projection_version": 1,
                        "task_id": task_id,
                        "task_envelope_id": "task-retention-v3",
                        "task_revision": 3,
                        "task_hash": tombstone_hash,
                        "note_envelope_id": "task-retention-note-v3",
                        "note_hash": "sha256:" + "4" * 64,
                        "linked": True,
                        "marker_hash": task_marker_hash(marker),
                    }
                },
                status="accepted",
                apply_status="applied",
                applied_at="2026-05-22T00:00:00+00:00",
                deleted=True,
            )
        )
        with product.transaction() as conn:
            product.task_store._execute(
                conn,
                "DELETE FROM task_note_projections WHERE owner_user_id = ? AND dataset_id = ?",
                ("task-retention-owner", "task-retention-dataset"),
            )

        dry_run = service.retention_dry_run(
            user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            domains=["notes.task"],
            audit_mode=False,
        )
        candidate = next(
            item
            for item in dry_run.candidates
            if item.server_sequence == tombstone.server_cursor
        )
        assert "retention_task_projection_anchor" in candidate.blockers
    finally:
        product.close_connection()


def test_notes_task_retention_revalidates_new_drift_under_dataset_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, product, note_id, task_id = _notes_task_retention_service(tmp_path)
    try:
        first, _second = _insert_task_retention_history(
            service,
            dataset_id="task-retention-dataset",
            note_id=note_id,
            task_id=task_id,
        )
        assert first.server_cursor is not None
        original = service.retention_dry_run
        inserted = False

        def stale_dry_run(**kwargs: Any):
            nonlocal inserted
            result = original(**kwargs)
            if not inserted:
                product.task_store.create_task_projection_drift(
                    owner_user_id="task-retention-owner",
                    dataset_id="task-retention-dataset",
                    drift_id="task-retention-stale-drift",
                    note_id=note_id,
                    task_id=task_id,
                    marker_base_revision=1,
                    marker_base_hash=str(first.payload_hash),
                    note_head_cursor=None,
                    note_head_hash=None,
                    task_head_cursor=first.server_cursor,
                    task_head_hash=str(first.payload_hash),
                    reason_code="both_changed",
                )
                inserted = True
            return result

        monkeypatch.setattr(service, "retention_dry_run", stale_dry_run)
        result = service.retention_compact(
            user_id="task-retention-owner",
            dataset_id="task-retention-dataset",
            domains=["notes.task"],
            confirm=True,
            apply_binding_release=False,
            apply_blob_gc=False,
        )

        assert result.applied_count == 0
        assert result.blocker_counts == {"retention_task_projection_drift": 1}
        assert service.store.get_domain_compaction_sequence(
            "task-retention-dataset", "notes.task"
        ) == 0
    finally:
        product.close_connection()
