from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncBlobObjectCreate,
    SyncDeviceBlobAckCreate,
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


def _ack_all_domains(service: SyncV2Service, through_sequence: int) -> None:
    for device_id in ("device-1", "device-2"):
        service.acknowledge_device_state(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id=device_id,
            domain_acks=[
                SyncDeviceDomainAckCreate(
                    dataset_id="dataset-1",
                    device_id=device_id,
                    domain="notes.note",
                    through_server_sequence=through_sequence,
                    applied_at=_clock(),
                ),
                SyncDeviceDomainAckCreate(
                    dataset_id="dataset-1",
                    device_id=device_id,
                    domain="attachment.ref",
                    through_server_sequence=through_sequence,
                    applied_at=_clock(),
                ),
            ],
        )


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
            storage_key="blob-1.bin",
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
