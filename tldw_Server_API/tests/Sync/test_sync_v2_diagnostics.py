from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncBlobObjectCreate,
    SyncConflictCreate,
    SyncDeviceUpsert,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-24T02:00:00+00:00"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _test_user() -> User:
    return User(id="user-1", username="user-1")


def _other_user() -> User:
    return User(id="user-2", username="user-2")


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _build_service(tmp_path: Path) -> SyncV2Service:
    registry = SyncAdapterRegistry(
        [
            StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1}),
            StaticSyncAdapter(domain="attachment.ref", supported_adapter_versions={1}),
        ]
    )
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_diagnostics.db")),
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs"),
        settings=SyncV2Settings(
            supports_attachments=True,
            max_blob_bytes=1024,
            max_chunk_bytes=128,
            user_blob_quota_bytes=2048,
            server_trusted_encryption=_ready_encryption(),
        ),
    )
    for device_id in ("device-1", "device-2"):
        service.store.upsert_device(
            SyncDeviceUpsert(
                device_id=device_id,
                user_id="user-1",
                display_name=f"Device {device_id[-1]}",
                client_type="chatbook",
            )
        )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )
    return service


def _client_for_service(service: SyncV2Service, user_factory=_test_user) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = user_factory
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
        "payload": {"title": "Research note secret"},
        "payload_hash": "sha256:note-v1",
        "created_at_client": "2026-05-24T01:00:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _attachment_ref_envelope(**overrides: Any) -> SyncEnvelopeCreate:
    payload_hash = _sha256(b"diagnostic paper payload")
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "attachment-env-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": "attachment-1",
        "device_id": "device-1",
        "client_sequence": 10,
        "schema_version": 1,
        "payload": {
            "attachment_id": "attachment-1",
            "parent_domain": "notes.note",
            "parent_object_id": "note-1",
            "filename": "diagnostic-secret.pdf",
            "content_type": "application/pdf",
            "size_bytes": 24,
            "payload_hash": payload_hash,
            "availability": "server",
        },
        "payload_hash": payload_hash,
        "created_at_client": "2026-05-24T01:05:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "adapter_version": 1,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _seed_diagnostics_state(service: SyncV2Service) -> None:
    first = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_note_envelope()],
    ).accepted[0]
    service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _note_envelope(
                client_envelope_id="note-env-2",
                client_sequence=2,
                object_revision=2,
                payload={"title": "Updated diagnostic secret"},
                payload_hash="sha256:note-v2",
                base_server_cursor=first.server_sequence,
                base_object_revision=first.object_revision,
                base_object_hash="sha256:note-v1",
            )
        ],
    )
    service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_attachment_ref_envelope()],
    )
    service.store.mark_envelope_apply_status(
        first.server_sequence,
        apply_status="failed",
        apply_error_code="projection_failed",
        apply_error_message="payload text must stay out of diagnostics",
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-1",
            dataset_id="dataset-1",
            domain="notes.note",
            object_id="note-1",
            conflict_type="revision_mismatch",
            metadata={"private_payload": "conflict-secret"},
        )
    )
    attachment_hash = _sha256(b"diagnostic paper payload")
    service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-1",
            payload_hash=attachment_hash,
            content_type="application/pdf",
            size_bytes=24,
            storage_backend="local_fs",
            storage_key="secret-storage-key",
        )
    )
    service.create_blob_upload_session(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        domain="attachment.ref",
        entity_id="attachment-2",
        attachment_id="attachment-2",
        content_type="application/pdf",
        size_bytes=32,
        payload_hash=_sha256(b"pending diagnostic blob"),
        chunk_size=16,
        chunk_count=2,
        idempotency_key="pending-upload",
    )
    service.store.store_key_record(
        SyncKeyRecordCreate(
            key_record_id="key-1",
            dataset_id="dataset-1",
            user_id="user-1",
            key_purpose="dataset_recovery",
            wrapped_key_blob="super-secret-wrapped-key",
            kdf_metadata={"algorithm": "argon2id", "salt": "super-secret-salt"},
            recovery_hint="private recovery hint",
        )
    )


def test_diagnostics_endpoint_returns_redacted_dataset_health(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    _seed_diagnostics_state(service)
    client = _client_for_service(service)

    response = client.get(
        "/api/v1/sync/diagnostics",
        params={"dataset_id": "dataset-1", "device_id": "device-1", "retention_limit": 20},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "dataset-1"
    assert body["generated_at"] == _clock()
    domains = {domain["domain"]: domain for domain in body["domains"]}
    assert domains["notes.note"]["envelope_count"] == 2
    assert domains["notes.note"]["object_count"] == 1
    assert domains["notes.note"]["failed_apply_count"] == 1
    assert domains["notes.note"]["unresolved_conflict_count"] == 1
    assert body["blob_health"]["blob_object_count"] == 1
    assert body["blob_health"]["available_blob_bytes"] == 24
    assert body["blob_health"]["active_upload_count"] == 1
    assert body["blob_health"]["reserved_blob_bytes"] == 32
    assert body["key_summary"] == {
        "key_record_count": 1,
        "active_key_record_count": 1,
        "revoked_key_record_count": 0,
        "superseded_key_record_count": 0,
        "rewrap_pending_count": 0,
        "recovery_available": True,
    }
    devices = {device["device_id"]: device for device in body["devices"]}
    device_2_lag = {
        lag["domain"]: lag for lag in devices["device-2"]["domain_lag"]
    }
    assert device_2_lag["notes.note"]["lag_count"] == 2
    assert device_2_lag["attachment.ref"]["lag_count"] == 1
    assert body["retention"]["dry_run"] is True
    assert body["retention"]["candidate_count"] >= 1
    assert body["retention"]["mutation_performed"] is False

    encoded = str(body)
    for secret in (
        "Research note secret",
        "Updated diagnostic secret",
        "diagnostic-secret.pdf",
        "payload text must stay out",
        "conflict-secret",
        "secret-storage-key",
        "super-secret-wrapped-key",
        "super-secret-salt",
        "private recovery hint",
    ):
        assert secret not in encoded


def test_diagnostics_endpoint_requires_dataset_access(tmp_path: Path) -> None:
    service = _build_service(tmp_path)
    client = _client_for_service(service, user_factory=_other_user)

    response = client.get("/api/v1/sync/diagnostics", params={"dataset_id": "dataset-1"})

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "sync_resource_not_found"
