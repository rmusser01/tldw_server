from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import SyncV2Envelope
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.factory import default_sync_v2_registry
from tldw_Server_API.app.core.Sync.v2.materializers import AttachmentRefMaterializer
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS, SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _test_user() -> User:
    return User(id="user-1", username="user-1")


@pytest.fixture()
def sync_service(tmp_path: Path) -> SyncV2Service:
    default_sync_v2_registry.cache_clear()
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_attachment_refs.db")),
        adapters=default_sync_v2_registry(),
        materializers={"attachment.ref": AttachmentRefMaterializer()},
        clock=lambda: "2026-05-23T18:12:00+00:00",
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )
    for device_id in ("device-1", "device-2"):
        service.register_device(
            user_id="user-1",
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
        )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=list(M1_SYNC_DOMAINS),
    )
    return service


@pytest.fixture()
def client(sync_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: sync_service
    return TestClient(app)


def _attachment_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attachment_id": "att-1",
        "parent_domain": "notes.note",
        "parent_object_id": "note-1",
        "content_type": "image/png",
        "size_bytes": 512,
        "payload_hash": "sha256:blob-v1",
        "availability": "client_local",
    }
    payload.update(overrides)
    return payload


def _attachment_ref(**overrides: Any) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-attachment-1",
        "domain": "attachment.ref",
        "operation": "upsert",
        "object_id": "att-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "schema_version": 1,
        "payload": _attachment_payload(),
        "payload_hash": "sha256:blob-v1",
        "payload_size_bytes": 128,
        "created_at_client": "2026-05-23T18:12:44+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
        "stable_key": "attachment:att-1",
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _push_one(service: SyncV2Service, envelope: SyncEnvelopeCreate):
    return service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[envelope],
    )


@pytest.mark.parametrize(
    "missing_key",
    [
        "attachment_id",
        "parent_domain",
        "parent_object_id",
        "content_type",
        "size_bytes",
        "payload_hash",
        "availability",
    ],
)
def test_attachment_ref_schema_requires_metadata_fields(missing_key: str) -> None:
    payload = _attachment_payload()
    payload.pop(missing_key)

    with pytest.raises(ValidationError, match="attachment.ref envelopes require payload metadata fields"):
        SyncV2Envelope(
            dataset_id="dataset-1",
            client_envelope_id=f"env-missing-{missing_key}",
            domain="attachment.ref",
            operation="upsert",
            object_id="att-1",
            payload=payload,
            payload_hash="sha256:blob-v1",
            encryption_metadata={"policy": "server_trusted_v1"},
        )


def test_attachment_ref_schema_rejects_mismatched_object_id_and_attachment_id() -> None:
    with pytest.raises(ValidationError, match="attachment.ref object_id must match payload attachment_id"):
        SyncV2Envelope(
            dataset_id="dataset-1",
            client_envelope_id="env-mismatched-object",
            domain="attachment.ref",
            operation="upsert",
            object_id="att-alias",
            payload=_attachment_payload(attachment_id="att-1"),
            payload_hash="sha256:blob-v1",
            encryption_metadata={"policy": "server_trusted_v1"},
        )


def test_attachment_ref_is_accepted_and_visible_through_pull(
    sync_service: SyncV2Service,
) -> None:
    result = _push_one(sync_service, _attachment_ref())

    assert [item.client_envelope_id for item in result.accepted] == ["env-attachment-1"]
    assert result.rejected == []
    assert result.conflicts == []
    state = sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-1")
    assert state is not None
    assert state.object_hash == "sha256:blob-v1"
    assert state.deleted is False

    pulled = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        domains=["attachment.ref"],
    )

    assert [item.client_envelope_id for item in pulled.envelopes] == ["env-attachment-1"]
    assert pulled.envelopes[0].payload["attachment_id"] == "att-1"
    assert pulled.envelopes[0].payload["parent_domain"] == "notes.note"


def test_duplicate_attachment_ref_same_payload_is_idempotent(
    sync_service: SyncV2Service,
) -> None:
    first = _push_one(sync_service, _attachment_ref())
    duplicate = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-duplicate",
            client_sequence=2,
        ),
    )

    assert [item.client_envelope_id for item in first.accepted] == ["env-attachment-1"]
    assert [item.client_envelope_id for item in duplicate.accepted] == ["env-attachment-duplicate"]
    assert duplicate.rejected == []
    assert duplicate.conflicts == []
    assert sync_service.store.list_conflicts("dataset-1") == []
    stored = sync_service.store.list_envelopes_for_entity(
        "dataset-1",
        "attachment.ref",
        entity_id="att-1",
        limit=10,
    )
    assert sorted((item.client_envelope_id, item.apply_status) for item in stored) == [
        ("env-attachment-1", "applied"),
        ("env-attachment-duplicate", "applied"),
    ]


def test_duplicate_attachment_ref_different_payload_hash_conflicts_without_overwrite(
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())

    divergent = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-divergent",
            client_sequence=2,
            payload=_attachment_payload(
                content_type="image/jpeg",
                payload_hash="sha256:blob-v2",
            ),
            payload_hash="sha256:blob-v2",
        ),
    )

    assert divergent.accepted == []
    assert [item.client_envelope_id for item in divergent.conflicts] == ["env-attachment-divergent"]
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert len(conflicts) == 1
    assert conflicts[0].domain == "attachment.ref"
    assert conflicts[0].conflict_type == "attachment_ref_hash_mismatch"

    history = sync_service.store.list_envelopes_after(
        "dataset-1",
        0,
        domains=["attachment.ref"],
        status=None,
    )
    assert [(item.client_envelope_id, item.status, item.payload_hash) for item in history] == [
        ("env-attachment-1", "accepted", "sha256:blob-v1"),
        ("env-attachment-divergent", "conflict", "sha256:blob-v2"),
    ]


def test_attachment_ref_mismatched_object_id_cannot_bypass_hash_guard(
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())

    bypass_attempt = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-alias-divergent",
            client_sequence=2,
            object_id="att-alias",
            stable_key="attachment:att-alias",
            payload=_attachment_payload(
                attachment_id="att-1",
                content_type="image/jpeg",
                payload_hash="sha256:blob-v2",
            ),
            payload_hash="sha256:blob-v2",
        ),
    )

    assert bypass_attempt.accepted == []
    assert bypass_attempt.conflicts == []
    assert [(item.client_envelope_id, item.error_code) for item in bypass_attempt.rejected] == [
        ("env-attachment-alias-divergent", "attachment_ref_object_id_mismatch")
    ]
    assert sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-alias") is None
    history = sync_service.store.list_envelopes_after(
        "dataset-1",
        0,
        domains=["attachment.ref"],
        status=None,
    )
    assert [(item.client_envelope_id, item.status, item.object_id) for item in history] == [
        ("env-attachment-1", "accepted", "att-1"),
    ]


def test_stale_upsert_after_tombstone_cannot_resurrect_attachment_ref(
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())
    tombstone = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-tombstone",
            client_sequence=2,
            operation="tombstone",
        ),
    )

    deleted_state = sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-1")
    assert [item.client_envelope_id for item in tombstone.accepted] == ["env-attachment-tombstone"]
    assert deleted_state is not None
    assert deleted_state.deleted is True

    stale_upsert = _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-stale-upsert",
            client_sequence=3,
        ),
    )

    current_state = sync_service.store.get_object_state("dataset-1", "attachment.ref", "att-1")
    assert stale_upsert.accepted == []
    assert stale_upsert.rejected == []
    assert [item.client_envelope_id for item in stale_upsert.conflicts] == ["env-attachment-stale-upsert"]
    assert current_state is not None
    assert current_state.deleted is True
    assert current_state.latest_server_cursor == deleted_state.latest_server_cursor
    conflicts = sync_service.store.list_conflicts("dataset-1")
    assert [(item.local_envelope_id, item.conflict_type) for item in conflicts] == [
        ("env-attachment-stale-upsert", "attachment_ref_tombstoned")
    ]


def test_restore_preview_reports_attachment_refs_and_missing_blobs(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())
    _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-server",
            object_id="att-server",
            stable_key="attachment:att-server",
            client_sequence=2,
            payload=_attachment_payload(
                attachment_id="att-server",
                payload_hash="sha256:server-blob",
                availability="server",
            ),
            payload_hash="sha256:server-blob",
        ),
    )

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"], "local_inventory": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert {item["attachment_id"] for item in body["attachment_refs"]} == {
        "att-1",
        "att-server",
    }
    assert body["attachment_refs"][0]["parent_domain"] == "notes.note"
    assert [item["attachment_id"] for item in body["missing_blobs"]] == ["att-1"]
    assert body["warnings"] == [
        {
            "code": "sync_key_recovery_missing",
            "message": "No active Sync v2 key recovery bundle is available for this dataset.",
            "dataset_id": "dataset-1",
            "attachment_id": None,
            "object_id": None,
            "payload_hash": None,
        },
        {
            "code": "sync_attachment_blob_missing",
            "message": "Attachment blob is not available from the Sync v2 M1 server.",
            "dataset_id": "dataset-1",
            "attachment_id": "att-1",
            "object_id": "att-1",
            "payload_hash": "sha256:blob-v1",
        }
    ]


def test_restore_preview_omits_tombstoned_attachment_refs(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    _push_one(sync_service, _attachment_ref())
    _push_one(
        sync_service,
        _attachment_ref(
            client_envelope_id="env-attachment-tombstone",
            client_sequence=2,
            operation="tombstone",
        ),
    )

    response = client.post(
        "/api/v1/sync/restore/preview",
        json={"dataset_ids": ["dataset-1"], "local_inventory": []},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["attachment_refs"] == []
    assert body["missing_blobs"] == []
    assert body["warnings"] == [
        {
            "code": "sync_key_recovery_missing",
            "message": "No active Sync v2 key recovery bundle is available for this dataset.",
            "dataset_id": "dataset-1",
            "attachment_id": None,
            "object_id": None,
            "payload_hash": None,
        }
    ]


def test_blob_upload_and_download_are_explicitly_unsupported_in_m1(
    client: TestClient,
) -> None:
    upload = client.post(
        "/api/v1/sync/attachments",
        json={
            "dataset_id": "dataset-1",
            "domain": "attachment.ref",
            "object_id": "att-1",
            "attachment_id": "att-1",
            "content_type": "image/png",
            "size_bytes": 512,
            "payload_ciphertext": "ciphertext",
            "payload_hash": "sha256:blob-v1",
            "encryption_policy": "server_trusted_v1",
        },
    )
    download = client.get(
        "/api/v1/sync/attachments/att-1",
        params={"dataset_id": "dataset-1"},
    )

    assert upload.status_code == 501
    assert upload.json()["detail"]["error_code"] == "sync_blob_transfer_not_supported"
    assert download.status_code == 501
    assert download.json()["detail"]["error_code"] == "sync_blob_transfer_not_supported"


def test_invalid_blob_upload_is_unsupported_before_m2_schema_validation(
    client: TestClient,
) -> None:
    upload = client.post(
        "/api/v1/sync/attachments",
        content=b"\x00not-json-attachment-bytes",
        headers={"content-type": "application/octet-stream"},
    )

    assert upload.status_code == 501
    assert upload.json()["detail"]["error_code"] == "sync_blob_transfer_not_supported"
