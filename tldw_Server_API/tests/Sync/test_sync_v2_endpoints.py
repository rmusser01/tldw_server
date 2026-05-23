from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.materializers import MaterializationResult
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    SyncDeviceUpsert,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-23T18:12:00+00:00"


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _test_user() -> User:
    return User(id="user-1", username="user-1")


def _ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode="managed_storage",
        server_trusted_enabled=True,
        auth_mode="multi_user",
    )


def _not_ready_encryption():
    return server_trusted_encryption_status_from_config(
        mode=None,
        server_trusted_enabled=False,
        auth_mode="multi_user",
    )


def _registry() -> SyncAdapterRegistry:
    return SyncAdapterRegistry(
        [StaticSyncAdapter(domain=domain, supported_adapter_versions={1}) for domain in M1_SYNC_DOMAINS]
    )


class _EndpointOutcomeMaterializer:
    domain = "notes.note"

    def apply(self, envelope, *, store: SyncV2Store) -> MaterializationResult:
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
        store.upsert_object_state(
            SyncObjectState(
                dataset_id=envelope.dataset_id,
                domain=envelope.domain,
                object_id=envelope.object_id,
                object_revision=envelope.object_revision or 1,
                object_hash=envelope.payload_hash or "",
                latest_server_cursor=envelope.server_cursor,
                deleted=False,
            )
        )
        store.mark_envelope_apply_status(envelope.server_cursor, apply_status="applied")
        return MaterializationResult(status="applied")


def _build_service(
    tmp_path: Path,
    *,
    encryption=None,
    materializers=None,
    supports_attachments: bool = False,
) -> SyncV2Service:
    return SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_endpoints.db")),
        adapters=_registry(),
        materializers=materializers,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        blob_store=LocalSyncBlobStore(tmp_path / "sync_blobs") if supports_attachments else None,
        settings=SyncV2Settings(
            supports_attachments=supports_attachments,
            max_attachment_bytes=64,
            max_blob_bytes=128,
            max_chunk_bytes=8,
            user_blob_quota_bytes=256,
            server_trusted_encryption=encryption or _ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )


def _client_for_service(service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
    if hasattr(sync_endpoint, "get_sync_v2_profile_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: service
    return TestClient(app)


@pytest.fixture()
def sync_service(tmp_path: Path) -> SyncV2Service:
    return _build_service(tmp_path)


@pytest.fixture()
def client(sync_service: SyncV2Service) -> TestClient:
    return _client_for_service(sync_service)


def test_capabilities_endpoint_reports_supported_domains_and_encryption_posture(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/sync/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["protocol_version"] == "sync-v2-m1"
    assert body["min_supported_protocol_version"] == "sync-v2-m1"
    assert body["domains"] == list(SYNC_V2_SUPPORTED_DOMAINS)
    assert body["encryption"]["policy"] == "server_trusted_v1"
    assert body["encryption"]["ready"] is True
    assert body["encryption"]["attestation"]["mode"] == "managed_storage"
    assert body["blob_transfer"] == {"supported": False}
    assert body["warnings"] == []


def test_profile_endpoint_is_read_only_when_no_dataset_exists(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    response = client.get("/api/v1/sync/profile", params={"device_id": "device-1"})

    assert response.status_code == 200
    body = response.json()
    assert body["profile_bootstrapped"] is False
    assert body["active_dataset_id"] is None
    assert body["dataset"] is None
    assert body["server_cursor"] == 0
    assert body["device"]["registered"] is False
    assert body["domain_status"] == []
    assert sync_service.store.list_datasets_for_user("user-1") == []
    assert sync_service.store.list_devices_for_user("user-1") == []


def test_device_lifecycle_endpoints_authorize_acknowledge_and_revoke(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    sync_service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-1",
            user_id="user-1",
            display_name="Trusted laptop",
            client_type="chatbook",
        )
    )
    sync_service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-2",
            user_id="user-1",
            display_name="New laptop",
            client_type="chatbook",
            status="pending_authorization",
            user_label="untrusted",
        )
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )

    renamed = client.patch(
        "/api/v1/sync/devices/device-2",
        json={"user_label": "travel laptop"},
    )
    requested = client.post(
        "/api/v1/sync/device-authorizations",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-2",
            "authorization_method": "existing_device",
            "idempotency_key": "authorize-device-2",
        },
    )
    retry = client.post(
        "/api/v1/sync/device-authorizations",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-2",
            "authorization_method": "existing_device",
            "idempotency_key": "authorize-device-2",
        },
    )
    authorization_id = requested.json().get("authorization_id", "missing")
    approved = client.post(
        f"/api/v1/sync/device-authorizations/{authorization_id}/approve",
        json={
            "dataset_id": "dataset-1",
            "approving_device_id": "device-1",
            "idempotency_key": "approve-device-2",
        },
    )
    paused = client.post("/api/v1/sync/devices/device-2/pause")
    paused_ack = client.post(
        "/api/v1/sync/device-acknowledgments",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-2",
            "domain_acks": [
                {
                    "domain": "notes.note",
                    "through_server_sequence": 4,
                    "applied_at": "2026-05-23T18:29:00+00:00",
                }
            ],
        },
    )
    resumed = client.post("/api/v1/sync/devices/device-2/resume")
    acknowledged = client.post(
        "/api/v1/sync/device-acknowledgments",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-2",
            "domain_acks": [
                {
                    "domain": "notes.note",
                    "through_server_sequence": 5,
                    "applied_at": "2026-05-23T18:30:00+00:00",
                    "idempotency_key": "notes-ack-5",
                }
            ],
            "blob_acks": [
                {
                    "attachment_id": "attachment-1",
                    "payload_hash": _sha256(b"attachment-1"),
                    "verified_at": "2026-05-23T18:31:00+00:00",
                    "idempotency_key": "blob-ack-1",
                }
            ],
        },
    )
    revoked = client.post(
        "/api/v1/sync/devices/device-2/revoke",
        json={"reason": "lost_device", "revoke_key_records": True},
    )
    revoked_restore_manifest = client.get(
        "/api/v1/sync/restore-manifest",
        params={"device_id": "device-2", "dataset_id": "dataset-1"},
    )
    revoked_restore_preview = client.post(
        "/api/v1/sync/restore/preview",
        json={"device_id": "device-2", "dataset_ids": ["dataset-1"]},
    )
    revoked_repair = client.post(
        "/api/v1/sync/repair",
        json={"dataset_id": "dataset-1", "device_id": "device-2"},
    )
    visible = client.get("/api/v1/sync/devices")
    auditable = client.get("/api/v1/sync/devices", params={"include_revoked": "true"})

    assert renamed.status_code == 200
    assert renamed.json()["user_label"] == "travel laptop"
    assert requested.status_code == 200
    assert retry.status_code == 200
    assert retry.json()["authorization_id"] == requested.json()["authorization_id"]
    assert requested.json()["status"] == "pending"
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
    assert approved.json()["approving_device_id"] == "device-1"
    assert paused.status_code == 200
    assert paused.json()["status"] == "paused"
    assert paused_ack.status_code == 404
    assert resumed.status_code == 200
    assert resumed.json()["status"] == "active"
    assert acknowledged.status_code == 200
    assert acknowledged.json()["domain_acks"]["notes.note"]["through_server_sequence"] == 5
    assert acknowledged.json()["blob_acks"][0]["attachment_id"] == "attachment-1"
    assert revoked.status_code == 200
    assert revoked.json()["status"] == "revoked"
    assert revoked.json()["revoked_reason"] == "lost_device"
    assert revoked_restore_manifest.status_code == 404
    assert revoked_restore_preview.status_code == 404
    assert revoked_repair.status_code == 404
    assert [device["device_id"] for device in visible.json()] == ["device-1"]
    assert {
        device["device_id"]: device["status"]
        for device in auditable.json()
    } == {"device-1": "active", "device-2": "revoked"}


def test_background_sync_policy_lease_and_status_endpoints(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    sync_service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-1",
            user_id="user-1",
            display_name="Trusted laptop",
            client_type="chatbook",
        )
    )
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )

    default_policy = client.get(
        "/api/v1/sync/background-policy",
        params={"dataset_id": "dataset-1", "device_id": "device-1"},
    )
    patched_policy = client.patch(
        "/api/v1/sync/background-policy",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "enabled": False,
            "paused_reason": "user_paused",
            "pending_local_changes": True,
        },
    )
    lease = client.post(
        "/api/v1/sync/background-leases",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "lease_id": "lease-1",
            "ttl_seconds": 120,
        },
    )
    held = client.post(
        "/api/v1/sync/background-leases",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "lease_id": "lease-2",
            "ttl_seconds": 120,
        },
    )
    status = client.get(
        "/api/v1/sync/background-status",
        params={"dataset_id": "dataset-1", "device_id": "device-1"},
    )

    assert default_policy.status_code == 200
    assert default_policy.json()["enabled"] is True
    assert patched_policy.status_code == 200
    assert patched_policy.json()["enabled"] is False
    assert patched_policy.json()["paused_reason"] == "user_paused"
    assert patched_policy.json()["pending_local_changes"] is True
    assert lease.status_code == 200
    assert lease.json()["status"] == "acquired"
    assert lease.json()["acquired"] is True
    assert held.status_code == 200
    assert held.json()["status"] == "held_by_other"
    assert held.json()["lease_id"] == "lease-1"
    assert status.status_code == 200
    assert status.json()["policy"]["enabled"] is False
    assert status.json()["lease"]["lease_id"] == "lease-1"
    assert {item["domain"] for item in status.json()["domains"]} == {
        "notes.note",
        "attachment.ref",
    }


def test_profile_endpoint_for_fresh_user_does_not_create_sync_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_db_path = tmp_path / "fresh_sync_v2.db"
    monkeypatch.setenv("SYNC_V2_SQLITE_PATH", str(sync_db_path))
    monkeypatch.setenv("SYNC_V2_AT_REST_ENCRYPTION_MODE", "managed_storage")
    monkeypatch.setenv("SYNC_V2_SERVER_TRUSTED_ENABLED", "true")
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    client = TestClient(app)

    response = client.get("/api/v1/sync/profile", params={"device_id": "device-1"})

    assert response.status_code == 200
    assert response.json()["profile_bootstrapped"] is False
    assert response.json()["dataset"] is None
    assert response.json()["device"]["registered"] is False
    assert not sync_db_path.exists()


def test_profile_bootstrap_endpoint_idempotently_creates_dataset_and_device(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    payload = {
        "client_family": "chatbook",
        "mode": "offline_sync",
        "device_id": "device-1",
        "device_name": "Laptop",
        "client_profile_id": "profile-1",
        "client_instance": {"app_version": "0.4.0", "platform": "macos"},
        "requested_domains": list(M1_SYNC_DOMAINS),
    }

    first = client.post("/api/v1/sync/profile/bootstrap", json=payload)
    second = client.post("/api/v1/sync/profile/bootstrap", json=payload)
    profile = client.get("/api/v1/sync/profile", params={"device_id": "device-1"})

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["created"] is True
    assert second_body["created"] is False
    assert first_body["profile_bootstrapped"] is True
    assert first_body["device"]["device_id"] == "device-1"
    assert first_body["device"]["registered"] is True
    assert first_body["device"]["client_profile_id"] == "profile-1"
    assert first_body["dataset"]["default_personal"] is True
    assert first_body["dataset"]["client_family"] == "chatbook"
    assert first_body["dataset"]["domains"] == list(M1_SYNC_DOMAINS)
    assert first_body["active_dataset_id"] == first_body["dataset"]["dataset_id"]
    assert second_body["dataset"]["dataset_id"] == first_body["dataset"]["dataset_id"]
    assert profile.json()["dataset"]["dataset_id"] == first_body["dataset"]["dataset_id"]
    assert {item["domain"] for item in profile.json()["domain_status"]} == set(M1_SYNC_DOMAINS)
    assert len(sync_service.store.list_datasets_for_user("user-1")) == 1
    assert len(sync_service.store.list_devices_for_user("user-1")) == 1


def test_profile_bootstrap_endpoint_reuses_omitted_device_by_client_profile_id(
    tmp_path: Path,
) -> None:
    issued: list[str] = []

    def _id_factory(prefix: str) -> str:
        value = f"{prefix}-{len(issued) + 1}"
        issued.append(value)
        return value

    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_endpoints.db")),
        adapters=_registry(),
        clock=_clock,
        id_factory=_id_factory,
        settings=SyncV2Settings(
            server_trusted_encryption=_ready_encryption(),
            restore_manifest_scan_limit=100,
        ),
    )
    client = _client_for_service(service)
    payload = {
        "client_family": "chatbook",
        "mode": "offline_sync",
        "device_name": "Laptop",
        "client_profile_id": "profile-1",
    }

    first = client.post("/api/v1/sync/profile/bootstrap", json=payload)
    second = client.post("/api/v1/sync/profile/bootstrap", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["device"]["device_id"] == first.json()["device"]["device_id"]
    assert [device.device_id for device in service.store.list_devices_for_user("user-1")] == [
        first.json()["device"]["device_id"]
    ]


def test_profile_bootstrap_endpoint_without_device_or_profile_generates_device(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    response = client.post(
        "/api/v1/sync/profile/bootstrap",
        json={
            "client_family": "chatbook",
            "mode": "offline_sync",
            "device_name": "Laptop",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["profile_bootstrapped"] is True
    assert body["active_dataset_id"] is not None
    assert body["device"]["device_id"] == "device-generated"
    assert body["device"]["registered"] is True
    assert body["device"]["client_profile_id"] is None
    devices = sync_service.store.list_devices_for_user("user-1")
    assert len(devices) == 1
    assert devices[0].device_id == "device-generated"
    assert devices[0].capabilities["client_profile_id"] is None
    assert len(sync_service.store.list_datasets_for_user("user-1")) == 1


def test_profile_bootstrap_endpoint_fails_closed_when_encryption_is_not_ready(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, encryption=_not_ready_encryption())
    client = _client_for_service(service)

    failed = client.post(
        "/api/v1/sync/profile/bootstrap",
        json={
            "client_family": "chatbook",
            "mode": "offline_sync",
            "device_id": "device-1",
            "device_name": "Laptop",
        },
    )
    profile = client.get("/api/v1/sync/profile", params={"device_id": "device-1"})

    assert failed.status_code == 412
    assert failed.json()["detail"]["error_code"] == "sync_encryption_attestation_required"
    assert profile.status_code == 200
    assert profile.json()["capabilities"]["encryption"]["ready"] is False
    assert profile.json()["warnings"][0]["code"] == "sync_encryption_attestation_required"
    assert service.store.list_datasets_for_user("user-1") == []
    assert service.store.list_devices_for_user("user-1") == []


def test_profile_endpoint_normalizes_unknown_lower_level_device_mode(
    sync_service: SyncV2Service,
) -> None:
    sync_service.store.upsert_device(
        SyncDeviceUpsert(
            device_id="device-legacy",
            user_id="user-1",
            display_name="Legacy",
            client_type="chatbook",
            capabilities={
                "client_profile_id": "profile-legacy",
                "sync_mode": "legacy_internal_mode",
            },
        )
    )
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = lambda: sync_service
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/api/v1/sync/profile", params={"device_id": "device-legacy"})

    assert response.status_code == 200
    assert response.json()["device"]["registered"] is True
    assert response.json()["device"]["mode"] is None


def test_lower_level_register_and_enroll_routes_remain_available_for_internal_callers(
    client: TestClient,
) -> None:
    registered = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "device-1",
            "display_name": "Laptop",
            "client_type": "chatbook",
            "client_version": "0.4.0",
            "capabilities": {"domains": list(M1_SYNC_DOMAINS)},
        },
    )
    enrolled = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "scope_type": "personal",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": {"default_personal": True, "client_family": "chatbook"},
        },
    )

    assert registered.status_code == 200
    assert registered.json()["device_id"] == "device-1"
    assert registered.json()["server_capabilities"]["domains"] == list(SYNC_V2_SUPPORTED_DOMAINS)
    assert enrolled.status_code == 200
    assert enrolled.json()["dataset_id"] == "dataset-1"
    assert enrolled.json()["encryption_policy"] == "server_trusted_v1"
    assert enrolled.json()["domains"] == list(M1_SYNC_DOMAINS)
    assert enrolled.json()["key_setup_required"] is False


def test_key_recovery_bundle_validation_error_does_not_expose_wrapped_material(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])
    secret = "wrapped:super-secret-key-material"
    log_messages: list[str] = []
    handler_id = logger.add(
        lambda message: log_messages.append(str(message)),
        format="{message} {extra}",
        level="WARNING",
    )
    try:
        response = client.post(
            "/api/v1/sync/keys/recovery-bundle",
            json={
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "key_purpose": "workspace_share",
                "wrapped_key_blob": secret,
                "kdf_metadata": {"algorithm": "scrypt", "salt": "secret-salt"},
            },
        )
    finally:
        logger.remove(handler_id)

    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "sync_validation_failed"
    assert secret not in response.text
    assert "secret-salt" not in response.text
    rendered_logs = "\n".join(log_messages)
    assert secret not in rendered_logs
    assert "secret-salt" not in rendered_logs


def test_datasets_enroll_endpoint_fails_closed_when_encryption_is_not_ready(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, encryption=_not_ready_encryption())
    client = _client_for_service(service)

    response = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "scope_type": "personal",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": {"default_personal": True, "client_family": "chatbook"},
        },
    )

    assert response.status_code == 412
    assert response.json()["detail"]["error_code"] == "sync_encryption_attestation_required"
    assert service.store.list_datasets_for_user("user-1") == []


def _note_envelope_json(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-note",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "client_sequence": 1,
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "object_revision": 1,
        "schema_version": 1,
        "payload": {"title": "Research note", "content": "Body"},
        "payload_hash": "sha256:note-1",
        "created_at_client": "2026-05-23T18:12:44+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def test_push_and_pull_endpoint_expose_apply_outcomes_for_replayable_failures(
    tmp_path: Path,
) -> None:
    service = _build_service(
        tmp_path,
        materializers={"notes.note": _EndpointOutcomeMaterializer()},
    )
    client = _client_for_service(service)
    client.post(
        "/api/v1/sync/devices/register",
        json={"device_id": "device-1", "display_name": "Laptop", "client_type": "chatbook"},
    )
    client.post(
        "/api/v1/sync/devices/register",
        json={"device_id": "device-2", "display_name": "Phone", "client_type": "chatbook"},
    )
    client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
        },
    )

    pushed = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                _note_envelope_json(),
                _note_envelope_json(
                    client_envelope_id="env-failed",
                    object_id="note-fail",
                    client_sequence=2,
                    payload_hash="sha256:failed",
                ),
            ],
        },
    )
    pulled = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-2",
            "cursor": "0",
            "domain": "notes.note",
            "include_own_changes": "true",
        },
    )

    assert pushed.status_code == 200
    accepted = pushed.json()["accepted"]
    assert [
        (item["client_envelope_id"], item["server_cursor"], item["object_revision"], item["apply_status"])
        for item in accepted
    ] == [
        ("env-note", 1, 1, "applied"),
        ("env-failed", 2, None, "failed"),
    ]
    assert accepted[1]["apply_error_code"] == "projection_failed"
    assert "replayable" in accepted[1]["apply_error_message"]
    assert pulled.status_code == 200
    failed = pulled.json()["envelopes"][1]
    assert failed["client_envelope_id"] == "env-failed"
    assert failed["apply_status"] == "failed"
    assert failed["apply_error_code"] == "projection_failed"
    assert "replayable" in failed["apply_error_message"]


def test_push_endpoint_reports_dataset_mismatch_per_envelope_in_mixed_batch(
    client: TestClient,
) -> None:
    client.post(
        "/api/v1/sync/devices/register",
        json={"device_id": "device-1", "display_name": "Laptop", "client_type": "chatbook"},
    )
    client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
        },
    )

    response = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                _note_envelope_json(),
                _note_envelope_json(
                    client_envelope_id="env-wrong-dataset",
                    dataset_id="dataset-other",
                    object_id="note-other",
                    client_sequence=2,
                    payload_hash="sha256:wrong-dataset",
                ),
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert [item["client_envelope_id"] for item in body["accepted"]] == ["env-note"]
    assert body["rejected"][0]["client_envelope_id"] == "env-wrong-dataset"
    assert body["rejected"][0]["error_code"] == "dataset_mismatch"


def test_legacy_send_and_get_routes_return_replaced_gone(
    sync_service: SyncV2Service,
) -> None:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    legacy_client = TestClient(app)

    send = legacy_client.post("/api/v1/sync/send", json={"client_id": "legacy-client", "changes": []})
    invalid_send = legacy_client.post("/api/v1/sync/send", json={"not": "a legacy media sync payload"})
    get = legacy_client.get(
        "/api/v1/sync/get",
        params={"client_id": "legacy-client", "since_change_id": 0},
    )
    invalid_get = legacy_client.get(
        "/api/v1/sync/get",
        params={"client_id": "legacy-client", "since_change_id": "not-an-int"},
    )

    assert send.status_code == 410
    assert send.json()["detail"]["error_code"] == "sync_legacy_endpoint_replaced"
    assert send.json()["detail"]["replacement"] == "/api/v1/sync/push"
    assert invalid_send.status_code == 410
    assert invalid_send.json()["detail"]["error_code"] == "sync_legacy_endpoint_replaced"
    assert invalid_send.json()["detail"]["replacement"] == "/api/v1/sync/push"
    assert get.status_code == 410
    assert get.json()["detail"]["error_code"] == "sync_legacy_endpoint_replaced"
    assert get.json()["detail"]["replacement"] == "/api/v1/sync/pull"
    assert invalid_get.status_code == 410
    assert invalid_get.json()["detail"]["error_code"] == "sync_legacy_endpoint_replaced"
    assert invalid_get.json()["detail"]["replacement"] == "/api/v1/sync/pull"


def test_resumable_blob_upload_endpoints_accept_raw_chunks_and_complete(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    client = _client_for_service(service)
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"hello world"

    create_response = client.post(
        "/api/v1/sync/blob-uploads",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "domain": "notes.note",
            "object_id": "note-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": len(payload),
            "payload_hash": _sha256(payload),
            "chunk_size": 6,
            "chunk_count": 2,
            "idempotency_key": "upload-key-1",
        },
    )
    assert create_response.status_code == 200
    upload_id = create_response.json()["upload_id"]

    first_response = client.put(
        f"/api/v1/sync/blob-uploads/{upload_id}/chunks/0",
        params={
            "dataset_id": "dataset-1",
            "offset_bytes": 0,
            "chunk_hash": _sha256(payload[:6]),
        },
        content=payload[:6],
        headers={"content-type": "application/octet-stream"},
    )
    second_response = client.put(
        f"/api/v1/sync/blob-uploads/{upload_id}/chunks/1",
        params={
            "dataset_id": "dataset-1",
            "offset_bytes": 6,
            "chunk_hash": _sha256(payload[6:]),
        },
        content=payload[6:],
        headers={"content-type": "application/octet-stream"},
    )
    complete_response = client.post(
        f"/api/v1/sync/blob-uploads/{upload_id}/complete",
        params={"dataset_id": "dataset-1"},
    )

    assert first_response.status_code == 200
    assert first_response.json()["missing_chunks"] == [1]
    assert second_response.status_code == 200
    assert second_response.json()["missing_chunks"] == []
    assert complete_response.status_code == 200
    body = complete_response.json()
    assert body["attachment_id"] == "attachment-1"
    assert body["status"] == "available"
    assert body["stored"] is True
    assert body["payload_hash"] == _sha256(payload)
    assert body["quota"]["used_blob_bytes"] == len(payload)


def test_blob_upload_endpoint_maps_validation_errors_to_safe_statuses(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    client = _client_for_service(service)
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note"])

    bad_hash_response = client.put(
        "/api/v1/sync/blob-uploads/upload-missing/chunks/0",
        params={
            "dataset_id": "dataset-1",
            "offset_bytes": 0,
            "chunk_hash": "sha256:" + "0" * 64,
        },
        content=b"bad",
        headers={"content-type": "application/octet-stream"},
    )
    quota_response = client.post(
        "/api/v1/sync/blob-uploads",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "domain": "notes.note",
            "object_id": "note-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 512,
            "payload_hash": _sha256(b"x" * 512),
            "chunk_size": 8,
            "chunk_count": 64,
        },
    )

    assert bad_hash_response.status_code == 404
    assert bad_hash_response.json()["detail"]["error_code"] == "sync_resource_not_found"
    assert quota_response.status_code == 413
    assert quota_response.json()["detail"]["error_code"] == "sync_attachment_too_large"


def test_small_attachment_endpoint_uses_blob_commit_path(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    client = _client_for_service(service)
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"small encrypted payload"

    response = client.post(
        "/api/v1/sync/attachments",
        json={
            "dataset_id": "dataset-1",
            "domain": "notes.note",
            "object_id": "note-1",
            "attachment_id": "attachment-small",
            "content_type": "application/octet-stream",
            "size_bytes": len(payload),
            "payload_ciphertext": payload.decode("utf-8"),
            "payload_hash": _sha256(payload),
        },
    )
    quota = service.store.summarize_blob_quota("user-1", dataset_id="dataset-1")

    assert response.status_code == 200
    body = response.json()
    assert body["attachment_id"] == "attachment-small"
    assert body["stored"] is True
    assert body["payload_hash"] == _sha256(payload)
    assert quota.used_blob_bytes == len(payload)


def test_attachment_download_manifest_and_byte_serving_are_dataset_scoped(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    id_counter = {"value": 0}

    def next_id(prefix: str) -> str:
        id_counter["value"] += 1
        return f"{prefix}-{id_counter['value']}"

    service.id_factory = next_id
    client = _client_for_service(service)
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
    )
    service.register_device(
        user_id="user-2",
        device_id="device-2",
        display_name="Other",
        client_type="chatbook",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    service.enroll_dataset(user_id="user-2", dataset_id="dataset-2", domains=["notes.note", "attachment.ref"])
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
    private_payload = b"private payload"
    service.store_attachment(
        user_id="user-2",
        dataset_id="dataset-2",
        domain="notes.note",
        entity_id="note-2",
        attachment_id="attachment-private",
        content_type="application/octet-stream",
        size_bytes=len(private_payload),
        payload_ciphertext=private_payload.decode("utf-8"),
        payload_hash=_sha256(private_payload),
    )

    manifest_response = client.get(
        "/api/v1/sync/attachments/attachment-download/manifest",
        params={"dataset_id": "dataset-1", "chunk_size": 8},
    )
    bytes_response = client.get(
        "/api/v1/sync/attachments/attachment-download",
        params={"dataset_id": "dataset-1", "offset": 5, "size": 8},
    )
    forbidden_response = client.get(
        "/api/v1/sync/attachments/attachment-private",
        params={"dataset_id": "dataset-2"},
    )

    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["availability"] == "available"
    assert manifest["payload_hash"] == _sha256(payload)
    assert [chunk["chunk_index"] for chunk in manifest["chunks"]] == [0, 1, 2]
    assert manifest["chunks"][0]["chunk_hash"] == _sha256(payload[:8])
    assert bytes_response.status_code == 200
    assert bytes_response.content == payload[5:13]
    assert bytes_response.headers["content-type"] == "application/octet-stream"
    assert forbidden_response.status_code == 404
