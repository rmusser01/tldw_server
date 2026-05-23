from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.models import SyncDeviceUpsert
from tldw_Server_API.app.core.Sync.v2.models import M1_SYNC_DOMAINS
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-23T18:12:00+00:00"


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


def _build_service(tmp_path: Path, *, encryption=None) -> SyncV2Service:
    return SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_endpoints.db")),
        adapters=_registry(),
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(
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


def test_capabilities_endpoint_reports_m1_domains_and_encryption_posture(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/sync/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["protocol_version"] == "sync-v2-m1"
    assert body["min_supported_protocol_version"] == "sync-v2-m1"
    assert body["domains"] == list(M1_SYNC_DOMAINS)
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
    assert registered.json()["server_capabilities"]["domains"] == list(M1_SYNC_DOMAINS)
    assert enrolled.status_code == 200
    assert enrolled.json()["dataset_id"] == "dataset-1"
    assert enrolled.json()["encryption_policy"] == "server_trusted_v1"
    assert enrolled.json()["domains"] == list(M1_SYNC_DOMAINS)
    assert enrolled.json()["key_setup_required"] is False


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


def test_legacy_send_and_get_routes_preserve_existing_policy(
    sync_service: SyncV2Service,
) -> None:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service

    class _Cursor:
        def __init__(self, rows: list[tuple[int]]) -> None:
            self._rows = rows

        def fetchall(self) -> list[tuple[int]]:
            return self._rows

        def fetchone(self) -> tuple[int] | None:
            return self._rows[0] if self._rows else None

    class _LegacyDb:
        db_path_str = "/tmp/legacy-media.db"

        def execute_query(self, query: str, params: tuple[Any, ...] = ()) -> _Cursor:
            if "MAX(change_id)" in query:
                return _Cursor([(7,)])
            if "FROM sync_log" in query:
                assert params[1] == "legacy-client"
                return _Cursor([])
            raise AssertionError(query)

    app.dependency_overrides[get_media_db_for_user] = lambda: _LegacyDb()
    legacy_client = TestClient(app)

    send = legacy_client.post("/api/v1/sync/send", json={"client_id": "legacy-client", "changes": []})
    get = legacy_client.get(
        "/api/v1/sync/get",
        params={"client_id": "legacy-client", "since_change_id": 0},
    )

    assert send.status_code == 200
    assert send.json() == {"status": "success", "message": "No changes received."}
    assert get.status_code == 200
    assert get.json() == {"changes": [], "latest_change_id": 7}
