from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User, get_request_user
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterConflict,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-10T12:00:00+00:00"


def _test_user() -> User:
    return User(id="user-1", username="user-1")


@pytest.fixture()
def registry() -> SyncAdapterRegistry:
    registry = SyncAdapterRegistry()
    registry.register(StaticSyncAdapter(domain="notes", supported_adapter_versions={1}))
    registry.register(
        StaticSyncAdapter(
            domain="chat",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="chat",
                    entity_id="conversation-1",
                    conflict_type="version_divergence",
                    message="chat conflict",
                )
            },
        )
    )
    registry.register(StaticSyncAdapter(domain="source_cache", supported_adapter_versions={1}))
    return registry


@pytest.fixture()
def sync_service(tmp_path: Path, registry: SyncAdapterRegistry) -> SyncV2Service:
    return SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_endpoints.db")),
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(max_pull_page_size=2),
    )


@pytest.fixture()
def client(sync_service: SyncV2Service) -> TestClient:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    if hasattr(sync_endpoint, "get_sync_v2_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service
    return TestClient(app)


def _envelope(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "client_envelope_id": "env-1",
        "dataset_id": "dataset-1",
        "domain": "notes",
        "entity_id": "note-1",
        "operation": "upsert",
        "adapter_version": 1,
        "device_id": "device-1",
        "stable_key": "note:note-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": "ciphertext:opaque",
        "payload_clear": {"status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 24,
    }
    payload.update(overrides)
    return payload


def _register_device(client: TestClient, device_id: str = "device-1") -> dict[str, Any]:
    response = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": device_id,
            "display_name": "Laptop",
            "client_type": "chatbook",
            "client_version": "0.1.0",
            "capabilities": {"domains": ["notes", "chat"]},
        },
    )
    assert response.status_code == 200
    return response.json()


def _enroll_dataset(
    client: TestClient,
    *,
    dataset_id: str = "dataset-1",
    domains: list[str] | None = None,
    encryption_policy: str = "client_private_v1",
) -> dict[str, Any]:
    response = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": dataset_id,
            "device_id": "device-1",
            "scope_type": "personal",
            "domains": domains or ["notes", "chat", "source_cache"],
            "encryption_policy": encryption_policy,
            "metadata": {"label": "private label"},
        },
    )
    assert response.status_code == 200
    return response.json()


def _push(client: TestClient, envelopes: list[dict[str, Any]], *, device_id: str = "device-1"):
    return client.post(
        "/api/v1/sync/push",
        json={"dataset_id": "dataset-1", "device_id": device_id, "envelopes": envelopes},
    )


def test_capabilities_endpoint_returns_sync_v2_contract(client: TestClient):
    response = client.get("/api/v1/sync/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["protocol_version"] == 2
    assert body["min_supported_protocol_version"] == 2
    assert body["supported_domains"] == ["chat", "notes", "source_cache"]
    assert body["supports_restore_manifest"] is True
    assert body["supports_conflicts"] is True
    assert body["supports_attachments"] is False
    assert body["server_time"] == _clock()


def test_devices_register_endpoint_registers_and_refreshes(client: TestClient):
    created = _register_device(client)
    refreshed = _register_device(client)

    assert created["device_id"] == "device-1"
    assert refreshed["device_id"] == "device-1"
    assert refreshed["server_capabilities"]["protocol_version"] == 2
    assert refreshed["last_seen_at"] >= created["last_seen_at"]


def test_datasets_enroll_endpoint_returns_dataset_cursors(client: TestClient):
    _register_device(client)

    body = _enroll_dataset(client, domains=["notes", "chat"])

    assert body["dataset_id"] == "dataset-1"
    assert body["scope_type"] == "personal"
    assert body["encryption_policy"] == "client_private_v1"
    assert body["domains"] == ["notes", "chat"]
    assert body["cursors"] == {"notes": "0", "chat": "0"}
    assert body["key_setup_required"] is True


def test_restore_manifest_endpoint_applies_dataset_and_domain_filters(client: TestClient):
    _register_device(client)
    _enroll_dataset(client, dataset_id="dataset-1", domains=["notes", "chat"])
    _enroll_dataset(client, dataset_id="dataset-2", domains=["source_cache"])
    assert _push(client, [_envelope()]).status_code == 200
    assert _push(
        client,
        [
            _envelope(
                client_envelope_id="env-chat",
                domain="chat",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:chat-1",
            )
        ],
    ).status_code == 200

    response = client.get(
        "/api/v1/sync/restore-manifest",
        params=[("dataset_id", "dataset-1"), ("domain", "notes")],
    )

    assert response.status_code == 200
    body = response.json()
    assert [dataset["dataset_id"] for dataset in body["datasets"]] == ["dataset-1"]
    assert body["datasets"][0]["domains"] == ["notes"]
    assert body["datasets"][0]["approximate_counts"] == {"notes": 1}
    assert body["filters_applied"] == {"dataset_ids": ["dataset-1"], "domains": ["notes"]}
    assert "private label" not in str(body)
    assert "ciphertext:opaque" not in str(body)


def test_push_endpoint_is_idempotent_and_rejects_unsupported_adapter_versions(
    client: TestClient,
):
    _register_device(client)
    _enroll_dataset(client, domains=["notes"])

    first = _push(
        client,
        [
            _envelope(client_envelope_id="env-idempotent"),
            _envelope(
                client_envelope_id="env-unsupported",
                entity_id="note-unsupported",
                stable_key="note:unsupported",
                payload_hash="sha256:unsupported",
                adapter_version=99,
            ),
        ],
    )
    second = _push(client, [_envelope(client_envelope_id="env-idempotent")])

    assert first.status_code == 200
    assert second.status_code == 200
    first_body = first.json()
    second_body = second.json()
    assert first_body["accepted"][0]["client_envelope_id"] == "env-idempotent"
    assert second_body["accepted"][0]["server_sequence"] == first_body["accepted"][0]["server_sequence"]
    assert first_body["rejected"][0]["client_envelope_id"] == "env-unsupported"
    assert first_body["rejected"][0]["error_code"] == "unsupported_adapter_version"


def test_push_endpoint_requires_top_level_device_id(client: TestClient):
    _register_device(client)
    _enroll_dataset(client, domains=["notes"])

    response = client.post(
        "/api/v1/sync/push",
        json={"dataset_id": "dataset-1", "envelopes": [_envelope()]},
    )

    assert response.status_code == 422


def test_pull_endpoint_filters_domains_excludes_echo_and_pages(client: TestClient):
    _register_device(client, "device-1")
    _register_device(client, "device-2")
    _enroll_dataset(client, domains=["notes", "chat"])
    assert _push(client, [_envelope(client_envelope_id="own-env")], device_id="device-1").status_code == 200
    assert _push(
        client,
        [
            _envelope(
                client_envelope_id="remote-note-1",
                device_id="device-2",
                entity_id="note-2",
                stable_key="note:2",
                payload_hash="sha256:note-2",
            ),
            _envelope(
                client_envelope_id="remote-chat",
                device_id="device-2",
                domain="chat",
                entity_id="conversation-1",
                stable_key="chat:1",
                payload_hash="sha256:chat-1",
            ),
            _envelope(
                client_envelope_id="remote-note-2",
                device_id="device-2",
                entity_id="note-3",
                stable_key="note:3",
                payload_hash="sha256:note-3",
            ),
        ],
        device_id="device-2",
    ).status_code == 200

    first = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "cursor": "0",
            "domain": "notes",
            "page_size": "1",
        },
    )
    second = client.get(
        "/api/v1/sync/pull",
        params=[
            ("dataset_id", "dataset-1"),
            ("device_id", "device-1"),
            ("cursor", first.json()["next_cursor"]),
            ("domain", "notes"),
            ("page_size", "1"),
        ],
    )

    assert first.status_code == 200
    assert [item["client_envelope_id"] for item in first.json()["envelopes"]] == ["remote-note-1"]
    assert first.json()["has_more"] is True
    assert [item["client_envelope_id"] for item in second.json()["envelopes"]] == ["remote-note-2"]
    assert second.json()["has_more"] is False


def test_pull_endpoint_preserves_dataset_encryption_policy(client: TestClient):
    _register_device(client, "device-1")
    _register_device(client, "device-2")
    _enroll_dataset(client, domains=["notes"], encryption_policy="server_trusted")
    pushed = _push(
        client,
        [
            _envelope(
                client_envelope_id="server-trusted-note",
                device_id="device-2",
                payload_ciphertext=None,
                payload_clear={"body": "clear server-managed content"},
                payload_hash="sha256:server-trusted-note",
                encryption_policy="server_trusted",
            )
        ],
        device_id="device-2",
    )

    pulled = client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": "dataset-1", "device_id": "device-1", "cursor": "0"},
    )

    assert pushed.status_code == 200
    assert pulled.status_code == 200
    envelope = pulled.json()["envelopes"][0]
    assert envelope["client_envelope_id"] == "server-trusted-note"
    assert envelope["encryption_policy"] == "server_trusted"
    assert envelope["payload_clear"]["body"] == "clear server-managed content"


def test_conflicts_list_and_resolve_endpoints(client: TestClient):
    _register_device(client)
    _enroll_dataset(client, domains=["chat"])
    conflict_push = _push(
        client,
        [
            _envelope(
                client_envelope_id="env-conflict",
                domain="chat",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = conflict_push.json()["conflicts"][0]["conflict_id"]

    listed = client.get("/api/v1/sync/conflicts", params={"dataset_id": "dataset-1"})
    resolved = client.post(
        f"/api/v1/sync/conflicts/{conflict_id}/resolve",
        json={"action": "dismiss", "resolved_by_device_id": "device-1", "notes": "duplicate"},
    )

    assert listed.status_code == 200
    assert listed.json()[0]["conflict_id"] == conflict_id
    assert listed.json()[0]["status"] == "unresolved"
    assert resolved.status_code == 200
    assert resolved.json()["conflict_id"] == conflict_id
    assert resolved.json()["status"] == "dismissed"
    assert resolved.json()["resolved_at"] is not None


def test_conflict_resolve_endpoint_persists_resolution_envelope(client: TestClient):
    _register_device(client, "device-1")
    _register_device(client, "device-2")
    _enroll_dataset(client, domains=["chat"])
    conflict_push = _push(
        client,
        [
            _envelope(
                client_envelope_id="env-conflict",
                domain="chat",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = conflict_push.json()["conflicts"][0]["conflict_id"]

    resolved = client.post(
        f"/api/v1/sync/conflicts/{conflict_id}/resolve",
        json={
            "action": "merge",
            "resolved_by_device_id": "device-1",
            "resolution_envelope": _envelope(
                client_envelope_id="env-resolution",
                domain="chat",
                entity_id="conversation-1",
                operation="resolve_conflict",
                stable_key="chat:conversation-1",
                payload_hash="sha256:resolution",
            ),
        },
    )
    pulled = client.get(
        "/api/v1/sync/pull",
        params={"dataset_id": "dataset-1", "device_id": "device-2", "domain": "chat", "cursor": "0"},
    )

    assert resolved.status_code == 200
    assert resolved.json()["status"] == "resolved"
    assert resolved.json()["resolved_by_envelope_id"] == "env-resolution"
    assert pulled.status_code == 200
    assert [item["client_envelope_id"] for item in pulled.json()["envelopes"]] == ["env-resolution"]


def test_conflict_resolve_endpoint_returns_client_error_for_invalid_private_resolution(
    client: TestClient,
):
    _register_device(client)
    _enroll_dataset(client, domains=["chat"])
    conflict_push = _push(
        client,
        [
            _envelope(
                client_envelope_id="env-conflict",
                domain="chat",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:conflict",
            )
        ],
    )
    conflict_id = conflict_push.json()["conflicts"][0]["conflict_id"]

    response = client.post(
        f"/api/v1/sync/conflicts/{conflict_id}/resolve",
        json={
            "action": "merge",
            "resolved_by_device_id": "device-1",
            "resolution_envelope": _envelope(
                client_envelope_id="env-invalid-resolution",
                domain="chat",
                entity_id="conversation-1",
                operation="resolve_conflict",
                stable_key="chat:conversation-1",
                payload_ciphertext=None,
                payload_clear={"body": "known plaintext"},
                payload_hash="sha256:invalid-resolution",
                encryption_policy="server_trusted",
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "sync_validation_failed"
    assert "known plaintext" not in str(response.json())


def test_attachments_endpoint_returns_feature_detect_response(client: TestClient):
    response = client.post(
        "/api/v1/sync/attachments",
        json={
            "dataset_id": "dataset-1",
            "domain": "notes",
            "entity_id": "note-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 12,
            "payload_ciphertext": "ciphertext:attachment-secret",
            "payload_hash": "sha256:attachment",
        },
    )

    assert response.status_code == 501
    assert response.json()["detail"]["error_code"] == "sync_attachments_not_enabled"
    assert "attachment-secret" not in str(response.json())


def test_attachments_endpoint_feature_detects_without_strict_body_validation(client: TestClient):
    response = client.post(
        "/api/v1/sync/attachments",
        json={"payload_ciphertext": "ciphertext:attachment-secret"},
    )

    assert response.status_code == 501
    assert response.json()["detail"]["error_code"] == "sync_attachments_not_enabled"
    assert "attachment-secret" not in str(response.json())


def test_key_recovery_bundle_endpoint_stores_safe_metadata(client: TestClient):
    _register_device(client)
    _enroll_dataset(client, domains=["notes"])

    response = client.post(
        "/api/v1/sync/keys/recovery-bundle",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped:very-secret-key",
            "kdf_metadata": {"algorithm": "argon2id"},
            "recovery_hint": "laptop",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["key_record_id"] == "key-generated"
    assert body["dataset_id"] == "dataset-1"
    assert body["device_id"] == "device-1"
    assert body["key_purpose"] == "dataset_recovery"
    assert body["recovery_hint"] == "laptop"
    assert "wrapped_key_blob" not in body
    assert "very-secret-key" not in str(body)


def test_key_recovery_bundle_endpoint_lists_opaque_material_without_manifest_leakage(
    client: TestClient,
):
    _register_device(client)
    _enroll_dataset(client, domains=["notes"])
    stored = client.post(
        "/api/v1/sync/keys/recovery-bundle",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "key_purpose": "dataset_recovery",
            "wrapped_key_blob": "wrapped:very-secret-key",
            "kdf_metadata": {"algorithm": "scrypt", "salt": "opaque-salt"},
            "recovery_hint": "laptop",
        },
    )

    response = client.get(
        "/api/v1/sync/keys/recovery-bundle",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "key_purpose": "dataset_recovery",
        },
    )
    manifest = client.get("/api/v1/sync/restore-manifest", params={"dataset_id": "dataset-1"})

    assert stored.status_code == 200
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_id"] == "dataset-1"
    assert len(body["key_records"]) == 1
    assert body["key_records"][0]["key_record_id"] == "key-generated"
    assert body["key_records"][0]["wrapped_key_blob"] == "wrapped:very-secret-key"
    assert body["key_records"][0]["kdf_metadata"] == {"algorithm": "scrypt", "salt": "opaque-salt"}
    assert body["key_records"][0]["recovery_hint"] == "laptop"
    assert body["key_records"][0]["revoked_at"] is None
    assert manifest.status_code == 200
    assert manifest.json()["datasets"][0]["key_recovery_available"] is True
    assert "wrapped:very-secret-key" not in str(manifest.json())
    assert "opaque-salt" not in str(manifest.json())


def test_key_recovery_bundle_endpoint_rejects_inaccessible_dataset_without_leakage(
    client: TestClient,
):
    response = client.get(
        "/api/v1/sync/keys/recovery-bundle",
        params={
            "dataset_id": "missing-dataset",
            "key_purpose": "dataset_recovery",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "sync_resource_not_found"
    assert "wrapped_key_blob" not in str(response.json())
    assert "kdf_metadata" not in str(response.json())


def test_sync_v2_errors_and_logs_do_not_expose_sensitive_material(client: TestClient):
    log_messages: list[str] = []
    sink_id = logger.add(
        lambda message: log_messages.append(str(message)),
        format="{message} | {extra}",
    )
    try:
        key_response = client.post(
            "/api/v1/sync/keys/recovery-bundle",
            json={
                "dataset_id": "missing-dataset",
                "device_id": "device-1",
                "key_purpose": "dataset_recovery",
                "wrapped_key_blob": "wrapped:leak-me",
                "kdf_metadata": {"known_plaintext": "do-not-leak"},
            },
        )
        conflict_response = client.post(
            "/api/v1/sync/conflicts/missing-conflict/resolve",
            json={
                "action": "merge",
                "resolved_by_device_id": "device-1",
                "resolution_envelope": _envelope(
                    client_envelope_id="resolution-secret",
                    payload_ciphertext="ciphertext:payload-secret",
                    payload_clear={"status": "clear-secret"},
                    payload_hash="sha256:resolution-secret",
                ),
            },
        )
    finally:
        logger.remove(sink_id)

    response_text = f"{key_response.json()}\n{conflict_response.json()}"
    log_output = "\n".join(log_messages)

    assert key_response.status_code == 404
    assert conflict_response.status_code == 404
    assert "Sync v2 request failed" in log_output
    for secret in ("leak-me", "do-not-leak", "payload-secret", "clear-secret"):
        assert secret not in response_text
        assert secret not in log_output


def test_legacy_send_and_get_routes_preserve_existing_policy(
    sync_service: SyncV2Service,
) -> None:
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    if hasattr(sync_endpoint, "get_sync_v2_service"):
        app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: sync_service

    class _Cursor:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

        def fetchone(self):
            return self._rows[0] if self._rows else None

    class _LegacyDb:
        db_path_str = "/tmp/legacy-media.db"

        def execute_query(self, query, params=()):
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
