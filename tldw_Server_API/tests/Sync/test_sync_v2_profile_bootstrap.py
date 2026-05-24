from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import StaticSyncAdapter, SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    M1_SYNC_DOMAINS,
    SyncConflictCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-23T18:12:00+00:00"


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


def _service(
    tmp_path: Path,
    *,
    encryption=None,
    id_factory=None,
    scan_limit: int = 100,
) -> tuple[SyncV2Service, SyncV2Store]:
    store = SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_profile.db"))
    service = SyncV2Service(
        store=store,
        adapters=_registry(),
        clock=_clock,
        id_factory=id_factory or (lambda prefix: f"{prefix}-generated"),
        settings=SyncV2Settings(
            server_trusted_encryption=encryption or _ready_encryption(),
            restore_manifest_scan_limit=scan_limit,
        ),
    )
    return service, store


def _note_envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-note-1",
        "domain": "notes.note",
        "operation": "upsert",
        "object_id": "note-1",
        "device_id": "device-1",
        "client_profile_id": "profile-1",
        "client_sequence": 1,
        "payload": {"title": "Research note"},
        "payload_hash": "sha256:note-1",
        "created_at_client": "2026-05-23T18:10:00+00:00",
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def test_profile_is_read_only_when_no_bootstrap_exists(tmp_path: Path) -> None:
    service, store = _service(tmp_path)

    profile = service.profile(user_id="user-1", device_id="device-1")

    assert profile.profile_bootstrapped is False
    assert profile.active_dataset_id is None
    assert profile.dataset is None
    assert profile.server_cursor == 0
    assert profile.device is not None
    assert profile.device.registered is False
    assert profile.device.device_id == "device-1"
    assert profile.capabilities.encryption["ready"] is True
    assert profile.domain_status == []
    assert store.list_datasets_for_user("user-1") == []
    assert store.list_devices_for_user("user-1") == []


def test_bootstrap_creates_default_dataset_and_is_idempotent(tmp_path: Path) -> None:
    service, store = _service(tmp_path)

    first = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        device_name="Laptop",
        client_profile_id="profile-1",
        client_instance={"app_version": "0.4.0", "platform": "macos"},
    )
    second = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        device_name="Laptop",
        client_profile_id="profile-1",
        client_instance={"app_version": "0.4.0", "platform": "macos"},
    )

    assert first.created is True
    assert second.created is False
    assert first.profile_bootstrapped is True
    assert first.dataset is not None
    assert second.dataset is not None
    assert second.dataset.dataset_id == first.dataset.dataset_id
    assert first.active_dataset_id == first.dataset.dataset_id
    assert first.dataset.default_personal is True
    assert first.dataset.client_family == "chatbook"
    assert first.dataset.domains == list(M1_SYNC_DOMAINS)
    assert first.device is not None
    assert first.device.registered is True
    assert first.device.client_profile_id == "profile-1"
    assert first.server_cursor == 0
    assert len(store.list_datasets_for_user("user-1")) == 1
    assert len(store.list_devices_for_user("user-1")) == 1


def test_bootstrap_supports_server_frontend_with_generated_device_id(tmp_path: Path) -> None:
    service, _store = _service(tmp_path)

    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="server_frontend",
        device_name="Browser session",
        client_profile_id="browser-profile-1",
    )

    assert profile.created is True
    assert profile.device is not None
    assert profile.device.device_id == "device-generated"
    assert profile.device.registered is True
    assert profile.device.mode == "server_frontend"


def test_profile_advertises_client_private_server_frontend_limitation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="server_frontend",
        device_id="frontend-device",
        device_name="Browser session",
    )
    dataset = store.get_dataset(profile.active_dataset_id or "")
    assert dataset is not None
    private_dataset = replace(dataset, encryption_policy="client_private_v1")
    monkeypatch.setattr(
        store,
        "list_datasets_for_user",
        lambda user_id: [private_dataset] if user_id == "user-1" else [],
    )

    status = service.profile(user_id="user-1", device_id="frontend-device")

    assert status.dataset is not None
    assert status.dataset.server_frontend_mutation_enabled is False
    assert status.dataset.server_frontend_mutation_blockers == [
        CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
    ]
    assert {
        item.domain: item.server_frontend_mutation_enabled
        for item in status.domain_status
    } == dict.fromkeys(M1_SYNC_DOMAINS, False)
    assert {
        item.domain: item.server_frontend_mutation_blockers
        for item in status.domain_status
    } == dict.fromkeys(
        M1_SYNC_DOMAINS,
        [CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE],
    )
    assert any(
        warning.get("code") == CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE
        for warning in status.warnings
    )


def test_bootstrap_without_device_id_and_profile_id_generates_device(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)

    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_name="Laptop",
    )

    assert profile.profile_bootstrapped is True
    assert profile.dataset is not None
    assert profile.device is not None
    assert profile.device.device_id == "device-generated"
    assert profile.device.registered is True
    assert profile.device.client_profile_id is None
    devices = store.list_devices_for_user("user-1")
    assert len(devices) == 1
    assert devices[0].device_id == "device-generated"
    assert devices[0].capabilities["client_profile_id"] is None
    assert len(store.list_datasets_for_user("user-1")) == 1


def test_bootstrap_without_device_id_reuses_device_by_client_profile_id(
    tmp_path: Path,
) -> None:
    issued: list[str] = []

    def _id_factory(prefix: str) -> str:
        value = f"{prefix}-{len(issued) + 1}"
        issued.append(value)
        return value

    service, store = _service(tmp_path, id_factory=_id_factory)

    first = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_name="Laptop",
        client_profile_id="profile-1",
    )
    second = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_name="Laptop",
        client_profile_id="profile-1",
    )

    assert first.device is not None
    assert second.device is not None
    assert second.device.device_id == first.device.device_id
    assert [device.device_id for device in store.list_devices_for_user("user-1")] == [
        first.device.device_id
    ]


def test_profile_status_reports_profile_and_per_domain_apply_health(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    bootstrapped = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        device_name="Laptop",
        client_profile_id="profile-1",
    )
    assert bootstrapped.dataset is not None
    dataset_id = bootstrapped.dataset.dataset_id

    note = store.insert_envelope(_note_envelope(dataset_id=dataset_id))
    message = store.insert_envelope(
        _note_envelope(
            dataset_id=dataset_id,
            client_envelope_id="env-message-1",
            domain="chat.message",
            operation="append",
            object_id="message-1",
            parent_id="conversation-1",
            client_sequence=2,
            payload={"role": "user"},
            payload_hash="sha256:message-1",
        )
    )
    store.mark_envelope_apply_status(
        message.server_cursor,
        apply_status="failed",
        apply_error_code="projection_failed",
        apply_error_message="projection failed",
    )
    store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-1",
            dataset_id=dataset_id,
            domain="chat.message",
            object_id="message-1",
            conflict_type="message_hash_mismatch",
            server_cursor=message.server_cursor,
        )
    )

    profile = service.profile_status(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="device-1",
    )
    domains = {item.domain: item for item in profile.domain_status}

    assert profile.profile_bootstrapped is True
    assert profile.server_cursor == message.server_cursor
    assert domains["notes.note"].envelope_count == 1
    assert domains["notes.note"].pending_apply_count == 1
    assert domains["notes.note"].failed_apply_count == 0
    assert domains["notes.note"].last_apply_status == "pending"
    assert domains["notes.note"].last_apply_result["server_cursor"] == note.server_cursor
    assert domains["chat.message"].envelope_count == 1
    assert domains["chat.message"].failed_apply_count == 1
    assert domains["chat.message"].unresolved_conflicts == 1
    assert domains["chat.message"].last_apply_status == "failed"
    assert domains["chat.message"].last_apply_result["error_code"] == "projection_failed"


def test_profile_status_uses_aggregates_beyond_scan_limit(tmp_path: Path) -> None:
    service, store = _service(tmp_path, scan_limit=2)
    bootstrapped = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        device_name="Laptop",
        client_profile_id="profile-1",
    )
    assert bootstrapped.dataset is not None
    dataset_id = bootstrapped.dataset.dataset_id
    cursors: list[int] = []
    for index in range(1, 6):
        envelope = store.insert_envelope(
            _note_envelope(
                dataset_id=dataset_id,
                client_envelope_id=f"env-note-{index}",
                object_id=f"note-{index}",
                client_sequence=index,
                payload={"title": f"Note {index}"},
                payload_hash=f"sha256:note-{index}",
            )
        )
        cursors.append(envelope.server_cursor)
        if index <= 3:
            store.mark_envelope_apply_status(
                envelope.server_cursor,
                apply_status="failed",
                apply_error_code="projection_failed",
                apply_error_message=f"projection failed {index}",
            )

    profile = service.profile_status(
        user_id="user-1",
        dataset_id=dataset_id,
        device_id="device-1",
    )
    domains = {item.domain: item for item in profile.domain_status}

    assert profile.server_cursor == max(cursors)
    assert domains["notes.note"].last_server_cursor == max(cursors)
    assert domains["notes.note"].envelope_count == 5
    assert domains["notes.note"].failed_apply_count == 3
    assert domains["notes.note"].pending_apply_count == 2
    assert domains["notes.note"].last_apply_status == "pending"
    assert domains["notes.note"].last_apply_result["server_cursor"] == max(cursors)


def test_bootstrap_refuses_when_server_trusted_encryption_is_not_ready(tmp_path: Path) -> None:
    service, store = _service(tmp_path, encryption=_not_ready_encryption())

    with pytest.raises(SyncStoreError, match="sync_encryption_attestation_required"):
        service.bootstrap_profile(
            user_id="user-1",
            mode="offline_sync",
            device_id="device-1",
            device_name="Laptop",
        )

    profile = service.profile(user_id="user-1", device_id="device-1")
    assert profile.capabilities.encryption["ready"] is False
    assert profile.warnings[0]["code"] == "sync_encryption_attestation_required"
    assert store.list_datasets_for_user("user-1") == []
    assert store.list_devices_for_user("user-1") == []
