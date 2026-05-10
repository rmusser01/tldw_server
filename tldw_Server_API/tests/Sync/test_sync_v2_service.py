from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    SyncAdapterRegistry,
    StaticSyncAdapter,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncConflictCreate,
    SyncDataset,
    SyncDomain,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _clock() -> str:
    return "2026-05-10T12:00:00+00:00"


@pytest.fixture()
def sync_store(tmp_path: Path) -> SyncV2Store:
    return SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_service.db"))


@pytest.fixture()
def registry() -> SyncAdapterRegistry:
    registry = SyncAdapterRegistry()
    registry.register(StaticSyncAdapter(domain="notes", supported_adapter_versions={1}))
    registry.register(StaticSyncAdapter(domain="chat", supported_adapter_versions={1}))
    registry.register(StaticSyncAdapter(domain="source_cache", supported_adapter_versions={1}))
    return registry


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
        ),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    return service


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes",
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


def test_capabilities_returns_protocol_domains_limits_and_encryption_policies(
    sync_service: SyncV2Service,
):
    capabilities = sync_service.capabilities()

    assert capabilities.protocol_version == 2
    assert capabilities.min_supported_protocol_version == 2
    assert capabilities.supported_domains == ["chat", "notes", "source_cache"]
    assert capabilities.max_batch_size == 10
    assert capabilities.max_envelope_payload_bytes == 1024
    assert capabilities.max_attachment_bytes == 4096
    assert capabilities.encryption_policies == [
        "client_private_v1",
        "server_trusted",
        "shared_workspace_v1",
    ]


def test_device_registration_creates_and_refreshes_same_device(sync_service: SyncV2Service):
    first = sync_service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        client_version="0.1.0",
        capabilities={"domains": ["notes"]},
        device_id="device-1",
    )
    refreshed = sync_service.register_device(
        user_id="user-1",
        display_name="Renamed Laptop",
        client_type="chatbook",
        client_version="0.1.1",
        capabilities={"domains": ["notes", "chat"]},
        device_id="device-1",
    )

    assert refreshed.device.device_id == first.device.device_id
    assert refreshed.device.registered_at == first.device.registered_at
    assert refreshed.device.display_name == "Renamed Laptop"
    assert refreshed.device.client_version == "0.1.1"
    assert refreshed.device.capabilities == {"domains": ["notes", "chat"]}


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


def test_dataset_enrollment_creates_personal_dataset_by_default(sync_service: SyncV2Service):
    enrolled = sync_service.enroll_dataset(user_id="user-1")

    assert enrolled.dataset.dataset_id == "dataset-generated"
    assert enrolled.dataset.owner_user_id == "user-1"
    assert enrolled.dataset.scope_type == "personal"
    assert enrolled.dataset.encryption_policy == "client_private_v1"
    assert enrolled.dataset.domains == ["chat", "notes", "source_cache"]
    assert enrolled.key_setup_required is True


def test_dataset_enrollment_rejects_cross_user_dataset_takeover(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="shared-dataset")

    with pytest.raises(SyncStoreError):
        sync_service.enroll_dataset(user_id="user-2", dataset_id="shared-dataset")


def test_adapter_registry_accepts_known_domains_and_rejects_unknown_domains(
    registry: SyncAdapterRegistry,
):
    assert registry.get("notes").domain == "notes"

    with pytest.raises(KeyError):
        registry.get("media")

    with pytest.raises(ValueError):
        registry.register(
            StaticSyncAdapter(domain=cast(SyncDomain, "bogus"), supported_adapter_versions={1})
        )


def test_push_supports_legacy_adapter_without_context_keyword(sync_store: SyncV2Store):
    class LegacyNotesAdapter:
        domain: SyncDomain = "notes"
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope()],
    )

    assert result.accepted[0].client_envelope_id == "env-1"
    assert adapter.dataset_id == "dataset-1"


def test_push_rejects_envelopes_for_datasets_user_cannot_access(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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


def test_push_returns_per_envelope_accepted_rejected_and_conflict_outcomes(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(StaticSyncAdapter(domain="notes", supported_adapter_versions={1}))
    registry.register(
        StaticSyncAdapter(
            domain="chat",
            supported_adapter_versions={1},
            outcomes={
                "env-rejected": AdapterRejected(
                    client_envelope_id="env-rejected",
                    error_code="domain_validation_failed",
                    message="invalid chat shape",
                ),
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="chat",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes", "chat"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(client_envelope_id="env-accepted"),
            _envelope(
                client_envelope_id="env-rejected",
                domain="chat",
                entity_id="conversation-1",
                stable_key="chat:conversation-1",
                payload_hash="sha256:chat-rejected",
            ),
            _envelope(
                client_envelope_id="env-conflict",
                domain="chat",
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
    assert result.next_cursor == str(result.accepted[0].server_sequence)


def test_push_rejects_unsupported_adapter_versions_per_envelope(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

    result = sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(adapter_version=2)],
    )

    assert result.accepted == []
    assert result.rejected[0].client_envelope_id == "env-1"
    assert result.rejected[0].error_code == "unsupported_adapter_version"


def test_push_rejects_mismatched_device_id_and_fills_missing_device_id(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

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
        domains=["notes"],
        include_own_changes=False,
    )

    assert spoofed.accepted == []
    assert spoofed.rejected[0].error_code == "device_mismatch"
    assert [item.client_envelope_id for item in accepted.accepted] == ["env-no-device"]
    assert same_device_pull.envelopes == []


def test_push_rejects_envelope_dataset_mismatch_before_persistence(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
    _register_devices(sync_service, "user-2", "user-2-device")
    sync_service.enroll_dataset(user_id="user-2", dataset_id="dataset-2", domains=["notes"])

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
        domains=["notes"],
        include_own_changes=True,
    )

    assert result.accepted == []
    assert result.conflicts == []
    assert result.rejected[0].client_envelope_id == "cross-dataset"
    assert result.rejected[0].error_code == "dataset_mismatch"
    assert leaked.envelopes == []


def test_push_rejects_envelopes_beyond_batch_limit(sync_store: SyncV2Store, registry: SyncAdapterRegistry):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(max_batch_size=2),
    )
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

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
    registry.register(StaticSyncAdapter(domain="notes", supported_adapter_versions={1}))
    registry.register(
        StaticSyncAdapter(
            domain="chat",
            supported_adapter_versions={1},
            outcomes={
                "unenrolled-conflict": AdapterConflict(
                    client_envelope_id="unenrolled-conflict",
                    domain="chat",
                    entity_id="conversation-1",
                    conflict_type="version_divergence",
                )
            },
        )
    )
    service = SyncV2Service(store=sync_store, adapters=registry, clock=_clock)
    _register_devices(service, "user-1", "device-1")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

    result = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="unenrolled-conflict",
                domain="chat",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

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
        domains=["notes"],
        encryption_policy="server_trusted",
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


def test_conflict_push_retry_reuses_existing_unresolved_conflict(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

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
    assert retried.conflicts[0].conflict_id == first.conflicts[0].conflict_id
    assert retried.conflicts[0].server_sequence == first.conflicts[0].server_sequence
    assert len(sync_store.list_conflicts("dataset-1", status="unresolved")) == 1
    assert manifest.datasets[0].unresolved_conflicts == 1


def test_conflict_push_rejects_idempotency_drift_without_aborting_batch(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

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
            domain="notes",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )
    conflict_id = pushed.conflicts[0].conflict_id

    resolved = service.resolve_conflict(
        user_id="user-1",
        conflict_id=conflict_id,
        action="merge",
        resolved_by_device_id="device-1",
        resolution_envelope=_envelope(
            client_envelope_id="env-resolution",
            operation="resolve_conflict",
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
    assert resolved.resolved_by_envelope_id == "env-resolution"
    assert resolved.resolution_action == "merge"
    assert [envelope.client_envelope_id for envelope in pulled.envelopes] == ["env-resolution"]
    assert pulled.envelopes[0].status == "accepted"


def test_device_scoped_operations_require_registered_user_device(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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
            action="dismiss",
            resolved_by_device_id="unregistered-device",
        )
    with pytest.raises(SyncStoreError, match="Sync device was not found"):
        service.resolve_conflict(
            user_id="user-1",
            conflict_id=conflict_id,
            action="merge",
            resolution_envelope=_envelope(
                client_envelope_id="env-unregistered-resolution",
                operation="resolve_conflict",
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
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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


def test_list_key_recovery_bundles_rejects_inaccessible_dataset(
    sync_service: SyncV2Service,
):
    with pytest.raises(SyncStoreError, match="Sync dataset was not found"):
        sync_service.list_key_recovery_bundles(
            user_id="user-1",
            dataset_id="missing-dataset",
            key_purpose="dataset_recovery",
        )


def test_resolve_conflict_rejects_invalid_private_resolution_payload(
    sync_store: SyncV2Store,
):
    registry = SyncAdapterRegistry()
    registry.register(
        StaticSyncAdapter(
            domain="notes",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
    pushed = service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[_envelope(client_envelope_id="env-conflict")],
    )

    with pytest.raises(SyncStoreError, match="private payload validation failed"):
        service.resolve_conflict(
            user_id="user-1",
            conflict_id=pushed.conflicts[0].conflict_id,
            action="merge",
            resolved_by_device_id="device-1",
            resolution_envelope=_envelope(
                client_envelope_id="env-invalid-resolution",
                operation="resolve_conflict",
                payload_ciphertext=None,
                payload_clear={"body": "known plaintext"},
                payload_hash="sha256:invalid-resolution",
            ),
        )


def test_pull_uses_stable_server_cursor(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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


def test_pull_does_not_persist_empty_explicit_high_cursor(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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


def test_pull_does_not_persist_visible_empty_explicit_cursor_over_echoes(
    sync_service: SyncV2Service,
):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

    with pytest.raises(SyncStoreError, match="Invalid sync cursor"):
        sync_service.pull(
            user_id="user-1",
            dataset_id="dataset-1",
            device_id="device-1",
            cursor="not-a-cursor",
        )


def test_pull_rejects_non_positive_page_size(sync_service: SyncV2Service):
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])

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
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes", "chat"])
    sync_service.push(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        envelopes=[
            _envelope(
                client_envelope_id="chat-before-notes",
                domain="chat",
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
        domains=["notes"],
        include_own_changes=True,
    )

    multi_domain = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-2",
        domains=["notes", "chat"],
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
    sync_service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes", "chat"])
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
                domain="chat",
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
        domains=["notes"],
        page_size=1,
        include_own_changes=False,
    )
    next_page = sync_service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor=page.next_cursor,
        domains=["notes"],
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
            domain="notes",
            supported_adapter_versions={1},
            outcomes={
                "env-conflict": AdapterConflict(
                    client_envelope_id="env-conflict",
                    domain="notes",
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
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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
        domains=["notes"],
        include_own_changes=True,
    )

    assert [conflict.client_envelope_id for conflict in push_result.conflicts] == ["env-conflict"]
    assert pull_result.envelopes == []


def test_pull_scans_past_echo_filled_raw_window_before_remote_change(
    sync_store: SyncV2Store,
    registry: SyncAdapterRegistry,
):
    service = SyncV2Service(
        store=sync_store,
        adapters=registry,
        clock=_clock,
        id_factory=lambda prefix: f"{prefix}-generated",
        settings=SyncV2Settings(max_batch_size=20, max_pull_page_size=1),
    )
    _register_devices(service, "user-1", "device-1", "device-2")
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes"])
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

    page = service.pull(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        cursor="0",
        domains=["notes"],
        page_size=1,
        include_own_changes=False,
    )

    assert [envelope.client_envelope_id for envelope in page.envelopes] == [
        "remote-after-echoes"
    ]
    assert page.next_cursor == "12"
    assert page.has_more is False


def test_restore_manifest_is_metadata_only_and_includes_inventory_status(
    sync_service: SyncV2Service,
    sync_store: SyncV2Store,
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
        domains=["notes", "source_cache"],
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
                domain="source_cache",
                entity_id="source-1",
                stable_key="source:1",
                payload_hash="sha256:source",
                payload_clear={
                    "attachment_id": "attachment-1",
                    "availability": "available",
                    "size_bytes": 512,
                },
                payload_size_bytes=512,
            ),
        ],
    )
    sync_store.insert_conflict(
        SyncConflictCreate(
            conflict_id="conflict-1",
            dataset_id="dataset-1",
            domain="notes",
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

    manifest = sync_service.restore_manifest(user_id="user-1")

    assert manifest.generated_at == _clock()
    assert manifest.devices[0].device_id == "device-1"
    assert manifest.devices[0].last_seen_at is not None
    assert manifest.datasets[0].dataset_id == "dataset-1"
    assert manifest.datasets[0].encryption_policy == "client_private_v1"
    assert manifest.datasets[0].metadata == {}
    assert manifest.datasets[0].approximate_counts == {"notes": 1, "source_cache": 1}
    assert manifest.datasets[0].unresolved_conflicts == 1
    assert manifest.datasets[0].attachment_availability == {"available": 1}
    assert manifest.datasets[0].attachment_size_classes == {"small": 1}
    assert manifest.datasets[0].key_recovery_available is True
    assert "known private label" not in repr(manifest)
    assert "ciphertext:known-private-note" not in repr(manifest)
    assert "wrapped:secret-key" not in repr(manifest)
