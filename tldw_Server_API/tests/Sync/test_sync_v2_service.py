from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from tldw_Server_API.app.core.DB_Management import Sync_DB as sync_db_module
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterConflict,
    AdapterRejected,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import MaterializationResult
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncConflictCreate,
    SyncDataset,
    SyncDomain,
    SyncEnvelopeCreate,
    SyncKeyRecordCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


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
    ]
    assert capabilities.max_batch_size == 10
    assert capabilities.max_envelope_payload_bytes == 1024
    assert capabilities.max_attachment_bytes == 4096
    assert capabilities.encryption_policies == ["server_trusted_v1"]
    assert capabilities.supports_attachments is False


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
    assert resolved.server_cursor == pulled.envelopes[0].server_cursor
    assert resolved.resolution_action == "overwrite"
    assert [envelope.client_envelope_id for envelope in pulled.envelopes] == ["env-resolution"]
    assert pulled.envelopes[0].status == "accepted"


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
    assert resolved.server_cursor == pulled.envelopes[0].server_cursor
    assert resolved.server_cursor != original_conflict_cursor


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
    resolution_envelope = next(
        item for item in envelopes if item.client_envelope_id == f"env-resolution-{expected_apply_status}"
    )

    assert conflict.status == "unresolved"
    assert conflict.resolution_action is None
    assert resolution_envelope.status == "accepted"
    assert resolution_envelope.apply_status == expected_apply_status


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
    monkeypatch: pytest.MonkeyPatch,
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
    original_insert = sync_store.insert_envelope

    def insert_and_mark_claim(envelope: SyncEnvelopeCreate):
        inserted = original_insert(envelope)
        sync_store.db.execute(
            """
            UPDATE sync_conflicts
               SET resolution_action = ?,
                   resolved_by_device_id = ?,
                   resolution_notes = ?
             WHERE conflict_id = ?
            """,
            ("overwrite", "device-1", "cleanup claim", conflict_id),
        )
        return inserted

    monkeypatch.setattr(sync_store, "insert_envelope", insert_and_mark_claim)

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
    resolution_envelope = next(
        item for item in envelopes if item.client_envelope_id == f"env-cleanup-{expected_apply_status}"
    )

    assert conflict.status == "unresolved"
    assert conflict.resolution_action is None
    assert conflict.resolved_by_device_id is None
    assert conflict.resolution_notes is None
    assert conflict.resolved_by_envelope_id is None
    assert resolution_envelope.status == "accepted"
    assert resolution_envelope.apply_status == expected_apply_status


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
    assert [(item.envelope_id, item.status, item.apply_status) for item in after] == [
        (item.envelope_id, item.status, item.apply_status) for item in before
    ]


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
