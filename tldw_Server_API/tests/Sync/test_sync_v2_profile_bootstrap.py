from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.sync_v2_models import SyncProfileBootstrapRequest
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AttachmentRefAdapter,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
    attachment_ref_v2_object_hash,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import (
    CLIENT_PRIVATE_SERVER_FRONTEND_LIMITATION_CODE,
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    SyncConflictCreate,
    SyncDataset,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit


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
    dataset_bootstrapper=None,
    notes_attachment_bootstrapper=None,
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
        dataset_bootstrapper=dataset_bootstrapper,
        notes_attachment_bootstrapper=notes_attachment_bootstrapper,
    )
    return service, store


class _PausedOrganizationBootstrapper:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def bootstrap(
        self,
        *,
        service: SyncV2Service,
        user_id: str,
        dataset: SyncDataset,
    ) -> SyncDataset:
        self.calls.append((user_id, dataset.dataset_id))
        return service.store.get_dataset(dataset.dataset_id) or dataset


class _PausedAttachmentBootstrapper(_PausedOrganizationBootstrapper):
    def dry_run(self, *, service: SyncV2Service, user_id: str):
        del service
        self.calls.append((user_id, "dry-run"))
        return {
            "candidate_count": 1_000,
            "candidate_count_is_lower_bound": True,
            "error_code": None,
        }


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


def _attachment_v2_envelope(*, dataset_id: str) -> SyncEnvelopeCreate:
    payload = {
        "attachment_id": "11111111-1111-4111-8111-111111111111",
        "parent_domain": "notes.note",
        "parent_object_id": "22222222-2222-4222-8222-222222222222",
        "file_name": "diagram.png",
        "original_file_name": "diagram.png",
        "content_type": "image/png",
        "size_bytes": 42,
        "blob_hash": "sha256:" + "a" * 64,
        "created_at": _clock(),
        "last_modified": _clock(),
        "created_by": "device-1",
    }
    return SyncEnvelopeCreate(
        dataset_id=dataset_id,
        client_envelope_id="env-attachment-v2",
        domain="attachment.ref",
        operation="upsert",
        object_id=payload["attachment_id"],
        device_id="device-1",
        client_sequence=1,
        schema_version=2,
        adapter_version=2,
        object_revision=1,
        payload=payload,
        payload_hash=attachment_ref_v2_object_hash(
            "upsert",
            payload,
            object_revision=1,
        ),
        created_at_client=_clock(),
        encryption_metadata={"policy": "server_trusted_v1"},
    )


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


def test_profile_capabilities_are_bound_to_active_ready_dataset(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    service.settings = replace(service.settings, supports_attachments=True)
    service.adapters.register(AttachmentRefAdapter(v2_writes_enabled=True))
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-ready",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note", "attachment.ref"],
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_attachment_v2": {
                    "bootstrap_id": "bootstrap-ready",
                    "state": "ready",
                    "target_adapter_version": 2,
                    "captured_count": 0,
                    "expected_count": 0,
                    "source_hash": "e3b0c44298fc1c149afbf4c8996fb924"
                    "27ae41e4649b934ca495991b7852b855",
                    "source_cursor": None,
                    "error_code": None,
                },
            },
        )
    )

    profile = service.profile(user_id="user-1")

    assert profile.active_dataset_id == "dataset-ready"
    assert profile.capabilities.writable_adapter_versions["notes.note"] == [1]
    assert profile.capabilities.writable_adapter_versions["attachment.ref"] == [2]

    bootstrap = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        requested_domains=["notes.note", "attachment.ref"],
    )
    assert bootstrap.capabilities.writable_adapter_versions["attachment.ref"] == [2]


def test_profile_bootstrap_begins_and_resumes_attachment_capture(
    tmp_path: Path,
) -> None:
    bootstrapper = _PausedAttachmentBootstrapper()
    service, store = _service(
        tmp_path,
        notes_attachment_bootstrapper=bootstrapper,
    )

    result = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        requested_domains=["notes.note", "attachment.ref"],
        client_instance={
            "supported_adapter_versions": {"attachment.ref": [2]},
        },
    )

    assert bootstrapper.calls == [("user-1", result.active_dataset_id)]
    dataset = store.get_dataset(result.active_dataset_id or "", owner_user_id="user-1")
    assert dataset is not None
    assert dataset.metadata["notes_attachment_v2"]["state"] == "initializing"
    assert dataset.metadata["notes_attachment_v2"]["target_adapter_version"] == 2
    assert result.capabilities.writable_adapter_versions["attachment.ref"] == []


def test_attachment_diagnostic_recovery_action_is_bounded_and_path_free(
    tmp_path: Path,
) -> None:
    bootstrapper = _PausedAttachmentBootstrapper()
    service, store = _service(
        tmp_path,
        notes_attachment_bootstrapper=bootstrapper,
    )
    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        requested_domains=[*M1_SYNC_DOMAINS, "attachment.ref"],
    )
    assert profile.dataset is not None
    dataset_id = profile.dataset.dataset_id
    dataset = store.get_dataset(dataset_id, owner_user_id="user-1")
    assert dataset is not None
    bootstrap_id = dataset.metadata["notes_attachment_v2"]["bootstrap_id"]
    source_key = "notes_attachments/note-1/private-name.pdf"
    mapping = store.resolve_notes_attachment_source_map(
        dataset_id,
        owner_user_id="user-1",
        bootstrap_id=bootstrap_id,
        note_id="note-1",
        source_key=source_key,
    )
    store.record_notes_attachment_cleanup_candidate(
        dataset_id,
        owner_user_id="user-1",
        bootstrap_id=bootstrap_id,
        source_key=source_key,
        source_relative_path=source_key,
        source_blob_hash="sha256:" + "1" * 64,
        source_size_bytes=12,
        source_modified_ns=42,
    )
    store.transition_notes_attachment_bootstrap(
        dataset_id,
        owner_user_id="user-1",
        bootstrap_id=bootstrap_id,
        expected_state="initializing",
        state="initializing",
        captured_count=1,
        expected_count=1,
        source_hash=None,
        source_cursor='{"private":"notes_attachments/note-1/private-name.pdf"}',
    )

    diagnostics = service.notes_attachment_bootstrap_diagnostics(
        user_id="user-1",
        dataset_id=dataset_id,
        sample_limit=1,
        dry_run=False,
    )

    assert diagnostics.state == "initializing"
    assert diagnostics.captured_count == diagnostics.expected_count == 1
    assert diagnostics.cursor is not None
    assert diagnostics.cursor.startswith("sha256:")
    assert diagnostics.cleanup_candidates[0].source_key_hash.startswith("sha256:")
    assert diagnostics.cleanup_candidates[0].attachment_id == mapping.attachment_id
    assert diagnostics.cleanup_candidates[0].state == "captured"
    assert diagnostics.cleanup_candidates[0].blocker_code is None
    assert [action.action for action in diagnostics.recovery_actions] == [
        "bootstrap_resume"
    ]
    serialized = repr(diagnostics)
    assert "private-name.pdf" not in serialized
    assert "notes_attachments" not in serialized
    assert bootstrap_id not in serialized


def test_attachment_bootstrap_dry_run_is_read_only_and_owner_scoped(
    tmp_path: Path,
) -> None:
    bootstrapper = _PausedAttachmentBootstrapper()
    service, store = _service(
        tmp_path,
        notes_attachment_bootstrapper=bootstrapper,
    )

    diagnostics = service.notes_attachment_bootstrap_diagnostics(
        user_id="user-1",
        sample_limit=0,
        dry_run=True,
    )

    assert diagnostics.state == "not_started"
    assert diagnostics.dry_run is True
    assert diagnostics.source_candidate_count == 1_000
    assert diagnostics.source_candidate_count_is_lower_bound is True
    assert store.list_datasets_for_user("user-1") == []
    assert store.list_devices_for_user("user-1") == []
    assert bootstrapper.calls == [("user-1", "dry-run")]

    other = store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="other-dataset",
            owner_user_id="other-user",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
        )
    )
    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        service.notes_attachment_bootstrap_diagnostics(
            user_id="user-1",
            dataset_id=other.dataset_id,
        )


def test_attachment_bootstrap_diagnostics_reject_oversized_samples(
    tmp_path: Path,
) -> None:
    service, _store = _service(tmp_path)

    with pytest.raises(
        SyncStoreError,
        match="sync_attachment_bootstrap_sample_limit_exceeded",
    ):
        service.notes_attachment_bootstrap_diagnostics(
            user_id="user-1",
            sample_limit=101,
        )


def test_moodboard_studio_readiness_diagnostics_are_safe_and_owner_scoped(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    dataset = store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-ready",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_moodboard_v1": {
                    "state": "blocked",
                    "source_cursor": "00000000-0000-4000-8000-000000000101",
                    "source_count": 1,
                    "source_fingerprint": "a" * 64,
                    "reason_code": "notes_moodboard_source_invalid",
                    "resume_phase": "bootstrapping",
                },
                "notes_moodboard_note_v1": {
                    "state": "ready",
                    "source_cursor": (
                        "00000000-0000-4000-8000-000000000101|"
                        "00000000-0000-4000-8000-000000000201"
                    ),
                    "source_count": 2,
                    "source_fingerprint": "b" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "notes_studio_document_v1": {
                    "state": "ready",
                    "source_cursor": "00000000-0000-4000-8000-000000000301",
                    "source_count": 3,
                    "source_fingerprint": "c" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "moodboard_capture_enabled": False,
                "studio_document_capture_enabled": False,
            },
        )
    )

    diagnostics = service._profile_manager().notes_moodboard_studio_readiness_diagnostics(
        user_id="user-1",
        dataset_id=dataset.dataset_id,
    )

    assert diagnostics.moodboard.state == "blocked"
    assert diagnostics.moodboard.source_count == 1
    assert diagnostics.moodboard.resume_phase == "bootstrapping"
    assert diagnostics.moodboard.cursor is not None
    assert diagnostics.moodboard.cursor.startswith("sha256:")
    assert diagnostics.moodboard_note.source_count == 2
    assert diagnostics.studio_document.source_count == 3
    assert diagnostics.moodboard_capture_enabled is False
    assert diagnostics.studio_document_capture_enabled is False
    serialized = repr(diagnostics)
    assert "00000000-0000-4000-8000-000000000101" not in serialized
    assert "00000000-0000-4000-8000-000000000201" not in serialized

    with pytest.raises(SyncStoreError, match="not found or is not accessible"):
        service._profile_manager().notes_moodboard_studio_readiness_diagnostics(
            user_id="other-user",
            dataset_id=dataset.dataset_id,
        )


def test_malformed_moodboard_studio_metadata_reports_stable_blocked_diagnostics(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    dataset = store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-ready",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_moodboard_v1": {
                    "state": "ready",
                    "source_cursor": "private board name",
                    "source_count": "many",
                    "source_fingerprint": "not-a-hash",
                    "reason_code": "private note text",
                    "resume_phase": None,
                },
            },
        )
    )

    diagnostics = service._profile_manager().notes_moodboard_studio_readiness_diagnostics(
        user_id="user-1",
        dataset_id=dataset.dataset_id,
    )

    assert diagnostics.moodboard.state == "blocked"
    assert diagnostics.moodboard.reason_code == "notes_moodboard_readiness_state_invalid"
    assert diagnostics.moodboard.cursor is None
    assert "private board name" not in repr(diagnostics)
    assert "private note text" not in repr(diagnostics)


def test_forged_ready_moodboard_studio_metadata_does_not_expose_capabilities(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    dormant = {
        "notes.moodboard",
        "notes.moodboard_note",
        "notes.studio_document",
    }
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-ready",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_moodboard_v1": {
                    "state": "ready",
                    "source_cursor": "00000000-0000-4000-8000-000000000101",
                    "source_count": 1,
                    "source_fingerprint": "a" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "notes_moodboard_note_v1": {
                    "state": "ready",
                    "source_cursor": (
                        "00000000-0000-4000-8000-000000000101|"
                        "00000000-0000-4000-8000-000000000201"
                    ),
                    "source_count": 1,
                    "source_fingerprint": "b" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "notes_studio_document_v1": {
                    "state": "ready",
                    "source_cursor": "00000000-0000-4000-8000-000000000301",
                    "source_count": 1,
                    "source_fingerprint": "c" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "moodboard_capture_enabled": False,
                "studio_document_capture_enabled": False,
            },
        )
    )

    profile = service.profile(user_id="user-1")

    assert dormant.isdisjoint(profile.capabilities.supported_domains)
    assert dormant.isdisjoint(profile.capabilities.operations)
    assert dormant.isdisjoint(profile.capabilities.domain_schemas)
    assert dormant.isdisjoint(profile.capabilities.supported_adapter_versions)
    assert dormant.isdisjoint(profile.capabilities.writable_adapter_versions)


def test_public_enrollment_and_manifest_redact_moodboard_studio_readiness(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    internal_metadata = {
        "notes_moodboard_v1": {
            "state": "ready",
            "source_cursor": "00000000-0000-4000-8000-000000000101",
            "source_count": 1,
            "source_fingerprint": "a" * 64,
            "reason_code": None,
            "resume_phase": None,
        },
        "notes_moodboard_note_v1": {
            "state": "ready",
            "source_cursor": (
                "00000000-0000-4000-8000-000000000101|"
                "00000000-0000-4000-8000-000000000201"
            ),
            "source_count": 1,
            "source_fingerprint": "b" * 64,
            "reason_code": None,
            "resume_phase": None,
        },
        "notes_studio_document_v1": {
            "state": "blocked",
            "source_cursor": "00000000-0000-4000-8000-000000000301",
            "source_count": 1,
            "source_fingerprint": "c" * 64,
            "reason_code": "notes_studio_document_source_invalid",
            "resume_phase": "bootstrapping",
        },
        "moodboard_capture_enabled": False,
        "studio_document_capture_enabled": False,
    }
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-internal-moodboard-studio-readiness",
            owner_user_id="user-1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={"label": "before", **internal_metadata},
        )
    )

    enrollment = service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-internal-moodboard-studio-readiness",
        metadata={"label": "after"},
    )
    manifest = service.restore_manifest(user_id="user-1")
    stored = store.get_dataset(
        "dataset-internal-moodboard-studio-readiness",
        owner_user_id="user-1",
    )

    assert enrollment.dataset.metadata == {"label": "after"}
    assert manifest.datasets[0].metadata == {"label": "after"}
    assert stored is not None
    assert stored.metadata == {"label": "after", **internal_metadata}


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


def test_bootstrap_persists_canonical_adapter_map_used_by_push(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    service.adapters.register(AttachmentRefAdapter(v2_writes_enabled=False))
    request = SyncProfileBootstrapRequest.model_validate(
        {
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": ["attachment.ref"],
            "supported_adapter_versions": {"attachment.ref": [2]},
            "client_instance": {"app_version": "0.4.0", "platform": "macos"},
        }
    )

    profile = service.bootstrap_profile(
        user_id="user-1",
        mode=request.mode,
        device_id=request.device_id,
        client_instance=request.client_instance,
        requested_domains=request.requested_domains,
    )

    stored = store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "attachment.ref": [2]
    }
    assert stored.capabilities["client_instance"] == {
        "app_version": "0.4.0",
        "platform": "macos",
    }
    assert profile.active_dataset_id is not None

    result = service.push(
        user_id="user-1",
        dataset_id=profile.active_dataset_id,
        device_id="device-1",
        envelopes=[_attachment_v2_envelope(dataset_id=profile.active_dataset_id)],
    )

    assert result.accepted == []
    assert result.rejected[0].error_code == "attachment_ref_v2_not_writable"


def test_repeat_profile_bootstrap_cannot_downgrade_active_adapter_versions(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    first = SyncProfileBootstrapRequest.model_validate(
        {
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
        }
    )
    service.bootstrap_profile(
        user_id="user-1",
        mode=first.mode,
        device_id=first.device_id,
        client_instance=first.client_instance,
        requested_domains=first.requested_domains,
    )
    downgrade = SyncProfileBootstrapRequest.model_validate(
        {
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1]},
        }
    )

    with pytest.raises(SyncStoreError, match="cannot remove active versions"):
        service.bootstrap_profile(
            user_id="user-1",
            mode=downgrade.mode,
            device_id=downgrade.device_id,
            client_instance=downgrade.client_instance,
            requested_domains=downgrade.requested_domains,
        )

    stored = store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "notes.note": [1, 2]
    }


def test_bootstrap_adapter_map_omission_remains_v1_only(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    service.adapters.register(AttachmentRefAdapter(v2_writes_enabled=True))
    request = SyncProfileBootstrapRequest.model_validate(
        {
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": ["attachment.ref"],
            "client_instance": {"platform": "macos"},
        }
    )

    profile = service.bootstrap_profile(
        user_id="user-1",
        mode=request.mode,
        device_id=request.device_id,
        client_instance=request.client_instance,
        requested_domains=request.requested_domains,
    )

    stored = store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "attachment.ref": [1]
    }
    assert profile.active_dataset_id is not None
    result = service.push(
        user_id="user-1",
        dataset_id=profile.active_dataset_id,
        device_id="device-1",
        envelopes=[_attachment_v2_envelope(dataset_id=profile.active_dataset_id)],
    )

    assert result.accepted == []
    assert result.rejected[0].error_code == "device_adapter_version_not_advertised"


def test_partial_bootstrap_adapter_map_defaults_omitted_domain_for_push(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    request = SyncProfileBootstrapRequest.model_validate(
        {
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": ["notes.note", "attachment.ref"],
            "supported_adapter_versions": {"attachment.ref": [2]},
        }
    )

    profile = service.bootstrap_profile(
        user_id="user-1",
        mode=request.mode,
        device_id=request.device_id,
        client_instance=request.client_instance,
        requested_domains=request.requested_domains,
    )

    stored = store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "notes.note": [1],
        "attachment.ref": [2],
    }
    assert profile.active_dataset_id is not None
    result = service.push(
        user_id="user-1",
        dataset_id=profile.active_dataset_id,
        device_id="device-1",
        envelopes=[
            _note_envelope(
                dataset_id=profile.active_dataset_id,
                device_id="device-1",
            )
        ],
    )

    assert len(result.accepted) == 1
    assert result.rejected == []


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
    assert domains["notes.note"].repair_status == {
        "status": "repair_needed",
        "pending_apply_count": 1,
        "failed_apply_count": 0,
    }
    assert domains["notes.note"].last_apply_status == "pending"
    assert domains["notes.note"].last_apply_result["server_cursor"] == note.server_cursor
    assert domains["chat.message"].envelope_count == 1
    assert domains["chat.message"].failed_apply_count == 1
    assert domains["chat.message"].unresolved_conflicts == 1
    assert domains["chat.message"].last_apply_status == "failed"
    assert domains["chat.message"].last_apply_result["error_code"] == "projection_failed"


def test_profile_status_uses_aggregates_beyond_scan_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    def fail_full_envelope_scan(*_args, **_kwargs):
        raise AssertionError("profile status should use aggregate envelope queries")

    monkeypatch.setattr(store, "list_envelopes_after", fail_full_envelope_scan)

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


def test_notes_organization_subset_enrollment_is_rejected_without_mutation(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)

    with pytest.raises(
        SyncStoreError,
        match="notes_organization_sync_domains_incomplete",
    ):
        service.bootstrap_profile(
            user_id="user-1",
            mode="offline_sync",
            device_id="device-1",
            requested_domains=["notes.keyword"],
        )

    assert store.list_datasets_for_user("user-1") == []
    assert store.list_devices_for_user("user-1") == []


def test_notes_organization_full_enrollment_is_atomic_initializing_and_safe(
    tmp_path: Path,
) -> None:
    bootstrapper = _PausedOrganizationBootstrapper()
    service, store = _service(tmp_path, dataset_bootstrapper=bootstrapper)
    requested = [*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]

    first = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        requested_domains=requested,
    )
    second = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        requested_domains=requested,
    )

    assert first.dataset is not None
    assert first.dataset.domains == requested
    assert first.dataset.notes_organization == {
        "state": "initializing",
        "captured_count": 0,
        "expected_count": 0,
        "error_code": None,
    }
    assert second.dataset is not None
    assert second.dataset.notes_organization == first.dataset.notes_organization
    assert bootstrapper.calls == [
        ("user-1", first.dataset.dataset_id),
        ("user-1", first.dataset.dataset_id),
    ]
    stored = store.get_dataset(first.dataset.dataset_id)
    assert stored is not None
    metadata = stored.metadata["notes_organization_v1"]
    assert metadata["state"] == "initializing"
    assert isinstance(metadata["bootstrap_id"], str)
    assert metadata["bootstrap_id"]
    assert not hasattr(first.dataset, "bootstrap_id")
    device = store.get_device("user-1", "device-1")
    assert device is not None
    assert device.capabilities["requested_domains"] == requested


def test_notes_organization_failed_profile_exposes_only_safe_summary(
    tmp_path: Path,
) -> None:
    bootstrapper = _PausedOrganizationBootstrapper()
    service, store = _service(tmp_path, dataset_bootstrapper=bootstrapper)
    requested = [*M1_SYNC_DOMAINS, *NOTES_ORGANIZATION_DOMAINS]
    profile = service.bootstrap_profile(
        user_id="user-1",
        mode="offline_sync",
        device_id="device-1",
        requested_domains=requested,
    )
    assert profile.dataset is not None
    stored = store.get_dataset(profile.dataset.dataset_id)
    assert stored is not None
    bootstrap_id = stored.metadata["notes_organization_v1"]["bootstrap_id"]

    store.transition_notes_organization_bootstrap(
        stored.dataset_id,
        bootstrap_id=bootstrap_id,
        expected_state="initializing",
        state="failed",
        captured_count=3,
        expected_count=4,
        error_code="notes_organization_bootstrap_source_invalid",
    )
    failed = service.profile_status(
        user_id="user-1",
        dataset_id=stored.dataset_id,
        device_id="device-1",
    )

    assert failed.dataset is not None
    assert failed.dataset.notes_organization == {
        "state": "failed",
        "captured_count": 3,
        "expected_count": 4,
        "error_code": "notes_organization_bootstrap_source_invalid",
    }
    assert "bootstrap_id" not in failed.dataset.notes_organization


def test_dormant_task_domain_readiness_diagnostics_are_owner_scoped_and_sanitized(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    private_cursor = "00000000-0000-4000-8000-000000000001"
    private_activity_cursor = (
        "2026-08-13T00:00:00+00:00|00000000-0000-4000-8000-000000000011"
    )
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-task-readiness",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_task_v1": {
                    "state": "blocked",
                    "source_cursor": private_cursor,
                    "source_count": 3,
                    "source_fingerprint": "a" * 64,
                    "reason_code": "notes_task_source_invalid",
                    "resume_phase": "bootstrapping",
                },
                "notes_task_activity_v1": {
                    "state": "ready",
                    "source_cursor": private_activity_cursor,
                    "source_count": 5,
                    "source_fingerprint": "b" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                },
                "task_activity_capture_enabled": True,
            },
        )
    )

    diagnostics = service._profile_manager().notes_task_readiness_diagnostics(
        user_id="user-1",
        dataset_id="dataset-task-readiness",
    )

    assert diagnostics.task.state == "blocked"
    assert diagnostics.task.source_count == 3
    assert diagnostics.task.cursor == "sha256:" + hashlib.sha256(
        private_cursor.encode("utf-8")
    ).hexdigest()
    assert diagnostics.task.source_fingerprint == "a" * 64
    assert diagnostics.task.reason_code == "notes_task_source_invalid"
    assert diagnostics.task_activity.state == "ready"
    assert diagnostics.task_activity.source_count == 5
    assert diagnostics.task_activity.cursor == "sha256:" + hashlib.sha256(
        private_activity_cursor.encode("utf-8")
    ).hexdigest()
    assert diagnostics.task_activity_capture_enabled is True
    serialized = repr(diagnostics)
    for secret in (
        private_cursor,
        private_activity_cursor,
    ):
        assert secret not in serialized

    with pytest.raises(
        SyncStoreError,
        match="Sync dataset was not found or is not accessible",
    ):
        service._profile_manager().notes_task_readiness_diagnostics(
            user_id="other-user",
            dataset_id="dataset-task-readiness",
        )


def test_dormant_task_domain_malformed_readiness_fails_closed(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-malformed-readiness",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "notes_task_v1": {
                    "state": "ready",
                    "source_cursor": ["private", "cursor"],
                    "source_count": True,
                    "source_fingerprint": "not-a-hash",
                    "reason_code": "private detail",
                    "resume_phase": None,
                },
                "notes_task_activity_v1": None,
                "task_activity_capture_enabled": "yes",
            },
        )
    )

    diagnostics = service._profile_manager().notes_task_readiness_diagnostics(
        user_id="user-1",
        dataset_id="dataset-malformed-readiness",
    )

    assert diagnostics.task.state == "blocked"
    assert diagnostics.task.source_count == 0
    assert diagnostics.task.cursor is None
    assert diagnostics.task.source_fingerprint is None
    assert diagnostics.task.reason_code == "notes_task_readiness_state_invalid"
    assert diagnostics.task_activity.state == "blocked"
    assert diagnostics.task_activity.reason_code == (
        "notes_task_activity_readiness_state_invalid"
    )
    assert diagnostics.task_activity_capture_enabled is False
    assert "private detail" not in repr(diagnostics)


@pytest.mark.parametrize(
    ("readiness_key", "raw", "reason_code"),
    [
        ("notes_task_v1", None, "notes_task_readiness_state_invalid"),
        ("notes_task_v1", [], "notes_task_readiness_state_invalid"),
        ("notes_task_v1", {}, "notes_task_readiness_state_invalid"),
        (
            "notes_task_v1",
            {
                "state": "not_enrolled",
                "source_cursor": None,
                "source_count": 0,
                "source_fingerprint": None,
                "reason_code": None,
                "resume_phase": None,
                "private_extra": "must not escape",
            },
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_v1",
            {
                "state": "bootstrapping",
                "source_cursor": "00000000-0000-4000-8000-000000000001",
                "source_count": 1,
                "source_fingerprint": {"private": "must not escape"},
                "reason_code": None,
                "resume_phase": None,
            },
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_activity_v1",
            {
                "state": "bootstrapping",
                "source_cursor": "00000000-0000-4000-8000-000000000001",
                "source_count": 1,
                "source_fingerprint": "a" * 64,
                "reason_code": None,
                "resume_phase": None,
            },
            "notes_task_activity_readiness_state_invalid",
        ),
        (
            "notes_task_v1",
            {
                "state": "blocked",
                "source_cursor": None,
                "source_count": 0,
                "source_fingerprint": "a" * 64,
                "reason_code": "notes_task_source_invalid",
                "resume_phase": [],
            },
            "notes_task_readiness_state_invalid",
        ),
        (
            "notes_task_activity_v1",
            {
                "state": "blocked",
                "source_cursor": None,
                "source_count": 0,
                "source_fingerprint": "a" * 64,
                "reason_code": "notes_task_activity_source_invalid",
                "resume_phase": {},
            },
            "notes_task_activity_readiness_state_invalid",
        ),
    ],
)
def test_dormant_task_domain_shared_parser_failures_are_sanitized(
    tmp_path: Path,
    readiness_key: str,
    raw: object,
    reason_code: str,
) -> None:
    service, store = _service(tmp_path)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-total-parser",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={readiness_key: raw},
        )
    )

    diagnostics = service._profile_manager().notes_task_readiness_diagnostics(
        user_id="user-1",
        dataset_id="dataset-total-parser",
    )
    domain = (
        diagnostics.task
        if readiness_key == "notes_task_v1"
        else diagnostics.task_activity
    )

    assert domain.state == "blocked"
    assert domain.reason_code == reason_code
    assert "must not escape" not in repr(diagnostics)


def test_dormant_task_domain_other_domain_reason_fails_closed(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-cross-domain-reason",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "notes_task_v1": {
                    "state": "blocked",
                    "source_cursor": "00000000-0000-4000-8000-000000000001",
                    "source_count": 1,
                    "source_fingerprint": "a" * 64,
                    "reason_code": "notes_task_activity_source_invalid",
                    "resume_phase": "bootstrapping",
                }
            },
        )
    )

    diagnostics = service._profile_manager().notes_task_readiness_diagnostics(
        user_id="user-1",
        dataset_id="dataset-cross-domain-reason",
    )

    assert diagnostics.task.state == "blocked"
    assert diagnostics.task.source_count == 0
    assert diagnostics.task.reason_code == "notes_task_readiness_state_invalid"


def test_dormant_task_domain_oversized_count_fails_closed(tmp_path: Path) -> None:
    service, store = _service(tmp_path)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-oversized-count",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "notes_task_v1": {
                    "state": "bootstrapping",
                    "source_cursor": "00000000-0000-4000-8000-000000000001",
                    "source_count": 9_223_372_036_854_775_808,
                    "source_fingerprint": "a" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                }
            },
        )
    )

    diagnostics = service._profile_manager().notes_task_readiness_diagnostics(
        user_id="user-1",
        dataset_id="dataset-oversized-count",
    )

    assert diagnostics.task.state == "blocked"
    assert diagnostics.task.source_count == 0
    assert diagnostics.task.reason_code == "notes_task_readiness_state_invalid"


def test_dormant_task_domain_unpaired_surrogate_cursor_fails_closed(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-surrogate-cursor",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "notes_task_v1": {
                    "state": "bootstrapping",
                    "source_cursor": "\ud800",
                    "source_count": 1,
                    "source_fingerprint": "a" * 64,
                    "reason_code": None,
                    "resume_phase": None,
                }
            },
        )
    )

    diagnostics = service._profile_manager().notes_task_readiness_diagnostics(
        user_id="user-1",
        dataset_id="dataset-surrogate-cursor",
    )

    assert diagnostics.task.state == "blocked"
    assert diagnostics.task.source_count == 0
    assert diagnostics.task.reason_code == "notes_task_readiness_state_invalid"


def test_dormant_task_domain_forged_ready_never_changes_capabilities(
    tmp_path: Path,
) -> None:
    service, store = _service(tmp_path)
    task_ready = {
        "state": "ready",
        "source_cursor": "00000000-0000-4000-8000-000000000001",
        "source_count": 1,
        "source_fingerprint": "a" * 64,
        "reason_code": None,
        "resume_phase": None,
    }
    activity_ready = {
        "state": "ready",
        "source_cursor": (
            "2026-08-13T00:00:00+00:00|"
            "00000000-0000-4000-8000-000000000011"
        ),
        "source_count": 1,
        "source_fingerprint": "a" * 64,
        "reason_code": None,
        "resume_phase": None,
    }
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-forged-ready",
            owner_user_id="user-1",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
            metadata={
                "default_personal": True,
                "client_family": "chatbook",
                "notes_task_v1": task_ready,
                "notes_task_activity_v1": activity_ready,
                "task_activity_capture_enabled": True,
            },
        )
    )

    profile = service.profile(user_id="user-1")

    assert "notes.task" not in profile.capabilities.supported_domains
    assert "notes.task_activity" not in profile.capabilities.supported_domains
    assert "notes.task" not in profile.capabilities.operations
    assert "notes.task_activity" not in profile.capabilities.operations
    assert "notes.task" not in profile.capabilities.domain_schemas
    assert "notes.task_activity" not in profile.capabilities.domain_schemas
    assert "notes.task" not in profile.capabilities.supported_adapter_versions
    assert "notes.task_activity" not in profile.capabilities.supported_adapter_versions
    assert "notes.task" not in profile.capabilities.writable_adapter_versions
    assert "notes.task_activity" not in profile.capabilities.writable_adapter_versions
    assert profile.dataset is not None
    assert "notes.task" not in profile.dataset.domains
    assert "notes.task_activity" not in profile.dataset.domains
