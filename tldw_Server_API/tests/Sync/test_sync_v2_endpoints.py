from __future__ import annotations

import base64
import hashlib
import hmac
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from loguru import logger
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    User,
    check_rate_limit,
    get_request_user,
)
from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Personalization.personal_context_repository_models import (
    ProfileStorageLockedError,
)
from tldw_Server_API.app.core.Sync.v2 import factory as sync_v2_factory
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AttachmentRefAdapter,
    StaticSyncAdapter,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
    attachment_ref_v2_object_hash,
)
from tldw_Server_API.app.core.Sync.v2.blob_store import LocalSyncBlobStore
from tldw_Server_API.app.core.Sync.v2.domain_adapters.notes_organization import (
    NotesOrganizationDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.materializers import MaterializationResult
from tldw_Server_API.app.core.Sync.v2.models import (
    M1_SYNC_DOMAINS,
    NOTES_ORGANIZATION_DOMAINS,
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SYNC_V2_SUPPORTED_DOMAINS,
    SyncAttachmentRevisionBindingCreate,
    SyncBlobObjectCreate,
    SyncBlobUploadSessionCreate,
    SyncConflictCreate,
    SyncDatasetCreate,
    SyncDeviceCursor,
    SyncDeviceUpsert,
    SyncEnvelope,
    SyncEnvelopeCreate,
    SyncObjectState,
)
from tldw_Server_API.app.core.Sync.v2.profile import PersonalContextBootstrapError
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import (
    PersonalContextSyncCapabilities,
    SyncV2Service,
    SyncV2Settings,
)
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store

pytestmark = pytest.mark.unit


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
        + [
            NotesOrganizationDomainAdapter(domain=domain)
            for domain in NOTES_ORGANIZATION_DOMAINS
        ]
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
            personal_context=PersonalContextSyncCapabilities(),
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


def _client_for_factory_service(service: SyncV2Service) -> TestClient:
    """Build a typed endpoint client with a storage-safe authenticated user ID."""

    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = lambda: User(
        id="101", username="factory-user"
    )
    app.dependency_overrides[sync_endpoint.get_sync_v2_service] = lambda: service
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
    assert body["encryption_policies"] == ["server_trusted_v1"]
    assert body["blob_transfer"] == {"supported": False}
    assert body["personal_context"] == {
        "available": False,
        "blockers": [
            "personal_context_profile_key_unavailable",
            "personal_context_transport_unavailable",
        ],
        "authorization_policy": "server_trusted_v1",
        "min_schema_version": 1,
        "max_schema_version": 1,
        "integrity_algorithm": "hmac-sha256-v1",
        "integrity_key_distribution": "wrapped-bootstrap-v1",
        "privacy_cleanup_ack": "personal-context-cleanup-v1",
        "purge_generation": "personal-context-purge-v1",
        "max_record_bytes": 16_384,
        "max_search_results": 20,
        "max_proposals_per_turn": 5,
        "max_proposals_per_session": 25,
        "max_unresolved_proposals": 200,
    }
    assert body["domain_schemas"]["notes.note"]["upsert"]["properties"] == {
        "title": {"type": "string", "max_length": 255},
        "content": {"type": "string", "max_length": 5_000_000},
        "conversation_id": {"type": ["string", "null"]},
        "message_id": {"type": ["string", "null"]},
    }
    assert body["domain_schemas"]["notes.note"]["restore"] == {
        "operation": "upsert",
        "routing_metadata": {"restore_intent": True},
        "requires_current_base": True,
    }
    assert {
        domain: body["domain_schemas"][domain]
        for domain in NOTES_ORGANIZATION_DOMAINS
    } == {
        "notes.keyword": {
            "schema_version": 1,
            "encryption_policy": "server_trusted_v1",
            "upsert": {
                "required": ["keyword"],
                "properties": {"keyword": {"type": "string", "max_length": 100}},
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
        },
        "notes.keyword_link": {
            "schema_version": 1,
            "encryption_policy": "server_trusted_v1",
            "upsert": {
                "required": ["subject_type", "subject_id", "keyword_sync_id"],
                "properties": {
                    "subject_type": {"enum": ["note", "conversation"]},
                    "subject_id": {"type": "string"},
                    "keyword_sync_id": {"type": "string"},
                },
                "additional_properties": False,
            },
            "tombstone": {
                "required": ["subject_type", "subject_id", "keyword_sync_id"],
                "properties": {
                    "subject_type": {"enum": ["note", "conversation"]},
                    "subject_id": {"type": "string"},
                    "keyword_sync_id": {"type": "string"},
                },
                "additional_properties": False,
            },
        },
        "notes.keyword_collection": {
            "schema_version": 1,
            "encryption_policy": "server_trusted_v1",
            "upsert": {
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "max_length": 255},
                    "parent_sync_id": {"type": ["string", "null"]},
                },
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
        },
        "notes.keyword_collection_link": {
            "schema_version": 1,
            "encryption_policy": "server_trusted_v1",
            "upsert": {
                "required": ["collection_sync_id", "keyword_sync_id"],
                "properties": {
                    "collection_sync_id": {"type": "string"},
                    "keyword_sync_id": {"type": "string"},
                },
                "additional_properties": False,
            },
            "tombstone": {
                "required": ["collection_sync_id", "keyword_sync_id"],
                "properties": {
                    "collection_sync_id": {"type": "string"},
                    "keyword_sync_id": {"type": "string"},
                },
                "additional_properties": False,
            },
        },
        "notes.folder": {
            "schema_version": 1,
            "encryption_policy": "server_trusted_v1",
            "upsert": {
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "max_length": 500},
                    "parent_sync_id": {"type": ["string", "null"]},
                },
                "additional_properties": False,
            },
            "tombstone": {"operation": "tombstone"},
        },
        "notes.folder_link": {
            "schema_version": 1,
            "encryption_policy": "server_trusted_v1",
            "upsert": {
                "required": ["note_id", "folder_sync_id"],
                "properties": {
                    "note_id": {"type": "string"},
                    "folder_sync_id": {"type": "string"},
                },
                "additional_properties": False,
            },
            "tombstone": {
                "required": ["note_id", "folder_sync_id"],
                "properties": {
                    "note_id": {"type": "string"},
                    "folder_sync_id": {"type": "string"},
                },
                "additional_properties": False,
            },
        },
    }
    assert body["warnings"] == []


@pytest.mark.parametrize(
    ("gate_enabled", "state", "expected"),
    [
        (False, "ready", []),
        (True, "initializing", []),
        (True, "ready", [2]),
    ],
)
def test_capabilities_endpoint_reports_selected_dataset_writable_versions(
    tmp_path: Path,
    gate_enabled: bool,
    state: str,
    expected: list[int],
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    service.adapters.register(
        AttachmentRefAdapter(v2_writes_enabled=gate_enabled)
    )
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note", "attachment.ref"],
            metadata={"notes_attachment_v2": {"state": state}},
        )
    )

    response = _client_for_service(service).get(
        "/api/v1/sync/capabilities",
        params={"dataset_id": "dataset-1"},
    )

    assert response.status_code == 200
    assert response.json()["supported_adapter_versions"]["attachment.ref"] == [1, 2]
    assert response.json()["writable_adapter_versions"]["attachment.ref"] == expected


@pytest.mark.unit
def test_capabilities_endpoint_hides_unauthorized_selected_dataset(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path)
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-private",
            owner_user_id="user-2",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note", "attachment.ref"],
            metadata={"notes_attachment_v2": {"state": "ready"}},
        )
    )

    response = _client_for_service(service).get(
        "/api/v1/sync/capabilities",
        params={"dataset_id": "dataset-private"},
    )

    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "sync_resource_not_found"


def _sync_diagnostics_snapshot(service: SyncV2Service) -> dict[str, list[dict[str, Any]]]:
    tables = (
        "sync_datasets",
        "sync_envelopes",
        "sync_current_heads",
        "sync_conflicts",
        "sync_attachment_revision_bindings",
        "sync_blob_objects",
        "sync_blob_upload_sessions",
        "sync_notes_attachment_cleanup_candidates",
    )
    return {
        table: service.store.db.execute(f"SELECT * FROM {table}").rows  # nosec B608
        for table in tables
    }


def _insert_diagnostic_attachment(
    service: SyncV2Service,
    *,
    attachment_id: str,
    parent_object_id: str,
    client_envelope_id: str,
    object_revision: int = 1,
    operation: str = "upsert",
    base: SyncEnvelope | None = None,
) -> SyncEnvelope:
    payload: dict[str, Any] = {
        "attachment_id": attachment_id,
        "parent_domain": "notes.note",
        "parent_object_id": parent_object_id,
        "file_name": f"{attachment_id[:8]}.pdf",
        "original_file_name": f"{attachment_id[:8]}.pdf",
        "content_type": "application/pdf",
        "size_bytes": 17,
        "blob_hash": "sha256:" + attachment_id.replace("-", "") * 2,
        "created_at": _clock(),
        "last_modified": _clock(),
        "created_by": "diagnostic-device",
    }
    if operation == "tombstone":
        payload["deleted_at"] = _clock()
    return service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id=client_envelope_id,
            domain="attachment.ref",
            operation=operation,
            object_id=attachment_id,
            object_revision=object_revision,
            schema_version=2,
            adapter_version=2,
            payload=payload,
            payload_hash=attachment_ref_v2_object_hash(
                operation,
                payload,
                object_revision=object_revision,
            ),
            created_at_client=_clock(),
            base_server_cursor=None if base is None else base.server_cursor,
            base_object_revision=None if base is None else base.object_revision,
            base_object_hash=None if base is None else base.payload_hash,
        )
    )


def test_attachment_diagnostics_are_read_only_bounded_and_actionable(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )
    parent_id = "b2222222-2222-4222-8222-222222222222"
    parent = service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="diagnostic-parent-upsert",
            domain="notes.note",
            operation="upsert",
            object_id=parent_id,
            object_revision=1,
            payload={"title": "Diagnostic parent", "content": ""},
            payload_hash="sha256:" + "d" * 64,
            apply_status="applied",
        )
    )
    service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="diagnostic-parent-tombstone",
            domain="notes.note",
            operation="tombstone",
            object_id=parent_id,
            object_revision=2,
            payload={},
            payload_hash="sha256:" + "e" * 64,
            apply_status="applied",
            base_server_cursor=parent.server_cursor,
            base_object_revision=parent.object_revision,
            base_object_hash=parent.payload_hash,
        )
    )
    live = _insert_diagnostic_attachment(
        service,
        attachment_id="a7111111-1111-4111-8111-111111111111",
        parent_object_id="b7111111-1111-4111-8111-111111111111",
        client_envelope_id="diagnostic-attachment-live",
    )
    hidden = _insert_diagnostic_attachment(
        service,
        attachment_id="a8111111-1111-4111-8111-111111111111",
        parent_object_id=parent_id,
        client_envelope_id="diagnostic-attachment-hidden",
    )
    tombstone_base = _insert_diagnostic_attachment(
        service,
        attachment_id="a9111111-1111-4111-8111-111111111111",
        parent_object_id="b9111111-1111-4111-8111-111111111111",
        client_envelope_id="diagnostic-attachment-before-tombstone",
    )
    _insert_diagnostic_attachment(
        service,
        attachment_id=tombstone_base.object_id,
        parent_object_id="b9111111-1111-4111-8111-111111111111",
        client_envelope_id="diagnostic-attachment-tombstone",
        object_revision=2,
        operation="tombstone",
        base=tombstone_base,
    )
    service.store.mark_envelope_apply_status(
        hidden.server_cursor,
        apply_status="failed",
        apply_error_code="sync_attachment_projection_failed",
        apply_error_message="private projection detail",
    )
    service.store.db.execute(
        """
        UPDATE sync_attachment_revision_bindings
           SET resolved_blob_id = 'blob-missing'
         WHERE dataset_id = ? AND attachment_id = ? AND attachment_revision = 1
        """,
        ("dataset-1", live.object_id),
    )
    service.store.insert_conflict(
        SyncConflictCreate(
            conflict_id="attachment-conflict-diagnostic",
            dataset_id="dataset-1",
            domain="attachment.ref",
            object_id=hidden.object_id,
            conflict_type="revision_mismatch",
            server_cursor=hidden.server_cursor,
            metadata={"private_detail": "must-not-leak"},
        )
    )
    service.store.begin_notes_attachment_bootstrap(
        "dataset-1",
        owner_user_id="user-1",
        bootstrap_id="bootstrap-diagnostic",
    )
    source_key = "notes_attachments/diagnostic/private.pdf"
    service.store.resolve_notes_attachment_source_map(
        "dataset-1",
        owner_user_id="user-1",
        bootstrap_id="bootstrap-diagnostic",
        note_id=parent_id,
        source_key=source_key,
    )
    service.store.record_notes_attachment_cleanup_candidate(
        "dataset-1",
        owner_user_id="user-1",
        bootstrap_id="bootstrap-diagnostic",
        source_key=source_key,
        source_relative_path=source_key,
        source_blob_hash="sha256:" + "7" * 64,
        source_size_bytes=17,
        source_modified_ns=1,
    )
    blob = service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-quarantined",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="a1111111-1111-4111-8111-111111111111",
            payload_hash="sha256:" + "a" * 64,
            content_type="application/pdf",
            size_bytes=12,
            storage_backend="local_fs",
            storage_key="blobs/v2/" + "b" * 32 + "/" + "a" * 64 + ".blob",
        )
    )
    service.store.db.execute(
        "UPDATE sync_blob_objects SET status = 'quarantined' WHERE blob_id = ?",
        (blob.blob_id,),
    )
    for index, status in enumerate(("verify_failed", "deleting", "deleted"), start=1):
        digest = f"{index}" * 64
        created = service.store.complete_blob_upload(
            SyncBlobObjectCreate(
                blob_id=f"blob-{status}",
                dataset_id="dataset-1",
                owner_user_id="user-1",
                attachment_id=f"a{index + 2}111111-1111-4111-8111-111111111111",
                payload_hash="sha256:" + digest,
                content_type="application/pdf",
                size_bytes=12 + index,
                storage_backend="local_fs",
                storage_key=f"blobs/v2/{digest[:32]}/{digest}.blob",
            )
        )
        service.store.db.execute(
            """
            UPDATE sync_blob_objects
               SET status = ?, deleted_at = CASE WHEN ? = 'deleted' THEN ? ELSE NULL END
             WHERE blob_id = ?
            """,
            (status, status, _clock(), created.blob_id),
        )
    service.store.create_blob_upload_session(
        SyncBlobUploadSessionCreate(
            upload_id="upload-diagnostic",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            device_id=None,
            attachment_id="a6111111-1111-4111-8111-111111111111",
            domain="attachment.ref",
            object_id="a6111111-1111-4111-8111-111111111111",
            content_type="application/pdf",
            size_bytes=16,
            payload_hash="sha256:" + "6" * 64,
            chunk_size=16,
            chunk_count=1,
            reserved_quota_bytes=16,
        )
    )
    with service.store.db.materialization_transaction(
        [("dataset-1", "attachment.ref", "a2222222-2222-4222-8222-222222222222")]
    ) as connection:
        service.store.db._create_attachment_revision_binding(
            SyncAttachmentRevisionBindingCreate(
                dataset_id="dataset-1",
                attachment_id="a2222222-2222-4222-8222-222222222222",
                attachment_revision=1,
                blob_hash="sha256:" + "c" * 64,
                size_bytes=8,
                establishing_server_cursor=1,
                availability_at_acceptance="metadata_only",
            ),
            connection=connection,
        )
    before = _sync_diagnostics_snapshot(service)

    response = _client_for_service(service).get(
        "/api/v1/sync/diagnostics",
        params={
            "dataset_id": "dataset-1",
            "attachment_sample_limit": 1,
            "attachment_total_sample_limit": 500,
        },
    )

    assert response.status_code == 200
    attachment = response.json()["attachment_lifecycle"]
    assert attachment["counts"]["quarantined"] == 1
    assert attachment["counts"]["verify_failed"] == 1
    assert attachment["counts"]["deleting"] == 1
    assert attachment["counts"]["deleted"] == 1
    assert attachment["counts"]["active_uploads"] == 1
    assert attachment["counts"]["registry_live"] == 1
    assert attachment["counts"]["registry_hidden"] == 1
    assert attachment["counts"]["registry_tombstoned"] == 1
    assert attachment["counts"]["metadata_only"] >= 1
    assert attachment["counts"]["missing"] == 1
    assert attachment["counts"]["cleanup_candidates"] == 1
    assert attachment["counts"]["projection_pending"] >= 1
    assert attachment["counts"]["projection_failed"] == 1
    assert attachment["counts"]["unresolved_conflicts"] == 1
    assert len(attachment["samples"]) <= 500
    assert all(
        sum(sample["category"] == category for sample in attachment["samples"]) <= 1
        for category in {sample["category"] for sample in attachment["samples"]}
    )
    assert {action["action"] for action in attachment["recovery_actions"]} >= {
        "release_quarantine",
        "retry_upload",
        "retry_verify",
        "gc_retry",
        "resume_upload",
        "restore_attachment",
        "restore_note",
        "repair_projection",
        "resolve_conflict",
        "bootstrap_resume",
        "wait_for_retention",
    }
    assert _sync_diagnostics_snapshot(service) == before
    assert all("storage_key" not in sample for sample in attachment["samples"])
    assert all("payload_hash" not in sample for sample in attachment["samples"])

    no_samples = _client_for_service(service).get(
        "/api/v1/sync/diagnostics",
        params={"dataset_id": "dataset-1"},
    )
    assert no_samples.status_code == 200
    assert no_samples.json()["attachment_lifecycle"]["samples"] == []

    aggregate_oversized = _client_for_service(service).get(
        "/api/v1/sync/diagnostics",
        params={
            "dataset_id": "dataset-1",
            "attachment_sample_limit": 1,
            "attachment_total_sample_limit": 1,
        },
    )
    assert aggregate_oversized.status_code == 413
    assert aggregate_oversized.json()["detail"]["error_code"] == (
        "sync_attachment_diagnostic_total_sample_limit_exceeded"
    )
    assert _sync_diagnostics_snapshot(service) == before


@pytest.mark.parametrize(
    "params",
    [
        {"attachment_sample_limit": 101},
        {"attachment_total_sample_limit": 501},
    ],
)
def test_attachment_diagnostics_reject_oversized_samples(
    sync_service: SyncV2Service,
    params: dict[str, int],
) -> None:
    sync_service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note", "attachment.ref"],
    )

    response = _client_for_service(sync_service).get(
        "/api/v1/sync/diagnostics",
        params={"dataset_id": "dataset-1", **params},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["error_code"].startswith(
        "sync_attachment_diagnostic_"
    )


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
    sync_service.store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-2",
            domain="notes.note",
            last_pulled_sequence=5,
            max_delivered_sequence=5,
        )
    )
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


def test_device_acknowledgment_endpoint_forwards_exact_adapter_version(
    tmp_path: Path,
) -> None:
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "version-ack.db")),
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="notes.note", supported_adapter_versions={1, 2})]
        ),
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
        capabilities={
            "requested_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
        },
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["notes.note"],
    )
    service.store.update_device_cursor(
        SyncDeviceCursor(
            dataset_id="dataset-1",
            device_id="device-1",
            domain="notes.note",
            adapter_version=2,
            last_pulled_sequence=5,
            max_delivered_sequence=5,
        )
    )

    response = _client_for_service(service).post(
        "/api/v1/sync/device-acknowledgments",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "domain_acks": [
                {
                    "domain": "notes.note",
                    "adapter_version": 2,
                    "through_server_sequence": 5,
                    "applied_at": "2026-05-23T18:30:00+00:00",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert len(response.json()["version_acks"]) == 1
    assert response.json()["version_acks"][0]["adapter_version"] == 2
    assert response.json()["domain_acks"] == {}


def _blob_id_ack_service(tmp_path: Path, *, supports_v2: bool = True) -> SyncV2Service:
    versions = {1, 2} if supports_v2 else {1}
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "blob-id-ack.db")),
        adapters=SyncAdapterRegistry(
            [StaticSyncAdapter(domain="attachment.ref", supported_adapter_versions=versions)]
        ),
        clock=_clock,
        settings=SyncV2Settings(server_trusted_encryption=_ready_encryption()),
    )
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
        capabilities={
            "requested_domains": ["attachment.ref"],
            "supported_adapter_versions": {"attachment.ref": sorted(versions)},
        },
    )
    service.enroll_dataset(
        user_id="user-1",
        dataset_id="dataset-1",
        domains=["attachment.ref"],
    )
    service.store.complete_blob_upload(
        SyncBlobObjectCreate(
            blob_id="blob-1",
            dataset_id="dataset-1",
            owner_user_id="user-1",
            attachment_id="attachment-provenance",
            payload_hash="sha256:" + "a" * 64,
            content_type="application/octet-stream",
            size_bytes=1,
            storage_backend="local_fs",
            storage_key="blob-1.bin",
        )
    )
    return service


def test_device_acknowledgment_endpoint_persists_reachable_blob_id_evidence(
    tmp_path: Path,
) -> None:
    response = _client_for_service(_blob_id_ack_service(tmp_path)).post(
        "/api/v1/sync/device-acknowledgments",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "blob_id_acks": [
                {
                    "blob_id": "blob-1",
                    "payload_hash": "sha256:" + "a" * 64,
                    "verified_at": "2026-08-11T20:30:00Z",
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["blob_acks"] == []
    assert response.json()["blob_id_acks"][0]["blob_id"] == "blob-1"


@pytest.mark.parametrize(
    ("supports_v2", "blob_id", "digest", "status_code", "error_code"),
    [
        (False, "blob-1", "sha256:" + "a" * 64, 400, "sync_blob_id_ack_adapter_v2_required"),
        (True, "missing", "sha256:" + "a" * 64, 404, "sync_blob_id_ack_not_authorized"),
        (True, "blob-1", "sha256:" + "b" * 64, 400, "sync_blob_id_ack_digest_mismatch"),
    ],
)
def test_device_acknowledgment_endpoint_sanitizes_blob_id_ack_errors(
    tmp_path: Path,
    supports_v2: bool,
    blob_id: str,
    digest: str,
    status_code: int,
    error_code: str,
) -> None:
    response = _client_for_service(
        _blob_id_ack_service(tmp_path, supports_v2=supports_v2)
    ).post(
        "/api/v1/sync/device-acknowledgments",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "blob_id_acks": [
                {
                    "blob_id": blob_id,
                    "payload_hash": digest,
                    "verified_at": "2026-08-11T20:30:00Z",
                }
            ],
        },
    )

    assert response.status_code == status_code
    assert response.json()["detail"]["error_code"] == error_code
    assert blob_id not in response.json()["detail"]["message"]


@pytest.mark.parametrize(
    ("error_code", "expected_status"),
    [
        ("sync_pull_token_invalid", 400),
        ("sync_pull_token_too_large", 413),
        ("sync_pull_restart_required", 409),
        ("sync_device_adapter_version_not_supported", 400),
    ],
)
def test_pull_endpoint_maps_versioned_token_errors(
    error_code: str,
    expected_status: int,
) -> None:
    class FailingPullService:
        def pull(self, **_kwargs):
            raise SyncStoreError(error_code)

    response = _client_for_service(FailingPullService()).get(  # type: ignore[arg-type]
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "cursor": "opaque-token",
        },
    )

    assert response.status_code == expected_status
    assert response.json()["detail"]["error_code"] == error_code


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


def test_attachment_bootstrap_diagnostics_endpoint_is_read_only_and_bounded(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    class _DryRunBootstrapper:
        def dry_run(self, *, service: SyncV2Service, user_id: str):
            assert service is sync_service
            assert user_id == "user-1"
            return {
                "candidate_count": 7,
                "candidate_count_is_lower_bound": False,
                "error_code": None,
            }

    sync_service.notes_attachment_bootstrapper = _DryRunBootstrapper()

    response = client.get(
        "/api/v1/sync/profile/attachment-bootstrap",
        params={"dry_run": "true", "sample_limit": 0},
    )

    assert response.status_code == 200
    assert response.json() == {
        "state": "not_started",
        "captured_count": 0,
        "expected_count": 0,
        "cursor": None,
        "error_code": None,
        "dry_run": True,
        "source_candidate_count": 7,
        "source_candidate_count_is_lower_bound": False,
        "cleanup_candidates": [],
        "recovery_actions": [],
    }
    assert sync_service.store.list_datasets_for_user("user-1") == []
    assert sync_service.store.list_devices_for_user("user-1") == []

    oversized = client.get(
        "/api/v1/sync/profile/attachment-bootstrap",
        params={"sample_limit": 101},
    )
    assert oversized.status_code == 413
    assert oversized.json()["detail"]["error_code"] == (
        "sync_attachment_bootstrap_sample_limit_exceeded"
    )


def test_attachment_bootstrap_diagnostics_for_fresh_user_does_not_create_sync_db(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_db_path = tmp_path / "fresh_attachment_diagnostics.db"
    monkeypatch.setenv("SYNC_V2_SQLITE_PATH", str(sync_db_path))
    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    client = TestClient(app)

    response = client.get("/api/v1/sync/profile/attachment-bootstrap")

    assert response.status_code == 200
    assert response.json() == {
        "state": "not_started",
        "captured_count": 0,
        "expected_count": 0,
        "cursor": None,
        "error_code": None,
        "dry_run": False,
        "source_candidate_count": None,
        "source_candidate_count_is_lower_bound": False,
        "cleanup_candidates": [],
        "recovery_actions": [],
    }
    assert not sync_db_path.exists()

    oversized = client.get(
        "/api/v1/sync/profile/attachment-bootstrap",
        params={"sample_limit": 101},
    )
    assert oversized.status_code == 413
    assert not sync_db_path.exists()


def test_attachment_bootstrap_diagnostics_enforces_ingress_rate_limit(
    sync_service: SyncV2Service,
) -> None:
    async def _deny_rate_limit() -> None:
        raise HTTPException(status_code=429, detail="rate limited")

    app = FastAPI()
    app.include_router(sync_endpoint.router, prefix="/api/v1/sync")
    app.dependency_overrides[get_request_user] = _test_user
    app.dependency_overrides[sync_endpoint.get_sync_v2_profile_service] = (
        lambda: sync_service
    )
    app.dependency_overrides[check_rate_limit] = _deny_rate_limit
    client = TestClient(app)

    response = client.get("/api/v1/sync/profile/attachment-bootstrap")

    assert response.status_code == 429


def test_profile_bootstrap_and_status_expose_safe_attachment_progress(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    started = client.post(
        "/api/v1/sync/profile/bootstrap",
        json={
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": [*M1_SYNC_DOMAINS, "attachment.ref"],
        },
    )
    status_response = client.get("/api/v1/sync/profile")

    assert started.status_code == status_response.status_code == 200
    expected = {
        "state": "initializing",
        "captured_count": 0,
        "expected_count": 0,
        "error_code": None,
    }
    assert started.json()["dataset"]["notes_attachment"] == expected
    assert status_response.json()["dataset"]["notes_attachment"] == expected
    assert "bootstrap_id" not in started.text
    assert "source_cursor" not in started.text

    dataset_id = started.json()["dataset"]["dataset_id"]
    dataset = sync_service.store.get_dataset(dataset_id, owner_user_id="user-1")
    assert dataset is not None
    bootstrap_id = dataset.metadata["notes_attachment_v2"]["bootstrap_id"]
    empty_hash = hashlib.sha256(b"").hexdigest()
    sync_service.store.transition_notes_attachment_bootstrap(
        dataset_id,
        owner_user_id="user-1",
        bootstrap_id=bootstrap_id,
        expected_state="initializing",
        state="ready",
        captured_count=0,
        expected_count=0,
        source_hash=empty_hash,
        source_cursor=None,
        ready_verifier=lambda: True,
    )
    ready = client.get("/api/v1/sync/profile")
    assert ready.json()["dataset"]["notes_attachment"]["state"] == "ready"

    sync_service.store.transition_notes_attachment_bootstrap(
        dataset_id,
        owner_user_id="user-1",
        bootstrap_id=bootstrap_id,
        expected_state="ready",
        state="failed",
        captured_count=0,
        expected_count=0,
        source_hash=empty_hash,
        source_cursor=None,
        error_code="notes_attachment_source_changed",
    )
    failed = client.get("/api/v1/sync/profile")
    assert failed.json()["dataset"]["notes_attachment"] == {
        "state": "failed",
        "captured_count": 0,
        "expected_count": 0,
        "error_code": "notes_attachment_source_changed",
    }


def test_attachment_bootstrap_diagnostics_enforce_owner_and_hide_legacy_path(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    own = sync_service.store.get_or_create_default_personal_dataset("user-1")
    own = sync_service.store.begin_notes_attachment_bootstrap(
        own.dataset_id,
        owner_user_id="user-1",
        bootstrap_id="bootstrap-private",
    )
    source_key = "notes_attachments/note-secret/private-name.pdf"
    mapping = sync_service.store.resolve_notes_attachment_source_map(
        own.dataset_id,
        owner_user_id="user-1",
        bootstrap_id="bootstrap-private",
        note_id="note-secret",
        source_key=source_key,
    )
    sync_service.store.record_notes_attachment_cleanup_candidate(
        own.dataset_id,
        owner_user_id="user-1",
        bootstrap_id="bootstrap-private",
        source_key=source_key,
        source_relative_path=source_key,
        source_blob_hash="sha256:" + "2" * 64,
        source_size_bytes=10,
        source_modified_ns=1,
    )
    other = sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="other-dataset",
            owner_user_id="other-user",
            scope_type="personal",
            domains=list(M1_SYNC_DOMAINS),
        )
    )

    response = client.get(
        "/api/v1/sync/profile/attachment-bootstrap",
        params={"dataset_id": own.dataset_id, "sample_limit": 1},
    )
    denied = client.get(
        "/api/v1/sync/profile/attachment-bootstrap",
        params={"dataset_id": other.dataset_id, "sample_limit": 1},
    )

    assert response.status_code == 200
    assert response.json()["cleanup_candidates"] == [
        {
            "source_key_hash": "sha256:"
            + hashlib.sha256(source_key.encode()).hexdigest(),
            "attachment_id": mapping.attachment_id,
            "state": "captured",
            "blocker_code": None,
        }
    ]
    assert "private-name.pdf" not in response.text
    assert "notes_attachments" not in response.text
    assert "bootstrap-private" not in response.text
    assert denied.status_code == 404
    assert "other-dataset" not in denied.text


@pytest.mark.parametrize(
    "version_map",
    [
        {"attachment.ref": [True]},
        {"attachment.ref": list(range(1, 10))},
    ],
)
def test_profile_bootstrap_endpoint_rejects_malformed_adapter_maps(
    client: TestClient,
    sync_service: SyncV2Service,
    version_map: dict[str, list[object]],
) -> None:
    response = client.post(
        "/api/v1/sync/profile/bootstrap",
        json={
            "client_family": "chatbook",
            "mode": "offline_sync",
            "device_id": "device-1",
            "requested_domains": ["attachment.ref"],
            "supported_adapter_versions": version_map,
        },
    )

    assert response.status_code == 422
    assert sync_service.store.list_devices_for_user("user-1") == []
    assert sync_service.store.list_datasets_for_user("user-1") == []


def test_profile_bootstrap_rejects_reserved_server_origin_device_id(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    registration = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "server-origin",
            "display_name": "Client",
            "client_type": "chatbook",
        },
    )
    response = client.post(
        "/api/v1/sync/profile/bootstrap",
        json={
            "client_family": "chatbook",
            "mode": "offline_sync",
            "device_id": "server-origin",
            "device_name": "Client",
        },
    )

    expected_detail = {
        "error_code": "reserved_device_id",
        "message": "The requested Sync device identifier is reserved.",
    }
    assert registration.status_code == response.status_code == 400
    assert registration.json()["detail"] == response.json()["detail"] == expected_detail
    assert sync_service.store.list_datasets_for_user("user-1") == []
    assert sync_service.store.list_devices_for_user("user-1") == []


def test_profile_endpoint_exposes_only_typed_notes_organization_summary(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    dataset = sync_service.store.get_or_create_default_personal_dataset("user-1")
    bootstrap_id = "private-bootstrap-id"
    sync_service.store.begin_notes_organization_bootstrap(
        dataset.dataset_id,
        owner_user_id="user-1",
        bootstrap_id=bootstrap_id,
    )

    initializing = client.get("/api/v1/sync/profile")
    assert initializing.status_code == 200
    assert initializing.json()["dataset"]["domains"] == [
        *M1_SYNC_DOMAINS,
        *NOTES_ORGANIZATION_DOMAINS,
    ]
    assert initializing.json()["dataset"]["notes_organization"] == {
        "state": "initializing",
        "captured_count": 0,
        "expected_count": 0,
        "error_code": None,
    }

    sync_service.store.transition_notes_organization_bootstrap(
        dataset.dataset_id,
        bootstrap_id=bootstrap_id,
        expected_state="initializing",
        state="ready",
        captured_count=0,
        expected_count=0,
        ready_verifier=lambda: True,
    )
    ready = client.get("/api/v1/sync/profile")
    assert ready.status_code == 200
    assert ready.json()["dataset"]["notes_organization"] == {
        "state": "ready",
        "captured_count": 0,
        "expected_count": 0,
        "error_code": None,
    }

    sync_service.store.transition_notes_organization_bootstrap(
        dataset.dataset_id,
        bootstrap_id=bootstrap_id,
        expected_state="ready",
        state="failed",
        captured_count=3,
        expected_count=4,
        error_code="notes_organization_bootstrap_source_invalid",
    )
    failed = client.get("/api/v1/sync/profile")
    assert failed.status_code == 200
    assert failed.json()["dataset"]["notes_organization"] == {
        "state": "failed",
        "captured_count": 3,
        "expected_count": 4,
        "error_code": "notes_organization_bootstrap_source_invalid",
    }
    serialized = failed.text
    assert bootstrap_id not in serialized
    assert "notes_organization_v1" not in serialized


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
            "metadata": {"label": "internal-caller"},
        },
    )

    assert registered.status_code == 200
    assert registered.json()["device_id"] == "device-1"
    assert registered.json()["server_capabilities"]["domains"] == list(SYNC_V2_SUPPORTED_DOMAINS)
    assert registered.json()["server_capabilities"]["encryption_policies"] == ["server_trusted_v1"]
    assert enrolled.status_code == 200
    assert enrolled.json()["dataset_id"] == "dataset-1"
    assert enrolled.json()["encryption_policy"] == "server_trusted_v1"
    assert enrolled.json()["domains"] == list(M1_SYNC_DOMAINS)
    assert enrolled.json()["key_setup_required"] is False
    assert enrolled.json()["metadata"] == {"label": "internal-caller"}


def test_device_patch_rejects_adapter_version_removal_without_mutating_registration(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    registered = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "device-1",
            "display_name": "Laptop",
            "supported_domains": ["notes.note"],
            "supported_adapter_versions": {"notes.note": [1, 2]},
            "capabilities": {"theme": "dark"},
        },
    )
    assert registered.status_code == 200

    response = client.patch(
        "/api/v1/sync/devices/device-1",
        json={
            "capabilities": {
                "supported_adapter_versions": {"notes.note": [2]},
                "theme": "light",
            }
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error_code": "sync_validation_failed",
        "message": "Sync request parameters are invalid.",
    }
    stored = sync_service.store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "notes.note": [1, 2]
    }
    assert stored.capabilities["theme"] == "dark"


@pytest.mark.parametrize(
    "requested_domains",
    [
        ["unknown.domain"],
        ["notes.note"] * 101,
    ],
)
def test_device_registration_rejects_invalid_requested_domains_without_version_map(
    client: TestClient,
    sync_service: SyncV2Service,
    requested_domains: list[str],
) -> None:
    response = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "invalid-device",
            "display_name": "Invalid device",
            "capabilities": {"requested_domains": requested_domains},
        },
    )

    assert response.status_code == 422
    assert sync_service.store.get_device("user-1", "invalid-device") is None


@pytest.mark.parametrize(
    "requested_domains",
    [
        ["unknown.domain"],
        ["notes.note"] * 101,
    ],
)
def test_device_patch_rejects_invalid_requested_domains_without_version_map(
    client: TestClient,
    sync_service: SyncV2Service,
    requested_domains: list[str],
) -> None:
    registered = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "device-1",
            "display_name": "Laptop",
            "supported_domains": ["notes.note"],
        },
    )
    assert registered.status_code == 200
    original = sync_service.store.get_device("user-1", "device-1")
    assert original is not None

    response = client.patch(
        "/api/v1/sync/devices/device-1",
        json={"capabilities": {"requested_domains": requested_domains}},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error_code": "sync_validation_failed",
        "message": "Sync request parameters are invalid.",
    }
    stored = sync_service.store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities == original.capabilities


def test_partial_registration_adapter_map_defaults_omitted_domain_for_push(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    registered = client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "device-1",
            "display_name": "Laptop",
            "supported_domains": ["notes.note", "attachment.ref"],
            "supported_adapter_versions": {"attachment.ref": [2]},
        },
    )
    enrolled = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "domains": ["notes.note"],
            "encryption_policy": "server_trusted_v1",
        },
    )
    assert registered.status_code == 200
    assert enrolled.status_code == 200

    stored = sync_service.store.get_device("user-1", "device-1")
    assert stored is not None
    assert stored.capabilities["supported_adapter_versions"] == {
        "notes.note": [1],
        "attachment.ref": [2],
    }
    pushed = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "base_server_cursor": 0,
            "envelopes": [_note_envelope_json()],
        },
    )

    assert pushed.status_code == 200
    assert [item["client_envelope_id"] for item in pushed.json()["accepted"]] == [
        "env-note"
    ]
    assert pushed.json()["rejected"] == []


def test_public_push_rejects_unattested_attachment_bootstrap_routing_safely(
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.Sync.v2.attachment_refs_v2 import (
        attachment_ref_v2_object_hash,
    )

    service = _build_service(tmp_path, supports_attachments=True)
    service.adapters.register(AttachmentRefAdapter(v2_writes_enabled=True))
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
        capabilities={
            "requested_domains": ["attachment.ref"],
            "supported_adapter_versions": {"attachment.ref": [2]},
        },
    )
    service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-1",
            owner_user_id="user-1",
            scope_type="personal",
            encryption_policy="server_trusted_v1",
            domains=["notes.note", "attachment.ref"],
            metadata={"notes_attachment_v2": {"state": "ready"}},
        )
    )
    payload = {
        "attachment_id": "a1111111-1111-4111-8111-111111111111",
        "parent_domain": "notes.note",
        "parent_object_id": "b2222222-2222-4222-8222-222222222222",
        "file_name": "diagram.png",
        "original_file_name": "diagram.png",
        "content_type": "image/png",
        "size_bytes": 42,
        "blob_hash": "sha256:" + "a" * 64,
        "created_at": _clock(),
        "last_modified": _clock(),
        "created_by": "device-1",
    }
    response = _client_for_service(service).post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                {
                    "dataset_id": "dataset-1",
                    "client_envelope_id": "env-untrusted-bootstrap",
                    "device_id": "device-1",
                    "client_sequence": 1,
                    "domain": "attachment.ref",
                    "operation": "upsert",
                    "object_id": payload["attachment_id"],
                    "object_revision": 1,
                    "schema_version": 2,
                    "adapter_version": 2,
                    "payload": payload,
                    "payload_hash": attachment_ref_v2_object_hash(
                        "upsert",
                        payload,
                        object_revision=1,
                    ),
                    "created_at_client": _clock(),
                    "encryption_metadata": {"policy": "server_trusted_v1"},
                    "routing_metadata": {
                        "bootstrap_capture": True,
                        "bootstrap_id": "client-forged-bootstrap",
                    },
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["accepted"] == []
    assert response.json()["rejected"] == [
        {
            "client_envelope_id": "env-untrusted-bootstrap",
            "error_code": "attachment_ref_v2_payload_invalid",
            "message": "attachment.ref v2 payload validation failed",
            "retryable": False,
        }
    ]
    assert service.store.list_envelopes_after("dataset-1", 0) == []


def test_dataset_enroll_endpoint_rejects_and_never_echoes_forged_server_metadata(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    forged = {
        "default_personal": True,
        "client_family": "chatbook",
        "notes_organization_v1": {
            "state": "ready",
            "bootstrap_id": "client-forged",
            "captured_count": 1,
            "expected_count": 1,
        },
    }

    response = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "forged-dataset",
            "scope_type": "personal",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": forged,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error_code": "sync_reserved_dataset_enrollment",
        "message": "Reserved Sync dataset capabilities require profile bootstrap.",
    }
    assert "client-forged" not in response.text
    assert sync_service.store.list_datasets_for_user("user-1") == []


def test_dataset_enroll_endpoint_rejects_forged_attachment_readiness(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    response = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "forged-attachment-ready",
            "scope_type": "personal",
            "domains": ["notes.note", "attachment.ref"],
            "encryption_policy": "server_trusted_v1",
            "metadata": {
                "notes_attachment_v2": {
                    "state": "ready",
                    "bootstrap_id": "private-bootstrap-id",
                }
            },
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error_code": "sync_reserved_dataset_enrollment",
        "message": "Reserved Sync dataset capabilities require profile bootstrap.",
    }
    assert "private-bootstrap-id" not in response.text
    assert sync_service.store.list_datasets_for_user("user-1") == []


@pytest.mark.parametrize(
    ("metadata_key", "value", "private_marker"),
    [
        (
            "notes_task_v1",
            {
                "state": "ready",
                "source_cursor": "00000000-0000-4000-8000-000000000001",
            },
            "00000000-0000-4000-8000-000000000001",
        ),
        (
            "notes_task_activity_v1",
            {
                "state": "ready",
                "source_cursor": (
                    "2026-08-13T00:00:00+00:00|"
                    "00000000-0000-4000-8000-000000000011"
                ),
            },
            "00000000-0000-4000-8000-000000000011",
        ),
        ("task_activity_capture_enabled", True, "task_activity_capture_enabled"),
    ],
)
def test_dataset_enroll_endpoint_rejects_forged_task_readiness(
    client: TestClient,
    sync_service: SyncV2Service,
    metadata_key: str,
    value: object,
    private_marker: str,
) -> None:
    response = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": f"forged-{metadata_key}",
            "scope_type": "personal",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": {metadata_key: value},
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "error_code": "sync_reserved_dataset_enrollment",
        "message": "Reserved Sync dataset capabilities require profile bootstrap.",
    }
    assert private_marker not in response.text
    assert sync_service.store.list_datasets_for_user("user-1") == []


def test_dataset_enrollment_and_manifest_never_disclose_task_readiness_metadata(
    client: TestClient,
    sync_service: SyncV2Service,
) -> None:
    server_metadata = {
        "notes_task_v1": {
            "state": "ready",
            "source_cursor": "00000000-0000-4000-8000-000000000001",
            "source_count": 1,
            "source_fingerprint": "a" * 64,
            "reason_code": None,
            "resume_phase": None,
        },
        "notes_task_activity_v1": {
            "state": "ready",
            "source_cursor": (
                "2026-08-13T00:00:00+00:00|"
                "00000000-0000-4000-8000-000000000011"
            ),
            "source_count": 1,
            "source_fingerprint": "b" * 64,
            "reason_code": None,
            "resume_phase": None,
        },
        "task_activity_capture_enabled": True,
    }
    sync_service.store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id="dataset-task-readiness-private",
            owner_user_id="user-1",
            domains=list(M1_SYNC_DOMAINS),
            metadata={"label": "before", **server_metadata},
        )
    )

    enrolled = client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-task-readiness-private",
            "scope_type": "personal",
            "domains": list(M1_SYNC_DOMAINS),
            "encryption_policy": "server_trusted_v1",
            "metadata": {"label": "after"},
        },
    )
    manifest = client.get(
        "/api/v1/sync/restore-manifest",
        params={"dataset_id": "dataset-task-readiness-private"},
    )
    stored = sync_service.store.get_dataset(
        "dataset-task-readiness-private",
        owner_user_id="user-1",
    )

    assert enrolled.status_code == 200
    assert enrolled.json()["metadata"] == {"label": "after"}
    assert manifest.status_code == 200
    assert manifest.json()["datasets"][0]["metadata"] == {"label": "after"}
    assert stored is not None
    assert stored.metadata == {"label": "after", **server_metadata}
    for private_marker in (
        "notes_task_v1",
        "notes_task_activity_v1",
        "task_activity_capture_enabled",
        "00000000-0000-4000-8000-000000000011",
    ):
        assert private_marker not in enrolled.text
        assert private_marker not in manifest.text


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


def test_key_rotation_preview_and_commit_endpoints_are_redacted(
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
    recovery_key = sync_service.store_key_recovery_bundle(
        user_id="user-1",
        dataset_id="dataset-1",
        device_id="device-1",
        key_purpose="dataset_recovery",
        wrapped_key_blob="wrapped:current-secret",
        kdf_metadata={"algorithm": "scrypt", "salt": "current-secret-salt"},
    )
    secret = "wrapped:new-secret-key-material"
    salt = "new-secret-salt"

    preview = client.post(
        "/api/v1/sync/key-rotation/preview",
        json={
            "dataset_id": "dataset-1",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "source_key_record_ids": [recovery_key.key_record_id],
        },
    )
    commit = client.post(
        "/api/v1/sync/key-rotation/commit",
        json={
            "dataset_id": "dataset-1",
            "rotation_id": "rotation-1",
            "target_encryption_policy": "passphrase_wrapped_v1",
            "wrapped_key_blob": secret,
            "kdf_metadata": {"algorithm": "scrypt", "salt": salt},
            "source_key_record_ids": [recovery_key.key_record_id],
            "wrapped_for": "passphrase",
        },
    )

    assert preview.status_code == 200
    assert preview.json()["can_commit"] is True
    assert preview.json()["next_key_epoch"] == 2
    assert preview.json()["affected_key_records"][0]["key_record_id"] == recovery_key.key_record_id
    assert commit.status_code == 200
    assert commit.json()["committed"] is True
    assert commit.json()["new_key_record"]["key_epoch"] == 2
    assert commit.json()["affected_key_records"][0]["superseded_at"] == _clock()
    assert secret not in preview.text
    assert salt not in preview.text
    assert secret not in commit.text
    assert salt not in commit.text


def test_key_rotation_commit_endpoint_validation_error_is_redacted(
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
    secret = "wrapped:new-secret-key-material"
    salt = "new-secret-salt"
    log_messages: list[str] = []
    handler_id = logger.add(
        lambda message: log_messages.append(str(message)),
        format="{message} {extra}",
        level="WARNING",
    )
    try:
        response = client.post(
            "/api/v1/sync/key-rotation/commit",
            json={
                "dataset_id": "dataset-1",
                "rotation_id": "rotation-1",
                "target_encryption_policy": "passphrase_wrapped_v1",
                "wrapped_key_blob": secret,
                "kdf_metadata": {"algorithm": "scrypt", "salt": salt},
                "source_key_record_ids": ["missing-key"],
                "wrapped_for": "passphrase",
            },
        )
    finally:
        logger.remove(handler_id)

    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "sync_validation_failed"
    assert secret not in response.text
    assert salt not in response.text
    rendered_logs = "\n".join(log_messages)
    assert secret not in rendered_logs
    assert salt not in rendered_logs


def test_key_rotation_commit_endpoint_422_validation_error_is_redacted(
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
    secret = "wrapped:new-secret-key-material"
    salt = "new-secret-salt"
    log_messages: list[str] = []
    handler_id = logger.add(
        lambda message: log_messages.append(str(message)),
        format="{message} {extra}",
        level="WARNING",
    )
    try:
        response = client.post(
            "/api/v1/sync/key-rotation/commit",
            json={
                "dataset_id": "dataset-1",
                "rotation_id": "rotation-1",
                "target_encryption_policy": "passphrase_wrapped_v1",
                "wrapped_key_blob": secret,
                "kdf_metadata": f"salt={salt}",
                "source_key_record_ids": ["missing-key"],
                "wrapped_for": "passphrase",
            },
        )
    finally:
        logger.remove(handler_id)

    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "sync_validation_failed"
    assert secret not in response.text
    assert salt not in response.text
    rendered_logs = "\n".join(log_messages)
    assert secret not in rendered_logs
    assert salt not in rendered_logs


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
            "metadata": {"label": "ordinary-enrollment"},
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


def test_public_pull_converter_strips_only_server_local_organization_routing() -> None:
    routing_metadata = {
        "notes_keyword_merge_response": {
            "source_keyword_id": 41,
            "target_keyword_id": 42,
        },
        "notes_folder_origin_provenance": {
            "operation": "source_upsert",
            "source_id": 73,
        },
        "notes_ingestion_expected_product_version": 19,
        "bootstrap_capture": True,
        "provenance": {"origin": "synthetic-client"},
    }
    envelope = SyncEnvelope(
        dataset_id="dataset-1",
        client_envelope_id="env-internal-routing",
        domain="notes.keyword",
        operation="tombstone",
        object_id="11111111-1111-4111-8111-111111111111",
        server_cursor=1,
        payload={},
        payload_hash="sha256:internal-routing",
        routing_metadata=routing_metadata,
        mutation_group_id="group-public-1",
        mutation_step=0,
        mutation_step_count=1,
        mutation_plan_hash="a" * 64,
    )

    public = sync_endpoint._api_envelope_from_core(
        envelope,
        encryption_policy="server_trusted_v1",
    )

    assert public.routing_metadata == {
        "bootstrap_capture": True,
        "provenance": {"origin": "synthetic-client"},
    }
    assert public.mutation_group_id == "group-public-1"
    assert public.mutation_step == 0
    assert public.mutation_step_count == 1
    assert public.mutation_plan_hash == "a" * 64
    assert envelope.routing_metadata == routing_metadata


def test_pull_endpoint_preserves_group_metadata_without_internal_routing(
    tmp_path: Path,
) -> None:
    service = _build_service(tmp_path)
    client = _client_for_service(service)
    assert client.post(
        "/api/v1/sync/devices/register",
        json={
            "device_id": "device-1",
            "display_name": "Laptop",
            "client_type": "chatbook",
        },
    ).status_code == 200
    assert client.post(
        "/api/v1/sync/datasets/enroll",
        json={
            "dataset_id": "dataset-1",
            "domains": ["notes.note"],
            "encryption_policy": "server_trusted_v1",
        },
    ).status_code == 200
    service.store.insert_envelope(
        SyncEnvelopeCreate(
            dataset_id="dataset-1",
            client_envelope_id="env-grouped-pull",
            domain="notes.note",
            operation="upsert",
            object_id="note-grouped",
            device_id="server-origin",
            client_sequence=1,
            object_revision=1,
            payload={"title": "Grouped", "content": "Body"},
            payload_hash="sha256:grouped-pull",
            routing_metadata={
                "notes_keyword_merge_response": {
                    "source_keyword_id": 41,
                    "target_keyword_id": 42,
                },
                "bootstrap_capture": True,
            },
            mutation_group_id="group-pull-1",
            mutation_step=0,
            mutation_step_count=1,
            mutation_plan_hash="b" * 64,
        )
    )

    response = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "cursor": "0",
            "domain": "notes.note",
        },
    )

    assert response.status_code == 200
    envelope = response.json()["envelopes"][0]
    assert envelope["mutation_group_id"] == "group-pull-1"
    assert envelope["mutation_step"] == 0
    assert envelope["mutation_step_count"] == 1
    assert envelope["mutation_plan_hash"] == "b" * 64
    assert envelope["routing_metadata"] == {"bootstrap_capture": True}


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
        ("env-failed", 2, 1, "failed"),
    ]
    assert accepted[1]["apply_error_code"] == "projection_failed"
    assert "replayable" in accepted[1]["apply_error_message"]
    assert pulled.status_code == 200
    failed = pulled.json()["envelopes"][1]
    assert failed["client_envelope_id"] == "env-failed"
    assert failed["object_revision"] == 1
    assert failed["apply_status"] == "failed"
    assert failed["apply_error_code"] == "projection_failed"
    assert "replayable" in failed["apply_error_message"]


def test_pull_endpoint_accepts_m1_contract_limit_and_echo_aliases(
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
    pushed = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "base_server_cursor": 0,
            "options": {"stop_on_conflict": False},
            "envelopes": [
                _note_envelope_json(client_envelope_id="env-note-1", payload_hash="sha256:note-1"),
                _note_envelope_json(
                    client_envelope_id="env-note-2",
                    object_id="note-2",
                    client_sequence=2,
                    payload_hash="sha256:note-2",
                ),
            ],
        },
    )

    pulled = client.get(
        "/api/v1/sync/pull",
        params={
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "cursor": "0",
            "domain": "notes.note",
            "limit": "1",
            "include_same_device_echoes": "true",
        },
    )

    assert pushed.status_code == 200
    assert pulled.status_code == 200
    assert [item["client_envelope_id"] for item in pulled.json()["envelopes"]] == ["env-note-1"]
    assert pulled.json()["has_more"] is True


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


def test_blob_chunk_endpoint_rejects_oversized_body_before_buffering(
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
    payload = b"12345678"
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
            "chunk_size": len(payload),
            "chunk_count": 1,
        },
    )
    assert create_response.status_code == 200
    upload_id = create_response.json()["upload_id"]

    response = client.put(
        f"/api/v1/sync/blob-uploads/{upload_id}/chunks/0",
        params={
            "dataset_id": "dataset-1",
            "offset_bytes": 0,
            "chunk_hash": _sha256(payload + b"x"),
        },
        content=payload + b"x",
        headers={"content-type": "application/octet-stream"},
    )

    assert response.status_code == 413
    assert response.json()["detail"]["error_code"] == "sync_blob_chunk_too_large"


def test_blob_completion_preserves_staged_chunks_when_db_commit_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _build_service(tmp_path, supports_attachments=True)
    service.register_device(
        user_id="user-1",
        device_id="device-1",
        display_name="Laptop",
        client_type="chatbook",
    )
    service.enroll_dataset(user_id="user-1", dataset_id="dataset-1", domains=["notes.note", "attachment.ref"])
    payload = b"retry me"
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
        chunk_size=len(payload),
        chunk_count=1,
    )
    service.upload_blob_chunk(
        user_id="user-1",
        dataset_id="dataset-1",
        upload_id=session.upload_id,
        chunk_index=0,
        offset_bytes=0,
        chunk_payload=payload,
        chunk_hash=_sha256(payload),
    )

    def fail_complete_blob_upload(_blob):
        raise SyncStoreError("database commit failed")

    monkeypatch.setattr(service.store, "complete_blob_upload", fail_complete_blob_upload)

    with pytest.raises(SyncStoreError):
        service.complete_blob_upload(
            user_id="user-1",
            dataset_id="dataset-1",
            upload_id=session.upload_id,
        )

    assert (tmp_path / "sync_blobs" / "_uploads" / session.upload_id / "0.part").exists()


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
    assert service.blob_store is not None

    def fail_read_blob(_storage_key: str) -> bytes:
        raise AssertionError("download endpoints should stream blob content")

    service.blob_store.read_blob = fail_read_blob  # type: ignore[method-assign]

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


@pytest.fixture()
def factory_personal_context_service(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SyncV2Service:
    """Build the production Sync composition with only temporary server storage."""

    monkeypatch.setenv("USER_DB_BASE_DIR", str(tmp_path / "user_databases"))
    monkeypatch.setenv(
        "TLDW_PERSONAL_CONTEXT_MASTER_KEY",
        base64.b64encode(b"p" * 32).decode("ascii"),
    )
    monkeypatch.setenv("SYNC_V2_AT_REST_ENCRYPTION_MODE", "managed_storage")
    monkeypatch.setenv("SYNC_V2_SERVER_TRUSTED_ENABLED", "true")
    monkeypatch.setenv(
        "SYNC_V2_PULL_TOKEN_SIGNING_SECRET",
        "personal-context-endpoint-test-signing-secret",
    )
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    for cached_factory in (
        sync_v2_factory._sync_v2_store_for_user,
        sync_v2_factory._chacha_notes_db_for_user,
        sync_v2_factory._sync_v2_blob_store_for_user,
        sync_v2_factory._personal_context_service_for_user,
    ):
        cached_factory.cache_clear()
    try:
        yield sync_v2_factory.sync_v2_service_for_user("101")
    finally:
        for cached_factory in (
            sync_v2_factory._sync_v2_store_for_user,
            sync_v2_factory._chacha_notes_db_for_user,
            sync_v2_factory._sync_v2_blob_store_for_user,
            sync_v2_factory._personal_context_service_for_user,
        ):
            cached_factory.cache_clear()


def _registered_personal_context_device_payload(public_key: rsa.RSAPublicKey) -> dict[str, object]:
    """Return the public endpoint payload for one Personal Context-capable device."""

    return {
        "device_id": "pc-device",
        "display_name": "Personal Context test device",
        "client_type": "chatbook",
        "supported_domains": list(PERSONAL_CONTEXT_SYNC_DOMAINS),
        "supported_adapter_versions": {
            domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        },
        "capabilities": {
            "personal_context_wrapping_public_key": public_key.public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            ).decode("utf-8"),
        },
    }


def _assert_personal_context_error_is_redacted(response, reason_code: str) -> None:
    """Assert the typed error exposes only the stable public reason shape."""

    body = response.json()
    assert body["detail"]["error_code"] == reason_code
    assert set(body["detail"]) == {"error_code", "message"}
    for secret in (
        "bootstrap-private-canary",
        "personal-context-integrity-v1",
        "ciphertext",
        "wrapped_key_blob",
    ):
        assert secret not in response.text


def test_personal_context_endpoints_use_real_factory_bootstrap_and_complete_flow(
    factory_personal_context_service: SyncV2Service,
) -> None:
    """The typed flow plans read-only state and materializes it only on completion."""

    client = _client_for_factory_service(factory_personal_context_service)
    missing = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "missing-device", "required_schema_version": 1},
    )
    assert missing.status_code == 404
    _assert_personal_context_error_is_redacted(
        missing, "personal_context_device_unavailable"
    )

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    registration = client.post(
        "/api/v1/sync/devices/register",
        json=_registered_personal_context_device_payload(private_key.public_key()),
    )
    assert registration.status_code == 200
    assert registration.json()["device_id"] == "pc-device"

    bootstrap_request = {
        "device_id": "pc-device",
        "required_schema_version": 1,
        "required_quotas": {"max_record_bytes": 16_384},
    }
    assert "authority_id" not in bootstrap_request
    bootstrap = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json=bootstrap_request,
    )
    assert bootstrap.status_code == 200, bootstrap.text
    body = bootstrap.json()
    assert body["authority_id"] == "tldw-server"
    assert body["dataset_id"]
    assert body["cursor"].startswith("personal-context-bootstrap-v1:")
    assert body["sync_transport_cursor"]
    assert body["sync_transport_cursor"] != body["cursor"]
    assert body["manifest"]["profile_id"]
    assert body["wrapped_key_blob"].startswith("rsa-oaep-sha256:")

    canonical = factory_personal_context_service.personal_context_service_resolver("101")
    with pytest.raises((KeyError, ProfileStorageLockedError)):
        canonical.get_manifest()
    assert canonical._repository.profile_ids() == ()
    with canonical._repository.database.transaction() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM personal_context_object_versions"
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM personal_context_object_heads"
        ).fetchone()[0] == 0
    retry = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json=bootstrap_request,
    )
    assert retry.status_code == 200
    assert retry.json()["manifest"] == body["manifest"]
    assert retry.json()["scopes"] == body["scopes"]
    assert retry.json()["cursor"] == body["cursor"]
    assert retry.json()["sync_transport_cursor"]

    integrity_key_id = body["integrity_key_id"]
    integrity_key = canonical._repository.sync_integrity_key(
        body["manifest"]["profile_id"]
    )[1]
    ciphertext = base64.urlsafe_b64decode(body["wrapped_key_blob"].split(":", 1)[1])
    assert private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=f"personal-context:{integrity_key_id}".encode(),
        ),
    ) == integrity_key

    stale_completion = client.post(
        "/api/v1/sync/personal-context/complete",
        json={
            "device_id": "pc-device",
            "dataset_id": body["dataset_id"],
            "bootstrap_cursor": "personal-context-bootstrap-v1:stale",
        },
    )
    assert stale_completion.status_code == 409
    _assert_personal_context_error_is_redacted(
        stale_completion, "personal_context_bootstrap_cursor_stale"
    )

    completion = client.post(
        "/api/v1/sync/personal-context/complete",
        json={
            "device_id": "pc-device",
            "dataset_id": body["dataset_id"],
            "bootstrap_cursor": body["cursor"],
        },
    )
    assert completion.status_code == 204
    manifest = canonical.get_manifest()
    assert manifest.model_dump(mode="json") == body["manifest"]
    assert [item.model_dump(mode="json") for item in canonical.list_scopes()] == body[
        "scopes"
    ]
    assert factory_personal_context_service.store.has_personal_context_link_receipt(
        user_id="101",
        dataset_id=body["dataset_id"],
        device_id="pc-device",
        profile_id=manifest.profile_id,
        integrity_key_id=integrity_key_id,
        purge_generation=body["purge_generation"],
    )

    manifest_payload = manifest.model_dump(mode="json")
    manifest_canonical = canonical_json_bytes(manifest_payload)
    manifest_tag = hmac.new(integrity_key, manifest_canonical, hashlib.sha256)
    push = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": body["dataset_id"],
            "device_id": "pc-device",
            "envelopes": [
                {
                    "dataset_id": body["dataset_id"],
                    "client_envelope_id": "pc-device:manifest:1",
                    "device_id": "pc-device",
                    "domain": "personal_context.manifest",
                    "operation": "upsert",
                    "object_id": manifest.profile_id,
                    "adapter_version": 1,
                    "schema_version": 1,
                    "payload": manifest_payload,
                    "payload_hash": f"hmac-sha256-v1:{manifest_tag.hexdigest()}",
                    "payload_size_bytes": len(manifest_canonical),
                    "routing_metadata": {
                        "integrity_key_id": integrity_key_id,
                        "profile_id": manifest.profile_id,
                        "purge_generation": body["purge_generation"],
                    },
                    "encryption_metadata": {"policy": "server_trusted_v1"},
                }
            ],
        },
    )
    assert push.status_code == 200, push.text
    assert [item["client_envelope_id"] for item in push.json()["accepted"]] == [
        "pc-device:manifest:1"
    ]


def test_personal_context_bootstrap_response_exposes_effective_zero_quota(
    factory_personal_context_service: SyncV2Service,
) -> None:
    client = _client_for_factory_service(factory_personal_context_service)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert client.post(
        "/api/v1/sync/devices/register",
        json=_registered_personal_context_device_payload(private_key.public_key()),
    ).status_code == 200

    response = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json={
            "device_id": "pc-device",
            "required_schema_version": 1,
            "required_quotas": {"future_sync_quota": 0},
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["quotas"]["future_sync_quota"] == 0


def test_personal_context_complete_endpoint_rejects_real_stale_integrity_binding(
    factory_personal_context_service: SyncV2Service,
) -> None:
    """Completion fails closed when the Sync binding no longer names the snapshot key."""

    client = _client_for_factory_service(factory_personal_context_service)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    registration = client.post(
        "/api/v1/sync/devices/register",
        json=_registered_personal_context_device_payload(private_key.public_key()),
    )
    assert registration.status_code == 200
    bootstrap = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "pc-device", "required_schema_version": 1},
    )
    assert bootstrap.status_code == 200, bootstrap.text
    body = bootstrap.json()
    canonical = factory_personal_context_service.personal_context_service_resolver("101")
    stale_key_id = "personal-context-integrity-vstale"
    dataset = factory_personal_context_service.store.get_dataset(body["dataset_id"])
    assert dataset is not None
    binding = dataset.metadata["personal_context"]
    factory_personal_context_service.store.bind_personal_context_dataset(
        dataset_id=dataset.dataset_id,
        user_id=dataset.owner_user_id,
        expected_binding=dict(binding),
        profile_id=str(binding["profile_id"]),
        authority_id=str(binding["authority_id"]),
        integrity_key_id=stale_key_id,
        purge_generation=int(binding["purge_generation"]),
        link_state=str(binding["link_state"]),
    )

    completion = client.post(
        "/api/v1/sync/personal-context/complete",
        json={
            "device_id": "pc-device",
            "dataset_id": body["dataset_id"],
            "bootstrap_cursor": body["cursor"],
        },
    )

    assert completion.status_code == 409, completion.text
    _assert_personal_context_error_is_redacted(
        completion, "personal_context_link_binding_stale"
    )
    assert canonical._repository.profile_ids() == ()
    assert not factory_personal_context_service.store.has_personal_context_link_receipt(
        user_id="101",
        dataset_id=body["dataset_id"],
        device_id="pc-device",
        profile_id=body["manifest"]["profile_id"],
        integrity_key_id=stale_key_id,
        purge_generation=body["purge_generation"],
    )


def test_personal_context_complete_endpoint_maps_real_receipt_cas_staleness(
    factory_personal_context_service: SyncV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A binding transition during receipt CAS reaches the typed 409 response."""

    client = _client_for_factory_service(factory_personal_context_service)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    registration = client.post(
        "/api/v1/sync/devices/register",
        json=_registered_personal_context_device_payload(private_key.public_key()),
    )
    assert registration.status_code == 200
    bootstrap = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "pc-device", "required_schema_version": 1},
    )
    assert bootstrap.status_code == 200, bootstrap.text
    body = bootstrap.json()
    stale_key_id = "personal-context-integrity-vstale"
    original_receipt = factory_personal_context_service.store.complete_personal_context_link_receipt

    def transition_binding_then_write_receipt(**values: object) -> None:
        dataset = factory_personal_context_service.store.get_dataset(body["dataset_id"])
        assert dataset is not None
        binding = dataset.metadata["personal_context"]
        factory_personal_context_service.store.bind_personal_context_dataset(
            dataset_id=dataset.dataset_id,
            user_id=dataset.owner_user_id,
            expected_binding=dict(binding),
            profile_id=str(binding["profile_id"]),
            authority_id=str(binding["authority_id"]),
            integrity_key_id=stale_key_id,
            purge_generation=int(binding["purge_generation"]),
            link_state=str(binding["link_state"]),
        )
        original_receipt(**values)

    monkeypatch.setattr(
        factory_personal_context_service.store,
        "complete_personal_context_link_receipt",
        transition_binding_then_write_receipt,
    )
    completion = client.post(
        "/api/v1/sync/personal-context/complete",
        json={
            "device_id": "pc-device",
            "dataset_id": body["dataset_id"],
            "bootstrap_cursor": body["cursor"],
        },
    )

    assert completion.status_code == 409, completion.text
    _assert_personal_context_error_is_redacted(
        completion, "personal_context_link_binding_stale"
    )
    assert not factory_personal_context_service.store.has_personal_context_link_receipt(
        user_id="101",
        dataset_id=body["dataset_id"],
        device_id="pc-device",
        profile_id=body["manifest"]["profile_id"],
        integrity_key_id=stale_key_id,
        purge_generation=body["purge_generation"],
    )


def test_personal_context_push_surfaces_receipt_storage_failure_without_reconciliation_hint(
    factory_personal_context_service: SyncV2Service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Receipt lookup outages are public operational failures, not false admission hints."""

    client = _client_for_factory_service(factory_personal_context_service)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert client.post(
        "/api/v1/sync/devices/register",
        json=_registered_personal_context_device_payload(private_key.public_key()),
    ).status_code == 200
    bootstrap = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "pc-device", "required_schema_version": 1},
    )
    assert bootstrap.status_code == 200, bootstrap.text
    body = bootstrap.json()
    assert client.post(
        "/api/v1/sync/personal-context/complete",
        json={
            "device_id": "pc-device",
            "dataset_id": body["dataset_id"],
            "bootstrap_cursor": body["cursor"],
        },
    ).status_code == 204

    original_execute = factory_personal_context_service.store.db.execute

    def fail_receipt_lookup(query: str, *args: object, **kwargs: object):
        if "sync_personal_context_link_receipts" in query:
            raise SyncStoreError("receipt storage unavailable")
        return original_execute(query, *args, **kwargs)

    monkeypatch.setattr(
        factory_personal_context_service.store.db,
        "execute",
        fail_receipt_lookup,
    )
    push = client.post(
        "/api/v1/sync/push",
        json={
            "dataset_id": body["dataset_id"],
            "device_id": "pc-device",
            "envelopes": [
                {
                    "dataset_id": body["dataset_id"],
                    "client_envelope_id": "pc-device:storage-failure:1",
                    "device_id": "pc-device",
                    "domain": "personal_context.manifest",
                    "operation": "upsert",
                    "object_id": body["manifest"]["profile_id"],
                    "adapter_version": 1,
                    "schema_version": 1,
                    "payload": {},
                    "payload_hash": "sha256:storage-failure",
                    "payload_size_bytes": 2,
                    "routing_metadata": {},
                    "encryption_metadata": {"policy": "server_trusted_v1"},
                }
            ],
        },
    )

    assert push.status_code == 500, push.text
    assert push.json()["detail"] == {
        "error_code": "sync_store_error",
        "message": "Internal sync storage error while processing request.",
    }
    assert "receipt storage unavailable" not in push.text


@pytest.mark.parametrize(
    ("reason_code", "expected_status"),
    [
        ("personal_context_bootstrap_unavailable", 503),
        ("personal_context_device_unavailable", 404),
        ("personal_context_authority_invalid", 400),
        ("personal_context_schema_incompatible", 409),
        ("personal_context_quota_incompatible", 409),
        ("personal_context_purge_generation_stale", 409),
        ("personal_context_authority_mismatch", 409),
        ("personal_context_snapshot_unavailable", 503),
        ("personal_context_snapshot_unstable", 409),
        ("personal_context_capability_unavailable", 503),
        ("personal_context_key_custody_unavailable", 503),
        ("personal_context_link_unavailable", 409),
        ("personal_context_link_binding_stale", 409),
        ("personal_context_bootstrap_cursor_stale", 409),
    ],
)
def test_personal_context_bootstrap_endpoint_maps_redacted_reason_codes(
    tmp_path: Path, reason_code: str, expected_status: int,
) -> None:
    """Authenticated API failures retain a stable code but no canonical body."""
    service = _build_service(tmp_path)

    def fail_bootstrap(**_kwargs: object):
        raise PersonalContextBootstrapError(reason_code)

    service.bootstrap_personal_context = fail_bootstrap  # type: ignore[method-assign]
    response = _client_for_service(service).post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "missing-device", "required_schema_version": 1},
    )
    assert response.status_code == expected_status
    _assert_personal_context_error_is_redacted(response, reason_code)


@pytest.mark.parametrize(
    ("request_patch", "reason_code", "attention"),
    [
        (
            {"required_schema_version": 2},
            "personal_context_schema_incompatible",
            {
                "kind": "schema_incompatible",
                "required_schema_version": 2,
                "server_min_schema_version": 1,
                "server_max_schema_version": 1,
            },
        ),
        (
            {"required_quotas": {"max_record_bytes": 16_385}},
            "personal_context_quota_incompatible",
            {
                "kind": "quota_incompatible",
                "required_quotas": {"max_record_bytes": 16_385},
                "available_quotas": {
                    "max_record_bytes": 16_384,
                    "max_search_results": 20,
                    "max_proposals_per_turn": 5,
                    "max_proposals_per_session": 25,
                    "max_unresolved_proposals": 200,
                },
                "insufficient_quotas": ["max_record_bytes"],
            },
        ),
        (
            {"required_quotas": {"future_sync_quota": 1}},
            "personal_context_quota_incompatible",
            {
                "kind": "quota_incompatible",
                "required_quotas": {"future_sync_quota": 1},
                "available_quotas": {
                    "future_sync_quota": 0,
                    "max_record_bytes": 16_384,
                    "max_search_results": 20,
                    "max_proposals_per_turn": 5,
                    "max_proposals_per_session": 25,
                    "max_unresolved_proposals": 200,
                },
                "insufficient_quotas": ["future_sync_quota"],
            },
        ),
        (
            {"expected_purge_generation": 1},
            "personal_context_purge_generation_stale",
            {
                "kind": "purge_generation_mismatch",
                "expected_purge_generation": 1,
                "current_purge_generation": 0,
            },
        ),
    ],
)
def test_personal_context_bootstrap_endpoint_exposes_exact_content_free_attention(
    factory_personal_context_service: SyncV2Service,
    request_patch: dict[str, object],
    reason_code: str,
    attention: dict[str, object],
) -> None:
    """Review blockers disclose exact numeric facts but no canonical body or key."""

    client = _client_for_factory_service(factory_personal_context_service)
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    assert client.post(
        "/api/v1/sync/devices/register",
        json=_registered_personal_context_device_payload(private_key.public_key()),
    ).status_code == 200

    response = client.post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "pc-device", **request_patch},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "error_code": reason_code,
        "message": response.json()["detail"]["message"],
        "attention": attention,
    }
    for secret in (
        '"wrapped_key_blob":',
        '"manifest":',
        '"scopes":',
        '"records":',
        '"proposals":',
        "ciphertext",
    ):
        assert secret not in response.text


@pytest.mark.parametrize(
    ("reason_code", "attention"),
    [
        (
            "personal_context_schema_incompatible",
            {
                "kind": "schema_incompatible",
                "required_schema_version": 2,
                "server_min_schema_version": 1,
                "server_max_schema_version": 1,
                "manifest": {"payload": "canonical-profile-canary"},
                "wrapped_key_blob": "wrapped-integrity-key-canary",
                "ciphertext": "ciphertext-canary",
            },
        ),
        (
            "personal_context_quota_incompatible",
            {
                "kind": "schema_incompatible",
                "required_schema_version": 2,
                "server_min_schema_version": 1,
                "server_max_schema_version": 1,
            },
        ),
    ],
)
def test_personal_context_bootstrap_endpoint_omits_untrusted_attention(
    tmp_path: Path,
    reason_code: str,
    attention: dict[str, object],
) -> None:
    """Malformed or mismatched attention never crosses the HTTP boundary."""

    service = _build_service(tmp_path)

    def fail_bootstrap(**_kwargs: object) -> None:
        raise PersonalContextBootstrapError(reason_code, attention=attention)

    service.bootstrap_personal_context = fail_bootstrap  # type: ignore[method-assign]
    response = _client_for_service(service).post(
        "/api/v1/sync/personal-context/bootstrap",
        json={"device_id": "device-a", "required_schema_version": 1},
    )

    assert response.status_code == 409
    assert set(response.json()["detail"]) == {"error_code", "message"}
    for canary in (
        "canonical-profile-canary",
        "wrapped-integrity-key-canary",
        "ciphertext-canary",
    ):
        assert canary not in response.text


@pytest.mark.parametrize(
    "reason_code",
    [
        "personal_context_bootstrap_cursor_stale",
        "personal_context_link_binding_stale",
    ],
)
def test_personal_context_completion_endpoint_maps_stale_reason_codes_without_content(
    tmp_path: Path,
    reason_code: str,
) -> None:
    service = _build_service(tmp_path)

    def fail_completion(**_kwargs: object) -> None:
        raise PersonalContextBootstrapError(reason_code)

    service.complete_personal_context_link = fail_completion  # type: ignore[method-assign]
    response = _client_for_service(service).post(
        "/api/v1/sync/personal-context/complete",
        json={"device_id": "device-a", "dataset_id": "dataset-a", "bootstrap_cursor": "stale"},
    )
    assert response.status_code == 409
    _assert_personal_context_error_is_redacted(response, reason_code)
