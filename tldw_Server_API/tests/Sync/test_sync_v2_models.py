import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.endpoints.sync import _core_envelope_from_api
from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    SYNC_V2_MAX_PUSH_ENVELOPES,
    SyncAttachmentUploadRequest,
    SyncAttachmentUploadResponse,
    SyncBlobChunkUploadResponse,
    SyncBlobDownloadManifestResponse,
    SyncBlobUploadCompleteResponse,
    SyncBlobUploadCreateRequest,
    SyncBlobUploadSessionResponse,
    SyncCapabilitiesResponse,
    SyncConflictResolveRequest,
    SyncDatasetEnrollRequest,
    SyncPushRequest,
    SyncPushResponse,
    SyncRestoreCompletenessResponse,
    SyncV2Envelope,
)
from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope as CoreSyncEnvelope

M1_DOMAINS = ["notes.note", "chat.conversation", "chat.message", "attachment.ref"]


def _m1_envelope_payload(**overrides):
    payload = {
        "client_envelope_id": "env-1",
        "dataset_id": "dataset-1",
        "device_id": "device-1",
        "client_profile_id": "chatbook-profile-1",
        "client_sequence": 17,
        "base_server_cursor": None,
        "base_object_revision": None,
        "base_object_hash": None,
        "server_cursor": None,
        "domain": "notes.note",
        "object_id": "note-1",
        "operation": "upsert",
        "parent_id": None,
        "schema_version": 1,
        "payload": {"title": "Research note"},
        "payload_hash": "sha256:test",
        "object_revision": None,
        "created_at_client": "2026-05-23T18:12:44Z",
        "received_at_server": None,
        "deleted": False,
        "encryption_metadata": {"policy": "server_trusted_v1"},
    }
    payload.update(overrides)
    return payload


def test_capabilities_advertise_only_m1_domains_and_server_trusted_encryption():
    capabilities = SyncCapabilitiesResponse()

    assert capabilities.protocol_version == "sync-v2-m1"
    assert capabilities.domains == M1_DOMAINS
    assert capabilities.operations == {
        "notes.note": ["upsert", "tombstone"],
        "chat.conversation": ["upsert", "tombstone"],
        "chat.message": ["append", "tombstone"],
        "attachment.ref": ["upsert", "tombstone"],
    }
    assert capabilities.encryption["policy"] == "server_trusted_v1"
    assert capabilities.encryption["ready"] is True
    assert capabilities.blob_transfer == {"supported": False}
    assert "client_private_v1" not in capabilities.model_dump_json()


def test_capabilities_normalize_legacy_supported_domains_to_m1_defaults():
    capabilities = SyncCapabilitiesResponse.model_validate(
        {"supported_domains": ["notes", "chat", "source_cache", "media"]}
    )

    assert capabilities.domains == M1_DOMAINS


def test_dataset_enroll_request_defaults_to_m1_personal_server_trusted_dataset():
    request = SyncDatasetEnrollRequest.model_validate({})

    assert request.scope_type == "personal"
    assert request.domains == M1_DOMAINS
    assert request.encryption_policy == "server_trusted_v1"


def test_sync_envelope_accepts_m1_fields_and_legacy_transition_aliases():
    envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            entity_id="note-from-old-client",
            object_id=None,
            server_sequence=101,
            payload=None,
            payload_clear={"title": "Legacy payload alias"},
        )
    )

    assert envelope.object_id == "note-from-old-client"
    assert envelope.entity_id == "note-from-old-client"
    assert envelope.server_cursor == 101
    assert envelope.server_sequence == 101
    assert envelope.payload == {"title": "Legacy payload alias"}
    assert envelope.payload_clear == {"title": "Legacy payload alias"}
    assert envelope.client_sequence == 17
    assert envelope.encryption_metadata == {"policy": "server_trusted_v1"}


def test_sync_envelope_rejects_domains_outside_m1_contract():
    with pytest.raises(ValidationError) as exc_info:
        SyncV2Envelope.model_validate(_m1_envelope_payload(domain="media"))

    assert "media" in str(exc_info.value)


def test_sync_envelope_rejects_client_private_as_m1_default():
    with pytest.raises(ValidationError) as exc_info:
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(encryption_metadata={"policy": "client_private_v1"})
        )

    assert "client_private_v1" in str(exc_info.value)


def test_sync_envelope_rejects_non_object_payload_as_validation_error():
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(_m1_envelope_payload(payload=[]))


def test_whole_object_tombstone_requires_base_metadata():
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(
                operation="tombstone",
                deleted=True,
                base_server_cursor=None,
                base_object_revision=None,
                base_object_hash=None,
            )
        )


def test_whole_object_update_requires_complete_base_metadata():
    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(base_server_cursor=98, base_object_revision=4)
        )


def test_chat_message_append_requires_stable_message_id_and_payload_hash():
    valid = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            domain="chat.message",
            operation="append",
            object_id="msg-1",
            parent_id="conversation-1",
            payload_hash="sha256:message",
        )
    )

    assert valid.object_id == "msg-1"
    assert valid.payload_hash == "sha256:message"

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(
                domain="chat.message",
                operation="append",
                object_id="",
                payload_hash="sha256:message",
            )
        )

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(
            _m1_envelope_payload(
                domain="chat.message",
                operation="append",
                object_id="msg-1",
                payload_hash="",
            )
        )


def test_core_sync_envelope_accepts_server_sequence_transition_alias():
    envelope = CoreSyncEnvelope(
        server_sequence=42,
        dataset_id="dataset-1",
        client_envelope_id="env-1",
        domain="notes.note",
        operation="upsert",
        object_id="note-1",
        payload={"title": "Alias"},
        payload_hash="sha256:alias",
    )

    assert envelope.server_cursor == 42
    assert envelope.server_sequence == 42


def test_core_envelope_mapping_strips_api_only_server_fields_before_persistence():
    api_envelope = SyncV2Envelope.model_validate(
        _m1_envelope_payload(
            envelope_id="srv_env_000000000001",
            server_cursor=1,
            received_at_server="2026-05-23T18:12:46Z",
            status="accepted",
            apply_status="applied",
        )
    )

    core_envelope = _core_envelope_from_api(api_envelope)

    assert core_envelope.client_envelope_id == "env-1"
    assert core_envelope.object_id == "note-1"
    assert core_envelope.server_cursor is None
    assert core_envelope.received_at_server is None
    assert core_envelope.status == "accepted"
    assert core_envelope.apply_status == "pending"


def test_conflict_resolution_request_uses_locked_m1_batch_shape():
    request = SyncConflictResolveRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "resolutions": [
                {
                    "conflict_id": "conflict-note-1",
                    "action": "overwrite",
                    "resolution_envelope": None,
                },
                {
                    "conflict_id": "conflict-note-2",
                    "action": "duplicate_rename",
                    "resolution_envelope": _m1_envelope_payload(
                        client_envelope_id="env-resolution",
                        object_id="note-copy",
                        payload_hash="sha256:copy",
                    ),
                },
                {
                    "conflict_id": "conflict-note-3",
                    "action": "skip",
                },
            ],
        }
    )

    assert request.dataset_id == "dataset-1"
    assert request.device_id == "device-1"
    assert [resolution.action for resolution in request.resolutions] == [
        "overwrite",
        "duplicate_rename",
        "skip",
    ]
    assert request.resolutions[1].resolution_envelope.object_id == "note-copy"


@pytest.mark.parametrize(
    "legacy_action",
    ["accept_local", "accept_remote", "merge", "dismiss"],
)
def test_conflict_resolution_request_rejects_pre_m1_actions(legacy_action: str):
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": legacy_action,
                    }
                ],
            }
        )


def test_conflict_resolution_request_rejects_legacy_single_conflict_shape():
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "conflict_id": "conflict-1",
                "action": "accept_local",
                "resolved_by_device_id": "device-1",
            }
        )


def test_conflict_resolution_request_rejects_duplicate_rename_without_envelope():
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": "duplicate_rename",
                    }
                ],
            }
        )


def test_conflict_resolution_request_rejects_skip_with_resolution_envelope():
    with pytest.raises(ValidationError):
        SyncConflictResolveRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "resolutions": [
                    {
                        "conflict_id": "conflict-1",
                        "action": "skip",
                        "resolution_envelope": _m1_envelope_payload(
                            client_envelope_id="env-skip-resolution",
                            object_id="note-1",
                            payload_hash="sha256:skip-resolution",
                        ),
                    }
                ],
            }
        )


def test_conflict_batch_endpoint_resolves_locked_m1_request_shape():
    from tldw_Server_API.app.api.v1.API_Deps.auth_deps import User
    from tldw_Server_API.app.api.v1.endpoints.sync import resolve_sync_v2_conflicts
    from tldw_Server_API.app.core.Sync.v2.models import SyncConflict

    class FakeSyncService:
        def __init__(self):
            self.calls = []

        def resolve_conflict(self, **kwargs):
            self.calls.append(kwargs)
            public_action = kwargs["action"]
            server_cursor = 123 if public_action == "duplicate_rename" else 12
            envelope_id = (
                "srv_env_000000000123"
                if public_action == "duplicate_rename"
                else kwargs.get("resolved_by_envelope_id")
            )
            return SyncConflict(
                conflict_id=kwargs["conflict_id"],
                dataset_id="dataset-1",
                domain="notes.note",
                object_id="note-1",
                conflict_type="version_divergence",
                status="dismissed" if public_action == "skip" else "resolved",
                base_envelope_id=None,
                local_envelope_id=None,
                remote_envelope_id=None,
                server_cursor=server_cursor,
                metadata={},
                created_at="2026-05-23T18:12:44Z",
                resolved_at="2026-05-23T18:13:44Z",
                resolved_by_device_id=kwargs["resolved_by_device_id"],
                resolved_by_envelope_id=envelope_id,
                resolution_action=public_action,
            )

    service = FakeSyncService()
    request = SyncConflictResolveRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "resolutions": [
                {"conflict_id": "conflict-1", "action": "overwrite"},
                {
                    "conflict_id": "conflict-2",
                    "action": "duplicate_rename",
                    "resolution_envelope": _m1_envelope_payload(
                        client_envelope_id="env-resolution",
                        object_id="note-copy",
                        payload_hash="sha256:copy",
                    ),
                },
                {"conflict_id": "conflict-3", "action": "skip"},
            ],
        }
    )

    response = resolve_sync_v2_conflicts(
        request,
        user=User(id="user-1", username="user-1"),
        service=service,
    )

    assert response.dataset_id == "dataset-1"
    assert [item.action for item in response.resolved] == [
        "overwrite",
        "duplicate_rename",
        "skip",
    ]
    assert response.rejected == []
    assert [call["conflict_id"] for call in service.calls] == [
        "conflict-1",
        "conflict-2",
        "conflict-3",
    ]
    assert [call["dataset_id"] for call in service.calls] == [
        "dataset-1",
        "dataset-1",
        "dataset-1",
    ]
    assert service.calls[1]["resolution_envelope"].client_envelope_id == "env-resolution"
    assert service.calls[1].get("resolved_by_envelope_id") is None
    assert service.calls[2]["action"] == "skip"
    response_payload = response.model_dump(exclude_none=True)
    assert response_payload["resolved"][1]["envelope_id"] == "srv_env_000000000123"
    assert response_payload["resolved"][1]["server_cursor"] == 123
    assert response_payload["server_cursor"] == 123
    assert "resolved_by_envelope_id" not in response_payload["resolved"][1]


def test_push_response_reports_per_envelope_outcomes():
    response = SyncPushResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": "env-1", "server_cursor": 1}],
            "rejected": [
                {
                    "client_envelope_id": "env-2",
                    "error_code": "unsupported_adapter_version",
                    "message": "adapter version is not supported",
                }
            ],
            "conflicts": [
                {
                    "conflict_id": "conflict-1",
                    "client_envelope_id": "env-3",
                    "object_id": "note-1",
                    "domain": "notes.note",
                    "server_cursor": 2,
                }
            ],
            "next_cursor": "2",
        }
    )

    assert response.accepted[0].server_cursor == 1
    assert response.rejected[0].error_code == "unsupported_adapter_version"
    assert response.conflicts[0].conflict_id == "conflict-1"


def test_push_request_allows_dataset_mismatch_for_per_envelope_outcomes():
    request = SyncPushRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "envelopes": [
                _m1_envelope_payload(
                    client_envelope_id="env-2",
                    dataset_id="dataset-2",
                )
            ],
        }
    )

    assert request.envelopes[0].dataset_id == "dataset-2"


def test_push_request_rejects_oversized_envelope_batches():
    envelopes = [
        _m1_envelope_payload(client_envelope_id=f"env-{index}")
        for index in range(SYNC_V2_MAX_PUSH_ENVELOPES + 1)
    ]

    with pytest.raises(ValidationError):
        SyncPushRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "device_id": "device-1",
                "envelopes": envelopes,
            }
        )


def test_push_request_requires_top_level_device_id():
    with pytest.raises(ValidationError):
        SyncPushRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "envelopes": [_m1_envelope_payload()],
            }
        )


def test_attachment_upload_request_response_models():
    request = SyncAttachmentUploadRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "domain": "attachment.ref",
            "object_id": "attachment-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 128,
            "payload_ciphertext": "encrypted-bytes",
            "payload_hash": "sha256:attachment",
            "metadata": {
                "parent_domain": "notes.note",
                "parent_object_id": "note-1",
                "availability": "metadata_only",
            },
        }
    )
    response = SyncAttachmentUploadResponse.model_validate(
        {
            "attachment_id": "attachment-1",
            "dataset_id": "dataset-1",
            "stored": True,
            "size_bytes": 128,
            "payload_hash": "sha256:attachment",
            "download_url": "/api/v1/sync/attachments/attachment-1",
        }
    )

    assert request.domain == "attachment.ref"
    assert request.object_id == "attachment-1"
    assert request.size_bytes == response.size_bytes
    assert response.stored is True


def test_m2_blob_protocol_models_validate_session_manifest_and_restore_completeness():
    payload_hash = "sha256:" + "a" * 64
    chunk_hash = "sha256:" + "b" * 64

    create_request = SyncBlobUploadCreateRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "device_id": "device-1",
            "domain": "attachment.ref",
            "object_id": "attachment-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 4096,
            "payload_hash": payload_hash,
            "chunk_size": 1024,
            "chunk_count": 4,
            "idempotency_key": "upload-once",
        }
    )
    session_response = SyncBlobUploadSessionResponse.model_validate(
        {
            "upload_id": "upload-1",
            "dataset_id": "dataset-1",
            "attachment_id": "attachment-1",
            "status": "uploading",
            "chunk_size": 1024,
            "chunk_count": 4,
            "uploaded_chunks": [0, 1],
            "missing_chunks": [2, 3],
            "quota": {"reserved_blob_bytes": 4096},
        }
    )
    chunk_response = SyncBlobChunkUploadResponse.model_validate(
        {
            "upload_id": "upload-1",
            "chunk_index": 2,
            "accepted": True,
            "size_bytes": 1024,
            "chunk_hash": chunk_hash,
            "missing_chunks": [3],
        }
    )
    complete_response = SyncBlobUploadCompleteResponse.model_validate(
        {
            "upload_id": "upload-1",
            "dataset_id": "dataset-1",
            "attachment_id": "attachment-1",
            "blob_id": "blob-1",
            "status": "available",
            "stored": True,
            "deduplicated": False,
            "size_bytes": 4096,
            "payload_hash": payload_hash,
        }
    )
    manifest_response = SyncBlobDownloadManifestResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "attachment_id": "attachment-1",
            "blob_id": "blob-1",
            "availability": "available",
            "content_type": "application/octet-stream",
            "size_bytes": 4096,
            "payload_hash": payload_hash,
            "chunks": [
                {
                    "chunk_index": 0,
                    "offset_bytes": 0,
                    "size_bytes": 1024,
                    "chunk_hash": chunk_hash,
                    "download_url": "/api/v1/sync/attachments/attachment-1/chunks/0",
                }
            ],
        }
    )
    completeness = SyncRestoreCompletenessResponse.model_validate(
        {
            "restore_status": "content_complete",
            "domain_details": [
                {
                    "domain": "attachment.ref",
                    "status": "content_complete",
                    "selected_count": 1,
                    "required_blob_count": 1,
                    "available_blob_count": 1,
                    "missing_blob_count": 0,
                    "verified_blob_count": 0,
                }
            ],
            "blob_details": [
                {
                    "attachment_id": "attachment-1",
                    "payload_hash": payload_hash,
                    "size_bytes": 4096,
                    "content_type": "application/octet-stream",
                    "parent_domain": "notes.note",
                    "parent_object_id": "note-1",
                    "server_availability": "available",
                    "required_for_restore": True,
                }
            ],
        }
    )

    assert create_request.encryption_policy == "server_trusted_v1"
    assert create_request.entity_id == "attachment-1"
    assert session_response.missing_chunks == [2, 3]
    assert chunk_response.chunk_index == 2
    assert complete_response.status == "available"
    assert manifest_response.chunks[0].chunk_hash == chunk_hash
    assert completeness.restore_status == "content_complete"


def test_m2_blob_upload_rejects_non_sha256_hashes():
    with pytest.raises(ValidationError):
        SyncBlobUploadCreateRequest.model_validate(
            {
                "dataset_id": "dataset-1",
                "domain": "attachment.ref",
                "object_id": "attachment-1",
                "attachment_id": "attachment-1",
                "content_type": "application/octet-stream",
                "size_bytes": 4096,
                "payload_hash": "md5:not-accepted",
                "chunk_size": 1024,
                "chunk_count": 4,
            }
        )


def test_capabilities_accept_m2_blob_transfer_and_quota_details():
    capabilities = SyncCapabilitiesResponse.model_validate(
        {
            "protocol_version": "sync-v2-m2",
            "min_supported_protocol_version": "sync-v2-m1",
            "supports_attachments": True,
            "blob_transfer": {
                "supported": True,
                "resumable_upload": True,
                "resumable_download": True,
                "chunk_checksums": True,
                "full_checksum": "sha256",
                "storage_backend": "local_fs",
            },
            "quota": {
                "max_blob_bytes": 104857600,
                "max_chunk_bytes": 4194304,
                "max_active_uploads": 8,
                "user_blob_quota_bytes": 10737418240,
            },
        }
    )

    assert capabilities.protocol_version == "sync-v2-m2"
    assert capabilities.blob_transfer["supported"] is True
    assert capabilities.blob_transfer["full_checksum"] == "sha256"
    assert capabilities.quota["max_chunk_bytes"] == 4194304
