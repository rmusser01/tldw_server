import pytest
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.schemas.sync_v2_models import (
    SyncAttachmentUploadRequest,
    SyncAttachmentUploadResponse,
    SyncPushResponse,
    SyncV2Envelope,
)


def _private_envelope_payload(**overrides):
    payload = {
        "client_envelope_id": "env-1",
        "dataset_id": "dataset-1",
        "domain": "notes",
        "entity_id": "note-1",
        "operation": "upsert",
        "adapter_version": 1,
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": "ciphertext:opaque",
        "payload_clear": {"status": "active", "tag_ids": ["tag-1"]},
        "payload_hash": "sha256:test",
    }
    payload.update(overrides)
    return payload


def test_sync_envelope_rejects_plaintext_private_payload():
    payload = _private_envelope_payload(
        payload_ciphertext=None,
        payload_clear={"body": "known plaintext"},
    )

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(payload)


def test_sync_envelope_allows_private_routing_metadata_in_clear_payload():
    envelope = SyncV2Envelope.model_validate(
        _private_envelope_payload(
            payload_clear={
                "status": "archived",
                "routing": {"entity_kind": "note"},
                "tag_ids": ["tag-1"],
            }
        )
    )

    assert envelope.payload_clear["status"] == "archived"
    assert envelope.payload_clear["routing"]["entity_kind"] == "note"


def test_sync_envelope_requires_adapter_version():
    payload = _private_envelope_payload()
    payload.pop("adapter_version")

    with pytest.raises(ValidationError):
        SyncV2Envelope.model_validate(payload)


def test_push_response_reports_per_envelope_outcomes():
    response = SyncPushResponse.model_validate(
        {
            "dataset_id": "dataset-1",
            "accepted": [{"client_envelope_id": "env-1", "server_sequence": 1}],
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
                    "entity_id": "note-1",
                    "domain": "notes",
                    "server_sequence": 2,
                }
            ],
            "next_cursor": "2",
        }
    )

    assert response.accepted[0].server_sequence == 1
    assert response.rejected[0].error_code == "unsupported_adapter_version"
    assert response.conflicts[0].conflict_id == "conflict-1"


def test_attachment_upload_request_response_models():
    request = SyncAttachmentUploadRequest.model_validate(
        {
            "dataset_id": "dataset-1",
            "domain": "source_cache",
            "entity_id": "source-1",
            "attachment_id": "attachment-1",
            "content_type": "application/octet-stream",
            "size_bytes": 128,
            "payload_ciphertext": "encrypted-bytes",
            "payload_hash": "sha256:attachment",
            "metadata": {"kind": "transcript_chunk"},
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

    assert request.domain == "source_cache"
    assert request.size_bytes == response.size_bytes
    assert response.stored is True
