from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Sync.v2.models import SyncEnvelope, SyncKeyRecord
from tldw_Server_API.app.core.Sync.v2.security import (
    PrivatePayloadValidationError,
    redact_envelope_for_log,
    redact_key_record_for_log,
    redact_private_mapping_for_log,
    validate_private_payload,
)


def _stored_envelope(**overrides) -> SyncEnvelope:
    payload = {
        "server_sequence": 1,
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "notes",
        "entity_id": "note-1",
        "operation": "upsert",
        "adapter_version": 1,
        "server_timestamp": "2026-05-10T00:00:00+00:00",
        "device_id": "device-1",
        "routing_metadata": {"entity_kind": "note"},
        "payload_ciphertext": "ciphertext:super-secret-private-note",
        "payload_clear": {"status": "active"},
        "payload_hash": "sha256:note-1",
        "payload_size_bytes": 512,
    }
    payload.update(overrides)
    return SyncEnvelope(**payload)


def _key_record(**overrides) -> SyncKeyRecord:
    payload = {
        "key_record_id": "key-1",
        "dataset_id": "dataset-1",
        "user_id": "user-1",
        "device_id": "device-1",
        "key_purpose": "dataset_recovery",
        "wrapped_key_blob": "wrapped:super-secret-key-material",
        "kdf_metadata": {"algorithm": "argon2id", "salt": "salt-secret"},
        "recovery_hint": "personal laptop",
        "rotation_of_key_record_id": None,
        "created_at": "2026-05-10T00:00:00+00:00",
    }
    payload.update(overrides)
    return SyncKeyRecord(**payload)


def test_private_payload_validation_rejects_plaintext_content_fields():
    with pytest.raises(PrivatePayloadValidationError):
        validate_private_payload(
            payload_ciphertext=None,
            payload_clear={"body": "known private note"},
        )


def test_private_payload_validation_allows_metadata_only_clear_fields():
    validate_private_payload(
        payload_ciphertext="ciphertext:opaque",
        payload_clear={
            "status": "active",
            "attachment_id": "attachment-1",
            "availability": "available",
            "size_bytes": 1024,
        },
    )


def test_envelope_redaction_never_leaks_ciphertext_or_private_clear_payload():
    envelope = _stored_envelope(
        payload_clear={"status": "active", "body": "known private note"}
    )

    redacted = redact_envelope_for_log(envelope)
    rendered = repr(redacted)

    assert redacted["payload_ciphertext"] == "<redacted>"
    assert redacted["payload_clear"] == "<redacted>"
    assert "ciphertext:super-secret-private-note" not in rendered
    assert "known private note" not in rendered
    assert redacted["client_envelope_id"] == "env-1"
    assert redacted["payload_hash"] == "sha256:note-1"


def test_key_record_redaction_never_leaks_wrapped_keys_or_kdf_secrets():
    record = _key_record()

    redacted = redact_key_record_for_log(record)
    rendered = repr(redacted)

    assert redacted["wrapped_key_blob"] == "<redacted>"
    assert redacted["kdf_metadata"] == "<redacted>"
    assert "wrapped:super-secret-key-material" not in rendered
    assert "salt-secret" not in rendered
    assert redacted["key_record_id"] == "key-1"
    assert redacted["recovery_hint"] == "personal laptop"


def test_private_mapping_redaction_is_recursive_and_preserves_safe_metadata():
    payload = {
        "dataset_id": "dataset-1",
        "payload_ciphertext": "ciphertext:secret",
        "wrapped_key_blob": "wrapped:secret",
        "payload_clear": {"body": "known private note"},
        "nested": {"ciphertext": "ciphertext:nested", "status": "active"},
    }

    redacted = redact_private_mapping_for_log(payload)
    rendered = repr(redacted)

    assert redacted["dataset_id"] == "dataset-1"
    assert redacted["payload_ciphertext"] == "<redacted>"
    assert redacted["wrapped_key_blob"] == "<redacted>"
    assert redacted["payload_clear"] == "<redacted>"
    assert redacted["nested"]["ciphertext"] == "<redacted>"
    assert redacted["nested"]["status"] == "active"
    assert "known private note" not in rendered
    assert "ciphertext:secret" not in rendered
    assert "wrapped:secret" not in rendered


def test_private_mapping_redacts_human_readable_private_fields_recursively():
    payload = {
        "dataset_id": "dataset-1",
        "status": "active",
        "title": "known private note title",
        "metadata": {
            "label": "known private label",
            "body": "known private body",
            "entity_kind": "note",
        },
        "items": [
            {
                "content": "known private item content",
                "stable_key": "note:1",
            }
        ],
        "batches": [
            [
                {
                    "content": "known nested private item content",
                    "stable_key": "note:2",
                }
            ]
        ],
    }

    redacted = redact_private_mapping_for_log(payload)
    rendered = repr(redacted)

    assert redacted["dataset_id"] == "dataset-1"
    assert redacted["status"] == "active"
    assert redacted["metadata"]["entity_kind"] == "note"
    assert redacted["items"][0]["stable_key"] == "note:1"
    assert redacted["title"] == "<redacted>"
    assert redacted["metadata"]["label"] == "<redacted>"
    assert redacted["metadata"]["body"] == "<redacted>"
    assert redacted["items"][0]["content"] == "<redacted>"
    assert redacted["batches"][0][0]["stable_key"] == "note:2"
    assert redacted["batches"][0][0]["content"] == "<redacted>"
    assert "known private" not in rendered
    assert "known nested private" not in rendered
