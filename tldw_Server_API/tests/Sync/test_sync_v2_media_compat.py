from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.endpoints import sync as sync_endpoint
from tldw_Server_API.app.api.v1.schemas.sync_server_models import SyncLogEntry
from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.domain_adapters.media import (
    MediaCompatibilityAdapter,
    legacy_media_sync_log_to_envelope,
)
from tldw_Server_API.app.core.Sync.v2.adapters import (
    AdapterAccepted,
    AdapterRejected,
    SyncAdapterRegistry,
)
from tldw_Server_API.app.core.Sync.v2.errors import SyncStoreError
from tldw_Server_API.app.core.Sync.v2.models import SyncDataset, SyncEnvelopeCreate
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store


def _dataset(*, encryption_policy: str = "client_private_v1") -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-1",
        owner_user_id="user-1",
        scope_type="personal",
        encryption_policy=encryption_policy,
        domains=["media"],
        workspace_id=None,
        metadata={},
        created_at="2026-05-10T00:00:00+00:00",
        updated_at="2026-05-10T00:00:00+00:00",
    )


def _envelope(**overrides) -> SyncEnvelopeCreate:
    payload = {
        "dataset_id": "dataset-1",
        "client_envelope_id": "env-1",
        "domain": "media",
        "entity_id": "media-1",
        "operation": "upsert",
        "adapter_version": 1,
        "device_id": "device-1",
        "stable_key": "media:media-1",
        "client_timestamp": "2026-05-10T00:00:00+00:00",
        "entity_version": 1,
        "routing_metadata": {
            "legacy_entity": "Media",
            "legacy_operation": "create",
        },
        "payload_ciphertext": "ciphertext:opaque",
        "payload_clear": {"entity_kind": "Media"},
        "payload_hash": "sha256:legacy-payload",
        "payload_size_bytes": 128,
    }
    payload.update(overrides)
    return SyncEnvelopeCreate(**payload)


def _ready_sync_settings() -> SyncV2Settings:
    return SyncV2Settings(
        server_trusted_encryption=server_trusted_encryption_status_from_config(
            mode="encrypted_volume",
            server_trusted_enabled=True,
            auth_mode="multi_user",
        )
    )


@pytest.mark.parametrize(
    ("legacy_entity", "legacy_operation", "v2_operation"),
    [
        ("Media", "create", "upsert"),
        ("Media", "update", "upsert"),
        ("Media", "delete", "delete"),
        ("Keywords", "create", "upsert"),
        ("Keywords", "update", "upsert"),
        ("Keywords", "delete", "delete"),
        ("MediaKeywords", "link", "link"),
        ("MediaKeywords", "unlink", "unlink"),
    ],
)
def test_media_adapter_accepts_legacy_media_semantics(
    legacy_entity: str,
    legacy_operation: str,
    v2_operation: str,
):
    adapter = MediaCompatibilityAdapter()
    routing_metadata = {
        "legacy_entity": legacy_entity,
        "legacy_operation": legacy_operation,
    }
    if legacy_entity == "MediaKeywords":
        routing_metadata.update({"media_uuid": "media-1", "keyword_uuid": "keyword-1"})

    outcome = adapter.evaluate_envelope(
        _envelope(
            client_envelope_id=f"{legacy_entity}-{legacy_operation}",
            entity_id="media-1" if legacy_entity != "MediaKeywords" else "media-1:keyword-1",
            operation=v2_operation,
            routing_metadata=routing_metadata,
            payload_clear={"entity_kind": legacy_entity},
        ),
        dataset=_dataset(),
    )

    assert outcome == AdapterAccepted(client_envelope_id=f"{legacy_entity}-{legacy_operation}")


@pytest.mark.parametrize(
    ("envelope", "error_code"),
    [
        (
            _envelope(routing_metadata={"legacy_entity": "Transcripts", "legacy_operation": "create"}),
            "unsupported_legacy_media_entity",
        ),
        (
            _envelope(
                operation="link",
                routing_metadata={"legacy_entity": "Media", "legacy_operation": "link"},
            ),
            "invalid_legacy_media_operation",
        ),
        (
            _envelope(
                operation="upsert",
                routing_metadata={"legacy_entity": "MediaKeywords", "legacy_operation": "create"},
            ),
            "invalid_legacy_media_operation",
        ),
        (
            _envelope(
                operation="link",
                routing_metadata={"legacy_entity": "MediaKeywords", "legacy_operation": "link"},
                payload_clear={"entity_kind": "MediaKeywords"},
            ),
            "missing_media_keyword_link_metadata",
        ),
    ],
)
def test_media_adapter_rejects_invalid_legacy_semantics(
    envelope: SyncEnvelopeCreate,
    error_code: str,
):
    outcome = MediaCompatibilityAdapter().evaluate_envelope(envelope, dataset=_dataset())

    assert isinstance(outcome, AdapterRejected)
    assert outcome.error_code == error_code


def test_default_sync_v2_registry_excludes_legacy_media_adapter():
    registry = sync_endpoint._default_sync_v2_registry()

    with pytest.raises(KeyError):
        registry.get("media")


def test_media_dataset_enrollment_is_rejected_by_m1_service(tmp_path: Path):
    registry = SyncAdapterRegistry([MediaCompatibilityAdapter()])
    service = SyncV2Service(
        store=SyncV2Store(SyncDatabase(sqlite_path=tmp_path / "sync_v2_media.db")),
        adapters=registry,
        settings=_ready_sync_settings(),
    )
    service.register_device(
        user_id="user-1",
        display_name="Laptop",
        client_type="chatbook",
        device_id="device-1",
    )

    with pytest.raises(SyncStoreError, match="unsupported domains: media"):
        service.enroll_dataset(
            user_id="user-1",
            dataset_id="dataset-1",
            domains=["media"],
        )


def test_legacy_media_sync_log_translates_without_private_plaintext_leakage():
    canonical_payload = json.dumps(
        {
            "content": "Private content",
            "deleted": False,
            "title": "Private title",
            "uuid": "media-1",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    entry = {
        "change_id": 42,
        "entity": "Media",
        "entity_uuid": "media-1",
        "operation": "update",
        "timestamp": "2026-05-10T00:00:00Z",
        "client_id": "client-1",
        "version": 7,
        "payload": '{"uuid":"media-1","title":"Private title","content":"Private content","deleted":false}',
    }

    envelope = legacy_media_sync_log_to_envelope(
        entry,
        dataset_id="dataset-1",
        payload_ciphertext="ciphertext:legacy-media",
    )

    assert envelope.domain == "media"
    assert envelope.operation == "upsert"
    assert envelope.entity_id == "media-1"
    assert envelope.stable_key == "media:media-1"
    assert envelope.client_envelope_id == "legacy-media:42"
    assert envelope.routing_metadata == {
        "legacy_entity": "Media",
        "legacy_operation": "update",
    }
    assert envelope.payload_ciphertext == "ciphertext:legacy-media"
    assert envelope.payload_clear == {"entity_kind": "Media", "deleted": False}
    assert envelope.payload_hash == (
        f"sha256:{hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()}"
    )
    assert envelope.payload_size_bytes == len(canonical_payload.encode("utf-8"))
    assert "Private title" not in str(envelope.payload_clear)
    assert "Private content" not in str(envelope.payload_clear)


def test_legacy_media_sync_log_accepts_decoded_dict_payload_from_media_db():
    decoded_payload = {
        "uuid": "media-1",
        "content": "Private content",
        "deleted": False,
        "title": "Private title",
    }
    canonical_payload = json.dumps(decoded_payload, sort_keys=True, separators=(",", ":"))
    entry = {
        "change_id": 44,
        "entity": "Media",
        "entity_uuid": "media-1",
        "operation": "update",
        "timestamp": "2026-05-10T00:00:00Z",
        "client_id": "client-1",
        "version": 8,
        "payload": decoded_payload,
    }

    envelope = legacy_media_sync_log_to_envelope(
        entry,
        dataset_id="dataset-1",
        payload_ciphertext="ciphertext:decoded-media",
    )

    assert envelope.payload_clear == {"entity_kind": "Media", "deleted": False}
    assert envelope.payload_hash == (
        f"sha256:{hashlib.sha256(canonical_payload.encode('utf-8')).hexdigest()}"
    )
    assert envelope.payload_size_bytes == len(canonical_payload.encode("utf-8"))
    assert "Private title" not in str(envelope.payload_clear)
    assert "Private content" not in str(envelope.payload_clear)


@pytest.mark.parametrize("payload", [["not", "an", "object"], '["not","an","object"]', 42])
def test_legacy_media_sync_log_rejects_non_object_payloads(payload):
    entry = {
        "change_id": 45,
        "entity": "Media",
        "entity_uuid": "media-1",
        "operation": "update",
        "timestamp": "2026-05-10T00:00:00Z",
        "client_id": "client-1",
        "version": 8,
        "payload": payload,
    }

    with pytest.raises(ValueError, match="payload must be a JSON object"):
        legacy_media_sync_log_to_envelope(
            entry,
            dataset_id="dataset-1",
            payload_ciphertext="ciphertext:bad-payload",
        )


def test_legacy_media_keyword_link_translates_to_pair_stable_key():
    entry = SyncLogEntry(
        change_id=43,
        entity="MediaKeywords",
        entity_uuid="media-1:keyword-1",
        operation="link",
        timestamp="2026-05-10T00:00:00Z",
        client_id="client-1",
        version=1,
        payload='{"media_uuid":"media-1","keyword_uuid":"keyword-1"}',
    )

    envelope = legacy_media_sync_log_to_envelope(
        entry,
        dataset_id="dataset-1",
        payload_ciphertext="ciphertext:legacy-link",
    )

    assert envelope.operation == "link"
    assert envelope.entity_id == "media-1:keyword-1"
    assert envelope.stable_key == "media_keywords:media-1:keyword-1"
    assert envelope.routing_metadata == {
        "legacy_entity": "MediaKeywords",
        "legacy_operation": "link",
        "media_uuid": "media-1",
        "keyword_uuid": "keyword-1",
    }
    assert envelope.payload_clear == {"entity_kind": "MediaKeywords", "link_type": "media_keyword"}
