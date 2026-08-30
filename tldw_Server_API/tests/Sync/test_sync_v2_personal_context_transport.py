from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace
from pathlib import Path

import pytest
from tldw_profile_core.canonical import canonical_json_bytes

from tldw_Server_API.app.core.DB_Management.Sync_DB import SyncDatabase
from tldw_Server_API.app.core.Sync.v2.adapters import SyncAdapterRegistry
from tldw_Server_API.app.core.Sync.v2.domain_adapters.personal_context import (
    PersonalContextDomainAdapter,
)
from tldw_Server_API.app.core.Sync.v2.materializers.personal_context import (
    PersonalContextMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    PERSONAL_CONTEXT_SYNC_DOMAINS,
    SyncDatasetCreate,
    SyncEnvelopeCreate,
)
from tldw_Server_API.app.core.Sync.v2.security import (
    server_trusted_encryption_status_from_config,
)
from tldw_Server_API.app.core.Sync.v2.service import SyncV2Service, SyncV2Settings
from tldw_Server_API.app.core.Sync.v2.store import SyncV2Store
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    preference_record,
)

pytestmark = pytest.mark.unit

DOMAIN = "personal_context.record"
DATASET_ID = "dataset-a"
PROFILE_ID = "profile-a"
INTEGRITY_KEY = b"i" * 32
ENCRYPTION_KEY = b"e" * 32
INTEGRITY_KEY_ID = "personal-context-integrity-v1"
CANARY = "SERVER-SYNC-PLAINTEXT-CANARY-21e8"


class _RecordingService:
    def __init__(self) -> None:
        self.values = []

    def apply_sync_object(self, **values):
        self.values.append(values["value"])
        return values["value"]


def _tag(payload: dict[str, object]) -> str:
    digest = hmac.new(INTEGRITY_KEY, canonical_json_bytes(payload), hashlib.sha256)
    return f"hmac-sha256-v1:{digest.hexdigest()}"


def _envelope(
    payload: dict[str, object],
    *,
    client_envelope_id: str,
    base_server_cursor: int | None = None,
    base_object_revision: int | None = None,
    base_object_hash: str | None = None,
) -> SyncEnvelopeCreate:
    return SyncEnvelopeCreate(
        dataset_id=DATASET_ID,
        client_envelope_id=client_envelope_id,
        device_id="device-a",
        domain=DOMAIN,
        operation="upsert",
        object_id=str(payload["record_id"]),
        parent_id=str(payload["scope_id"]),
        adapter_version=1,
        schema_version=1,
        payload=payload,
        payload_hash=_tag(payload),
        payload_size_bytes=len(canonical_json_bytes(payload)),
        base_server_cursor=base_server_cursor,
        base_object_revision=base_object_revision,
        base_object_hash=base_object_hash,
        base_version=payload.get("parent_version_id"),
        entity_version=str(payload["version_id"]),
        encryption_metadata={"policy": "server_trusted_v1"},
        routing_metadata={
            "integrity_key_id": INTEGRITY_KEY_ID,
            "profile_id": PROFILE_ID,
            "purge_generation": None,
        },
    )


def _service(tmp_path: Path) -> tuple[SyncV2Service, _RecordingService, Path]:
    sqlite_path = tmp_path / "sync.db"
    store = SyncV2Store(SyncDatabase(sqlite_path=sqlite_path))
    adapters = SyncAdapterRegistry(
        [
            PersonalContextDomainAdapter(
                domain=domain,
                integrity_key_resolver=lambda _dataset, _key_id: INTEGRITY_KEY,
                encryption_key_resolver=lambda _dataset: (ENCRYPTION_KEY, 1),
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        ]
    )
    target = _RecordingService()
    service = SyncV2Service(
        store=store,
        adapters=adapters,
        materializers={
            domain: PersonalContextMaterializer(
                domain=domain,
                service_resolver=lambda _user_id: target,
            )
            for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
        },
        settings=SyncV2Settings(
            server_trusted_encryption=server_trusted_encryption_status_from_config(
                mode="managed_storage",
                server_trusted_enabled=True,
                auth_mode="multi_user",
            )
        ),
    )
    for device_id in ("device-a", "device-b"):
        service.register_device(
            user_id="user-a",
            display_name=device_id,
            client_type="chatbook",
            device_id=device_id,
            capabilities={
                "supported_adapter_versions": {
                    domain: [1] for domain in PERSONAL_CONTEXT_SYNC_DOMAINS
                }
            },
        )
    store.enroll_dataset(
        SyncDatasetCreate(
            dataset_id=DATASET_ID,
            owner_user_id="user-a",
            encryption_policy="server_trusted_v1",
            domains=list(PERSONAL_CONTEXT_SYNC_DOMAINS),
            metadata={
                "personal_context": {
                    "profile_id": PROFILE_ID,
                    "integrity_key_id": INTEGRITY_KEY_ID,
                    "purge_generation": 0,
                    "link_state": "complete",
                }
            },
        )
    )
    return service, target, sqlite_path


def test_two_version_push_satisfies_sync_cas_and_encrypts_transport_history(
    tmp_path: Path,
) -> None:
    service, target, sqlite_path = _service(tmp_path)
    first_payload = preference_record(value=CANARY).model_dump(mode="json")
    first = _envelope(first_payload, client_envelope_id="device-a:record:1")

    first_result = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[first],
    )

    assert first_result.rejected == []
    assert first_result.conflicts == []
    assert len(first_result.accepted) == 1
    first_cursor = first_result.accepted[0].server_sequence
    replayed = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[first],
    )
    assert replayed.rejected == []
    assert replayed.conflicts == []
    assert replayed.accepted[0].server_sequence == first_cursor
    assert len(target.values) == 1
    second_payload = preference_record(
        version_id="record-v2",
        parent_version_id=str(first_payload["version_id"]),
        value="structured",
    ).model_dump(mode="json")
    second = _envelope(
        second_payload,
        client_envelope_id="device-a:record:2",
        base_server_cursor=first_cursor,
        base_object_revision=first_result.accepted[0].object_revision,
        base_object_hash=first.payload_hash,
    )

    second_result = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[second],
    )

    assert second_result.rejected == []
    assert second_result.conflicts == []
    assert len(second_result.accepted) == 1
    assert [value.version_id for value in target.values] == [
        first_payload["version_id"],
        "record-v2",
    ]
    pulled = service.pull(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-b",
        domains=[DOMAIN],
    )
    assert [item.payload["version_id"] for item in pulled.envelopes] == [
        first_payload["version_id"],
        "record-v2",
    ]
    durable = b"".join(
        path.read_bytes()
        for path in tmp_path.iterdir()
        if path.name.startswith(sqlite_path.name)
    )
    assert CANARY.encode() not in durable


def test_conflict_fails_closed_when_profile_storage_key_is_unavailable(
    tmp_path: Path,
) -> None:
    service, _target, _sqlite_path = _service(tmp_path)
    for domain in PERSONAL_CONTEXT_SYNC_DOMAINS:
        service.adapters.get(domain).encryption_key_resolver = None
    payload = preference_record().model_dump(mode="json")
    envelope = _envelope(payload, client_envelope_id="device-a:record:conflict")
    envelope = replace(envelope, object_id="wrong-record-id")

    result = service.push(
        user_id="user-a",
        dataset_id=DATASET_ID,
        device_id="device-a",
        envelopes=[envelope],
    )

    assert result.conflicts == []
    assert result.accepted == []
    assert len(result.rejected) == 1
    assert result.rejected[0].error_code == "personal_context_storage_unavailable"
