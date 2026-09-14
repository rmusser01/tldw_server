from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Personalization.personal_context_publication import (
    CanonicalApplyReceipt,
)
from tldw_Server_API.app.core.Personalization.personal_context_service import (
    ProfileConflictError,
)
from tldw_Server_API.app.core.Sync.v2.materializers.personal_context import (
    PersonalContextMaterializer,
)
from tldw_Server_API.app.core.Sync.v2.models import (
    SyncDataset,
    SyncEnvelope,
    SyncObjectState,
)
from tldw_Server_API.tests.Personalization.personal_context_test_support import (
    preference_record,
)

pytestmark = pytest.mark.unit


@dataclass
class _RecordingService:
    calls: list[dict[str, object]] = field(default_factory=list)
    ingress_calls: list[dict[str, object]] = field(default_factory=list)
    conflict: bool = False
    invalid_ingress_receipt: bool = False

    def apply_sync_object(self, **values: object) -> object:
        if self.conflict:
            raise ProfileConflictError("changed")
        self.calls.append(values)
        return values["value"]

    def apply_sync_ingress(self, **values: object) -> object:
        self.ingress_calls.append(values)
        if self.invalid_ingress_receipt:
            return SimpleNamespace(receipt_id="not-a-canonical-receipt")
        identity = values["identity"]
        value = values["value"]
        return CanonicalApplyReceipt(
            resulting_object_id=value.record_id,
            resulting_version_id=value.version_id,
            manifest_revision=1,
            manifest_version_id="manifest-v1",
            purge_generation=identity.purge_generation,
            publication_batch_id="batch-1",
            profile_publication_sequence=1,
            receipt_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    "tldw:personal-context:ingress:"
                    f"{identity.dataset_id}:{identity.device_id}:{identity.client_envelope_id}",
                )
            ),
            dataset_id=identity.dataset_id,
            device_id=identity.device_id,
            client_envelope_id=identity.client_envelope_id,
            canonical_payload_digest=identity.canonical_payload_digest,
            wire_entity_version=identity.wire_entity_version,
        )


class _Store:
    def __init__(self, dataset: SyncDataset | None) -> None:
        self.dataset = dataset
        self.statuses: list[tuple[int, str, str | None]] = []
        self.object_states: list[SyncObjectState] = []

    def get_dataset(self, _dataset_id: str) -> SyncDataset | None:
        return self.dataset

    def get_object_state(
        self,
        _dataset_id: str,
        _domain: str,
        _object_id: str,
    ) -> SyncObjectState | None:
        return self.object_states[-1] if self.object_states else None

    def upsert_object_state(self, state: SyncObjectState) -> None:
        self.object_states.append(state)

    def mark_envelope_apply_status(
        self,
        cursor: int,
        *,
        apply_status: str,
        apply_error_code: str | None = None,
        apply_error_message: str | None = None,
    ) -> None:
        del apply_error_message
        self.statuses.append((cursor, apply_status, apply_error_code))

    def mark_personal_context_ingress_applied(
        self,
        *,
        server_cursor: int,
        receipt: CanonicalApplyReceipt,
    ) -> None:
        assert receipt.client_envelope_id == "device-a:record-a:1"
        assert receipt.receipt_id == str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                "tldw:personal-context:ingress:dataset-a:device-a:device-a:record-a:1",
            )
        )
        self.statuses.append((server_cursor, "applied", None))


def _dataset() -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-a",
        owner_user_id="user-a",
        scope_type="personal",
        encryption_policy="server_trusted_v1",
        domains=["personal_context.record"],
        workspace_id=None,
        metadata={"personal_context": {"profile_id": "profile-a", "purge_generation": 0}},
        created_at="2026-08-30T12:00:00Z",
        updated_at="2026-08-30T12:00:00Z",
    )


def _envelope() -> SyncEnvelope:
    record = preference_record()
    return SyncEnvelope(
        dataset_id="dataset-a",
        client_envelope_id="device-a:record-a:1",
        server_cursor=1,
        envelope_id="envelope-1",
        device_id="device-a",
        domain="personal_context.record",
        operation="upsert",
        object_id=record.record_id,
        parent_id=record.scope_id,
        adapter_version=1,
        schema_version=1,
        payload=record.model_dump(mode="json"),
        payload_hash="hmac-sha256-v1:" + "a" * 64,
        object_revision=1,
        entity_version=record.version_id,
        encryption_metadata={"policy": "server_trusted_v1"},
    )


def test_materializer_applies_through_authenticated_personal_context_service() -> None:
    service = _RecordingService()
    store = _Store(_dataset())
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda user_id: service if user_id == "user-a" else None,
    )

    result = materializer.apply(_envelope(), store=store)

    assert result.status == "applied"
    assert service.calls[0]["actor_type"] == "sync"
    assert service.calls[0]["actor_id"] == "device-a"
    assert service.calls[0]["domain"] == "personal_context.record"
    assert service.calls[0]["value"] == preference_record()
    assert store.statuses == [(1, "applied", None)]


def test_materializer_maps_service_cas_failure_to_content_free_conflict() -> None:
    service = _RecordingService(conflict=True)
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda _user_id: service,
    )

    result = materializer.apply(_envelope(), store=_Store(_dataset()))

    assert result.status == "conflict"
    assert result.conflict_type == "personal_context_base_conflict"
    assert "concise" not in (result.message or "")


def test_materializer_applies_client_ingress_through_a_canonical_receipt() -> None:
    service = _RecordingService()
    store = _Store(_dataset())
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda _user_id: service,
    )
    envelope = replace(
        _envelope(),
        routing_metadata={"personal_context_authority": {"role": "client_ingress"}},
    )

    result = materializer.apply(envelope, store=store)

    assert result.status == "applied"
    assert service.calls == []
    assert service.ingress_calls[0]["identity"].client_envelope_id == "device-a:record-a:1"
    assert store.statuses == [(1, "applied", None)]


def test_materializer_derives_omitted_update_revision_from_immutable_lineage() -> None:
    service = _RecordingService()
    store = _Store(_dataset())
    envelope = replace(
        _envelope(),
        server_cursor=9,
        object_revision=None,
        base_server_cursor=8,
        base_object_revision=4,
        base_object_hash="hmac-sha256-v1:" + "b" * 64,
        base_version="record-v4",
        routing_metadata={"personal_context_authority": {"role": "client_ingress"}},
    )
    store.object_states.append(
        SyncObjectState(
            dataset_id="dataset-a",
            domain="personal_context.record",
            object_id=envelope.object_id,
            object_revision=4,
            object_hash="hmac-sha256-v1:" + "b" * 64,
            latest_server_cursor=8,
        )
    )
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda _user_id: service,
    )

    result = materializer.apply(envelope, store=store)

    assert result.status == "applied"
    assert store.object_states[-1].object_revision == 5


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("dataset_id", "other-dataset"),
        ("domain", "personal_context.scope"),
        ("object_id", "other-record"),
        ("latest_server_cursor", 7),
        ("object_revision", 3),
        ("object_hash", "hmac-sha256-v1:" + "0" * 64),
        ("deleted", True),
    ],
)
def test_materializer_rejects_omitted_revision_predecessor_mismatch_before_receipt(
    field: str,
    changed: object,
) -> None:
    service = _RecordingService()
    envelope = replace(
        _envelope(),
        server_cursor=9,
        object_revision=None,
        base_server_cursor=8,
        base_object_revision=4,
        base_object_hash="hmac-sha256-v1:" + "b" * 64,
        base_version="record-v4",
        routing_metadata={"personal_context_authority": {"role": "client_ingress"}},
    )
    state = SyncObjectState(
        dataset_id="dataset-a",
        domain="personal_context.record",
        object_id=envelope.object_id,
        object_revision=4,
        object_hash="hmac-sha256-v1:" + "b" * 64,
        latest_server_cursor=8,
    )
    store = _Store(_dataset())
    store.object_states.append(replace(state, **{field: changed}))
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda _user_id: service,
    )

    result = materializer.apply(envelope, store=store)

    assert result.status == "conflict"
    assert service.ingress_calls == []
    assert store.statuses == [(9, "conflict", "personal_context_base_conflict")]


def test_materializer_rejects_partial_omitted_revision_lineage_before_receipt() -> None:
    service = _RecordingService()
    store = _Store(_dataset())
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda _user_id: service,
    )
    envelope = replace(
        _envelope(),
        object_revision=None,
        base_server_cursor=8,
        routing_metadata={"personal_context_authority": {"role": "client_ingress"}},
    )

    result = materializer.apply(envelope, store=store)

    assert result.status == "failed"
    assert service.ingress_calls == []
    assert store.statuses == [(1, "failed", "personal_context_payload_invalid")]


def test_materializer_fails_before_service_resolution_without_authorized_dataset() -> None:
    resolver_calls: list[str] = []
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda user_id: resolver_calls.append(user_id),
    )

    result = materializer.apply(_envelope(), store=_Store(None))

    assert result.status == "failed"
    assert result.error_code == "personal_context_authorization_unavailable"
    assert resolver_calls == []


def test_materializer_rejects_fake_ingress_receipts_before_sync_terminalization() -> None:
    service = _RecordingService(invalid_ingress_receipt=True)
    store = _Store(_dataset())
    materializer = PersonalContextMaterializer(
        domain="personal_context.record",
        service_resolver=lambda _user_id: service,
    )
    envelope = replace(
        _envelope(),
        routing_metadata={"personal_context_authority": {"role": "client_ingress"}},
    )

    result = materializer.apply(envelope, store=store)

    assert result.status == "failed"
    assert result.error_code == "personal_context_payload_invalid"
    assert store.statuses == [(1, "failed", "personal_context_payload_invalid")]
