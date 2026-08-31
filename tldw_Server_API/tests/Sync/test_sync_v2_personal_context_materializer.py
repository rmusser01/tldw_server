from __future__ import annotations

from dataclasses import dataclass, field

import pytest

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
    conflict: bool = False

    def apply_sync_object(self, **values: object) -> object:
        if self.conflict:
            raise ProfileConflictError("changed")
        self.calls.append(values)
        return values["value"]


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


def _dataset() -> SyncDataset:
    return SyncDataset(
        dataset_id="dataset-a",
        owner_user_id="user-a",
        scope_type="personal",
        encryption_policy="server_trusted_v1",
        domains=["personal_context.record"],
        workspace_id=None,
        metadata={"personal_context": {"profile_id": "profile-a"}},
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
