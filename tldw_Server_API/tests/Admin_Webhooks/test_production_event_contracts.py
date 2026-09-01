from __future__ import annotations

import base64
import json
from dataclasses import replace
from datetime import datetime, timezone
from inspect import signature
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.catalog import EVENT_API_VERSION
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
    WebhookRouteSelection,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    EVENT_BODY_MAX_BYTES,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    EventSourceKind,
    WebhookError,
    WebhookErrorCode,
    WebhookEvent,
)
from tldw_Server_API.app.core.Admin_Webhooks.events import (
    canonical_event_body,
    prepare_event_insert,
    snapshot_json_object,
    verify_event_replay,
)
from tldw_Server_API.app.core.Admin_Webhooks.producer import (
    AdminWebhookEventProducer,
    build_incident_created_data,
    build_incident_notify_data,
    build_incident_resolved_data,
    build_incident_updated_data,
    build_user_created_data,
    build_user_deleted_data,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    EventCaptureResult,
    StoredWebhookEvent,
)

CREATED_AT = datetime(2026, 8, 31, 12, 0, tzinfo=timezone.utc)
UPDATED_AT = datetime(2026, 8, 31, 12, 5, tzinfo=timezone.utc)
RESOLVED_AT = datetime(2026, 8, 31, 12, 10, tzinfo=timezone.utc)
PROFILE_VERSION = datetime(2026, 8, 31, 12, 5, tzinfo=timezone.utc)
EVENT_ID = "5f16ee87-4e72-49f7-b027-41f15f26a90f"
KEY_ID = "key-2026-08"


def _ring() -> WebhookKeyRing:
    return WebhookKeyRing(
        {KEY_ID: base64.b64encode(b"k" * 32).decode("ascii")},
        primary_id=KEY_ID,
    )


def _settings(mode: AdminWebhookMode = AdminWebhookMode.ON) -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=mode,
        route_selection=WebhookRouteSelection.CANONICAL,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
    )


def _migration_state(**changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "phase": "complete",
        "completed_at": CREATED_AT,
        "rotation_phase": None,
        "active_primary_key_id": KEY_ID,
    }
    values.update(changes)
    return SimpleNamespace(**values)


class _ProducerRepositoryProbe:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state
        self.reads = 0

    async def get_migration_state(self) -> SimpleNamespace:
        self.reads += 1
        return self.state


class _CaptureUnitProbe:
    def __init__(self, state: SimpleNamespace) -> None:
        self.state = state
        self.lock_calls = 0
        self.capture_calls = 0
        self.event = None

    async def lock_migration_state(self) -> SimpleNamespace:
        self.lock_calls += 1
        return self.state

    async def capture_event_and_expand(
        self,
        event,
        delivery_id_factory,
        expires_at: datetime,
    ) -> EventCaptureResult:
        self.capture_calls += 1
        self.event = event
        assert expires_at.timestamp() - event.created_at.timestamp() == 72 * 60 * 60
        delivery_id_factory()
        stored = StoredWebhookEvent(
            event=WebhookEvent(
                id=event.id,
                event_type=event.event_type,
                api_version=event.api_version,
                source_kind=event.source_kind,
                created_at=event.created_at,
            ),
            aggregate_type=event.aggregate_type,
            aggregate_id=event.aggregate_id,
            aggregate_version=event.aggregate_version,
            source_command_id=event.source_command_id,
            source_component=event.source_component,
            source_request_id=event.source_request_id,
            body=event.body,
            body_size_bytes=event.body_size_bytes,
        )
        return EventCaptureResult(event=stored, deliveries=(), inserted=True)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("builder", "kwargs", "expected"),
    (
        (
            build_user_created_data,
            {
                "user_id": 7,
                "resource_version": PROFILE_VERSION,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
            },
            {
                "user_id": 7,
                "status": "active",
                "resource_version": "2026-08-31T12:05:00Z",
                "created_at": "2026-08-31T12:00:00Z",
                "updated_at": "2026-08-31T12:05:00Z",
            },
        ),
        (
            build_user_deleted_data,
            {
                "user_id": 7,
                "resource_version": PROFILE_VERSION,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
            },
            {
                "user_id": 7,
                "status": "inactive",
                "resource_version": "2026-08-31T12:05:00Z",
                "created_at": "2026-08-31T12:00:00Z",
                "updated_at": "2026-08-31T12:05:00Z",
            },
        ),
        (
            build_incident_created_data,
            {
                "incident_id": "inc_123",
                "state": "open",
                "severity": "high",
                "resource_version": 1,
                "created_at": CREATED_AT,
                "updated_at": CREATED_AT,
                "resolved_at": None,
            },
            {
                "incident_id": "inc_123",
                "state": "open",
                "severity": "high",
                "resource_version": 1,
                "created_at": "2026-08-31T12:00:00Z",
                "updated_at": "2026-08-31T12:00:00Z",
                "resolved_at": None,
            },
        ),
        (
            build_incident_updated_data,
            {
                "incident_id": "inc_123",
                "state": "investigating",
                "severity": "critical",
                "resource_version": 2,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
                "resolved_at": None,
            },
            {
                "incident_id": "inc_123",
                "state": "investigating",
                "severity": "critical",
                "resource_version": 2,
                "created_at": "2026-08-31T12:00:00Z",
                "updated_at": "2026-08-31T12:05:00Z",
                "resolved_at": None,
            },
        ),
        (
            build_incident_resolved_data,
            {
                "incident_id": "inc_123",
                "state": "resolved",
                "severity": "critical",
                "resource_version": 3,
                "created_at": CREATED_AT,
                "updated_at": RESOLVED_AT,
                "resolved_at": RESOLVED_AT,
            },
            {
                "incident_id": "inc_123",
                "state": "resolved",
                "severity": "critical",
                "resource_version": 3,
                "created_at": "2026-08-31T12:00:00Z",
                "updated_at": "2026-08-31T12:10:00Z",
                "resolved_at": "2026-08-31T12:10:00Z",
            },
        ),
        (
            build_incident_notify_data,
            {
                "incident_id": "inc_123",
                "state": "investigating",
                "severity": "high",
                "resource_version": 2,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
                "resolved_at": None,
                "narrative": "Mitigation is in progress.",
            },
            {
                "incident_id": "inc_123",
                "state": "investigating",
                "severity": "high",
                "resource_version": 2,
                "created_at": "2026-08-31T12:00:00Z",
                "updated_at": "2026-08-31T12:05:00Z",
                "resolved_at": None,
                "narrative": "Mitigation is in progress.",
            },
        ),
    ),
)
def test_production_builders_emit_exact_privacy_bounded_shapes(
    builder,
    kwargs: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert builder(**kwargs) == expected


@pytest.mark.unit
def test_production_builder_signatures_expose_only_approved_fields() -> None:
    user_fields = {
        "user_id",
        "resource_version",
        "created_at",
        "updated_at",
    }
    incident_fields = {
        "incident_id",
        "state",
        "severity",
        "resource_version",
        "created_at",
        "updated_at",
        "resolved_at",
    }

    assert set(signature(build_user_created_data).parameters) == user_fields
    assert set(signature(build_user_deleted_data).parameters) == user_fields
    assert set(signature(build_incident_created_data).parameters) == incident_fields
    assert set(signature(build_incident_updated_data).parameters) == incident_fields
    assert set(signature(build_incident_resolved_data).parameters) == incident_fields
    assert set(signature(build_incident_notify_data).parameters) == {
        *incident_fields,
        "narrative",
    }


@pytest.mark.unit
@pytest.mark.parametrize("user_id", (True, 0, -1, "7"))
def test_user_builders_reject_invalid_stable_ids(user_id: object) -> None:
    with pytest.raises(ValueError, match="user ID"):
        build_user_created_data(
            user_id=user_id,  # type: ignore[arg-type]
            resource_version=PROFILE_VERSION,
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("incident_id", ""),
        ("incident_id", "x" * 256),
        ("state", "unknown"),
        ("severity", "urgent"),
        ("resource_version", True),
        ("resource_version", 0),
    ),
)
def test_incident_builders_reject_invalid_lifecycle_fields(
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "incident_id": "inc_123",
        "state": "investigating",
        "severity": "high",
        "resource_version": 2,
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
        "resolved_at": None,
    }
    kwargs[field] = value

    error_field = "incident ID" if field == "incident_id" else field.replace("_", " ")
    with pytest.raises(ValueError, match=error_field):
        build_incident_updated_data(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_event_builders_reject_naive_or_reversed_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_user_created_data(
            user_id=7,
            resource_version=PROFILE_VERSION.replace(tzinfo=None),
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
        )

    with pytest.raises(ValueError, match="timestamp order"):
        build_incident_updated_data(
            incident_id="inc_123",
            state="investigating",
            severity="high",
            resource_version=2,
            created_at=UPDATED_AT,
            updated_at=CREATED_AT,
            resolved_at=None,
        )


@pytest.mark.unit
def test_resolved_builder_requires_resolved_state_and_timestamp() -> None:
    with pytest.raises(ValueError, match="resolved incident"):
        build_incident_resolved_data(
            incident_id="inc_123",
            state="mitigating",
            severity="high",
            resource_version=3,
            created_at=CREATED_AT,
            updated_at=RESOLVED_AT,
            resolved_at=RESOLVED_AT,
        )


@pytest.mark.unit
def test_incident_notify_narrative_is_verbatim_and_bounded() -> None:
    narrative = " x "
    data = build_incident_notify_data(
        incident_id="inc_123",
        state="investigating",
        severity="high",
        resource_version=2,
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
        resolved_at=None,
        narrative=narrative,
    )
    assert data["narrative"] == narrative

    with pytest.raises(ValueError, match="narrative"):
        build_incident_notify_data(
            incident_id="inc_123",
            state="investigating",
            severity="high",
            resource_version=2,
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
            resolved_at=None,
            narrative="x" * 4097,
        )


@pytest.mark.unit
def test_event_snapshot_and_body_are_deterministic_utf8_bytes() -> None:
    source = {"unicode": "caf\u00e9", "nested": {"b": 2, "a": [1]}}
    snapshot = snapshot_json_object(source)
    source["nested"]["a"].append(2)  # type: ignore[index,union-attr]

    body = canonical_event_body(
        event_id=EVENT_ID,
        event_type="user.created",
        api_version="2026-07-01",
        created_at=CREATED_AT,
        data=snapshot,
    )

    assert body == (
        b'{"api_version":"2026-07-01","created_at":"2026-08-31T12:00:00Z",'
        b'"data":{"nested":{"a":[1],"b":2},"unicode":"caf\xc3\xa9"},'
        b'"id":"5f16ee87-4e72-49f7-b027-41f15f26a90f",'
        b'"type":"user.created"}'
    )
    assert json.loads(body)["data"] == snapshot


@pytest.mark.unit
@pytest.mark.parametrize(
    "data",
    (
        {"invalid": float("nan")},
        {"invalid": float("inf")},
        {1: "non-string-key"},
        {"invalid": {1, 2}},
    ),
)
def test_event_snapshot_rejects_non_json_data(data: dict[object, object]) -> None:
    with pytest.raises(ValueError, match="event data"):
        snapshot_json_object(data)  # type: ignore[arg-type]


@pytest.mark.unit
def test_event_snapshot_rejects_excessive_nesting() -> None:
    nested: dict[str, object] = {}
    cursor = nested
    for _ in range(65):
        child: dict[str, object] = {}
        cursor["child"] = child
        cursor = child

    with pytest.raises(ValueError, match="nesting"):
        snapshot_json_object(nested)


@pytest.mark.unit
def test_canonical_body_accepts_65536_bytes_and_rejects_one_more() -> None:
    prefix = (
        b'{"api_version":"2026-07-01","created_at":"2026-08-31T12:00:00Z",'
        b'"data":{"blob":"'
    )
    suffix = (
        b'"},"id":"'
        + EVENT_ID.encode("ascii")
        + b'","type":"incident.notify"}'
    )
    blob_size = EVENT_BODY_MAX_BYTES - len(prefix) - len(suffix)

    accepted = canonical_event_body(
        event_id=EVENT_ID,
        event_type="incident.notify",
        api_version="2026-07-01",
        created_at=CREATED_AT,
        data={"blob": "x" * blob_size},
    )
    assert len(accepted) == EVENT_BODY_MAX_BYTES

    with pytest.raises(ValueError, match="too large"):
        canonical_event_body(
            event_id=EVENT_ID,
            event_type="incident.notify",
            api_version="2026-07-01",
            created_at=CREATED_AT,
            data={"blob": "x" * (blob_size + 1)},
        )


@pytest.mark.unit
def test_prepared_event_is_encrypted_and_replay_verification_is_exact() -> None:
    ring = _ring()
    prepared = prepare_event_insert(
        ring=ring,
        event_id=EVENT_ID,
        event_type="user.created",
        api_version="2026-07-01",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id="register-command-1",
        source_component="registration-service",
        source_request_id="request-1",
        created_at=CREATED_AT,
        data=build_user_created_data(
            user_id=7,
            resource_version=PROFILE_VERSION,
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
        ),
    )
    plaintext = ring.decrypt_event_body(
        event_id=prepared.event.id,
        api_version=prepared.event.api_version,
        protected=prepared.event.body,
    )

    assert prepared.event.body_size_bytes == len(plaintext)
    assert b'"user_id":7' in plaintext
    assert "user_id" not in prepared.event.body.ciphertext_json
    stored = StoredWebhookEvent(
        event=WebhookEvent(
            id=prepared.event.id,
            event_type=prepared.event.event_type,
            api_version=prepared.event.api_version,
            source_kind=prepared.event.source_kind,
            created_at=prepared.event.created_at,
        ),
        aggregate_type=prepared.event.aggregate_type,
        aggregate_id=prepared.event.aggregate_id,
        aggregate_version=prepared.event.aggregate_version,
        source_command_id=prepared.event.source_command_id,
        source_component=prepared.event.source_component,
        source_request_id=prepared.event.source_request_id,
        body=prepared.event.body,
        body_size_bytes=prepared.event.body_size_bytes,
    )
    replay = EventCaptureResult(event=stored, deliveries=(), inserted=False)

    verify_event_replay(ring=ring, result=replay, prepared=prepared)

    wrong_source = EventCaptureResult(
        event=replace(stored, source_component="another-service"),
        deliveries=(),
        inserted=False,
    )
    with pytest.raises(WebhookError) as source_conflict:
        verify_event_replay(ring=ring, result=wrong_source, prepared=prepared)
    assert source_conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT

    different_body = prepare_event_insert(
        ring=ring,
        event_id="644574d3-6db0-44f0-bddb-c3203730c5e1",
        event_type="user.created",
        api_version="2026-07-01",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id="register-command-1",
        source_component="registration-service",
        source_request_id="request-1",
        created_at=UPDATED_AT,
        data=build_user_created_data(
            user_id=8,
            resource_version=PROFILE_VERSION,
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
        ),
    )
    with pytest.raises(WebhookError) as body_conflict:
        verify_event_replay(ring=ring, result=replay, prepared=different_body)
    assert body_conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.unit
@pytest.mark.parametrize("mode", (AdminWebhookMode.OFF, AdminWebhookMode.MIGRATE))
async def test_production_capture_is_a_noop_until_mode_on(
    mode: AdminWebhookMode,
) -> None:
    repository = _ProducerRepositoryProbe(_migration_state())
    producer = AdminWebhookEventProducer(
        repository=repository,
        settings=_settings(mode),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=_ring(),
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: "b4663f95-9f18-48ea-817e-6a98f4f596bd",
        clock=lambda: CREATED_AT,
    )

    preparation = await producer.begin_capture(
        source_component="registration-service",
        source_request_id="request-1",
    )

    assert preparation is None
    assert repository.reads == 0


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "key_result", "code"),
    (
        (
            _migration_state(phase="migration_pending", completed_at=None),
            WebhookKeyRingLoadResult(
                ring=_ring(),
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            WebhookErrorCode.MIGRATION_PENDING,
        ),
        (
            _migration_state(rotation_phase="rewriting"),
            WebhookKeyRingLoadResult(
                ring=_ring(),
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            WebhookErrorCode.KEY_ROTATION_IN_PROGRESS,
        ),
        (
            _migration_state(),
            WebhookKeyRingLoadResult(
                ring=None,
                code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
            ),
            WebhookErrorCode.KEY_UNAVAILABLE,
        ),
        (
            _migration_state(active_primary_key_id="another-key"),
            WebhookKeyRingLoadResult(
                ring=_ring(),
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            WebhookErrorCode.KEY_CONFIGURATION_MISMATCH,
        ),
    ),
)
async def test_production_capture_preflight_fails_closed(
    state: SimpleNamespace,
    key_result: WebhookKeyRingLoadResult,
    code: WebhookErrorCode,
) -> None:
    producer = AdminWebhookEventProducer(
        repository=_ProducerRepositoryProbe(state),
        settings=_settings(),
        key_ring_result=key_result,
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: "b4663f95-9f18-48ea-817e-6a98f4f596bd",
        clock=lambda: CREATED_AT,
    )

    with pytest.raises(WebhookError) as denied:
        await producer.begin_capture(
            source_component="registration-service",
            source_request_id="request-1",
        )

    assert denied.value.code is code


@pytest.mark.unit
async def test_production_capture_revalidates_and_encrypts_inside_source_transaction() -> None:
    ring = _ring()
    repository = _ProducerRepositoryProbe(_migration_state())
    producer = AdminWebhookEventProducer(
        repository=repository,
        settings=_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: "b4663f95-9f18-48ea-817e-6a98f4f596bd",
        clock=lambda: CREATED_AT,
    )
    preparation = await producer.begin_capture(
        source_component="registration-service",
        source_request_id="request-1",
    )
    assert preparation is not None
    tx = _CaptureUnitProbe(_migration_state())
    data = build_user_created_data(
        user_id=7,
        resource_version=PROFILE_VERSION,
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )

    result = await producer.capture_in_transaction(
        preparation,
        tx=tx,
        event_type="user.created",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id="registration-command-1",
        data=data,
    )

    assert result.inserted is True
    assert tx.lock_calls == 1
    assert tx.capture_calls == 1
    assert tx.event.api_version == EVENT_API_VERSION
    plaintext = ring.decrypt_event_body(
        event_id=tx.event.id,
        api_version=tx.event.api_version,
        protected=tx.event.body,
    )
    assert json.loads(plaintext)["data"] == data


@pytest.mark.unit
async def test_production_capture_rejects_unreviewed_payload_and_source_coordinates() -> None:
    producer = AdminWebhookEventProducer(
        repository=_ProducerRepositoryProbe(_migration_state()),
        settings=_settings(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=_ring(),
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: "b4663f95-9f18-48ea-817e-6a98f4f596bd",
        clock=lambda: CREATED_AT,
    )
    preparation = await producer.begin_capture(
        source_component="registration-service",
        source_request_id="request-1",
    )
    assert preparation is not None
    data = build_user_created_data(
        user_id=7,
        resource_version=PROFILE_VERSION,
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )
    data["email"] = "private@example.com"
    tx = _CaptureUnitProbe(_migration_state())

    with pytest.raises(WebhookError) as denied:
        await producer.capture_in_transaction(
            preparation,
            tx=tx,
            event_type="user.created",
            source_kind=EventSourceKind.AGGREGATE,
            aggregate_type="user",
            aggregate_id="7",
            aggregate_version="1",
            source_command_id=None,
            data=data,
        )

    assert denied.value.code is WebhookErrorCode.VALIDATION_FAILED
    assert tx.capture_calls == 0
