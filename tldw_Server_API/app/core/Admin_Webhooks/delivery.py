"""Internal synthetic capture and delivery-lifecycle composition."""

from __future__ import annotations

import hmac
import json
import logging
import math
import os
import re
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, cast
from uuid import UUID, uuid4

from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError

from .audit import DeliveryMutationAudit, DeliveryMutationAuditSink
from .catalog import EVENT_API_VERSION, EVENT_CATALOG
from .config import AdminWebhookMode, AdminWebhookSettings
from .crypto import (
    EVENT_BODY_MAX_BYTES,
    WebhookKeyError,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from .domain import (
    AttemptState,
    DeliveryHistoryPage,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryState,
    EventSourceKind,
    IdempotencyScope,
    ValidatedWebhookTarget,
    WebhookDelivery,
    WebhookDeliveryAttempt,
    WebhookError,
    WebhookErrorCode,
    WebhookRegistration,
    build_idempotency_scope,
    canonical_request_hash,
    idempotency_lookup_digest,
    parse_registration_etag,
    validate_idempotency_key,
    validate_webhook_target,
)
from .executor import (
    AttemptExecutionRequest,
    AttemptOutcome,
    DeliveryAttemptExecutor,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        AdminWebhookRepository,
        EventCaptureResult,
        EventInsert,
        IdempotencyLookup,
        MigrationState,
        StoredWebhookEvent,
        StoredWebhookRegistration,
    )

_ACTIVE_ROTATION_PHASES = frozenset(
    {"rewriting", "verifying", "awaiting_primary_cutover"}
)
_CATALOG_EVENTS = frozenset(item.event_type for item in EVENT_CATALOG)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
_SIGNING_SECRET = re.compile(r"^whsec_[0-9a-f]{64}$")
_MAX_JSON_DEPTH = 64
_TEST_REPLAY_RETRY_SECONDS = 5

logger = logging.getLogger(__name__)

EventCaptureOutcome: TypeAlias = Literal["accepted", "failed"]


def _delivery_mode_error(
    settings: AdminWebhookSettings | None,
) -> WebhookErrorCode | None:
    """Return the sole public denial for the configured delivery mode."""

    if settings is None or settings.mode is AdminWebhookMode.OFF:
        return WebhookErrorCode.DISABLED
    if settings.mode is AdminWebhookMode.MIGRATE:
        return WebhookErrorCode.MIGRATION_PENDING
    if settings.mode is not AdminWebhookMode.ON:
        return WebhookErrorCode.DISABLED
    return None


def _bounded_text(
    value: object,
    *,
    field_name: str,
    maximum: int,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{field_name} is invalid")
    return value


def _canonical_uuid4(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} is invalid")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        raise ValueError(f"{field_name} is invalid") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field_name} is invalid")
    return value


@dataclass(frozen=True)
class CaptureSyntheticEventCommand:
    """Validated internal command with no transport- or receiver-owned fields."""

    actor_id: int
    request_id: str
    event_type: str
    source_kind: EventSourceKind
    aggregate_type: str | None
    aggregate_id: str | None
    aggregate_version: str | None
    source_command_id: str | None
    source_component: str
    source_request_id: str | None
    data: dict[str, object] = field(repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.actor_id, bool) or not isinstance(self.actor_id, int) or self.actor_id < 1:
            raise ValueError("actor ID is invalid")
        if not isinstance(self.request_id, str) or _SAFE_ID.fullmatch(self.request_id) is None:
            raise ValueError("request ID is invalid")
        if not isinstance(self.event_type, str) or self.event_type not in _CATALOG_EVENTS:
            raise ValueError("event type is invalid")
        if not isinstance(self.source_kind, EventSourceKind):
            raise ValueError("source identity is invalid")
        _bounded_text(
            self.source_component,
            field_name="source component",
            maximum=64,
        )
        _bounded_text(
            self.source_request_id,
            field_name="source request ID",
            maximum=128,
            optional=True,
        )
        aggregate = (
            self.aggregate_type,
            self.aggregate_id,
            self.aggregate_version,
        )
        if self.source_kind is EventSourceKind.AGGREGATE:
            if self.source_command_id is not None or any(value is None for value in aggregate):
                raise ValueError("source identity is invalid")
            _bounded_text(self.aggregate_type, field_name="aggregate type", maximum=64)
            _bounded_text(self.aggregate_id, field_name="aggregate ID", maximum=255)
            _bounded_text(
                self.aggregate_version,
                field_name="aggregate version",
                maximum=255,
            )
        elif any(value is not None for value in aggregate) or self.source_command_id is None:
            raise ValueError("source identity is invalid")
        else:
            _bounded_text(
                self.source_command_id,
                field_name="source command ID",
                maximum=255,
            )
        if not isinstance(self.data, dict):
            raise ValueError("data must be a JSON object")


@dataclass(frozen=True)
class EventCaptureAudit:
    """Closed mandatory-audit record for one capture decision."""

    event_type: str
    event_id: str | None
    fanout_count: int
    actor_id: int
    request_id: str
    outcome: EventCaptureOutcome
    reason_code: WebhookErrorCode | None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or self.event_type not in _CATALOG_EVENTS:
            raise ValueError("event type is invalid")
        if self.event_id is not None:
            _canonical_uuid4(self.event_id, field_name="event ID")
        if (
            isinstance(self.fanout_count, bool)
            or not isinstance(self.fanout_count, int)
            or self.fanout_count < 0
        ):
            raise ValueError("fanout count is invalid")
        if isinstance(self.actor_id, bool) or not isinstance(self.actor_id, int) or self.actor_id < 1:
            raise ValueError("actor ID is invalid")
        if not isinstance(self.request_id, str) or _SAFE_ID.fullmatch(self.request_id) is None:
            raise ValueError("request ID is invalid")
        if self.outcome not in {"accepted", "failed"}:
            raise ValueError("capture outcome is invalid")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            WebhookErrorCode,
        ):
            raise TypeError("capture reason code is invalid")
        if (self.outcome == "accepted") != (self.reason_code is None):
            raise ValueError("capture reason code does not match outcome")


EventCaptureAuditSink: TypeAlias = Callable[[EventCaptureAudit], Awaitable[None]]

TestWebhookOutcome: TypeAlias = Literal["accepted", "succeeded", "failed"]
TestWebhookReasonCode: TypeAlias = DeliveryReasonCode | WebhookErrorCode


@dataclass(frozen=True)
class TestWebhookCommand:
    """Internal inputs for one idempotent synchronous test delivery."""

    actor_id: int
    webhook_id: int
    if_match: str | None = field(repr=False)
    delivery_config_version: int
    idempotency_key: str = field(repr=False)
    request_id: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.actor_id, "actor ID"),
            (self.webhook_id, "webhook ID"),
            (self.delivery_config_version, "delivery config version"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} is invalid")
        if self.if_match is not None and not isinstance(self.if_match, str):
            raise ValueError("registration ETag is invalid")
        if not isinstance(self.idempotency_key, str):
            raise ValueError("idempotency key is invalid")
        if not isinstance(self.request_id, str) or _SAFE_ID.fullmatch(self.request_id) is None:
            raise ValueError("request ID is invalid")


@dataclass(frozen=True)
class TestWebhookResult:
    """Stored one-attempt result projected for the future Task 10 route."""

    delivery: WebhookDelivery
    attempt: WebhookDeliveryAttempt
    idempotent_replay: bool
    in_progress: bool
    retry_after_seconds: int | None
    completed_after_config_change: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.delivery, WebhookDelivery)
            or not isinstance(self.attempt, WebhookDeliveryAttempt)
            or self.delivery.kind is not DeliveryKind.TEST
            or self.attempt.delivery_id != self.delivery.id
            or self.attempt.attempt_number != 1
        ):
            raise ValueError("test result coordinates are invalid")
        if not isinstance(self.idempotent_replay, bool) or not isinstance(
            self.in_progress,
            bool,
        ):
            raise TypeError("test replay state is invalid")
        if self.in_progress != (
            self.delivery.state is DeliveryState.PROCESSING
            and self.attempt.state is AttemptState.PROCESSING
        ):
            raise ValueError("test processing state is invalid")
        if self.retry_after_seconds != (
            _TEST_REPLAY_RETRY_SECONDS if self.in_progress else None
        ):
            raise ValueError("test retry guidance is invalid")
        if not isinstance(self.completed_after_config_change, bool):
            raise TypeError("test configuration-change state is invalid")


@dataclass(frozen=True)
class TestWebhookAudit:
    """Bounded audit projection without request or receiver secrets."""

    actor_id: int
    webhook_id: int
    delivery_id: str
    attempt_id: str
    target_hostname: str
    request_id: str
    outcome: TestWebhookOutcome
    status_code: int | None
    reason_code: TestWebhookReasonCode | None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.actor_id, "actor ID"),
            (self.webhook_id, "webhook ID"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} is invalid")
        _canonical_uuid4(self.delivery_id, field_name="delivery ID")
        _canonical_uuid4(self.attempt_id, field_name="attempt ID")
        _bounded_text(
            self.target_hostname,
            field_name="target hostname",
            maximum=253,
        )
        if not isinstance(self.request_id, str) or _SAFE_ID.fullmatch(self.request_id) is None:
            raise ValueError("request ID is invalid")
        if self.outcome not in {"accepted", "succeeded", "failed"}:
            raise ValueError("test audit outcome is invalid")
        if self.status_code is not None and not 100 <= self.status_code <= 599:
            raise ValueError("test audit status is invalid")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            (DeliveryReasonCode, WebhookErrorCode),
        ):
            raise TypeError("test audit reason is invalid")
        if self.outcome in {"accepted", "succeeded"} and self.reason_code is not None:
            raise ValueError("test audit reason does not match outcome")
        if self.outcome == "failed" and self.reason_code is None:
            raise ValueError("failed test audit requires a reason")


TestWebhookAuditSink: TypeAlias = Callable[[TestWebhookAudit], Awaitable[None]]


@dataclass(frozen=True)
class RedeliverWebhookCommand:
    """Internal inputs for one idempotent manual redelivery."""

    actor_id: int
    webhook_id: int
    source_delivery_id: str
    if_match: str | None = field(repr=False)
    delivery_config_version: int
    confirm_changed_configuration: bool
    idempotency_key: str = field(repr=False)
    request_id: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.actor_id, "actor ID"),
            (self.webhook_id, "webhook ID"),
            (self.delivery_config_version, "delivery config version"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} is invalid")
        _canonical_uuid4(self.source_delivery_id, field_name="source delivery ID")
        if self.if_match is not None and not isinstance(self.if_match, str):
            raise ValueError("registration ETag is invalid")
        if not isinstance(self.confirm_changed_configuration, bool):
            raise TypeError("redelivery confirmation is invalid")
        if not isinstance(self.idempotency_key, str):
            raise ValueError("idempotency key is invalid")
        if not isinstance(self.request_id, str) or _SAFE_ID.fullmatch(self.request_id) is None:
            raise ValueError("request ID is invalid")


@dataclass(frozen=True)
class RedeliverWebhookResult:
    """Sanitized manual delivery result without Jobs or protected material."""

    delivery: WebhookDelivery
    event_type: str
    completed_after_config_change: bool
    idempotent_replay: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.delivery, WebhookDelivery)
            or self.delivery.kind is not DeliveryKind.MANUAL
            or self.delivery.redelivery_of_id is None
        ):
            raise ValueError("manual redelivery result is invalid")
        _bounded_text(self.event_type, field_name="event type", maximum=64)
        if not isinstance(self.completed_after_config_change, bool) or not isinstance(
            self.idempotent_replay,
            bool,
        ):
            raise TypeError("manual redelivery result state is invalid")


@dataclass(frozen=True)
class _PreparedTest:
    registration: StoredWebhookRegistration = field(repr=False)
    event: EventInsert = field(repr=False)
    target: ValidatedWebhookTarget = field(repr=False)
    body: bytes = field(repr=False)
    signing_secret: str = field(repr=False)
    scope: IdempotencyScope
    lookup_digest: str
    request_fingerprint: str
    expected_revision: int
    started_at: datetime
    delivery_id: str
    attempt_id: str
    test_attempt_token: str = field(repr=False)


class _CaptureAuditUnavailable(MandatoryAuditWriteError):
    pass


def _utc(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _canonical_timestamp(value: datetime) -> str:
    return _utc(value, field_name="created_at").isoformat().replace("+00:00", "Z")


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > _MAX_JSON_DEPTH:
        raise ValueError("event data nesting is invalid")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("event data number is invalid")
        return
    if isinstance(value, list):
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("event data key is invalid")
        for item in value.values():
            _validate_json_value(item, depth=depth + 1)
        return
    raise ValueError("event data value is invalid")


def _snapshot_json_object(data: dict[str, object]) -> dict[str, object]:
    _validate_json_value(data)
    try:
        encoded = json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        snapshot = json.loads(encoded)
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("event data is invalid") from exc
    if not isinstance(snapshot, dict):
        raise ValueError("event data must be a JSON object")
    return snapshot


def _canonical_event_body(
    *,
    event_id: str,
    event_type: str,
    api_version: str,
    created_at: datetime,
    data: dict[str, object],
) -> bytes:
    try:
        encoded = json.dumps(
            {
                "id": event_id,
                "type": event_type,
                "api_version": api_version,
                "created_at": _canonical_timestamp(created_at),
                "data": data,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (OverflowError, RecursionError, TypeError, ValueError) as exc:
        raise ValueError("event body is invalid") from exc
    if len(encoded) > EVENT_BODY_MAX_BYTES:
        raise ValueError("event body is too large")
    return encoded


def validate_stored_event_body(
    event: StoredWebhookEvent,
    plaintext: bytes,
) -> None:
    """Require exact canonical bytes matching immutable event metadata."""

    if not isinstance(plaintext, bytes) or len(plaintext) != event.body_size_bytes:
        raise ValueError("persisted event body is invalid")
    try:
        decoded = json.loads(plaintext)
        if not isinstance(decoded, dict) or not isinstance(decoded.get("data"), dict):
            raise ValueError("persisted event body is invalid")
        data = _snapshot_json_object(decoded["data"])
        expected = _canonical_event_body(
            event_id=event.event.id,
            event_type=event.event.event_type,
            api_version=event.event.api_version,
            created_at=event.event.created_at,
            data=data,
        )
    except (OverflowError, RecursionError, TypeError, UnicodeError, ValueError) as exc:
        raise ValueError("persisted event body is invalid") from exc
    if (
        len(expected) != event.body_size_bytes
        or not hmac.compare_digest(plaintext, expected)
    ):
        raise ValueError("persisted event body is invalid")


def registration_work_lifecycle_reason(
    delivery: WebhookDelivery,
    registration: WebhookRegistration,
) -> DeliveryReasonCode | None:
    """Select the specific no-I/O reason for stale registration work."""
    if not isinstance(delivery, WebhookDelivery) or not isinstance(
        registration,
        WebhookRegistration,
    ):
        raise TypeError("delivery lifecycle inputs are invalid")
    if registration.deleted_at is not None:
        return DeliveryReasonCode.CANCELED_DELETED
    if not registration.active:
        return DeliveryReasonCode.CANCELED_DISABLED
    if delivery.secret_version != registration.secret_version:
        return DeliveryReasonCode.CANCELED_SECRET_ROTATION
    if delivery.delivery_config_version != registration.delivery_config_version:
        return DeliveryReasonCode.SUPERSEDED_CONFIG
    return None


def _map_capture_error(exc: BaseException) -> WebhookError:
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        WebhookRepositoryError,
        WebhookRepositoryErrorCode,
    )

    if isinstance(exc, WebhookError):
        return exc
    if isinstance(exc, WebhookRepositoryError):
        if exc.code is WebhookRepositoryErrorCode.DATABASE_BUSY:
            return WebhookError(WebhookErrorCode.DATABASE_BUSY)
        return WebhookError(WebhookErrorCode.OPERATION_FAILED)
    if isinstance(exc, WebhookKeyError):
        return WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
    if isinstance(exc, (TypeError, ValueError, OverflowError, RecursionError)):
        return WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    return WebhookError(WebhookErrorCode.OPERATION_FAILED)


def _require_migration_ready(state: MigrationState) -> None:
    if state.phase != "complete" or state.completed_at is None:
        raise WebhookError(WebhookErrorCode.MIGRATION_PENDING)


def _require_writable_ring(
    state: MigrationState,
    key_ring_result: WebhookKeyRingLoadResult,
) -> WebhookKeyRing:
    _require_migration_ready(state)
    if state.rotation_phase in _ACTIVE_ROTATION_PHASES:
        raise WebhookError(WebhookErrorCode.KEY_ROTATION_IN_PROGRESS)
    ring = key_ring_result.ring
    if ring is None:
        raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
    if state.active_primary_key_id != ring.primary_id:
        raise WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH)
    return ring


def _source_matches(
    event: StoredWebhookEvent,
    command: CaptureSyntheticEventCommand,
) -> bool:
    return (
        event.event.event_type == command.event_type
        and event.event.source_kind is command.source_kind
        and event.aggregate_type == command.aggregate_type
        and event.aggregate_id == command.aggregate_id
        and event.aggregate_version == command.aggregate_version
        and event.source_command_id == command.source_command_id
        and event.source_component == command.source_component
        and event.source_request_id == command.source_request_id
    )


def _verify_replay(
    *,
    ring: WebhookKeyRing,
    result: EventCaptureResult,
    command: CaptureSyntheticEventCommand,
    data: dict[str, object],
) -> None:
    event = result.event
    if not _source_matches(event, command):
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
    expected = _canonical_event_body(
        event_id=event.id,
        event_type=event.event.event_type,
        api_version=event.event.api_version,
        created_at=event.event.created_at,
        data=data,
    )
    try:
        plaintext = ring.decrypt_event_body(
            event_id=event.id,
            api_version=event.event.api_version,
            protected=event.body,
        )
    except WebhookKeyError:
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT) from None
    if (
        event.body_size_bytes != len(expected)
        or event.body_size_bytes != len(plaintext)
        or not hmac.compare_digest(plaintext, expected)
    ):
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)


class _DeliveryMetrics(Protocol):
    def events_committed(self, *, event_type: str, fanout_count: int) -> None: ...

    def attempt_committed(
        self,
        *,
        state: DeliveryState,
        kind: DeliveryKind,
        reason_code: DeliveryReasonCode | None,
        delivery_reason_code: DeliveryReasonCode | None,
        status_code: int | None,
        latency_ms: int | None,
    ) -> None: ...


class AdminWebhookDeliveryService:
    """Compose protected event capture without owning SQL or cryptography."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        key_ring_result: WebhookKeyRingLoadResult,
        event_id_factory: Callable[[], str],
        delivery_id_factory: Callable[[], str],
        clock: Callable[[], datetime],
        settings: AdminWebhookSettings | None = None,
        executor: DeliveryAttemptExecutor | None = None,
        test_attempt_id_factory: Callable[[], str] | None = None,
        test_token_factory: Callable[[], str] | None = None,
        metrics: _DeliveryMetrics | None = None,
    ) -> None:
        if not callable(event_id_factory) or not callable(delivery_id_factory):
            raise TypeError("event and delivery ID factories are required")
        if not callable(clock):
            raise TypeError("capture clock is required")
        self._repository = repository
        self._key_ring_result = key_ring_result
        self._event_id_factory = event_id_factory
        self._delivery_id_factory = delivery_id_factory
        self._clock = clock
        self._settings = settings
        self._executor = executor
        self._test_attempt_id_factory = test_attempt_id_factory
        self._test_token_factory = test_token_factory
        self._metrics = metrics

    def _require_delivery_mode(self) -> AdminWebhookSettings:
        settings = self._settings
        error = _delivery_mode_error(settings)
        if error is not None:
            raise WebhookError(error)
        return cast(AdminWebhookSettings, settings)

    def _require_test_dependencies(
        self,
    ) -> tuple[AdminWebhookSettings, DeliveryAttemptExecutor, Callable[[], str], Callable[[], str]]:
        if (
            self._settings is None
            or self._executor is None
            or self._test_attempt_id_factory is None
            or self._test_token_factory is None
            or not callable(self._test_attempt_id_factory)
            or not callable(self._test_token_factory)
        ):
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
        return (
            self._settings,
            self._executor,
            self._test_attempt_id_factory,
            self._test_token_factory,
        )

    @staticmethod
    def _test_result(
        snapshot: object,
        *,
        replay: bool,
    ) -> TestWebhookResult:
        from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
            TestAttemptSnapshot,
        )

        if not isinstance(snapshot, TestAttemptSnapshot):
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
        processing = (
            snapshot.delivery.delivery.state is DeliveryState.PROCESSING
            and snapshot.attempt.state is AttemptState.PROCESSING
        )
        return TestWebhookResult(
            delivery=snapshot.delivery.delivery,
            attempt=snapshot.attempt,
            idempotent_replay=replay,
            in_progress=processing,
            retry_after_seconds=(
                _TEST_REPLAY_RETRY_SECONDS if processing else None
            ),
            completed_after_config_change=(
                snapshot.delivery.completed_after_config_change
            ),
        )

    async def _resolve_test_lookup(
        self,
        lookup: IdempotencyLookup,
        *,
        command: TestWebhookCommand,
    ) -> TestWebhookResult | None:
        from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
            IdempotencyLookupKind,
        )

        if lookup.kind is IdempotencyLookupKind.NEW:
            return None
        if lookup.kind is IdempotencyLookupKind.CONFLICT:
            raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
        if lookup.kind is IdempotencyLookupKind.IN_PROGRESS and (
            lookup.test_delivery_id is None or lookup.test_attempt_id is None
        ):
            raise WebhookError(WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS)
        if lookup.test_delivery_id is None or lookup.test_attempt_id is None:
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
        snapshot = await self._repository.get_test_attempt_snapshot(
            lookup.test_delivery_id,
            lookup.test_attempt_id,
        )
        if (
            snapshot is None
            or snapshot.delivery.delivery.webhook_id != command.webhook_id
        ):
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
        return self._test_result(snapshot, replay=True)

    async def _prepare_test(
        self,
        command: TestWebhookCommand,
        *,
        settings: AdminWebhookSettings,
        test_attempt_id_factory: Callable[[], str],
        test_token_factory: Callable[[], str],
    ) -> tuple[_PreparedTest | None, TestWebhookResult | None]:
        from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
            EventInsert,
        )

        validate_idempotency_key(command.idempotency_key)
        expected_revision = parse_registration_etag(
            command.if_match,
            expected_webhook_id=command.webhook_id,
        )
        scope = build_idempotency_scope(
            actor_id=command.actor_id,
            operation="test",
            route=f"/admin/webhooks/{command.webhook_id}/test",
            webhook_id=command.webhook_id,
        )
        lookup_digest = idempotency_lookup_digest(command.idempotency_key, scope)
        request_fingerprint = canonical_request_hash(
            command.idempotency_key,
            scope=scope,
            body={"delivery_config_version": command.delivery_config_version},
            conditional_version=expected_revision,
        )
        started_at = _utc(self._clock(), field_name="test start time")
        lookup = await self._repository.lookup_idempotency(
            lookup_digest=lookup_digest,
            scope=scope,
            request_fingerprint=request_fingerprint,
            now=started_at,
        )
        replay = await self._resolve_test_lookup(lookup, command=command)
        if replay is not None:
            return None, replay

        state = await self._repository.get_migration_state()
        ring = _require_writable_ring(state, self._key_ring_result)
        registration = await self._repository.get_protected_registration(
            command.webhook_id,
            include_deleted=False,
        )
        if registration is None:
            raise WebhookError(WebhookErrorCode.NOT_FOUND)
        current = registration.registration
        if (
            current.revision != expected_revision
            or current.delivery_config_version != command.delivery_config_version
        ):
            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
        if current.secret_rotation_required:
            raise WebhookError(WebhookErrorCode.SECRET_ROTATION_REQUIRED)
        if (
            registration.target.key_id != ring.primary_id
            or registration.secret.key_id != ring.primary_id
        ):
            raise WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH)
        target_url = ring.decrypt_text(
            purpose="registration.target",
            identity={
                "registration_id": current.id,
                "target_version": current.target_version,
            },
            protected=registration.target,
        )
        signing_secret = ring.decrypt_text(
            purpose="registration.secret",
            identity={
                "registration_id": current.id,
                "secret_version": current.secret_version,
            },
            protected=registration.secret,
        )
        target = validate_webhook_target(
            target_url,
            allow_http_dev=settings.allow_http_dev,
        )
        if (
            target.hostname != current.target_hostname
            or _SIGNING_SECRET.fullmatch(signing_secret) is None
        ):
            raise WebhookError(WebhookErrorCode.OPERATION_FAILED)

        event_id = _canonical_uuid4(
            self._event_id_factory(),
            field_name="event ID",
        )
        delivery_id = _canonical_uuid4(
            self._delivery_id_factory(),
            field_name="delivery ID",
        )
        attempt_id = _canonical_uuid4(
            test_attempt_id_factory(),
            field_name="attempt ID",
        )
        test_attempt_token = test_token_factory()
        if not isinstance(test_attempt_token, str) or re.fullmatch(
            r"[0-9a-f]{64}",
            test_attempt_token,
        ) is None:
            raise WebhookError(WebhookErrorCode.OPERATION_FAILED)
        data = {"test": True, "webhook_id": current.id}
        body = _canonical_event_body(
            event_id=event_id,
            event_type="webhook.test",
            api_version=EVENT_API_VERSION,
            created_at=started_at,
            data=data,
        )
        event = EventInsert(
            id=event_id,
            event_type="webhook.test",
            api_version=EVENT_API_VERSION,
            source_kind=EventSourceKind.COMMAND,
            aggregate_type=None,
            aggregate_id=None,
            aggregate_version=None,
            source_command_id=f"test:{event_id}",
            source_component="admin_webhooks.test",
            source_request_id=command.request_id,
            body=ring.encrypt_event_body(
                event_id=event_id,
                api_version=EVENT_API_VERSION,
                body=body,
            ),
            body_size_bytes=len(body),
            created_at=started_at,
        )
        return (
            _PreparedTest(
                registration=registration,
                event=event,
                target=target,
                body=body,
                signing_secret=signing_secret,
                scope=scope,
                lookup_digest=lookup_digest,
                request_fingerprint=request_fingerprint,
                expected_revision=expected_revision,
                started_at=started_at,
                delivery_id=delivery_id,
                attempt_id=attempt_id,
                test_attempt_token=test_attempt_token,
            ),
            None,
        )

    @staticmethod
    def _test_audit(
        command: TestWebhookCommand,
        *,
        delivery_id: str,
        attempt_id: str,
        target_hostname: str,
        outcome: TestWebhookOutcome,
        status_code: int | None = None,
        reason_code: TestWebhookReasonCode | None = None,
    ) -> TestWebhookAudit:
        return TestWebhookAudit(
            actor_id=command.actor_id,
            webhook_id=command.webhook_id,
            delivery_id=delivery_id,
            attempt_id=attempt_id,
            target_hostname=target_hostname,
            request_id=command.request_id,
            outcome=outcome,
            status_code=status_code,
            reason_code=reason_code,
        )

    async def test_webhook(
        self,
        command: TestWebhookCommand,
        *,
        audit_sink: TestWebhookAuditSink,
    ) -> TestWebhookResult:
        """Persist, execute, and close exactly one synchronous test attempt."""

        from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
            IdempotencyLookupKind,
            TestAttemptCompletion,
        )

        if not isinstance(command, TestWebhookCommand):
            raise TypeError("test command is required")
        if not callable(audit_sink):
            raise TypeError("test audit sink is required")
        self._require_delivery_mode()
        settings, executor, attempt_id_factory, token_factory = (
            self._require_test_dependencies()
        )
        try:
            prepared, replay = await self._prepare_test(
                command,
                settings=settings,
                test_attempt_id_factory=attempt_id_factory,
                test_token_factory=token_factory,
            )
        except Exception as exc:  # noqa: BLE001 - internal boundary is closed
            raise _map_capture_error(exc) from None
        if replay is not None:
            return replay
        if prepared is None:
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)

        accepted_emitted = False
        reservation = None
        raced_lookup = None
        try:
            async with self._repository.transaction() as tx:
                claim = await tx.claim_idempotency(
                    lookup_digest=prepared.lookup_digest,
                    scope=prepared.scope,
                    request_fingerprint=prepared.request_fingerprint,
                    now=prepared.started_at,
                    expires_at=prepared.started_at
                    + timedelta(seconds=settings.idempotency_ttl_seconds),
                )
                if claim.kind is IdempotencyLookupKind.CONFLICT:
                    raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
                if claim.kind is not IdempotencyLookupKind.NEW:
                    raced_lookup = claim
                else:
                    locked_state = await tx.lock_migration_state()
                    _require_writable_ring(locked_state, self._key_ring_result)
                    current = prepared.registration.registration
                    reservation = await tx.start_test_attempt(
                        prepared.event,
                        webhook_id=command.webhook_id,
                        delivery_id=prepared.delivery_id,
                        attempt_id=prepared.attempt_id,
                        test_attempt_token=prepared.test_attempt_token,
                        request_timeout_seconds=current.timeout_seconds,
                        expected_revision=prepared.expected_revision,
                        expected_delivery_config_version=(
                            command.delivery_config_version
                        ),
                        expected_target_version=current.target_version,
                        expected_secret_version=current.secret_version,
                        expected_target=prepared.registration.target,
                        expected_secret=prepared.registration.secret,
                        lookup_digest=prepared.lookup_digest,
                        request_fingerprint=prepared.request_fingerprint,
                        started_at=prepared.started_at,
                        expires_at=prepared.started_at + timedelta(hours=72),
                    )
                    if reservation is None:
                        raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                    try:
                        await audit_sink(
                            self._test_audit(
                                command,
                                delivery_id=prepared.delivery_id,
                                attempt_id=prepared.attempt_id,
                                target_hostname=(
                                    prepared.registration.registration.target_hostname
                                ),
                                outcome="accepted",
                            )
                        )
                        accepted_emitted = True
                    except Exception:  # noqa: BLE001 - accepted audit is mandatory
                        raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
        except Exception as exc:  # noqa: BLE001 - start failures cannot reach egress
            error = _map_capture_error(exc)
            if accepted_emitted:
                try:
                    await audit_sink(
                        self._test_audit(
                            command,
                            delivery_id=prepared.delivery_id,
                            attempt_id=prepared.attempt_id,
                            target_hostname=(
                                prepared.registration.registration.target_hostname
                            ),
                            outcome="failed",
                            reason_code=error.code,
                        )
                    )
                except Exception:  # noqa: BLE001 - preserve the commit failure
                    logger.warning(
                        "Admin webhook test commit-failure audit could not be recorded"
                    )
            raise error from None

        if raced_lookup is not None:
            raced_replay = await self._resolve_test_lookup(
                raced_lookup,
                command=command,
            )
            if raced_replay is None:
                raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
            return raced_replay
        if reservation is None or not reservation.start_owner:
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)

        execution = await executor.execute(
            AttemptExecutionRequest(
                target=prepared.target,
                body=prepared.body,
                signing_secret=prepared.signing_secret,
                timeout_seconds=prepared.registration.registration.timeout_seconds,
                event_type="webhook.test",
                event_id=prepared.event.id,
                delivery_id=prepared.delivery_id,
                attempt_number=1,
                secret_version=prepared.registration.registration.secret_version,
                kind=DeliveryKind.TEST,
            )
        )
        finished_at = _utc(self._clock(), field_name="test finish time")
        if execution.outcome is AttemptOutcome.SUCCESS:
            attempt_state = AttemptState.SUCCEEDED
            delivery_state = DeliveryState.SUCCEEDED
            reason_code = None
            audit_outcome: TestWebhookOutcome = "succeeded"
        else:
            if execution.reason_code is None:
                raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
            attempt_state = AttemptState.FAILED
            delivery_state = DeliveryState.DEAD
            reason_code = DeliveryReasonCode(execution.reason_code.value)
            audit_outcome = "failed"
        completion = TestAttemptCompletion(
            attempt_state=attempt_state,
            delivery_state=delivery_state,
            status_code=execution.status_code,
            latency_ms=execution.latency_ms,
            reason_code=reason_code,
            finished_at=finished_at,
        )
        try:
            async with self._repository.transaction() as tx:
                completed = await tx.finish_test_attempt(
                    prepared.delivery_id,
                    prepared.attempt_id,
                    prepared.test_attempt_token,
                    lookup_digest=prepared.lookup_digest,
                    request_fingerprint=prepared.request_fingerprint,
                    outcome=completion,
                )
        except Exception as exc:  # noqa: BLE001 - completion boundary is closed
            raise _map_capture_error(exc) from None
        if completed is None:
            raise WebhookError(WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE)
        if self._metrics is not None:
            try:
                self._metrics.attempt_committed(
                    state=completion.delivery_state,
                    kind=DeliveryKind.TEST,
                    reason_code=completion.reason_code,
                    delivery_reason_code=completion.reason_code,
                    status_code=completion.status_code,
                    latency_ms=completion.latency_ms,
                )
            except Exception:  # noqa: BLE001 - metrics cannot alter durable truth
                pass
        try:
            await audit_sink(
                self._test_audit(
                    command,
                    delivery_id=prepared.delivery_id,
                    attempt_id=prepared.attempt_id,
                    target_hostname=(
                        prepared.registration.registration.target_hostname
                    ),
                    outcome=audit_outcome,
                    status_code=execution.status_code,
                    reason_code=reason_code,
                )
            )
        except Exception:  # noqa: BLE001 - durable receiver truth wins
            logger.warning("Admin webhook test completion audit unavailable")
        return self._test_result(completed, replay=False)

    async def list_delivery_history(
        self,
        webhook_id: int,
        *,
        limit: int,
        offset: int,
    ) -> DeliveryHistoryPage:
        """Return key-independent sanitized history for one retained registration."""
        if isinstance(webhook_id, bool) or not isinstance(webhook_id, int) or webhook_id < 1:
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
        self._require_delivery_mode()
        try:
            return await self._repository.list_delivery_history(
                webhook_id,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:  # noqa: BLE001 - public boundary is closed
            from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
                WebhookRepositoryError,
                WebhookRepositoryErrorCode,
            )

            if (
                isinstance(exc, WebhookRepositoryError)
                and exc.code is WebhookRepositoryErrorCode.NOT_FOUND
            ):
                raise WebhookError(WebhookErrorCode.NOT_FOUND) from None
            raise _map_capture_error(exc) from None

    @staticmethod
    async def _emit_delivery_mutation(
        audit_sink: DeliveryMutationAuditSink,
        record: DeliveryMutationAudit,
    ) -> None:
        try:
            await audit_sink(record)
        except Exception:  # noqa: BLE001 - mandatory audit failure is closed
            raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None

    @staticmethod
    def _redelivery_audit(
        command: RedeliverWebhookCommand,
        *,
        delivery_id: str | None,
        target_hostname: str | None,
        source_config_version: int | None,
        current_config_version: int | None,
        changed_config: bool | None,
        outcome: Literal["accepted", "no_op", "denied", "failed"],
        reason_code: WebhookErrorCode | None = None,
    ) -> DeliveryMutationAudit:
        return DeliveryMutationAudit(
            actor_id=command.actor_id,
            action="admin_webhook.redeliver",
            webhook_id=command.webhook_id,
            source_delivery_id=command.source_delivery_id,
            delivery_id=delivery_id,
            attempt_id=None,
            target_hostname=target_hostname,
            source_config_version=source_config_version,
            current_config_version=current_config_version,
            redelivery_to_changed_config=changed_config,
            status_code=None,
            outcome=outcome,
            request_id=command.request_id,
            reason_code=reason_code,
        )

    @staticmethod
    def _redelivery_error(exc: BaseException) -> WebhookError:
        return _map_capture_error(exc)

    @staticmethod
    def _redelivery_failure_outcome(
        error: WebhookError,
    ) -> Literal["denied", "failed"]:
        if error.code in {
            WebhookErrorCode.DISABLED,
            WebhookErrorCode.MIGRATION_PENDING,
        }:
            return "denied"
        return "failed" if error.code.http_status >= 500 else "denied"

    async def _resolve_redelivery_lookup(
        self,
        lookup: IdempotencyLookup,
        *,
        command: RedeliverWebhookCommand,
        audit_sink: DeliveryMutationAuditSink,
    ) -> RedeliverWebhookResult | None:
        from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
            IdempotencyLookupKind,
        )

        if lookup.kind is IdempotencyLookupKind.NEW:
            return None
        if lookup.kind is IdempotencyLookupKind.CONFLICT:
            raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
        if lookup.kind is IdempotencyLookupKind.IN_PROGRESS:
            raise WebhookError(WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS)
        replay_delivery_id = lookup.redelivery_delivery_id
        if replay_delivery_id is None:
            raise WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
        created = await self._repository.get_delivery_history_item(
            command.webhook_id,
            replay_delivery_id,
        )
        source = await self._repository.get_delivery_history_item(
            command.webhook_id,
            command.source_delivery_id,
        )
        if (
            created is None
            or source is None
            or created.delivery.kind is not DeliveryKind.MANUAL
            or created.delivery.redelivery_of_id != command.source_delivery_id
            or created.delivery.event_id != source.delivery.event_id
        ):
            raise WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
        changed_config = (
            source.delivery.delivery_config_version
            != created.delivery.delivery_config_version
        )
        await self._emit_delivery_mutation(
            audit_sink,
            self._redelivery_audit(
                command,
                delivery_id=created.delivery.id,
                target_hostname=None,
                source_config_version=source.delivery.delivery_config_version,
                current_config_version=created.delivery.delivery_config_version,
                changed_config=changed_config,
                outcome="no_op",
            ),
        )
        return RedeliverWebhookResult(
            delivery=created.delivery,
            event_type=created.event_type,
            completed_after_config_change=(
                created.completed_after_config_change
            ),
            idempotent_replay=True,
        )

    async def _resolve_redelivery_lookup_audited(
        self,
        lookup: IdempotencyLookup,
        *,
        command: RedeliverWebhookCommand,
        audit_sink: DeliveryMutationAuditSink,
    ) -> RedeliverWebhookResult | None:
        try:
            return await self._resolve_redelivery_lookup(
                lookup,
                command=command,
                audit_sink=audit_sink,
            )
        except ValueError:
            resolved_error = WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
        except WebhookError as exc:
            if exc.code is WebhookErrorCode.AUDIT_UNAVAILABLE:
                raise
            resolved_error = exc
        except Exception as exc:  # noqa: BLE001 - replay reads fail closed
            resolved_error = self._redelivery_error(exc)
        await self._emit_delivery_mutation(
            audit_sink,
            self._redelivery_audit(
                command,
                delivery_id=None,
                target_hostname=None,
                source_config_version=None,
                current_config_version=None,
                changed_config=None,
                outcome=self._redelivery_failure_outcome(resolved_error),
                reason_code=resolved_error.code,
            ),
        )
        raise resolved_error from None

    async def redeliver_webhook(
        self,
        command: RedeliverWebhookCommand,
        *,
        audit_sink: DeliveryMutationAuditSink,
    ) -> RedeliverWebhookResult:
        """Create one pending manual delivery without admitting Jobs work."""
        from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
            IdempotencyLookupKind,
        )

        if not isinstance(command, RedeliverWebhookCommand):
            raise TypeError("redelivery command is required")
        if not callable(audit_sink):
            raise TypeError("redelivery audit sink is required")
        expected_revision: int | None = None
        lookup_digest: str | None = None
        request_fingerprint: str | None = None
        scope: IdempotencyScope | None = None
        try:
            settings = self._require_delivery_mode()
            observed_at = _utc(self._clock(), field_name="redelivery start time")
            validate_idempotency_key(command.idempotency_key)
            expected_revision = parse_registration_etag(
                command.if_match,
                expected_webhook_id=command.webhook_id,
            )
            scope = build_idempotency_scope(
                actor_id=command.actor_id,
                operation="redeliver",
                route=(
                    f"/admin/webhooks/{command.webhook_id}/deliveries/"
                    f"{command.source_delivery_id}/redeliver"
                ),
                webhook_id=command.webhook_id,
                delivery_id=command.source_delivery_id,
            )
            lookup_scope = build_idempotency_scope(
                actor_id=command.actor_id,
                operation="redeliver",
                route=(
                    f"/admin/webhooks/{command.webhook_id}/deliveries/redeliver"
                ),
                webhook_id=command.webhook_id,
            )
            lookup_digest = idempotency_lookup_digest(
                command.idempotency_key,
                lookup_scope,
            )
            request_fingerprint = canonical_request_hash(
                command.idempotency_key,
                scope=scope,
                body={
                    "delivery_config_version": command.delivery_config_version,
                    "confirm_changed_configuration": (
                        command.confirm_changed_configuration
                    ),
                },
                conditional_version=expected_revision,
            )
            lookup = await self._repository.lookup_idempotency(
                lookup_digest=lookup_digest,
                scope=scope,
                request_fingerprint=request_fingerprint,
                now=observed_at,
            )
        except ValueError:
            error = WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
            await self._emit_delivery_mutation(
                audit_sink,
                self._redelivery_audit(
                    command,
                    delivery_id=None,
                    target_hostname=None,
                    source_config_version=None,
                    current_config_version=None,
                    changed_config=None,
                    outcome="failed",
                    reason_code=error.code,
                ),
            )
            raise error from None
        except Exception as exc:  # noqa: BLE001 - lookup boundary is closed
            error = self._redelivery_error(exc)
            await self._emit_delivery_mutation(
                audit_sink,
                self._redelivery_audit(
                    command,
                    delivery_id=None,
                    target_hostname=None,
                    source_config_version=None,
                    current_config_version=None,
                    changed_config=None,
                    outcome=self._redelivery_failure_outcome(error),
                    reason_code=error.code,
                ),
            )
            raise error from None
        replay = await self._resolve_redelivery_lookup_audited(
            lookup,
            command=command,
            audit_sink=audit_sink,
        )
        if replay is not None:
            return replay
        if expected_revision is None or scope is None or lookup_digest is None or request_fingerprint is None:
            raise WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)

        try:
            delivery_id = _canonical_uuid4(
                self._delivery_id_factory(),
                field_name="delivery ID",
            )
        except Exception as exc:  # noqa: BLE001 - secure factory boundary is closed
            error = (
                WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
                if isinstance(exc, ValueError)
                else self._redelivery_error(exc)
            )
            await self._emit_delivery_mutation(
                audit_sink,
                self._redelivery_audit(
                    command,
                    delivery_id=None,
                    target_hostname=None,
                    source_config_version=None,
                    current_config_version=None,
                    changed_config=None,
                    outcome="failed",
                    reason_code=error.code,
                ),
            )
            raise error from None
        accepted_emitted = False
        error_audited = False
        raced_lookup = None
        created = None
        event_type: str | None = None
        target_hostname: str | None = None
        source_config_version: int | None = None
        current_config_version: int | None = None
        changed_config: bool | None = None
        try:
            async with self._repository.transaction() as tx:
                try:
                    claim = await tx.claim_idempotency(
                        lookup_digest=lookup_digest,
                        scope=scope,
                        request_fingerprint=request_fingerprint,
                        now=observed_at,
                        expires_at=observed_at
                        + timedelta(seconds=settings.idempotency_ttl_seconds),
                    )
                    if claim.kind is IdempotencyLookupKind.CONFLICT:
                        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_CONFLICT)
                    if claim.kind is not IdempotencyLookupKind.NEW:
                        raced_lookup = claim
                    else:
                        locked_state = await tx.lock_migration_state()
                        ring = _require_writable_ring(
                            locked_state,
                            self._key_ring_result,
                        )
                        current_stored = await tx.get_protected_registration(
                            command.webhook_id,
                            include_deleted=False,
                            lock=True,
                        )
                        if current_stored is None:
                            raise WebhookError(WebhookErrorCode.NOT_FOUND)
                        current = current_stored.registration
                        target_hostname = current.target_hostname
                        current_config_version = current.delivery_config_version
                        if not current.active:
                            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                        if (
                            current.revision != expected_revision
                            or current.delivery_config_version
                            != command.delivery_config_version
                        ):
                            raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
                        if current.secret_rotation_required:
                            raise WebhookError(
                                WebhookErrorCode.SECRET_ROTATION_REQUIRED
                            )
                        source_bundle = await tx.get_delivery_bundle(
                            command.source_delivery_id
                        )
                        if (
                            source_bundle is None
                            or source_bundle.delivery.delivery.webhook_id
                            != command.webhook_id
                        ):
                            raise WebhookError(WebhookErrorCode.NOT_FOUND)
                        source = source_bundle.delivery.delivery
                        source_config_version = source.delivery_config_version
                        changed_config = (
                            source_config_version != current_config_version
                        )
                        if (
                            changed_config
                            and not command.confirm_changed_configuration
                        ):
                            raise WebhookError(
                                WebhookErrorCode.REDELIVERY_CONFIRMATION_REQUIRED
                            )
                        plaintext = ring.decrypt_event_body(
                            event_id=source_bundle.event.event.id,
                            api_version=source_bundle.event.event.api_version,
                            protected=source_bundle.event.body,
                        )
                        try:
                            validate_stored_event_body(
                                source_bundle.event,
                                plaintext,
                            )
                        except ValueError:
                            raise WebhookError(WebhookErrorCode.OPERATION_FAILED) from None
                        event_type = source_bundle.event.event.event_type
                        created = await tx.insert_delivery(
                            delivery_id,
                            event_id=source.event_id,
                            webhook_id=command.webhook_id,
                            kind=DeliveryKind.MANUAL,
                            expires_at=observed_at + timedelta(hours=72),
                            now=observed_at,
                            redelivery_of_id=source.id,
                        )
                        await tx.complete_idempotency(
                            lookup_digest=lookup_digest,
                            request_fingerprint=request_fingerprint,
                            resource_id=None,
                            resource_version=None,
                            response_status=202,
                            response_metadata={
                                "redelivery_delivery_id": delivery_id
                            },
                            at=observed_at,
                        )
                        await tx.mark_first_canonical_activity(
                            "delivery_attempt",
                            observed_at,
                        )
                        await self._emit_delivery_mutation(
                            audit_sink,
                            self._redelivery_audit(
                                command,
                                delivery_id=delivery_id,
                                target_hostname=target_hostname,
                                source_config_version=source_config_version,
                                current_config_version=current_config_version,
                                changed_config=changed_config,
                                outcome="accepted",
                            ),
                        )
                        accepted_emitted = True
                except Exception as exc:  # noqa: BLE001 - audit before rollback
                    error = self._redelivery_error(exc)
                    if error.code is WebhookErrorCode.AUDIT_UNAVAILABLE:
                        raise error from None
                    await self._emit_delivery_mutation(
                        audit_sink,
                        self._redelivery_audit(
                            command,
                            delivery_id=(delivery_id if created is not None else None),
                            target_hostname=target_hostname,
                            source_config_version=source_config_version,
                            current_config_version=current_config_version,
                            changed_config=changed_config,
                            outcome=self._redelivery_failure_outcome(error),
                            reason_code=error.code,
                        ),
                    )
                    error_audited = True
                    raise error from None
        except Exception as exc:  # noqa: BLE001 - correlate commit failure
            error = self._redelivery_error(exc)
            if accepted_emitted:
                try:
                    await self._emit_delivery_mutation(
                        audit_sink,
                        self._redelivery_audit(
                            command,
                            delivery_id=delivery_id,
                            target_hostname=target_hostname,
                            source_config_version=source_config_version,
                            current_config_version=current_config_version,
                            changed_config=changed_config,
                            outcome="failed",
                            reason_code=error.code,
                        ),
                    )
                except WebhookError:
                    logger.warning(
                        "Admin webhook redelivery commit-failure audit unavailable"
                    )
            elif not error_audited and error.code is not WebhookErrorCode.AUDIT_UNAVAILABLE:
                await self._emit_delivery_mutation(
                    audit_sink,
                    self._redelivery_audit(
                        command,
                        delivery_id=None,
                        target_hostname=target_hostname,
                        source_config_version=source_config_version,
                        current_config_version=current_config_version,
                        changed_config=changed_config,
                        outcome=self._redelivery_failure_outcome(error),
                        reason_code=error.code,
                    ),
                )
            raise error from None

        if raced_lookup is not None:
            raced = await self._resolve_redelivery_lookup_audited(
                raced_lookup,
                command=command,
                audit_sink=audit_sink,
            )
            if raced is None:
                raise WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
            return raced
        if created is None or event_type is None:
            raise WebhookError(WebhookErrorCode.DELIVERY_UNAVAILABLE)
        return RedeliverWebhookResult(
            delivery=created.delivery,
            event_type=event_type,
            completed_after_config_change=created.completed_after_config_change,
            idempotent_replay=False,
        )

    @staticmethod
    async def _emit(
        sink: EventCaptureAuditSink,
        record: EventCaptureAudit,
    ) -> None:
        try:
            await sink(record)
        except Exception as exc:  # noqa: BLE001 - audit sink failures are fail-closed
            raise _CaptureAuditUnavailable from exc

    @staticmethod
    def _audit_record(
        command: CaptureSyntheticEventCommand,
        *,
        event_id: str | None,
        fanout_count: int,
        outcome: EventCaptureOutcome,
        reason_code: WebhookErrorCode | None,
    ) -> EventCaptureAudit:
        return EventCaptureAudit(
            event_type=command.event_type,
            event_id=event_id,
            fanout_count=fanout_count,
            actor_id=command.actor_id,
            request_id=command.request_id,
            outcome=outcome,
            reason_code=reason_code,
        )

    async def _emit_failed(
        self,
        command: CaptureSyntheticEventCommand,
        sink: EventCaptureAuditSink,
        *,
        event_id: str | None,
        fanout_count: int,
        error: WebhookError,
    ) -> None:
        await self._emit(
            sink,
            self._audit_record(
                command,
                event_id=event_id,
                fanout_count=fanout_count,
                outcome="failed",
                reason_code=error.code,
            ),
        )

    async def capture_synthetic_event(
        self,
        command: CaptureSyntheticEventCommand,
        *,
        audit_sink: EventCaptureAuditSink,
    ) -> EventCaptureResult:
        """Persist and fan out one encrypted synthetic event transactionally."""
        if not isinstance(command, CaptureSyntheticEventCommand):
            raise TypeError("capture command is required")
        event_id: str | None = None
        fanout_count = 0
        try:
            self._require_delivery_mode()
            from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
                EventInsert,
            )

            data = _snapshot_json_object(command.data)
            event_id = _canonical_uuid4(
                self._event_id_factory(),
                field_name="event ID",
            )
            created_at = _utc(self._clock(), field_name="created_at")
            body = _canonical_event_body(
                event_id=event_id,
                event_type=command.event_type,
                api_version=EVENT_API_VERSION,
                created_at=created_at,
                data=data,
            )
            state = await self._repository.get_migration_state()
            ring = _require_writable_ring(state, self._key_ring_result)
            event = EventInsert(
                id=event_id,
                event_type=command.event_type,
                api_version=EVENT_API_VERSION,
                source_kind=command.source_kind,
                aggregate_type=command.aggregate_type,
                aggregate_id=command.aggregate_id,
                aggregate_version=command.aggregate_version,
                source_command_id=command.source_command_id,
                source_component=command.source_component,
                source_request_id=command.source_request_id,
                body=ring.encrypt_event_body(
                    event_id=event_id,
                    api_version=EVENT_API_VERSION,
                    body=body,
                ),
                body_size_bytes=len(body),
                created_at=created_at,
            )
        except Exception as exc:  # noqa: BLE001 - boundary must audit preparation failures
            error = _map_capture_error(exc)
            try:
                await self._emit_failed(
                    command,
                    audit_sink,
                    event_id=event_id,
                    fanout_count=0,
                    error=error,
                )
            except _CaptureAuditUnavailable:
                raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
            raise error from None
        accepted_emitted = False
        failed_emitted = False
        try:
            async with self._repository.transaction() as tx:
                try:
                    locked = await tx.lock_migration_state()
                    locked_ring = _require_writable_ring(
                        locked,
                        self._key_ring_result,
                    )
                    result = await tx.capture_event_and_expand(
                        event,
                        self._delivery_id_factory,
                        created_at + timedelta(hours=72),
                    )
                    event_id = result.event.id
                    fanout_count = len(result.deliveries)
                    if not result.inserted:
                        _verify_replay(
                            ring=locked_ring,
                            result=result,
                            command=command,
                            data=data,
                        )
                except Exception as exc:  # noqa: BLE001 - boundary must audit transaction failures
                    error = _map_capture_error(exc)
                    await self._emit_failed(
                        command,
                        audit_sink,
                        event_id=event_id,
                        fanout_count=fanout_count,
                        error=error,
                    )
                    failed_emitted = True
                    raise error from None
                await self._emit(
                    audit_sink,
                    self._audit_record(
                        command,
                        event_id=result.event.id,
                        fanout_count=len(result.deliveries),
                        outcome="accepted",
                        reason_code=None,
                    ),
                )
                accepted_emitted = True
            if result.inserted and self._metrics is not None:
                try:
                    self._metrics.events_committed(
                        event_type=result.event.event.event_type,
                        fanout_count=len(result.deliveries),
                    )
                except Exception:  # noqa: BLE001 - metrics are fail-open
                    pass
            return result
        except _CaptureAuditUnavailable:
            raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
        except Exception as exc:  # noqa: BLE001 - boundary must correlate commit failures
            error = _map_capture_error(exc)
            if accepted_emitted:
                try:
                    await self._emit_failed(
                        command,
                        audit_sink,
                        event_id=event_id,
                        fanout_count=fanout_count,
                        error=error,
                    )
                except _CaptureAuditUnavailable:
                    pass
            elif not failed_emitted:
                try:
                    await self._emit_failed(
                        command,
                        audit_sink,
                        event_id=event_id,
                        fanout_count=fanout_count,
                        error=error,
                    )
                except _CaptureAuditUnavailable:
                    raise WebhookError(WebhookErrorCode.AUDIT_UNAVAILABLE) from None
            raise error from None


async def get_admin_webhook_delivery_service() -> AdminWebhookDeliveryService:
    """Compose delivery operations from application-scoped validated resources."""
    settings = AdminWebhookSettings.from_environment(os.environ)
    if _delivery_mode_error(settings) is not None:
        return AdminWebhookDeliveryService(
            repository=object(),
            key_ring_result=WebhookKeyRingLoadResult(
                ring=None,
                code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
            ),
            event_id_factory=lambda: str(uuid4()),
            delivery_id_factory=lambda: str(uuid4()),
            clock=lambda: datetime.now(timezone.utc),
            settings=settings,
        )

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        AdminWebhookRepository,
    )

    from .crypto import load_webhook_key_ring
    from .observability import AdminWebhookMetrics

    pool = await get_db_pool()
    return AdminWebhookDeliveryService(
        repository=AdminWebhookRepository(pool),
        key_ring_result=load_webhook_key_ring(),
        event_id_factory=lambda: str(uuid4()),
        delivery_id_factory=lambda: str(uuid4()),
        clock=lambda: datetime.now(timezone.utc),
        settings=settings,
        executor=DeliveryAttemptExecutor(
            allow_http_dev=settings.allow_http_dev,
        ),
        test_attempt_id_factory=lambda: str(uuid4()),
        test_token_factory=lambda: secrets.token_hex(32),
        metrics=AdminWebhookMetrics(),
    )


__all__ = [
    "AdminWebhookDeliveryService",
    "CaptureSyntheticEventCommand",
    "EventCaptureAudit",
    "EventCaptureAuditSink",
    "get_admin_webhook_delivery_service",
    "registration_work_lifecycle_reason",
    "validate_stored_event_body",
]
