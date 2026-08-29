"""Internal synthetic capture and delivery-lifecycle composition."""

from __future__ import annotations

import hmac
import json
import math
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Literal, TypeAlias
from uuid import UUID

from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError

from .catalog import EVENT_API_VERSION, EVENT_CATALOG
from .crypto import EVENT_BODY_MAX_BYTES, WebhookKeyError, WebhookKeyRing, WebhookKeyRingLoadResult
from .domain import (
    DeliveryReasonCode,
    EventSourceKind,
    WebhookDelivery,
    WebhookError,
    WebhookErrorCode,
    WebhookRegistration,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        AdminWebhookRepository,
        EventCaptureResult,
        MigrationState,
        StoredWebhookEvent,
    )

_ACTIVE_ROTATION_PHASES = frozenset(
    {"rewriting", "verifying", "awaiting_primary_cutover"}
)
_CATALOG_EVENTS = frozenset(item.event_type for item in EVENT_CATALOG)
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
_MAX_JSON_DEPTH = 64

EventCaptureOutcome: TypeAlias = Literal["accepted", "failed"]


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
    created_at: datetime,
    data: dict[str, object],
) -> bytes:
    try:
        encoded = json.dumps(
            {
                "id": event_id,
                "type": event_type,
                "api_version": EVENT_API_VERSION,
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
        and event.event.api_version == EVENT_API_VERSION
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


__all__ = [
    "AdminWebhookDeliveryService",
    "CaptureSyntheticEventCommand",
    "EventCaptureAudit",
    "EventCaptureAuditSink",
    "registration_work_lifecycle_reason",
]
