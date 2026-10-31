"""Bounded mandatory-audit adapters for canonical admin webhooks."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias
from uuid import UUID

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
    UnifiedAuditService,
)

from .catalog import EVENT_CATALOG
from .domain import DeliveryReasonCode, WebhookErrorCode

AUDIT_WRITE_TIMEOUT_SECONDS = 5.0
AUDIT_STOP_TIMEOUT_SECONDS = 1.0

MutationAction: TypeAlias = Literal[
    "admin_webhook.create",
    "admin_webhook.patch",
    "admin_webhook.delete",
    "admin_webhook.rotate_secret",
]
MutationOutcome: TypeAlias = Literal["accepted", "no_op", "denied", "failed"]
OperationalAction: TypeAlias = Literal[
    "admin_webhook.import.apply",
    "admin_webhook.import.reject_source",
    "admin_webhook.key_rotation.start",
    "admin_webhook.key_rotation.resume",
    "admin_webhook.key_rotation.verify",
    "admin_webhook.key_rotation.finalize",
    "admin_webhook.rollback.extract",
    "admin_webhook.rollback.destroy",
]
OperationalOutcome: TypeAlias = Literal["accepted", "completed", "failed"]
DeliveryMutationAction: TypeAlias = Literal[
    "admin_webhook.test",
    "admin_webhook.redeliver",
]
DeliveryMutationOutcome: TypeAlias = Literal[
    "accepted",
    "succeeded",
    "no_op",
    "denied",
    "failed",
]
DeliveryMutationReasonCode: TypeAlias = DeliveryReasonCode | WebhookErrorCode

_MUTATION_ACTIONS = frozenset(
    {
        "admin_webhook.create",
        "admin_webhook.patch",
        "admin_webhook.delete",
        "admin_webhook.rotate_secret",
    }
)
_MUTATION_OUTCOMES = frozenset({"accepted", "no_op", "denied", "failed"})
_OPERATIONAL_ACTIONS = frozenset(
    {
        "admin_webhook.import.apply",
        "admin_webhook.import.reject_source",
        "admin_webhook.key_rotation.start",
        "admin_webhook.key_rotation.resume",
        "admin_webhook.key_rotation.verify",
        "admin_webhook.key_rotation.finalize",
        "admin_webhook.rollback.extract",
        "admin_webhook.rollback.destroy",
    }
)
_OPERATIONAL_OUTCOMES = frozenset({"accepted", "completed", "failed"})
_DELIVERY_MUTATION_ACTIONS = frozenset(
    {"admin_webhook.test", "admin_webhook.redeliver"}
)
_DELIVERY_MUTATION_OUTCOMES = frozenset(
    {"accepted", "succeeded", "no_op", "denied", "failed"}
)
_PRINCIPAL_KINDS = frozenset({"user", "api_key", "service", "anonymous"})
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:@-]{1,128}$")
_SAFE_ROLE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_SAFE_HOSTNAME = re.compile(r"^[A-Za-z0-9.:-]{1,253}$")
_CATALOG_EVENTS = frozenset(item.event_type for item in EVENT_CATALOG)


class WebhookOperationalReasonCode(str, Enum):
    """Closed reason codes shared by import, rotation, and rollback commands."""

    VALIDATION_FAILED = "admin_webhook_validation_failed"
    PRECONDITION_FAILED = "precondition_failed"
    DATABASE_BUSY = "admin_webhook_database_busy"
    AUDIT_UNAVAILABLE = "admin_webhook_audit_unavailable"
    KEY_UNAVAILABLE = "admin_webhook_key_unavailable"
    KEY_CONFIGURATION_MISMATCH = "admin_webhook_key_configuration_mismatch"
    LEGACY_SOURCE_CHANGED = "admin_webhook_legacy_source_changed"
    ROLLBACK_WINDOW_CLOSED = "admin_webhook_rollback_window_closed"
    ROLLBACK_ARTIFACTS_ALREADY_RETIRED = (
        "admin_webhook_rollback_artifacts_already_retired"
    )
    ROLLBACK_ARTIFACTS_NOT_APPLICABLE = (
        "admin_webhook_rollback_artifacts_not_applicable"
    )
    OPERATION_FAILED = "admin_webhook_operation_failed"


def _validate_positive_id(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_safe_id(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise ValueError(f"{field} is invalid")
    return value


def _validate_canonical_uuid4(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is invalid")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        raise ValueError(f"{field} is invalid") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field} is invalid")
    return value


def validate_actor_principal_id(value: str) -> str:
    """Validate a bounded pseudonymous principal ID without truncation."""
    return _validate_safe_id(value, field="actor principal ID")


def validate_actor_kind(value: str | None) -> str | None:
    """Validate the closed AuthNZ principal-kind vocabulary."""
    if value is None:
        return None
    if not isinstance(value, str) or value not in _PRINCIPAL_KINDS:
        raise ValueError("actor kind is invalid")
    return value


def validate_actor_roles(values: tuple[str, ...]) -> tuple[str, ...]:
    """Return a deterministic, bounded role set suitable for audit metadata."""
    if not isinstance(values, tuple) or len(values) > 32:
        raise ValueError("actor roles are invalid")
    if any(not isinstance(value, str) or _SAFE_ROLE.fullmatch(value) is None for value in values):
        raise ValueError("actor roles are invalid")
    return tuple(sorted(set(values)))


def _validate_event_types(values: object) -> tuple[str, ...]:
    if not isinstance(values, tuple) or len(values) > len(EVENT_CATALOG):
        raise ValueError("audit event types are invalid")
    if len(set(values)) != len(values) or any(
        not isinstance(value, str) or value not in _CATALOG_EVENTS
        for value in values
    ):
        raise ValueError("audit event types are invalid")
    return values


@dataclass(frozen=True)
class MutationAudit:
    """Closed, redacted mutation record accepted by the mandatory sink."""

    actor_id: int
    action: MutationAction
    webhook_id: int | None
    target_hostname: str | None
    event_types: tuple[str, ...]
    outcome: MutationOutcome
    request_id: str
    reason_code: WebhookErrorCode | None

    def __post_init__(self) -> None:
        _validate_positive_id(self.actor_id, field="actor ID")
        if self.action not in _MUTATION_ACTIONS:
            raise ValueError("mutation action is invalid")
        if self.webhook_id is not None:
            _validate_positive_id(self.webhook_id, field="webhook ID")
        if self.target_hostname is not None and (
            not isinstance(self.target_hostname, str)
            or _SAFE_HOSTNAME.fullmatch(self.target_hostname) is None
        ):
            raise ValueError("target hostname is invalid")
        _validate_event_types(self.event_types)
        if self.outcome not in _MUTATION_OUTCOMES:
            raise ValueError("mutation outcome is invalid")
        _validate_safe_id(self.request_id, field="request ID")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            WebhookErrorCode,
        ):
            raise TypeError("mutation reason code is invalid")


MutationAuditSink: TypeAlias = Callable[[MutationAudit], Awaitable[None]]


@dataclass(frozen=True)
class DeliveryMutationAudit:
    """Closed test/redelivery audit record without transport-owned content."""

    actor_id: int
    action: DeliveryMutationAction
    webhook_id: int
    source_delivery_id: str | None
    delivery_id: str | None
    attempt_id: str | None
    target_hostname: str | None
    source_config_version: int | None
    current_config_version: int | None
    redelivery_to_changed_config: bool | None
    status_code: int | None
    outcome: DeliveryMutationOutcome
    request_id: str
    reason_code: DeliveryMutationReasonCode | None

    def __post_init__(self) -> None:
        _validate_positive_id(self.actor_id, field="actor ID")
        _validate_positive_id(self.webhook_id, field="webhook ID")
        if self.action not in _DELIVERY_MUTATION_ACTIONS:
            raise ValueError("delivery mutation action is invalid")
        for value, field_name in (
            (self.source_delivery_id, "source delivery ID"),
            (self.delivery_id, "delivery ID"),
            (self.attempt_id, "attempt ID"),
        ):
            if value is not None:
                _validate_canonical_uuid4(value, field=field_name)
        if self.target_hostname is not None and (
            not isinstance(self.target_hostname, str)
            or _SAFE_HOSTNAME.fullmatch(self.target_hostname) is None
        ):
            raise ValueError("target hostname is invalid")
        for value, field_name in (
            (self.source_config_version, "source config version"),
            (self.current_config_version, "current config version"),
        ):
            if value is not None:
                _validate_positive_id(value, field=field_name)
        if self.redelivery_to_changed_config is not None and not isinstance(
            self.redelivery_to_changed_config,
            bool,
        ):
            raise TypeError("redelivery configuration-change state is invalid")
        if self.status_code is not None and (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("delivery mutation status is invalid")
        if self.outcome not in _DELIVERY_MUTATION_OUTCOMES:
            raise ValueError("delivery mutation outcome is invalid")
        _validate_safe_id(self.request_id, field="request ID")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            (DeliveryReasonCode, WebhookErrorCode),
        ):
            raise TypeError("delivery mutation reason code is invalid")
        if (self.outcome in {"denied", "failed"}) != (
            self.reason_code is not None
        ):
            raise ValueError("delivery mutation reason does not match outcome")
        if self.action == "admin_webhook.test":
            self._validate_test_shape()
        else:
            self._validate_redelivery_shape()

    def _validate_test_shape(self) -> None:
        if (
            self.source_delivery_id is not None
            or self.source_config_version is not None
            or self.current_config_version is not None
            or self.redelivery_to_changed_config is not None
            or self.outcome == "no_op"
        ):
            raise ValueError("test audit shape is invalid")
        if self.outcome in {"accepted", "succeeded"} and (
            self.delivery_id is None
            or self.attempt_id is None
            or self.target_hostname is None
        ):
            raise ValueError("test audit coordinates are required")

    def _validate_redelivery_shape(self) -> None:
        if self.attempt_id is not None or self.status_code is not None or self.outcome == "succeeded":
            raise ValueError("redelivery audit shape is invalid")
        if self.outcome in {"accepted", "no_op"} and (
            self.source_delivery_id is None
            or self.delivery_id is None
            or self.source_config_version is None
            or self.current_config_version is None
            or self.redelivery_to_changed_config is None
        ):
            raise ValueError("redelivery audit coordinates are required")


DeliveryMutationAuditSink: TypeAlias = Callable[
    [DeliveryMutationAudit], Awaitable[None]
]


@dataclass(frozen=True)
class OperationalAudit:
    """Closed operational record that cannot carry content or artifact paths."""

    operator_id: int
    action: OperationalAction
    operation_id: str
    outcome: OperationalOutcome
    request_id: str
    reason_code: WebhookOperationalReasonCode | None

    def __post_init__(self) -> None:
        _validate_positive_id(self.operator_id, field="operator ID")
        if self.action not in _OPERATIONAL_ACTIONS:
            raise ValueError("operational action is invalid")
        _validate_safe_id(self.operation_id, field="operation ID")
        if self.outcome not in _OPERATIONAL_OUTCOMES:
            raise ValueError("operational outcome is invalid")
        _validate_safe_id(self.request_id, field="request ID")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            WebhookOperationalReasonCode,
        ):
            raise TypeError("operational reason code is invalid")


OperationalAuditSink: TypeAlias = Callable[[OperationalAudit], Awaitable[None]]


async def _create_isolated_audit_service(user_id: int) -> UnifiedAuditService:
    """Create an isolated audit service for one fail-closed write."""
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
        _create_audit_service_for_user,
    )

    return await _create_audit_service_for_user(user_id)


async def _emit_bounded(
    *,
    user_id: int,
    write: Callable[[UnifiedAuditService], Awaitable[None]],
) -> None:
    audit_service: UnifiedAuditService | None = None

    async def write_event() -> None:
        nonlocal audit_service
        audit_service = await _create_isolated_audit_service(user_id)
        await write(audit_service)
        await audit_service.flush(raise_on_failure=True)

    try:
        await asyncio.wait_for(write_event(), timeout=AUDIT_WRITE_TIMEOUT_SECONDS)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        raise MandatoryAuditWriteError(
            "Mandatory audit persistence unavailable"
        ) from exc
    finally:
        if audit_service is not None:
            with suppress(Exception):
                await asyncio.wait_for(
                    audit_service.stop(),
                    timeout=AUDIT_STOP_TIMEOUT_SECONDS,
                )


async def emit_mandatory_webhook_audit(
    record: MutationAudit,
    *,
    actor_principal_id: str,
    actor_kind: str | None,
    actor_roles: tuple[str, ...],
) -> None:
    """Persist one bounded mutation event or fail the caller closed."""
    if not isinstance(record, MutationAudit):
        raise TypeError("mutation audit record is required")
    principal_id = validate_actor_principal_id(actor_principal_id)
    principal_kind = validate_actor_kind(actor_kind)
    roles = validate_actor_roles(actor_roles)
    event_type, category = {
        "admin_webhook.create": (
            AuditEventType.DATA_WRITE,
            AuditEventCategory.DATA_MODIFICATION,
        ),
        "admin_webhook.patch": (
            AuditEventType.DATA_UPDATE,
            AuditEventCategory.DATA_MODIFICATION,
        ),
        "admin_webhook.delete": (
            AuditEventType.DATA_DELETE,
            AuditEventCategory.DATA_MODIFICATION,
        ),
        "admin_webhook.rotate_secret": (
            AuditEventType.DATA_UPDATE,
            AuditEventCategory.SECURITY,
        ),
    }[record.action]

    async def write(service: UnifiedAuditService) -> None:
        await service.log_event(
            event_type=event_type,
            category=category,
            context=AuditContext(user_id=str(record.actor_id)),
            resource_type="admin_webhook",
            resource_id=(str(record.webhook_id) if record.webhook_id is not None else None),
            action=record.action,
            metadata={
                "actor_principal_id": principal_id,
                "actor_kind": principal_kind or "unknown",
                "actor_roles": list(roles),
                "target_hostname": record.target_hostname,
                "event_types": list(record.event_types),
                "outcome": record.outcome,
                "request_id": record.request_id,
                "reason_code": (
                    record.reason_code.value if record.reason_code is not None else None
                ),
            },
        )

    await _emit_bounded(user_id=record.actor_id, write=write)


async def emit_mandatory_webhook_operation_audit(
    record: OperationalAudit,
) -> None:
    """Persist one bounded import/rotation/rollback event or fail closed."""
    if not isinstance(record, OperationalAudit):
        raise TypeError("operational audit record is required")

    async def write(service: UnifiedAuditService) -> None:
        await service.log_event(
            event_type=AuditEventType.CONFIG_CHANGED,
            category=AuditEventCategory.SYSTEM,
            context=AuditContext(user_id=str(record.operator_id)),
            resource_type="admin_webhook_operation",
            resource_id=record.operation_id,
            action=record.action,
            metadata={
                "operator_id": record.operator_id,
                "operation_id": record.operation_id,
                "outcome": record.outcome,
                "request_id": record.request_id,
                "reason_code": (
                    record.reason_code.value if record.reason_code is not None else None
                ),
            },
        )

    await _emit_bounded(user_id=record.operator_id, write=write)


async def emit_mandatory_webhook_delivery_audit(
    record: DeliveryMutationAudit,
) -> None:
    """Persist one bounded test/redelivery event or fail the caller closed."""
    if not isinstance(record, DeliveryMutationAudit):
        raise TypeError("delivery mutation audit record is required")
    event_type = (
        AuditEventType.DATA_WRITE
        if record.action == "admin_webhook.redeliver"
        else AuditEventType.DATA_UPDATE
    )

    async def write(service: UnifiedAuditService) -> None:
        await service.log_event(
            event_type=event_type,
            category=AuditEventCategory.DATA_MODIFICATION,
            context=AuditContext(user_id=str(record.actor_id)),
            resource_type="admin_webhook_delivery",
            resource_id=record.delivery_id or record.source_delivery_id,
            action=record.action,
            metadata={
                "actor_id": record.actor_id,
                "webhook_id": record.webhook_id,
                "source_delivery_id": record.source_delivery_id,
                "delivery_id": record.delivery_id,
                "attempt_id": record.attempt_id,
                "target_hostname": record.target_hostname,
                "source_config_version": record.source_config_version,
                "current_config_version": record.current_config_version,
                "redelivery_to_changed_config": (
                    record.redelivery_to_changed_config
                ),
                "status_code": record.status_code,
                "outcome": record.outcome,
                "request_id": record.request_id,
                "reason_code": (
                    record.reason_code.value
                    if record.reason_code is not None
                    else None
                ),
            },
        )

    await _emit_bounded(user_id=record.actor_id, write=write)
