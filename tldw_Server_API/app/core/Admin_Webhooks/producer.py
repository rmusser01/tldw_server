"""Privacy-bounded production event payload builders."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from .catalog import EVENT_API_VERSION
from .config import AdminWebhookMode, AdminWebhookSettings
from .crypto import (
    WebhookKeyError,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
    load_webhook_key_ring,
)
from .domain import EventSourceKind, WebhookError, WebhookErrorCode
from .events import prepare_event_insert, verify_event_replay

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        AdminWebhookRepository,
        AdminWebhookUnitOfWork,
        EventCaptureResult,
        MigrationState,
    )

_INCIDENT_SEVERITIES = frozenset({"low", "medium", "high", "critical"})
_INCIDENT_STATES = frozenset({"open", "investigating", "mitigating", "resolved"})
_MAX_INCIDENT_NARRATIVE_LENGTH = 4096
_ACTIVE_ROTATION_PHASES = frozenset(
    {"rewriting", "verifying", "awaiting_primary_cutover"}
)


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} is invalid")
    return value


def _bounded_text(value: object, *, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{field_name} is invalid")
    return value


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("event timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _timestamp(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


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


def _public_timestamp(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field_name} is invalid")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError:
        raise ValueError(f"{field_name} is invalid") from None
    if _timestamp(parsed) != value:
        raise ValueError(f"{field_name} is invalid")
    return parsed


def _validate_timestamp_order(
    *,
    created_at: datetime,
    updated_at: datetime,
    resolved_at: datetime | None = None,
) -> None:
    created = _utc(created_at)
    updated = _utc(updated_at)
    if created > updated:
        raise ValueError("event timestamp order is invalid")
    if resolved_at is not None:
        resolved = _utc(resolved_at)
        if resolved < created or resolved > updated:
            raise ValueError("event timestamp order is invalid")


def _user_data(
    *,
    user_id: int,
    status: str,
    resource_version: datetime,
    created_at: datetime,
    updated_at: datetime,
) -> dict[str, object]:
    _positive_int(user_id, field_name="user ID")
    _validate_timestamp_order(created_at=created_at, updated_at=updated_at)
    return {
        "user_id": user_id,
        "status": status,
        "resource_version": _timestamp(resource_version),
        "created_at": _timestamp(created_at),
        "updated_at": _timestamp(updated_at),
    }


def build_user_created_data(
    *,
    user_id: int,
    is_active: bool,
    resource_version: datetime,
    created_at: datetime,
    updated_at: datetime,
) -> dict[str, object]:
    """Build the complete public payload for ``user.created``."""

    if not isinstance(is_active, bool):
        raise ValueError("user status is invalid")
    return _user_data(
        user_id=user_id,
        status="active" if is_active else "inactive",
        resource_version=resource_version,
        created_at=created_at,
        updated_at=updated_at,
    )


def build_user_deleted_data(
    *,
    user_id: int,
    resource_version: datetime,
    created_at: datetime,
    updated_at: datetime,
) -> dict[str, object]:
    """Build the complete public payload for ``user.deleted``."""

    return _user_data(
        user_id=user_id,
        status="inactive",
        resource_version=resource_version,
        created_at=created_at,
        updated_at=updated_at,
    )


def _incident_data(
    *,
    incident_id: str,
    state: str,
    severity: str,
    resource_version: int,
    created_at: datetime,
    updated_at: datetime,
    resolved_at: datetime | None,
) -> dict[str, object]:
    _bounded_text(incident_id, field_name="incident ID", maximum=255)
    if not isinstance(state, str) or state not in _INCIDENT_STATES:
        raise ValueError("state is invalid")
    if not isinstance(severity, str) or severity not in _INCIDENT_SEVERITIES:
        raise ValueError("severity is invalid")
    _positive_int(resource_version, field_name="resource version")
    _validate_timestamp_order(
        created_at=created_at,
        updated_at=updated_at,
        resolved_at=resolved_at,
    )
    if (state == "resolved") != (resolved_at is not None):
        raise ValueError("resolved incident state is invalid")
    return {
        "incident_id": incident_id,
        "state": state,
        "severity": severity,
        "resource_version": resource_version,
        "created_at": _timestamp(created_at),
        "updated_at": _timestamp(updated_at),
        "resolved_at": _timestamp(resolved_at) if resolved_at is not None else None,
    }


def build_incident_created_data(
    *,
    incident_id: str,
    state: str,
    severity: str,
    resource_version: int,
    created_at: datetime,
    updated_at: datetime,
    resolved_at: datetime | None,
) -> dict[str, object]:
    """Build the complete public payload for ``incident.created``."""

    return _incident_data(
        incident_id=incident_id,
        state=state,
        severity=severity,
        resource_version=resource_version,
        created_at=created_at,
        updated_at=updated_at,
        resolved_at=resolved_at,
    )


def build_incident_updated_data(
    *,
    incident_id: str,
    state: str,
    severity: str,
    resource_version: int,
    created_at: datetime,
    updated_at: datetime,
    resolved_at: datetime | None,
) -> dict[str, object]:
    """Build the complete public payload for ``incident.updated``."""

    return _incident_data(
        incident_id=incident_id,
        state=state,
        severity=severity,
        resource_version=resource_version,
        created_at=created_at,
        updated_at=updated_at,
        resolved_at=resolved_at,
    )


def build_incident_resolved_data(
    *,
    incident_id: str,
    state: str,
    severity: str,
    resource_version: int,
    created_at: datetime,
    updated_at: datetime,
    resolved_at: datetime,
) -> dict[str, object]:
    """Build the complete public payload for ``incident.resolved``."""

    if state != "resolved":
        raise ValueError("resolved incident state is invalid")
    return _incident_data(
        incident_id=incident_id,
        state=state,
        severity=severity,
        resource_version=resource_version,
        created_at=created_at,
        updated_at=updated_at,
        resolved_at=resolved_at,
    )


def build_incident_notify_data(
    *,
    incident_id: str,
    state: str,
    severity: str,
    resource_version: int,
    created_at: datetime,
    updated_at: datetime,
    resolved_at: datetime | None,
    narrative: str | None,
) -> dict[str, object]:
    """Build the reviewed public payload for ``incident.notify``."""

    if narrative is not None:
        _bounded_text(
            narrative,
            field_name="narrative",
            maximum=_MAX_INCIDENT_NARRATIVE_LENGTH,
        )
    data = _incident_data(
        incident_id=incident_id,
        state=state,
        severity=severity,
        resource_version=resource_version,
        created_at=created_at,
        updated_at=updated_at,
        resolved_at=resolved_at,
    )
    data["narrative"] = narrative
    return data


def require_writable_event_ring(
    state: MigrationState,
    key_ring_result: WebhookKeyRingLoadResult,
) -> WebhookKeyRing:
    """Return the active event key only when canonical writes are safe."""

    if state.phase != "complete" or state.completed_at is None:
        raise WebhookError(WebhookErrorCode.MIGRATION_PENDING)
    if state.rotation_phase in _ACTIVE_ROTATION_PHASES:
        raise WebhookError(WebhookErrorCode.KEY_ROTATION_IN_PROGRESS)
    ring = key_ring_result.ring
    if ring is None:
        raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)
    if state.active_primary_key_id != ring.primary_id:
        raise WebhookError(WebhookErrorCode.KEY_CONFIGURATION_MISMATCH)
    return ring


@dataclass(frozen=True)
class ProductionEventPreparation:
    """Stable event coordinates generated before a source mutation starts."""

    event_id: str
    created_at: datetime
    source_component: str
    source_request_id: str | None

    def __post_init__(self) -> None:
        _canonical_uuid4(self.event_id, field_name="event ID")
        _utc(self.created_at)
        _bounded_text(
            self.source_component,
            field_name="source component",
            maximum=64,
        )
        if self.source_request_id is not None:
            _bounded_text(
                self.source_request_id,
                field_name="source request ID",
                maximum=128,
            )


def _validated_user_data(
    event_type: str,
    data: Mapping[str, object],
) -> dict[str, object]:
    expected_fields = {
        "user_id",
        "status",
        "resource_version",
        "created_at",
        "updated_at",
    }
    if set(data) != expected_fields:
        raise ValueError("user event data is invalid")
    builder = (
        build_user_created_data
        if event_type == "user.created"
        else build_user_deleted_data
    )
    if data["status"] not in {"active", "inactive"}:
        raise ValueError("user event status is invalid")
    if event_type == "user.deleted" and data["status"] != "inactive":
        raise ValueError("user event status is invalid")
    kwargs = {
        "user_id": data["user_id"],
        "resource_version": _public_timestamp(
            data["resource_version"],
            field_name="resource version",
        ),
        "created_at": _public_timestamp(data["created_at"], field_name="created at"),
        "updated_at": _public_timestamp(data["updated_at"], field_name="updated at"),
    }
    if event_type == "user.created":
        return builder(  # type: ignore[call-arg]
            **kwargs,
            is_active=data["status"] == "active",
        )
    return builder(
        **kwargs,  # type: ignore[arg-type]
    )


def _validated_incident_data(
    event_type: str,
    data: Mapping[str, object],
) -> dict[str, object]:
    expected_fields = {
        "incident_id",
        "state",
        "severity",
        "resource_version",
        "created_at",
        "updated_at",
        "resolved_at",
    }
    if event_type == "incident.notify":
        expected_fields.add("narrative")
    if set(data) != expected_fields:
        raise ValueError("incident event data is invalid")
    resolved_value = data["resolved_at"]
    resolved_at = (
        _public_timestamp(resolved_value, field_name="resolved at")
        if resolved_value is not None
        else None
    )
    common = {
        "incident_id": data["incident_id"],
        "state": data["state"],
        "severity": data["severity"],
        "resource_version": data["resource_version"],
        "created_at": _public_timestamp(data["created_at"], field_name="created at"),
        "updated_at": _public_timestamp(data["updated_at"], field_name="updated at"),
        "resolved_at": resolved_at,
    }
    if event_type == "incident.created":
        return build_incident_created_data(**common)  # type: ignore[arg-type]
    if event_type == "incident.updated":
        return build_incident_updated_data(**common)  # type: ignore[arg-type]
    if event_type == "incident.resolved":
        if resolved_at is None:
            raise ValueError("resolved incident timestamp is invalid")
        return build_incident_resolved_data(**common)  # type: ignore[arg-type]
    return build_incident_notify_data(  # type: ignore[arg-type]
        **common,
        narrative=data["narrative"],
    )


def _validated_production_data(
    event_type: str,
    data: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(data, Mapping):
        raise TypeError("event data must be a mapping")
    if event_type in {"user.created", "user.deleted"}:
        return _validated_user_data(event_type, data)
    if event_type in {
        "incident.created",
        "incident.updated",
        "incident.resolved",
        "incident.notify",
    }:
        return _validated_incident_data(event_type, data)
    raise WebhookError(WebhookErrorCode.EVENT_UNSUPPORTED)


def _validate_source_coordinates(
    *,
    event_type: str,
    source_kind: EventSourceKind,
    aggregate_type: str | None,
    aggregate_id: str | None,
    aggregate_version: str | None,
    source_command_id: str | None,
    data: Mapping[str, object],
) -> None:
    aggregate_event = event_type in {
        "incident.created",
        "incident.updated",
        "incident.resolved",
    }
    if aggregate_event:
        if (
            source_kind is not EventSourceKind.AGGREGATE
            or aggregate_type != "incident"
            or aggregate_id != data["incident_id"]
            or aggregate_version != str(data["resource_version"])
            or source_command_id is not None
        ):
            raise ValueError("aggregate event source is invalid")
        return
    if (
        source_kind is not EventSourceKind.COMMAND
        or any(
            coordinate is not None
            for coordinate in (aggregate_type, aggregate_id, aggregate_version)
        )
        or source_command_id is None
    ):
        raise ValueError("command event source is invalid")


class AdminWebhookEventProducer:
    """Capture reviewed production events inside source-owned transactions."""

    def __init__(
        self,
        *,
        repository: AdminWebhookRepository,
        settings: AdminWebhookSettings,
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
        self._settings = settings
        self._key_ring_result = key_ring_result
        self._event_id_factory = event_id_factory
        self._delivery_id_factory = delivery_id_factory
        self._clock = clock

    def bind_transaction(self, connection: object) -> AdminWebhookUnitOfWork:
        """Bind capture to a source-owned AuthNZ transaction."""

        return self._repository.unit_of_work(connection)

    async def begin_capture(
        self,
        *,
        source_component: str,
        source_request_id: str | None,
    ) -> ProductionEventPreparation | None:
        """Preflight production capture before opening the source mutation."""

        if self._settings.mode is not AdminWebhookMode.ON:
            return None
        state = await self._repository.get_migration_state()
        require_writable_event_ring(state, self._key_ring_result)
        return ProductionEventPreparation(
            event_id=_canonical_uuid4(
                self._event_id_factory(),
                field_name="event ID",
            ),
            created_at=_utc(self._clock()),
            source_component=source_component,
            source_request_id=source_request_id,
        )

    async def capture_in_transaction(
        self,
        preparation: ProductionEventPreparation,
        *,
        tx: AdminWebhookUnitOfWork,
        event_type: str,
        source_kind: EventSourceKind,
        aggregate_type: str | None,
        aggregate_id: str | None,
        aggregate_version: str | None,
        source_command_id: str | None,
        data: Mapping[str, object],
    ) -> EventCaptureResult:
        """Atomically capture one reviewed event through the caller's unit."""

        if not isinstance(preparation, ProductionEventPreparation):
            raise TypeError("production event preparation is invalid")
        if self._settings.mode is not AdminWebhookMode.ON:
            raise WebhookError(WebhookErrorCode.DISABLED)
        try:
            validated_data = _validated_production_data(event_type, data)
            _validate_source_coordinates(
                event_type=event_type,
                source_kind=source_kind,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                aggregate_version=aggregate_version,
                source_command_id=source_command_id,
                data=validated_data,
            )
        except WebhookError:
            raise
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from None

        state = await tx.lock_migration_state()
        ring = require_writable_event_ring(state, self._key_ring_result)
        try:
            prepared = prepare_event_insert(
                ring=ring,
                event_id=preparation.event_id,
                event_type=event_type,
                api_version=EVENT_API_VERSION,
                source_kind=source_kind,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id,
                aggregate_version=aggregate_version,
                source_command_id=source_command_id,
                source_component=preparation.source_component,
                source_request_id=preparation.source_request_id,
                created_at=preparation.created_at,
                data=validated_data,
            )
        except WebhookKeyError:
            raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE) from None
        except (TypeError, ValueError, OverflowError, RecursionError):
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from None
        result = await tx.capture_event_and_expand(
            prepared.event,
            self._delivery_id_factory,
            preparation.created_at
            + timedelta(seconds=self._settings.delivery_expiry_seconds),
        )
        if not result.inserted:
            verify_event_replay(
                ring=ring,
                result=result,
                prepared=prepared,
            )
        return result


def build_admin_webhook_event_producer(
    pool: DatabasePool,
    *,
    environ: Mapping[str, str] | None = None,
) -> AdminWebhookEventProducer:
    """Compose the application production-event boundary for one AuthNZ pool."""

    from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
        AdminWebhookRepository,
    )

    settings = AdminWebhookSettings.from_environment(
        os.environ if environ is None else environ
    )
    return AdminWebhookEventProducer(
        repository=AdminWebhookRepository(pool),
        settings=settings,
        key_ring_result=load_webhook_key_ring(
            os.environ if environ is None else environ
        ),
        event_id_factory=lambda: str(uuid4()),
        delivery_id_factory=lambda: str(uuid4()),
        clock=lambda: datetime.now(timezone.utc),
    )


__all__ = [
    "AdminWebhookEventProducer",
    "ProductionEventPreparation",
    "build_incident_created_data",
    "build_incident_notify_data",
    "build_incident_resolved_data",
    "build_incident_updated_data",
    "build_admin_webhook_event_producer",
    "build_user_created_data",
    "build_user_deleted_data",
    "require_writable_event_ring",
]
