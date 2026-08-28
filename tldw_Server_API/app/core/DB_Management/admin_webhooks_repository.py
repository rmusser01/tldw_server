"""Transactional SQLite/PostgreSQL repository for canonical admin webhooks."""

from __future__ import annotations

import hmac
import json
import re
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import asyncpg

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    EVENT_BODY_MAX_BYTES,
    ProtectedValue,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AttemptState,
    DeliveryHistoryPage,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryRuntimeComponent,
    DeliveryRuntimeHeartbeat,
    DeliveryRuntimeReasonCode,
    DeliveryState,
    EventSourceKind,
    IdempotencyScope,
    JobsDispositionKind,
    WebhookDelivery,
    WebhookDeliveryAttempt,
    WebhookError,
    WebhookEvent,
    WebhookRegistration,
    redact_target,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
)
from tldw_Server_API.app.core.exceptions import TransactionPassthroughError

_MAX_PAGE_SIZE = 500
_MAX_RETENTION_BATCH_SIZE = 200
_MIN_TOMBSTONE_RETENTION_DAYS = 30
_DISPOSITION_TOKEN = re.compile(r"^[0-9a-f]{64}$")
DATABASE_PROTECTED_TABLE_ORDER = (
    "registration_targets",
    "registration_secrets",
    "event_bodies",
    "idempotency_replay_secrets",
)
_DATABASE_PROTECTED_TABLES = frozenset(DATABASE_PROTECTED_TABLE_ORDER)
_ACTIVITY_KINDS = frozenset({"registration_mutation", "event_capture", "delivery_attempt"})
_SAFE_METADATA_KEYS = frozenset(
    {
        "completed_after_config_change",
        "latency_ms",
        "reason_code",
        "redelivery_to_changed_config",
        "result_kind",
        "retry_after_seconds",
        "status_code",
    }
)
_SAFE_METADATA_STRING = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_LOOKUP_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_REQUEST_FINGERPRINT = re.compile(r"^hmac-sha256:[0-9a-f]{64}$")
_MIGRATION_JSON_COLUMNS = {
    "source_mapping_json": dict,
    "source_rejections_json": list,
}
_DELIVERY_SCHEMA_COLUMNS = {
    "admin_webhook_deliveries": frozenset(
        {
            "pending_jobs_disposition_token",
            "pending_jobs_disposition_not_before_at",
        }
    ),
    "admin_webhook_delivery_attempts": frozenset({"request_timeout_seconds"}),
    "admin_webhook_runtime_heartbeats": frozenset(
        {
            "component",
            "instance_id",
            "ready",
            "reason_code",
            "heartbeat_at",
            "last_success_at",
            "created_at",
            "updated_at",
        }
    ),
}
_DELIVERY_SCHEMA_INDEXES = frozenset(
    {
        "idx_admin_webhook_deliveries_recovery",
        "idx_admin_webhook_deliveries_disposition_recovery",
        "idx_admin_webhook_runtime_heartbeats_freshness",
    }
)
_DELIVERY_SCHEMA_COLUMN_CONTRACT = {
    "admin_webhook_deliveries": {
        "pending_jobs_disposition_token": ("TEXT", False, 0),
        "pending_jobs_disposition_not_before_at": ("TEXT", False, 0),
    },
    "admin_webhook_delivery_attempts": {
        "request_timeout_seconds": ("INTEGER", False, 0),
    },
    "admin_webhook_runtime_heartbeats": {
        "component": ("TEXT", True, 1),
        "instance_id": ("TEXT", True, 2),
        "ready": ("INTEGER", True, 0),
        "reason_code": ("TEXT", False, 0),
        "heartbeat_at": ("TEXT", True, 0),
        "last_success_at": ("TEXT", False, 0),
        "created_at": ("TEXT", True, 0),
        "updated_at": ("TEXT", True, 0),
    },
}
_DELIVERY_POSTGRES_COLUMN_CONTRACT = {
    "admin_webhook_deliveries": {
        "pending_jobs_disposition_token": ("text", "YES"),
        "pending_jobs_disposition_not_before_at": ("timestamp with time zone", "YES"),
    },
    "admin_webhook_delivery_attempts": {
        "request_timeout_seconds": ("integer", "YES"),
    },
    "admin_webhook_runtime_heartbeats": {
        "component": ("text", "NO"),
        "instance_id": ("text", "NO"),
        "ready": ("boolean", "NO"),
        "reason_code": ("text", "YES"),
        "heartbeat_at": ("timestamp with time zone", "NO"),
        "last_success_at": ("timestamp with time zone", "YES"),
        "created_at": ("timestamp with time zone", "NO"),
        "updated_at": ("timestamp with time zone", "NO"),
    },
}
_DELIVERY_SCHEMA_INDEX_COLUMNS = {
    "idx_admin_webhook_deliveries_recovery": (
        "state",
        "enqueue_claim_expires_at",
        "expires_at",
        "created_at",
    ),
    "idx_admin_webhook_deliveries_disposition_recovery": (
        "jobs_disposition_applied",
        "pending_jobs_disposition_not_before_at",
        "updated_at",
    ),
    "idx_admin_webhook_runtime_heartbeats_freshness": (
        "component",
        "ready",
        "heartbeat_at",
    ),
}
_DELIVERY_SCHEMA_INDEX_TABLES = {
    "idx_admin_webhook_deliveries_recovery": "admin_webhook_deliveries",
    "idx_admin_webhook_deliveries_disposition_recovery": "admin_webhook_deliveries",
    "idx_admin_webhook_runtime_heartbeats_freshness": (
        "admin_webhook_runtime_heartbeats"
    ),
}
_DELIVERY_POSTGRES_INDEX_DESCENDING = {
    "idx_admin_webhook_deliveries_recovery": (False, False, False, False),
    "idx_admin_webhook_deliveries_disposition_recovery": (False, False, False),
    "idx_admin_webhook_runtime_heartbeats_freshness": (False, False, True),
}
_DELIVERY_RUNTIME_REASON_VALUES = (
    "mode_off",
    "mode_migrate",
    "schema_unready",
    "migration_pending",
    "key_unavailable",
    "key_configuration_mismatch",
    "jobs_unavailable",
    "database_unavailable",
    "worker_unavailable",
    "reconciler_unavailable",
    "retention_unavailable",
    "heartbeat_stale",
)
_DELIVERY_SQLITE_INDEX_DEFINITIONS = {
    "idx_admin_webhook_deliveries_recovery": (
        "createindexidx_admin_webhook_deliveries_recovery"
        "onadmin_webhook_deliveries(state,enqueue_claim_expires_at,expires_at,created_at)"
        "wherestatein('pending','enqueue_claimed')"
    ),
    "idx_admin_webhook_deliveries_disposition_recovery": (
        "createindexidx_admin_webhook_deliveries_disposition_recovery"
        "onadmin_webhook_deliveries(jobs_disposition_applied,"
        "pending_jobs_disposition_not_before_at,updated_at)"
        "wherepending_jobs_dispositionisnotnull"
    ),
    "idx_admin_webhook_runtime_heartbeats_freshness": (
        "createindexidx_admin_webhook_runtime_heartbeats_freshness"
        "onadmin_webhook_runtime_heartbeats(component,ready,heartbeat_atdesc)"
    ),
}


def _compact_schema_sql(value: str) -> str:
    """Normalize backend DDL for fail-closed structural comparisons."""
    return re.sub(r"\s+", "", value.lower()).replace("::text", "")


def _strip_outer_parentheses(value: str) -> str:
    """Remove only balanced outer parentheses from a catalog expression."""
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        for index, character in enumerate(value):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            if depth == 0 and index != len(value) - 1:
                return value
        value = value[1:-1]
    return value


def _has_required_delivery_checks(
    definitions_by_table: Mapping[str, Sequence[str]], *, is_postgres: bool
) -> bool:
    """Return whether the extension's bounded values are enforced by checks."""
    compact_checks = {
        table: " ".join(
            _compact_schema_sql(definition)
            for definition in definitions_by_table.get(table, ())
        )
        for table in _DELIVERY_SCHEMA_COLUMNS
    }
    structural_checks = {
        table: checks.replace("(", "").replace(")", "")
        for table, checks in compact_checks.items()
    }
    delivery_checks = compact_checks["admin_webhook_deliveries"]
    attempt_checks = compact_checks["admin_webhook_delivery_attempts"]
    heartbeat_checks = compact_checks["admin_webhook_runtime_heartbeats"]
    heartbeat_structural_checks = structural_checks[
        "admin_webhook_runtime_heartbeats"
    ]
    token_contract = (
        "pending_jobs_disposition_token~'^[0-9a-f]{64}$'" in delivery_checks
        if is_postgres
        else (
            "length(pending_jobs_disposition_token)=64" in delivery_checks
            and "pending_jobs_disposition_tokennotglob'*[^0-9a-f]*'"
            in delivery_checks
        )
    )
    instance_length_contract = (
        "char_length(instance_id)>=1" in heartbeat_checks
        and "char_length(instance_id)<=128" in heartbeat_checks
        if is_postgres
        else "length(instance_id)between1and128" in heartbeat_checks
    )
    timeout_contract = (
        "request_timeout_seconds>=1" in attempt_checks
        and "request_timeout_seconds<=30" in attempt_checks
        if is_postgres
        else "request_timeout_secondsbetween1and30" in attempt_checks
    )
    ready_reason_contract = (
        "component=anyarray['worker','reconciler','retention']"
        in heartbeat_structural_checks
        and "readyandreason_codeisnull" in heartbeat_structural_checks
        and "notreadyandreason_codeisnotnullandreason_code=anyarray["
        in heartbeat_structural_checks
        if is_postgres
        else (
            "componentin'worker','reconciler','retention'"
            in heartbeat_structural_checks
            and "ready=1andreason_codeisnull" in heartbeat_structural_checks
            and "ready=0andreason_codeisnotnullandreason_codein"
            in heartbeat_structural_checks
        )
    )
    return (
        token_contract
        and timeout_contract
        and instance_length_contract
        and ready_reason_contract
        and "reason_code" in heartbeat_checks
        and all(reason in heartbeat_checks for reason in _DELIVERY_RUNTIME_REASON_VALUES)
    )
_MIGRATION_MUTABLE_COLUMNS = frozenset(
    {
        "phase",
        "import_operation_id",
        "import_operator_id",
        "import_started_at",
        "import_approved_at",
        "artifacts_ready_at",
        "database_committed_at",
        "fingerprint_key_id",
        "active_primary_key_id",
        "system_ops_webhook_fingerprint",
        "legacy_table_fingerprint",
        "source_mapping_json",
        "redacted_report_digest",
        "protected_backup_ciphertext_digest",
        "source_rejections_json",
        "completed_at",
        "active_report_path",
        "active_backup_path",
        "active_key_path",
        "staging_report_path",
        "staging_backup_path",
        "staging_key_path",
        "report_owner_id",
        "report_group_id",
        "report_mode",
        "report_file_identity",
        "backup_owner_id",
        "backup_group_id",
        "backup_mode",
        "backup_file_identity",
        "rollback_key_owner_id",
        "rollback_key_group_id",
        "rollback_key_mode",
        "rollback_key_file_identity",
        "rollback_expires_at",
        "rollback_retirement_phase",
        "rollback_retirement_operator_id",
        "rollback_retirement_started_at",
        "rollback_retirement_completed_at",
        "expected_ciphertext_digest",
        "rotation_operation_id",
        "rotation_source_key_id",
        "rotation_target_key_id",
        "rotation_phase",
        "rotation_table_cursor",
        "rotation_key_cursor",
        "rotation_processed_count",
        "rotation_verified_count",
        "rotation_started_at",
        "rotation_completed_at",
    }
)


class WebhookRepositoryErrorCode(str, Enum):
    """Stable failures owned by the canonical persistence boundary."""

    DATABASE_BUSY = "admin_webhook_database_busy"
    SEQUENCE_UNAVAILABLE = "admin_webhook_sequence_unavailable"
    NOT_FOUND = "admin_webhook_not_found"
    STALE_REVISION = "admin_webhook_stale_revision"
    REGISTRATION_LIMIT = "admin_webhook_registration_limit"
    ACTIVE_LIMIT = "admin_webhook_active_limit"
    IDEMPOTENCY_NOT_CLAIMED = "admin_webhook_idempotency_not_claimed"
    IDEMPOTENCY_COMPLETION_INVALID = "admin_webhook_idempotency_completion_invalid"
    MIGRATION_STATE_UNAVAILABLE = "admin_webhook_migration_state_unavailable"
    STALE_MIGRATION_STATE = "admin_webhook_migration_state_stale"


class WebhookRepositoryError(TransactionPassthroughError):
    """Sanitized repository failure with no database or caller-controlled text."""

    def __init__(self, code: WebhookRepositoryErrorCode | str) -> None:
        self.code = WebhookRepositoryErrorCode(code)
        self.http_status = 503 if self.code is WebhookRepositoryErrorCode.DATABASE_BUSY else None
        super().__init__(self.code.value)


@dataclass(frozen=True)
class RegistrationTarget:
    """Protected destination plus the only redacted searchable metadata."""

    protected: ProtectedValue
    hostname: str
    display: str


@dataclass(frozen=True)
class RegistrationInsert:
    """Complete canonical registration insert with no plaintext secret or URL."""

    id: int
    description: str
    target: RegistrationTarget
    event_types: tuple[str, ...]
    active: bool
    timeout_seconds: int
    secret: ProtectedValue
    secret_rotation_required: bool
    actor_user_id: int
    now: datetime


@dataclass(frozen=True)
class LegacyWebhookRow:
    """One raw legacy database row kept inside the migration boundary."""

    source_identity: str
    values: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, str) or not self.source_identity:
            raise ValueError("legacy source identity is invalid")
        if not isinstance(self.values, Mapping):
            raise TypeError("legacy row values must be a mapping")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True)
class LegacyImportDatabaseSnapshot:
    """Deterministic legacy and canonical allocator state for one import scan."""

    table_present: bool
    rows: tuple[LegacyWebhookRow, ...]
    canonical_registration_ids: tuple[int, ...]
    canonical_non_deleted_count: int
    next_registration_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.table_present, bool):
            raise TypeError("legacy table presence must be boolean")
        if self.next_registration_id < 1:
            raise ValueError("next registration ID must be positive")
        if tuple(sorted(set(self.canonical_registration_ids))) != (
            self.canonical_registration_ids
        ):
            raise ValueError("canonical registration IDs must be unique and sorted")
        if (
            isinstance(self.canonical_non_deleted_count, bool)
            or not isinstance(self.canonical_non_deleted_count, int)
            or not 0 <= self.canonical_non_deleted_count <= len(self.canonical_registration_ids)
        ):
            raise ValueError("canonical non-deleted count is invalid")


class _Unset:
    __slots__ = ()


UNSET = _Unset()


@dataclass(frozen=True)
class RegistrationPatch:
    """Partial registration mutation; omitted fields are distinct from values."""

    description: str | _Unset = UNSET
    target: RegistrationTarget | _Unset = UNSET
    event_types: tuple[str, ...] | _Unset = UNSET
    active: bool | _Unset = UNSET
    timeout_seconds: int | _Unset = UNSET
    secret: ProtectedValue | _Unset = UNSET
    secret_rotation_required: bool | _Unset = UNSET


@dataclass(frozen=True)
class RegistrationPatchResult:
    """Conditional mutation result including explicit effective-no-op state."""

    changed: bool
    registration: WebhookRegistration


@dataclass(frozen=True)
class StoredWebhookRegistration:
    """Internal registration row including protected values for trusted services."""

    registration: WebhookRegistration
    target: ProtectedValue
    secret: ProtectedValue


@dataclass(frozen=True)
class ProtectedRow:
    """One closed protected-value inventory entry with contextual identity."""

    table: str
    row_identity: str
    field: str
    protected: ProtectedValue
    purpose: str
    envelope_identity: Mapping[str, str | int]

    def __post_init__(self) -> None:
        if self.table not in _DATABASE_PROTECTED_TABLES:
            raise ValueError("protected row table is invalid")
        if not isinstance(self.row_identity, str) or not self.row_identity:
            raise ValueError("protected row identity is invalid")
        if not isinstance(self.protected, ProtectedValue):
            raise TypeError("protected row value is invalid")
        expected = {
            "registration_targets": ("target", "registration.target"),
            "registration_secrets": ("secret", "registration.secret"),
            "event_bodies": ("body", "event.body"),
            "idempotency_replay_secrets": (
                "replay_secret",
                "idempotency.secret_replay",
            ),
        }[self.table]
        if (self.field, self.purpose) != expected:
            raise ValueError("protected row field is invalid")
        if not isinstance(self.envelope_identity, Mapping) or not self.envelope_identity:
            raise ValueError("protected row envelope identity is invalid")
        object.__setattr__(
            self,
            "envelope_identity",
            MappingProxyType(dict(self.envelope_identity)),
        )


@dataclass(frozen=True)
class RegistrationLimitState:
    """Read-only admission state that remains valid while already over limit."""

    current: int
    limit: int
    at_limit: bool
    over_limit: bool


class IdempotencyLookupKind(str, Enum):
    """Outcome of one scoped idempotency lookup or claim."""

    NEW = "new"
    REPLAY = "replay"
    CONFLICT = "conflict"
    IN_PROGRESS = "in_progress"


@dataclass(frozen=True)
class IdempotencyLookup:
    """Protected replay state without a raw key or canonical request body."""

    kind: IdempotencyLookupKind
    resource_id: int | None = None
    resource_version: int | None = None
    secret_version: int | None = None
    replay_secret: ProtectedValue | None = None
    test_delivery_id: str | None = None
    test_attempt_id: str | None = None
    response_status: int | None = None
    response_metadata: Mapping[str, object] | None = None
    resource_superseded: bool = False
    expires_at: datetime | None = None


@dataclass(frozen=True)
class MigrationState:
    """Typed durable singleton shared by import, rotation, and status services."""

    singleton_id: int
    schema_version: int
    state_revision: int
    phase: str
    import_operation_id: str | None
    import_operator_id: int | None
    import_started_at: datetime | None
    import_approved_at: datetime | None
    artifacts_ready_at: datetime | None
    database_committed_at: datetime | None
    fingerprint_key_id: str | None
    active_primary_key_id: str | None
    system_ops_webhook_fingerprint: str | None
    legacy_table_fingerprint: str | None
    source_mapping: Mapping[str, object]
    redacted_report_digest: str | None
    protected_backup_ciphertext_digest: str | None
    source_rejections: tuple[object, ...]
    completed_at: datetime | None
    active_report_path: str | None
    active_backup_path: str | None
    active_key_path: str | None
    staging_report_path: str | None
    staging_backup_path: str | None
    staging_key_path: str | None
    report_owner_id: int | None
    report_group_id: int | None
    report_mode: int | None
    report_file_identity: str | None
    backup_owner_id: int | None
    backup_group_id: int | None
    backup_mode: int | None
    backup_file_identity: str | None
    rollback_key_owner_id: int | None
    rollback_key_group_id: int | None
    rollback_key_mode: int | None
    rollback_key_file_identity: str | None
    rollback_expires_at: datetime | None
    rollback_retirement_phase: str
    rollback_retirement_operator_id: int | None
    rollback_retirement_started_at: datetime | None
    rollback_retirement_completed_at: datetime | None
    expected_ciphertext_digest: str | None
    first_canonical_activity_at: datetime | None
    first_canonical_activity_kind: str | None
    rotation_operation_id: str | None
    rotation_source_key_id: str | None
    rotation_target_key_id: str | None
    rotation_phase: str | None
    rotation_table_cursor: str | None
    rotation_key_cursor: str | None
    rotation_processed_count: int
    rotation_verified_count: int
    rotation_started_at: datetime | None
    rotation_completed_at: datetime | None
    updated_at: datetime


def _bounded_text(
    value: object,
    *,
    field: str,
    maximum: int,
    optional: bool = False,
) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{field} is invalid")
    return value


@dataclass(frozen=True)
class EventInsert:
    """Immutable protected event write supplied by the delivery service."""

    id: str
    event_type: str
    api_version: str
    source_kind: EventSourceKind
    aggregate_type: str | None
    aggregate_id: str | None
    aggregate_version: str | None
    source_command_id: str | None
    source_component: str
    source_request_id: str | None
    body: ProtectedValue
    body_size_bytes: int
    created_at: datetime

    def __post_init__(self) -> None:
        _bounded_text(self.id, field="event ID", maximum=128)
        _bounded_text(self.event_type, field="event type", maximum=64)
        _bounded_text(self.api_version, field="API version", maximum=32)
        _bounded_text(self.source_component, field="source component", maximum=64)
        _bounded_text(
            self.source_request_id,
            field="source request ID",
            maximum=128,
            optional=True,
        )
        if not isinstance(self.source_kind, EventSourceKind):
            raise ValueError("event source kind is invalid")
        if not isinstance(self.body, ProtectedValue):
            raise TypeError("event body must be protected")
        if (
            isinstance(self.body_size_bytes, bool)
            or not isinstance(self.body_size_bytes, int)
            or not 0 <= self.body_size_bytes <= EVENT_BODY_MAX_BYTES
        ):
            raise ValueError("event body size is invalid")
        aggregate = (
            self.aggregate_type,
            self.aggregate_id,
            self.aggregate_version,
        )
        if self.source_kind is EventSourceKind.AGGREGATE:
            if self.source_command_id is not None or any(value is None for value in aggregate):
                raise ValueError("event source identity is invalid")
            _bounded_text(self.aggregate_type, field="aggregate type", maximum=64)
            _bounded_text(self.aggregate_id, field="aggregate ID", maximum=255)
            _bounded_text(self.aggregate_version, field="aggregate version", maximum=255)
        elif any(value is not None for value in aggregate) or self.source_command_id is None:
            raise ValueError("event source identity is invalid")
        else:
            _bounded_text(self.source_command_id, field="source command ID", maximum=255)
        object.__setattr__(
            self,
            "created_at",
            _utc_datetime(self.created_at, field="created_at"),
        )


@dataclass(frozen=True)
class StoredWebhookEvent:
    """Protected event readback with exact source identity."""

    event: WebhookEvent
    aggregate_type: str | None
    aggregate_id: str | None
    aggregate_version: str | None
    source_command_id: str | None
    source_component: str
    source_request_id: str | None
    body: ProtectedValue
    body_size_bytes: int

    @property
    def id(self) -> str:
        return self.event.id


@dataclass(frozen=True)
class StoredWebhookDelivery:
    """Internal delivery state including opaque recovery coordinates."""

    delivery: WebhookDelivery
    jobs_job_id: str | None
    enqueue_claim_token: str | None
    enqueue_claim_expires_at: datetime | None
    current_attempt_id: str | None
    pending_jobs_disposition: JobsDispositionKind | None
    pending_jobs_disposition_delay_seconds: int | None
    pending_jobs_disposition_token: str | None
    pending_jobs_disposition_not_before_at: datetime | None
    jobs_disposition_applied: bool
    completed_after_config_change: bool


@dataclass(frozen=True)
class DeliveryBundle:
    """Trusted execution readback containing protected registration material."""

    event: StoredWebhookEvent
    delivery: StoredWebhookDelivery
    registration: StoredWebhookRegistration


@dataclass(frozen=True)
class EventCaptureResult:
    """Atomic event-capture result, including idempotent replay evidence."""

    event: StoredWebhookEvent
    deliveries: tuple[StoredWebhookDelivery, ...]
    inserted: bool


@dataclass(frozen=True)
class EnqueueClaim:
    """One pending delivery claimed for external Jobs admission."""

    delivery: StoredWebhookDelivery
    claim_token: str
    claimed_until: datetime


@dataclass(frozen=True)
class AttemptReservation:
    """Guarded attempt reservation or closed hard-budget rejection."""

    reserved: bool
    delivery: StoredWebhookDelivery
    attempt: WebhookDeliveryAttempt | None
    bundle: DeliveryBundle | None
    reason_code: DeliveryReasonCode | None = None

    def __post_init__(self) -> None:
        if self.reserved != (self.attempt is not None and self.bundle is not None):
            raise ValueError("attempt reservation shape is invalid")
        if self.reserved == (self.reason_code is not None):
            raise ValueError("attempt reservation reason is invalid")


@dataclass(frozen=True)
class AttemptCompletion:
    """Bounded service-owned outcome used to close one exact attempt."""

    attempt_state: AttemptState
    delivery_state: DeliveryState
    disposition: JobsDispositionKind | None
    status_code: int | None
    latency_ms: int | None
    reason_code: DeliveryReasonCode | None
    requested_retry_delay_seconds: int | None
    finished_at: datetime
    completed_after_config_change: bool

    def __post_init__(self) -> None:
        if self.attempt_state not in AttemptState.terminal_states():
            raise ValueError("attempt completion state is invalid")
        if self.delivery_state not in {
            DeliveryState.RETRY_WAIT,
            *DeliveryState.terminal_states(),
        }:
            raise ValueError("delivery completion state is invalid")
        if self.status_code is not None and not 100 <= self.status_code <= 599:
            raise ValueError("attempt status code is invalid")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("attempt latency is invalid")
        retry = self.attempt_state is AttemptState.RETRYABLE
        if retry != (self.requested_retry_delay_seconds is not None):
            raise ValueError("attempt retry delay is invalid")
        if self.requested_retry_delay_seconds is not None and not (
            1 <= self.requested_retry_delay_seconds <= 1_800
        ):
            raise ValueError("attempt retry delay is invalid")
        expected_dispositions = {
            (AttemptState.SUCCEEDED, DeliveryState.SUCCEEDED): JobsDispositionKind.COMPLETE,
            (AttemptState.RETRYABLE, DeliveryState.RETRY_WAIT): JobsDispositionKind.RETRY,
            (AttemptState.FAILED, DeliveryState.DEAD): JobsDispositionKind.FAIL,
            (AttemptState.CANCELED, DeliveryState.CANCELED): JobsDispositionKind.CANCEL,
            (AttemptState.SUPERSEDED, DeliveryState.SUPERSEDED): JobsDispositionKind.CANCEL,
            (
                AttemptState.OUTCOME_UNKNOWN,
                DeliveryState.RETRY_WAIT,
            ): JobsDispositionKind.DEFER,
        }
        expected_disposition = expected_dispositions.get(
            (self.attempt_state, self.delivery_state)
        )
        if expected_disposition is None or (
            self.disposition is not None and self.disposition is not expected_disposition
        ):
            raise ValueError("attempt and delivery outcomes are inconsistent")
        object.__setattr__(
            self,
            "finished_at",
            _utc_datetime(self.finished_at, field="finished_at"),
        )


@dataclass(frozen=True)
class PendingJobsDisposition:
    """Exact durable Jobs transition awaiting acknowledgement."""

    delivery_id: str
    jobs_job_id: str
    attempt_id: str | None
    kind: JobsDispositionKind
    delay_seconds: int | None
    token: str
    not_before_at: datetime

    def __post_init__(self) -> None:
        _bounded_text(self.delivery_id, field="delivery ID", maximum=128)
        _bounded_text(self.jobs_job_id, field="Jobs job ID", maximum=255)
        _bounded_text(self.attempt_id, field="attempt ID", maximum=128, optional=True)
        if _DISPOSITION_TOKEN.fullmatch(self.token) is None:
            raise ValueError("disposition token is invalid")
        if self.kind is JobsDispositionKind.RETRY:
            if self.delay_seconds is None or not 1 <= self.delay_seconds <= 1_800:
                raise ValueError("retry disposition requires a bounded delay")
        elif self.delay_seconds is not None:
            raise ValueError("non-retry disposition cannot carry a delay")
        object.__setattr__(
            self,
            "not_before_at",
            _utc_datetime(self.not_before_at, field="not_before_at"),
        )


@dataclass(frozen=True)
class RuntimeHeartbeatWrite:
    """Bounded runtime heartbeat upsert supplied by one component instance."""

    component: DeliveryRuntimeComponent
    instance_id: str
    ready: bool
    reason_code: DeliveryRuntimeReasonCode | None
    heartbeat_at: datetime
    last_success_at: datetime | None

    def __post_init__(self) -> None:
        _bounded_text(self.instance_id, field="runtime instance ID", maximum=128)
        if not isinstance(self.component, DeliveryRuntimeComponent):
            raise ValueError("runtime component is invalid")
        if not isinstance(self.ready, bool):
            raise ValueError("runtime readiness is invalid")
        if self.ready and self.reason_code is not None:
            raise ValueError("ready heartbeat cannot carry a reason")
        if not self.ready and not isinstance(
            self.reason_code, DeliveryRuntimeReasonCode
        ):
            raise ValueError("unready heartbeat requires a runtime reason")
        object.__setattr__(
            self,
            "heartbeat_at",
            _utc_datetime(self.heartbeat_at, field="heartbeat_at"),
        )
        if self.last_success_at is not None:
            object.__setattr__(
                self,
                "last_success_at",
                _utc_datetime(self.last_success_at, field="last_success_at"),
            )


@dataclass(frozen=True)
class RetentionBatchResult:
    """Per-table deletion counts from one bounded retention transaction."""

    expired_idempotency: int = 0
    deliveries: int = 0
    events: int = 0
    heartbeats: int = 0
    registrations: int = 0


def _utc_datetime(value: datetime, *, field: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        parsed = datetime.fromisoformat(text)
    else:
        raise ValueError("invalid persisted datetime")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _validate_registration_target(target: RegistrationTarget) -> None:
    try:
        canonical_display = redact_target(target.display)
        parsed_display = urlsplit(target.display)
        display_hostname = parsed_display.hostname
    except (TypeError, ValueError, WebhookError) as exc:
        raise ValueError("registration target metadata is invalid") from exc
    if (
        canonical_display != target.display
        or parsed_display.scheme not in {"https", "http"}
        or display_hostname is None
        or display_hostname.rstrip(".").lower() != target.hostname.rstrip(".").lower()
    ):
        raise ValueError("registration target metadata is not a redacted origin")


def _strict_json_object(value: object) -> dict[str, object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("persisted JSON must be an object")
    return parsed


def _strict_json_array(value: object) -> list[object]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError("persisted JSON must be an array")
    return parsed


def _row_dict(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return row
    keys = getattr(row, "keys", None)
    if callable(keys):
        return {str(key): row[key] for key in keys()}
    return dict(row)


def _question_marks_to_postgres(query: str, count: int) -> str:
    if query.count("?") != count:
        raise ValueError("repository SQL placeholder mismatch")
    parts = query.split("?")
    return "".join(part + (f"${index + 1}" if index < count else "") for index, part in enumerate(parts))


def _registration_from_row(row: Mapping[str, Any]) -> WebhookRegistration:
    raw_events = _strict_json_array(row["event_types_json"])
    if any(not isinstance(item, str) for item in raw_events):
        raise ValueError("persisted event set is invalid")
    created_at = _parse_datetime(row["created_at"])
    updated_at = _parse_datetime(row["updated_at"])
    if created_at is None or updated_at is None:
        raise ValueError("persisted registration timestamps are invalid")
    return WebhookRegistration(
        id=int(row["id"]),
        description=str(row["description"]),
        target_display=str(row["target_display"]),
        target_hostname=str(row["target_hostname"]),
        event_types=tuple(str(item) for item in raw_events),
        active=bool(row["active"]),
        timeout_seconds=int(row["timeout_seconds"]),
        revision=int(row["revision"]),
        delivery_config_version=int(row["delivery_config_version"]),
        target_version=int(row["target_version"]),
        secret_version=int(row["secret_version"]),
        secret_rotation_required=bool(row["secret_rotation_required"]),
        created_by_user_id=int(row["created_by_user_id"]),
        updated_by_user_id=int(row["updated_by_user_id"]),
        created_at=created_at,
        updated_at=updated_at,
        deleted_at=_parse_datetime(row["deleted_at"]),
        deleted_by_user_id=(int(row["deleted_by_user_id"]) if row["deleted_by_user_id"] is not None else None),
    )


def _stored_registration_from_row(
    row: Mapping[str, Any],
) -> StoredWebhookRegistration:
    return StoredWebhookRegistration(
        registration=_registration_from_row(row),
        target=ProtectedValue(
            ciphertext_json=str(row["target_ciphertext_json"]),
            key_id=str(row["target_key_id"]),
        ),
        secret=ProtectedValue(
            ciphertext_json=str(row["secret_ciphertext_json"]),
            key_id=str(row["secret_key_id"]),
        ),
    )


_EVENT_COLUMNS = """
    id, event_type, api_version, source_kind, aggregate_type, aggregate_id,
    aggregate_version, source_command_id, source_component, source_request_id,
    body_ciphertext_json, body_key_id, body_size_bytes, created_at
"""
_DELIVERY_COLUMNS = """
    id, event_id, webhook_id, kind, delivery_config_version, secret_version,
    jobs_job_id, enqueue_claim_token, enqueue_claim_expires_at, state,
    attempt_count, current_attempt_id, status_code, latency_ms, reason_code,
    pending_jobs_disposition, pending_jobs_disposition_delay_seconds,
    pending_jobs_disposition_token, pending_jobs_disposition_not_before_at,
    jobs_disposition_applied, completed_after_config_change, terminal_at,
    expires_at, redelivery_of_id, created_at, updated_at
"""
_ATTEMPT_COLUMNS = """
    id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
    test_attempt_token, request_timeout_seconds, started_at, finished_at, state,
    status_code, latency_ms, reason_code, requested_retry_delay_seconds,
    jobs_disposition_applied, created_at
"""


def _stored_event_from_row(row: Mapping[str, Any]) -> StoredWebhookEvent:
    created_at = _parse_datetime(row["created_at"])
    if created_at is None:
        raise ValueError("persisted event timestamp is invalid")
    try:
        source_kind = EventSourceKind(str(row["source_kind"]))
    except ValueError as exc:
        raise ValueError("persisted event source kind is invalid") from exc
    aggregate_type = str(row["aggregate_type"]) if row["aggregate_type"] is not None else None
    aggregate_id = str(row["aggregate_id"]) if row["aggregate_id"] is not None else None
    aggregate_version = (
        str(row["aggregate_version"]) if row["aggregate_version"] is not None else None
    )
    source_command_id = (
        str(row["source_command_id"]) if row["source_command_id"] is not None else None
    )
    if source_kind is EventSourceKind.AGGREGATE:
        valid_source = (
            source_command_id is None
            and aggregate_type is not None
            and aggregate_id is not None
            and aggregate_version is not None
        )
    else:
        valid_source = (
            source_command_id is not None
            and aggregate_type is None
            and aggregate_id is None
            and aggregate_version is None
        )
    if not valid_source:
        raise ValueError("persisted event source identity is invalid")
    size = int(row["body_size_bytes"])
    if not 0 <= size <= EVENT_BODY_MAX_BYTES:
        raise ValueError("persisted event body size is invalid")
    return StoredWebhookEvent(
        event=WebhookEvent(
            id=str(row["id"]),
            event_type=str(row["event_type"]),
            api_version=str(row["api_version"]),
            source_kind=source_kind,
            created_at=created_at,
        ),
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        aggregate_version=aggregate_version,
        source_command_id=source_command_id,
        source_component=str(row["source_component"]),
        source_request_id=(
            str(row["source_request_id"])
            if row["source_request_id"] is not None
            else None
        ),
        body=ProtectedValue(
            ciphertext_json=str(row["body_ciphertext_json"]),
            key_id=str(row["body_key_id"]),
        ),
        body_size_bytes=size,
    )


def _stored_delivery_from_row(row: Mapping[str, Any]) -> StoredWebhookDelivery:
    required_times = {
        name: _parse_datetime(row[name])
        for name in ("expires_at", "created_at", "updated_at")
    }
    if any(value is None for value in required_times.values()):
        raise ValueError("persisted delivery timestamp is invalid")
    try:
        kind = DeliveryKind(str(row["kind"]))
        state = DeliveryState(str(row["state"]))
        reason = (
            DeliveryReasonCode(str(row["reason_code"]))
            if row["reason_code"] is not None
            else None
        )
        disposition = (
            JobsDispositionKind(str(row["pending_jobs_disposition"]))
            if row["pending_jobs_disposition"] is not None
            else None
        )
    except ValueError as exc:
        raise ValueError("persisted delivery enum is invalid") from exc
    token = (
        str(row["pending_jobs_disposition_token"])
        if row["pending_jobs_disposition_token"] is not None
        else None
    )
    not_before = _parse_datetime(row["pending_jobs_disposition_not_before_at"])
    delay = (
        int(row["pending_jobs_disposition_delay_seconds"])
        if row["pending_jobs_disposition_delay_seconds"] is not None
        else None
    )
    if disposition is None:
        if token is not None or not_before is not None or delay is not None:
            raise ValueError("persisted pending disposition shape is invalid")
    else:
        if token is None or _DISPOSITION_TOKEN.fullmatch(token) is None or not_before is None:
            raise ValueError("persisted pending disposition shape is invalid")
        if (disposition is JobsDispositionKind.RETRY) != (delay is not None):
            raise ValueError("persisted pending disposition delay is invalid")
    terminal_at = _parse_datetime(row["terminal_at"])
    if (state in DeliveryState.terminal_states()) != (terminal_at is not None):
        raise ValueError("persisted delivery terminal state is invalid")
    delivery = WebhookDelivery(
        id=str(row["id"]),
        event_id=str(row["event_id"]),
        webhook_id=int(row["webhook_id"]),
        kind=kind,
        state=state,
        delivery_config_version=int(row["delivery_config_version"]),
        secret_version=int(row["secret_version"]),
        attempt_count=int(row["attempt_count"]),
        status_code=int(row["status_code"]) if row["status_code"] is not None else None,
        latency_ms=int(row["latency_ms"]) if row["latency_ms"] is not None else None,
        reason_code=reason,
        expires_at=required_times["expires_at"],  # type: ignore[arg-type]
        created_at=required_times["created_at"],  # type: ignore[arg-type]
        updated_at=required_times["updated_at"],  # type: ignore[arg-type]
        terminal_at=terminal_at,
        redelivery_of_id=(
            str(row["redelivery_of_id"])
            if row["redelivery_of_id"] is not None
            else None
        ),
    )
    return StoredWebhookDelivery(
        delivery=delivery,
        jobs_job_id=str(row["jobs_job_id"]) if row["jobs_job_id"] is not None else None,
        enqueue_claim_token=(
            str(row["enqueue_claim_token"])
            if row["enqueue_claim_token"] is not None
            else None
        ),
        enqueue_claim_expires_at=_parse_datetime(row["enqueue_claim_expires_at"]),
        current_attempt_id=(
            str(row["current_attempt_id"])
            if row["current_attempt_id"] is not None
            else None
        ),
        pending_jobs_disposition=disposition,
        pending_jobs_disposition_delay_seconds=delay,
        pending_jobs_disposition_token=token,
        pending_jobs_disposition_not_before_at=not_before,
        jobs_disposition_applied=bool(row["jobs_disposition_applied"]),
        completed_after_config_change=bool(row["completed_after_config_change"]),
    )


def _attempt_from_row(row: Mapping[str, Any]) -> WebhookDeliveryAttempt:
    started_at = _parse_datetime(row["started_at"])
    finished_at = _parse_datetime(row["finished_at"])
    if started_at is None:
        raise ValueError("persisted attempt timestamp is invalid")
    try:
        state = AttemptState(str(row["state"]))
        reason = (
            DeliveryReasonCode(str(row["reason_code"]))
            if row["reason_code"] is not None
            else None
        )
    except ValueError as exc:
        raise ValueError("persisted attempt enum is invalid") from exc
    if (state is AttemptState.PROCESSING) != (finished_at is None):
        raise ValueError("persisted attempt terminal state is invalid")
    return WebhookDeliveryAttempt(
        id=str(row["id"]),
        delivery_id=str(row["delivery_id"]),
        attempt_number=int(row["attempt_number"]),
        state=state,
        request_timeout_seconds=(
            int(row["request_timeout_seconds"])
            if row["request_timeout_seconds"] is not None
            else None
        ),
        status_code=int(row["status_code"]) if row["status_code"] is not None else None,
        latency_ms=int(row["latency_ms"]) if row["latency_ms"] is not None else None,
        reason_code=reason,
        requested_retry_delay_seconds=(
            int(row["requested_retry_delay_seconds"])
            if row["requested_retry_delay_seconds"] is not None
            else None
        ),
        started_at=started_at,
        finished_at=finished_at,
    )


def _heartbeat_from_row(row: Mapping[str, Any]) -> DeliveryRuntimeHeartbeat:
    heartbeat_at = _parse_datetime(row["heartbeat_at"])
    created_at = _parse_datetime(row["created_at"])
    updated_at = _parse_datetime(row["updated_at"])
    if heartbeat_at is None or created_at is None or updated_at is None:
        raise ValueError("persisted runtime heartbeat timestamp is invalid")
    try:
        component = DeliveryRuntimeComponent(str(row["component"]))
        reason = (
            DeliveryRuntimeReasonCode(str(row["reason_code"]))
            if row["reason_code"] is not None
            else None
        )
    except ValueError as exc:
        raise ValueError("persisted runtime heartbeat enum is invalid") from exc
    ready = bool(row["ready"])
    if ready == (reason is not None):
        raise ValueError("persisted runtime heartbeat readiness is invalid")
    return DeliveryRuntimeHeartbeat(
        component=component,
        instance_id=str(row["instance_id"]),
        ready=ready,
        reason_code=reason,
        heartbeat_at=heartbeat_at,
        last_success_at=_parse_datetime(row["last_success_at"]),
        created_at=created_at,
        updated_at=updated_at,
    )


def _migration_state_from_row(row: Mapping[str, Any]) -> MigrationState:
    source_mapping = _strict_json_object(row["source_mapping_json"])
    source_rejections = _strict_json_array(row["source_rejections_json"])
    updated_at = _parse_datetime(row["updated_at"])
    if updated_at is None:
        raise ValueError("persisted migration timestamp is invalid")

    def optional_int(name: str) -> int | None:
        value = row[name]
        return int(value) if value is not None else None

    def optional_text(name: str) -> str | None:
        value = row[name]
        return str(value) if value is not None else None

    return MigrationState(
        singleton_id=int(row["singleton_id"]),
        schema_version=int(row["schema_version"]),
        state_revision=int(row["state_revision"]),
        phase=str(row["phase"]),
        import_operation_id=optional_text("import_operation_id"),
        import_operator_id=optional_int("import_operator_id"),
        import_started_at=_parse_datetime(row["import_started_at"]),
        import_approved_at=_parse_datetime(row["import_approved_at"]),
        artifacts_ready_at=_parse_datetime(row["artifacts_ready_at"]),
        database_committed_at=_parse_datetime(row["database_committed_at"]),
        fingerprint_key_id=optional_text("fingerprint_key_id"),
        active_primary_key_id=optional_text("active_primary_key_id"),
        system_ops_webhook_fingerprint=optional_text("system_ops_webhook_fingerprint"),
        legacy_table_fingerprint=optional_text("legacy_table_fingerprint"),
        source_mapping=MappingProxyType(source_mapping),
        redacted_report_digest=optional_text("redacted_report_digest"),
        protected_backup_ciphertext_digest=optional_text("protected_backup_ciphertext_digest"),
        source_rejections=tuple(source_rejections),
        completed_at=_parse_datetime(row["completed_at"]),
        active_report_path=optional_text("active_report_path"),
        active_backup_path=optional_text("active_backup_path"),
        active_key_path=optional_text("active_key_path"),
        staging_report_path=optional_text("staging_report_path"),
        staging_backup_path=optional_text("staging_backup_path"),
        staging_key_path=optional_text("staging_key_path"),
        report_owner_id=optional_int("report_owner_id"),
        report_group_id=optional_int("report_group_id"),
        report_mode=optional_int("report_mode"),
        report_file_identity=optional_text("report_file_identity"),
        backup_owner_id=optional_int("backup_owner_id"),
        backup_group_id=optional_int("backup_group_id"),
        backup_mode=optional_int("backup_mode"),
        backup_file_identity=optional_text("backup_file_identity"),
        rollback_key_owner_id=optional_int("rollback_key_owner_id"),
        rollback_key_group_id=optional_int("rollback_key_group_id"),
        rollback_key_mode=optional_int("rollback_key_mode"),
        rollback_key_file_identity=optional_text("rollback_key_file_identity"),
        rollback_expires_at=_parse_datetime(row["rollback_expires_at"]),
        rollback_retirement_phase=str(row["rollback_retirement_phase"]),
        rollback_retirement_operator_id=optional_int("rollback_retirement_operator_id"),
        rollback_retirement_started_at=_parse_datetime(row["rollback_retirement_started_at"]),
        rollback_retirement_completed_at=_parse_datetime(row["rollback_retirement_completed_at"]),
        expected_ciphertext_digest=optional_text("expected_ciphertext_digest"),
        first_canonical_activity_at=_parse_datetime(row["first_canonical_activity_at"]),
        first_canonical_activity_kind=optional_text("first_canonical_activity_kind"),
        rotation_operation_id=optional_text("rotation_operation_id"),
        rotation_source_key_id=optional_text("rotation_source_key_id"),
        rotation_target_key_id=optional_text("rotation_target_key_id"),
        rotation_phase=optional_text("rotation_phase"),
        rotation_table_cursor=optional_text("rotation_table_cursor"),
        rotation_key_cursor=optional_text("rotation_key_cursor"),
        rotation_processed_count=int(row["rotation_processed_count"]),
        rotation_verified_count=int(row["rotation_verified_count"]),
        rotation_started_at=_parse_datetime(row["rotation_started_at"]),
        rotation_completed_at=_parse_datetime(row["rotation_completed_at"]),
        updated_at=updated_at,
    )


def _safe_response_metadata(
    value: Mapping[str, object] | None,
) -> tuple[str | None, Mapping[str, object] | None]:
    if value is None:
        return None, None
    normalized: dict[str, object] = {}
    for key, item in value.items():
        if key not in _SAFE_METADATA_KEYS:
            raise ValueError("response metadata key is not allowed")
        if isinstance(item, str):
            if _SAFE_METADATA_STRING.fullmatch(item) is None:
                raise ValueError("response metadata string is invalid")
        elif isinstance(item, bool) or item is None:
            pass
        elif isinstance(item, int):
            if not -(2**31) <= item <= 2**31 - 1:
                raise ValueError("response metadata integer is out of range")
        else:
            raise ValueError("response metadata value is invalid")
        normalized[key] = item
    encoded = _canonical_json(normalized)
    if len(encoded.encode("utf-8")) > 16_384:
        raise ValueError("response metadata is too large")
    return encoded, MappingProxyType(normalized)


def _referenced_registration_ids(value: object) -> set[int]:
    referenced: set[int] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, bool):
            continue
        if isinstance(current, int) and current > 0:
            referenced.add(current)
        elif isinstance(current, Mapping):
            pending.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current,
            (str, bytes, bytearray),
        ):
            pending.extend(current)
    return referenced


def _is_database_busy(exc: BaseException) -> bool:
    current: BaseException | None = exc
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(
            current,
            (
                ConnectionPoolExhaustedError,
                DatabaseLockError,
                asyncpg.LockNotAvailableError,
                asyncpg.QueryCanceledError,
                asyncpg.DeadlockDetectedError,
            ),
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


class _ConnectionAdapter:
    def __init__(self, connection: object, *, is_postgres: bool) -> None:
        self._connection = connection
        self._is_postgres = is_postgres

    def _query_and_params(
        self,
        query: str,
        params: Sequence[object],
    ) -> tuple[str, tuple[object, ...]]:
        normalized = tuple(params)
        if self._is_postgres:
            return _question_marks_to_postgres(query, len(normalized)), normalized
        sqlite_params = tuple(
            item.astimezone(timezone.utc).isoformat(sep=" ")
            if isinstance(item, datetime)
            else int(item)
            if isinstance(item, bool)
            else item
            for item in normalized
        )
        return query, sqlite_params

    async def _fetchrow(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> dict[str, Any] | None:
        normalized_query, normalized_params = self._query_and_params(query, params)
        if self._is_postgres:
            row = await self._connection.fetchrow(  # type: ignore[attr-defined]
                normalized_query,
                *normalized_params,
            )
        else:
            cursor = await self._connection.execute(  # type: ignore[attr-defined]
                normalized_query,
                normalized_params,
            )
            row = await cursor.fetchone()
        return _row_dict(row) if row is not None else None

    async def _fetch(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> list[dict[str, Any]]:
        normalized_query, normalized_params = self._query_and_params(query, params)
        if self._is_postgres:
            rows = await self._connection.fetch(  # type: ignore[attr-defined]
                normalized_query,
                *normalized_params,
            )
        else:
            cursor = await self._connection.execute(  # type: ignore[attr-defined]
                normalized_query,
                normalized_params,
            )
            rows = await cursor.fetchall()
        return [_row_dict(row) for row in rows]

    async def _execute(
        self,
        query: str,
        params: Sequence[object] = (),
    ) -> int:
        normalized_query, normalized_params = self._query_and_params(query, params)
        if self._is_postgres:
            status = await self._connection.execute(  # type: ignore[attr-defined]
                normalized_query,
                *normalized_params,
            )
            tail = str(status).rsplit(" ", 1)[-1]
            return int(tail) if tail.isdigit() else 0
        cursor = await self._connection.execute(  # type: ignore[attr-defined]
            normalized_query,
            normalized_params,
        )
        return max(int(cursor.rowcount), 0)

    async def _executemany(
        self,
        query: str,
        rows: Sequence[Sequence[object]],
    ) -> int:
        normalized_rows = tuple(tuple(row) for row in rows)
        if not normalized_rows:
            return 0
        normalized_query, _ = self._query_and_params(query, normalized_rows[0])
        converted_rows = tuple(
            self._query_and_params(query, row)[1] for row in normalized_rows
        )
        await self._connection.executemany(  # type: ignore[attr-defined]
            normalized_query,
            converted_rows,
        )
        return len(converted_rows)


class AdminWebhookRepository:
    """Application-scoped repository that owns all canonical webhook SQL."""

    def __init__(
        self,
        pool: DatabasePool,
        *,
        postgres_lock_timeout_ms: int = 5_000,
        postgres_statement_timeout_ms: int = 15_000,
    ) -> None:
        if postgres_lock_timeout_ms < 1 or postgres_statement_timeout_ms < 1:
            raise ValueError("database timeouts must be positive")
        self._pool = pool
        self._postgres_lock_timeout_ms = postgres_lock_timeout_ms
        self._postgres_statement_timeout_ms = postgres_statement_timeout_ms

    @property
    def database_path(self) -> Path | None:
        """Return the SQLite filesystem path without inventing one for PostgreSQL."""
        value = getattr(self._pool, "_sqlite_fs_path", None)
        return Path(value) if value and value != ":memory:" else None

    @property
    def is_postgres(self) -> bool:
        return self._pool.pool is not None

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AdminWebhookUnitOfWork]:
        """Open one atomic unit and map only bounded contention to a stable error."""
        try:
            async with self._pool.transaction() as connection:
                unit = AdminWebhookUnitOfWork(
                    connection,
                    is_postgres=self.is_postgres,
                )
                if self.is_postgres:
                    await unit._fetchrow(
                        "SELECT set_config('lock_timeout', ?, true) AS value",
                        (f"{self._postgres_lock_timeout_ms}ms",),
                    )
                    await unit._fetchrow(
                        "SELECT set_config('statement_timeout', ?, true) AS value",
                        (f"{self._postgres_statement_timeout_ms}ms",),
                    )
                yield unit
        except MandatoryAuditWriteError:
            raise
        except Exception as exc:
            if _is_database_busy(exc):
                raise WebhookRepositoryError(WebhookRepositoryErrorCode.DATABASE_BUSY) from None
            raise

    @asynccontextmanager
    async def _read_connection(self) -> AsyncIterator[object]:
        try:
            async with self._pool.acquire(timeout=5.0) as connection:
                yield connection
        except TimeoutError:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.DATABASE_BUSY) from None
        except Exception as exc:
            if _is_database_busy(exc):
                raise WebhookRepositoryError(WebhookRepositoryErrorCode.DATABASE_BUSY) from None
            raise

    async def get_registration(
        self,
        webhook_id: int,
        *,
        include_deleted: bool = False,
    ) -> WebhookRegistration | None:
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            stored = await unit.get_protected_registration(
                webhook_id,
                include_deleted=include_deleted,
            )
            return stored.registration if stored is not None else None

    async def get_protected_registration(
        self,
        webhook_id: int,
        *,
        include_deleted: bool = False,
    ) -> StoredWebhookRegistration | None:
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            return await unit.get_protected_registration(
                webhook_id,
                include_deleted=include_deleted,
            )

    async def list_registrations(
        self,
        *,
        limit: int,
        before_id: int | None = None,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[WebhookRegistration]:
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            return await unit.list_registrations(
                limit=limit,
                before_id=before_id,
                offset=offset,
                include_deleted=include_deleted,
            )

    async def count_registrations(self) -> int:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).count_registrations()

    async def count_active_registrations(self) -> int:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).count_active_registrations()

    async def count_secret_rotation_required(self) -> int:
        """Return non-deleted registrations awaiting canonical secret rotation."""
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).count_secret_rotation_required()

    async def registration_limit_state(self, *, limit: int) -> RegistrationLimitState:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).registration_limit_state(limit=limit)

    async def active_registration_limit_state(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).active_registration_limit_state(limit=limit)

    async def get_migration_state(self) -> MigrationState:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).get_migration_state(lock=False)

    async def list_delivery_history(
        self,
        webhook_id: int,
        *,
        limit: int,
        offset: int = 0,
    ) -> DeliveryHistoryPage:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).list_delivery_history(webhook_id, limit=limit, offset=offset)

    async def get_delivery_for_registration(
        self,
        webhook_id: int,
        delivery_id: str,
    ) -> WebhookDelivery | None:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).get_delivery_for_registration(webhook_id, delivery_id)

    async def get_delivery_bundle(self, delivery_id: str) -> DeliveryBundle | None:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).get_delivery_bundle(delivery_id)

    async def list_delivery_attempts(
        self,
        webhook_id: int,
        delivery_id: str,
    ) -> tuple[WebhookDeliveryAttempt, ...]:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).list_delivery_attempts(webhook_id, delivery_id)

    async def list_runtime_heartbeats(self) -> tuple[DeliveryRuntimeHeartbeat, ...]:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).list_runtime_heartbeats()

    async def upsert_runtime_heartbeat(
        self,
        write: RuntimeHeartbeatWrite,
    ) -> DeliveryRuntimeHeartbeat:
        if not isinstance(write, RuntimeHeartbeatWrite):
            raise TypeError("runtime heartbeat write is invalid")
        async with self.transaction() as unit:
            return await unit.upsert_runtime_heartbeat(write)

    async def purge_retained_rows(
        self,
        now: datetime,
        retention_cutoff: datetime,
        batch_size: int,
    ) -> RetentionBatchResult:
        if not 1 <= batch_size <= _MAX_RETENTION_BATCH_SIZE:
            raise ValueError("batch_size must be between 1 and 200")
        async with self.transaction() as unit:
            return await unit.purge_retained_rows(now, retention_cutoff, batch_size)

    async def delivery_schema_ready(self) -> bool:
        """Return whether the additive delivery extension is fully present."""
        async with self._read_connection() as connection:
            unit = AdminWebhookUnitOfWork(connection, is_postgres=self.is_postgres)
            if self.is_postgres:
                table_rows = await unit._fetch(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = current_schema()
                      AND table_name = ANY(?::text[])
                    """,
                    (tuple(_DELIVERY_SCHEMA_COLUMNS),),
                )
                column_rows = await unit._fetch(
                    """
                    SELECT table_name, column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = ANY(?::text[])
                    """,
                    (tuple(_DELIVERY_SCHEMA_COLUMNS),),
                )
                constraint_rows = await unit._fetch(
                    """
                    SELECT table_constraints.table_name,
                           table_constraints.constraint_type,
                           COALESCE(check_constraints.check_clause, '') AS definition
                    FROM information_schema.table_constraints AS table_constraints
                    LEFT JOIN information_schema.check_constraints AS check_constraints
                        ON check_constraints.constraint_catalog
                            = table_constraints.constraint_catalog
                       AND check_constraints.constraint_schema
                            = table_constraints.constraint_schema
                       AND check_constraints.constraint_name
                            = table_constraints.constraint_name
                    WHERE table_constraints.table_schema = current_schema()
                      AND table_constraints.table_name = ANY(?::text[])
                    """,
                    (tuple(_DELIVERY_SCHEMA_COLUMNS),),
                )
                primary_key_rows = await unit._fetch(
                    """
                    SELECT key_column_usage.column_name
                    FROM information_schema.table_constraints AS table_constraints
                    JOIN information_schema.key_column_usage AS key_column_usage
                        ON key_column_usage.constraint_catalog
                            = table_constraints.constraint_catalog
                       AND key_column_usage.constraint_schema
                            = table_constraints.constraint_schema
                       AND key_column_usage.constraint_name
                            = table_constraints.constraint_name
                    WHERE table_constraints.table_schema = current_schema()
                      AND table_constraints.table_name = 'admin_webhook_runtime_heartbeats'
                      AND table_constraints.constraint_type = 'PRIMARY KEY'
                    ORDER BY key_column_usage.ordinal_position
                    """,
                )
                index_rows = await unit._fetch(
                    """
                    SELECT index_class.relname AS index_name,
                           relation.relname AS table_name,
                           array_agg(attribute.attname ORDER BY key_columns.ordinality)
                               FILTER (WHERE key_columns.ordinality <= index_data.indnkeyatts)
                               AS column_names,
                           array_agg((key_columns.option_flags & 1) = 1
                               ORDER BY key_columns.ordinality)
                               FILTER (WHERE key_columns.ordinality <= index_data.indnkeyatts)
                               AS descending_flags,
                           pg_get_expr(index_data.indpred, index_data.indrelid)
                               AS predicate
                    FROM pg_index AS index_data
                    JOIN pg_class AS index_class ON index_class.oid = index_data.indexrelid
                    JOIN pg_class AS relation ON relation.oid = index_data.indrelid
                    JOIN pg_namespace AS namespace ON namespace.oid = relation.relnamespace
                    JOIN LATERAL unnest(
                        index_data.indkey::smallint[],
                        index_data.indoption::smallint[]
                    ) WITH ORDINALITY AS key_columns(
                        attribute_number, option_flags, ordinality
                    )
                        ON TRUE
                    JOIN pg_attribute AS attribute
                        ON attribute.attrelid = relation.oid
                       AND attribute.attnum = key_columns.attribute_number
                    WHERE namespace.nspname = current_schema()
                      AND index_class.relname = ANY(?::text[])
                    GROUP BY index_class.relname, relation.relname, index_data.indexrelid,
                             index_data.indnkeyatts,
                             index_data.indpred, index_data.indrelid
                    """,
                    (tuple(_DELIVERY_SCHEMA_INDEXES),),
                )
                tables = {str(row["table_name"]) for row in table_rows}
                columns = {
                    table: {
                        str(row["column_name"]): (
                            str(row["data_type"]),
                            str(row["is_nullable"]),
                        )
                        for row in column_rows
                        if str(row["table_name"]) == table
                    }
                    for table in _DELIVERY_POSTGRES_COLUMN_CONTRACT
                }
                constraint_definitions = {
                    table: tuple(
                        str(row["definition"])
                        for row in constraint_rows
                        if str(row["table_name"]) == table
                        and str(row["constraint_type"]) == "CHECK"
                    )
                    for table in _DELIVERY_SCHEMA_COLUMNS
                }
                heartbeat_primary_key = tuple(
                    str(row["column_name"]) for row in primary_key_rows
                )
                index_contracts = {
                    str(row["index_name"]): (
                        str(row["table_name"]),
                        tuple(str(column) for column in row["column_names"]),
                        tuple(bool(flag) for flag in row["descending_flags"]),
                        _strip_outer_parentheses(
                            _compact_schema_sql(str(row["predicate"] or ""))
                        ),
                    )
                    for row in index_rows
                }
            else:
                table_rows = await unit._fetch(
                    """
                    SELECT name, sql
                    FROM sqlite_master
                    WHERE type = 'table' AND name IN (?, ?, ?)
                    """,
                    tuple(_DELIVERY_SCHEMA_COLUMNS),
                )
                columns: dict[str, dict[str, tuple[str, bool, int]]] = {}
                for table in _DELIVERY_SCHEMA_COLUMNS:
                    rows = await unit._fetch(f"PRAGMA table_info({table})")
                    columns[table] = {
                        str(row["name"]): (
                            str(row["type"]).upper(),
                            bool(row["notnull"]),
                            int(row["pk"]),
                        )
                        for row in rows
                    }
                index_rows = await unit._fetch(
                    """
                    SELECT name, tbl_name, sql
                    FROM sqlite_master
                    WHERE type = 'index' AND name IN (?, ?, ?)
                    """,
                    tuple(_DELIVERY_SCHEMA_INDEXES),
                )
                tables = {str(row["name"]) for row in table_rows}
                table_definitions = {
                    str(row["name"]): (str(row["sql"] or ""),)
                    for row in table_rows
                }
                index_contracts = {
                    str(row["name"]): (
                        str(row["tbl_name"]),
                        _compact_schema_sql(str(row["sql"] or "")),
                    )
                    for row in index_rows
                }
                return (
                    tables == set(_DELIVERY_SCHEMA_COLUMNS)
                    and all(
                        columns[table].get(column) == contract
                        for table, contract_columns in _DELIVERY_SCHEMA_COLUMN_CONTRACT.items()
                        for column, contract in contract_columns.items()
                    )
                    and _has_required_delivery_checks(
                        table_definitions, is_postgres=False
                    )
                    and set(index_contracts) == _DELIVERY_SCHEMA_INDEXES
                    and all(
                        index_contracts[index_name]
                        == (
                            _DELIVERY_SCHEMA_INDEX_TABLES[index_name],
                            expected_definition,
                        )
                        for index_name, expected_definition in _DELIVERY_SQLITE_INDEX_DEFINITIONS.items()
                    )
                )

            expected_indexes = _DELIVERY_SCHEMA_INDEX_COLUMNS
            recovery_predicate = "state=any(array['pending','enqueue_claimed'])"
            disposition_predicate = "pending_jobs_dispositionisnotnull"
            return (
                tables == set(_DELIVERY_SCHEMA_COLUMNS)
                and all(
                    columns[table].get(column) == contract
                    for table, contract_columns in _DELIVERY_POSTGRES_COLUMN_CONTRACT.items()
                    for column, contract in contract_columns.items()
                )
                and _has_required_delivery_checks(
                    constraint_definitions, is_postgres=True
                )
                and heartbeat_primary_key == ("component", "instance_id")
                and set(index_contracts) == _DELIVERY_SCHEMA_INDEXES
                and all(
                    index_contracts[index_name][0]
                    == _DELIVERY_SCHEMA_INDEX_TABLES[index_name]
                    and index_contracts[index_name][1] == expected_columns
                    and index_contracts[index_name][2]
                    == _DELIVERY_POSTGRES_INDEX_DESCENDING[index_name]
                    for index_name, expected_columns in expected_indexes.items()
                )
                and index_contracts["idx_admin_webhook_deliveries_recovery"][3]
                == recovery_predicate
                and index_contracts[
                    "idx_admin_webhook_deliveries_disposition_recovery"
                ][3]
                == disposition_predicate
                and index_contracts[
                    "idx_admin_webhook_runtime_heartbeats_freshness"
                ][3]
                == ""
            )

    async def get_legacy_import_snapshot(self) -> LegacyImportDatabaseSnapshot:
        """Read legacy rows and canonical ID-allocation state without mutation."""
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).get_legacy_import_snapshot(lock=False)

    async def page_protected_rows(
        self,
        *,
        table: str,
        after: str | None,
        limit: int,
        inventory_at: datetime | None = None,
    ) -> list[ProtectedRow]:
        """Read a bounded page, optionally freezing replay expiry eligibility."""
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).page_protected_rows(
                table=table,
                after=after,
                limit=limit,
                inventory_at=inventory_at,
            )

    async def find_purge_eligible_registration_ids(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[int]:
        async with self._read_connection() as connection:
            return await AdminWebhookUnitOfWork(
                connection,
                is_postgres=self.is_postgres,
            ).find_purge_eligible_registration_ids(now=now, limit=limit)


class AdminWebhookUnitOfWork(_ConnectionAdapter):
    """Transaction-bound operations shared by both supported database backends."""

    async def _event_by_source(self, event: EventInsert) -> StoredWebhookEvent | None:
        if event.source_kind is EventSourceKind.AGGREGATE:
            row = await self._fetchrow(
                f"""
                SELECT {_EVENT_COLUMNS}
                FROM admin_webhook_events
                WHERE event_type = ?
                  AND aggregate_type = ?
                  AND aggregate_id = ?
                  AND aggregate_version = ?
                """,
                (
                    event.event_type,
                    event.aggregate_type,
                    event.aggregate_id,
                    event.aggregate_version,
                ),
            )
        else:
            row = await self._fetchrow(
                f"""
                SELECT {_EVENT_COLUMNS}
                FROM admin_webhook_events
                WHERE event_type = ? AND source_command_id = ?
                """,
                (event.event_type, event.source_command_id),
            )
        return _stored_event_from_row(row) if row is not None else None

    async def _automatic_deliveries(
        self,
        event_id: str,
    ) -> tuple[StoredWebhookDelivery, ...]:
        rows = await self._fetch(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE event_id = ? AND kind = 'automatic'
            ORDER BY webhook_id ASC, id ASC
            """,
            (event_id,),
        )
        return tuple(_stored_delivery_from_row(row) for row in rows)

    async def capture_event_and_expand(
        self,
        event: EventInsert,
        delivery_id_factory: Callable[[], str],
        expires_at: datetime,
    ) -> EventCaptureResult:
        """Insert one protected event and set-expand its active subscriptions."""
        if not isinstance(event, EventInsert):
            raise TypeError("event insert is invalid")
        if not callable(delivery_id_factory):
            raise TypeError("delivery ID factory is invalid")
        expires_at = _utc_datetime(expires_at, field="expires_at")
        if expires_at != event.created_at + timedelta(hours=72):
            raise ValueError("automatic delivery expiry must be exactly 72 hours")
        inserted = await self._fetchrow(
            """
            INSERT INTO admin_webhook_events (
                id, event_type, api_version, source_kind, aggregate_type,
                aggregate_id, aggregate_version, source_command_id,
                source_component, source_request_id, body_ciphertext_json,
                body_key_id, body_size_bytes, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT DO NOTHING
            RETURNING id
            """,
            (
                event.id,
                event.event_type,
                event.api_version,
                event.source_kind.value,
                event.aggregate_type,
                event.aggregate_id,
                event.aggregate_version,
                event.source_command_id,
                event.source_component,
                event.source_request_id,
                event.body.ciphertext_json,
                event.body.key_id,
                event.body_size_bytes,
                event.created_at,
            ),
        )
        if inserted is None:
            existing = await self._event_by_source(event)
            if existing is None:
                raise ValueError("event ID conflicts with another source")
            return EventCaptureResult(
                event=existing,
                deliveries=await self._automatic_deliveries(existing.id),
                inserted=False,
            )

        if self._is_postgres:
            registrations = await self._fetch(
                """
                SELECT /* admin_webhook_match_subscriptions */
                    id, delivery_config_version, secret_version
                FROM admin_webhook_registrations
                WHERE active = TRUE
                  AND deleted_at IS NULL
                  AND secret_rotation_required = FALSE
                  AND event_types_json::jsonb @> ?::jsonb
                ORDER BY id ASC
                """,
                (_canonical_json([event.event_type]),),
            )
        else:
            registrations = await self._fetch(
                """
                SELECT /* admin_webhook_match_subscriptions */
                    registration.id,
                    registration.delivery_config_version,
                    registration.secret_version
                FROM admin_webhook_registrations AS registration
                WHERE registration.active = 1
                  AND registration.deleted_at IS NULL
                  AND registration.secret_rotation_required = 0
                  AND EXISTS (
                      SELECT 1
                      FROM json_each(registration.event_types_json)
                      WHERE json_each.value = ?
                  )
                ORDER BY registration.id ASC
                """,
                (event.event_type,),
            )
        delivery_rows: list[tuple[object, ...]] = []
        for registration in registrations:
            delivery_id = delivery_id_factory()
            _bounded_text(delivery_id, field="delivery ID", maximum=128)
            delivery_rows.append(
                (
                    delivery_id,
                    event.id,
                    int(registration["id"]),
                    DeliveryKind.AUTOMATIC.value,
                    int(registration["delivery_config_version"]),
                    int(registration["secret_version"]),
                    DeliveryState.PENDING.value,
                    expires_at,
                    event.created_at,
                    event.created_at,
                )
            )
        await self._executemany(
            """
            INSERT /* admin_webhook_delivery_fanout */
            INTO admin_webhook_deliveries (
                id, event_id, webhook_id, kind, delivery_config_version,
                secret_version, state, expires_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            delivery_rows,
        )
        await self.mark_first_canonical_activity("event_capture", event.created_at)
        stored_event = StoredWebhookEvent(
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
        return EventCaptureResult(
            event=stored_event,
            deliveries=await self._automatic_deliveries(event.id),
            inserted=True,
        )

    async def insert_delivery(
        self,
        delivery_id: str,
        *,
        event_id: str,
        webhook_id: int,
        kind: DeliveryKind,
        expires_at: datetime,
        now: datetime,
        redelivery_of_id: str | None = None,
    ) -> StoredWebhookDelivery:
        _bounded_text(delivery_id, field="delivery ID", maximum=128)
        _bounded_text(event_id, field="event ID", maximum=128)
        if kind not in {DeliveryKind.MANUAL, DeliveryKind.TEST}:
            raise ValueError("explicit delivery kind is invalid")
        expires_at = _utc_datetime(expires_at, field="expires_at")
        now = _utc_datetime(now, field="now")
        registration = await self._fetchrow(
            """
            SELECT delivery_config_version, secret_version
            FROM admin_webhook_registrations
            WHERE id = ? AND deleted_at IS NULL
            """,
            (webhook_id,),
        )
        if registration is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)
        row = await self._fetchrow(
            f"""
            INSERT INTO admin_webhook_deliveries (
                id, event_id, webhook_id, kind, delivery_config_version,
                secret_version, state, expires_at, redelivery_of_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
            RETURNING {_DELIVERY_COLUMNS}
            """,
            (
                delivery_id,
                event_id,
                webhook_id,
                kind.value,
                int(registration["delivery_config_version"]),
                int(registration["secret_version"]),
                expires_at,
                redelivery_of_id,
                now,
                now,
            ),
        )
        if row is None:
            raise RuntimeError("delivery insert returned no row")
        return _stored_delivery_from_row(row)

    async def list_delivery_history(
        self,
        webhook_id: int,
        *,
        limit: int,
        offset: int = 0,
    ) -> DeliveryHistoryPage:
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if not 0 <= offset <= 10_000:
            raise ValueError("offset must be between 0 and 10000")
        count = await self._fetchrow(
            "SELECT COUNT(*) AS count FROM admin_webhook_deliveries WHERE webhook_id = ?",
            (webhook_id,),
        )
        rows = await self._fetch(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE webhook_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (webhook_id, limit, offset),
        )
        return DeliveryHistoryPage(
            items=tuple(_stored_delivery_from_row(row).delivery for row in rows),
            total=int(count["count"]) if count is not None else 0,
            limit=limit,
            offset=offset,
        )

    async def get_delivery_for_registration(
        self,
        webhook_id: int,
        delivery_id: str,
    ) -> WebhookDelivery | None:
        row = await self._fetchrow(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE webhook_id = ? AND id = ?
            """,
            (webhook_id, delivery_id),
        )
        return _stored_delivery_from_row(row).delivery if row is not None else None

    async def get_delivery_bundle(self, delivery_id: str) -> DeliveryBundle | None:
        delivery_row = await self._fetchrow(
            f"SELECT {_DELIVERY_COLUMNS} FROM admin_webhook_deliveries WHERE id = ?",
            (delivery_id,),
        )
        if delivery_row is None:
            return None
        delivery = _stored_delivery_from_row(delivery_row)
        event_row = await self._fetchrow(
            f"SELECT {_EVENT_COLUMNS} FROM admin_webhook_events WHERE id = ?",
            (delivery.delivery.event_id,),
        )
        registration = await self.get_protected_registration(
            delivery.delivery.webhook_id,
            include_deleted=True,
        )
        if event_row is None or registration is None:
            raise ValueError("delivery bundle references are invalid")
        return DeliveryBundle(
            event=_stored_event_from_row(event_row),
            delivery=delivery,
            registration=registration,
        )

    async def list_delivery_attempts(
        self,
        webhook_id: int,
        delivery_id: str,
    ) -> tuple[WebhookDeliveryAttempt, ...]:
        rows = await self._fetch(
            f"""
            SELECT {_ATTEMPT_COLUMNS}
            FROM admin_webhook_delivery_attempts AS attempt
            WHERE EXISTS (
                SELECT 1 FROM admin_webhook_deliveries AS delivery
                WHERE delivery.id = attempt.delivery_id
                  AND delivery.webhook_id = ?
            )
              AND attempt.delivery_id = ?
            ORDER BY attempt.attempt_number ASC, attempt.id ASC
            """,
            (webhook_id, delivery_id),
        )
        return tuple(_attempt_from_row(row) for row in rows)

    async def claim_pending_delivery(
        self,
        claim_token: str,
        claimed_until: datetime,
        now: datetime,
    ) -> EnqueueClaim | None:
        _bounded_text(claim_token, field="enqueue claim token", maximum=255)
        claimed_until = _utc_datetime(claimed_until, field="claimed_until")
        now = _utc_datetime(now, field="now")
        if claimed_until <= now:
            raise ValueError("enqueue claim expiry must follow claim time")
        row = await self._fetchrow(
            f"""
            UPDATE admin_webhook_deliveries
            SET state = 'enqueue_claimed',
                enqueue_claim_token = ?,
                enqueue_claim_expires_at = ?,
                updated_at = ?
            WHERE id = (
                SELECT id
                FROM admin_webhook_deliveries
                WHERE state = 'pending' AND expires_at > ?
                ORDER BY created_at ASC, id ASC
                LIMIT 1
            )
              AND state = 'pending'
            RETURNING {_DELIVERY_COLUMNS}
            """,
            (claim_token, claimed_until, now, now),
        )
        if row is None:
            return None
        return EnqueueClaim(
            delivery=_stored_delivery_from_row(row),
            claim_token=claim_token,
            claimed_until=claimed_until,
        )

    async def attach_jobs_job(
        self,
        delivery_id: str,
        claim_token: str,
        jobs_job_id: str,
        now: datetime,
    ) -> StoredWebhookDelivery | None:
        _bounded_text(delivery_id, field="delivery ID", maximum=128)
        _bounded_text(claim_token, field="enqueue claim token", maximum=255)
        _bounded_text(jobs_job_id, field="Jobs job ID", maximum=255)
        now = _utc_datetime(now, field="now")
        row = await self._fetchrow(
            f"""
            UPDATE admin_webhook_deliveries
            SET jobs_job_id = ?, state = 'queued',
                enqueue_claim_token = NULL,
                enqueue_claim_expires_at = NULL,
                updated_at = ?
            WHERE id = ?
              AND state = 'enqueue_claimed'
              AND enqueue_claim_token = ?
              AND enqueue_claim_expires_at >= ?
              AND jobs_job_id IS NULL
            RETURNING {_DELIVERY_COLUMNS}
            """,
            (jobs_job_id, now, delivery_id, claim_token, now),
        )
        return _stored_delivery_from_row(row) if row is not None else None

    async def release_expired_enqueue_claim(
        self,
        delivery_id: str,
        expected_token: str,
        now: datetime,
    ) -> bool:
        now = _utc_datetime(now, field="now")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_deliveries
            SET state = 'pending', enqueue_claim_token = NULL,
                enqueue_claim_expires_at = NULL, updated_at = ?
            WHERE id = ?
              AND state = 'enqueue_claimed'
              AND enqueue_claim_token = ?
              AND enqueue_claim_expires_at <= ?
            RETURNING id
            """,
            (now, delivery_id, expected_token, now),
        )
        return row is not None

    async def reserve_jobs_attempt(
        self,
        delivery_id: str,
        jobs_job_id: str,
        lease_id: str,
        attempt_id: str,
        request_timeout_seconds: int,
        now: datetime,
        required_horizon: datetime,
    ) -> AttemptReservation | None:
        for value, field, maximum in (
            (delivery_id, "delivery ID", 128),
            (jobs_job_id, "Jobs job ID", 255),
            (lease_id, "Jobs lease ID", 255),
            (attempt_id, "attempt ID", 128),
        ):
            _bounded_text(value, field=field, maximum=maximum)
        if not 1 <= request_timeout_seconds <= 30:
            raise ValueError("request timeout must be between 1 and 30")
        now = _utc_datetime(now, field="now")
        required_horizon = _utc_datetime(required_horizon, field="required_horizon")
        if required_horizon < now + timedelta(seconds=request_timeout_seconds):
            raise ValueError("required horizon is shorter than request timeout")
        delivery_row = await self._fetchrow(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE id = ? AND jobs_job_id = ?
            """,
            (delivery_id, jobs_job_id),
        )
        if delivery_row is None:
            return None
        delivery = _stored_delivery_from_row(delivery_row)
        if delivery.delivery.attempt_count >= 4:
            return AttemptReservation(
                reserved=False,
                delivery=delivery,
                attempt=None,
                bundle=None,
                reason_code=DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED,
            )
        if delivery.delivery.expires_at <= now:
            return None
        if (
            delivery.pending_jobs_disposition is not None
            and not delivery.jobs_disposition_applied
        ):
            return None
        if delivery.delivery.state not in {
            DeliveryState.QUEUED,
            DeliveryState.RETRY_WAIT,
        }:
            return None
        attempt_number = delivery.delivery.attempt_count + 1
        inserted = await self._fetchrow(
            f"""
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, jobs_job_id, jobs_lease_id,
                request_timeout_seconds, started_at, state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'processing', ?)
            RETURNING {_ATTEMPT_COLUMNS}
            """,
            (
                attempt_id,
                delivery_id,
                attempt_number,
                jobs_job_id,
                lease_id,
                request_timeout_seconds,
                now,
                now,
            ),
        )
        updated = await self._fetchrow(
            f"""
            UPDATE admin_webhook_deliveries
            SET state = 'processing', attempt_count = ?, current_attempt_id = ?,
                pending_jobs_disposition = NULL,
                pending_jobs_disposition_delay_seconds = NULL,
                pending_jobs_disposition_token = NULL,
                pending_jobs_disposition_not_before_at = NULL,
                jobs_disposition_applied = FALSE,
                updated_at = ?
            WHERE id = ? AND jobs_job_id = ?
              AND state IN ('queued', 'retry_wait')
              AND attempt_count = ?
            RETURNING {_DELIVERY_COLUMNS}
            """,
            (
                attempt_number,
                attempt_id,
                now,
                delivery_id,
                jobs_job_id,
                attempt_number - 1,
            ),
        )
        if inserted is None or updated is None:
            raise ValueError("attempt reservation compare-and-set failed")
        await self.mark_first_canonical_activity("delivery_attempt", now)
        bundle = await self.get_delivery_bundle(delivery_id)
        if bundle is None:
            raise ValueError("reserved delivery bundle is unavailable")
        return AttemptReservation(
            reserved=True,
            delivery=_stored_delivery_from_row(updated),
            attempt=_attempt_from_row(inserted),
            bundle=bundle,
        )

    async def reserve_test_attempt(
        self,
        test_attempt_token: str,
        delivery_id: str,
        attempt_id: str,
        request_timeout_seconds: int,
        started_at: datetime,
    ) -> AttemptReservation | None:
        _bounded_text(test_attempt_token, field="test attempt token", maximum=255)
        _bounded_text(delivery_id, field="delivery ID", maximum=128)
        _bounded_text(attempt_id, field="attempt ID", maximum=128)
        if not 1 <= request_timeout_seconds <= 30:
            raise ValueError("request timeout must be between 1 and 30")
        started_at = _utc_datetime(started_at, field="started_at")
        delivery_row = await self._fetchrow(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE id = ? AND kind = 'test' AND jobs_job_id IS NULL
            """,
            (delivery_id,),
        )
        if delivery_row is None:
            return None
        delivery = _stored_delivery_from_row(delivery_row)
        if delivery.delivery.attempt_count >= 4:
            return AttemptReservation(
                reserved=False,
                delivery=delivery,
                attempt=None,
                bundle=None,
                reason_code=DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED,
            )
        if delivery.delivery.state is not DeliveryState.PENDING:
            return None
        inserted = await self._fetchrow(
            f"""
            INSERT INTO admin_webhook_delivery_attempts (
                id, delivery_id, attempt_number, test_attempt_token,
                request_timeout_seconds, started_at, state, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'processing', ?)
            RETURNING {_ATTEMPT_COLUMNS}
            """,
            (
                attempt_id,
                delivery_id,
                delivery.delivery.attempt_count + 1,
                test_attempt_token,
                request_timeout_seconds,
                started_at,
                started_at,
            ),
        )
        updated = await self._fetchrow(
            f"""
            UPDATE admin_webhook_deliveries
            SET state = 'processing', attempt_count = attempt_count + 1,
                current_attempt_id = ?, updated_at = ?
            WHERE id = ? AND kind = 'test' AND state = 'pending'
              AND attempt_count = ?
            RETURNING {_DELIVERY_COLUMNS}
            """,
            (attempt_id, started_at, delivery_id, delivery.delivery.attempt_count),
        )
        if inserted is None or updated is None:
            raise ValueError("test attempt reservation compare-and-set failed")
        await self.mark_first_canonical_activity("delivery_attempt", started_at)
        bundle = await self.get_delivery_bundle(delivery_id)
        if bundle is None:
            raise ValueError("reserved test delivery bundle is unavailable")
        return AttemptReservation(
            reserved=True,
            delivery=_stored_delivery_from_row(updated),
            attempt=_attempt_from_row(inserted),
            bundle=bundle,
        )

    async def finish_attempt_and_prepare_disposition(
        self,
        attempt_token: str,
        outcome: AttemptCompletion,
        disposition_token: str | None,
        not_before_at: datetime | None,
    ) -> PendingJobsDisposition | None:
        _bounded_text(attempt_token, field="attempt token", maximum=255)
        if not isinstance(outcome, AttemptCompletion):
            raise TypeError("attempt outcome is invalid")
        attempt_rows = await self._fetch(
            f"""
            SELECT {_ATTEMPT_COLUMNS}
            FROM admin_webhook_delivery_attempts
            WHERE state = 'processing'
              AND (jobs_lease_id = ? OR test_attempt_token = ?)
            ORDER BY id ASC
            LIMIT 2
            """,
            (attempt_token, attempt_token),
        )
        if len(attempt_rows) != 1:
            return None
        attempt_row = attempt_rows[0]
        delivery_row = await self._fetchrow(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE id = ? AND state = 'processing' AND current_attempt_id = ?
            """,
            (attempt_row["delivery_id"], attempt_row["id"]),
        )
        if delivery_row is None:
            return None
        delivery = _stored_delivery_from_row(delivery_row)
        pending: PendingJobsDisposition | None = None
        if delivery.jobs_job_id is None:
            if outcome.disposition is not None or disposition_token is not None or not_before_at is not None:
                raise ValueError("test attempt cannot prepare a Jobs disposition")
        else:
            if outcome.disposition is None or disposition_token is None or not_before_at is None:
                raise ValueError("Jobs attempt disposition coordinates are required")
            pending = PendingJobsDisposition(
                delivery_id=delivery.delivery.id,
                jobs_job_id=delivery.jobs_job_id,
                attempt_id=str(attempt_row["id"]),
                kind=outcome.disposition,
                delay_seconds=outcome.requested_retry_delay_seconds,
                token=disposition_token,
                not_before_at=not_before_at,
            )
        attempt_updated = await self._fetchrow(
            """
            UPDATE admin_webhook_delivery_attempts
            SET state = ?, status_code = ?, latency_ms = ?, reason_code = ?,
                requested_retry_delay_seconds = ?, finished_at = ?
            WHERE id = ? AND state = 'processing'
              AND (jobs_lease_id = ? OR test_attempt_token = ?)
            RETURNING id
            """,
            (
                outcome.attempt_state.value,
                outcome.status_code,
                outcome.latency_ms,
                outcome.reason_code.value if outcome.reason_code is not None else None,
                outcome.requested_retry_delay_seconds,
                outcome.finished_at,
                attempt_row["id"],
                attempt_token,
                attempt_token,
            ),
        )
        terminal_at = (
            outcome.finished_at
            if outcome.delivery_state in DeliveryState.terminal_states()
            else None
        )
        delivery_updated = await self._fetchrow(
            """
            UPDATE admin_webhook_deliveries
            SET state = ?, status_code = ?, latency_ms = ?, reason_code = ?,
                pending_jobs_disposition = ?,
                pending_jobs_disposition_delay_seconds = ?,
                pending_jobs_disposition_token = ?,
                pending_jobs_disposition_not_before_at = ?,
                jobs_disposition_applied = FALSE,
                completed_after_config_change = ?, terminal_at = ?,
                updated_at = ?
            WHERE id = ? AND state = 'processing' AND current_attempt_id = ?
            RETURNING id
            """,
            (
                outcome.delivery_state.value,
                outcome.status_code,
                outcome.latency_ms,
                outcome.reason_code.value if outcome.reason_code is not None else None,
                pending.kind.value if pending is not None else None,
                pending.delay_seconds if pending is not None else None,
                pending.token if pending is not None else None,
                pending.not_before_at if pending is not None else None,
                outcome.completed_after_config_change,
                terminal_at,
                outcome.finished_at,
                delivery.delivery.id,
                attempt_row["id"],
            ),
        )
        if attempt_updated is None or delivery_updated is None:
            raise ValueError("attempt completion compare-and-set failed")
        return pending

    async def acknowledge_jobs_disposition(
        self,
        delivery_id: str,
        disposition_token: str,
        jobs_state: str,
    ) -> bool:
        if _DISPOSITION_TOKEN.fullmatch(disposition_token) is None:
            return False
        expected_states = {
            JobsDispositionKind.COMPLETE: "completed",
            JobsDispositionKind.RETRY: "queued",
            JobsDispositionKind.FAIL: "failed",
            JobsDispositionKind.CANCEL: "cancelled",
            JobsDispositionKind.DEFER: "queued",
        }
        row = await self._fetchrow(
            """
            SELECT current_attempt_id, pending_jobs_disposition
            FROM admin_webhook_deliveries
            WHERE id = ? AND pending_jobs_disposition_token = ?
              AND pending_jobs_disposition IS NOT NULL
              AND jobs_disposition_applied = FALSE
            """,
            (delivery_id, disposition_token),
        )
        if row is None:
            return False
        try:
            disposition = JobsDispositionKind(str(row["pending_jobs_disposition"]))
        except ValueError:
            return False
        if expected_states[disposition] != jobs_state:
            return False
        updated = await self._fetchrow(
            """
            UPDATE admin_webhook_deliveries
            SET jobs_disposition_applied = TRUE
            WHERE id = ? AND pending_jobs_disposition_token = ?
              AND jobs_disposition_applied = FALSE
            RETURNING current_attempt_id
            """,
            (delivery_id, disposition_token),
        )
        if updated is None:
            return False
        attempt_id = updated["current_attempt_id"]
        if attempt_id is not None:
            await self._execute(
                """
                UPDATE admin_webhook_delivery_attempts
                SET jobs_disposition_applied = TRUE
                WHERE id = ? AND delivery_id = ? AND state != 'processing'
                """,
                (attempt_id, delivery_id),
            )
        return True

    async def close_stale_attempt_as_unknown(
        self,
        delivery_id: str,
        attempt_id: str,
        stale_before: datetime,
    ) -> bool:
        stale_before = _utc_datetime(stale_before, field="stale_before")
        row = await self._fetchrow(
            f"""
            SELECT {_ATTEMPT_COLUMNS}
            FROM admin_webhook_delivery_attempts AS attempt
            WHERE attempt.id = ? AND attempt.delivery_id = ?
              AND attempt.state = 'processing'
              AND EXISTS (
                  SELECT 1 FROM admin_webhook_deliveries AS delivery
                  WHERE delivery.id = attempt.delivery_id
                    AND delivery.state = 'processing'
                    AND delivery.current_attempt_id = attempt.id
              )
            """,
            (attempt_id, delivery_id),
        )
        if row is None:
            return False
        started_at = _parse_datetime(row["started_at"])
        if started_at is None:
            raise ValueError("persisted attempt timestamp is invalid")
        timeout = (
            int(row["request_timeout_seconds"])
            if row["request_timeout_seconds"] is not None
            else 30
        )
        if started_at + timedelta(seconds=timeout) > stale_before:
            return False
        attempt_updated = await self._fetchrow(
            """
            UPDATE admin_webhook_delivery_attempts
            SET state = 'outcome_unknown', finished_at = ?,
                reason_code = 'outcome_unknown'
            WHERE id = ? AND delivery_id = ? AND state = 'processing'
            RETURNING id
            """,
            (stale_before, attempt_id, delivery_id),
        )
        delivery_updated = await self._fetchrow(
            """
            UPDATE admin_webhook_deliveries
            SET state = 'retry_wait', reason_code = 'outcome_unknown',
                updated_at = ?
            WHERE id = ? AND state = 'processing' AND current_attempt_id = ?
            RETURNING id
            """,
            (stale_before, delivery_id, attempt_id),
        )
        if attempt_updated is None or delivery_updated is None:
            raise ValueError("stale attempt compare-and-set failed")
        return True

    async def cancel_registration_work(
        self,
        webhook_id: int,
        cutoff_versions: tuple[int, int],
        reason: DeliveryReasonCode,
        disposition_token_factory: Callable[[], str],
        now: datetime,
    ) -> tuple[PendingJobsDisposition, ...]:
        if (
            not isinstance(cutoff_versions, tuple)
            or len(cutoff_versions) != 2
            or any(isinstance(value, bool) or value < 1 for value in cutoff_versions)
        ):
            raise ValueError("delivery cutoff versions are invalid")
        if reason not in {
            DeliveryReasonCode.CANCELED_DELETED,
            DeliveryReasonCode.CANCELED_DISABLED,
            DeliveryReasonCode.CANCELED_SECRET_ROTATION,
            DeliveryReasonCode.SUPERSEDED_CONFIG,
        }:
            raise ValueError("delivery cancellation reason is invalid")
        if not callable(disposition_token_factory):
            raise TypeError("disposition token factory is invalid")
        now = _utc_datetime(now, field="now")
        rows = await self._fetch(
            f"""
            SELECT {_DELIVERY_COLUMNS}
            FROM admin_webhook_deliveries
            WHERE webhook_id = ?
              AND state NOT IN ('succeeded', 'dead', 'canceled', 'superseded')
              AND (
                  delivery_config_version < ? OR secret_version < ?
              )
            ORDER BY created_at ASC, id ASC
            """,
            (webhook_id, cutoff_versions[0], cutoff_versions[1]),
        )
        pending: list[PendingJobsDisposition] = []
        target_state = (
            DeliveryState.SUPERSEDED
            if reason is DeliveryReasonCode.SUPERSEDED_CONFIG
            else DeliveryState.CANCELED
        )
        for raw in rows:
            delivery = _stored_delivery_from_row(raw)
            disposition: PendingJobsDisposition | None = None
            disposition_attempt_id = (
                delivery.current_attempt_id
                if delivery.delivery.state is DeliveryState.PROCESSING
                else None
            )
            if delivery.jobs_job_id is not None:
                disposition = PendingJobsDisposition(
                    delivery_id=delivery.delivery.id,
                    jobs_job_id=delivery.jobs_job_id,
                    attempt_id=disposition_attempt_id,
                    kind=JobsDispositionKind.CANCEL,
                    delay_seconds=None,
                    token=disposition_token_factory(),
                    not_before_at=now,
                )
                pending.append(disposition)
            if delivery.current_attempt_id is not None:
                await self._execute(
                    """
                    UPDATE admin_webhook_delivery_attempts
                    SET state = ?, reason_code = ?, finished_at = ?
                    WHERE id = ? AND delivery_id = ? AND state = 'processing'
                    """,
                    (
                        AttemptState.SUPERSEDED.value
                        if target_state is DeliveryState.SUPERSEDED
                        else AttemptState.CANCELED.value,
                        reason.value,
                        now,
                        delivery.current_attempt_id,
                        delivery.delivery.id,
                    ),
                )
            updated = await self._execute(
                """
                UPDATE admin_webhook_deliveries
                SET state = ?, reason_code = ?, terminal_at = ?,
                    current_attempt_id = ?,
                    pending_jobs_disposition = ?,
                    pending_jobs_disposition_delay_seconds = NULL,
                    pending_jobs_disposition_token = ?,
                    pending_jobs_disposition_not_before_at = ?,
                    jobs_disposition_applied = FALSE, updated_at = ?
                WHERE id = ?
                  AND state NOT IN ('succeeded', 'dead', 'canceled', 'superseded')
                  AND delivery_config_version = ? AND secret_version = ?
                """,
                (
                    target_state.value,
                    reason.value,
                    now,
                    disposition_attempt_id,
                    disposition.kind.value if disposition is not None else None,
                    disposition.token if disposition is not None else None,
                    disposition.not_before_at if disposition is not None else None,
                    now,
                    delivery.delivery.id,
                    delivery.delivery.delivery_config_version,
                    delivery.delivery.secret_version,
                ),
            )
            if updated != 1:
                raise ValueError("delivery cancellation compare-and-set failed")
        return tuple(pending)

    async def expire_delivery(
        self,
        delivery_id: str,
        expected_state: DeliveryState,
        now: datetime,
    ) -> bool:
        if expected_state in DeliveryState.terminal_states():
            return False
        now = _utc_datetime(now, field="now")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_deliveries
            SET state = 'dead', reason_code = 'delivery_expired',
                terminal_at = ?, updated_at = ?
            WHERE id = ? AND state = ? AND expires_at <= ?
            RETURNING id
            """,
            (now, now, delivery_id, expected_state.value, now),
        )
        return row is not None

    async def upsert_runtime_heartbeat(
        self,
        write: RuntimeHeartbeatWrite,
    ) -> DeliveryRuntimeHeartbeat:
        if not isinstance(write, RuntimeHeartbeatWrite):
            raise TypeError("runtime heartbeat write is invalid")
        row = await self._fetchrow(
            """
            INSERT INTO admin_webhook_runtime_heartbeats (
                component, instance_id, ready, reason_code, heartbeat_at,
                last_success_at, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(component, instance_id) DO UPDATE SET
                ready = excluded.ready,
                reason_code = excluded.reason_code,
                heartbeat_at = excluded.heartbeat_at,
                last_success_at = excluded.last_success_at,
                updated_at = excluded.updated_at
            RETURNING component, instance_id, ready, reason_code, heartbeat_at,
                      last_success_at, created_at, updated_at
            """,
            (
                write.component.value,
                write.instance_id,
                write.ready,
                write.reason_code.value if write.reason_code is not None else None,
                write.heartbeat_at,
                write.last_success_at,
                write.heartbeat_at,
                write.heartbeat_at,
            ),
        )
        if row is None:
            raise RuntimeError("runtime heartbeat upsert returned no row")
        return _heartbeat_from_row(row)

    async def list_runtime_heartbeats(
        self,
    ) -> tuple[DeliveryRuntimeHeartbeat, ...]:
        rows = await self._fetch(
            """
            SELECT component, instance_id, ready, reason_code, heartbeat_at,
                   last_success_at, created_at, updated_at
            FROM admin_webhook_runtime_heartbeats
            ORDER BY component ASC, instance_id ASC
            """
        )
        return tuple(_heartbeat_from_row(row) for row in rows)

    async def _delete_by_ids(
        self,
        table: str,
        ids: Sequence[object],
    ) -> int:
        allowed = {
            "admin_webhook_idempotency",
            "admin_webhook_deliveries",
            "admin_webhook_events",
            "admin_webhook_runtime_heartbeats",
            "admin_webhook_registrations",
        }
        if table not in allowed or not ids:
            return 0
        placeholders = ", ".join("?" for _ in ids)
        return await self._execute(
            f"DELETE FROM {table} WHERE id IN ({placeholders})",  # noqa: S608
            tuple(ids),
        )

    async def purge_retained_rows(
        self,
        now: datetime,
        retention_cutoff: datetime,
        batch_size: int,
    ) -> RetentionBatchResult:
        now = _utc_datetime(now, field="now")
        retention_cutoff = _utc_datetime(retention_cutoff, field="retention_cutoff")
        if not 1 <= batch_size <= _MAX_RETENTION_BATCH_SIZE:
            raise ValueError("batch_size must be between 1 and 200")
        retention_cutoff = min(
            retention_cutoff,
            now - timedelta(days=_MIN_TOMBSTONE_RETENTION_DAYS),
        )
        remaining = batch_size

        idempotency_rows = await self._fetch(
            """
            SELECT id FROM admin_webhook_idempotency
            WHERE expires_at <= ?
            ORDER BY expires_at ASC, id ASC LIMIT ?
            """,
            (now, remaining),
        )
        expired_idempotency = await self._delete_by_ids(
            "admin_webhook_idempotency",
            [row["id"] for row in idempotency_rows],
        )
        remaining -= expired_idempotency

        delivery_rows = await self._fetch(
            """
            SELECT delivery.id
            FROM admin_webhook_deliveries AS delivery
            WHERE delivery.state IN ('succeeded', 'dead', 'canceled', 'superseded')
              AND delivery.terminal_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM admin_webhook_deliveries AS redelivery
                  WHERE redelivery.redelivery_of_id = delivery.id
              )
            ORDER BY delivery.terminal_at ASC, delivery.id ASC
            LIMIT ?
            """,
            (retention_cutoff, remaining),
        ) if remaining else []
        deliveries = await self._delete_by_ids(
            "admin_webhook_deliveries",
            [row["id"] for row in delivery_rows],
        )
        remaining -= deliveries

        event_rows = await self._fetch(
            """
            SELECT event.id
            FROM admin_webhook_events AS event
            WHERE event.created_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM admin_webhook_deliveries AS delivery
                  WHERE delivery.event_id = event.id
              )
            ORDER BY event.created_at ASC, event.id ASC
            LIMIT ?
            """,
            (retention_cutoff, remaining),
        ) if remaining else []
        events = await self._delete_by_ids(
            "admin_webhook_events",
            [row["id"] for row in event_rows],
        )
        remaining -= events

        heartbeat_rows = await self._fetch(
            """
            SELECT component, instance_id
            FROM admin_webhook_runtime_heartbeats
            WHERE heartbeat_at <= ?
            ORDER BY heartbeat_at ASC, component ASC, instance_id ASC
            LIMIT ?
            """,
            (retention_cutoff, remaining),
        ) if remaining else []
        heartbeats = 0
        for row in heartbeat_rows:
            heartbeats += await self._execute(
                """
                DELETE FROM admin_webhook_runtime_heartbeats
                WHERE component = ? AND instance_id = ? AND heartbeat_at <= ?
                """,
                (row["component"], row["instance_id"], retention_cutoff),
            )
        remaining -= heartbeats

        registration_ids = (
            await self.find_purge_eligible_registration_ids(now=now, limit=remaining)
            if remaining
            else []
        )
        registrations = await self._delete_by_ids(
            "admin_webhook_registrations",
            registration_ids,
        )
        return RetentionBatchResult(
            expired_idempotency=expired_idempotency,
            deliveries=deliveries,
            events=events,
            heartbeats=heartbeats,
            registrations=registrations,
        )

    async def allocate_registration_id(self) -> int:
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_sequences
            SET next_value = next_value + 1
            WHERE name = ?
            RETURNING next_value - 1 AS allocated_id
            """,
            ("registration",),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)
        return int(row["allocated_id"])

    async def ensure_registration_sequence_above(self, webhook_id: int) -> int:
        """Advance the allocator past one imported ID without moving it backward."""
        if isinstance(webhook_id, bool) or not 1 <= webhook_id < 2**63 - 1:
            raise ValueError("imported webhook ID cannot be followed by a positive ID")
        next_value = webhook_id + 1
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_sequences
            SET next_value = CASE
                WHEN next_value < ? THEN ?
                ELSE next_value
            END
            WHERE name = ?
            RETURNING next_value
            """,
            (next_value, next_value, "registration"),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)
        return int(row["next_value"])

    async def get_legacy_import_snapshot(
        self,
        *,
        lock: bool,
    ) -> LegacyImportDatabaseSnapshot:
        """Read one fixed-table legacy snapshot for planning or import validation."""
        if self._is_postgres:
            table_row = await self._fetchrow(
                "SELECT to_regclass(?) AS table_name",
                ("public.admin_webhooks",),
            )
            table_present = bool(table_row and table_row["table_name"] is not None)
        else:
            table_row = await self._fetchrow(
                """
                SELECT name AS table_name
                FROM sqlite_master
                WHERE type = ? AND name = ?
                """,
                ("table", "admin_webhooks"),
            )
            table_present = table_row is not None

        raw_rows: list[dict[str, Any]] = []
        if table_present:
            legacy_query = "SELECT * FROM admin_webhooks ORDER BY id ASC"
            if lock and self._is_postgres:
                legacy_query += " FOR SHARE"
            raw_rows = await self._fetch(legacy_query)

        canonical_query = "SELECT id, deleted_at FROM admin_webhook_registrations ORDER BY id ASC"
        if lock and self._is_postgres:
            canonical_query += " FOR SHARE"
        canonical_rows = await self._fetch(canonical_query)

        sequence_query = (
            "SELECT next_value FROM admin_webhook_sequences WHERE name = ?"
        )
        if lock and self._is_postgres:
            sequence_query += " FOR UPDATE"
        sequence_row = await self._fetchrow(sequence_query, ("registration",))
        if sequence_row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)

        rows: list[LegacyWebhookRow] = []
        for position, raw_row in enumerate(raw_rows):
            raw_identity = raw_row.get("id")
            if isinstance(raw_identity, bool):
                source_identity = "true" if raw_identity else "false"
            elif raw_identity is None:
                source_identity = "null"
            elif isinstance(raw_identity, (int, str)):
                source_identity = str(raw_identity)
            else:
                source_identity = f"row-{position + 1}"
            rows.append(
                LegacyWebhookRow(
                    source_identity=source_identity,
                    values=raw_row,
                )
            )

        canonical_ids = tuple(int(row["id"]) for row in canonical_rows)
        return LegacyImportDatabaseSnapshot(
            table_present=table_present,
            rows=tuple(rows),
            canonical_registration_ids=canonical_ids,
            canonical_non_deleted_count=sum(row["deleted_at"] is None for row in canonical_rows),
            next_registration_id=int(sequence_row["next_value"]),
        )

    async def insert_registration(
        self,
        registration: RegistrationInsert,
    ) -> WebhookRegistration:
        _utc_datetime(registration.now, field="now")
        _validate_registration_target(registration.target)
        if registration.id < 1 or registration.actor_user_id < 1:
            raise ValueError("registration and actor IDs must be positive")
        row = await self._fetchrow(
            """
            INSERT INTO admin_webhook_registrations (
                id, description, target_ciphertext_json, target_key_id,
                target_hostname, target_display, event_types_json, active,
                timeout_seconds, secret_ciphertext_json, secret_key_id,
                secret_rotation_required, created_by_user_id,
                updated_by_user_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING *
            """,
            (
                registration.id,
                registration.description,
                registration.target.protected.ciphertext_json,
                registration.target.protected.key_id,
                registration.target.hostname,
                registration.target.display,
                _canonical_json(list(registration.event_types)),
                registration.active,
                registration.timeout_seconds,
                registration.secret.ciphertext_json,
                registration.secret.key_id,
                registration.secret_rotation_required,
                registration.actor_user_id,
                registration.actor_user_id,
                registration.now,
                registration.now,
            ),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)
        return _registration_from_row(row)

    async def _registration_row(
        self,
        webhook_id: int,
        *,
        include_deleted: bool,
        lock: bool = False,
    ) -> dict[str, Any] | None:
        query = "SELECT * FROM admin_webhook_registrations WHERE id = ?"
        if not include_deleted:
            query += " AND deleted_at IS NULL"
        if lock and self._is_postgres:
            query += " FOR UPDATE"
        return await self._fetchrow(query, (webhook_id,))

    async def get_protected_registration(
        self,
        webhook_id: int,
        *,
        include_deleted: bool = False,
        lock: bool = False,
    ) -> StoredWebhookRegistration | None:
        row = await self._registration_row(
            webhook_id,
            include_deleted=include_deleted,
            lock=lock,
        )
        return _stored_registration_from_row(row) if row is not None else None

    async def list_registrations(
        self,
        *,
        limit: int,
        before_id: int | None = None,
        offset: int = 0,
        include_deleted: bool = False,
    ) -> list[WebhookRegistration]:
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if not 0 <= offset <= 1_000:
            raise ValueError("offset must be between 0 and 1000")
        clauses: list[str] = []
        params: list[object] = []
        if not include_deleted:
            clauses.append("deleted_at IS NULL")
        if before_id is not None:
            if before_id < 1:
                raise ValueError("before_id must be positive")
            clauses.append("id < ?")
            params.append(before_id)
        query = "SELECT * FROM admin_webhook_registrations"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params.extend((limit, offset))
        return [_registration_from_row(row) for row in await self._fetch(query, params)]

    async def count_registrations(self) -> int:
        row = await self._fetchrow(
            """
            SELECT COUNT(*) AS count
            FROM admin_webhook_registrations
            WHERE deleted_at IS NULL
            """
        )
        return int(row["count"]) if row is not None else 0

    async def count_active_registrations(self) -> int:
        row = await self._fetchrow(
            """
            SELECT COUNT(*) AS count
            FROM admin_webhook_registrations
            WHERE deleted_at IS NULL AND active = ?
            """,
            (True,),
        )
        return int(row["count"]) if row is not None else 0

    async def count_secret_rotation_required(self) -> int:
        row = await self._fetchrow(
            """
            SELECT COUNT(*) AS count
            FROM admin_webhook_registrations
            WHERE deleted_at IS NULL AND secret_rotation_required = ?
            """,
            (True,),
        )
        return int(row["count"]) if row is not None else 0

    async def page_protected_rows(
        self,
        *,
        table: str,
        after: str | None,
        limit: int,
        inventory_at: datetime | None = None,
    ) -> list[ProtectedRow]:
        """Return a deterministic page from one explicitly known table/field."""
        if table not in _DATABASE_PROTECTED_TABLES:
            raise ValueError("protected row table is invalid")
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        if inventory_at is not None:
            inventory_at = _utc_datetime(inventory_at, field="inventory_at")

        if table in {
            "registration_targets",
            "registration_secrets",
            "idempotency_replay_secrets",
        }:
            if after is None:
                after_id = 0
            elif not isinstance(after, str) or not after.isdigit() or int(after) < 1:
                raise ValueError("protected row cursor is invalid")
            else:
                after_id = int(after)

        if table == "registration_targets":
            rows = await self._fetch(
                """
                SELECT id, target_ciphertext_json, target_key_id, target_version
                FROM admin_webhook_registrations
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, limit),
            )
            return [
                ProtectedRow(
                    table=table,
                    row_identity=str(row["id"]),
                    field="target",
                    protected=ProtectedValue(
                        ciphertext_json=str(row["target_ciphertext_json"]),
                        key_id=str(row["target_key_id"]),
                    ),
                    purpose="registration.target",
                    envelope_identity={
                        "registration_id": int(row["id"]),
                        "target_version": int(row["target_version"]),
                    },
                )
                for row in rows
            ]

        if table == "registration_secrets":
            rows = await self._fetch(
                """
                SELECT id, secret_ciphertext_json, secret_key_id, secret_version
                FROM admin_webhook_registrations
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, limit),
            )
            return [
                ProtectedRow(
                    table=table,
                    row_identity=str(row["id"]),
                    field="secret",
                    protected=ProtectedValue(
                        ciphertext_json=str(row["secret_ciphertext_json"]),
                        key_id=str(row["secret_key_id"]),
                    ),
                    purpose="registration.secret",
                    envelope_identity={
                        "registration_id": int(row["id"]),
                        "secret_version": int(row["secret_version"]),
                    },
                )
                for row in rows
            ]

        if table == "event_bodies":
            if after is not None and (not isinstance(after, str) or not after):
                raise ValueError("protected row cursor is invalid")
            rows = await self._fetch(
                """
                SELECT id, api_version, body_ciphertext_json, body_key_id
                FROM admin_webhook_events
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after or "", limit),
            )
            return [
                ProtectedRow(
                    table=table,
                    row_identity=str(row["id"]),
                    field="body",
                    protected=ProtectedValue(
                        ciphertext_json=str(row["body_ciphertext_json"]),
                        key_id=str(row["body_key_id"]),
                    ),
                    purpose="event.body",
                    envelope_identity={
                        "event_id": str(row["id"]),
                        "api_version": str(row["api_version"]),
                    },
                )
                for row in rows
            ]

        if inventory_at is None:
            rows = await self._fetch(
                """
                SELECT id, lookup_digest, resource_id, secret_version,
                       replay_secret_ciphertext_json, replay_secret_key_id
                FROM admin_webhook_idempotency
                WHERE id > ?
                  AND state = 'completed'
                  AND replay_secret_ciphertext_json IS NOT NULL
                  AND replay_secret_key_id IS NOT NULL
                  AND resource_id IS NOT NULL
                  AND secret_version IS NOT NULL
                  AND expires_at > CURRENT_TIMESTAMP
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, limit),
            )
        else:
            rows = await self._fetch(
                """
                SELECT id, lookup_digest, resource_id, secret_version,
                       replay_secret_ciphertext_json, replay_secret_key_id
                FROM admin_webhook_idempotency
                WHERE id > ?
                  AND state = 'completed'
                  AND replay_secret_ciphertext_json IS NOT NULL
                  AND replay_secret_key_id IS NOT NULL
                  AND resource_id IS NOT NULL
                  AND secret_version IS NOT NULL
                  AND expires_at > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (after_id, inventory_at, limit),
            )
        return [
            ProtectedRow(
                table=table,
                row_identity=str(row["id"]),
                field="replay_secret",
                protected=ProtectedValue(
                    ciphertext_json=str(row["replay_secret_ciphertext_json"]),
                    key_id=str(row["replay_secret_key_id"]),
                ),
                purpose="idempotency.secret_replay",
                envelope_identity={
                    "lookup_digest": str(row["lookup_digest"]),
                    "registration_id": int(row["resource_id"]),
                    "secret_version": int(row["secret_version"]),
                },
            )
            for row in rows
        ]

    async def replace_protected_value(
        self,
        row: ProtectedRow,
        *,
        expected_ciphertext: str,
        replacement: ProtectedValue,
    ) -> bool:
        """Replace one known envelope only if its full persisted identity matches."""
        if not isinstance(row, ProtectedRow) or not isinstance(replacement, ProtectedValue):
            raise TypeError("protected row and replacement are required")
        if not isinstance(expected_ciphertext, str) or not hmac.compare_digest(
            expected_ciphertext,
            row.protected.ciphertext_json,
        ):
            raise ValueError("expected protected ciphertext is invalid")

        identity = row.envelope_identity
        if row.table == "registration_targets":
            changed = await self._execute(
                """
                UPDATE admin_webhook_registrations
                SET target_ciphertext_json = ?, target_key_id = ?
                WHERE id = ?
                  AND target_version = ?
                  AND target_ciphertext_json = ?
                  AND target_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    int(row.row_identity),
                    identity["target_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        elif row.table == "registration_secrets":
            changed = await self._execute(
                """
                UPDATE admin_webhook_registrations
                SET secret_ciphertext_json = ?, secret_key_id = ?
                WHERE id = ?
                  AND secret_version = ?
                  AND secret_ciphertext_json = ?
                  AND secret_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    int(row.row_identity),
                    identity["secret_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        elif row.table == "event_bodies":
            changed = await self._execute(
                """
                UPDATE admin_webhook_events
                SET body_ciphertext_json = ?, body_key_id = ?
                WHERE id = ?
                  AND api_version = ?
                  AND body_ciphertext_json = ?
                  AND body_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    row.row_identity,
                    identity["api_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        else:
            changed = await self._execute(
                """
                UPDATE admin_webhook_idempotency
                SET replay_secret_ciphertext_json = ?, replay_secret_key_id = ?
                WHERE id = ?
                  AND lookup_digest = ?
                  AND resource_id = ?
                  AND secret_version = ?
                  AND replay_secret_ciphertext_json = ?
                  AND replay_secret_key_id = ?
                """,
                (
                    replacement.ciphertext_json,
                    replacement.key_id,
                    int(row.row_identity),
                    identity["lookup_digest"],
                    identity["registration_id"],
                    identity["secret_version"],
                    expected_ciphertext,
                    row.protected.key_id,
                ),
            )
        if changed not in {0, 1}:
            raise RuntimeError("protected value compare-and-set affected multiple rows")
        return changed == 1

    async def _lock_registration_admission(self) -> None:
        query = (
            "SELECT next_value FROM admin_webhook_sequences WHERE name = ? FOR UPDATE"
            if self._is_postgres
            else "SELECT next_value FROM admin_webhook_sequences WHERE name = ?"
        )
        if await self._fetchrow(query, ("registration",)) is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.SEQUENCE_UNAVAILABLE)

    async def registration_limit_state(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        if limit < 1:
            raise ValueError("limit must be positive")
        current = await self.count_registrations()
        return RegistrationLimitState(
            current=current,
            limit=limit,
            at_limit=current >= limit,
            over_limit=current > limit,
        )

    async def active_registration_limit_state(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        if limit < 1:
            raise ValueError("limit must be positive")
        current = await self.count_active_registrations()
        return RegistrationLimitState(
            current=current,
            limit=limit,
            at_limit=current >= limit,
            over_limit=current > limit,
        )

    async def enforce_registration_limit(self, *, limit: int) -> RegistrationLimitState:
        """Serialize admission and reject create when no registration slot remains."""
        await self._lock_registration_admission()
        state = await self.registration_limit_state(limit=limit)
        if state.at_limit:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.REGISTRATION_LIMIT)
        return state

    async def enforce_active_registration_limit(
        self,
        *,
        limit: int,
    ) -> RegistrationLimitState:
        """Serialize admission and reject activation when no active slot remains."""
        await self._lock_registration_admission()
        state = await self.active_registration_limit_state(limit=limit)
        if state.at_limit:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.ACTIVE_LIMIT)
        return state

    async def patch_registration(
        self,
        webhook_id: int,
        *,
        expected_revision: int,
        patch: RegistrationPatch,
        actor_user_id: int,
        at: datetime,
    ) -> RegistrationPatchResult:
        at = _utc_datetime(at, field="at")
        if webhook_id < 1 or expected_revision < 1 or actor_user_id < 1:
            raise ValueError("IDs and expected revision must be positive")
        current_row = await self._registration_row(
            webhook_id,
            include_deleted=False,
            lock=True,
        )
        if current_row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.NOT_FOUND)
        current = _stored_registration_from_row(current_row)
        if current.registration.revision != expected_revision:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.STALE_REVISION)

        assignments: list[str] = []
        params: list[object] = []
        changes_delivery_config = False
        changes_target = False
        changes_secret = False

        def assign(column: str, value: object) -> None:
            assignments.append(f"{column} = ?")
            params.append(value)

        if isinstance(patch.description, str):
            if patch.description != current.registration.description:
                assign("description", patch.description)
        elif patch.description is not UNSET:
            raise TypeError("description patch is invalid")

        if isinstance(patch.target, RegistrationTarget):
            _validate_registration_target(patch.target)
            target_changed = (
                patch.target.protected != current.target
                or patch.target.hostname != current.registration.target_hostname
                or patch.target.display != current.registration.target_display
            )
            if target_changed:
                assign("target_ciphertext_json", patch.target.protected.ciphertext_json)
                assign("target_key_id", patch.target.protected.key_id)
                assign("target_hostname", patch.target.hostname)
                assign("target_display", patch.target.display)
                changes_delivery_config = True
                changes_target = True
        elif patch.target is not UNSET:
            raise TypeError("target patch is invalid")

        if isinstance(patch.event_types, tuple):
            if any(not isinstance(item, str) for item in patch.event_types):
                raise TypeError("event types patch is invalid")
            if patch.event_types != current.registration.event_types:
                assign("event_types_json", _canonical_json(list(patch.event_types)))
                changes_delivery_config = True
        elif patch.event_types is not UNSET:
            raise TypeError("event types patch is invalid")

        if isinstance(patch.active, bool):
            if patch.active != current.registration.active:
                assign("active", patch.active)
                changes_delivery_config = True
        elif patch.active is not UNSET:
            raise TypeError("active patch is invalid")

        if isinstance(patch.timeout_seconds, int) and not isinstance(
            patch.timeout_seconds,
            bool,
        ):
            if patch.timeout_seconds != current.registration.timeout_seconds:
                assign("timeout_seconds", patch.timeout_seconds)
                changes_delivery_config = True
        elif patch.timeout_seconds is not UNSET:
            raise TypeError("timeout patch is invalid")

        if isinstance(patch.secret, ProtectedValue):
            if patch.secret != current.secret:
                assign("secret_ciphertext_json", patch.secret.ciphertext_json)
                assign("secret_key_id", patch.secret.key_id)
                changes_delivery_config = True
                changes_secret = True
        elif patch.secret is not UNSET:
            raise TypeError("secret patch is invalid")

        if isinstance(patch.secret_rotation_required, bool):
            if patch.secret_rotation_required != current.registration.secret_rotation_required:
                if patch.secret_rotation_required is False and not changes_secret:
                    raise ValueError("secret rotation requirement cannot be cleared without a new secret")
                assign("secret_rotation_required", patch.secret_rotation_required)
        elif patch.secret_rotation_required is not UNSET:
            raise TypeError("secret rotation patch is invalid")

        if not assignments:
            return RegistrationPatchResult(
                changed=False,
                registration=current.registration,
            )

        assignments.extend(["revision = revision + 1", "updated_by_user_id = ?", "updated_at = ?"])
        params.extend([actor_user_id, at])
        if changes_delivery_config:
            assignments.append("delivery_config_version = delivery_config_version + 1")
        if changes_target:
            assignments.append("target_version = target_version + 1")
        if changes_secret:
            assignments.append("secret_version = secret_version + 1")
        params.extend([webhook_id, expected_revision])
        # Identifiers are selected only by the closed patch fields above.
        update_query = " ".join(
            (
                "UPDATE admin_webhook_registrations SET",
                ", ".join(assignments),
                "WHERE id = ? AND revision = ? AND deleted_at IS NULL RETURNING *",
            )
        )
        row = await self._fetchrow(update_query, params)
        if row is None:
            latest = await self._registration_row(
                webhook_id,
                include_deleted=False,
                lock=False,
            )
            code = WebhookRepositoryErrorCode.NOT_FOUND if latest is None else WebhookRepositoryErrorCode.STALE_REVISION
            raise WebhookRepositoryError(code)
        return RegistrationPatchResult(
            changed=True,
            registration=_registration_from_row(row),
        )

    async def soft_delete_registration(
        self,
        webhook_id: int,
        *,
        expected_revision: int,
        actor_user_id: int,
        at: datetime,
    ) -> WebhookRegistration:
        at = _utc_datetime(at, field="at")
        if webhook_id < 1 or expected_revision < 1 or actor_user_id < 1:
            raise ValueError("IDs and expected revision must be positive")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_registrations
            SET active = ?, deleted_at = ?, deleted_by_user_id = ?,
                updated_by_user_id = ?, updated_at = ?, revision = revision + 1,
                delivery_config_version = delivery_config_version + 1
            WHERE id = ? AND revision = ? AND deleted_at IS NULL
            RETURNING *
            """,
            (
                False,
                at,
                actor_user_id,
                actor_user_id,
                at,
                webhook_id,
                expected_revision,
            ),
        )
        if row is None:
            latest = await self._registration_row(
                webhook_id,
                include_deleted=False,
                lock=False,
            )
            code = WebhookRepositoryErrorCode.NOT_FOUND if latest is None else WebhookRepositoryErrorCode.STALE_REVISION
            raise WebhookRepositoryError(code)
        return _registration_from_row(row)

    async def claim_idempotency(
        self,
        *,
        lookup_digest: str,
        scope: IdempotencyScope,
        request_fingerprint: str,
        now: datetime,
        expires_at: datetime,
    ) -> IdempotencyLookup:
        now = _utc_datetime(now, field="now")
        expires_at = _utc_datetime(expires_at, field="expires_at")
        if expires_at <= now:
            raise ValueError("expires_at must be after now")
        if _LOOKUP_DIGEST.fullmatch(lookup_digest) is None:
            raise ValueError("lookup digest is invalid")
        if _REQUEST_FINGERPRINT.fullmatch(request_fingerprint) is None:
            raise ValueError("request fingerprint is invalid")
        lookup_query = (
            "SELECT * FROM admin_webhook_idempotency WHERE lookup_digest = ? FOR UPDATE"
            if self._is_postgres
            else "SELECT * FROM admin_webhook_idempotency WHERE lookup_digest = ?"
        )
        row = await self._fetchrow(
            lookup_query,
            (lookup_digest,),
        )
        if row is not None:
            row_expires_at = _parse_datetime(row["expires_at"])
            if row_expires_at is None:
                raise ValueError("persisted idempotency expiry is invalid")
            if row_expires_at <= now:
                await self._execute(
                    "DELETE FROM admin_webhook_idempotency WHERE id = ?",
                    (row["id"],),
                )
                row = None
        if row is None:
            row = await self._fetchrow(
                """
                INSERT INTO admin_webhook_idempotency (
                    lookup_digest, actor_id, operation, route, webhook_id,
                    delivery_id, request_fingerprint, state, created_at,
                    updated_at, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'in_progress', ?, ?, ?)
                ON CONFLICT (lookup_digest) DO NOTHING
                RETURNING *
                """,
                (
                    lookup_digest,
                    scope.actor_id,
                    scope.operation,
                    scope.route,
                    scope.webhook_id,
                    scope.delivery_id,
                    request_fingerprint,
                    now,
                    now,
                    expires_at,
                ),
            )
            if row is not None:
                return IdempotencyLookup(
                    kind=IdempotencyLookupKind.NEW,
                    expires_at=expires_at,
                )
            row = await self._fetchrow(
                lookup_query,
                (lookup_digest,),
            )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.IDEMPOTENCY_NOT_CLAIMED)
        return await self._idempotency_lookup_from_row(
            row,
            scope=scope,
            request_fingerprint=request_fingerprint,
        )

    async def _idempotency_lookup_from_row(
        self,
        row: Mapping[str, Any],
        *,
        scope: IdempotencyScope,
        request_fingerprint: str,
    ) -> IdempotencyLookup:
        scope_matches = (
            hmac.compare_digest(str(row["actor_id"]), scope.actor_id)
            and hmac.compare_digest(str(row["operation"]), scope.operation)
            and hmac.compare_digest(str(row["route"]), scope.route)
            and row["webhook_id"] == scope.webhook_id
            and row["delivery_id"] == scope.delivery_id
        )
        if not scope_matches or not hmac.compare_digest(
            str(row["request_fingerprint"]),
            request_fingerprint,
        ):
            return IdempotencyLookup(kind=IdempotencyLookupKind.CONFLICT)
        if str(row["state"]) == "in_progress":
            return IdempotencyLookup(
                kind=IdempotencyLookupKind.IN_PROGRESS,
                expires_at=_parse_datetime(row["expires_at"]),
            )

        replay_secret = None
        if row["replay_secret_ciphertext_json"] is not None:
            replay_secret = ProtectedValue(
                ciphertext_json=str(row["replay_secret_ciphertext_json"]),
                key_id=str(row["replay_secret_key_id"]),
            )
        metadata: Mapping[str, object] | None = None
        if row["response_metadata_json"] is not None:
            metadata = MappingProxyType(_strict_json_object(row["response_metadata_json"]))
        resource_id = int(row["resource_id"]) if row["resource_id"] is not None else None
        resource_version = int(row["resource_version"]) if row["resource_version"] is not None else None
        secret_version = int(row["secret_version"]) if row["secret_version"] is not None else None
        superseded = False
        if replay_secret is not None and resource_id is not None:
            current = await self._registration_row(
                resource_id,
                include_deleted=True,
                lock=False,
            )
            superseded = (
                current is None
                or current["deleted_at"] is not None
                or (secret_version is not None and int(current["secret_version"]) != secret_version)
            )
        return IdempotencyLookup(
            kind=IdempotencyLookupKind.REPLAY,
            resource_id=resource_id,
            resource_version=resource_version,
            secret_version=secret_version,
            replay_secret=replay_secret,
            test_delivery_id=(str(row["test_delivery_id"]) if row["test_delivery_id"] is not None else None),
            test_attempt_id=(str(row["test_attempt_id"]) if row["test_attempt_id"] is not None else None),
            response_status=(int(row["response_status"]) if row["response_status"] is not None else None),
            response_metadata=metadata,
            resource_superseded=superseded,
            expires_at=_parse_datetime(row["expires_at"]),
        )

    async def complete_idempotency(
        self,
        *,
        lookup_digest: str,
        request_fingerprint: str,
        resource_id: int | None,
        resource_version: int | None,
        secret_version: int | None = None,
        replay_secret: ProtectedValue | None = None,
        test_delivery_id: str | None = None,
        test_attempt_id: str | None = None,
        response_status: int,
        response_metadata: Mapping[str, object] | None,
        at: datetime,
    ) -> IdempotencyLookup:
        at = _utc_datetime(at, field="at")
        metadata_json, _ = _safe_response_metadata(response_metadata)
        if _LOOKUP_DIGEST.fullmatch(lookup_digest) is None:
            raise ValueError("lookup digest is invalid")
        if _REQUEST_FINGERPRINT.fullmatch(request_fingerprint) is None:
            raise ValueError("request fingerprint is invalid")
        if (resource_id is None) != (resource_version is None):
            raise ValueError("resource ID and version must be paired")
        if resource_id is not None:
            if resource_version is None or resource_id < 1 or resource_version < 1:
                raise ValueError("resource ID and version must be positive")
        if not 100 <= response_status <= 599:
            raise ValueError("response status is invalid")
        if (replay_secret is None) != (secret_version is None):
            raise ValueError("replay secret and secret version must be paired")
        if (test_delivery_id is None) != (test_attempt_id is None):
            raise ValueError("test delivery and attempt IDs must be paired")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_idempotency
            SET state = 'completed', resource_id = ?, resource_version = ?,
                secret_version = ?, replay_secret_ciphertext_json = ?,
                replay_secret_key_id = ?, test_delivery_id = ?,
                test_attempt_id = ?, response_status = ?,
                response_metadata_json = ?, updated_at = ?
            WHERE lookup_digest = ? AND request_fingerprint = ?
              AND state = 'in_progress'
            RETURNING *
            """,
            (
                resource_id,
                resource_version,
                secret_version,
                replay_secret.ciphertext_json if replay_secret is not None else None,
                replay_secret.key_id if replay_secret is not None else None,
                test_delivery_id,
                test_attempt_id,
                response_status,
                metadata_json,
                at,
                lookup_digest,
                request_fingerprint,
            ),
        )
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.IDEMPOTENCY_COMPLETION_INVALID)
        scope = IdempotencyScope(
            actor_id=str(row["actor_id"]),
            operation=str(row["operation"]),
            route=str(row["route"]),
            webhook_id=(int(row["webhook_id"]) if row["webhook_id"] is not None else None),
            delivery_id=(str(row["delivery_id"]) if row["delivery_id"] is not None else None),
        )
        return await self._idempotency_lookup_from_row(
            row,
            scope=scope,
            request_fingerprint=request_fingerprint,
        )

    async def get_migration_state(self, *, lock: bool = False) -> MigrationState:
        query = "SELECT * FROM admin_webhook_migration_state WHERE singleton_id = 1"
        if lock and self._is_postgres:
            query += " FOR UPDATE"
        row = await self._fetchrow(query)
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.MIGRATION_STATE_UNAVAILABLE)
        return _migration_state_from_row(row)

    async def lock_migration_state(self) -> MigrationState:
        return await self.get_migration_state(lock=True)

    async def compare_and_set_migration_state(
        self,
        *,
        expected_revision: int,
        updates: Mapping[str, object],
        at: datetime,
    ) -> MigrationState:
        at = _utc_datetime(at, field="at")
        if expected_revision < 1 or not updates:
            raise ValueError("expected revision and updates are required")
        invalid = set(updates) - _MIGRATION_MUTABLE_COLUMNS
        if invalid:
            raise ValueError("migration state update contains invalid fields")
        assignments: list[str] = []
        params: list[object] = []
        for column in sorted(updates):
            value = updates[column]
            expected_type = _MIGRATION_JSON_COLUMNS.get(column)
            if expected_type is not None:
                if not isinstance(value, expected_type):
                    raise ValueError(f"{column} has an invalid JSON shape")
                value = _canonical_json(value)
            assignments.append(f"{column} = ?")
            params.append(value)
        assignments.extend(["state_revision = state_revision + 1", "updated_at = ?"])
        params.extend([at, expected_revision])
        # Identifiers are admitted only from _MIGRATION_MUTABLE_COLUMNS.
        update_query = " ".join(
            (
                "UPDATE admin_webhook_migration_state SET",
                ", ".join(assignments),
                "WHERE singleton_id = 1 AND state_revision = ? RETURNING *",
            )
        )
        row = await self._fetchrow(update_query, params)
        if row is None:
            raise WebhookRepositoryError(WebhookRepositoryErrorCode.STALE_MIGRATION_STATE)
        return _migration_state_from_row(row)

    async def mark_first_canonical_activity(
        self,
        kind: str,
        at: datetime,
    ) -> MigrationState:
        if kind not in _ACTIVITY_KINDS:
            raise ValueError("activity kind is invalid")
        at = _utc_datetime(at, field="at")
        row = await self._fetchrow(
            """
            UPDATE admin_webhook_migration_state
            SET first_canonical_activity_at = ?,
                first_canonical_activity_kind = ?,
                state_revision = state_revision + 1,
                updated_at = ?
            WHERE singleton_id = 1
              AND first_canonical_activity_at IS NULL
              AND first_canonical_activity_kind IS NULL
            RETURNING *
            """,
            (at, kind, at),
        )
        if row is not None:
            return _migration_state_from_row(row)
        return await self.get_migration_state(lock=True)

    async def find_purge_eligible_registration_ids(
        self,
        *,
        now: datetime,
        limit: int,
    ) -> list[int]:
        now = _utc_datetime(now, field="now")
        if not 1 <= limit <= _MAX_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
        state = await self.get_migration_state(lock=False)
        referenced = _referenced_registration_ids(state.source_mapping)
        cutoff = now - timedelta(days=_MIN_TOMBSTONE_RETENTION_DAYS)
        result: list[int] = []
        after_id = 0
        batch_size = min(_MAX_PAGE_SIZE, max(50, limit * 2))
        while len(result) < limit:
            rows = await self._fetch(
                """
                SELECT registration.id
                FROM admin_webhook_registrations AS registration
                WHERE registration.id > ?
                  AND registration.deleted_at IS NOT NULL
                  AND registration.deleted_at <= ?
                  AND NOT EXISTS (
                      SELECT 1 FROM admin_webhook_deliveries AS delivery
                      WHERE delivery.webhook_id = registration.id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM admin_webhook_idempotency AS command
                      WHERE command.expires_at > ?
                        AND (
                            command.webhook_id = registration.id
                            OR command.resource_id = registration.id
                        )
                  )
                ORDER BY registration.id ASC
                LIMIT ?
                """,
                (after_id, cutoff, now, batch_size),
            )
            if not rows:
                break
            for row in rows:
                webhook_id = int(row["id"])
                if webhook_id not in referenced:
                    result.append(webhook_id)
                    if len(result) == limit:
                        break
            after_id = int(rows[-1]["id"])
            if len(rows) < batch_size:
                break
        return result


__all__ = [
    "AdminWebhookRepository",
    "AdminWebhookUnitOfWork",
    "IdempotencyLookup",
    "IdempotencyLookupKind",
    "MigrationState",
    "RegistrationInsert",
    "RegistrationLimitState",
    "RegistrationPatch",
    "RegistrationPatchResult",
    "RegistrationTarget",
    "StoredWebhookRegistration",
    "WebhookRepositoryError",
    "WebhookRepositoryErrorCode",
]
