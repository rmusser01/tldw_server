"""Pure domain records and stable helpers for canonical admin webhooks."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from urllib.parse import SplitResult, urlsplit

from tldw_Server_API.app.core.exceptions import WebhookError
from tldw_Server_API.app.core.Security.egress import (
    evaluate_platform_webhook_url_policy,
)

from .crypto import ProtectedValue


class WebhookErrorCode(str, Enum):
    """Closed canonical error codes used across webhook layers."""

    VALIDATION_FAILED = "admin_webhook_validation_failed"
    REQUEST_REJECTED = "admin_webhook_request_rejected"
    EVENT_UNSUPPORTED = "admin_webhook_event_unsupported"
    IDEMPOTENCY_KEY_INVALID = "admin_webhook_idempotency_key_invalid"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    IDEMPOTENCY_IN_PROGRESS = "idempotency_in_progress"
    IDEMPOTENCY_RESULT_SUPERSEDED = "idempotency_result_superseded"
    PRECONDITION_REQUIRED = "precondition_required"
    PRECONDITION_FAILED = "precondition_failed"
    TARGET_REJECTED = "admin_webhook_target_rejected"
    NOT_FOUND = "admin_webhook_not_found"
    DISABLED = "admin_webhooks_disabled"
    MODE_UNAVAILABLE = "admin_webhooks_disabled"
    MIGRATION_PENDING = "admin_webhook_migration_pending"
    REGISTRATION_LIMIT = "admin_webhook_registration_limit"
    ACTIVE_LIMIT = "admin_webhook_active_limit"
    # Stable public error code; this value is not a credential.
    SECRET_ROTATION_REQUIRED = "admin_webhook_secret_rotation_required"  # nosec B105
    REGISTRATION_ACTIVE = "admin_webhook_registration_active"
    KEY_UNAVAILABLE = "admin_webhook_key_unavailable"
    KEY_CONFIGURATION_MISMATCH = "admin_webhook_key_configuration_mismatch"
    KEY_ROTATION_IN_PROGRESS = "admin_webhook_key_rotation_in_progress"
    DATABASE_BUSY = "admin_webhook_database_busy"
    AUDIT_UNAVAILABLE = "admin_webhook_audit_unavailable"
    OPERATION_FAILED = "admin_webhook_operation_failed"
    USER_PRINCIPAL_REQUIRED = "admin_webhook_user_principal_required"
    DELIVERY_UNAVAILABLE = "admin_webhook_delivery_unavailable"
    TEST_DELIVERY_UNAVAILABLE = "admin_webhook_test_delivery_unavailable"
    REDELIVERY_CONFIRMATION_REQUIRED = "admin_webhook_redelivery_confirmation_required"
    DELIVERY_HISTORY_UNAVAILABLE = "admin_webhook_delivery_history_unavailable"
    RECOVERY_UNAVAILABLE = "admin_webhook_recovery_unavailable"

    @property
    def http_status(self) -> int:
        """Return the default HTTP status for this closed error code."""

        return _ERROR_STATUS[self]


_ERROR_STATUS = {
    WebhookErrorCode.VALIDATION_FAILED: 422,
    WebhookErrorCode.EVENT_UNSUPPORTED: 422,
    WebhookErrorCode.IDEMPOTENCY_KEY_INVALID: 422,
    WebhookErrorCode.IDEMPOTENCY_CONFLICT: 409,
    WebhookErrorCode.IDEMPOTENCY_IN_PROGRESS: 409,
    WebhookErrorCode.IDEMPOTENCY_RESULT_SUPERSEDED: 409,
    WebhookErrorCode.PRECONDITION_REQUIRED: 428,
    WebhookErrorCode.PRECONDITION_FAILED: 412,
    WebhookErrorCode.TARGET_REJECTED: 422,
    WebhookErrorCode.NOT_FOUND: 404,
    WebhookErrorCode.DISABLED: 503,
    WebhookErrorCode.MIGRATION_PENDING: 503,
    WebhookErrorCode.REGISTRATION_LIMIT: 409,
    WebhookErrorCode.ACTIVE_LIMIT: 409,
    WebhookErrorCode.SECRET_ROTATION_REQUIRED: 409,
    WebhookErrorCode.REGISTRATION_ACTIVE: 409,
    WebhookErrorCode.KEY_UNAVAILABLE: 503,
    WebhookErrorCode.KEY_CONFIGURATION_MISMATCH: 503,
    WebhookErrorCode.KEY_ROTATION_IN_PROGRESS: 503,
    WebhookErrorCode.DATABASE_BUSY: 503,
    WebhookErrorCode.AUDIT_UNAVAILABLE: 503,
    WebhookErrorCode.OPERATION_FAILED: 503,
    WebhookErrorCode.USER_PRINCIPAL_REQUIRED: 403,
    WebhookErrorCode.DELIVERY_UNAVAILABLE: 503,
    WebhookErrorCode.TEST_DELIVERY_UNAVAILABLE: 503,
    WebhookErrorCode.REDELIVERY_CONFIRMATION_REQUIRED: 428,
    WebhookErrorCode.DELIVERY_HISTORY_UNAVAILABLE: 503,
    WebhookErrorCode.RECOVERY_UNAVAILABLE: 503,
    WebhookErrorCode.REQUEST_REJECTED: 400,
}


class DeliveryKind(str, Enum):
    """Closed origin for one persisted webhook delivery."""

    AUTOMATIC = "automatic"
    MANUAL = "manual"
    TEST = "test"


class DeliveryState(str, Enum):
    """Closed state machine for persisted webhook delivery work."""

    PENDING = "pending"
    ENQUEUE_CLAIMED = "enqueue_claimed"
    QUEUED = "queued"
    PROCESSING = "processing"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    DEAD = "dead"
    CANCELED = "canceled"
    SUPERSEDED = "superseded"

    @classmethod
    def terminal_states(cls) -> frozenset[DeliveryState]:
        """Return terminal delivery states that never regress."""
        return frozenset({cls.SUCCEEDED, cls.DEAD, cls.CANCELED, cls.SUPERSEDED})


class AttemptState(str, Enum):
    """Closed state machine for one append-only attempt record."""

    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    RETRYABLE = "retryable"
    FAILED = "failed"
    CANCELED = "canceled"
    SUPERSEDED = "superseded"
    OUTCOME_UNKNOWN = "outcome_unknown"

    @classmethod
    def terminal_states(cls) -> frozenset[AttemptState]:
        """Return attempt states that no longer own network I/O."""
        return frozenset(set(cls) - {cls.PROCESSING})


class JobsDispositionKind(str, Enum):
    """Closed acknowledgement required from the Jobs scheduler."""

    COMPLETE = "complete"
    RETRY = "retry"
    FAIL = "fail"
    CANCEL = "cancel"
    DEFER = "defer"


class DeliveryReasonCode(str, Enum):
    """Stable bounded reasons retained in delivery and attempt metadata."""

    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    CANCELED_DELETED = "canceled_deleted"
    CANCELED_DISABLED = "canceled_disabled"
    CANCELED_SECRET_ROTATION = "canceled_secret_rotation"
    DELIVERY_EXPIRED = "delivery_expired"
    JOBS_IDENTITY_CONFLICT = "jobs_identity_conflict"
    OUTCOME_UNKNOWN = "outcome_unknown"
    SUPERSEDED_CONFIG = "superseded_config"
    TEST_ATTEMPT_INTERRUPTED = "test_attempt_interrupted"
    TARGET_INVALID = "target_invalid"
    TARGET_REJECTED = "target_rejected"
    POLICY_ERROR = "policy_error"
    CLOCK_ERROR = "clock_error"
    TRANSPORT_ERROR = "transport_error"
    HTTP_REDIRECT = "http_redirect"
    HTTP_CLIENT_ERROR = "http_client_error"
    HTTP_REQUEST_TIMEOUT = "http_request_timeout"
    HTTP_RATE_LIMITED = "http_rate_limited"
    HTTP_SERVER_ERROR = "http_server_error"
    HTTP_STATUS_INVALID = "http_status_invalid"
    HTTP_HOP_INVALID_REQUEST = "http_hop_invalid_request"
    HTTP_HOP_DNS_RESOLUTION_FAILED = "http_hop_dns_resolution_failed"
    HTTP_HOP_DNS_TIMEOUT = "http_hop_dns_timeout"
    HTTP_HOP_DNS_ADDRESS_DENIED = "http_hop_dns_address_denied"
    HTTP_HOP_CONNECT_TIMEOUT = "http_hop_connect_timeout"
    HTTP_HOP_READ_TIMEOUT = "http_hop_read_timeout"
    HTTP_HOP_WRITE_TIMEOUT = "http_hop_write_timeout"
    HTTP_HOP_TOTAL_TIMEOUT = "http_hop_total_timeout"
    HTTP_HOP_PEER_VERIFICATION_FAILED = "http_hop_peer_verification_failed"
    HTTP_HOP_TLS_ERROR = "http_hop_tls_error"
    HTTP_HOP_PROTOCOL_ERROR = "http_hop_protocol_error"
    HTTP_HOP_RESPONSE_HEADERS_TOO_LARGE = "http_hop_response_headers_too_large"
    HTTP_HOP_RESPONSE_TOO_LARGE = "http_hop_response_too_large"
    HTTP_HOP_DECOMPRESSED_RESPONSE_TOO_LARGE = (
        "http_hop_decompressed_response_too_large"
    )
    HTTP_HOP_PARSER_INPUT_TOO_LARGE = "http_hop_parser_input_too_large"
    HTTP_HOP_UNSUPPORTED_CONTENT_ENCODING = (
        "http_hop_unsupported_content_encoding"
    )
    HTTP_HOP_INVALID_CONTENT_ENCODING = "http_hop_invalid_content_encoding"
    HTTP_HOP_TRANSPORT_ERROR = "http_hop_transport_error"


class DeliveryRuntimeReasonCode(str, Enum):
    """Closed readiness reasons retained in runtime heartbeat metadata."""

    MODE_OFF = "mode_off"
    MODE_MIGRATE = "mode_migrate"
    SCHEMA_UNREADY = "schema_unready"
    MIGRATION_PENDING = "migration_pending"
    KEY_UNAVAILABLE = "key_unavailable"
    KEY_CONFIGURATION_MISMATCH = "key_configuration_mismatch"
    JOBS_UNAVAILABLE = "jobs_unavailable"
    DATABASE_UNAVAILABLE = "database_unavailable"
    WORKER_UNAVAILABLE = "worker_unavailable"
    RECONCILER_UNAVAILABLE = "reconciler_unavailable"
    RETENTION_UNAVAILABLE = "retention_unavailable"
    HEARTBEAT_STALE = "heartbeat_stale"


class EventSourceKind(str, Enum):
    """Closed source identity shape for an immutable webhook event."""

    AGGREGATE = "aggregate"
    COMMAND = "command"


class DeliveryRuntimeComponent(str, Enum):
    """Closed runtime components that write durable heartbeat evidence."""

    WORKER = "worker"
    RECONCILER = "reconciler"
    RETENTION = "retention"


@dataclass(frozen=True)
class WebhookEvent:
    """Public immutable event metadata without a body or encryption material."""

    id: str
    event_type: str
    api_version: str
    source_kind: EventSourceKind
    created_at: datetime


@dataclass(frozen=True)
class WebhookDelivery:
    """Sanitized delivery history metadata without destination or Jobs tokens."""

    id: str
    event_id: str
    webhook_id: int
    kind: DeliveryKind
    state: DeliveryState
    delivery_config_version: int
    secret_version: int
    attempt_count: int
    status_code: int | None
    latency_ms: int | None
    reason_code: DeliveryReasonCode | None
    expires_at: datetime
    created_at: datetime
    updated_at: datetime
    terminal_at: datetime | None = None
    redelivery_of_id: str | None = None


@dataclass(frozen=True)
class WebhookDeliveryAttempt:
    """Sanitized append-only attempt metadata without network request material."""

    id: str
    delivery_id: str
    attempt_number: int
    state: AttemptState
    request_timeout_seconds: int | None
    status_code: int | None
    latency_ms: int | None
    reason_code: DeliveryReasonCode | None
    requested_retry_delay_seconds: int | None
    started_at: datetime
    finished_at: datetime | None


@dataclass(frozen=True)
class DeliveryHistoryItem:
    """One sanitized delivery with its ordered append-only attempts."""

    delivery: WebhookDelivery
    event_type: str
    completed_after_config_change: bool
    attempts: tuple[WebhookDeliveryAttempt, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.delivery, WebhookDelivery):
            raise TypeError("history delivery is invalid")
        if not isinstance(self.event_type, str) or not 1 <= len(self.event_type) <= 64:
            raise ValueError("history event type is invalid")
        if not isinstance(self.completed_after_config_change, bool):
            raise TypeError("history configuration-change state is invalid")
        if not isinstance(self.attempts, tuple):
            raise TypeError("history attempts are invalid")
        numbers = tuple(attempt.attempt_number for attempt in self.attempts)
        if (
            any(
                not isinstance(attempt, WebhookDeliveryAttempt)
                or attempt.delivery_id != self.delivery.id
                for attempt in self.attempts
            )
            or numbers != tuple(sorted(numbers))
            or len(set(numbers)) != len(numbers)
        ):
            raise ValueError("history attempts are invalid")

    @property
    def id(self) -> str:
        """Return the delivery ID for legacy internal callers."""
        return self.delivery.id

    @property
    def kind(self) -> DeliveryKind:
        """Return the delivery kind for legacy internal callers."""
        return self.delivery.kind


@dataclass(frozen=True)
class DeliveryHistoryPage:
    """Bounded page of sanitized delivery-history items."""

    items: tuple[DeliveryHistoryItem, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True)
class DeliveryRuntimeHeartbeat:
    """Durable per-instance readiness metadata without host or error text."""

    component: DeliveryRuntimeComponent
    instance_id: str
    ready: bool
    reason_code: DeliveryRuntimeReasonCode | None
    heartbeat_at: datetime
    last_success_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class DeliveryComponentStatus:
    """Sanitized readiness for one runtime component."""

    component: DeliveryRuntimeComponent
    ready: bool
    reason_code: DeliveryRuntimeReasonCode | None
    heartbeat_age_seconds: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.component, DeliveryRuntimeComponent):
            raise TypeError("delivery component is invalid")
        if not isinstance(self.ready, bool):
            raise TypeError("delivery component readiness is invalid")
        if self.ready == (self.reason_code is not None):
            raise ValueError("delivery component reason is invalid")
        if self.heartbeat_age_seconds is not None and (
            isinstance(self.heartbeat_age_seconds, bool)
            or not isinstance(self.heartbeat_age_seconds, int)
            or self.heartbeat_age_seconds < 0
        ):
            raise ValueError("delivery component heartbeat age is invalid")


@dataclass(frozen=True)
class DeliveryBacklogCounts:
    """Closed nonterminal delivery counts from one read snapshot."""

    pending: int = 0
    enqueue_claimed: int = 0
    queued: int = 0
    processing: int = 0
    retry_wait: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.__dict__.values()
        ):
            raise ValueError("delivery backlog count is invalid")


@dataclass(frozen=True)
class DeliveryHealthSnapshot:
    """Sanitized AuthNZ delivery facts from one bounded read snapshot."""

    canonical_schema_version: int
    delivery_schema_ready: bool
    migration_complete: bool
    key_ready: bool
    key_primary_match: bool
    worker: DeliveryComponentStatus
    reconciler: DeliveryComponentStatus
    retention: DeliveryComponentStatus
    backlog: DeliveryBacklogCounts
    oldest_nonterminal_created_at: datetime | None


@dataclass(frozen=True)
class DeliveryCapabilityStatus:
    """Closed public delivery capability without instance or secret identity."""

    canonical_schema_version: int
    schema_ready: bool
    delivery_schema_ready: bool
    migration_complete: bool
    key_ready: bool
    key_primary_match: bool
    jobs_database_ready: bool
    queue_ready: bool
    job_type_ready: bool
    jobs_backend: str
    worker: DeliveryComponentStatus
    reconciler: DeliveryComponentStatus
    retention: DeliveryComponentStatus
    backlog: DeliveryBacklogCounts
    oldest_nonterminal_age_seconds: int | None
    acquisition_ready: bool
    acquisition_reason_code: DeliveryRuntimeReasonCode | None
    delivery_capability_ready: bool


@dataclass(frozen=True)
class WebhookRegistration:
    """Redacted immutable registration metadata returned by the control plane."""

    id: int
    description: str
    target_display: str
    target_hostname: str
    event_types: tuple[str, ...]
    active: bool
    timeout_seconds: int
    revision: int
    delivery_config_version: int
    target_version: int
    secret_version: int
    secret_rotation_required: bool
    created_by_user_id: int
    updated_by_user_id: int
    created_at: datetime
    updated_at: datetime
    deleted_at: datetime | None = None
    deleted_by_user_id: int | None = None


@dataclass(frozen=True)
class WebhookLimits:
    """Effective bounded registration limits."""

    registrations: int
    active_registrations: int
    current_registrations: int = 0
    current_active_registrations: int = 0
    registrations_over_limit: bool = False
    active_registrations_over_limit: bool = False


@dataclass(frozen=True)
class WebhookMigrationSummary:
    """Sanitized migration state exposed to operators."""

    phase: str
    imported_count: int = 0
    unresolved_count: int = 0
    rejected_count: int = 0
    secret_rotation_required_count: int = 0
    legacy_file_restore_permitted: bool = False
    rollback_expires_at: datetime | None = None


@dataclass(frozen=True)
class WebhookStatus:
    """Sanitized canonical control-plane and delivery status projection."""

    mode: str
    route_selection: str
    schema_ready: bool
    key_state: str
    delivery_capability_ready: bool
    delivery: DeliveryCapabilityStatus
    limits: WebhookLimits
    migration: WebhookMigrationSummary


@dataclass(frozen=True)
class IdempotencyScope:
    """Normalized identity for one idempotent command family."""

    actor_id: str
    operation: str
    route: str
    webhook_id: int | None = None
    delivery_id: str | None = None


class IdempotencyClaimState(str, Enum):
    """Repository result for an idempotency claim."""

    CLAIMED = "claimed"
    REPLAY = "replay"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class IdempotencyClaim:
    """Bounded idempotency claim result shared with the control plane."""

    state: IdempotencyClaimState
    resource_id: int | None = None
    resource_version: int | None = None
    secret_version: int | None = None


@dataclass(frozen=True)
class ValidatedWebhookTarget:
    """Validated full target plus safe metadata for separate persistence."""

    url: str
    hostname: str
    target_display: str


_PENDING_MARKER_FIELDS = frozenset(
    {
        "event_id",
        "event_type",
        "api_version",
        "source_kind",
        "aggregate_type",
        "aggregate_id",
        "aggregate_version",
        "source_command_id",
        "body_ciphertext_json",
        "body_key_id",
        "created_at",
    }
)
_MARKER_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,255}$")
_MARKER_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")


@dataclass(frozen=True)
class PendingIncidentWebhookMarker:
    """Dormant crash-safe incident event marker with only protected body bytes."""

    event_id: str
    event_type: str
    api_version: str
    source_kind: str
    aggregate_type: str | None
    aggregate_id: str | None
    aggregate_version: str | None
    source_command_id: str | None
    body: ProtectedValue
    created_at: datetime

    def __post_init__(self) -> None:
        if _MARKER_ID_PATTERN.fullmatch(self.event_id) is None:
            raise ValueError("pending marker event ID is invalid")
        if not self.event_type.startswith("incident.") or (
            _MARKER_COMPONENT_PATTERN.fullmatch(self.event_type) is None
        ):
            raise ValueError("pending marker event type is invalid")
        if _MARKER_COMPONENT_PATTERN.fullmatch(self.api_version) is None:
            raise ValueError("pending marker API version is invalid")
        if self.source_kind not in {"aggregate", "command"}:
            raise ValueError("pending marker source kind is invalid")
        if not isinstance(self.body, ProtectedValue):
            raise TypeError("pending marker body must be protected")
        if not isinstance(self.created_at, datetime) or self.created_at.tzinfo is None:
            raise ValueError("pending marker timestamp must be timezone-aware")

        aggregate_values = (
            self.aggregate_type,
            self.aggregate_id,
            self.aggregate_version,
        )
        if self.source_kind == "aggregate":
            if self.source_command_id is not None or any(
                value is None or _MARKER_ID_PATTERN.fullmatch(value) is None
                for value in aggregate_values
            ):
                raise ValueError("pending aggregate marker identity is invalid")
        elif (
            any(value is not None for value in aggregate_values)
            or self.source_command_id is None
            or _MARKER_ID_PATTERN.fullmatch(self.source_command_id) is None
        ):
            raise ValueError("pending command marker identity is invalid")

    @property
    def envelope_purpose(self) -> str:
        return "pending_incident.body"

    @property
    def envelope_identity(self) -> Mapping[str, str | int]:
        identity: dict[str, str | int] = {
            "event_id": self.event_id,
            "api_version": self.api_version,
        }
        if self.source_kind == "command":
            source_command_id = self.source_command_id
            if source_command_id is None:
                raise ValueError("pending command marker identity is invalid")
            identity["source_command_id"] = source_command_id
        else:
            aggregate_type = self.aggregate_type
            aggregate_id = self.aggregate_id
            aggregate_version = self.aggregate_version
            if (
                aggregate_type is None
                or aggregate_id is None
                or aggregate_version is None
            ):
                raise ValueError("pending aggregate marker identity is invalid")
            identity.update(
                {
                    "aggregate_type": aggregate_type,
                    "aggregate_id": aggregate_id,
                    "aggregate_version": aggregate_version,
                }
            )
        return identity

    def to_store_record(self) -> dict[str, object]:
        """Return the exact structural representation persisted in system ops."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "api_version": self.api_version,
            "source_kind": self.source_kind,
            "aggregate_type": self.aggregate_type,
            "aggregate_id": self.aggregate_id,
            "aggregate_version": self.aggregate_version,
            "source_command_id": self.source_command_id,
            "body_ciphertext_json": self.body.ciphertext_json,
            "body_key_id": self.body.key_id,
            "created_at": self.created_at.astimezone(timezone.utc).isoformat(),
        }

    @classmethod
    def from_store_record(cls, value: object) -> PendingIncidentWebhookMarker:
        """Parse one exact marker record without permissive coercion."""
        if not isinstance(value, dict) or set(value) != _PENDING_MARKER_FIELDS:
            raise ValueError("pending incident marker record is invalid")

        def optional_text(name: str) -> str | None:
            item = value[name]
            if item is None:
                return None
            if not isinstance(item, str):
                raise ValueError("pending incident marker record is invalid")
            return item

        required_text: dict[str, str] = {}
        for name in (
            "event_id",
            "event_type",
            "api_version",
            "source_kind",
            "body_ciphertext_json",
            "body_key_id",
            "created_at",
        ):
            item = value[name]
            if not isinstance(item, str):
                raise ValueError("pending incident marker record is invalid")
            required_text[name] = item
        try:
            created_at = datetime.fromisoformat(
                required_text["created_at"].replace("Z", "+00:00")
            )
        except ValueError:
            raise ValueError("pending incident marker record is invalid") from None
        return cls(
            event_id=required_text["event_id"],
            event_type=required_text["event_type"],
            api_version=required_text["api_version"],
            source_kind=required_text["source_kind"],
            aggregate_type=optional_text("aggregate_type"),
            aggregate_id=optional_text("aggregate_id"),
            aggregate_version=optional_text("aggregate_version"),
            source_command_id=optional_text("source_command_id"),
            body=ProtectedValue(
                ciphertext_json=required_text["body_ciphertext_json"],
                key_id=required_text["body_key_id"],
            ),
            created_at=created_at,
        )


_ETAG_PATTERN = re.compile(r'^"admin-webhook-([1-9][0-9]*)-r([1-9][0-9]*)"$')
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{16,255}$")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_OPERATION_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_DNS_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_LOOKUP_DOMAIN = b"tldw-admin-webhook-idempotency-lookup-v1\x00"
_REQUEST_DOMAIN = "tldw-admin-webhook-request-v1"


def build_registration_etag(*, webhook_id: int, revision: int) -> str:
    """Build the strong ETag for one positive registration revision."""
    if webhook_id < 1 or revision < 1:
        raise ValueError("webhook_id and revision must be positive")
    return f'"admin-webhook-{webhook_id}-r{revision}"'


def parse_registration_etag(
    value: str | None,
    *,
    expected_webhook_id: int,
) -> int:
    """Parse an exact strong registration ETag and return its revision."""
    if value is None:
        raise WebhookError(WebhookErrorCode.PRECONDITION_REQUIRED)
    match = _ETAG_PATTERN.fullmatch(value)
    if match is None or int(match.group(1)) != expected_webhook_id:
        raise WebhookError(WebhookErrorCode.PRECONDITION_FAILED)
    return int(match.group(2))


def validate_idempotency_key(value: str) -> str:
    """Validate an opaque idempotency key without normalizing it."""
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is None:
        raise WebhookError(WebhookErrorCode.IDEMPOTENCY_KEY_INVALID)
    return value


def build_idempotency_scope(
    *,
    actor_id: int | str,
    operation: str,
    route: str,
    webhook_id: int | None = None,
    delivery_id: str | None = None,
) -> IdempotencyScope:
    """Build a normalized, resource-bound command scope."""
    normalized_actor = str(actor_id).strip()
    normalized_operation = operation.strip().lower()
    normalized_route = "/" + route.strip().strip("/")
    if not normalized_actor:
        raise ValueError("actor_id is required")
    if _OPERATION_PATTERN.fullmatch(normalized_operation) is None:
        raise ValueError("operation is invalid")
    if webhook_id is not None and webhook_id < 1:
        raise ValueError("webhook_id must be positive")
    normalized_delivery = delivery_id.strip() if delivery_id is not None else None
    if delivery_id is not None and not normalized_delivery:
        raise ValueError("delivery_id cannot be empty")
    return IdempotencyScope(
        actor_id=normalized_actor,
        operation=normalized_operation,
        route=normalized_route,
        webhook_id=webhook_id,
        delivery_id=normalized_delivery,
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def idempotency_lookup_digest(
    idempotency_key: str,
    scope: IdempotencyScope,
) -> str:
    """Return the domain-separated lookup digest for a raw command key."""
    key = validate_idempotency_key(idempotency_key)
    payload = _canonical_json_bytes({"scope": asdict(scope)})
    digest = hashlib.sha256(
        _LOOKUP_DOMAIN + payload + b"\x00" + key.encode("ascii")
    ).hexdigest()
    return f"sha256:{digest}"


def canonical_request_hash(
    idempotency_key: str,
    *,
    scope: IdempotencyScope,
    body: Mapping[str, Any],
    conditional_version: int | None,
) -> str:
    """Return a keyed canonical-request fingerprint for conflict detection."""
    key = validate_idempotency_key(idempotency_key)
    payload = _canonical_json_bytes(
        {
            "version": _REQUEST_DOMAIN,
            "scope": asdict(scope),
            "body": body,
            "conditional_version": conditional_version,
        }
    )
    digest = hmac.new(key.encode("ascii"), payload, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


def normalize_request_id(
    value: str | None,
    *,
    generator: Callable[[], object] = uuid.uuid4,
) -> str:
    """Return a bounded safe request ID or a server-generated fallback."""
    if isinstance(value, str) and _REQUEST_ID_PATTERN.fullmatch(value) is not None:
        return value
    return str(generator())


def _parse_and_normalize_target(url: str) -> tuple[SplitResult, str]:
    if not isinstance(url, str) or not url or len(url) > 2_048:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    try:
        encoded = url.encode("utf-8")
    except UnicodeError as exc:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from exc
    if len(encoded) > 2_048 or "\\" in url or any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except (TypeError, UnicodeError, ValueError) as exc:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from exc
    if not parsed.scheme or not parsed.netloc or not hostname:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    if parsed.username is not None or parsed.password is not None or parsed.fragment:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    if port is not None and not 1 <= port <= 65_535:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    if "%" in hostname:
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    try:
        normalized_host = str(ipaddress.ip_address(hostname))
    except ValueError:
        try:
            normalized_host = hostname.rstrip(".").encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from exc
        labels = normalized_host.split(".")
        if (
            not normalized_host
            or len(normalized_host) > 253
            or any(_DNS_LABEL_PATTERN.fullmatch(label) is None for label in labels)
        ):
            raise WebhookError(WebhookErrorCode.VALIDATION_FAILED) from None
    return parsed, normalized_host


def _redacted_origin(parsed: SplitResult, normalized_host: str) -> str:
    scheme = parsed.scheme.lower()
    host_display = normalized_host
    if ":" in normalized_host:
        host_display = f"[{normalized_host}]"
    port = parsed.port
    default_port = 443 if scheme == "https" else 80
    suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{scheme}://{host_display}{suffix}"


def validate_webhook_target(
    url: str,
    *,
    allow_http_dev: bool,
) -> ValidatedWebhookTarget:
    """Apply strict syntax and central destination policy to a target URL."""
    parsed, normalized_host = _parse_and_normalize_target(url)
    scheme = parsed.scheme.lower()
    if scheme != "https" and not (scheme == "http" and allow_http_dev):
        raise WebhookError(WebhookErrorCode.VALIDATION_FAILED)
    result = evaluate_platform_webhook_url_policy(url)
    if not result.allowed:
        raise WebhookError(WebhookErrorCode.TARGET_REJECTED)
    return ValidatedWebhookTarget(
        url=url,
        hostname=normalized_host,
        target_display=_redacted_origin(parsed, normalized_host),
    )


def redact_target(url: str) -> str:
    """Return a safe origin-only display for a previously validated target."""
    parsed, normalized_host = _parse_and_normalize_target(url)
    return _redacted_origin(parsed, normalized_host)
