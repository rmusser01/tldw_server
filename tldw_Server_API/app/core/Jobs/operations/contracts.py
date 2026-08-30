"""Typed operation contracts for future Jobs backend extraction."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import UUID


class OperationOutcome(str, Enum):
    """High-level result categories returned by backend operations."""

    APPLIED = "applied"
    NO_TRANSITION = "no_transition"
    ADMISSION_REJECTED = "admission_rejected"
    BACKEND_CONFLICT = "backend_conflict"
    BACKEND_SCHEMA_ERROR = "backend_schema_error"
    BACKEND_ERROR = "backend_error"


class PreparedDispositionKind(str, Enum):
    """Closed Jobs transitions returned by prepared workers."""

    COMPLETE = "complete"
    RETRY = "retry"
    FAIL = "fail"
    CANCEL = "cancel"
    DEFER = "defer"


class PreparedDispositionOrigin(str, Enum):
    """Authority that prepared a Jobs disposition."""

    AUTHNZ = "authnz"
    INFRASTRUCTURE = "infrastructure"
    RECOVERY = "recovery"


class ExpiredLeasePolicy(str, Enum):
    """Per-job behavior when a processing lease expires."""

    CONSUME_RETRY = "consume_retry"
    REQUEUE_NO_ATTEMPT = "requeue_no_attempt"


class JobIdentityLookupState(str, Enum):
    """Closed results for exact active/archive identity lookup."""

    ACTIVE = "active"
    ARCHIVED = "archived"
    MISSING = "missing"
    CONFLICT = "conflict"


_OPAQUE_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_REASON_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")

ADMIN_WEBHOOK_DELIVERY_DOMAIN = "admin_webhooks"
ADMIN_WEBHOOK_DELIVERY_QUEUE = "delivery"
ADMIN_WEBHOOK_DELIVERY_JOB_TYPE = "admin_webhook_delivery"
ADMIN_WEBHOOK_DELIVERY_IDEMPOTENCY_PREFIX = "admin-webhook-delivery:"
ADMIN_WEBHOOK_DELIVERY_PRIORITY = 5
ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES = 3
ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD = 5
_ADMIN_WEBHOOK_MARKER_MAX_BYTES = 2_048
_ADMIN_WEBHOOK_NO_ATTEMPT_FAIL_REASONS = frozenset(
    {"attempt_budget_exhausted", "delivery_expired"}
)


def _canonical_uuid4(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a canonical UUIDv4")
    try:
        parsed = UUID(value)
    except (AttributeError, ValueError):
        raise ValueError(f"{field_name} must be a canonical UUIDv4") from None
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{field_name} must be a canonical UUIDv4")
    return value


def _aware_utc(value: datetime, *, field_name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _bounded_identity(value: object, *, field_name: str, maximum: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= maximum:
        raise ValueError(f"{field_name} must be between 1 and {maximum} characters")
    return value


def canonical_admin_webhook_delivery_id(payload: object) -> str:
    """Return the sole canonical delivery ID or reject the payload shape."""

    if not isinstance(payload, dict) or set(payload) != {"delivery_id"}:
        raise ValueError("canonical admin webhook payload must contain only delivery_id")
    return _canonical_uuid4(payload["delivery_id"], field_name="delivery_id")


def canonical_admin_webhook_idempotency_key(delivery_id: str) -> str:
    """Return the canonical Jobs idempotency key for one delivery."""

    return f"{ADMIN_WEBHOOK_DELIVERY_IDEMPOTENCY_PREFIX}{delivery_id}"


def is_admin_webhook_delivery_queue(domain: object, queue: object) -> bool:
    """Return whether admission targets the reserved canonical queue."""

    return (
        domain == ADMIN_WEBHOOK_DELIVERY_DOMAIN
        and queue == ADMIN_WEBHOOK_DELIVERY_QUEUE
    )


def canonical_admin_webhook_row_matches(
    row: dict[str, Any],
    *,
    expected_payload: dict[str, Any],
    archived: bool = False,
) -> bool:
    """Verify immutable canonical identity and controls on a persisted row."""

    if not isinstance(archived, bool):
        return False
    try:
        delivery_id = canonical_admin_webhook_delivery_id(expected_payload)
        _canonical_uuid4(row.get("uuid"), field_name="job_uuid")
    except ValueError:
        return False
    if (
        row.get("domain") != ADMIN_WEBHOOK_DELIVERY_DOMAIN
        or row.get("queue") != ADMIN_WEBHOOK_DELIVERY_QUEUE
        or row.get("job_type") != ADMIN_WEBHOOK_DELIVERY_JOB_TYPE
        or row.get("payload") != expected_payload
        or row.get("idempotency_key")
        != canonical_admin_webhook_idempotency_key(delivery_id)
    ):
        return False
    marker_valid, marker = _canonical_admin_webhook_marker(
        row,
        delivery_id=delivery_id,
    )
    if not marker_valid:
        return False
    if (
        row.get("owner_user_id") is not None
        or row.get("project_id") is not None
        or row.get("batch_group") is not None
        or row.get("priority") != ADMIN_WEBHOOK_DELIVERY_PRIORITY
        or row.get("max_retries") != ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES
    ):
        return False
    raw_available_at = row.get("available_at")
    available_at = _stored_utc(raw_available_at)
    if raw_available_at is not None and available_at is None:
        return False
    recovery_fingerprint = row.get("no_attempt_recovery_fingerprint")
    if recovery_fingerprint is not None and (
        not isinstance(recovery_fingerprint, str)
        or _OPAQUE_TOKEN_RE.fullmatch(recovery_fingerprint) is None
    ):
        return False
    prepared_fingerprint = row.get("prepared_disposition_fingerprint")
    status = row.get("status")
    if status == "queued":
        if marker is None:
            if available_at is not None or recovery_fingerprint is not None:
                return False
        elif marker["kind"] not in {
            PreparedDispositionKind.RETRY.value,
            PreparedDispositionKind.DEFER.value,
        }:
            return False
        elif available_at is None:
            if (
                recovery_fingerprint is None
                or recovery_fingerprint != prepared_fingerprint
            ):
                return False
        elif (
            recovery_fingerprint is not None
            or not _canonical_schedule_matches(
                marker,
                available_at=available_at,
                sqlite_second_precision=isinstance(raw_available_at, str),
            )
        ):
            return False
    elif status == "processing":
        if recovery_fingerprint is not None:
            return False
        if marker is not None and marker["kind"] not in {
            PreparedDispositionKind.RETRY.value,
            PreparedDispositionKind.DEFER.value,
        }:
            return False
        if available_at is not None:
            acquired_at = _stored_utc(row.get("acquired_at"))
            if (
                marker is None
                or marker["kind"]
                not in {
                    PreparedDispositionKind.RETRY.value,
                    PreparedDispositionKind.DEFER.value,
                }
                or not _canonical_schedule_matches(
                    marker,
                    available_at=available_at,
                    sqlite_second_precision=isinstance(raw_available_at, str),
                )
                or acquired_at is None
                or available_at > acquired_at
            ):
                return False
    else:
        terminal_kind = {
            "completed": PreparedDispositionKind.COMPLETE.value,
            "failed": PreparedDispositionKind.FAIL.value,
            "cancelled": PreparedDispositionKind.CANCEL.value,
            "quarantined": PreparedDispositionKind.RETRY.value,
        }.get(status)
        if (
            terminal_kind is None
            or marker is None
            or marker["kind"] != terminal_kind
            or row.get("completion_token") != marker["token"]
            or recovery_fingerprint is not None
        ):
            return False
    return (
        row.get("expired_lease_policy")
        == ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT.value
        and row.get("quarantine_threshold")
        == ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD
    )


def _stored_utc(value: object) -> datetime | None:
    """Parse one SQLite/PostgreSQL stored timestamp for invariant comparison."""

    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _canonical_admin_webhook_marker(
    row: dict[str, Any],
    *,
    delivery_id: str,
    require_fingerprint: bool = True,
) -> tuple[bool, dict[str, Any] | None]:
    """Validate the strict public disposition marker without exposing facts."""

    marker = row.get("result")
    fingerprint = row.get("prepared_disposition_fingerprint")
    if marker is None:
        return fingerprint is None, None
    if isinstance(marker, str):
        if len(marker.encode("utf-8")) > _ADMIN_WEBHOOK_MARKER_MAX_BYTES:
            return False, None
        try:
            marker = json.loads(marker)
        except (TypeError, ValueError):
            return False, None
    if not isinstance(marker, dict):
        return False, None
    try:
        encoded_marker = json.dumps(
            marker,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError):
        return False, None
    if len(encoded_marker) > _ADMIN_WEBHOOK_MARKER_MAX_BYTES:
        return False, None
    if require_fingerprint and (
        not isinstance(fingerprint, str)
        or _OPAQUE_TOKEN_RE.fullmatch(fingerprint) is None
    ):
        return False, None
    if not require_fingerprint and fingerprint is not None:
        return False, None
    kind = marker.get("kind")
    origin = marker.get("origin")
    expected_keys = {
        "schema_version",
        "token",
        "kind",
        "origin",
        "delivery_id",
        "applied_at",
    }
    if kind in {
        PreparedDispositionKind.COMPLETE.value,
        PreparedDispositionKind.RETRY.value,
    }:
        expected_keys.add("attempt_id")
    elif kind in {
        PreparedDispositionKind.FAIL.value,
        PreparedDispositionKind.CANCEL.value,
    }:
        if marker.get("attempt_id") is not None:
            expected_keys.add("attempt_id")
    elif kind != PreparedDispositionKind.DEFER.value:
        return False, None
    if kind in {
        PreparedDispositionKind.RETRY.value,
        PreparedDispositionKind.DEFER.value,
    }:
        expected_keys.add("original_not_before_at")
    if set(marker) != expected_keys:
        return False, None
    if kind == PreparedDispositionKind.DEFER.value:
        if origin not in {
            PreparedDispositionOrigin.INFRASTRUCTURE.value,
            PreparedDispositionOrigin.RECOVERY.value,
        }:
            return False, None
    elif origin != PreparedDispositionOrigin.AUTHNZ.value:
        return False, None
    try:
        if "attempt_id" in expected_keys:
            _canonical_uuid4(marker.get("attempt_id"), field_name="attempt_id")
    except ValueError:
        return False, None
    valid = bool(
        marker.get("schema_version") == 1
        and not isinstance(marker.get("schema_version"), bool)
        and marker.get("delivery_id") == delivery_id
        and isinstance(marker.get("token"), str)
        and _OPAQUE_TOKEN_RE.fullmatch(marker["token"]) is not None
        and _stored_utc(marker.get("applied_at")) is not None
        and (
            "original_not_before_at" not in expected_keys
            or _stored_utc(marker.get("original_not_before_at")) is not None
        )
    )
    return valid, marker if valid else None


@dataclass(frozen=True)
class AdminWebhookDispositionMarker:
    """Strict bounded Jobs evidence for one applied canonical disposition."""

    token: str
    kind: PreparedDispositionKind
    origin: PreparedDispositionOrigin
    delivery_id: str
    attempt_id: str | None
    original_not_before_at: datetime | None
    applied_at: datetime
    fingerprint: str


def project_admin_webhook_disposition_marker(
    row: dict[str, Any],
    *,
    expected_payload: dict[str, Any],
    archived: bool = False,
) -> AdminWebhookDispositionMarker | None:
    """Project one canonical marker only after complete row validation."""

    if not isinstance(row, dict) or not canonical_admin_webhook_row_matches(
        row,
        expected_payload=expected_payload,
        archived=archived,
    ):
        return None
    try:
        delivery_id = canonical_admin_webhook_delivery_id(expected_payload)
    except ValueError:
        return None
    marker_valid, marker = _canonical_admin_webhook_marker(
        row,
        delivery_id=delivery_id,
    )
    fingerprint = row.get("prepared_disposition_fingerprint")
    if (
        not marker_valid
        or marker is None
        or not isinstance(fingerprint, str)
        or _OPAQUE_TOKEN_RE.fullmatch(fingerprint) is None
    ):
        return None
    try:
        kind = PreparedDispositionKind(marker["kind"])
        origin = PreparedDispositionOrigin(marker["origin"])
    except (KeyError, TypeError, ValueError):
        return None
    applied_at = _stored_utc(marker.get("applied_at"))
    original_not_before = _stored_utc(marker.get("original_not_before_at"))
    if applied_at is None:
        return None
    return AdminWebhookDispositionMarker(
        token=marker["token"],
        kind=kind,
        origin=origin,
        delivery_id=delivery_id,
        attempt_id=marker.get("attempt_id"),
        original_not_before_at=original_not_before,
        applied_at=applied_at,
        fingerprint=fingerprint,
    )


def _canonical_schedule_matches(
    marker: dict[str, Any],
    *,
    available_at: datetime,
    sqlite_second_precision: bool,
) -> bool:
    """Verify the stored schedule is the marker's database-clock maximum."""

    applied_at = _stored_utc(marker.get("applied_at"))
    original_not_before = _stored_utc(marker.get("original_not_before_at"))
    if applied_at is None or original_not_before is None:
        return False
    expected = max(applied_at, original_not_before)
    if sqlite_second_precision:
        return (
            available_at.microsecond == 0
            and available_at == expected.replace(microsecond=0)
        )
    return available_at == expected


class NoTransitionReason(str, Enum):
    """Reasons a lifecycle operation made no durable state transition."""

    MISSING = "missing"
    WRONG_STATUS = "wrong_status"
    STALE_LEASE = "stale_lease"
    ALREADY_TERMINAL = "already_terminal"
    IDEMPOTENT_EXISTING = "idempotent_existing"
    RLS_FILTERED = "rls_filtered"
    NO_ELIGIBLE_JOB = "no_eligible_job"


class AdmissionRejectionReason(str, Enum):
    """Reasons a create/admission request can be rejected before insertion."""

    QUEUE_PAUSED = "queue_paused"
    QUEUE_DRAINING = "queue_draining"
    QUOTA_EXCEEDED = "quota_exceeded"
    FAIR_SHARE_LIMIT = "fair_share_limit"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    POLICY_REJECTED = "policy_rejected"


class IdempotentOperationDisposition(str, Enum):
    """How a durable idempotent operation admission was resolved."""

    CREATED = "created"
    REPLAYED = "replayed"
    CONVERGED = "converged"


class IdempotentOperationConflictReason(str, Enum):
    """Stable conflicts returned by durable idempotent admission."""

    KEY_REUSED = "idempotency_key_reused"
    SCOPE_ACTIVE = "operation_already_in_progress"


class IdempotentOperationConflict(RuntimeError):
    """A durable admission conflicts with previously accepted work."""

    def __init__(
        self,
        reason: IdempotentOperationConflictReason,
        job_uuid: str | None = None,
    ) -> None:
        super().__init__(reason.value)
        self.reason = reason
        self.job_uuid = job_uuid


class IdempotentOperationUnavailableError(RuntimeError):
    """A receipt-to-Job correlation cannot be proven safe."""


class SlidesArchiveNormalizationError(Exception):
    """A compressed Slides archive projection is invalid."""


class TerminalOperationResultPatchOutcome(str, Enum):
    """Closed outcomes for an exact terminal operation-result patch."""

    APPLIED = "applied"
    IDEMPOTENT = "idempotent"
    MISSING = "missing"
    CONFLICT = "conflict"


def terminal_operation_result_fingerprint(result: dict[str, Any]) -> str:
    """Return the canonical SHA-256 fingerprint for a terminal result object."""

    if not isinstance(result, dict):
        raise ValueError("terminal operation result must be an object")
    try:
        encoded = json.dumps(
            result,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal operation result must be JSON-serializable") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TerminalOperationResultPatchCommand:
    """Exact correlation and compare-and-set data for terminal result repair."""

    job_uuid: str
    owner_user_id: str
    domain: str
    queue: str
    job_type: str
    operation_scope: str
    allowed_statuses: tuple[str, ...]
    expected_result_fingerprint: str
    replacement_result: dict[str, Any]

    def __post_init__(self) -> None:
        """Reject malformed correlation and defensively copy mutable result data."""

        for field_name in (
            "job_uuid",
            "owner_user_id",
            "domain",
            "queue",
            "job_type",
            "operation_scope",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value or len(value) > 200:
                raise ValueError(f"{field_name} must be between 1 and 200 characters")
        statuses = tuple(self.allowed_statuses)
        if not statuses or len(statuses) != len(set(statuses)):
            raise ValueError("allowed_statuses must contain unique terminal statuses")
        if any(
            status not in {"completed", "failed", "cancelled", "quarantined"}
            for status in statuses
        ):
            raise ValueError("allowed_statuses contains a nonterminal status")
        fingerprint = self.expected_result_fingerprint
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(
                "expected_result_fingerprint must be a lowercase SHA-256 digest"
            )
        terminal_operation_result_fingerprint(self.replacement_result)
        object.__setattr__(self, "allowed_statuses", statuses)
        object.__setattr__(self, "replacement_result", copy.deepcopy(self.replacement_result))


@dataclass(frozen=True)
class CreateJobCommand:
    """Backend-neutral command payload for creating a Jobs row."""

    domain: str
    queue: str
    job_type: str
    payload: dict[str, Any] | None
    owner_user_id: str | None
    idempotency_key: str | None = None
    priority: int = 100
    max_retries: int = 3
    available_at: datetime | None = None
    project_id: int | str | None = None
    batch_group: str | None = None
    request_id: str | None = None
    trace_id: str | None = None
    expired_lease_policy: ExpiredLeasePolicy = ExpiredLeasePolicy.CONSUME_RETRY
    quarantine_threshold: int | None = None

    def __post_init__(self) -> None:
        """Validate controls that must fail before any backend operation."""

        try:
            policy = ExpiredLeasePolicy(self.expired_lease_policy)
        except (TypeError, ValueError):
            raise ValueError("expired_lease_policy is invalid") from None
        threshold = self.quarantine_threshold
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, int)
            or threshold <= 0
        ):
            raise ValueError("quarantine_threshold must be a positive integer")
        object.__setattr__(self, "expired_lease_policy", policy)


@dataclass(frozen=True)
class PreparedJobDisposition:
    """Validated exact transition prepared outside the Jobs database."""

    token: str
    kind: PreparedDispositionKind
    origin: PreparedDispositionOrigin
    delivery_id: str
    attempt_id: str | None = None
    delay_seconds: int | None = None
    not_before_at: datetime | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = PreparedDispositionKind(self.kind)
            origin = PreparedDispositionOrigin(self.origin)
        except (TypeError, ValueError):
            raise ValueError("prepared disposition kind or origin is invalid") from None
        if not isinstance(self.token, str) or _OPAQUE_TOKEN_RE.fullmatch(self.token) is None:
            raise ValueError("token must be 64 lowercase hexadecimal characters")
        _canonical_uuid4(self.delivery_id, field_name="delivery_id")
        if self.attempt_id is not None:
            _canonical_uuid4(self.attempt_id, field_name="attempt_id")
        if self.reason_code is not None and (
            not isinstance(self.reason_code, str)
            or _REASON_CODE_RE.fullmatch(self.reason_code) is None
        ):
            raise ValueError("reason_code is invalid")

        authnz_kinds = {
            PreparedDispositionKind.COMPLETE,
            PreparedDispositionKind.RETRY,
            PreparedDispositionKind.FAIL,
            PreparedDispositionKind.CANCEL,
        }
        if kind in authnz_kinds and origin is not PreparedDispositionOrigin.AUTHNZ:
            raise ValueError("complete, retry, fail, and cancel require AuthNZ origin")
        if kind is PreparedDispositionKind.DEFER and origin is PreparedDispositionOrigin.AUTHNZ:
            raise ValueError("defer requires infrastructure or recovery origin")

        if kind in {
            PreparedDispositionKind.COMPLETE,
            PreparedDispositionKind.RETRY,
        } and self.attempt_id is None:
            raise ValueError(f"{kind.value} requires attempt_id")
        if (
            kind is PreparedDispositionKind.FAIL
            and self.attempt_id is None
            and self.reason_code not in _ADMIN_WEBHOOK_NO_ATTEMPT_FAIL_REASONS
        ):
            raise ValueError("no-attempt fail reason is invalid")
        if kind is PreparedDispositionKind.DEFER and self.attempt_id is not None:
            raise ValueError("defer cannot include attempt_id")

        if kind is PreparedDispositionKind.RETRY:
            if (
                isinstance(self.delay_seconds, bool)
                or not isinstance(self.delay_seconds, int)
                or not 1 <= self.delay_seconds <= 1800
            ):
                raise ValueError("delay_seconds must be between 1 and 1800")
            if self.not_before_at is None:
                raise ValueError("retry requires not_before_at")
            normalized_not_before = _aware_utc(
                self.not_before_at,
                field_name="not_before_at",
            )
        elif self.delay_seconds is not None:
            raise ValueError("delay_seconds is legal only for retry")
        else:
            normalized_not_before = self.not_before_at

        if kind is not PreparedDispositionKind.RETRY:
            if origin is PreparedDispositionOrigin.INFRASTRUCTURE:
                if self.not_before_at is not None:
                    raise ValueError("infrastructure defer cannot provide not_before_at")
                normalized_not_before = None
            elif origin is PreparedDispositionOrigin.RECOVERY:
                if self.not_before_at is None:
                    raise ValueError("recovery defer requires not_before_at")
                normalized_not_before = _aware_utc(
                    self.not_before_at,
                    field_name="not_before_at",
                )
            elif self.not_before_at is not None:
                raise ValueError("not_before_at is legal only for retry or recovery defer")

        requires_reason = kind in {
            PreparedDispositionKind.RETRY,
            PreparedDispositionKind.FAIL,
            PreparedDispositionKind.CANCEL,
            PreparedDispositionKind.DEFER,
        }
        if requires_reason and self.reason_code is None:
            raise ValueError(f"{kind.value} requires reason_code")
        if kind is PreparedDispositionKind.COMPLETE and self.reason_code is not None:
            raise ValueError("complete cannot include reason_code")

        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "not_before_at", normalized_not_before)

    @classmethod
    def complete(
        cls,
        *,
        token: str,
        delivery_id: str,
        attempt_id: str,
    ) -> PreparedJobDisposition:
        return cls(
            token=token,
            kind=PreparedDispositionKind.COMPLETE,
            origin=PreparedDispositionOrigin.AUTHNZ,
            delivery_id=delivery_id,
            attempt_id=attempt_id,
        )

    @classmethod
    def retry(
        cls,
        *,
        token: str,
        delivery_id: str,
        attempt_id: str,
        delay_seconds: int,
        not_before_at: datetime,
        reason_code: str,
    ) -> PreparedJobDisposition:
        return cls(
            token=token,
            kind=PreparedDispositionKind.RETRY,
            origin=PreparedDispositionOrigin.AUTHNZ,
            delivery_id=delivery_id,
            attempt_id=attempt_id,
            delay_seconds=delay_seconds,
            not_before_at=not_before_at,
            reason_code=reason_code,
        )

    @classmethod
    def fail(
        cls,
        *,
        token: str,
        delivery_id: str,
        reason_code: str,
        attempt_id: str | None = None,
    ) -> PreparedJobDisposition:
        return cls(
            token=token,
            kind=PreparedDispositionKind.FAIL,
            origin=PreparedDispositionOrigin.AUTHNZ,
            delivery_id=delivery_id,
            attempt_id=attempt_id,
            reason_code=reason_code,
        )

    @classmethod
    def cancel(
        cls,
        *,
        token: str,
        delivery_id: str,
        reason_code: str,
        attempt_id: str | None = None,
    ) -> PreparedJobDisposition:
        return cls(
            token=token,
            kind=PreparedDispositionKind.CANCEL,
            origin=PreparedDispositionOrigin.AUTHNZ,
            delivery_id=delivery_id,
            attempt_id=attempt_id,
            reason_code=reason_code,
        )

    @classmethod
    def infrastructure_defer(
        cls,
        *,
        token: str,
        delivery_id: str,
        reason_code: str,
    ) -> PreparedJobDisposition:
        return cls(
            token=token,
            kind=PreparedDispositionKind.DEFER,
            origin=PreparedDispositionOrigin.INFRASTRUCTURE,
            delivery_id=delivery_id,
            reason_code=reason_code,
        )

    @classmethod
    def recovery_defer_until(
        cls,
        *,
        token: str,
        delivery_id: str,
        not_before_at: datetime,
        reason_code: str,
    ) -> PreparedJobDisposition:
        return cls(
            token=token,
            kind=PreparedDispositionKind.DEFER,
            origin=PreparedDispositionOrigin.RECOVERY,
            delivery_id=delivery_id,
            not_before_at=not_before_at,
            reason_code=reason_code,
        )


def prepared_disposition_fingerprint(disposition: PreparedJobDisposition) -> str:
    """Hash every validated replay fact without expanding public result evidence."""

    facts = {
        "attempt_id": disposition.attempt_id,
        "delay_seconds": disposition.delay_seconds,
        "delivery_id": disposition.delivery_id,
        "kind": disposition.kind.value,
        "not_before_at": (
            disposition.not_before_at.isoformat()
            if disposition.not_before_at is not None
            else None
        ),
        "origin": disposition.origin.value,
        "reason_code": disposition.reason_code,
        "token": disposition.token,
    }
    encoded = json.dumps(
        facts,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reconstruct_legacy_admin_webhook_archive_fingerprint(
    row: dict[str, Any],
) -> str | None:
    """Return a safe legacy fingerprint upgrade or reject reserved evidence."""

    error = RuntimeError(
        "canonical admin webhook legacy archive cannot be reconstructed"
    )
    if not isinstance(row, dict):
        raise error
    try:
        _canonical_uuid4(row.get("uuid"), field_name="job_uuid")
        payload = row.get("payload")
        if isinstance(payload, str):
            if len(payload.encode("utf-8")) > _ADMIN_WEBHOOK_MARKER_MAX_BYTES:
                raise ValueError
            payload = json.loads(payload)
        delivery_id = canonical_admin_webhook_delivery_id(payload)
    except (RecursionError, TypeError, ValueError):
        raise error from None
    normalized = {**row, "payload": payload}
    if canonical_admin_webhook_row_matches(
        normalized,
        expected_payload=payload,
        archived=True,
    ):
        return None
    if (
        normalized.get("expired_lease_policy")
        != ExpiredLeasePolicy.CONSUME_RETRY.value
        or normalized.get("quarantine_threshold") is not None
        or normalized.get("prepared_disposition_fingerprint") is not None
        or normalized.get("no_attempt_recovery_fingerprint") is not None
    ):
        raise error
    marker_valid, marker = _canonical_admin_webhook_marker(
        normalized,
        delivery_id=delivery_id,
        require_fingerprint=False,
    )
    if (
        not marker_valid
        or marker is None
        or normalized.get("completion_token") != marker.get("token")
    ):
        raise error
    try:
        kind = PreparedDispositionKind(marker["kind"])
        if kind is PreparedDispositionKind.COMPLETE:
            if normalized.get("status") != "completed":
                raise ValueError
            disposition = PreparedJobDisposition.complete(
                token=marker["token"],
                delivery_id=delivery_id,
                attempt_id=marker["attempt_id"],
            )
        elif kind is PreparedDispositionKind.FAIL:
            reason = normalized.get("error_code")
            if (
                normalized.get("status") != "failed"
                or reason != normalized.get("error_message")
                or reason != normalized.get("last_error")
            ):
                raise ValueError
            disposition = PreparedJobDisposition.fail(
                token=marker["token"],
                delivery_id=delivery_id,
                attempt_id=marker.get("attempt_id"),
                reason_code=reason,
            )
        elif kind is PreparedDispositionKind.CANCEL:
            if normalized.get("status") != "cancelled":
                raise ValueError
            disposition = PreparedJobDisposition.cancel(
                token=marker["token"],
                delivery_id=delivery_id,
                attempt_id=marker.get("attempt_id"),
                reason_code=normalized.get("cancellation_reason"),
            )
        else:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise error from None

    fingerprint = prepared_disposition_fingerprint(disposition)
    upgraded = {
        **normalized,
        "expired_lease_policy": ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT.value,
        "quarantine_threshold": ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD,
        "prepared_disposition_fingerprint": fingerprint,
        "no_attempt_recovery_fingerprint": None,
    }
    if not canonical_admin_webhook_row_matches(
        upgraded,
        expected_payload=payload,
        archived=True,
    ):
        raise error
    return fingerprint


def admin_webhook_disposition_marker_matches(
    marker: AdminWebhookDispositionMarker,
    disposition: PreparedJobDisposition,
) -> bool:
    """Compare every public marker fact plus the complete private fingerprint."""

    if not isinstance(marker, AdminWebhookDispositionMarker) or not isinstance(
        disposition,
        PreparedJobDisposition,
    ):
        return False
    return bool(
        marker.token == disposition.token
        and marker.kind is disposition.kind
        and marker.origin is disposition.origin
        and marker.delivery_id == disposition.delivery_id
        and marker.attempt_id == disposition.attempt_id
        and (
            disposition.not_before_at is None
            or marker.original_not_before_at == disposition.not_before_at
        )
        and marker.fingerprint == prepared_disposition_fingerprint(disposition)
    )


@dataclass(frozen=True)
class ApplyPreparedDispositionCommand:
    """Exact Jobs row, lease, and canonical payload for one disposition."""

    job_id: int
    domain: str
    queue: str
    job_type: str
    expected_payload: dict[str, Any]
    disposition: PreparedJobDisposition
    worker_id: str | None = None
    lease_id: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.job_id, bool) or not isinstance(self.job_id, int) or self.job_id <= 0:
            raise ValueError("job_id must be a positive integer")
        _bounded_identity(self.domain, field_name="domain", maximum=64)
        _bounded_identity(self.queue, field_name="queue", maximum=64)
        _bounded_identity(self.job_type, field_name="job_type", maximum=128)
        if not isinstance(self.expected_payload, dict):
            raise ValueError("expected_payload must be an object")
        if (self.worker_id is None) != (self.lease_id is None):
            raise ValueError("worker_id and lease_id must be provided together")
        for field_name in ("worker_id", "lease_id"):
            value = getattr(self, field_name)
            if value is not None:
                _bounded_identity(value, field_name=field_name, maximum=200)
        object.__setattr__(self, "expected_payload", copy.deepcopy(self.expected_payload))


@dataclass(frozen=True)
class EnsureLeaseHorizonCommand:
    """Exact processing lease whose remaining horizon must be ensured."""

    job_id: int
    domain: str
    queue: str
    job_type: str
    expected_payload: dict[str, Any]
    worker_id: str
    lease_id: str
    minimum_seconds: int

    def __post_init__(self) -> None:
        if isinstance(self.job_id, bool) or not isinstance(self.job_id, int) or self.job_id <= 0:
            raise ValueError("job_id must be a positive integer")
        if (
            isinstance(self.minimum_seconds, bool)
            or not isinstance(self.minimum_seconds, int)
            or self.minimum_seconds <= 0
        ):
            raise ValueError("minimum_seconds must be a positive integer")
        for field_name, maximum in (
            ("domain", 64),
            ("queue", 64),
            ("job_type", 128),
            ("worker_id", 200),
            ("lease_id", 200),
        ):
            _bounded_identity(
                getattr(self, field_name),
                field_name=field_name,
                maximum=maximum,
            )
        if not isinstance(self.expected_payload, dict):
            raise ValueError("expected_payload must be an object")
        object.__setattr__(self, "expected_payload", copy.deepcopy(self.expected_payload))


@dataclass(frozen=True)
class FindJobByIdentityCommand:
    """Read-only exact active/archive Jobs identity lookup."""

    domain: str
    queue: str
    job_type: str
    idempotency_key: str
    expected_payload: dict[str, Any]

    def __post_init__(self) -> None:
        for field_name, maximum in (
            ("domain", 64),
            ("queue", 64),
            ("job_type", 128),
            ("idempotency_key", 512),
        ):
            _bounded_identity(
                getattr(self, field_name),
                field_name=field_name,
                maximum=maximum,
            )
        if not isinstance(self.expected_payload, dict):
            raise ValueError("expected_payload must be an object")
        object.__setattr__(self, "expected_payload", copy.deepcopy(self.expected_payload))


@dataclass(frozen=True)
class PreparedDispositionResult:
    """Immutable result of one exact prepared-disposition application."""

    outcome: OperationOutcome
    state: str | None = None
    already_applied: bool = False
    metadata: dict[str, Any] | None = None
    not_before_at: datetime | None = None
    no_transition_reason: NoTransitionReason | None = None

    def __post_init__(self) -> None:
        if self.outcome is OperationOutcome.APPLIED:
            if self.state is None or self.metadata is None:
                raise ValueError("applied prepared disposition requires state and metadata")
        elif self.already_applied:
            raise ValueError("only applied dispositions may be already_applied")
        if self.outcome is OperationOutcome.NO_TRANSITION and self.no_transition_reason is None:
            raise ValueError("no-transition prepared disposition requires a reason")
        if self.outcome is not OperationOutcome.NO_TRANSITION and self.no_transition_reason is not None:
            raise ValueError("only no-transition prepared dispositions include a reason")
        if self.not_before_at is not None:
            object.__setattr__(
                self,
                "not_before_at",
                _aware_utc(self.not_before_at, field_name="not_before_at"),
            )
        object.__setattr__(
            self,
            "metadata",
            copy.deepcopy(self.metadata) if self.metadata is not None else None,
        )

    @classmethod
    def applied(
        cls,
        *,
        state: str,
        metadata: dict[str, Any],
        already_applied: bool,
        not_before_at: datetime | None = None,
    ) -> PreparedDispositionResult:
        return cls(
            outcome=OperationOutcome.APPLIED,
            state=state,
            already_applied=already_applied,
            metadata=metadata,
            not_before_at=not_before_at,
        )

    @classmethod
    def no_transition(
        cls,
        reason: NoTransitionReason,
        *,
        state: str | None = None,
    ) -> PreparedDispositionResult:
        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            state=state,
            no_transition_reason=reason,
        )

    @classmethod
    def conflict(cls, *, state: str | None = None) -> PreparedDispositionResult:
        return cls(outcome=OperationOutcome.BACKEND_CONFLICT, state=state)


@dataclass(frozen=True)
class LeaseHorizonResult:
    """Observed lease horizon after an atomic ensure operation."""

    outcome: OperationOutcome
    ensured: bool
    leased_until: datetime | None = None
    no_transition_reason: NoTransitionReason | None = None
    guaranteed_seconds: int | None = None

    def __post_init__(self) -> None:
        if type(self.outcome) is not OperationOutcome or self.outcome not in {
            OperationOutcome.APPLIED,
            OperationOutcome.NO_TRANSITION,
            OperationOutcome.BACKEND_CONFLICT,
        }:
            raise ValueError("lease horizon outcome is invalid")
        if type(self.ensured) is not bool:
            raise ValueError("ensured must be an exact bool")
        if self.outcome is OperationOutcome.APPLIED:
            if self.ensured is not True:
                raise ValueError("applied lease horizon must be ensured")
            if self.leased_until is None:
                raise ValueError("applied lease horizon requires leased_until")
            if self.no_transition_reason is not None:
                raise ValueError("applied lease horizon cannot include a reason")
            if (
                type(self.guaranteed_seconds) is not int
                or self.guaranteed_seconds <= 0
            ):
                raise ValueError(
                    "applied lease horizon requires positive exact-int guaranteed_seconds"
                )
        elif self.outcome is OperationOutcome.NO_TRANSITION:
            if self.ensured is not False:
                raise ValueError("no-transition lease horizon cannot be ensured")
            if type(self.no_transition_reason) is not NoTransitionReason:
                raise ValueError("no-transition lease horizon requires an exact reason")
            if self.guaranteed_seconds is not None:
                raise ValueError(
                    "no-transition lease horizon cannot include guaranteed_seconds"
                )
        else:
            if self.ensured is not False:
                raise ValueError("conflicting lease horizon cannot be ensured")
            if self.no_transition_reason is not None:
                raise ValueError("conflicting lease horizon cannot include a reason")
            if self.guaranteed_seconds is not None:
                raise ValueError(
                    "conflicting lease horizon cannot include guaranteed_seconds"
                )
        if self.leased_until is not None:
            object.__setattr__(
                self,
                "leased_until",
                _aware_utc(self.leased_until, field_name="leased_until"),
            )

    @classmethod
    def applied(
        cls,
        *,
        leased_until: datetime,
        guaranteed_seconds: int,
    ) -> LeaseHorizonResult:
        return cls(
            outcome=OperationOutcome.APPLIED,
            ensured=True,
            leased_until=leased_until,
            guaranteed_seconds=guaranteed_seconds,
        )

    @classmethod
    def no_transition(
        cls,
        reason: NoTransitionReason,
        *,
        leased_until: datetime | None = None,
    ) -> LeaseHorizonResult:
        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            ensured=False,
            leased_until=leased_until,
            no_transition_reason=reason,
        )


@dataclass(frozen=True)
class JobIdentityLookupResult:
    """Immutable exact identity lookup result."""

    state: JobIdentityLookupState
    row: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.state in {JobIdentityLookupState.ACTIVE, JobIdentityLookupState.ARCHIVED}:
            if self.row is None:
                raise ValueError("found identity lookup requires a row")
        elif self.row is not None:
            raise ValueError("missing or conflicting identity lookup cannot include a row")
        object.__setattr__(
            self,
            "row",
            copy.deepcopy(self.row) if self.row is not None else None,
        )

    @classmethod
    def found(
        cls,
        state: JobIdentityLookupState,
        row: dict[str, Any],
    ) -> JobIdentityLookupResult:
        if state not in {JobIdentityLookupState.ACTIVE, JobIdentityLookupState.ARCHIVED}:
            raise ValueError("found lookup state must be active or archived")
        return cls(state=state, row=row)

    @classmethod
    def missing(cls) -> JobIdentityLookupResult:
        return cls(state=JobIdentityLookupState.MISSING)

    @classmethod
    def conflict(cls) -> JobIdentityLookupResult:
        return cls(state=JobIdentityLookupState.CONFLICT)


@dataclass(frozen=True)
class IdempotentOperationCommand:
    """Backend-neutral command for receipt-backed Job admission."""

    job: CreateJobCommand
    key_digest: str
    request_fingerprint: str
    operation_scope: str
    receipt_expires_at: datetime

    def __post_init__(self) -> None:
        """Reject unowned, malformed, or unbounded receipt metadata."""

        owner_user_id = self.job.owner_user_id
        if not isinstance(owner_user_id, str) or not owner_user_id.strip():
            raise ValueError("owner_user_id must be a non-empty string")
        for field_name in ("key_digest", "request_fingerprint"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 hex digest")
        if (
            not isinstance(self.operation_scope, str)
            or not self.operation_scope.strip()
            or len(self.operation_scope) > 200
        ):
            raise ValueError("operation_scope must be between 1 and 200 ASCII characters")
        try:
            self.operation_scope.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError(
                "operation_scope must be between 1 and 200 ASCII characters"
            ) from exc
        if (
            not isinstance(self.receipt_expires_at, datetime)
            or self.receipt_expires_at.tzinfo is None
            or self.receipt_expires_at.utcoffset() is None
        ):
            raise ValueError("receipt_expires_at must be timezone-aware")
        if self.job.batch_group != self.operation_scope:
            raise ValueError("job batch_group must equal operation_scope")
        if self.job.idempotency_key is not None:
            raise ValueError(
                "job idempotency_key must be unset for owner-scoped receipt admission"
            )


@dataclass(frozen=True)
class IdempotentOperationAdmission:
    """Immutable result of admitting or replaying a receipt-backed Job."""

    job: dict[str, Any]
    disposition: IdempotentOperationDisposition

    def __post_init__(self) -> None:
        """Defensively copy mutable Job data returned by database backends."""

        object.__setattr__(self, "job", copy.deepcopy(self.job))

    @classmethod
    def created(cls, job: dict[str, Any]) -> IdempotentOperationAdmission:
        """Build a result for a newly admitted Job."""

        return cls(job=job, disposition=IdempotentOperationDisposition.CREATED)

    @classmethod
    def replayed(cls, job: dict[str, Any]) -> IdempotentOperationAdmission:
        """Build a result for an exact idempotency-key replay."""

        return cls(job=job, disposition=IdempotentOperationDisposition.REPLAYED)

    @classmethod
    def converged(cls, job: dict[str, Any]) -> IdempotentOperationAdmission:
        """Build a result for a second key converged on active work."""

        return cls(job=job, disposition=IdempotentOperationDisposition.CONVERGED)


@dataclass(frozen=True)
class AcquireJobCommand:
    """Backend-neutral command payload for acquiring one eligible Jobs row."""

    domain: str
    queue: str
    lease_seconds: int
    worker_id: str
    lease_id: str
    owner_user_id: str | None = None
    job_type: str | None = None
    max_inflight_quota: int = 0
    priority_direction: str = "ASC"
    tie_break: str | None = None
    single_update: bool = False

    def __post_init__(self) -> None:
        if self.priority_direction not in {"ASC", "DESC"}:
            raise ValueError("priority_direction must be ASC or DESC")
        if self.tie_break not in {None, "fifo", "lifo"}:
            raise ValueError("tie_break must be fifo, lifo, or None")
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")


@dataclass(frozen=True)
class RenewLeaseCommand:
    """Backend-neutral command payload for renewing one job lease."""

    job_id: int
    seconds: int
    enforce: bool
    worker_id: str | None = None
    lease_id: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None

    def __post_init__(self) -> None:
        """Reject non-positive lease extension durations."""

        if self.seconds < 1:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True)
class BatchRenewLeaseItem:
    """One facade-normalized lease renewal attempt."""

    job_id: int
    seconds: int
    worker_id: str | None = None
    lease_id: str | None = None

    def __post_init__(self) -> None:
        if self.seconds < 1:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True)
class BatchRenewLeasesCommand:
    """Ordered immutable lease renewal attempts for one transaction."""

    items: tuple[BatchRenewLeaseItem, ...]
    enforce: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))


@dataclass(frozen=True)
class BatchRenewLeasesResult:
    """Counts produced by one atomic batch renewal operation."""

    requested_count: int
    applied_count: int

    def __post_init__(self) -> None:
        if self.requested_count < 0:
            raise ValueError("requested_count must be non-negative")
        if not 0 <= self.applied_count <= self.requested_count:
            raise ValueError("applied_count must be between zero and requested_count")


@dataclass(frozen=True)
class ReleaseJobCommand:
    """Backend-neutral command payload for releasing one processing job."""

    job_id: int
    enforce: bool
    worker_id: str | None = None
    lease_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class AdmissionResult:
    """Result facts produced by a create/admission operation."""

    outcome: OperationOutcome
    row: dict[str, Any] | None = None
    was_inserted: bool = False
    no_transition_reason: NoTransitionReason | None = None
    admission_rejection_reason: AdmissionRejectionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate admission invariants and freeze mutable facts."""

        if self.outcome is OperationOutcome.APPLIED and not self.was_inserted:
            raise ValueError("applied admission results must mark was_inserted")
        if self.outcome is OperationOutcome.APPLIED and self.row is None:
            raise ValueError("applied admission results require a row")
        if self.outcome is OperationOutcome.NO_TRANSITION and self.no_transition_reason is None:
            raise ValueError("no-transition admission results require a reason")
        if self.outcome is OperationOutcome.ADMISSION_REJECTED and self.admission_rejection_reason is None:
            raise ValueError("rejected admission results require a rejection reason")
        if self.outcome is not OperationOutcome.APPLIED and self.was_inserted:
            raise ValueError("only applied admission results may mark was_inserted")
        if self.outcome is not OperationOutcome.NO_TRANSITION and self.no_transition_reason is not None:
            raise ValueError("only no-transition admission results may include a no-transition reason")
        if self.outcome is not OperationOutcome.ADMISSION_REJECTED and self.admission_rejection_reason is not None:
            raise ValueError("only rejected admission results may include a rejection reason")
        if (
            self.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
            and self.row is None
        ):
            raise ValueError("idempotent-existing admission results require a row")
        can_include_durable_events = self.outcome is OperationOutcome.APPLIED or (
            self.outcome is OperationOutcome.NO_TRANSITION
            and self.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
        )
        if not can_include_durable_events and self.durable_events:
            raise ValueError("only applied or idempotent-existing admission results may include durable events")
        object.__setattr__(self, "row", copy.deepcopy(self.row) if self.row is not None else None)
        object.__setattr__(self, "durable_events", tuple(copy.deepcopy(event) for event in self.durable_events))

    @property
    def inserted(self) -> bool:
        """Return whether the admission inserted a new row."""

        return self.was_inserted

    @classmethod
    def applied(cls, *, row: dict[str, Any], durable_events: Sequence[dict[str, Any]] = ()) -> AdmissionResult:
        """Build an applied admission result for a newly inserted row."""

        return cls(
            outcome=OperationOutcome.APPLIED,
            row=row,
            was_inserted=True,
            durable_events=durable_events,
        )

    @classmethod
    def existing(
        cls,
        *,
        row: dict[str, Any],
        durable_events: Sequence[dict[str, Any]] = (),
    ) -> AdmissionResult:
        """Build a no-transition result for an idempotent existing row."""

        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            row=row,
            was_inserted=False,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
            durable_events=durable_events,
        )

    @classmethod
    def rejected(cls, reason: AdmissionRejectionReason, *, message: str | None = None) -> AdmissionResult:
        """Build an admission-rejected result with an explicit reason."""

        return cls(
            outcome=OperationOutcome.ADMISSION_REJECTED,
            admission_rejection_reason=reason,
            message=message,
        )


@dataclass(frozen=True)
class LifecycleResult:
    """Result facts produced by a Jobs lifecycle state transition."""

    outcome: OperationOutcome
    transition_applied: bool
    row: dict[str, Any] | None = None
    no_transition_reason: NoTransitionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate lifecycle invariants and freeze mutable facts."""

        if self.outcome is OperationOutcome.APPLIED and not self.transition_applied:
            raise ValueError("applied lifecycle results must mark transition_applied")
        if self.outcome is OperationOutcome.APPLIED and self.row is None:
            raise ValueError("applied lifecycle results require a row")
        if self.outcome is OperationOutcome.NO_TRANSITION and self.no_transition_reason is None:
            raise ValueError("no-transition lifecycle results require a reason")
        if self.outcome is not OperationOutcome.APPLIED and self.transition_applied:
            raise ValueError("only applied lifecycle results may mark transition_applied")
        if self.outcome is not OperationOutcome.NO_TRANSITION and self.no_transition_reason is not None:
            raise ValueError("only no-transition lifecycle results may include a no-transition reason")
        if self.outcome is not OperationOutcome.APPLIED and self.durable_events:
            raise ValueError("only applied lifecycle results may include durable events")
        object.__setattr__(self, "row", copy.deepcopy(self.row) if self.row is not None else None)
        object.__setattr__(self, "durable_events", tuple(copy.deepcopy(event) for event in self.durable_events))

    @classmethod
    def applied(
        cls,
        *,
        row: dict[str, Any],
        durable_events: Sequence[dict[str, Any]] = (),
    ) -> LifecycleResult:
        """Build an applied lifecycle result for a successful transition."""

        return cls(
            outcome=OperationOutcome.APPLIED,
            transition_applied=True,
            row=row,
            durable_events=durable_events,
        )

    @classmethod
    def no_transition(
        cls,
        reason: NoTransitionReason,
        *,
        row: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> LifecycleResult:
        """Build a no-transition lifecycle result with a reason."""

        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=False,
            row=row,
            no_transition_reason=reason,
            message=message,
        )


__all__ = [
    "ADMIN_WEBHOOK_DELIVERY_DOMAIN",
    "ADMIN_WEBHOOK_DELIVERY_IDEMPOTENCY_PREFIX",
    "ADMIN_WEBHOOK_DELIVERY_JOB_TYPE",
    "ADMIN_WEBHOOK_DELIVERY_MAX_RETRIES",
    "ADMIN_WEBHOOK_DELIVERY_PRIORITY",
    "ADMIN_WEBHOOK_DELIVERY_QUARANTINE_THRESHOLD",
    "ADMIN_WEBHOOK_DELIVERY_QUEUE",
    "ApplyPreparedDispositionCommand",
    "AdmissionRejectionReason",
    "AdmissionResult",
    "AdminWebhookDispositionMarker",
    "AcquireJobCommand",
    "BatchRenewLeaseItem",
    "BatchRenewLeasesCommand",
    "BatchRenewLeasesResult",
    "canonical_admin_webhook_delivery_id",
    "canonical_admin_webhook_idempotency_key",
    "canonical_admin_webhook_row_matches",
    "admin_webhook_disposition_marker_matches",
    "CreateJobCommand",
    "EnsureLeaseHorizonCommand",
    "ExpiredLeasePolicy",
    "FindJobByIdentityCommand",
    "JobIdentityLookupResult",
    "JobIdentityLookupState",
    "LeaseHorizonResult",
    "LifecycleResult",
    "NoTransitionReason",
    "OperationOutcome",
    "PreparedDispositionKind",
    "PreparedDispositionOrigin",
    "PreparedDispositionResult",
    "PreparedJobDisposition",
    "prepared_disposition_fingerprint",
    "project_admin_webhook_disposition_marker",
    "reconstruct_legacy_admin_webhook_archive_fingerprint",
    "ReleaseJobCommand",
    "RenewLeaseCommand",
    "is_admin_webhook_delivery_queue",
]
