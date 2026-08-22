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
from datetime import datetime
from enum import Enum
from typing import Any
from urllib.parse import SplitResult, urlsplit

from tldw_Server_API.app.core.Security.egress import (
    evaluate_platform_webhook_url_policy,
)


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
    USER_PRINCIPAL_REQUIRED = "admin_webhook_user_principal_required"
    DELIVERY_UNAVAILABLE = "admin_webhook_delivery_unavailable"


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
    WebhookErrorCode.USER_PRINCIPAL_REQUIRED: 403,
    WebhookErrorCode.DELIVERY_UNAVAILABLE: 503,
    WebhookErrorCode.REQUEST_REJECTED: 400,
}


class WebhookError(Exception):
    """Expected domain failure with no caller-controlled message text."""

    def __init__(
        self,
        code: WebhookErrorCode,
        http_status: int | None = None,
    ) -> None:
        self.code = code
        self.http_status = http_status or _ERROR_STATUS[code]
        super().__init__(code.value)


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
    """Sanitized PR 1 status projection."""

    mode: str
    route_selection: str
    schema_ready: bool
    key_state: str
    delivery_capability_ready: bool
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
