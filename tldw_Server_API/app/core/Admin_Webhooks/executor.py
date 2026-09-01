"""One-attempt outbound boundary for canonical admin webhooks."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import math
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol, TypeAlias
from urllib.parse import quote, urlunsplit
from uuid import UUID

from tldw_Server_API.app.core.exceptions import HTTPHopError
from tldw_Server_API.app.core.Security.egress import (
    evaluate_admin_webhook_e2e_loopback_policy,
    evaluate_platform_webhook_url_policy,
)
from tldw_Server_API.app.core.Security.http_hop import (
    HTTPHopLimits,
    NormalizedHTTPHopRequest,
    StatusOnlyHTTPHopResponse,
    request_admin_webhook_e2e_loopback_status,
    request_http_hop_status,
)

from .domain import DeliveryKind, ValidatedWebhookTarget
from .target import normalize_webhook_hostname, parse_webhook_target_url

_MAX_BODY_BYTES = 64 * 1_024
_MAX_TIMEOUT_SECONDS = 30
_RETRY_DELAYS_SECONDS = (60, 300, 1_800)
_SIGNING_SECRET_PATTERN = re.compile(r"whsec_[0-9a-f]{64}\Z")
_EVENT_TYPE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,63}\Z")
_PERCENT_ESCAPE_PATTERN = re.compile(r"%[0-9A-Fa-f]{2}")
_PATH_SAFE = "/!$&'()*+,;=:@-._~%"
_QUERY_SAFE = "/?!$&'()*+,;=:@-._~%"


class AttemptOutcome(str, Enum):
    """Closed outcome from exactly one delivery attempt."""

    SUCCESS = "success"
    RETRYABLE = "retryable"
    FAILED = "failed"


class AttemptReasonCode(str, Enum):
    """Sanitized reasons owned by the one-attempt boundary."""

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
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
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


@dataclass(frozen=True, slots=True)
class AttemptExecutionRequest:
    """Immutable reviewed inputs for one outbound webhook attempt."""

    target: ValidatedWebhookTarget = field(repr=False)
    body: bytes = field(repr=False)
    signing_secret: str = field(repr=False)
    timeout_seconds: int
    event_type: str
    event_id: str
    delivery_id: str
    attempt_number: int
    secret_version: int
    kind: DeliveryKind


@dataclass(frozen=True, slots=True)
class AttemptExecutionResult:
    """Bounded attempt evidence safe to pass to durable lifecycle code."""

    outcome: AttemptOutcome
    status_code: int | None
    latency_ms: int | None
    reason_code: AttemptReasonCode | None
    retry_delay_seconds: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, AttemptOutcome):
            raise ValueError("outcome is invalid")
        if self.status_code is not None and (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status code is invalid")
        if self.latency_ms is not None and (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, int)
            or self.latency_ms < 0
        ):
            raise ValueError("latency is invalid")
        if self.reason_code is not None and not isinstance(
            self.reason_code,
            AttemptReasonCode,
        ):
            raise ValueError("reason code is invalid")
        if self.retry_delay_seconds is not None and (
            isinstance(self.retry_delay_seconds, bool)
            or not isinstance(self.retry_delay_seconds, int)
            or not 1 <= self.retry_delay_seconds <= 1_800
        ):
            raise ValueError("retry delay is invalid")


class _Clock(Protocol):
    def monotonic(self) -> float: ...

    def utc_now(self) -> datetime: ...


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)


_Egress: TypeAlias = Callable[
    [NormalizedHTTPHopRequest],
    Awaitable[StatusOnlyHTTPHopResponse],
]


@dataclass(frozen=True, slots=True)
class _NormalizedTarget:
    url: str = field(repr=False)
    scheme: str
    host: str
    port: int
    request_target: str = field(repr=False)


_HTTP_HOP_REASONS = {
    "invalid_request": AttemptReasonCode.HTTP_HOP_INVALID_REQUEST,
    "dns_resolution_failed": AttemptReasonCode.HTTP_HOP_DNS_RESOLUTION_FAILED,
    "dns_timeout": AttemptReasonCode.HTTP_HOP_DNS_TIMEOUT,
    "dns_address_denied": AttemptReasonCode.HTTP_HOP_DNS_ADDRESS_DENIED,
    "connect_timeout": AttemptReasonCode.HTTP_HOP_CONNECT_TIMEOUT,
    "read_timeout": AttemptReasonCode.HTTP_HOP_READ_TIMEOUT,
    "write_timeout": AttemptReasonCode.HTTP_HOP_WRITE_TIMEOUT,
    "total_timeout": AttemptReasonCode.HTTP_HOP_TOTAL_TIMEOUT,
    "peer_verification_failed": AttemptReasonCode.HTTP_HOP_PEER_VERIFICATION_FAILED,
    "tls_error": AttemptReasonCode.HTTP_HOP_TLS_ERROR,
    "protocol_error": AttemptReasonCode.HTTP_HOP_PROTOCOL_ERROR,
    "response_headers_too_large": AttemptReasonCode.HTTP_HOP_RESPONSE_HEADERS_TOO_LARGE,
    "response_too_large": AttemptReasonCode.HTTP_HOP_RESPONSE_TOO_LARGE,
    "decompressed_response_too_large": (
        AttemptReasonCode.HTTP_HOP_DECOMPRESSED_RESPONSE_TOO_LARGE
    ),
    "parser_input_too_large": AttemptReasonCode.HTTP_HOP_PARSER_INPUT_TOO_LARGE,
    "unsupported_content_encoding": (
        AttemptReasonCode.HTTP_HOP_UNSUPPORTED_CONTENT_ENCODING
    ),
    "invalid_content_encoding": AttemptReasonCode.HTTP_HOP_INVALID_CONTENT_ENCODING,
    "transport_error": AttemptReasonCode.HTTP_HOP_TRANSPORT_ERROR,
}
_RETRYABLE_HTTP_HOP_CODES = frozenset(
    {
        "dns_resolution_failed",
        "dns_timeout",
        "connect_timeout",
        "read_timeout",
        "write_timeout",
        "total_timeout",
        "tls_error",
        "transport_error",
    }
)


def _validate_uuid4(value: object, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical UUIDv4")
    try:
        parsed = UUID(value)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a canonical UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{name} must be a canonical UUIDv4")
    return value


def _validate_execution_request(request: AttemptExecutionRequest) -> None:
    if not isinstance(request, AttemptExecutionRequest):
        raise TypeError("request must be an AttemptExecutionRequest")
    if (
        isinstance(request.attempt_number, bool)
        or not isinstance(request.attempt_number, int)
        or not 1 <= request.attempt_number <= 4
    ):
        raise ValueError("attempt number must be an integer from 1 through 4")
    if not isinstance(request.target, ValidatedWebhookTarget):
        raise ValueError("target must be a validated webhook target")
    if type(request.body) is not bytes or len(request.body) > _MAX_BODY_BYTES:
        raise ValueError("body must be bounded immutable bytes")
    if (
        not isinstance(request.signing_secret, str)
        or _SIGNING_SECRET_PATTERN.fullmatch(request.signing_secret) is None
    ):
        raise ValueError("signing secret is invalid")
    if (
        isinstance(request.timeout_seconds, bool)
        or not isinstance(request.timeout_seconds, int)
        or not 1 <= request.timeout_seconds <= _MAX_TIMEOUT_SECONDS
    ):
        raise ValueError("timeout must be an integer from 1 through 30")
    if (
        not isinstance(request.event_type, str)
        or _EVENT_TYPE_PATTERN.fullmatch(request.event_type) is None
    ):
        raise ValueError("event type is invalid")
    event_id = _validate_uuid4(request.event_id, name="event id")
    delivery_id = _validate_uuid4(request.delivery_id, name="delivery id")
    if event_id == delivery_id:
        raise ValueError("event and delivery ids must differ")
    if (
        isinstance(request.secret_version, bool)
        or not isinstance(request.secret_version, int)
        or not 1 <= request.secret_version <= 2_147_483_647
    ):
        raise ValueError("secret version is invalid")
    if not isinstance(request.kind, DeliveryKind):
        raise ValueError("delivery kind is invalid")


def _uppercase_percent_escapes(value: str) -> str:
    return _PERCENT_ESCAPE_PATTERN.sub(lambda match: match.group(0).upper(), value)


def _normalize_target(
    target: ValidatedWebhookTarget,
    *,
    allow_http_dev: bool,
) -> _NormalizedTarget:
    url = target.url
    try:
        parsed, host = parse_webhook_target_url(url)
        expected_host = normalize_webhook_hostname(target.hostname)
    except ValueError as exc:
        raise ValueError("target URL is invalid") from exc

    scheme = parsed.scheme.lower()
    if scheme != "https" and not (scheme == "http" and allow_http_dev):
        raise ValueError("target URL scheme is invalid")
    if expected_host != host:
        raise ValueError("target hostname does not match URL")
    port = parsed.port
    if port is not None and not 1 <= port <= 65_535:
        raise ValueError("target port is invalid")
    effective_port = port if port is not None else (443 if scheme == "https" else 80)

    try:
        path = _uppercase_percent_escapes(quote(parsed.path or "/", safe=_PATH_SAFE))
        query = _uppercase_percent_escapes(quote(parsed.query, safe=_QUERY_SAFE))
    except UnicodeError as exc:
        raise ValueError("target URL is invalid") from exc
    request_target = path if not query else f"{path}?{query}"
    if request_target.startswith("//"):
        raise ValueError("target request path is invalid")

    authority_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    authority = (
        authority_host
        if effective_port == default_port
        else f"{authority_host}:{effective_port}"
    )
    normalized_url = urlunsplit((scheme, authority, path, query, ""))
    return _NormalizedTarget(
        url=normalized_url,
        scheme=scheme,
        host=host,
        port=effective_port,
        request_target=request_target,
    )


def _clock_utc_timestamp(clock: _Clock) -> int:
    try:
        value = clock.utc_now()
        offset = value.utcoffset() if isinstance(value, datetime) else None
        timestamp = value.timestamp() if offset == timedelta(0) else None
    except Exception:  # noqa: BLE001 - injected clock detail is not boundary-safe
        raise ValueError("clock error") from None
    if timestamp is None or not math.isfinite(timestamp):
        raise ValueError("clock error")
    return int(timestamp)


def _clock_monotonic(clock: _Clock) -> float:
    try:
        value = clock.monotonic()
    except Exception:  # noqa: BLE001 - injected clock detail is not boundary-safe
        raise ValueError("clock error") from None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError("clock error")
    return float(value)


def _elapsed_ms(clock: _Clock, started: float) -> int:
    finished = _clock_monotonic(clock)
    elapsed = (finished - started) * 1_000
    if finished < started or not math.isfinite(elapsed):
        raise ValueError("clock error")
    return int(elapsed)


def _retry_result(
    request: AttemptExecutionRequest,
    *,
    reason: AttemptReasonCode,
    status_code: int | None = None,
    latency_ms: int | None = None,
    receiver_delay_seconds: int | None = None,
) -> AttemptExecutionResult:
    if request.attempt_number == 4:
        return AttemptExecutionResult(
            outcome=AttemptOutcome.FAILED,
            status_code=status_code,
            latency_ms=latency_ms,
            reason_code=AttemptReasonCode.ATTEMPT_BUDGET_EXHAUSTED,
            retry_delay_seconds=None,
        )
    delay = _RETRY_DELAYS_SECONDS[request.attempt_number - 1]
    if (
        isinstance(receiver_delay_seconds, int)
        and not isinstance(receiver_delay_seconds, bool)
        and 1 <= receiver_delay_seconds <= 1_800
    ):
        delay = max(delay, receiver_delay_seconds)
    return AttemptExecutionResult(
        outcome=AttemptOutcome.RETRYABLE,
        status_code=status_code,
        latency_ms=latency_ms,
        reason_code=reason,
        retry_delay_seconds=delay,
    )


def _failed_result(
    reason: AttemptReasonCode,
    *,
    status_code: int | None = None,
    latency_ms: int | None = None,
) -> AttemptExecutionResult:
    return AttemptExecutionResult(
        outcome=AttemptOutcome.FAILED,
        status_code=status_code,
        latency_ms=latency_ms,
        reason_code=reason,
        retry_delay_seconds=None,
    )


def _status_result(
    request: AttemptExecutionRequest,
    response: StatusOnlyHTTPHopResponse,
    *,
    latency_ms: int,
) -> AttemptExecutionResult:
    status = response.status_code
    if 200 <= status <= 299:
        return AttemptExecutionResult(
            outcome=AttemptOutcome.SUCCESS,
            status_code=status,
            latency_ms=latency_ms,
            reason_code=None,
            retry_delay_seconds=None,
        )
    if 300 <= status <= 399:
        return _failed_result(
            AttemptReasonCode.HTTP_REDIRECT,
            status_code=status,
            latency_ms=latency_ms,
        )
    if status == 408:
        return _retry_result(
            request,
            reason=AttemptReasonCode.HTTP_REQUEST_TIMEOUT,
            status_code=status,
            latency_ms=latency_ms,
        )
    if status == 429:
        return _retry_result(
            request,
            reason=AttemptReasonCode.HTTP_RATE_LIMITED,
            status_code=status,
            latency_ms=latency_ms,
            receiver_delay_seconds=response.retry_after_seconds,
        )
    if 400 <= status <= 499:
        return _failed_result(
            AttemptReasonCode.HTTP_CLIENT_ERROR,
            status_code=status,
            latency_ms=latency_ms,
        )
    if 500 <= status <= 599:
        return _retry_result(
            request,
            reason=AttemptReasonCode.HTTP_SERVER_ERROR,
            status_code=status,
            latency_ms=latency_ms,
            receiver_delay_seconds=(
                response.retry_after_seconds if status == 503 else None
            ),
        )
    return _failed_result(
        AttemptReasonCode.HTTP_STATUS_INVALID,
        status_code=status,
        latency_ms=latency_ms,
    )


class DeliveryAttemptExecutor:
    """Revalidate, sign, and execute exactly one canonical webhook request."""

    def __init__(
        self,
        *,
        egress: _Egress = request_http_hop_status,
        clock: _Clock | None = None,
        allow_http_dev: bool = False,
        allow_e2e_loopback: bool = False,
    ) -> None:
        if not callable(egress):
            raise TypeError("egress must be callable")
        if clock is not None and (
            not callable(getattr(clock, "monotonic", None))
            or not callable(getattr(clock, "utc_now", None))
        ):
            raise TypeError("clock must provide monotonic and utc_now")
        if not isinstance(allow_http_dev, bool):
            raise TypeError("allow_http_dev must be a boolean")
        if not isinstance(allow_e2e_loopback, bool):
            raise TypeError("allow_e2e_loopback must be a boolean")
        if allow_e2e_loopback and not allow_http_dev:
            raise ValueError("allow_e2e_loopback requires allow_http_dev")
        self._egress = (
            request_admin_webhook_e2e_loopback_status
            if allow_e2e_loopback and egress is request_http_hop_status
            else egress
        )
        self._clock = clock or _SystemClock()
        self._allow_http_dev = allow_http_dev
        self._allow_e2e_loopback = allow_e2e_loopback

    async def execute(
        self,
        request: AttemptExecutionRequest,
    ) -> AttemptExecutionResult:
        """Execute one request and return only bounded disposition evidence."""
        _validate_execution_request(request)
        try:
            target = _normalize_target(
                request.target,
                allow_http_dev=self._allow_http_dev,
            )
        except ValueError:
            return _failed_result(AttemptReasonCode.TARGET_INVALID)

        try:
            policy = (
                evaluate_admin_webhook_e2e_loopback_policy(target.url)
                if self._allow_e2e_loopback
                else evaluate_platform_webhook_url_policy(target.url)
            )
        except Exception:  # noqa: BLE001 - policy detail must not cross this boundary
            return _retry_result(request, reason=AttemptReasonCode.POLICY_ERROR)
        if policy.allowed is not True:
            return _failed_result(AttemptReasonCode.TARGET_REJECTED)

        try:
            timestamp = _clock_utc_timestamp(self._clock)
        except ValueError:
            return _retry_result(request, reason=AttemptReasonCode.CLOCK_ERROR)
        signature = hmac.new(
            request.signing_secret.encode("ascii"),
            str(timestamp).encode("ascii") + b"." + request.body,
            hashlib.sha256,
        ).hexdigest()
        headers = (
            ("content-type", "application/json"),
            ("x-tldw-webhook-event", request.event_type),
            ("x-tldw-webhook-event-id", request.event_id),
            ("x-tldw-webhook-delivery-id", request.delivery_id),
            ("x-tldw-webhook-timestamp", str(timestamp)),
            ("x-tldw-webhook-secret-version", str(request.secret_version)),
            ("x-tldw-webhook-signature", f"v1={signature}"),
        )
        if request.kind is DeliveryKind.TEST:
            headers += (("x-tldw-webhook-test", "true"),)

        timeout = float(request.timeout_seconds)
        try:
            hop_request = NormalizedHTTPHopRequest(
                scheme=target.scheme,  # type: ignore[arg-type]
                host=target.host,
                port=target.port,
                method="POST",
                target=target.request_target,
                headers=headers,
                body=request.body,
                limits=HTTPHopLimits(
                    dns_timeout_seconds=min(2.0, timeout),
                    connect_timeout_seconds=min(5.0, timeout),
                    read_timeout_seconds=timeout,
                    write_timeout_seconds=min(5.0, timeout),
                    total_timeout_seconds=timeout,
                    max_request_body_bytes=_MAX_BODY_BYTES,
                ),
            )
        except (HTTPHopError, TypeError, ValueError, UnicodeError):
            return _failed_result(AttemptReasonCode.TARGET_INVALID)
        try:
            started = _clock_monotonic(self._clock)
        except ValueError:
            return _retry_result(request, reason=AttemptReasonCode.CLOCK_ERROR)

        try:
            response = await self._egress(hop_request)
        except asyncio.CancelledError:
            raise
        except HTTPHopError as exc:
            try:
                latency_ms = _elapsed_ms(self._clock, started)
            except ValueError:
                return _retry_result(request, reason=AttemptReasonCode.CLOCK_ERROR)
            reason = _HTTP_HOP_REASONS.get(
                exc.code,
                AttemptReasonCode.TRANSPORT_ERROR,
            )
            if exc.code in _RETRYABLE_HTTP_HOP_CODES:
                return _retry_result(
                    request,
                    reason=reason,
                    latency_ms=latency_ms,
                )
            return _failed_result(reason, latency_ms=latency_ms)
        except Exception:  # noqa: BLE001 - exception text must not cross this boundary
            try:
                latency_ms = _elapsed_ms(self._clock, started)
            except ValueError:
                return _retry_result(request, reason=AttemptReasonCode.CLOCK_ERROR)
            return _retry_result(
                request,
                reason=AttemptReasonCode.TRANSPORT_ERROR,
                latency_ms=latency_ms,
            )

        try:
            latency_ms = _elapsed_ms(self._clock, started)
        except ValueError:
            return _retry_result(request, reason=AttemptReasonCode.CLOCK_ERROR)
        if not isinstance(response, StatusOnlyHTTPHopResponse):
            return _retry_result(
                request,
                reason=AttemptReasonCode.TRANSPORT_ERROR,
                latency_ms=latency_ms,
            )
        return _status_result(request, response, latency_ms=latency_ms)


__all__ = [
    "AttemptExecutionRequest",
    "AttemptExecutionResult",
    "AttemptOutcome",
    "AttemptReasonCode",
    "DeliveryAttemptExecutor",
]
