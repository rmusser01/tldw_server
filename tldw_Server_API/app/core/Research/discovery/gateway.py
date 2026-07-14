"""Credential-free, one-hop transport facade for research discovery."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias
from urllib.parse import quote, urlencode

from tldw_Server_API.app.core.Research.discovery.contracts import (
    AccessRoute,
    CredentialRequirement,
    DispatchIntent,
    canonical_policy_digest,
)
from tldw_Server_API.app.core.Security.http_hop import (
    HTTPHopError,
    HTTPHopLimits,
    HTTPHopResponse,
    NormalizedHTTPHopRequest,
    request_http_hop,
)

GatewayErrorCode = Literal[
    "request_rejected",
    "policy_inactive",
    "hop_failed",
    "invalid_hop_response",
]
OneHop: TypeAlias = Callable[[NormalizedHTTPHopRequest], Awaitable[HTTPHopResponse]]
PolicyActivityCheck: TypeAlias = Callable[[str, str], bool]

_ERROR_MESSAGES: dict[GatewayErrorCode, str] = {
    "request_rejected": "Discovery gateway request rejected",
    "policy_inactive": "Discovery gateway policy inactive",
    "hop_failed": "Discovery gateway hop failed",
    "invalid_hop_response": "Discovery gateway hop response rejected",
}
_MAX_CONTENT_TYPE_BYTES = 256


class DiscoveryGatewayError(Exception):
    """Stable failure without request, response, or provider detail."""

    __slots__ = ("code", "retryable")

    def __init__(self, code: GatewayErrorCode, *, retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(_ERROR_MESSAGES[code])


@dataclass(frozen=True, slots=True)
class DiscoveryGatewayTrace:
    """Public and derived evidence for one physical hop."""

    route_id: str
    policy_digest: str
    scheme: str
    requested_host: str
    tls_server_name: str | None
    port: int
    method: str
    path: str
    query_keys: tuple[str, ...]
    timeout_ms: int
    max_response_bytes: int
    http_limits: HTTPHopLimits
    status_code: int
    resolved_ips: tuple[str, ...]
    connected_ip: str
    response_header_bytes: int
    wire_bytes: int
    decoded_bytes: int
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class DiscoveryGatewayResponse:
    """Bounded body, safe metadata, and evidence from exactly one hop."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    trace: DiscoveryGatewayTrace


def _reject(code: GatewayErrorCode) -> DiscoveryGatewayError:
    return DiscoveryGatewayError(code)


def _validate_binding(route: AccessRoute, intent: DispatchIntent) -> None:
    if not isinstance(route, AccessRoute) or not isinstance(intent, DispatchIntent):
        raise _reject("request_rejected")
    try:
        digest = canonical_policy_digest(route.policy)
    except (TypeError, ValueError):
        raise _reject("request_rejected") from None
    if (
        route.route_id != intent.route_id
        or route.policy.policy_digest != digest
        or intent.policy_digest != digest
        or route.credential_requirement is not CredentialRequirement.NONE
        or intent.method not in route.policy.methods
        or intent.path not in route.policy.paths
        or intent.limits != route.policy.limits
        or any(pair.name not in route.policy.allowed_query_keys for pair in intent.query_pairs)
    ):
        raise _reject("request_rejected")


def _hop_limits(intent: DispatchIntent) -> HTTPHopLimits:
    timeout_seconds = intent.limits.timeout_ms / 1000
    response_bytes = intent.limits.max_response_bytes
    return HTTPHopLimits(
        dns_timeout_seconds=timeout_seconds,
        connect_timeout_seconds=timeout_seconds,
        read_timeout_seconds=timeout_seconds,
        write_timeout_seconds=timeout_seconds,
        total_timeout_seconds=timeout_seconds,
        max_wire_bytes=response_bytes,
        max_decompressed_bytes=response_bytes,
        max_parser_input_bytes=response_bytes,
    )


def _build_request(
    route: AccessRoute,
    intent: DispatchIntent,
) -> NormalizedHTTPHopRequest | None:
    query = urlencode(
        tuple((pair.name, pair.value) for pair in intent.query_pairs),
        doseq=False,
        quote_via=quote,
    )
    target = intent.path if not query else f"{intent.path}?{query}"
    try:
        return NormalizedHTTPHopRequest(
            scheme=route.policy.origin.scheme,
            host=route.policy.origin.host,
            port=route.policy.origin.port,
            method=intent.method,
            target=target,
            headers=(),
            body=b"",
            limits=_hop_limits(intent),
        )
    except (HTTPHopError, TypeError, ValueError):
        return None


def _active(
    check: PolicyActivityCheck,
    route_id: str,
    policy_digest: str,
) -> bool:
    try:
        return check(route_id, policy_digest) is True
    except Exception:  # noqa: BLE001 - revocation checks must fail closed.
        return False


def _safe_headers(headers: tuple[tuple[str, str], ...]) -> tuple[tuple[str, str], ...]:
    for name, value in headers:
        if not isinstance(name, str) or not isinstance(value, str):
            continue
        if name.lower() != "content-type":
            continue
        normalized = value.strip()
        if (
            normalized
            and len(normalized.encode("utf-8")) <= _MAX_CONTENT_TYPE_BYTES
            and all(" " <= character <= "~" for character in normalized)
        ):
            return (("content-type", normalized),)
    return ()


def _valid_ip_evidence(response: HTTPHopResponse) -> bool:
    if (
        not isinstance(response.resolved_ips, tuple)
        or not response.resolved_ips
        or not all(isinstance(address, str) for address in response.resolved_ips)
        or not isinstance(response.connected_ip, str)
        or response.connected_ip not in response.resolved_ips
    ):
        return False
    try:
        tuple(ipaddress.ip_address(address) for address in response.resolved_ips)
        ipaddress.ip_address(response.connected_ip)
    except (TypeError, ValueError):
        return False
    return True


def _valid_headers(headers: object) -> bool:
    return isinstance(headers, tuple) and all(
        isinstance(header, tuple) and len(header) == 2 and all(isinstance(value, str) for value in header)
        for header in headers
    )


def _valid_response(response: object, limits: HTTPHopLimits) -> bool:
    return bool(
        isinstance(response, HTTPHopResponse)
        and isinstance(response.status_code, int)
        and not isinstance(response.status_code, bool)
        and 100 <= response.status_code <= 599
        and _valid_headers(response.headers)
        and isinstance(response.body, bytes)
        and len(response.body) <= limits.max_parser_input_bytes
        and isinstance(response.response_header_bytes, int)
        and not isinstance(response.response_header_bytes, bool)
        and 0 <= response.response_header_bytes <= limits.max_response_header_bytes
        and isinstance(response.wire_bytes, int)
        and not isinstance(response.wire_bytes, bool)
        and 0 <= response.wire_bytes <= limits.max_wire_bytes
        and _valid_ip_evidence(response)
    )


def _requested_host(route: AccessRoute) -> str:
    origin = route.policy.origin
    default_port = 443 if origin.scheme == "https" else 80
    return origin.host if origin.port == default_port else f"{origin.host}:{origin.port}"


async def dispatch_once(
    route: AccessRoute,
    intent: DispatchIntent,
    *,
    is_policy_active: PolicyActivityCheck,
    one_hop: OneHop = request_http_hop,
) -> DiscoveryGatewayResponse:
    """Validate and perform exactly one credential-free physical HTTP hop."""
    _validate_binding(route, intent)
    request = _build_request(route, intent)
    if request is None:
        raise _reject("request_rejected")
    if not _active(is_policy_active, route.route_id, intent.policy_digest):
        raise _reject("policy_inactive")

    started_at = time.monotonic()
    mapped_error: DiscoveryGatewayError | None = None
    response: HTTPHopResponse | None = None
    try:
        response = await one_hop(request)
    except HTTPHopError as error:
        mapped_error = DiscoveryGatewayError("hop_failed", retryable=error.retryable)
    except Exception:  # noqa: BLE001 - never expose unexpected provider detail.
        mapped_error = DiscoveryGatewayError("hop_failed")
    elapsed_ms = max(0, int((time.monotonic() - started_at) * 1000))
    if mapped_error is not None:
        raise mapped_error
    if not _valid_response(response, request.limits) or response is None:
        raise _reject("invalid_hop_response")

    trace = DiscoveryGatewayTrace(
        route_id=route.route_id,
        policy_digest=intent.policy_digest,
        scheme=route.policy.origin.scheme,
        requested_host=_requested_host(route),
        tls_server_name=(route.policy.origin.host if route.policy.origin.scheme == "https" else None),
        port=route.policy.origin.port,
        method=intent.method,
        path=intent.path,
        query_keys=tuple(pair.name for pair in intent.query_pairs),
        timeout_ms=intent.limits.timeout_ms,
        max_response_bytes=intent.limits.max_response_bytes,
        http_limits=request.limits,
        status_code=response.status_code,
        resolved_ips=response.resolved_ips,
        connected_ip=response.connected_ip,
        response_header_bytes=response.response_header_bytes,
        wire_bytes=response.wire_bytes,
        decoded_bytes=len(response.body),
        elapsed_ms=elapsed_ms,
    )
    return DiscoveryGatewayResponse(
        status_code=response.status_code,
        headers=_safe_headers(response.headers),
        body=response.body,
        trace=trace,
    )


__all__ = [
    "DiscoveryGatewayError",
    "DiscoveryGatewayResponse",
    "DiscoveryGatewayTrace",
    "dispatch_once",
]
