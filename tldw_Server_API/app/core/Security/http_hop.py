"""Strict contracts and DNS policy for a single outbound HTTP request hop."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from tldw_Server_API.app.core.Security.egress import is_private_ip, resolve_host_ips

HTTPHopErrorCode = Literal[
    "invalid_request",
    "dns_resolution_failed",
    "dns_timeout",
    "dns_address_denied",
    "connect_timeout",
    "read_timeout",
    "write_timeout",
    "total_timeout",
    "peer_verification_failed",
    "tls_error",
    "protocol_error",
    "response_headers_too_large",
    "response_too_large",
    "decompressed_response_too_large",
    "parser_input_too_large",
    "unsupported_content_encoding",
    "invalid_content_encoding",
    "transport_error",
]

_ERROR_MESSAGES: dict[HTTPHopErrorCode, str] = {
    "invalid_request": "The outbound request is invalid.",
    "dns_resolution_failed": "The destination could not be resolved.",
    "dns_timeout": "Destination resolution timed out.",
    "dns_address_denied": "The destination address is not allowed.",
    "connect_timeout": "The destination connection timed out.",
    "read_timeout": "The destination response timed out.",
    "write_timeout": "The outbound request timed out.",
    "total_timeout": "The outbound request exceeded its time limit.",
    "peer_verification_failed": "The connected destination could not be verified.",
    "tls_error": "The secure destination connection failed.",
    "protocol_error": "The destination returned an invalid response.",
    "response_headers_too_large": "The destination response headers are too large.",
    "response_too_large": "The destination response is too large.",
    "decompressed_response_too_large": "The decoded destination response is too large.",
    "parser_input_too_large": "The destination response exceeds the parser limit.",
    "unsupported_content_encoding": "The destination used an unsupported content encoding.",
    "invalid_content_encoding": "The destination returned invalid encoded content.",
    "transport_error": "The destination request failed.",
}

_FLOAT_LIMIT_FIELDS = (
    "dns_timeout_seconds",
    "connect_timeout_seconds",
    "read_timeout_seconds",
    "write_timeout_seconds",
    "total_timeout_seconds",
)
_INTEGER_LIMIT_FIELDS = (
    "max_request_target_bytes",
    "max_request_header_bytes",
    "max_request_headers",
    "max_request_body_bytes",
    "max_response_header_bytes",
    "max_response_headers",
    "max_wire_bytes",
    "max_decompressed_bytes",
    "max_parser_input_bytes",
)
_FORBIDDEN_HEADERS = frozenset(
    {
        "accept-encoding",
        "connection",
        "content-length",
        "expect",
        "host",
        "keep-alive",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_HEADER_NAME_PATTERN = re.compile(r"[!#$%&'*+.^_`|~0-9a-z-]+\Z")
_DNS_LABEL_PATTERN = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z")
_LEGACY_NUMERIC_HOST_PATTERN = re.compile(r"(?:0x[0-9a-f]+|[0-9]+)(?:\.(?:0x[0-9a-f]+|[0-9]+)){0,3}\Z")
_TARGET_PATTERN = re.compile(r"(?:[A-Za-z0-9._~!$&'()*+,;=:@/?-]|%[0-9A-Fa-f]{2})*\Z")
_DENIED_IPV6_TRANSITION_NETWORKS = (
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/32"),
    ipaddress.ip_network("2002::/16"),
)


class HTTPHopError(Exception):
    """A stable, sanitized failure from the one-hop HTTP boundary."""

    def __init__(self, code: HTTPHopErrorCode, *, retryable: bool = False) -> None:
        message = _ERROR_MESSAGES.get(code)
        if message is None:
            raise ValueError("Unsupported HTTP hop error code")
        if not isinstance(retryable, bool):
            raise TypeError("retryable must be a boolean")
        self.code = code
        self.retryable = retryable
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class HTTPHopLimits:
    """Finite positive ceilings applied to one physical HTTP request."""

    dns_timeout_seconds: float = 2.0
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 10.0
    write_timeout_seconds: float = 5.0
    total_timeout_seconds: float = 20.0
    max_request_target_bytes: int = 8 * 1024
    max_request_header_bytes: int = 16 * 1024
    max_request_headers: int = 64
    max_request_body_bytes: int = 1024 * 1024
    max_response_header_bytes: int = 64 * 1024
    max_response_headers: int = 128
    max_wire_bytes: int = 2 * 1024 * 1024
    max_decompressed_bytes: int = 4 * 1024 * 1024
    max_parser_input_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in _FLOAT_LIMIT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in _INTEGER_LIMIT_FIELDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_parser_input_bytes > self.max_decompressed_bytes:
            raise ValueError("max_parser_input_bytes cannot exceed max_decompressed_bytes")


@dataclass(frozen=True, slots=True)
class NormalizedHTTPHopRequest:
    """Canonical, bounded input for exactly one HTTP request hop."""

    scheme: Literal["http", "https"]
    host: str
    port: int
    method: Literal["GET", "HEAD", "POST"]
    target: str
    headers: tuple[tuple[str, str], ...] = ()
    body: bytes = b""
    limits: HTTPHopLimits = field(default_factory=HTTPHopLimits)

    def __post_init__(self) -> None:
        invalid = False
        try:
            _validate_request(self)
        except HTTPHopError:
            raise
        except (TypeError, ValueError, UnicodeError):
            invalid = True
        if invalid:
            raise HTTPHopError("invalid_request")


@dataclass(frozen=True, slots=True)
class HTTPHopResponse:
    """Bounded response evidence returned by the one-hop HTTP primitive."""

    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    resolved_ips: tuple[str, ...]
    connected_ip: str
    response_header_bytes: int
    wire_bytes: int


_Resolver = Callable[[str, int, float], Awaitable[Sequence[str]]]


def _invalid_request() -> HTTPHopError:
    return HTTPHopError("invalid_request")


def _validate_request(request: NormalizedHTTPHopRequest) -> None:
    if request.scheme not in {"http", "https"}:
        raise _invalid_request()
    _validate_host(request.host)
    if isinstance(request.port, bool) or not isinstance(request.port, int):
        raise _invalid_request()
    if not 1 <= request.port <= 65535:
        raise _invalid_request()
    if request.method not in {"GET", "HEAD", "POST"}:
        raise _invalid_request()
    if not isinstance(request.limits, HTTPHopLimits):
        raise _invalid_request()
    _validate_target(request.target, request.limits)
    _validate_headers(request.headers, request.limits)
    if not isinstance(request.body, bytes):
        raise _invalid_request()
    if len(request.body) > request.limits.max_request_body_bytes:
        raise _invalid_request()
    if request.method in {"GET", "HEAD"} and request.body:
        raise _invalid_request()


def _validate_host(host: str) -> None:
    if not isinstance(host, str) or not host or len(host) > 253:
        raise _invalid_request()
    if not host.isascii() or host != host.lower():
        raise _invalid_request()
    if any(ord(character) <= 32 or ord(character) == 127 for character in host):
        raise _invalid_request()
    if any(character in host for character in "%@[]/\\"):
        raise _invalid_request()

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address is not None:
        if address.compressed != host:
            raise _invalid_request()
        return

    if ":" in host or _LEGACY_NUMERIC_HOST_PATTERN.fullmatch(host):
        raise _invalid_request()
    labels = host.split(".")
    if any(not _DNS_LABEL_PATTERN.fullmatch(label) for label in labels):
        raise _invalid_request()
    canonical_idna: str | None
    try:
        canonical_idna = host.encode("ascii").decode("idna").encode("idna").decode("ascii")
    except UnicodeError:
        canonical_idna = None
    if canonical_idna != host:
        raise _invalid_request()


def _validate_target(target: str, limits: HTTPHopLimits) -> None:
    if not isinstance(target, str) or not target.startswith("/") or target.startswith("//"):
        raise _invalid_request()
    if not _TARGET_PATTERN.fullmatch(target):
        raise _invalid_request()
    if len(target.encode("ascii")) > limits.max_request_target_bytes:
        raise _invalid_request()


def _validate_headers(
    headers: tuple[tuple[str, str], ...],
    limits: HTTPHopLimits,
) -> None:
    if not isinstance(headers, tuple) or len(headers) > limits.max_request_headers:
        raise _invalid_request()
    seen: set[str] = set()
    byte_count = 0
    for pair in headers:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise _invalid_request()
        name, value = pair
        if not isinstance(name, str) or not isinstance(value, str):
            raise _invalid_request()
        if (
            not name.isascii()
            or name != name.lower()
            or not _HEADER_NAME_PATTERN.fullmatch(name)
            or name in seen
            or name in _FORBIDDEN_HEADERS
        ):
            raise _invalid_request()
        if not value.isascii() or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise _invalid_request()
        seen.add(name)
        byte_count += len(name) + 2 + len(value) + 2
        if byte_count > limits.max_request_header_bytes:
            raise _invalid_request()


def _is_allowed_public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if is_private_ip(address.compressed):
        return False
    if (
        not address.is_global
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
        or address.is_loopback
        or address.is_link_local
        or address.is_private
    ):
        return False
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None or address.is_site_local:
            return False
        if any(address in network for network in _DENIED_IPV6_TRANSITION_NETWORKS):
            return False
    return True


def _validate_resolved_ips(resolved_ips: Sequence[str]) -> tuple[str, ...]:
    """Validate the complete DNS answer set, preserving resolver order."""
    if isinstance(resolved_ips, (str, bytes)) or not resolved_ips:
        raise HTTPHopError("dns_resolution_failed", retryable=True)
    validated: list[str] = []
    seen: set[str] = set()
    for raw_ip in resolved_ips:
        if not isinstance(raw_ip, str) or "%" in raw_ip:
            raise HTTPHopError("dns_address_denied")
        address: ipaddress.IPv4Address | ipaddress.IPv6Address | None
        try:
            address = ipaddress.ip_address(raw_ip)
        except ValueError:
            address = None
        if address is None:
            raise HTTPHopError("dns_address_denied")
        canonical = address.compressed
        if raw_ip != canonical or not _is_allowed_public_address(address):
            raise HTTPHopError("dns_address_denied")
        if canonical not in seen:
            seen.add(canonical)
            validated.append(canonical)
    if not validated:
        raise HTTPHopError("dns_resolution_failed", retryable=True)
    return tuple(validated)


async def _default_resolver(
    host: str,
    _port: int,
    timeout_seconds: float,
) -> Sequence[str]:
    """Run the bounded blocking resolver away from the event loop."""
    return await asyncio.to_thread(resolve_host_ips, host, timeout_s=timeout_seconds)


async def _resolve_validated_ips(
    request: NormalizedHTTPHopRequest,
    *,
    resolver: _Resolver = _default_resolver,
) -> tuple[str, ...]:
    """Resolve once and reject the complete answer set if any address is unsafe."""
    try:
        literal = ipaddress.ip_address(request.host)
    except ValueError:
        literal = None
    if literal is not None:
        return _validate_resolved_ips((request.host,))

    failure: HTTPHopError | None = None
    try:
        resolved = await asyncio.wait_for(
            resolver(request.host, request.port, request.limits.dns_timeout_seconds),
            timeout=request.limits.dns_timeout_seconds,
        )
    except asyncio.TimeoutError:
        failure = HTTPHopError("dns_timeout", retryable=True)
    except HTTPHopError:
        raise
    except Exception:  # noqa: BLE001 - resolver failures must cross a sanitized boundary
        failure = HTTPHopError("dns_resolution_failed", retryable=True)
    if failure is not None:
        raise failure
    return _validate_resolved_ips(resolved)


__all__ = [
    "HTTPHopError",
    "HTTPHopErrorCode",
    "HTTPHopLimits",
    "HTTPHopResponse",
    "NormalizedHTTPHopRequest",
]
