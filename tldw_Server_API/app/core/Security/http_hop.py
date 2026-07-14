"""Strict contracts and DNS policy for a single outbound HTTP request hop."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import re
import ssl
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import certifi
import httpcore

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
_MAX_NETWORK_READ_BYTES = 16 * 1024


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


@dataclass(slots=True)
class _PeerEvidence:
    connected_ip: str | None = None


def _verified_peer_ip(
    peer: object,
    *,
    selected_ip: str,
    expected_port: int,
) -> str:
    """Return canonical peer evidence or fail closed without exposing metadata."""
    if not isinstance(peer, tuple) or len(peer) not in {2, 4}:
        raise HTTPHopError("peer_verification_failed")
    peer_host, peer_port = peer[0], peer[1]
    if not isinstance(peer_host, str) or "%" in peer_host:
        raise HTTPHopError("peer_verification_failed")
    if isinstance(peer_port, bool) or not isinstance(peer_port, int):
        raise HTTPHopError("peer_verification_failed")
    if peer_port != expected_port:
        raise HTTPHopError("peer_verification_failed")
    if len(peer) == 4:
        flow_info, scope_id = peer[2], peer[3]
        if (
            isinstance(flow_info, bool)
            or not isinstance(flow_info, int)
            or isinstance(scope_id, bool)
            or not isinstance(scope_id, int)
            or scope_id != 0
        ):
            raise HTTPHopError("peer_verification_failed")

    peer_address: ipaddress.IPv4Address | ipaddress.IPv6Address | None
    try:
        peer_address = ipaddress.ip_address(peer_host)
    except ValueError:
        peer_address = None
    selected_address = ipaddress.ip_address(selected_ip)
    if peer_address is None or peer_address != selected_address:
        raise HTTPHopError("peer_verification_failed")
    if len(peer) == 4 and not isinstance(peer_address, ipaddress.IPv6Address):
        raise HTTPHopError("peer_verification_failed")
    return peer_address.compressed


async def _close_stream_quietly(stream: httpcore.AsyncNetworkStream) -> None:
    """Best-effort close that never replaces the stable boundary error."""
    try:
        await stream.aclose()
    except asyncio.CancelledError:
        raise
    except Exception:  # noqa: BLE001 - cleanup errors must not expose backend detail
        return


def _peer_from_stream(
    stream: httpcore.AsyncNetworkStream,
    *,
    selected_ip: str,
    expected_port: int,
) -> str:
    peer: object | None = None
    metadata_failed = False
    try:
        peer = stream.get_extra_info("server_addr")
    except Exception:  # noqa: BLE001 - peer metadata crosses a sanitized boundary
        metadata_failed = True
    if metadata_failed:
        raise HTTPHopError("peer_verification_failed")
    return _verified_peer_ip(
        peer,
        selected_ip=selected_ip,
        expected_port=expected_port,
    )


class _PeerVerifiedStream(httpcore.AsyncNetworkStream):
    """Delegate stream that preserves route identity and rechecks TLS peers."""

    def __init__(
        self,
        stream: httpcore.AsyncNetworkStream,
        *,
        scheme: Literal["http", "https"],
        route_host: str,
        selected_ip: str,
        expected_port: int,
        tls_context: ssl.SSLContext | None,
        evidence: _PeerEvidence,
        tls_active: bool = False,
        ssl_object: object | None = None,
    ) -> None:
        self._stream = stream
        self._scheme = scheme
        self._route_host = route_host
        self._selected_ip = selected_ip
        self._expected_port = expected_port
        self._tls_context = tls_context
        self._evidence = evidence
        self._tls_active = tls_active
        self._tls_started = tls_active
        self._ssl_object = ssl_object

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._scheme == "https" and not self._tls_active:
            await _close_stream_quietly(self._stream)
            raise HTTPHopError("peer_verification_failed")
        return await self._stream.read(
            min(max_bytes, _MAX_NETWORK_READ_BYTES),
            timeout=timeout,
        )

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        if self._scheme == "https" and not self._tls_active:
            await _close_stream_quietly(self._stream)
            raise HTTPHopError("peer_verification_failed")
        await self._stream.write(buffer, timeout=timeout)

    async def aclose(self) -> None:
        await self._stream.aclose()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if (
            self._scheme != "https"
            or self._tls_active
            or self._tls_started
            or ssl_context is not self._tls_context
            or server_hostname != self._route_host
        ):
            await _close_stream_quietly(self._stream)
            raise HTTPHopError("peer_verification_failed")
        self._tls_started = True

        tls_stream: httpcore.AsyncNetworkStream | None = None
        failure: HTTPHopError | None = None
        try:
            tls_stream = await self._stream.start_tls(
                ssl_context,
                server_hostname=server_hostname,
                timeout=timeout,
            )
        except asyncio.CancelledError:
            await _close_stream_quietly(self._stream)
            raise
        except httpcore.ConnectTimeout:
            failure = HTTPHopError("connect_timeout", retryable=True)
        except Exception:  # noqa: BLE001 - TLS failures cross a sanitized boundary
            failure = HTTPHopError("tls_error", retryable=True)
        if failure is not None:
            await _close_stream_quietly(self._stream)
            raise failure
        assert tls_stream is not None

        ssl_object: object | None = None
        ssl_metadata_failed = False
        try:
            ssl_object = tls_stream.get_extra_info("ssl_object")
            selected_alpn = (
                ssl_object.selected_alpn_protocol() if ssl_object is not None else None  # type: ignore[attr-defined]
            )
        except Exception:  # noqa: BLE001 - TLS metadata crosses a sanitized boundary
            ssl_metadata_failed = True
            selected_alpn = None
        if ssl_metadata_failed or ssl_object is None or selected_alpn not in {None, "http/1.1"}:
            await _close_stream_quietly(tls_stream)
            if tls_stream is not self._stream:
                await _close_stream_quietly(self._stream)
            raise HTTPHopError("tls_error")

        try:
            connected_ip = _peer_from_stream(
                tls_stream,
                selected_ip=self._selected_ip,
                expected_port=self._expected_port,
            )
        except HTTPHopError:
            await _close_stream_quietly(tls_stream)
            if tls_stream is not self._stream:
                await _close_stream_quietly(self._stream)
            raise
        self._evidence.connected_ip = connected_ip
        return _PeerVerifiedStream(
            tls_stream,
            scheme=self._scheme,
            route_host=self._route_host,
            selected_ip=self._selected_ip,
            expected_port=self._expected_port,
            tls_context=self._tls_context,
            evidence=self._evidence,
            tls_active=True,
            ssl_object=ssl_object,
        )

    def get_extra_info(self, info: str) -> object:
        if info == "ssl_object":
            return self._ssl_object if self._tls_active else None
        return self._stream.get_extra_info(info)


class _PinnedBackend(httpcore.AsyncNetworkBackend):
    """One-use backend that replaces only the physical dial hostname."""

    def __init__(
        self,
        delegate: httpcore.AsyncNetworkBackend,
        *,
        scheme: Literal["http", "https"],
        route_host: str,
        selected_ip: str,
        expected_port: int,
        tls_context: ssl.SSLContext | None,
        evidence: _PeerEvidence,
    ) -> None:
        self._delegate = delegate
        self._scheme = scheme
        self._route_host = route_host
        self._selected_ip = selected_ip
        self._expected_port = expected_port
        self._tls_context = tls_context
        self._evidence = evidence
        self._dial_started = False

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if self._dial_started:
            raise HTTPHopError("transport_error")
        self._dial_started = True
        if (
            host != self._route_host
            or port != self._expected_port
            or local_address is not None
            or socket_options is not None
        ):
            raise HTTPHopError("peer_verification_failed")

        stream = await self._delegate.connect_tcp(
            self._selected_ip,
            self._expected_port,
            timeout=timeout,
            local_address=None,
            socket_options=None,
        )
        try:
            connected_ip = _peer_from_stream(
                stream,
                selected_ip=self._selected_ip,
                expected_port=self._expected_port,
            )
        except HTTPHopError:
            await _close_stream_quietly(stream)
            raise
        self._evidence.connected_ip = connected_ip
        return _PeerVerifiedStream(
            stream,
            scheme=self._scheme,
            route_host=self._route_host,
            selected_ip=self._selected_ip,
            expected_port=self._expected_port,
            tls_context=self._tls_context,
            evidence=self._evidence,
        )

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del path, timeout, socket_options
        raise HTTPHopError("peer_verification_failed")

    async def sleep(self, seconds: float) -> None:
        del seconds
        raise HTTPHopError("transport_error")


def _build_tls_context() -> ssl.SSLContext:
    """Build a fresh client context from only the explicit certifi CA bundle."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.check_hostname = True
    context.verify_mode = ssl.CERT_REQUIRED
    context.load_verify_locations(cafile=certifi.where())
    context.keylog_filename = None
    return context


def _host_header_value(request: NormalizedHTTPHopRequest) -> bytes:
    host = request.host
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    authority = f"[{host}]" if isinstance(address, ipaddress.IPv6Address) else host
    default_port = 443 if request.scheme == "https" else 80
    if request.port != default_port:
        authority = f"{authority}:{request.port}"
    return authority.encode("ascii")


def _transport_headers(
    request: NormalizedHTTPHopRequest,
) -> list[tuple[bytes, bytes]]:
    headers = [
        (b"Host", _host_header_value(request)),
        (b"Connection", b"close"),
        (b"Accept-Encoding", b"identity"),
    ]
    headers.extend((name.encode("ascii"), value.encode("ascii")) for name, value in request.headers)
    return headers


def _response_headers(response: httpcore.Response) -> tuple[tuple[str, str], ...]:
    return tuple((name.decode("ascii").lower(), value.decode("latin-1")) for name, value in response.headers)


async def _perform_http_hop(
    request: NormalizedHTTPHopRequest,
    *,
    resolved_ips: tuple[str, ...],
    network_backend: httpcore.AsyncNetworkBackend,
) -> HTTPHopResponse:
    selected_ip = resolved_ips[0]
    evidence = _PeerEvidence()
    tls_context = _build_tls_context() if request.scheme == "https" else None
    backend = _PinnedBackend(
        network_backend,
        scheme=request.scheme,
        route_host=request.host,
        selected_ip=selected_ip,
        expected_port=request.port,
        tls_context=tls_context,
        evidence=evidence,
    )
    url = httpcore.URL(
        scheme=request.scheme,
        host=request.host,
        port=request.port,
        target=request.target,
    )
    content = request.body if request.method == "POST" else None
    timeouts = {
        "pool": request.limits.connect_timeout_seconds,
        "connect": request.limits.connect_timeout_seconds,
        "read": request.limits.read_timeout_seconds,
        "write": request.limits.write_timeout_seconds,
    }
    body = bytearray()
    body_ceiling = min(
        request.limits.max_decompressed_bytes,
        request.limits.max_parser_input_bytes,
    )

    async with httpcore.AsyncConnectionPool(
        ssl_context=tls_context,
        proxy=None,
        max_connections=1,
        max_keepalive_connections=0,
        keepalive_expiry=0,
        http1=True,
        http2=False,
        retries=0,
        network_backend=backend,
    ) as pool:
        async with pool.stream(
            request.method,
            url,
            headers=_transport_headers(request),
            content=content,
            extensions={"timeout": timeouts},
        ) as response:
            if response.status == 101:
                raise HTTPHopError("protocol_error")
            content_encodings = [
                value.strip().lower() for name, value in response.headers if name.lower() == b"content-encoding"
            ]
            if content_encodings not in ([], [b"identity"]):
                raise HTTPHopError("unsupported_content_encoding")
            async for chunk in response.aiter_stream():
                if len(body) + len(chunk) > body_ceiling:
                    raise HTTPHopError("response_too_large")
                body.extend(chunk)
            headers = _response_headers(response)

    if evidence.connected_ip is None:
        raise HTTPHopError("peer_verification_failed")
    return HTTPHopResponse(
        status_code=response.status,
        headers=headers,
        body=bytes(body),
        resolved_ips=resolved_ips,
        connected_ip=evidence.connected_ip,
        response_header_bytes=0,
        wire_bytes=len(body),
    )


async def _execute_http_hop(
    request: NormalizedHTTPHopRequest,
    *,
    resolved_ips: Sequence[str],
    network_backend: httpcore.AsyncNetworkBackend,
) -> HTTPHopResponse:
    """Private deterministic transport seam; public callers cannot inject I/O."""
    validated_ips = _validate_resolved_ips(resolved_ips)
    failure: HTTPHopError | None = None
    try:
        return await _perform_http_hop(
            request,
            resolved_ips=validated_ips,
            network_backend=network_backend,
        )
    except HTTPHopError:
        raise
    except httpcore.ConnectTimeout:
        failure = HTTPHopError("connect_timeout", retryable=True)
    except httpcore.ReadTimeout:
        failure = HTTPHopError("read_timeout", retryable=True)
    except httpcore.WriteTimeout:
        failure = HTTPHopError("write_timeout", retryable=True)
    except (httpcore.LocalProtocolError, httpcore.RemoteProtocolError, httpcore.UnsupportedProtocol):
        failure = HTTPHopError("protocol_error")
    except (httpcore.NetworkError, httpcore.PoolTimeout, httpcore.ConnectionNotAvailable):
        failure = HTTPHopError("transport_error", retryable=True)
    except Exception:  # noqa: BLE001 - transport failures cross a sanitized boundary
        failure = HTTPHopError("transport_error")
    raise failure


__all__ = [
    "HTTPHopError",
    "HTTPHopErrorCode",
    "HTTPHopLimits",
    "HTTPHopResponse",
    "NormalizedHTTPHopRequest",
]
