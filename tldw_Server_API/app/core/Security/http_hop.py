"""Strict contracts and DNS policy for a single outbound HTTP request hop."""

from __future__ import annotations

import asyncio
import ipaddress
import math
import re
import ssl
import time
import zlib
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Literal, Protocol

import certifi
import httpcore

from tldw_Server_API.app.core.exceptions import HTTPHopError, HTTPHopErrorCode
from tldw_Server_API.app.core.Security.egress import is_private_ip, resolve_host_ips

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
_HEADER_BLOCK_END_PATTERN = re.compile(rb"\n\r?\n")
_STATUS_LINE_PATTERN = re.compile(rb"HTTP/(1\.[01]) ([0-9]{3})(?: [^\r\n]*)?\Z")
_DENIED_IPV6_TRANSITION_NETWORKS = (
    ipaddress.ip_network("::/96"),
    ipaddress.ip_network("::ffff:0:0/96"),
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2001::/32"),
    ipaddress.ip_network("2002::/16"),
)
_MAX_NETWORK_READ_BYTES = 16 * 1024


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
    target: str = field(repr=False)
    headers: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    body: bytes = field(default=b"", repr=False)
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


@dataclass(frozen=True, slots=True)
class StatusOnlyHTTPHopResponse:
    """Status evidence that cannot expose receiver content or peer metadata."""

    status_code: int
    latency_ms: int
    retry_after_seconds: int | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.status_code, bool)
            or not isinstance(self.status_code, int)
            or not 100 <= self.status_code <= 599
        ):
            raise ValueError("status_code must be a valid HTTP status")
        if (
            isinstance(self.latency_ms, bool)
            or not isinstance(self.latency_ms, int)
            or self.latency_ms < 0
        ):
            raise ValueError("latency_ms must be a non-negative integer")
        if self.retry_after_seconds is not None:
            if (
                isinstance(self.retry_after_seconds, bool)
                or not isinstance(self.retry_after_seconds, int)
                or not 1 <= self.retry_after_seconds <= 1_800
                or self.status_code not in {429, 503}
            ):
                raise ValueError("retry_after_seconds is invalid")


_Resolver = Callable[[str, int, float], Awaitable[Sequence[str]]]


class _Clock(Protocol):
    """Clock shape shared by latency and HTTP-date calculations."""

    def monotonic(self) -> float: ...

    def utc_now(self) -> datetime: ...


class _SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def utc_now(self) -> datetime:
        return datetime.now(timezone.utc)


_SYSTEM_CLOCK = _SystemClock()


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
    _validate_headers(request.headers)
    if request.scheme != "https" and any(name == "authorization" for name, _value in request.headers):
        raise _invalid_request()
    if not isinstance(request.body, bytes):
        raise _invalid_request()
    if len(request.body) > request.limits.max_request_body_bytes:
        raise _invalid_request()
    if request.method in {"GET", "HEAD"} and request.body:
        raise _invalid_request()
    _validate_transport_header_limits(_transport_headers(request), request.limits)


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
) -> None:
    if not isinstance(headers, tuple):
        raise _invalid_request()
    seen: set[str] = set()
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


def _validate_resolved_ips(
    resolved_ips: Sequence[str],
    *,
    allow_e2e_loopback: bool = False,
) -> tuple[str, ...]:
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
        e2e_loopback = allow_e2e_loopback and canonical == "127.0.0.1"
        if raw_ip != canonical or not (
            _is_allowed_public_address(address) or e2e_loopback
        ):
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
    allow_e2e_loopback: bool = False,
) -> tuple[str, ...]:
    """Resolve once and reject the complete answer set if any address is unsafe."""
    try:
        literal = ipaddress.ip_address(request.host)
    except ValueError:
        literal = None
    if literal is not None:
        return _validate_resolved_ips(
            (request.host,),
            allow_e2e_loopback=allow_e2e_loopback,
        )

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
    return _validate_resolved_ips(
        resolved,
        allow_e2e_loopback=allow_e2e_loopback,
    )


@dataclass(slots=True)
class _PeerEvidence:
    connected_ip: str | None = None
    response_header_bytes: int = 0
    response_header_count: int = 0
    wire_bytes: int = 0
    final_http_version: bytes | None = None
    final_status: int | None = None
    final_content_lengths: tuple[bytes, ...] = ()
    final_retry_after_values: tuple[bytes, ...] = ()


class _ContentDecoder(Protocol):
    """Small structural subset shared by zlib and deterministic test decoders."""

    eof: bool
    unused_data: bytes
    unconsumed_tail: bytes

    def decompress(self, data: bytes, max_length: int = 0) -> bytes: ...


class _RawResponseGuard:
    """Bound plaintext response headers and wire bytes before h11 sees them."""

    def __init__(
        self,
        *,
        evidence: _PeerEvidence,
        limits: HTTPHopLimits,
        response_mode: Literal["bounded_body", "status_only"],
    ) -> None:
        self._evidence = evidence
        self._limits = limits
        self._response_mode = response_mode
        self._pending = bytearray()
        self._scan_position = 0
        self._output_position = 0
        self._final_headers_seen = False

    async def read(
        self,
        stream: httpcore.AsyncNetworkStream,
        max_bytes: int,
        timeout: float | None,
    ) -> bytes:
        """Return only bytes already admitted by the header/wire ceilings."""
        if self._output_position < len(self._pending):
            return self._read_pending(max_bytes)

        if self._final_headers_seen:
            if self._response_mode == "status_only":
                return b""
            return await self._read_wire(stream, max_bytes, timeout)

        while not self._final_headers_seen:
            partial_header_bytes = len(self._pending) - self._scan_position
            remaining = (
                self._limits.max_response_header_bytes - self._evidence.response_header_bytes - partial_header_bytes
            )
            read_size = min(max_bytes, _MAX_NETWORK_READ_BYTES, max(1, remaining + 1))
            data = await stream.read(read_size, timeout=timeout)
            if not data:
                raise HTTPHopError("protocol_error")
            self._pending.extend(data)
            self._scan_header_blocks()

            if not self._final_headers_seen:
                observed = self._evidence.response_header_bytes + len(self._pending) - self._scan_position
                if observed > self._limits.max_response_header_bytes:
                    raise HTTPHopError("response_headers_too_large")

        return self._read_pending(max_bytes)

    def _read_pending(self, max_bytes: int) -> bytes:
        end = min(len(self._pending), self._output_position + max_bytes)
        data = bytes(self._pending[self._output_position : end])
        self._output_position = end
        if self._output_position == len(self._pending):
            self._pending.clear()
            self._scan_position = 0
            self._output_position = 0
        return data

    async def _read_wire(
        self,
        stream: httpcore.AsyncNetworkStream,
        max_bytes: int,
        timeout: float | None,
    ) -> bytes:
        remaining = self._limits.max_wire_bytes - self._evidence.wire_bytes
        read_size = min(max_bytes, _MAX_NETWORK_READ_BYTES, max(1, remaining + 1))
        data = await stream.read(read_size, timeout=timeout)
        if len(data) > remaining:
            raise HTTPHopError("response_too_large")
        self._evidence.wire_bytes += len(data)
        return data

    def _scan_header_blocks(self) -> None:
        while True:
            match = _HEADER_BLOCK_END_PATTERN.search(self._pending, self._scan_position)
            if match is None:
                return
            block_end = match.end()
            block = bytes(self._pending[self._scan_position : block_end])
            self._record_header_block(block)
            self._scan_position = block_end
            if self._final_headers_seen:
                body_bytes = len(self._pending) - block_end
                if self._response_mode == "status_only":
                    del self._pending[block_end:]
                    self._evidence.wire_bytes = 0
                    return
                if body_bytes > self._limits.max_wire_bytes:
                    raise HTTPHopError("response_too_large")
                self._evidence.wire_bytes = body_bytes
                return

    def _record_header_block(self, block: bytes) -> None:
        self._evidence.response_header_bytes += len(block)
        if self._evidence.response_header_bytes > self._limits.max_response_header_bytes:
            raise HTTPHopError("response_headers_too_large")

        lines = block.split(b"\n")
        if len(lines) < 3:
            raise HTTPHopError("protocol_error")
        status_line = lines[0].removesuffix(b"\r")
        status_match = _STATUS_LINE_PATTERN.fullmatch(status_line)
        if status_match is None:
            raise HTTPHopError("protocol_error")
        http_version = status_match.group(1)
        status = int(status_match.group(2))
        if status < 100 or status > 599 or status == 101:
            raise HTTPHopError("protocol_error")

        field_lines = [line.removesuffix(b"\r") for line in lines[1:-2]]
        self._evidence.response_header_count += len(field_lines)
        if self._evidence.response_header_count > self._limits.max_response_headers:
            raise HTTPHopError("response_headers_too_large")

        if 100 <= status < 200:
            return

        content_lengths: list[bytes] = []
        retry_after_values: list[bytes] = []
        for line in field_lines:
            if not line or line[:1] in {b" ", b"\t"}:
                continue
            name, separator, value = line.partition(b":")
            if not separator:
                continue
            normalized_name = name.lower()
            if normalized_name == b"content-length":
                content_lengths.append(value.strip(b" \t"))
            elif normalized_name == b"retry-after":
                retry_after_values.append(value)
        if len(content_lengths) > 1:
            raise HTTPHopError("protocol_error")
        self._evidence.final_content_lengths = tuple(content_lengths)
        self._evidence.final_retry_after_values = tuple(retry_after_values)
        self._evidence.final_http_version = http_version
        self._evidence.final_status = status
        self._final_headers_seen = True


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
        response_guard: _RawResponseGuard,
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
        self._response_guard = response_guard
        self._tls_active = tls_active
        self._tls_started = tls_active
        self._ssl_object = ssl_object

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self._scheme == "https" and not self._tls_active:
            await _close_stream_quietly(self._stream)
            raise HTTPHopError("peer_verification_failed")
        return await self._response_guard.read(
            self._stream,
            max_bytes,
            timeout,
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
        if tls_stream is None:
            await _close_stream_quietly(self._stream)
            raise HTTPHopError("tls_error")

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
            response_guard=self._response_guard,
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
        response_guard: _RawResponseGuard,
    ) -> None:
        self._delegate = delegate
        self._scheme = scheme
        self._route_host = route_host
        self._selected_ip = selected_ip
        self._expected_port = expected_port
        self._tls_context = tls_context
        self._evidence = evidence
        self._response_guard = response_guard
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
            response_guard=self._response_guard,
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
        (b"Accept-Encoding", b"gzip, deflate"),
    ]
    if request.method == "POST":
        headers.append((b"Content-Length", str(len(request.body)).encode("ascii")))
    headers.extend((name.encode("ascii"), value.encode("ascii")) for name, value in request.headers)
    return headers


def _validate_transport_header_limits(
    headers: list[tuple[bytes, bytes]],
    limits: HTTPHopLimits,
) -> None:
    if len(headers) > limits.max_request_headers:
        raise _invalid_request()
    serialized_bytes = sum(len(name) + 2 + len(value) + 2 for name, value in headers)
    if serialized_bytes > limits.max_request_header_bytes:
        raise _invalid_request()


def _response_headers(response: httpcore.Response) -> tuple[tuple[str, str], ...]:
    return tuple((name.decode("ascii").lower(), value.decode("latin-1")) for name, value in response.headers)


def _clock_failure() -> HTTPHopError:
    return HTTPHopError("transport_error", retryable=True)


def _monotonic_now(clock: _Clock) -> float:
    try:
        value = clock.monotonic()
    except Exception:  # noqa: BLE001 - injected clock detail is not boundary-safe
        raise _clock_failure() from None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _clock_failure()
    return float(value)


def _utc_now(clock: _Clock) -> datetime:
    try:
        value = clock.utc_now()
        offset = value.utcoffset() if isinstance(value, datetime) else None
        normalized = value.astimezone(timezone.utc) if offset is not None else None
    except Exception:  # noqa: BLE001 - injected clock detail is not boundary-safe
        raise _clock_failure() from None
    if normalized is None:
        raise _clock_failure()
    return normalized


def _latency_ms(clock: _Clock, started: float) -> int:
    finished = _monotonic_now(clock)
    elapsed_ms = (finished - started) * 1_000
    if finished < started or not math.isfinite(elapsed_ms):
        raise _clock_failure()
    return int(elapsed_ms)


def _retry_after_seconds(
    evidence: _PeerEvidence,
    *,
    clock: _Clock,
) -> int | None:
    if evidence.final_status not in {429, 503}:
        return None
    if len(evidence.final_retry_after_values) != 1:
        return None
    raw_value = evidence.final_retry_after_values[0]
    if any(byte < 32 or byte > 126 for byte in raw_value):
        return None
    value = raw_value.strip(b" ").decode("ascii")
    if not value:
        return None
    if value.isdigit():
        significant = value.lstrip("0") or "0"
        seconds = 1_800 if len(significant) > 4 else int(significant)
        return min(1_800, max(1, seconds))
    try:
        parsed = parsedate_to_datetime(value)
        offset = parsed.utcoffset()
    except (OverflowError, TypeError, ValueError):
        return None
    if offset is None:
        return None
    delay = (parsed.astimezone(timezone.utc) - _utc_now(clock)).total_seconds()
    if not math.isfinite(delay):
        raise _clock_failure()
    return min(1_800, max(1, math.ceil(delay)))


def _body_is_permitted(request: NormalizedHTTPHopRequest, status_code: int) -> bool:
    return request.method != "HEAD" and status_code not in {204, 205, 304}


def _transfer_is_chunked(response: httpcore.Response) -> bool:
    values = [value.strip().lower() for name, value in response.headers if name.lower() == b"transfer-encoding"]
    if not values:
        return False
    if values != [b"chunked"]:
        raise HTTPHopError("protocol_error")
    return True


def _declared_content_length(evidence: _PeerEvidence) -> int | None:
    if not evidence.final_content_lengths:
        return None
    if len(evidence.final_content_lengths) != 1:
        raise HTTPHopError("protocol_error")
    value = evidence.final_content_lengths[0]
    if not value or not value.isdigit():
        raise HTTPHopError("protocol_error")
    return int(value)


def _content_encoding(response: httpcore.Response) -> Literal["identity", "gzip", "deflate"]:
    values = [value.strip().lower() for name, value in response.headers if name.lower() == b"content-encoding"]
    if not values:
        return "identity"
    if len(values) != 1 or b"," in values[0]:
        raise HTTPHopError("unsupported_content_encoding")
    if values[0] == b"identity":
        return "identity"
    if values[0] == b"gzip":
        return "gzip"
    if values[0] == b"deflate":
        return "deflate"
    raise HTTPHopError("unsupported_content_encoding")


def _extend_decoded_body(body: bytearray, data: bytes, limits: HTTPHopLimits) -> None:
    new_size = len(body) + len(data)
    if new_size > limits.max_decompressed_bytes:
        raise HTTPHopError("decompressed_response_too_large")
    if new_size > limits.max_parser_input_bytes:
        raise HTTPHopError("parser_input_too_large")
    body.extend(data)


def _decoder_output_limit(body: bytearray, limits: HTTPHopLimits) -> int:
    remaining = min(
        limits.max_decompressed_bytes - len(body),
        limits.max_parser_input_bytes - len(body),
    )
    return remaining + 1


def _decompress_bounded(
    decoder: _ContentDecoder,
    data: bytes,
    max_length: int,
) -> bytes:
    try:
        return decoder.decompress(data, max_length)
    except zlib.error:
        pass
    raise HTTPHopError("invalid_content_encoding")


def _decode_compressed_input(
    decoder: _ContentDecoder,
    encoded: bytes,
    body: bytearray,
    limits: HTTPHopLimits,
) -> None:
    pending = encoded
    while pending:
        if decoder.eof:
            raise HTTPHopError("invalid_content_encoding")
        output = _decompress_bounded(
            decoder,
            pending,
            _decoder_output_limit(body, limits),
        )
        _extend_decoded_body(body, output, limits)
        if decoder.unused_data:
            raise HTTPHopError("invalid_content_encoding")
        tail = decoder.unconsumed_tail
        if tail:
            if tail == pending and not output:
                raise HTTPHopError("invalid_content_encoding")
            pending = tail
            continue
        return


def _finish_content_decoder(
    decoder: _ContentDecoder,
    body: bytearray,
    limits: HTTPHopLimits,
) -> None:
    while not decoder.eof:
        output = _decompress_bounded(
            decoder,
            b"",
            _decoder_output_limit(body, limits),
        )
        _extend_decoded_body(body, output, limits)
        if decoder.unused_data or decoder.unconsumed_tail:
            raise HTTPHopError("invalid_content_encoding")
        if decoder.eof:
            return
        if not output:
            raise HTTPHopError("invalid_content_encoding")


async def _read_decoded_body(
    response: httpcore.Response,
    *,
    encoding: Literal["identity", "gzip", "deflate"],
    limits: HTTPHopLimits,
) -> bytes:
    body = bytearray()
    if encoding == "identity":
        async for chunk in response.aiter_stream():
            _extend_decoded_body(body, chunk, limits)
        return bytes(body)

    window_bits = zlib.MAX_WBITS | 16 if encoding == "gzip" else zlib.MAX_WBITS
    decoder: _ContentDecoder = zlib.decompressobj(window_bits)
    async for chunk in response.aiter_stream():
        _decode_compressed_input(decoder, chunk, body, limits)
    _finish_content_decoder(decoder, body, limits)
    return bytes(body)


@dataclass(frozen=True, slots=True)
class _StatusOnlyEvidence:
    status_code: int
    retry_after_seconds: int | None


async def _perform_http_hop(
    request: NormalizedHTTPHopRequest,
    *,
    resolved_ips: tuple[str, ...],
    network_backend: httpcore.AsyncNetworkBackend,
    response_mode: Literal["bounded_body", "status_only"],
    clock: _Clock | None,
) -> HTTPHopResponse | _StatusOnlyEvidence:
    if response_mode not in {"bounded_body", "status_only"}:
        raise HTTPHopError("transport_error")
    selected_ip = resolved_ips[0]
    transport_headers = _transport_headers(request)
    evidence = _PeerEvidence()
    response_guard = _RawResponseGuard(
        evidence=evidence,
        limits=request.limits,
        response_mode=response_mode,
    )
    tls_context = _build_tls_context() if request.scheme == "https" else None
    backend = _PinnedBackend(
        network_backend,
        scheme=request.scheme,
        route_host=request.host,
        selected_ip=selected_ip,
        expected_port=request.port,
        tls_context=tls_context,
        evidence=evidence,
        response_guard=response_guard,
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
    status_code: int | None = None
    status_evidence: _StatusOnlyEvidence | None = None
    headers: tuple[tuple[str, str], ...] | None = None
    body: bytes | None = None
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
            headers=transport_headers,
            content=content,
            extensions={"timeout": timeouts},
        ) as response:
            if evidence.final_status is None or evidence.final_status != response.status:
                raise HTTPHopError("protocol_error")
            status_code = response.status
            if response_mode == "status_only":
                if clock is None:
                    raise HTTPHopError("transport_error")
                status_evidence = _StatusOnlyEvidence(
                    status_code=response.status,
                    retry_after_seconds=_retry_after_seconds(evidence, clock=clock),
                )
            else:
                body_permitted = _body_is_permitted(request, response.status)
                chunked = _transfer_is_chunked(response)
                declared_length = _declared_content_length(evidence)
                if chunked and evidence.final_http_version != b"1.1":
                    raise HTTPHopError("protocol_error")
                if chunked and declared_length is not None:
                    raise HTTPHopError("protocol_error")
                if response.status == 204 and (chunked or declared_length is not None):
                    raise HTTPHopError("protocol_error")
                if body_permitted and not chunked and declared_length is not None:
                    if declared_length > request.limits.max_wire_bytes:
                        raise HTTPHopError("response_too_large")
                encoding = _content_encoding(response) if body_permitted else "identity"
                body = await _read_decoded_body(
                    response,
                    encoding=encoding,
                    limits=request.limits,
                )
                if not body_permitted:
                    empty_chunked_205 = (
                        request.method != "HEAD"
                        and response.status == 205
                        and chunked
                        and not body
                        and evidence.wire_bytes == 5
                    )
                    if body or (evidence.wire_bytes and not empty_chunked_205):
                        raise HTTPHopError("protocol_error")
                if (
                    body_permitted
                    and not chunked
                    and declared_length is not None
                    and evidence.wire_bytes != declared_length
                ):
                    raise HTTPHopError("protocol_error")
                headers = _response_headers(response)

    if evidence.connected_ip is None:
        raise HTTPHopError("peer_verification_failed")
    if status_evidence is not None:
        return status_evidence
    if status_code is None or headers is None or body is None:
        raise HTTPHopError("transport_error")
    return HTTPHopResponse(
        status_code=status_code,
        headers=headers,
        body=body,
        resolved_ips=resolved_ips,
        connected_ip=evidence.connected_ip,
        response_header_bytes=evidence.response_header_bytes,
        wire_bytes=evidence.wire_bytes,
    )


async def _execute_http_hop_mode(
    request: NormalizedHTTPHopRequest,
    *,
    resolved_ips: Sequence[str],
    network_backend: httpcore.AsyncNetworkBackend,
    response_mode: Literal["bounded_body", "status_only"],
    clock: _Clock | None,
    allow_e2e_loopback: bool = False,
) -> HTTPHopResponse | _StatusOnlyEvidence:
    validated_ips = _validate_resolved_ips(
        resolved_ips,
        allow_e2e_loopback=allow_e2e_loopback,
    )
    failure: HTTPHopError | None = None
    try:
        return await _perform_http_hop(
            request,
            resolved_ips=validated_ips,
            network_backend=network_backend,
            response_mode=response_mode,
            clock=clock,
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


async def _execute_http_hop(
    request: NormalizedHTTPHopRequest,
    *,
    resolved_ips: Sequence[str],
    network_backend: httpcore.AsyncNetworkBackend,
) -> HTTPHopResponse:
    """Private deterministic bounded-body seam; public callers cannot inject I/O."""
    response = await _execute_http_hop_mode(
        request,
        resolved_ips=resolved_ips,
        network_backend=network_backend,
        response_mode="bounded_body",
        clock=None,
    )
    if not isinstance(response, HTTPHopResponse):
        raise HTTPHopError("transport_error")
    return response


async def _execute_http_hop_status(
    request: NormalizedHTTPHopRequest,
    *,
    resolved_ips: Sequence[str],
    network_backend: httpcore.AsyncNetworkBackend,
    clock: _Clock,
    allow_e2e_loopback: bool = False,
) -> _StatusOnlyEvidence:
    response = await _execute_http_hop_mode(
        request,
        resolved_ips=resolved_ips,
        network_backend=network_backend,
        response_mode="status_only",
        clock=clock,
        allow_e2e_loopback=allow_e2e_loopback,
    )
    if not isinstance(response, _StatusOnlyEvidence):
        raise HTTPHopError("transport_error")
    return response


async def _request_http_hop(
    request: NormalizedHTTPHopRequest,
    *,
    resolver: _Resolver,
    network_backend: httpcore.AsyncNetworkBackend,
) -> HTTPHopResponse:
    """Private deterministic seam covered by the whole-hop deadline."""
    if not isinstance(request, NormalizedHTTPHopRequest):
        raise HTTPHopError("invalid_request")

    async def execute() -> HTTPHopResponse:
        resolved_ips = await _resolve_validated_ips(request, resolver=resolver)
        return await _execute_http_hop(
            request,
            resolved_ips=resolved_ips,
            network_backend=network_backend,
        )

    try:
        return await asyncio.wait_for(
            execute(),
            timeout=request.limits.total_timeout_seconds,
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        raise HTTPHopError("total_timeout", retryable=True) from None


async def _request_http_hop_status(
    request: NormalizedHTTPHopRequest,
    *,
    resolver: _Resolver,
    network_backend: httpcore.AsyncNetworkBackend,
    clock: _Clock,
    allow_e2e_loopback: bool = False,
) -> StatusOnlyHTTPHopResponse:
    """Private deterministic status-only seam covered by the whole-hop deadline."""
    if not isinstance(request, NormalizedHTTPHopRequest):
        raise HTTPHopError("invalid_request")
    started = _monotonic_now(clock)

    async def execute() -> _StatusOnlyEvidence:
        resolved_ips = await _resolve_validated_ips(
            request,
            resolver=resolver,
            allow_e2e_loopback=allow_e2e_loopback,
        )
        return await _execute_http_hop_status(
            request,
            resolved_ips=resolved_ips,
            network_backend=network_backend,
            clock=clock,
            allow_e2e_loopback=allow_e2e_loopback,
        )

    try:
        evidence = await asyncio.wait_for(
            execute(),
            timeout=min(request.limits.total_timeout_seconds, 30.0),
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        raise HTTPHopError("total_timeout", retryable=True) from None
    return StatusOnlyHTTPHopResponse(
        status_code=evidence.status_code,
        latency_ms=_latency_ms(clock, started),
        retry_after_seconds=evidence.retry_after_seconds,
    )


async def request_http_hop(
    request: NormalizedHTTPHopRequest,
) -> HTTPHopResponse:
    """Perform one bounded request using only production resolver/transport inputs."""
    return await _request_http_hop(
        request,
        resolver=_default_resolver,
        network_backend=httpcore.AnyIOBackend(),
    )


async def request_http_hop_status(
    request: NormalizedHTTPHopRequest,
) -> StatusOnlyHTTPHopResponse:
    """Perform one production-only hop and retain no receiver body or metadata."""
    return await _request_http_hop_status(
        request,
        resolver=_default_resolver,
        network_backend=httpcore.AnyIOBackend(),
        clock=_SYSTEM_CLOCK,
    )


async def request_admin_webhook_e2e_loopback_status(
    request: NormalizedHTTPHopRequest,
) -> StatusOnlyHTTPHopResponse:
    """Perform one status-only hop to the exact isolated admin E2E receiver."""
    if (
        not isinstance(request, NormalizedHTTPHopRequest)
        or request.scheme != "http"
        or request.host != "127.0.0.1"
    ):
        raise HTTPHopError("dns_address_denied")
    return await _request_http_hop_status(
        request,
        resolver=_default_resolver,
        network_backend=httpcore.AnyIOBackend(),
        clock=_SYSTEM_CLOCK,
        allow_e2e_loopback=True,
    )


__all__ = [
    "HTTPHopError",
    "HTTPHopErrorCode",
    "HTTPHopLimits",
    "HTTPHopResponse",
    "NormalizedHTTPHopRequest",
    "StatusOnlyHTTPHopResponse",
    "request_http_hop",
    "request_http_hop_status",
]
