from __future__ import annotations

import asyncio
import ipaddress
import logging
import ssl
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpcore
import pytest

from tldw_Server_API.app.core.Security import http_hop

pytestmark = pytest.mark.unit

_OK_RESPONSE = (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",)


class DeterministicClock:
    """One clock source for deterministic latency and HTTP-date parsing."""

    def __init__(
        self,
        *,
        monotonic_values: Sequence[float] = (10.0, 10.125),
        utc_now: datetime = datetime(2026, 8, 23, tzinfo=timezone.utc),
    ) -> None:
        self.monotonic_values = list(monotonic_values)
        self.now = utc_now

    def monotonic(self) -> float:
        return self.monotonic_values.pop(0)

    def utc_now(self) -> datetime:
        return self.now


class FakeSSLObject:
    """Minimal TLS evidence used by HTTPcore's negotiated-protocol check."""

    def selected_alpn_protocol(self) -> str | None:
        return None


class RecordingStream(httpcore.AsyncNetworkStream):
    """Deterministic HTTPcore stream with observable peer and TLS behavior."""

    def __init__(
        self,
        *,
        server_addr: object,
        response: Sequence[bytes] = (),
        tls_stream: RecordingStream | None = None,
        tls_error: Exception | None = None,
        provide_tls_object: bool = True,
    ) -> None:
        self.server_addr = server_addr
        self.response = list(response)
        self.tls_stream = tls_stream
        self.tls_error = tls_error
        self.provide_tls_object = provide_tls_object
        self.ssl_object: object | None = None
        self.writes: list[bytes] = []
        self.tls_calls: list[tuple[ssl.SSLContext, str | None, float | None]] = []
        self.closed = False

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        del timeout
        if not self.response:
            return b""
        chunk = self.response.pop(0)
        if len(chunk) > max_bytes:
            self.response.insert(0, chunk[max_bytes:])
            return chunk[:max_bytes]
        return chunk

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    async def aclose(self) -> None:
        self.closed = True

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.tls_calls.append((ssl_context, server_hostname, timeout))
        if self.tls_error is not None:
            raise self.tls_error
        tls_stream = self if self.tls_stream is None else self.tls_stream
        if self.provide_tls_object:
            tls_stream.ssl_object = FakeSSLObject()
        return tls_stream

    def get_extra_info(self, info: str) -> Any:
        if info == "server_addr":
            return self.server_addr
        if info == "ssl_object":
            return self.ssl_object
        if info == "is_readable":
            return False
        return None


class NullTLSResultStream(RecordingStream):
    """Simulate a backend that violates the start_tls return contract."""

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.tls_calls.append((ssl_context, server_hostname, timeout))
        return None  # type: ignore[return-value]


class RecordingBackend(httpcore.AsyncNetworkBackend):
    """Delegate backend that records the address selected by the hop."""

    def __init__(
        self,
        stream: RecordingStream,
        *,
        connect_error: Exception | None = None,
    ) -> None:
        self.stream = stream
        self.connect_error = connect_error
        self.connect_calls: list[tuple[str, int, float | None, str | None, Iterable[object] | None]] = []
        self.unix_connect_calls = 0
        self.sleep_calls: list[float] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[object] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.connect_calls.append((host, port, timeout, local_address, socket_options))
        if len(self.connect_calls) > 1:
            raise AssertionError("A hop must never perform a second physical dial")
        if self.connect_error is not None:
            raise self.connect_error
        return self.stream

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[object] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del path, timeout, socket_options
        self.unix_connect_calls += 1
        raise AssertionError("HTTP hops must never use Unix sockets")

    async def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        raise AssertionError("HTTP hops must never retry")


class BlockingReadStream(RecordingStream):
    """Stream that exposes a deterministic cancellation point during read."""

    def __init__(
        self,
        *,
        server_addr: object,
        response: Sequence[bytes] = (),
    ) -> None:
        super().__init__(server_addr=server_addr, response=response)
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        self.read_started.set()
        await self.read_release.wait()
        return await super().read(max_bytes, timeout=timeout)


class BlockingTLSStream(RecordingStream):
    """Stream that exposes a deterministic cancellation point during TLS."""

    def __init__(self, *, server_addr: object) -> None:
        super().__init__(server_addr=server_addr)
        self.tls_started = asyncio.Event()
        self.tls_release = asyncio.Event()

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        self.tls_calls.append((ssl_context, server_hostname, timeout))
        self.tls_started.set()
        await self.tls_release.wait()
        return self


class HostileRawSSLMetadataStream(RecordingStream):
    """Raw HTTP stream whose SSL metadata must never be consulted."""

    def get_extra_info(self, info: str) -> Any:
        if info == "ssl_object":
            raise RuntimeError("raw stream injected ssl metadata")
        return super().get_extra_info(info)


def _request(**overrides: object) -> http_hop.NormalizedHTTPHopRequest:
    values: dict[str, object] = {
        "scheme": "http",
        "host": "api.example.com",
        "port": 80,
        "method": "GET",
        "target": "/works?q=bounded",
        "headers": (("accept", "application/json"),),
    }
    values.update(overrides)
    return http_hop.NormalizedHTTPHopRequest(**values)  # type: ignore[arg-type]


def _header_values(request_bytes: bytes, name: bytes) -> list[bytes]:
    prefix = name.lower() + b":"
    return [line.split(b":", 1)[1].strip() for line in request_bytes.split(b"\r\n") if line.lower().startswith(prefix)]


async def _execute(
    request: http_hop.NormalizedHTTPHopRequest,
    resolved_ips: tuple[str, ...],
    backend: RecordingBackend,
) -> http_hop.HTTPHopResponse:
    return await http_hop._execute_http_hop(
        request,
        resolved_ips=resolved_ips,
        network_backend=backend,
    )


async def _execute_status(
    request: http_hop.NormalizedHTTPHopRequest,
    resolved_ips: tuple[str, ...],
    backend: RecordingBackend,
    *,
    clock: DeterministicClock | None = None,
) -> http_hop.StatusOnlyHTTPHopResponse:
    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        return resolved_ips

    return await http_hop._request_http_hop_status(
        request,
        resolver=resolver,
        network_backend=backend,
        clock=clock or DeterministicClock(),
    )


async def test_dials_selected_validated_ip_and_keeps_original_origin() -> None:
    stream = RecordingStream(
        server_addr=("8.8.8.8", 8080),
        response=_OK_RESPONSE,
    )
    backend = RecordingBackend(stream)
    request = _request(port=8080)

    response = await _execute(request, ("8.8.8.8", "1.1.1.1"), backend)

    assert [(call[0], call[1], call[2]) for call in backend.connect_calls] == [("8.8.8.8", 8080, 5.0)]
    assert backend.unix_connect_calls == 0
    assert backend.sleep_calls == []
    request_bytes = b"".join(stream.writes)
    assert request_bytes.startswith(b"GET /works?q=bounded HTTP/1.1\r\n")
    assert _header_values(request_bytes, b"host") == [b"api.example.com:8080"]
    assert response.body == b"ok"
    assert response.resolved_ips == ("8.8.8.8", "1.1.1.1")
    assert response.connected_ip == "8.8.8.8"
    assert stream.closed is True


async def test_status_only_pins_dns_preserves_host_ignores_proxies_and_does_not_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://ambient-proxy-secret@127.0.0.1:1")
    response_bytes = (
        b"HTTP/1.1 302 Found\r\n"
        b"Location: https://redirect.example/secret\r\n"
        b"Content-Length: 999999\r\nConnection: close\r\n\r\n"
    )
    stream = RecordingStream(
        server_addr=("8.8.8.8", 8080),
        response=(response_bytes,),
    )
    backend = RecordingBackend(stream)
    request = _request(port=8080, target="/status-only?opaque=1")

    response = await _execute_status(
        request,
        ("8.8.8.8", "1.1.1.1"),
        backend,
    )

    assert response == http_hop.StatusOnlyHTTPHopResponse(
        status_code=302,
        latency_ms=125,
        retry_after_seconds=None,
    )
    assert len(backend.connect_calls) == 1
    assert backend.connect_calls[0][:2] == ("8.8.8.8", 8080)
    request_bytes = b"".join(stream.writes)
    assert request_bytes.startswith(b"GET /status-only?opaque=1 HTTP/1.1\r\n")
    assert _header_values(request_bytes, b"host") == [b"api.example.com:8080"]
    assert b"ambient-proxy-secret" not in request_bytes
    assert stream.closed is True


async def test_status_only_https_keeps_hostname_tls_and_host_semantics() -> None:
    response_bytes = (
        b"HTTP/1.1 503 Service Unavailable\r\n"
        b"Retry-After: 300\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
    )
    tls_stream = RecordingStream(
        server_addr=("8.8.8.8", 8443),
        response=(response_bytes,),
    )
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 8443),
        tls_stream=tls_stream,
    )
    backend = RecordingBackend(tcp_stream)

    response = await _execute_status(
        _request(scheme="https", port=8443),
        ("8.8.8.8",),
        backend,
    )

    assert response.status_code == 503
    assert response.retry_after_seconds == 300
    assert len(tcp_stream.tls_calls) == 1
    context, server_hostname, _timeout = tcp_stream.tls_calls[0]
    assert server_hostname == "api.example.com"
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert _header_values(b"".join(tls_stream.writes), b"host") == [b"api.example.com:8443"]
    assert tls_stream.closed is True


async def test_status_only_rejects_unverified_connected_peer_and_closes() -> None:
    stream = RecordingStream(
        server_addr=("1.1.1.1", 80),
        response=_OK_RESPONSE,
    )
    backend = RecordingBackend(stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute_status(_request(), ("8.8.8.8",), backend)

    assert exc.value.code == "peer_verification_failed"
    assert stream.writes == []
    assert stream.closed is True


async def test_status_only_cancellation_propagates_and_closes_stream() -> None:
    stream = BlockingReadStream(server_addr=("8.8.8.8", 80))
    backend = RecordingBackend(stream)
    task = asyncio.create_task(_execute_status(_request(), ("8.8.8.8",), backend))

    await asyncio.wait_for(stream.read_started.wait(), timeout=1.0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True


async def test_status_only_backward_monotonic_clock_fails_closed_after_cleanup() -> None:
    stream = RecordingStream(server_addr=("8.8.8.8", 80), response=_OK_RESPONSE)
    backend = RecordingBackend(stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute_status(
            _request(),
            ("8.8.8.8",),
            backend,
            clock=DeterministicClock(monotonic_values=(10.0, 9.0)),
        )

    assert exc.value.code == "transport_error"
    assert stream.closed is True


async def test_public_status_only_wrapper_uses_production_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel_backend = object()
    captured: dict[str, object] = {}

    async def private_seam(
        request: http_hop.NormalizedHTTPHopRequest,
        *,
        resolver: object,
        network_backend: object,
        clock: object,
    ) -> http_hop.StatusOnlyHTTPHopResponse:
        captured.update(
            request=request,
            resolver=resolver,
            network_backend=network_backend,
            clock=clock,
        )
        return http_hop.StatusOnlyHTTPHopResponse(200, 1, None)

    monkeypatch.setattr(http_hop.httpcore, "AnyIOBackend", lambda: sentinel_backend)
    monkeypatch.setattr(http_hop, "_request_http_hop_status", private_seam)
    request = _request()

    response = await http_hop.request_http_hop_status(request)

    assert response == http_hop.StatusOnlyHTTPHopResponse(200, 1, None)
    assert captured["request"] is request
    assert captured["resolver"] is http_hop._default_resolver
    assert captured["network_backend"] is sentinel_backend
    assert callable(captured["clock"].monotonic)
    assert callable(captured["clock"].utc_now)


async def test_plain_http_ignores_delegate_ssl_metadata() -> None:
    stream = HostileRawSSLMetadataStream(
        server_addr=("8.8.8.8", 80),
        response=_OK_RESPONSE,
    )
    backend = RecordingBackend(stream)

    response = await _execute(_request(), ("8.8.8.8",), backend)

    assert response.status_code == 200
    assert b"".join(stream.writes).startswith(b"GET /works?q=bounded HTTP/1.1\r\n")
    assert stream.closed is True


async def test_builds_request_framing_only_from_explicit_contract() -> None:
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=_OK_RESPONSE,
    )
    backend = RecordingBackend(stream)
    request = _request(
        method="POST",
        body=b"payload",
        headers=(
            ("accept", "application/json"),
            ("x-request-id", "explicit-route-id"),
        ),
    )

    await _execute(request, ("8.8.8.8",), backend)

    request_bytes = b"".join(stream.writes)
    assert _header_values(request_bytes, b"host") == [b"api.example.com"]
    assert _header_values(request_bytes, b"connection") == [b"close"]
    assert _header_values(request_bytes, b"content-length") == [b"7"]
    assert _header_values(request_bytes, b"transfer-encoding") == []
    assert _header_values(request_bytes, b"accept-encoding") == [b"gzip, deflate"]
    assert _header_values(request_bytes, b"x-request-id") == [b"explicit-route-id"]
    assert request_bytes.endswith(b"\r\n\r\npayload")


async def test_rejects_unvalidated_resolved_set_before_connect() -> None:
    stream = RecordingStream(server_addr=("10.0.0.1", 80), response=_OK_RESPONSE)
    backend = RecordingBackend(stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(_request(), ("10.0.0.1",), backend)

    assert exc.value.code == "dns_address_denied"
    assert backend.connect_calls == []


@pytest.mark.parametrize(
    ("selected_ip", "server_addr"),
    [
        ("8.8.8.8", ("8.8.8.8", 8080)),
        ("2001:4860:4860::8888", ("2001:4860:4860::8888", 8080)),
        (
            "2001:4860:4860::8888",
            ("2001:4860:4860:0:0:0:0:8888", 8080),
        ),
        ("2001:4860:4860::8888", ("2001:4860:4860::8888", 8080, 0, 0)),
    ],
)
async def test_accepts_ipv4_and_ipv6_peer_tuple_forms(
    selected_ip: str,
    server_addr: object,
) -> None:
    stream = RecordingStream(server_addr=server_addr, response=_OK_RESPONSE)
    backend = RecordingBackend(stream)
    request = _request(host=selected_ip, port=8080)

    response = await _execute(request, (selected_ip,), backend)

    assert response.connected_ip == selected_ip
    assert backend.connect_calls[0][:2] == (selected_ip, 8080)


@pytest.mark.parametrize(
    "server_addr",
    [
        None,
        (),
        "8.8.8.8:80",
        ("not-an-ip", 80),
        ("10.0.0.1", 80),
        ("1.1.1.1", 80),
        ("8.8.8.8", 81),
        ("8.8.8.8", "80"),
        ("8.8.8.8", 80, 0),
        ("2001:4860:4860::8888", 80, False, 0),
        ("2001:4860:4860::8888", 80, 0, 2),
    ],
)
async def test_rejects_unverified_peer_before_any_http_write(
    server_addr: object,
) -> None:
    stream = RecordingStream(server_addr=server_addr, response=_OK_RESPONSE)
    backend = RecordingBackend(stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(_request(), ("8.8.8.8",), backend)

    assert exc.value.code == "peer_verification_failed"
    assert stream.writes == []
    assert stream.closed is True
    assert len(backend.connect_calls) == 1


async def test_peer_must_equal_selected_ip_not_another_validated_answer() -> None:
    stream = RecordingStream(
        server_addr=("1.1.1.1", 80),
        response=_OK_RESPONSE,
    )
    backend = RecordingBackend(stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(_request(), ("8.8.8.8", "1.1.1.1"), backend)

    assert exc.value.code == "peer_verification_failed"
    assert stream.writes == []
    assert stream.closed is True


async def test_https_uses_original_hostname_and_secure_context() -> None:
    tls_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        response=_OK_RESPONSE,
    )
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        tls_stream=tls_stream,
    )
    backend = RecordingBackend(tcp_stream)
    request = _request(scheme="https", port=443)

    response = await _execute(request, ("8.8.8.8",), backend)

    assert len(tcp_stream.tls_calls) == 1
    context, server_hostname, timeout = tcp_stream.tls_calls[0]
    assert server_hostname == "api.example.com"
    assert timeout == 5.0
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.minimum_version >= ssl.TLSVersion.TLSv1_2
    assert context.keylog_filename is None
    assert _header_values(b"".join(tls_stream.writes), b"host") == [b"api.example.com"]
    assert response.connected_ip == "8.8.8.8"
    assert tls_stream.closed is True


async def test_https_reverifies_and_closes_the_post_tls_peer() -> None:
    tls_stream = RecordingStream(
        server_addr=("1.1.1.1", 443),
        response=_OK_RESPONSE,
    )
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        tls_stream=tls_stream,
    )
    backend = RecordingBackend(tcp_stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(
            _request(scheme="https", port=443),
            ("8.8.8.8",),
            backend,
        )

    assert exc.value.code == "peer_verification_failed"
    assert len(tcp_stream.tls_calls) == 1
    assert tls_stream.writes == []
    assert tls_stream.closed is True


async def test_https_rejects_stream_without_tls_evidence_before_write() -> None:
    tls_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        response=_OK_RESPONSE,
    )
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        tls_stream=tls_stream,
        provide_tls_object=False,
    )
    backend = RecordingBackend(tcp_stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(
            _request(scheme="https", port=443),
            ("8.8.8.8",),
            backend,
        )

    assert exc.value.code == "tls_error"
    assert tls_stream.writes == []
    assert tls_stream.closed is True
    assert tcp_stream.closed is True


async def test_https_rejects_missing_tls_stream_with_typed_error() -> None:
    tcp_stream = NullTLSResultStream(server_addr=("8.8.8.8", 443))
    backend = RecordingBackend(tcp_stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(
            _request(scheme="https", port=443),
            ("8.8.8.8",),
            backend,
        )

    assert exc.value.code == "tls_error"
    assert tcp_stream.writes == []
    assert tcp_stream.closed is True


async def test_connect_timeout_is_typed_and_not_retried() -> None:
    stream = RecordingStream(server_addr=("8.8.8.8", 80))
    backend = RecordingBackend(
        stream,
        connect_error=httpcore.ConnectTimeout("upstream timeout detail"),
    )

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(_request(), ("8.8.8.8",), backend)

    assert exc.value.code == "connect_timeout"
    assert "upstream timeout detail" not in repr(exc.value)
    assert exc.value.__context__ is None
    assert len(backend.connect_calls) == 1
    assert backend.sleep_calls == []


async def test_tls_timeout_is_connect_timeout_and_closes_stream() -> None:
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        tls_error=httpcore.ConnectTimeout("tls timeout detail"),
    )
    backend = RecordingBackend(tcp_stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(
            _request(scheme="https", port=443),
            ("8.8.8.8",),
            backend,
        )

    assert exc.value.code == "connect_timeout"
    assert "tls timeout detail" not in repr(exc.value)
    assert exc.value.__context__ is None
    assert tcp_stream.closed is True
    assert len(backend.connect_calls) == 1
    assert backend.sleep_calls == []


async def test_cancellation_during_tls_preserves_cancel_and_closes_stream() -> None:
    tcp_stream = BlockingTLSStream(server_addr=("8.8.8.8", 443))
    backend = RecordingBackend(tcp_stream)
    task = asyncio.create_task(
        _execute(
            _request(scheme="https", port=443),
            ("8.8.8.8",),
            backend,
        )
    )
    await asyncio.wait_for(tcp_stream.tls_started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert tcp_stream.closed is True
    assert len(backend.connect_calls) == 1
    assert backend.sleep_calls == []


async def test_cancellation_during_response_read_closes_stream() -> None:
    stream = BlockingReadStream(server_addr=("8.8.8.8", 80))
    backend = RecordingBackend(stream)
    task = asyncio.create_task(_execute(_request(), ("8.8.8.8",), backend))
    await asyncio.wait_for(stream.read_started.wait(), timeout=1.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.closed is True
    assert len(backend.connect_calls) == 1
    assert backend.sleep_calls == []


@pytest.mark.parametrize(
    ("scheme", "host", "port", "expected_host"),
    [
        ("http", "api.example.com", 80, b"api.example.com"),
        ("http", "api.example.com", 8080, b"api.example.com:8080"),
        ("https", "api.example.com", 443, b"api.example.com"),
        ("https", "api.example.com", 8443, b"api.example.com:8443"),
        ("http", "8.8.8.8", 80, b"8.8.8.8"),
        ("http", "8.8.8.8", 8080, b"8.8.8.8:8080"),
        ("https", "8.8.8.8", 443, b"8.8.8.8"),
        ("https", "8.8.8.8", 8443, b"8.8.8.8:8443"),
        (
            "http",
            "2001:4860:4860::8888",
            80,
            b"[2001:4860:4860::8888]",
        ),
        (
            "http",
            "2001:4860:4860::8888",
            8080,
            b"[2001:4860:4860::8888]:8080",
        ),
        (
            "https",
            "2001:4860:4860::8888",
            443,
            b"[2001:4860:4860::8888]",
        ),
        (
            "https",
            "2001:4860:4860::8888",
            8443,
            b"[2001:4860:4860::8888]:8443",
        ),
    ],
)
async def test_emits_one_exact_host_header(
    scheme: str,
    host: str,
    port: int,
    expected_host: bytes,
) -> None:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        selected_ip = "8.8.8.8"
    else:
        selected_ip = host
    peer = (selected_ip, port, 0, 0) if ":" in selected_ip else (selected_ip, port)
    active_stream = RecordingStream(server_addr=peer, response=_OK_RESPONSE)
    if scheme == "https":
        tcp_stream = RecordingStream(server_addr=peer, tls_stream=active_stream)
        backend = RecordingBackend(tcp_stream)
    else:
        tcp_stream = active_stream
        backend = RecordingBackend(active_stream)

    await _execute(
        _request(scheme=scheme, host=host, port=port),
        (selected_ip,),
        backend,
    )

    assert _header_values(b"".join(active_stream.writes), b"host") == [expected_host]
    if scheme == "https":
        assert tcp_stream.tls_calls[0][1] == host


async def test_returns_redirect_without_another_connect_or_request() -> None:
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=(
            b"HTTP/1.1 302 Found\r\n"
            b"Location: https://redirect.invalid/elsewhere\r\n"
            b"Content-Length: 0\r\nConnection: close\r\n\r\n",
        ),
    )
    backend = RecordingBackend(stream)

    response = await _execute(_request(target="/start"), ("8.8.8.8",), backend)

    request_bytes = b"".join(stream.writes)
    assert response.status_code == 302
    assert ("location", "https://redirect.invalid/elsewhere") in response.headers
    assert len(backend.connect_calls) == 1
    assert request_bytes.count(b"GET /start HTTP/1.1\r\n") == 1
    assert b"redirect.invalid" not in request_bytes
    assert backend.sleep_calls == []


async def test_rejects_upgrade_response_and_closes_the_stream() -> None:
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=(b"HTTP/1.1 101 Switching Protocols\r\n" b"Connection: Upgrade\r\nUpgrade: websocket\r\n\r\n",),
    )
    backend = RecordingBackend(stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(_request(target="/upgrade"), ("8.8.8.8",), backend)

    assert exc.value.code == "protocol_error"
    assert len(backend.connect_calls) == 1
    assert stream.closed is True


async def test_connect_failure_is_not_retried_or_slept() -> None:
    stream = RecordingStream(server_addr=("8.8.8.8", 80))
    backend = RecordingBackend(
        stream,
        connect_error=httpcore.ConnectError("upstream detail"),
    )

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(_request(), ("8.8.8.8",), backend)

    assert exc.value.code == "transport_error"
    assert "upstream detail" not in str(exc.value)
    assert "upstream detail" not in repr(exc.value)
    assert exc.value.__context__ is None
    assert len(backend.connect_calls) == 1
    assert backend.sleep_calls == []
    assert stream.writes == []


async def test_tls_failure_closes_connected_stream() -> None:
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        tls_error=httpcore.ConnectError("tls upstream detail"),
    )
    backend = RecordingBackend(tcp_stream)

    with pytest.raises(http_hop.HTTPHopError) as exc:
        await _execute(
            _request(scheme="https", port=443),
            ("8.8.8.8",),
            backend,
        )

    assert exc.value.code == "tls_error"
    assert "tls upstream detail" not in str(exc.value)
    assert "tls upstream detail" not in repr(exc.value)
    assert exc.value.__context__ is None
    assert len(tcp_stream.tls_calls) == 1
    assert tcp_stream.writes == []
    assert tcp_stream.closed is True


async def test_public_http_hop_uses_real_socket_without_following_redirect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[bytes] = []
    served = asyncio.Event()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(
                b"HTTP/1.1 302 Found\r\n"
                + f"Location: http://api.example.com:{port}/not-followed\r\n".encode("ascii")
                + b"Content-Length: 0\r\nConnection: close\r\n\r\n"
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            served.set()

    resolver_calls: list[tuple[str, int, float]] = []

    async def loopback_resolver(host: str, port: int, timeout_seconds: float) -> Sequence[str]:
        resolver_calls.append((host, port, timeout_seconds))
        return ("127.0.0.1",)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    assert server.sockets
    port = int(server.sockets[0].getsockname()[1])
    original_classifier = http_hop._is_allowed_public_address
    monkeypatch.setattr(
        http_hop,
        "_is_allowed_public_address",
        lambda address: address.is_loopback or original_classifier(address),
    )
    monkeypatch.setattr(http_hop, "_default_resolver", loopback_resolver)

    async with server:
        response = await http_hop.request_http_hop(
            _request(
                port=port,
                target="/smoke",
                headers=(("x-request-id", "loopback-smoke"),),
            )
        )
        await asyncio.wait_for(served.wait(), timeout=1.0)

    assert response.status_code == 302
    assert ("location", f"http://api.example.com:{port}/not-followed") in response.headers
    assert response.resolved_ips == ("127.0.0.1",)
    assert response.connected_ip == "127.0.0.1"
    assert resolver_calls == [("api.example.com", port, 2.0)]
    assert len(requests) == 1
    assert requests[0].startswith(b"GET /smoke HTTP/1.1\r\n")
    assert _header_values(requests[0], b"host") == [f"api.example.com:{port}".encode("ascii")]
    assert _header_values(requests[0], b"authorization") == []
    assert _header_values(requests[0], b"x-request-id") == [b"loopback-smoke"]


async def test_admin_webhook_e2e_loopback_hop_is_narrow_and_status_only() -> None:
    requests: list[bytes] = []

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            requests.append(await reader.readuntil(b"\r\n\r\n"))
            writer.write(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    assert server.sockets
    port = int(server.sockets[0].getsockname()[1])

    async with server:
        response = await http_hop.request_admin_webhook_e2e_loopback_status(
            _request(
                scheme="http",
                host="127.0.0.1",
                port=port,
                method="POST",
                target="/admin-webhooks",
                body=b"{}",
            )
        )

    assert response.status_code == 204
    assert response.retry_after_seconds is None
    assert len(requests) == 1

    for host in ("localhost", "127.0.0.2", "10.0.0.7", "::1"):
        with pytest.raises(http_hop.HTTPHopError) as exc_info:
            await http_hop.request_admin_webhook_e2e_loopback_status(
                _request(scheme="http", host=host, port=port)
            )
        assert exc_info.value.code == "dns_address_denied"


async def test_ambient_http_client_state_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    netrc_path = tmp_path / ".netrc"
    netrc_path.write_text(
        "machine api.example.com login ambient-user password ambient-netrc-secret\n",
        encoding="utf-8",
    )
    netrc_path.chmod(0o600)
    hostile_path = tmp_path / "must-not-be-read"
    ambient = {
        "HOME": str(tmp_path),
        "NETRC": str(netrc_path),
        "HTTP_PROXY": "http://ambient-user:ambient-proxy-secret@127.0.0.1:1",
        "HTTPS_PROXY": "http://ambient-user:ambient-proxy-secret@127.0.0.1:1",
        "ALL_PROXY": "http://ambient-user:ambient-proxy-secret@127.0.0.1:1",
        "NO_PROXY": "unrelated.invalid",
        "HTTP_COOKIE": "session=ambient-cookie-secret",
        "HTTP_AUTHORIZATION": "Bearer ambient-auth-secret",
        "SSL_CERT_FILE": str(hostile_path),
        "SSL_CERT_DIR": str(hostile_path),
        "SSLKEYLOGFILE": str(hostile_path),
        "SSL_CLIENT_CERT": str(hostile_path),
        "SSL_CLIENT_KEY": str(hostile_path),
        "REQUESTS_CLIENT_CERT": str(hostile_path),
    }
    for name, value in ambient.items():
        monkeypatch.setenv(name, value)

    tls_stream = RecordingStream(server_addr=("8.8.8.8", 443), response=_OK_RESPONSE)
    tcp_stream = RecordingStream(
        server_addr=("8.8.8.8", 443),
        tls_stream=tls_stream,
    )
    backend = RecordingBackend(tcp_stream)

    response = await _execute(
        _request(
            scheme="https",
            port=443,
            headers=(("authorization", "Bearer explicit-route-secret"),),
        ),
        ("8.8.8.8",),
        backend,
    )

    request_bytes = b"".join(tls_stream.writes)
    context = tcp_stream.tls_calls[0][0]
    assert response.body == b"ok"
    assert backend.connect_calls[0][:2] == ("8.8.8.8", 443)
    assert tcp_stream.tls_calls[0][1] == "api.example.com"
    assert _header_values(request_bytes, b"authorization") == [b"Bearer explicit-route-secret"]
    assert _header_values(request_bytes, b"cookie") == []
    assert _header_values(request_bytes, b"proxy-authorization") == []
    assert b"ambient-" not in request_bytes
    assert context.check_hostname is True
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.keylog_filename is None
    assert context.get_ca_certs()
    assert not hostile_path.exists()


async def test_concurrent_failure_does_not_share_streams_counters_or_request_headers() -> None:
    first_stream = BlockingReadStream(
        server_addr=("8.8.8.8", 80),
        response=(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nConnection: close\r\n\r\none",),
    )
    second_stream = BlockingReadStream(
        server_addr=("8.8.8.8", 80),
        response=(b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\nsecond",),
    )
    first_backend = RecordingBackend(first_stream)
    second_backend = RecordingBackend(second_stream)
    first = asyncio.create_task(
        _execute(
            _request(
                target="/first",
                headers=(("x-request-id", "first-route"),),
                limits=http_hop.HTTPHopLimits(max_wire_bytes=3),
            ),
            ("8.8.8.8",),
            first_backend,
        )
    )
    second = asyncio.create_task(
        _execute(
            _request(
                target="/second",
                headers=(("x-request-id", "second-route"),),
                limits=http_hop.HTTPHopLimits(max_wire_bytes=5),
            ),
            ("8.8.8.8",),
            second_backend,
        )
    )

    try:
        await asyncio.wait_for(
            asyncio.gather(first_stream.read_started.wait(), second_stream.read_started.wait()),
            timeout=1.0,
        )
    except BaseException:
        first.cancel()
        second.cancel()
        first_stream.read_release.set()
        second_stream.read_release.set()
        await asyncio.gather(first, second, return_exceptions=True)
        raise
    first_stream.read_release.set()
    second_stream.read_release.set()
    first_response, second_result = await asyncio.gather(first, second, return_exceptions=True)

    assert isinstance(first_response, http_hop.HTTPHopResponse)
    assert first_response.body == b"one"
    assert first_response.wire_bytes == 3
    assert isinstance(second_result, http_hop.HTTPHopError)
    assert second_result.code == "response_too_large"
    assert len(first_backend.connect_calls) == len(second_backend.connect_calls) == 1
    assert first_stream is not second_stream
    assert first_stream.closed is second_stream.closed is True
    first_request = b"".join(first_stream.writes)
    second_request = b"".join(second_stream.writes)
    assert _header_values(first_request, b"x-request-id") == [b"first-route"]
    assert _header_values(second_request, b"x-request-id") == [b"second-route"]
    assert b"second-route" not in first_request
    assert b"first-route" not in second_request


async def test_public_http_hop_preserves_caller_httpcore_log_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=_OK_RESPONSE,
    )
    backend = RecordingBackend(stream)

    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        return ("8.8.8.8",)

    monkeypatch.setattr(http_hop, "_default_resolver", resolver)
    monkeypatch.setattr(http_hop.httpcore, "AnyIOBackend", lambda: backend)
    httpcore_logger = logging.getLogger("httpcore")
    http11_logger = logging.getLogger("httpcore.http11")
    monkeypatch.setattr(httpcore_logger, "level", logging.WARNING)
    monkeypatch.setattr(http11_logger, "level", logging.DEBUG)

    response = await http_hop.request_http_hop(_request())

    assert response.status_code == 200
    assert (httpcore_logger.level, http11_logger.level) == (logging.WARNING, logging.DEBUG)


async def test_explicit_log_hardening_protects_standalone_public_hop(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from tldw_Server_API.app import main as app_main

    secret = "STANDALONE-SECRET"
    stream = RecordingStream(
        server_addr=("8.8.8.8", 80),
        response=(
            b"HTTP/1.1 200 OK\r\n"
            + f"Set-Cookie: token={secret}\r\n".encode("ascii")
            + b"Content-Length: 0\r\nConnection: close\r\n\r\n",
        ),
    )
    backend = RecordingBackend(stream)

    async def resolver(_host: str, _port: int, _timeout_seconds: float) -> Sequence[str]:
        return ("8.8.8.8",)

    monkeypatch.setattr(http_hop, "_default_resolver", resolver)
    monkeypatch.setattr(http_hop.httpcore, "AnyIOBackend", lambda: backend)
    httpcore_logger = logging.getLogger("httpcore")
    http11_logger = logging.getLogger("httpcore.http11")
    monkeypatch.setattr(httpcore_logger, "level", logging.DEBUG)
    monkeypatch.setattr(http11_logger, "level", logging.DEBUG)
    caplog.set_level(logging.DEBUG)

    app_main.harden_httpcore_logging()
    caplog.clear()
    response = await http_hop.request_http_hop(_request())

    assert response.status_code == 200
    assert (httpcore_logger.level, http11_logger.level) == (logging.INFO, logging.INFO)
    assert secret not in caplog.text


def test_log_hardening_preserves_stricter_httpcore_log_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as app_main

    httpcore_logger = logging.getLogger("httpcore")
    http11_logger = logging.getLogger("httpcore.http11")
    monkeypatch.setattr(httpcore_logger, "level", logging.WARNING)
    monkeypatch.setattr(http11_logger, "level", logging.ERROR)

    app_main.harden_httpcore_logging()

    assert (httpcore_logger.level, http11_logger.level) == (logging.WARNING, logging.ERROR)


def test_log_hardening_raises_inherited_debug_levels_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app import main as app_main

    root_logger = logging.getLogger()
    httpcore_logger = logging.getLogger("httpcore")
    http11_logger = logging.getLogger("httpcore.http11")
    monkeypatch.setattr(root_logger, "level", logging.DEBUG)
    monkeypatch.setattr(httpcore_logger, "level", logging.NOTSET)
    monkeypatch.setattr(http11_logger, "level", logging.NOTSET)

    app_main.harden_httpcore_logging()

    assert httpcore_logger.getEffectiveLevel() == logging.INFO
    assert http11_logger.getEffectiveLevel() == logging.INFO
