from __future__ import annotations

import asyncio
import ipaddress
import ssl
from collections.abc import Iterable, Sequence
from typing import Any

import httpcore
import pytest

from tldw_Server_API.app.core.Security import http_hop

pytestmark = pytest.mark.unit

_OK_RESPONSE = (b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",)


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

    def __init__(self, *, server_addr: object) -> None:
        super().__init__(server_addr=server_addr)
        self.read_started = asyncio.Event()
        self.read_release = asyncio.Event()

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        del max_bytes, timeout
        self.read_started.set()
        await self.read_release.wait()
        return b""


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
            ("authorization", "Bearer explicit-route-secret"),
        ),
    )

    await _execute(request, ("8.8.8.8",), backend)

    request_bytes = b"".join(stream.writes)
    assert _header_values(request_bytes, b"host") == [b"api.example.com"]
    assert _header_values(request_bytes, b"connection") == [b"close"]
    assert _header_values(request_bytes, b"content-length") == [b"7"]
    assert _header_values(request_bytes, b"transfer-encoding") == []
    assert _header_values(request_bytes, b"accept-encoding") == [b"identity"]
    assert _header_values(request_bytes, b"authorization") == [b"Bearer explicit-route-secret"]
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
