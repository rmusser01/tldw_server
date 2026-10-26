"""Pinned connections fall back through the vetted DNS answers (TASK-13950.11 group C).

http_client pins every connection to an address the egress policy approved. When the
first approved address refuses the connect (``localhost`` answering ``::1`` first for a
server bound to 127.0.0.1 only), the next approved address from the same resolution is
tried, with the same Host and SNI. Nothing is re-resolved, and an address the policy did
not approve is never tried. Only the last test opens real (loopback) sockets; the others
run httpx over a MockTransport, aiohttp over a fake session, and record
``socket.create_connection``.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from urllib.parse import urlsplit

import aiohttp
import httpx
import pytest

from tldw_Server_API.app.core import http_client
from tldw_Server_API.app.core.exceptions import EgressPolicyError, NetworkError
from tldw_Server_API.app.core.Security import egress
from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

pytestmark = pytest.mark.unit

_LOCAL = "http://localhost:8080/health"
_PUBLIC = "https://dual.example/resource"
_PUBLIC_V6 = "2606:2800:220:1:248:1893:25c8:1946"
_UP = frozenset({"127.0.0.1", "93.184.216.34"})


@pytest.fixture
def dns(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """The production egress policy over a DNS table; returns the table."""
    answers = {"localhost": ["::1", "127.0.0.1"], "dual.example": [_PUBLIC_V6, "93.184.216.34"]}
    monkeypatch.setattr(http_client, "is_explicit_pytest_runtime", lambda: False)
    monkeypatch.setattr(http_client, "is_test_mode", lambda: False)
    monkeypatch.setattr(http_client, "env_flag_enabled", lambda _name: False)
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.setattr(egress, "_resolve_host_ips", lambda host, **_kwargs: list(answers.get(host, [])))
    return answers


@pytest.fixture
def sockets(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Record every stdlib socket connect and refuse it."""
    attempts: list[tuple[str, int]] = []

    def refuse(address, *_args, **_kwargs):
        attempts.append(address)
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr(socket, "create_connection", refuse)
    return attempts


class _Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    """An unread body, so the streaming and byte-capped paths can read it."""

    def __iter__(self):
        yield b"ok"

    async def __aiter__(self):
        yield b"ok"


def _transport(seen: list[tuple[str, str, object]], up=_UP) -> httpx.MockTransport:
    """Record (address, Host, SNI) per connect; refuse addresses not in ``up``."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers["host"], request.extensions.get("sni_hostname")))
        if request.url.host not in up:
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(200, stream=_Body(), request=request)

    return httpx.MockTransport(handle)


class _AiohttpSession:
    """Just enough of aiohttp.ClientSession for http_client's request and stream I/O."""

    def __init__(self, seen: list[tuple[str, str, object]], up=_UP) -> None:
        self.seen = seen
        self.up = up
        self.cookie_jar = SimpleNamespace(clear=lambda: None)

    @asynccontextmanager
    async def request(self, method, url, **kwargs):
        host = urlsplit(url).hostname or ""
        self.seen.append((host, kwargs["headers"]["Host"], kwargs.get("server_hostname")))
        if host not in self.up:
            raise aiohttp.ClientConnectorError(
                SimpleNamespace(host=host, port=8080, ssl=None),
                ConnectionRefusedError(61, "Connection refused"),
            )
        remaining = bytearray(b"ok")

        async def read(size=None):
            chunk = bytes(remaining if size is None or size < 0 else remaining[:size])
            del remaining[: len(chunk)]
            return chunk

        async def iter_any():
            yield await read()

        yield SimpleNamespace(
            status=200,
            headers={},
            url=url,
            read=read,
            release=lambda: None,
            content=SimpleNamespace(read=read, iter_any=iter_any, iter_chunked=lambda _size: iter_any()),
        )


_ONE_ATTEMPT = http_client.RetryPolicy(attempts=1)
_LOCAL_SCOPE = ConfiguredEndpointScope.from_url(_LOCAL)


async def _afetch_httpx(seen, *, url=_LOCAL, up=_UP, **kwargs):
    async with httpx.AsyncClient(transport=_transport(seen, up)) as client:
        response = await http_client.afetch(
            method="GET",
            url=url,
            client=client,
            configured_endpoint=_LOCAL_SCOPE,
            retry=_ONE_ATTEMPT,
            **kwargs,
        )
        return response.content


async def _afetch_httpx_bounded(seen, **kwargs):
    return await _afetch_httpx(seen, max_response_bytes=10, **kwargs)


async def _afetch_aiohttp(seen, *, url=_LOCAL, up=_UP):
    response = await http_client._afetch_aiohttp(
        method="GET",
        url=url,
        client=_AiohttpSession(seen, up),
        configured_endpoint=_LOCAL_SCOPE,
        retry=_ONE_ATTEMPT,
    )
    return response.content


async def _fetch_sync(seen, *, url=_LOCAL, up=_UP):
    with httpx.Client(transport=_transport(seen, up)) as client:
        response = http_client.fetch(
            method="GET",
            url=url,
            client=client,
            configured_endpoint=_LOCAL_SCOPE,
            retry=_ONE_ATTEMPT,
        )
        return response.content


async def _astream_httpx(seen, *, url=_PUBLIC, up=_UP):
    async with httpx.AsyncClient(transport=_transport(seen, up)) as client:
        chunks = [
            chunk
            async for chunk in http_client.astream_bytes(
                method="GET", url=url, client=client, retry=_ONE_ATTEMPT, timeout=5.0
            )
        ]
        return b"".join(chunks)


async def _astream_aiohttp(seen, *, url=_PUBLIC, up=_UP):
    chunks = [
        chunk
        async for chunk in http_client._astream_bytes_aiohttp(
            method="GET", url=url, client=_AiohttpSession(seen, up), retry=_ONE_ATTEMPT, timeout=5.0
        )
    ]
    return b"".join(chunks)


async def _stream_response_sync(seen, *, url=_PUBLIC, up=_UP):
    with httpx.Client(transport=_transport(seen, up)) as client:
        with http_client.stream_response(method="GET", url=url, client=client) as response:
            return response.read()


async def _download_sync(seen, *, url=_PUBLIC, up=_UP, dest):
    with httpx.Client(transport=_transport(seen, up)) as client:
        path = http_client.download(url=url, dest=dest, client=client, retry=_ONE_ATTEMPT)
        return path.read_bytes()


# path -> (runner, Host header, SNI name). httpx always names the SNI host on a pinned
# connection; aiohttp only for https.
_RUNNERS: dict[str, tuple[Callable[..., Awaitable[bytes]], str, str | None]] = {
    "afetch-httpx": (_afetch_httpx, "localhost:8080", "localhost"),
    "afetch-httpx-bounded": (_afetch_httpx_bounded, "localhost:8080", "localhost"),
    "afetch-aiohttp": (_afetch_aiohttp, "localhost:8080", None),
    "fetch-sync": (_fetch_sync, "localhost:8080", "localhost"),
    "astream-httpx": (_astream_httpx, "dual.example", "dual.example"),
    "astream-aiohttp": (_astream_aiohttp, "dual.example", "dual.example"),
    "stream-response-sync": (_stream_response_sync, "dual.example", "dual.example"),
    "download-sync": (_download_sync, "dual.example", "dual.example"),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", sorted(_RUNNERS))
async def test_a_refused_first_address_falls_back_to_the_next_vetted_one(dns, sockets, tmp_path, path):
    run, host_header, sni = _RUNNERS[path]
    seen: list[tuple[str, str, object]] = []
    kwargs = {"dest": tmp_path / "out.bin"} if path == "download-sync" else {}

    assert await run(seen, **kwargs) == b"ok"

    first, second = dns[host_header.split(":")[0]]
    # Same Host and SNI on both addresses, the vetted answers in order, nothing re-resolved.
    assert [address for address, _host, _sni in seen] == [first, second]
    assert {(host, server_name) for _address, host, server_name in seen} == {(host_header, sni)}
    assert sockets == []


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["afetch-httpx", "afetch-httpx-bounded", "afetch-aiohttp", "fetch-sync"])
async def test_every_vetted_address_failing_raises_one_network_error(dns, sockets, path):
    run, _host_header, _sni = _RUNNERS[path]
    seen: list[tuple[str, str, object]] = []

    with pytest.raises(NetworkError):
        await run(seen, up=frozenset())

    assert [address for address, _host, _sni in seen] == ["::1", "127.0.0.1"]


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["afetch-httpx", "afetch-aiohttp", "fetch-sync"])
async def test_an_egress_refused_answer_refuses_the_request_before_any_connect(dns, sockets, path):
    """One forbidden answer refuses the whole host, as before: no address is tried."""
    run, _host_header, _sni = _RUNNERS[path]
    dns["localhost"] = ["169.254.169.254", "127.0.0.1"]
    seen: list[tuple[str, str, object]] = []

    with pytest.raises(EgressPolicyError):
        await run(seen)

    assert seen == []
    assert sockets == []


@pytest.mark.asyncio
async def test_the_fallback_is_bounded(dns, sockets):
    dns["localhost"] = [f"10.0.0.{index}" for index in range(1, 9)]
    seen: list[tuple[str, str, object]] = []

    with pytest.raises(NetworkError):
        await _afetch_httpx(seen, up=frozenset())

    assert [address for address, _host, _sni in seen] == dns["localhost"][: http_client._PINNED_ADDRESS_ATTEMPTS]


def test_certificate_pin_check_falls_back_through_vetted_addresses(dns, sockets):
    with pytest.raises(EgressPolicyError, match="TLS pinning verification failed"):
        http_client._check_cert_pinning(
            "dual.example",
            443,
            {"00"},
            None,
            accepted_resolved_ips=(_PUBLIC_V6, "93.184.216.34"),
        )

    assert sockets == [(_PUBLIC_V6, 443), ("93.184.216.34", 443)]


# --- timeout=None uses the client's configured timeout -------------------------------------


@pytest.mark.asyncio
async def test_afetch_without_timeout_uses_the_client_timeout(dns):
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=7.0) as client:
        inherited = await http_client.afetch(
            method="GET", url="http://93.184.216.34/a", client=client, retry=_ONE_ATTEMPT
        )
        explicit = await http_client.afetch(
            method="GET", url="http://93.184.216.34/a", client=client, retry=_ONE_ATTEMPT, timeout=2.0
        )

    assert (inherited.status_code, explicit.status_code) == (200, 200)
    assert seen == [
        {"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0},
        {"connect": 2.0, "read": 2.0, "write": 2.0, "pool": 2.0},
    ]


def test_fetch_without_timeout_uses_the_client_timeout(dns):
    seen: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, request=request)

    with httpx.Client(transport=httpx.MockTransport(handle), timeout=7.0) as client:
        response = http_client.fetch(method="GET", url="http://93.184.216.34/a", client=client, retry=_ONE_ATTEMPT)

    assert response.status_code == 200
    assert seen == [{"connect": 7.0, "read": 7.0, "write": 7.0, "pool": 7.0}]


# --- The fallback's connect budget ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [True, False])
async def test_only_the_final_address_gets_the_full_connect_timeout_on_httpx(dns, sockets, explicit):
    """A 300 s caller must not wait 4 x 300 s: non-final tries cap connect at the default."""
    timeouts: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        timeouts.append(request.extensions["timeout"])
        if request.url.host == "::1":
            raise httpx.ConnectError("Connection refused", request=request)
        return httpx.Response(200, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle), timeout=300.0) as client:
        response = await http_client.afetch(
            method="GET",
            url=_LOCAL,
            client=client,
            configured_endpoint=_LOCAL_SCOPE,
            retry=_ONE_ATTEMPT,
            **({"timeout": 300.0} if explicit else {}),
        )

    assert response.status_code == 200
    capped = http_client.DEFAULT_CONNECT_TIMEOUT
    assert timeouts == [
        {"connect": capped, "read": 300.0, "write": 300.0, "pool": 300.0},
        {"connect": 300.0, "read": 300.0, "write": 300.0, "pool": 300.0},
    ]


@pytest.mark.asyncio
async def test_only_the_final_address_gets_the_full_connect_timeout_on_aiohttp(dns, sockets):
    seen: list[tuple[str, str, object]] = []
    session = _AiohttpSession(seen)
    timeouts: list[aiohttp.ClientTimeout] = []
    request = session.request

    def recording_request(method, url, **kwargs):
        timeouts.append(kwargs["timeout"])
        return request(method, url, **kwargs)

    session.request = recording_request
    response = await http_client._afetch_aiohttp(
        method="GET",
        url=_LOCAL,
        client=session,
        configured_endpoint=_LOCAL_SCOPE,
        retry=_ONE_ATTEMPT,
        timeout=300.0,
    )

    assert response.content == b"ok"
    capped = http_client.DEFAULT_CONNECT_TIMEOUT
    assert timeouts == [
        aiohttp.ClientTimeout(total=300.0, connect=capped, sock_connect=capped),
        aiohttp.ClientTimeout(total=300.0),
    ]


def test_certificate_pin_preflight_shares_one_connect_budget(dns, monkeypatch):
    """The preflight is synchronous on the event loop, so its addresses share one budget."""
    import time

    monkeypatch.setattr(http_client, "DEFAULT_CONNECT_TIMEOUT", 0.2)
    budgets: list[float] = []

    def slow_then_refuse(address, *, timeout):
        budgets.append(timeout)
        if len(budgets) == 1:
            time.sleep(0.15)
            raise TimeoutError("timed out")
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr(socket, "create_connection", slow_then_refuse)

    with pytest.raises(EgressPolicyError, match="TLS pinning verification failed"):
        http_client._check_cert_pinning(
            "dual.example", 443, {"00"}, None, accepted_resolved_ips=(_PUBLIC_V6, "93.184.216.34")
        )

    assert len(budgets) == 2
    assert budgets[0] == pytest.approx(0.2, abs=0.02)
    assert budgets[1] <= 0.07


# --- Real sockets: an IPv4-only loopback server behind a ::1-first answer --------------------


def _ipv6_loopback_available() -> bool:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as probe:
            probe.bind(("::1", 0))
    except OSError:
        return False
    return True


@pytest.fixture
def ipv4_only_server() -> Iterator[tuple[int, list[bytes]]]:
    """An HTTP server on 127.0.0.1 only; yields its port and every POST body it received."""
    if not _ipv6_loopback_available():
        pytest.skip("IPv6 loopback is unavailable, so ::1 cannot be the refused first answer")
    bodies: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *_args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1], bodies
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.asyncio
async def test_localhost_answering_ipv6_first_reaches_an_ipv4_only_server_once(dns, ipv4_only_server):
    port, bodies = ipv4_only_server
    url = f"http://localhost:{port}/echo"
    scope = ConfiguredEndpointScope.from_url(url)
    request = {"method": "POST", "url": url, "configured_endpoint": scope, "retry": _ONE_ATTEMPT, "timeout": 2.0}

    async with http_client.create_async_client(timeout=2.0) as client:
        via_httpx = await http_client.afetch(client=client, json={"via": "httpx"}, **request)
    session = http_client.create_aiohttp_session(timeout=2.0)
    try:
        via_aiohttp = await http_client.afetch(client=session, json={"via": "aiohttp"}, **request)
    finally:
        await session.close()
    with http_client.create_client(timeout=2.0) as sync_client:
        via_sync = http_client.fetch(client=sync_client, json={"via": "sync"}, **request)

    assert [via_httpx.status_code, via_aiohttp.status_code, via_sync.status_code] == [200, 200, 200]
    assert [json.loads(body) for body in bodies] == [{"via": "httpx"}, {"via": "aiohttp"}, {"via": "sync"}]
