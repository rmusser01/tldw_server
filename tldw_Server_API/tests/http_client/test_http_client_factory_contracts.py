"""Regression coverage for dynamic HTTP factory contracts."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import TYPE_CHECKING

import aiohttp
import httpx
import pytest

from tldw_Server_API.app.core import http_client as hc

pytestmark = pytest.mark.unit

if TYPE_CHECKING:
    _sync_response: hc.SyncResponseLike = httpx.Response(200)
    _async_response: hc.AsyncResponseLike = hc._AiohttpResponse(SimpleNamespace(), b"")
    _adapters: tuple[hc.TransportAdapter, hc.TransportAdapter] = (
        hc.HttpxAdapter(),
        hc.AiohttpAdapter(),
    )
    _bytes: AsyncIterator[bytes] = _adapters[0].stream_bytes(method="GET", url="https://example.com")
    _events: AsyncIterator[hc.SSEEvent] = _adapters[1].stream_sse(url="https://example.com")


@pytest.mark.parametrize("factory", [hc.create_client, hc.create_async_client])
@pytest.mark.parametrize("stub_timeout", [None, float])
def test_dynamic_factory_preserves_real_timeout(monkeypatch, factory, stub_timeout):
    """A partial httpx module must not coerce a real Timeout to float."""
    timeout = httpx.Timeout(connect=1, read=2, write=3, pool=4)

    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

    stub = SimpleNamespace(Client=Client, AsyncClient=Client)
    if stub_timeout is not None:
        stub.Timeout = stub_timeout
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: stub)
    client = factory(timeout=timeout)
    assert client.timeout is timeout


@pytest.mark.parametrize("factory", [hc.create_client, hc.create_async_client])
def test_dynamic_factory_keeps_default_timeout_components(monkeypatch, factory):
    class Client:
        def __init__(self, timeout):
            self.timeout = timeout

    stub = SimpleNamespace(Client=Client, AsyncClient=Client, Timeout=httpx.Timeout)
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: stub)
    timeout = factory().timeout
    assert (timeout.connect, timeout.read, timeout.write, timeout.pool) == (
        hc.DEFAULT_CONNECT_TIMEOUT,
        hc.DEFAULT_READ_TIMEOUT,
        hc.DEFAULT_WRITE_TIMEOUT,
        hc.DEFAULT_POOL_TIMEOUT,
    )


def test_dynamic_limits_factory_remains_selected(monkeypatch):
    class Limits:
        def __init__(self, max_connections):
            self.max_connections = max_connections

    monkeypatch.setattr(hc, "_resolve_httpx", lambda: SimpleNamespace(Limits=Limits))
    limits = hc.build_limits(max_connections=7)
    assert isinstance(limits, Limits)
    assert limits.max_connections == 7


@pytest.mark.asyncio
async def test_default_aiohttp_connector_still_verifies_certificates(monkeypatch):
    monkeypatch.setattr(hc, "_build_ssl_context", lambda *_: None)
    monkeypatch.setattr(hc, "_AIOHTTP_SESSION_CACHE", {})
    session = hc._get_aiohttp_session()
    try:
        connector = session.connector
        assert isinstance(connector, aiohttp.TCPConnector)
        assert connector._ssl is True
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_create_aiohttp_session_carries_default_headers_pins_and_tls_floor(monkeypatch):
    import ssl

    monkeypatch.setenv("HTTP_CERT_PINS", "audio.example.test=AbCd|EF01")
    monkeypatch.setenv("HTTP_DEFAULT_USER_AGENT", "tldw-test-agent")
    monkeypatch.setattr(hc, "ENFORCE_TLS_MIN", True)
    monkeypatch.setattr(hc, "TLS_MIN_VERSION", "1.3")
    session = hc.create_aiohttp_session(timeout=7.5, trust_env=False)
    try:
        assert session.headers["User-Agent"] == "tldw-test-agent"
        assert hc._get_client_cert_pins(session) == {"audio.example.test": {"abcd", "ef01"}}
        assert session.timeout.total == 7.5
        assert session.trust_env is False
        assert isinstance(session.connector, aiohttp.TCPConnector)
        assert session.connector._ssl.minimum_version == ssl.TLSVersion.TLSv1_3
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_create_aiohttp_session_defaults_match_the_shared_session(monkeypatch):
    monkeypatch.setattr(hc, "_AIOHTTP_SESSION_CACHE", {})
    owned = hc.create_aiohttp_session()
    shared = hc._get_aiohttp_session()
    try:
        assert owned is not shared
        assert owned.timeout == shared.timeout
        assert owned.trust_env is shared.trust_env is hc.DEFAULT_TRUST_ENV
        assert dict(owned.headers) == dict(shared.headers)
    finally:
        await owned.close()
        await shared.close()
