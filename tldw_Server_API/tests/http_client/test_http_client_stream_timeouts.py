import asyncio
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _has_httpx():
    try:
        import httpx  # noqa: F401
        return True
    except Exception:
        return False


def _has_aiohttp():
    try:
        import aiohttp  # noqa: F401
        return True
    except Exception:
        return False


requires_httpx = pytest.mark.skipif(not _has_httpx(), reason="httpx not installed")
requires_aiohttp = pytest.mark.skipif(not _has_aiohttp(), reason="aiohttp not installed")


@requires_httpx
def test_stream_response_scoped_lan_lifetime_and_borrowed_client(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress as egress_mod
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
    import httpx

    monkeypatch.setattr(
        egress_mod,
        "_resolve_host_ips",
        lambda _host: ["192.168.1.50"],
    )
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")
    monkeypatch.setenv("WORKFLOWS_EGRESS_ALLOWED_PORTS", "80,443")
    url = "http://192.168.1.50:11434/v1/chat/completions"
    scope = ConfiguredEndpointScope.from_url(url)
    calls: list[str] = []

    class StreamingBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"one\n"
            yield b"two\n"

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, request=request, stream=StreamingBody())

    client = hc.create_client(transport=httpx.MockTransport(handler))
    try:
        with hc.stream_response(
            method="POST",
            url=url,
            client=client,
            configured_endpoint=scope,
            json={"stream": True},
        ) as response:
            assert not response.is_closed
            assert b"".join(response.iter_bytes()) == b"one\ntwo\n"
        assert response.is_closed
        assert not client.is_closed
    finally:
        client.close()

    assert calls == [url]


@requires_httpx
def test_stream_response_connects_to_vetted_ip_and_preserves_http_identity(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress as egress_mod
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope
    import httpx
    import types

    original_url = "https://models.internal:11434/v1/chat/completions"
    scope = ConfiguredEndpointScope.from_url(original_url)
    observed: dict[str, object] = {}

    class StreamingBody(httpx.SyncByteStream):
        def __iter__(self):
            yield b"ok"

    def allow(_url, **_kwargs):
        return types.SimpleNamespace(
            allowed=True,
            reason=None,
            reason_code=None,
            resolved_ips=("192.0.2.10",),
        )

    def handler(request: httpx.Request) -> httpx.Response:
        observed["transport_url"] = str(request.url)
        observed["host"] = request.headers.get("host")
        observed["sni"] = request.extensions.get("sni_hostname")
        return httpx.Response(200, request=request, stream=StreamingBody())

    monkeypatch.setattr(egress_mod, "evaluate_url_policy", allow)
    client = hc.create_client(transport=httpx.MockTransport(handler))
    try:
        with hc.stream_response(
            method="POST",
            url=original_url,
            client=client,
            configured_endpoint=scope,
        ) as response:
            assert b"".join(response.iter_bytes()) == b"ok"
            assert str(response.request.url) == original_url
    finally:
        client.close()

    assert observed == {
        "transport_url": "https://192.0.2.10:11434/v1/chat/completions",
        "host": "models.internal:11434",
        "sni": "models.internal",
    }


@requires_httpx
def test_stream_response_owns_client_and_forces_redirects_off(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda _proxies: None)
    state = {"closed": False, "create": None, "stream": None}
    response = SimpleNamespace(status_code=302)

    class DummyClient:
        @contextmanager
        def stream(self, method, url, **kwargs):  # noqa: ARG002
            state["stream"] = kwargs
            yield response

        def close(self):
            state["closed"] = True

    def fake_create_client(**kwargs):
        state["create"] = kwargs
        return DummyClient()

    monkeypatch.setattr(hc, "create_client", fake_create_client)

    with hc.stream_response(
        method="GET",
        url="https://example.com/start",
        follow_redirects=True,
        headers={"Accept-Encoding": "gzip, zstd"},
        proxies="http://proxy.example:8080",
        timeout=7.5,
        trust_env="false",
        verify=False,
    ) as yielded:
        assert yielded is response
        assert not state["closed"]

    assert state["closed"] is True
    assert state["create"] == {
        "timeout": 7.5,
        "proxies": "http://proxy.example:8080",
        "trust_env": False,
        "cert_pinning": None,
        "verify": False,
    }
    assert state["stream"]["follow_redirects"] is False
    assert state["stream"]["timeout"] == 7.5
    assert state["stream"]["headers"]["Accept-Encoding"] == "gzip"


@requires_httpx
def test_stream_response_rejects_scope_mismatch_before_io():
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    calls = {"stream": 0}

    class DummyClient:
        def stream(self, *_args, **_kwargs):
            calls["stream"] += 1
            raise AssertionError("network I/O must not start")

    scope = ConfiguredEndpointScope.from_url("http://192.168.1.50:11434")
    with pytest.raises(EgressPolicyError) as exc:
        with hc.stream_response(
            method="GET",
            url="http://192.168.1.50:11435/models",
            client=DummyClient(),
            configured_endpoint=scope,
        ):
            pass

    assert exc.value.reason_code == "origin_mismatch"
    assert calls["stream"] == 0


@requires_httpx
def test_stream_response_enforces_certificate_pin(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import EgressPolicyError
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("https://93.184.216.34:11434")
    calls = {"stream": 0}

    class DummyClient:
        def stream(self, *_args, **_kwargs):
            calls["stream"] += 1
            raise AssertionError("network I/O must not start")

    def deny_pin(*_args, **kwargs):
        assert kwargs["configured_endpoint"] is scope
        assert kwargs["accepted_resolved_ips"] == ("93.184.216.34",)
        raise EgressPolicyError("pin error", reason_code="tls_pin_error")

    monkeypatch.setattr(hc, "_check_cert_pinning", deny_pin)

    with pytest.raises(EgressPolicyError) as exc:
        with hc.stream_response(
            method="GET",
            url="https://93.184.216.34:11434/stream",
            client=DummyClient(),
            configured_endpoint=scope,
            cert_pinning={"93.184.216.34": {"pin"}},
        ):
            pass

    assert exc.value.reason_code == "tls_pin_error"
    assert calls["stream"] == 0


class StreamResponse:
    def __init__(self, url: str, status_code: int = 200) -> None:
        self.status_code = status_code
        self.request = SimpleNamespace(url=url)

    def raise_for_status(self) -> None:
        return None


class AioStreamResponse:
    def __init__(self, url: str, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self.url = url
        self.headers = headers or {}

    async def read(self) -> bytes:
        return b""


@requires_httpx
@pytest.mark.asyncio
async def test_stream_bytes_retries_http_503_before_first_chunk_httpx(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    import httpx

    calls = {"n": 0}

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            request = httpx.Request("POST", "http://93.184.216.34/stream")
            response = httpx.Response(503, request=request)

            async def iter_bytes():
                if False:  # pragma: no cover
                    yield b""

            yield response, iter_bytes()
            return

        async def iter_bytes():
            yield b"ok"

        resp = StreamResponse("http://93.184.216.34/stream")
        yield resp, iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)

    chunks = []
    async for chunk in hc.astream_bytes(
        method="POST",
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=2, retry_on_unsafe=True),
    ):
        chunks.append(chunk)

    assert calls["n"] == 2
    assert chunks == [b"ok"]


@requires_httpx
@pytest.mark.asyncio
async def test_stream_bytes_honors_date_form_retry_after_before_first_chunk_httpx(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    import httpx

    calls = {"n": 0}
    delays: list[float] = []
    retry_after = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=5), usegmt=True)

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            request = httpx.Request("POST", "http://93.184.216.34/stream")
            response = httpx.Response(
                503,
                request=request,
                headers={"Retry-After": retry_after},
            )

            async def iter_bytes():
                if False:  # pragma: no cover
                    yield b""

            yield response, iter_bytes()
            return

        async def iter_bytes():
            yield b"ok"

        resp = StreamResponse("http://93.184.216.34/stream")
        yield resp, iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", fake_sleep)

    chunks = []
    async for chunk in hc.astream_bytes(
        method="POST",
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=2, retry_on_unsafe=True),
    ):
        chunks.append(chunk)

    assert calls["n"] == 2
    assert chunks == [b"ok"]
    assert len(delays) == 1
    assert delays[0] >= 3.0


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_bytes_retries_aiohttp_transport_error_before_first_chunk(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {"n": 0}

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("socket reset")

        async def iter_bytes():
            yield b"ok"

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)

    chunks = []
    async for chunk in hc._astream_bytes_aiohttp(
        method="POST",
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=2, retry_on_unsafe=True),
    ):
        chunks.append(chunk)

    assert calls["n"] == 2
    assert chunks == [b"ok"]


@requires_httpx
@pytest.mark.asyncio
async def test_stream_bytes_does_not_retry_after_first_chunk_httpx(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {"n": 0}

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        calls["n"] += 1

        async def iter_bytes():
            yield b"first"
            raise hc.NetworkError("stream dropped")

        resp = StreamResponse("http://93.184.216.34/stream")
        yield resp, iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)

    chunks = []
    with pytest.raises(hc.NetworkError) as exc:
        async for chunk in hc.astream_bytes(
            method="POST",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2, retry_on_unsafe=True),
        ):
            chunks.append(chunk)

    assert calls["n"] == 1
    assert chunks == [b"first"]
    assert "stream dropped" in str(exc.value)


@requires_httpx
@pytest.mark.asyncio
async def test_stream_bytes_first_byte_timeout_httpx(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        async def iter_bytes():
            await asyncio.sleep(0.05)
            yield b"late"

        resp = StreamResponse("http://93.184.216.34/stream")
        yield resp, iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)

    timeout = SimpleNamespace(connect=0.01, read=0.1)
    with pytest.raises(hc.NetworkError) as exc:
        async for _ in hc.astream_bytes(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            timeout=timeout,
        ):
            pass
    assert "StreamTimeout:first_byte" in str(exc.value)


@requires_httpx
@pytest.mark.asyncio
async def test_stream_bytes_idle_timeout_httpx(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        async def iter_bytes():
            yield b"first"
            await asyncio.sleep(0.05)
            yield b"second"

        resp = StreamResponse("http://93.184.216.34/stream")
        yield resp, iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)

    timeout = SimpleNamespace(connect=0.01, read=0.01)
    chunks = []
    with pytest.raises(hc.NetworkError) as exc:
        async for chunk in hc.astream_bytes(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            timeout=timeout,
        ):
            chunks.append(chunk)
    assert chunks == [b"first"]
    assert "StreamTimeout:idle" in str(exc.value)


@requires_httpx
@pytest.mark.asyncio
async def test_sse_retries_on_timeout_httpx(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {"n": 0}

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            async def iter_bytes():
                await asyncio.sleep(0.05)
                yield b"data: timeout\n\n"
        else:
            async def iter_bytes():
                yield b"data: ok\n\n"

        resp = StreamResponse("http://93.184.216.34/stream")
        yield resp, iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)

    timeout = SimpleNamespace(connect=0.01, read=0.01)
    events = []
    async for ev in hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=2),
        timeout=timeout,
    ):
        events.append(ev)
        if events:
            break
    assert calls["n"] == 2
    assert events[0].data == "ok"
