import asyncio
import socket
from collections.abc import Mapping
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
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
    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress as egress_mod
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

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
    import types

    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security import egress as egress_mod
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

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
    def __init__(
        self,
        url: str,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
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


class ClosableByteIterator:
    def __init__(
        self,
        chunks: list[bytes] | None = None,
        *,
        block: bool = False,
    ) -> None:
        self._chunks = iter(chunks or [])
        self._block = block
        self.close_calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        if self._block:
            self.started.set()
            await self.release.wait()
            raise StopAsyncIteration
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_timeout_iterator_closes_delegate_on_early_close() -> None:
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator([b"one", b"two"])
    stream = hc._iter_bytes_with_timeouts(source, None)

    assert await stream.__anext__() == b"one"
    await stream.aclose()

    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_timeout_iterator_preserves_context_bound_delegate_through_exhaustion() -> None:
    from tldw_Server_API.app.core import http_client as hc

    owned = ContextVar("timeout_stream_owned", default=False)
    finalized: list[str] = []

    async def source():
        token = owned.set(True)
        try:
            yield b"one"
            assert owned.get() is True
            yield b"two"
        finally:
            owned.reset(token)
            finalized.append("closed")

    chunks = [chunk async for chunk in hc._iter_bytes_with_timeouts(source(), None)]

    assert chunks == [b"one", b"two"]
    assert finalized == ["closed"]


@pytest.mark.asyncio
async def test_timeout_iterator_closes_context_bound_delegate_in_owner_context() -> None:
    from tldw_Server_API.app.core import http_client as hc

    owned = ContextVar("timeout_stream_owned", default=False)
    finalized: list[str] = []

    async def source():
        token = owned.set(True)
        try:
            yield b"one"
            yield b"two"
        finally:
            owned.reset(token)
            finalized.append("closed")

    stream = hc._iter_bytes_with_timeouts(source(), None)

    assert await stream.__anext__() == b"one"
    await stream.aclose()

    assert finalized == ["closed"]


@pytest.mark.asyncio
async def test_timeout_iterator_closes_delegate_after_timeout() -> None:
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator(block=True)
    stream = hc._iter_bytes_with_timeouts(
        source,
        SimpleNamespace(connect=0.001, read=0.001),
    )

    with pytest.raises(hc.NetworkError, match="StreamTimeout:first_byte"):
        await stream.__anext__()

    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_timeout_iterator_closes_delegate_after_cancellation() -> None:
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator(block=True)
    stream = hc._iter_bytes_with_timeouts(source, None)
    task = asyncio.create_task(stream.__anext__())
    await source.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert source.close_calls == 1


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_byte_stream_closes_raw_iterator_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator([b"one", b"two"])

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        yield StreamResponse("http://93.184.216.34/stream"), source

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc._astream_bytes_httpx(
        method="GET",
        url="http://93.184.216.34/stream",
        client=object(),
    )

    assert await stream.__anext__() == b"one"
    await stream.aclose()

    assert source.close_calls == 1


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_byte_stream_closes_raw_iterator_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator([b"one", b"two"])

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        yield AioStreamResponse("http://93.184.216.34/stream"), source

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc._astream_bytes_aiohttp(
        method="GET",
        url="http://93.184.216.34/stream",
        client=object(),
    )

    assert await stream.__anext__() == b"one"
    await stream.aclose()

    assert source.close_calls == 1


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sse_stream_closes_raw_iterator_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator([b"data: one\n\n", b"data: two\n\n"])

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        yield StreamResponse("http://93.184.216.34/stream"), source

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
    )

    assert (await stream.__anext__()).data == "one"
    await stream.aclose()

    assert source.close_calls == 1


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sse_stream_closes_raw_iterator_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    source = ClosableByteIterator([b"data: one\n\n", b"data: two\n\n"])

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        yield AioStreamResponse("http://93.184.216.34/stream"), source

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
    )

    assert (await stream.__anext__()).data == "one"
    await stream.aclose()

    assert source.close_calls == 1


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sse_sensitive_status_preserves_safe_error_semantics(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        yield StreamResponse("http://93.184.216.34/stream", status_code=401), ClosableByteIterator()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=1),
        sensitive_observability=True,
    )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 401$"):
        await stream.__anext__()


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sse_sensitive_status_preserves_safe_error_semantics(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        yield AioStreamResponse("http://93.184.216.34/stream", status=401), ClosableByteIterator()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=1),
        sensitive_observability=True,
    )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 401$"):
        await stream.__anext__()


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sse_does_not_retry_non_retriable_status(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        yield StreamResponse("http://93.184.216.34/stream", status_code=404), ClosableByteIterator()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
    )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 404$") as exc_info:
        await stream.__anext__()

    assert calls == 1
    assert exc_info.value.status_code == 404
    assert getattr(exc_info.value, "request", None) is None
    assert getattr(exc_info.value, "response", None) is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sse_does_not_retry_non_retriable_status(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        yield AioStreamResponse("http://93.184.216.34/stream", status=404), ClosableByteIterator()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
    )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 404$") as exc_info:
        await stream.__anext__()

    assert calls == 1
    assert exc_info.value.status_code == 404
    assert getattr(exc_info.value, "request", None) is None
    assert getattr(exc_info.value, "response", None) is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sensitive_sse_preserves_stream_timeout_reason(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        async def iter_bytes():
            await asyncio.sleep(0.05)
            yield b"data: late\n\n"

        yield StreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=1),
        timeout=SimpleNamespace(connect=0.001, read=0.001),
        sensitive_observability=True,
    )

    with pytest.raises(hc.NetworkError, match=r"^StreamTimeout:first_byte$"):
        await stream.__anext__()


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sensitive_sse_preserves_stream_timeout_reason(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        async def iter_bytes():
            await asyncio.sleep(0.05)
            yield b"data: late\n\n"

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=1),
        timeout=SimpleNamespace(connect=0.001, read=0.001),
        sensitive_observability=True,
    )

    with pytest.raises(hc.NetworkError, match=r"^StreamTimeout:first_byte$"):
        await stream.__anext__()


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sse_does_not_retry_after_first_event(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            yield b"data: one\n\n"
            raise hc.NetworkError("stream dropped")

        yield StreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    events = []

    with pytest.raises(hc.NetworkError, match=r"^stream dropped$"):
        async for event in hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        ):
            events.append(event.data)

    assert events == ["one"]
    assert calls == 1


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sse_does_not_retry_after_first_event(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            yield b"data: one\n\n"
            raise hc.NetworkError("stream dropped")

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    events = []

    with pytest.raises(hc.NetworkError, match=r"^stream dropped$"):
        async for event in hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        ):
            events.append(event.data)

    assert events == ["one"]
    assert calls == 1


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sse_does_not_retry_after_partial_body_chunk(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            yield b"data: partial"
            raise hc.NetworkError("stream dropped")

        yield StreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    events = []

    with pytest.raises(hc.NetworkError, match=r"^stream dropped$"):
        async for event in hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        ):
            events.append(event.data)

    assert events == []
    assert calls == 1


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sse_does_not_retry_after_partial_body_chunk(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            yield b"data: partial"
            raise hc.NetworkError("stream dropped")

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    events = []

    with pytest.raises(hc.NetworkError, match=r"^stream dropped$"):
        async for event in hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        ):
            events.append(event.data)

    assert events == []
    assert calls == 1


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_byte_stream_does_not_retry_non_retriable_status(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        yield AioStreamResponse("http://93.184.216.34/stream", status=404), ClosableByteIterator()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_bytes(
        method="GET",
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
    )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 404$"):
        await stream.__anext__()

    assert calls == 1


@requires_httpx
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["bytes", "sse"])
async def test_httpx_terminal_status_survives_stream_context_exit_failure(
    monkeypatch,
    stream_kind,
):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    request = httpx.Request("GET", "http://93.184.216.34/stream")

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        try:
            yield httpx.Response(404, request=request), ClosableByteIterator()
        finally:
            raise httpx.ReadError("response cleanup failed", request=request)

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    if stream_kind == "bytes":
        stream = hc.astream_bytes(
            method="GET",
            url=str(request.url),
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        )
    else:
        stream = hc.astream_sse(
            url=str(request.url),
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 404$"):
        await stream.__anext__()

    assert calls == 1


@requires_httpx
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["bytes", "sse"])
async def test_httpx_invalid_terminal_status_is_sanitized_and_detached(
    monkeypatch,
    stream_kind,
):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    request = httpx.Request("GET", "http://93.184.216.34/stream")

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        yield httpx.Response(700, request=request), ClosableByteIterator()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    if stream_kind == "bytes":
        stream = hc.astream_bytes(
            method="GET",
            url=str(request.url),
            client=object(),
            retry=hc.RetryPolicy(attempts=1),
        )
    else:
        stream = hc.astream_sse(
            url=str(request.url),
            client=object(),
            retry=hc.RetryPolicy(attempts=1),
        )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 700$") as exc_info:
        await stream.__anext__()

    assert exc_info.value.status_code is None
    assert getattr(exc_info.value, "request", None) is None
    assert getattr(exc_info.value, "response", None) is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@requires_httpx
@pytest.mark.asyncio
async def test_terminal_status_does_not_hide_non_transport_cleanup_failure(
    monkeypatch,
):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    request = httpx.Request("GET", "http://93.184.216.34/stream")

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        try:
            yield httpx.Response(404, request=request), ClosableByteIterator()
        finally:
            raise ValueError("cleanup implementation bug")

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc.astream_bytes(
        method="GET",
        url=str(request.url),
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
    )

    with pytest.raises(ValueError, match="cleanup implementation bug"):
        await stream.__anext__()

    assert calls == 1


@requires_aiohttp
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["bytes", "sse"])
async def test_aiohttp_terminal_status_survives_stream_context_exit_failure(
    monkeypatch,
    stream_kind,
):
    import aiohttp

    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        try:
            yield AioStreamResponse(
                "http://93.184.216.34/stream",
                status=404,
            ), ClosableByteIterator()
        finally:
            raise aiohttp.ClientConnectionError("response cleanup failed")

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    if stream_kind == "bytes":
        stream = hc.astream_bytes(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        )
    else:
        stream = hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
        )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 404$"):
        await stream.__anext__()

    assert calls == 1


@requires_aiohttp
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["bytes", "sse"])
async def test_aiohttp_invalid_terminal_status_is_sanitized_and_detached(
    monkeypatch,
    stream_kind,
):
    from tldw_Server_API.app.core import http_client as hc

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        yield AioStreamResponse(
            "http://93.184.216.34/stream",
            status=700,
        ), ClosableByteIterator()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    if stream_kind == "bytes":
        stream = hc.astream_bytes(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=1),
        )
    else:
        stream = hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=1),
        )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 700$") as exc_info:
        await stream.__anext__()

    assert exc_info.value.status_code is None
    assert getattr(exc_info.value, "request", None) is None
    assert getattr(exc_info.value, "response", None) is None
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_byte_terminal_status_does_not_read_error_body(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    read_calls = 0

    class GuardedReadResponse(AioStreamResponse):
        async def read(self) -> bytes:
            nonlocal read_calls
            read_calls += 1
            raise RuntimeError("error body unavailable")

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        yield GuardedReadResponse("http://93.184.216.34/stream", status=404), ClosableByteIterator()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_bytes(
        method="GET",
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
    )

    with pytest.raises(hc.NetworkError, match=r"^HTTP 404$"):
        await stream.__anext__()

    assert calls == 1
    assert read_calls == 0


@requires_httpx
@pytest.mark.asyncio
async def test_sensitive_httpx_sse_dns_error_is_not_retried_or_leaked(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        raise socket.gaierror(socket.EAI_NONAME, "secret.example.invalid")
        yield  # pragma: no cover

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
        sensitive_observability=True,
    )

    with pytest.raises(hc.NetworkError) as exc_info:
        await stream.__anext__()

    assert calls == 1
    assert str(exc_info.value) == "gaierror"
    assert "secret.example.invalid" not in str(exc_info.value)


@requires_aiohttp
@pytest.mark.asyncio
async def test_sensitive_aiohttp_sse_dns_error_is_not_retried_or_leaked(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        raise socket.gaierror(socket.EAI_NONAME, "secret.example.invalid")
        yield  # pragma: no cover

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    stream = hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=3),
        sensitive_observability=True,
    )

    with pytest.raises(hc.NetworkError) as exc_info:
        await stream.__anext__()

    assert calls == 1
    assert str(exc_info.value) == "gaierror"
    assert "secret.example.invalid" not in str(exc_info.value)


@requires_aiohttp
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["bytes", "sse"])
async def test_aiohttp_client_error_retries_before_first_body(monkeypatch, stream_kind):
    import aiohttp

    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            if calls == 1:
                raise aiohttp.ClientPayloadError("payload failed at secret.example")
            if stream_kind == "sse":
                yield b"data: ok\n\n"
            else:
                yield b"ok"

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    retry = hc.RetryPolicy(attempts=2, backoff_base_ms=1, backoff_cap_s=1)

    if stream_kind == "sse":
        output = [
            event.data
            async for event in hc.astream_sse(
                url="http://93.184.216.34/stream",
                client=object(),
                retry=retry,
                sensitive_observability=True,
            )
        ]
    else:
        output = [
            chunk
            async for chunk in hc.astream_bytes(
                method="GET",
                url="http://93.184.216.34/stream",
                client=object(),
                retry=retry,
                sensitive_observability=True,
            )
        ]

    expected_output = ["ok"] if stream_kind == "sse" else [b"ok"]
    assert calls == 2
    assert output == expected_output


@requires_aiohttp
@pytest.mark.asyncio
@pytest.mark.parametrize("stream_kind", ["bytes", "sse"])
async def test_aiohttp_client_error_after_body_is_not_retried_or_leaked(
    monkeypatch,
    stream_kind,
):
    import aiohttp

    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            if stream_kind == "sse":
                yield b"data: partial"
            else:
                yield b"first"
            raise aiohttp.ClientPayloadError("payload failed at secret.example")

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    retry = hc.RetryPolicy(attempts=3, backoff_base_ms=1, backoff_cap_s=1)
    output = []

    with pytest.raises(hc.NetworkError) as exc_info:
        if stream_kind == "sse":
            async for event in hc.astream_sse(
                url="http://93.184.216.34/stream",
                client=object(),
                retry=retry,
                sensitive_observability=True,
            ):
                output.append(event.data)
        else:
            async for chunk in hc.astream_bytes(
                method="GET",
                url="http://93.184.216.34/stream",
                client=object(),
                retry=retry,
                sensitive_observability=True,
            ):
                output.append(chunk)

    expected_output = [] if stream_kind == "sse" else [b"first"]
    assert calls == 1
    assert output == expected_output
    assert str(exc_info.value) == "ClientPayloadError"
    assert "secret.example" not in str(exc_info.value)


@requires_aiohttp
@pytest.mark.asyncio
async def test_afetch_aiohttp_retries_client_error(monkeypatch):
    import aiohttp

    from tldw_Server_API.app.core import http_client as hc

    calls = 0

    async def fake_aiohttp_request_io(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise aiohttp.ServerDisconnectedError("secret.example disconnected")
        return SimpleNamespace(
            status_code=200,
            headers={},
            url=kwargs["url"],
        )

    async def fake_sleep(_delay):
        return None

    monkeypatch.setattr(hc, "_aiohttp_request_io", fake_aiohttp_request_io)
    monkeypatch.setattr(hc.asyncio, "sleep", fake_sleep)

    response = await hc._afetch_aiohttp(
        method="GET",
        url="http://93.184.216.34/request",
        client=object(),
        retry=hc.RetryPolicy(attempts=2),
        sensitive_observability=True,
    )

    assert calls == 2
    assert response.status_code == 200


@requires_aiohttp
@pytest.mark.asyncio
async def test_concurrent_aiohttp_client_errors_keep_retry_state_isolated(monkeypatch):
    import aiohttp

    from tldw_Server_API.app.core import http_client as hc

    calls = {"partial": 0, "retry": 0}
    partial_started = asyncio.Event()
    retry_started = asyncio.Event()

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**kwargs):
        stream_name = "partial" if str(kwargs["url"]).endswith("/partial") else "retry"
        calls[stream_name] += 1

        async def iter_bytes():
            if stream_name == "partial":
                partial_started.set()
                await retry_started.wait()
                yield b"data: partial"
                raise aiohttp.ClientPayloadError("partial secret.example")
            if calls[stream_name] == 1:
                retry_started.set()
                await partial_started.wait()
                raise aiohttp.ClientPayloadError("retry secret.example")
            yield b"data: ok\n\n"

        yield AioStreamResponse(str(kwargs["url"])), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    retry = hc.RetryPolicy(attempts=3, backoff_base_ms=1, backoff_cap_s=1)

    async def consume_partial():
        events = []
        with pytest.raises(hc.NetworkError) as exc_info:
            async for event in hc.astream_sse(
                url="http://93.184.216.34/partial",
                client=object(),
                retry=retry,
                sensitive_observability=True,
            ):
                events.append(event.data)
        return events, str(exc_info.value)

    async def consume_retry():
        return [
            event.data
            async for event in hc.astream_sse(
                url="http://93.184.216.34/retry",
                client=object(),
                retry=retry,
                sensitive_observability=True,
            )
        ]

    (partial_events, partial_error), retry_events = await asyncio.gather(
        consume_partial(),
        consume_retry(),
    )

    assert partial_events == []
    assert partial_error == "ClientPayloadError"
    assert "secret.example" not in partial_error
    assert retry_events == ["ok"]
    assert calls == {"partial": 1, "retry": 2}


@requires_httpx
@pytest.mark.asyncio
async def test_concurrent_sse_retry_state_is_isolated_per_stream(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {"partial": 0, "retry": 0}

    @asynccontextmanager
    async def fake_httpx_stream_io(**kwargs):
        stream_name = "partial" if str(kwargs["url"]).endswith("/partial") else "retry"
        calls[stream_name] += 1
        if stream_name == "retry" and calls[stream_name] == 1:
            raise hc.NetworkError("connect failed")

        async def iter_bytes():
            if stream_name == "partial":
                yield b"data: partial"
                raise hc.NetworkError("stream dropped")
            yield b"data: ok\n\n"

        yield StreamResponse(str(kwargs["url"])), iter_bytes()

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    retry = hc.RetryPolicy(attempts=3, backoff_base_ms=1, backoff_cap_s=1)

    async def consume_partial():
        events = []
        with pytest.raises(hc.NetworkError, match=r"^stream dropped$"):
            async for event in hc.astream_sse(
                url="http://93.184.216.34/partial",
                client=object(),
                retry=retry,
            ):
                events.append(event.data)
        return events

    async def consume_retry():
        return [
            event.data
            async for event in hc.astream_sse(
                url="http://93.184.216.34/retry",
                client=object(),
                retry=retry,
            )
        ]

    partial_events, retry_events = await asyncio.gather(consume_partial(), consume_retry())

    assert partial_events == []
    assert retry_events == ["ok"]
    assert calls == {"partial": 1, "retry": 2}


@requires_httpx
@pytest.mark.asyncio
async def test_stream_bytes_retries_http_503_before_first_chunk_httpx(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

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
    import httpx

    from tldw_Server_API.app.core import http_client as hc

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


@requires_aiohttp
@pytest.mark.asyncio
async def test_sse_retries_on_timeout_aiohttp(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {"n": 0}

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:

            async def iter_bytes():
                await asyncio.sleep(0.05)
                yield b"data: timeout\n\n"

        else:

            async def iter_bytes():
                yield b"data: ok\n\n"

        yield AioStreamResponse("http://93.184.216.34/stream"), iter_bytes()

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)

    timeout = SimpleNamespace(connect=0.01, read=0.01)
    events = []
    async for event in hc.astream_sse(
        url="http://93.184.216.34/stream",
        client=object(),
        retry=hc.RetryPolicy(attempts=2),
        timeout=timeout,
    ):
        events.append(event)
        break

    assert calls["n"] == 2
    assert events[0].data == "ok"


@requires_httpx
@pytest.mark.asyncio
async def test_httpx_sse_honors_retry_after(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    delays = []

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            if calls == 2:
                yield b"data: ok\n\n"

        yield StreamResponse(
            "http://93.184.216.34/stream",
            status_code=429 if calls == 1 else 200,
            headers={"retry-after": "17"} if calls == 1 else {},
        ), iter_bytes()

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", fake_sleep)

    events = [
        event.data
        async for event in hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2, respect_retry_after=True),
        )
    ]

    assert calls == 2
    assert delays == [17.0]
    assert events == ["ok"]


@requires_httpx
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_httpx_skips_retryable_response(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    statuses: list[int] = []

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "http://93.184.216.34/stream")
        response = httpx.Response(503 if calls == 1 else 200, request=request)

        async def body():
            if calls == 2:
                yield b"ok"

        yield response, body()

    async def no_sleep(_delay: float) -> None:
        return None

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    chunks = [
        chunk
        async for chunk in hc._astream_bytes_httpx(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        )
    ]

    assert chunks == [b"ok"]
    assert statuses == [200]


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_aiohttp_skips_retryable_response(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    statuses: list[int] = []

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def body():
            if calls == 2:
                yield b"ok"

        yield AioStreamResponse(
            "http://93.184.216.34/stream",
            status=503 if calls == 1 else 200,
        ), body()

    async def no_sleep(_delay: float) -> None:
        return None

    async def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    chunks = [
        chunk
        async for chunk in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        )
    ]

    assert chunks == [b"ok"]
    assert statuses == [200]


@requires_httpx
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_httpx_reports_exhausted_retry_once(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    events: list[str] = []
    calls = 0

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal calls
        calls += 1
        request = httpx.Request("GET", "http://93.184.216.34/stream")
        response = httpx.Response(503, request=request)

        async def body():
            events.append("body")
            yield b"error"

        yield response, body()

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        events.append(f"callback:{status}")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    with pytest.raises(hc.NetworkError, match=r"^HTTP 503$"):
        async for _ in hc._astream_bytes_httpx(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        ):
            pass

    assert calls == 2
    assert events == ["callback:503"]


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_aiohttp_reports_exhausted_retry_once(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    events: list[str] = []
    calls = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def body():
            events.append("body")
            yield b"error"

        yield AioStreamResponse("http://93.184.216.34/stream", status=503), body()

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        events.append(f"callback:{status}")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    with pytest.raises(hc.NetworkError, match=r"^HTTP 503$"):
        async for _ in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        ):
            pass

    assert calls == 2
    assert events == ["callback:503"]


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_aiohttp_does_not_retry_non_retryable_status(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    statuses: list[int] = []

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def body():
            yield b"error"

        yield AioStreamResponse("http://93.184.216.34/stream", status=400), body()

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)

    with pytest.raises(hc.NetworkError, match=r"^HTTP 400$"):
        async for _ in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=3),
            on_response=on_response,
        ):
            pass

    assert calls == 1
    assert statuses == [400]


@requires_httpx
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_httpx_reports_redirect_without_following(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    urls: list[str] = []
    statuses: list[int] = []

    @asynccontextmanager
    async def fake_httpx_stream_io(**kwargs):
        request_url = str(kwargs["url"])
        urls.append(request_url)
        request = httpx.Request("GET", request_url)
        response = httpx.Response(302, request=request, headers={"Location": "/final"})

        async def body():
            yield b"redirect"

        yield response, body()

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)

    with pytest.raises(hc.NetworkError, match=r"^HTTP 302$"):
        async for _ in hc._astream_bytes_httpx(
            method="GET",
            url="http://93.184.216.34/start",
            client=object(),
            on_response=on_response,
        ):
            pass

    assert urls == ["http://93.184.216.34/start"]
    assert statuses == [302]


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_aiohttp_reports_redirect_without_following(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    urls: list[str] = []
    statuses: list[int] = []

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**kwargs):
        request_url = str(kwargs["url"])
        urls.append(request_url)

        async def body():
            yield b"redirect"

        yield AioStreamResponse(
            request_url,
            status=302,
            headers={"Location": "/final"},
        ), body()

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)

    chunks = [
        chunk
        async for chunk in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/start",
            client=object(),
            on_response=on_response,
        )
    ]

    assert urls == ["http://93.184.216.34/start"]
    assert chunks == [b"redirect"]
    assert statuses == [302]


@requires_aiohttp
@pytest.mark.asyncio
async def test_aiohttp_sse_honors_retry_after(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    delays = []

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def iter_bytes():
            if calls == 2:
                yield b"data: ok\n\n"

        yield AioStreamResponse(
            "http://93.184.216.34/stream",
            status=429 if calls == 1 else 200,
            headers={"retry-after": "17"} if calls == 1 else {},
        ), iter_bytes()

    async def fake_sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(hc, "_is_aiohttp_client", lambda _client: True)
    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", fake_sleep)

    events = [
        event.data
        async for event in hc.astream_sse(
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2, respect_retry_after=True),
        )
    ]

    assert calls == 2
    assert delays == [17.0]
    assert events == ["ok"]
@requires_httpx
@pytest.mark.asyncio
async def test_stream_committed_response_httpx_does_not_retry_first_byte_failure(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    attempts = 0
    closes = 0
    statuses: list[int] = []
    error = hc.NetworkError("first byte failed")

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal attempts, closes
        attempts += 1
        request = httpx.Request("GET", "http://93.184.216.34/stream")
        response = httpx.Response(200, request=request)

        async def body():
            raise error
            yield b""  # pragma: no cover

        try:
            yield response, body()
        finally:
            closes += 1

    async def no_sleep(_delay: float) -> None:
        return None

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    with pytest.raises(hc.NetworkError) as exc_info:
        async for _ in hc._astream_bytes_httpx(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        ):
            pass

    assert exc_info.value is error
    assert attempts == 1
    assert statuses == [200]
    assert closes == 1


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_committed_response_aiohttp_does_not_retry_first_byte_failure(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    attempts = 0
    closes = 0
    statuses: list[int] = []
    error = hc.NetworkError("first byte failed")

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal attempts, closes
        attempts += 1

        async def body():
            raise error
            yield b""  # pragma: no cover

        try:
            yield AioStreamResponse("http://93.184.216.34/stream"), body()
        finally:
            closes += 1

    async def no_sleep(_delay: float) -> None:
        return None

    async def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    with pytest.raises(hc.NetworkError) as exc_info:
        async for _ in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        ):
            pass

    assert exc_info.value is error
    assert attempts == 1
    assert statuses == [200]
    assert closes == 1


@requires_httpx
@pytest.mark.asyncio
async def test_stream_uncommitted_response_httpx_retries_first_byte_failure(monkeypatch):
    import httpx

    from tldw_Server_API.app.core import http_client as hc

    attempts = 0
    closes = 0

    @asynccontextmanager
    async def fake_httpx_stream_io(**_kwargs):
        nonlocal attempts, closes
        attempts += 1
        current_attempt = attempts
        request = httpx.Request("GET", "http://93.184.216.34/stream")
        response = httpx.Response(200, request=request)

        async def body():
            if current_attempt == 1:
                raise hc.NetworkError("first byte failed")
            yield b"ok"

        try:
            yield response, body()
        finally:
            closes += 1

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(hc, "_httpx_stream_io", fake_httpx_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    chunks = [
        chunk
        async for chunk in hc._astream_bytes_httpx(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
        )
    ]

    assert chunks == [b"ok"]
    assert attempts == 2
    assert closes == 2


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_uncommitted_response_aiohttp_retries_first_byte_failure(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    attempts = 0
    closes = 0

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal attempts, closes
        attempts += 1
        current_attempt = attempts

        async def body():
            if current_attempt == 1:
                raise hc.NetworkError("first byte failed")
            yield b"ok"

        try:
            yield AioStreamResponse("http://93.184.216.34/stream"), body()
        finally:
            closes += 1

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    chunks = [
        chunk
        async for chunk in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
        )
    ]

    assert chunks == [b"ok"]
    assert attempts == 2
    assert closes == 2
