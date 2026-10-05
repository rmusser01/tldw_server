import pytest
from opentelemetry import context as otel_context

pytestmark = pytest.mark.unit


class DummySyncResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    url = "http://example.com"
    text = '{"ok": true}'

    def json(self):
        return {"ok": True}

    def raise_for_status(self) -> None:
        return None

    def close(self) -> None:
        return None


class DummyAsyncResponse:
    status_code = 200
    headers = {"content-type": "application/json"}
    url = "http://example.com"
    text = '{"ok": true}'

    def json(self):
        return {"ok": True}

    def raise_for_status(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def test_httpx_adapter_request_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    calls = {}

    def fake_fetch_httpx_response(**kwargs):
        calls["kwargs"] = kwargs
        calls["sensitive_context"] = hc._SENSITIVE_HTTP_LOG_CONTEXT.get()
        calls["otel_suppressed"] = otel_context.get_value(
            hc._OTEL_HTTP_SUPPRESSION_KEY
        )
        return DummySyncResponse()

    monkeypatch.setattr(hc, "_fetch_httpx_response", fake_fetch_httpx_response)

    adapter = hc.HttpxAdapter()
    scope = ConfiguredEndpointScope.from_url("http://example.com")
    resp = adapter.request(
        method="GET",
        url="http://example.com",
        headers={"x": "y"},
        client=object(),
        configured_endpoint=scope,
        sensitive_observability=True,
    )

    assert isinstance(resp, DummySyncResponse)
    assert calls["kwargs"]["method"] == "GET"
    assert calls["kwargs"]["url"] == "http://example.com"
    assert calls["kwargs"]["headers"] == {"x": "y"}
    assert calls["kwargs"]["configured_endpoint"] is scope
    assert calls["sensitive_context"] is True
    assert calls["otel_suppressed"] is True
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_httpx_adapter_arequest_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    calls = {}

    async def fake_afetch_httpx(**kwargs):
        calls["kwargs"] = kwargs
        calls["sensitive_context"] = hc._SENSITIVE_HTTP_LOG_CONTEXT.get()
        calls["otel_suppressed"] = otel_context.get_value(
            hc._OTEL_HTTP_SUPPRESSION_KEY
        )
        return DummyAsyncResponse()

    monkeypatch.setattr(hc, "_afetch_httpx", fake_afetch_httpx)

    adapter = hc.HttpxAdapter()
    scope = ConfiguredEndpointScope.from_url("http://example.com")
    resp = await adapter.arequest(
        method="POST",
        url="http://example.com",
        json={"k": "v"},
        client=object(),
        configured_endpoint=scope,
        sensitive_observability=True,
    )

    assert isinstance(resp, DummyAsyncResponse)
    assert calls["kwargs"]["method"] == "POST"
    assert calls["kwargs"]["url"] == "http://example.com"
    assert calls["kwargs"]["json"] == {"k": "v"}
    assert calls["kwargs"]["configured_endpoint"] is scope
    assert calls["sensitive_context"] is True
    assert calls["otel_suppressed"] is True
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_httpx_adapter_stream_bytes_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    context_states = []

    async def fake_stream_bytes_httpx(**_kwargs):
        context_states.append(
            (
                hc._SENSITIVE_HTTP_LOG_CONTEXT.get(),
                otel_context.get_value(hc._OTEL_HTTP_SUPPRESSION_KEY),
            )
        )
        yield b"one"
        context_states.append(
            (
                hc._SENSITIVE_HTTP_LOG_CONTEXT.get(),
                otel_context.get_value(hc._OTEL_HTTP_SUPPRESSION_KEY),
            )
        )
        yield b"two"

    monkeypatch.setattr(hc, "_astream_bytes_httpx", fake_stream_bytes_httpx)

    adapter = hc.HttpxAdapter()
    chunks = [
        chunk
        async for chunk in adapter.stream_bytes(
            method="GET",
            url="http://example.com",
            client=object(),
            sensitive_observability=True,
        )
    ]

    assert chunks == [b"one", b"two"]
    assert context_states == [(True, True), (True, True)]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_httpx_adapter_stream_bytes_closes_delegate_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    finalized = []

    async def fake_stream_bytes_httpx(**_kwargs):
        try:
            yield b"one"
            yield b"two"
        finally:
            finalized.append("closed")

    monkeypatch.setattr(hc, "_astream_bytes_httpx", fake_stream_bytes_httpx)

    stream = hc.HttpxAdapter().stream_bytes(
        method="GET",
        url="http://example.com",
        client=object(),
        sensitive_observability=True,
    )
    assert await stream.__anext__() == b"one"
    await stream.aclose()

    assert finalized == ["closed"]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_httpx_adapter_stream_response_callback_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {}

    async def fake_stream_bytes_httpx(**kwargs):
        calls["kwargs"] = kwargs
        yield b"one"

    def on_response(_status, _headers):
        return None

    monkeypatch.setattr(hc, "_astream_bytes_httpx", fake_stream_bytes_httpx)

    chunks = [
        chunk
        async for chunk in hc.HttpxAdapter().stream_bytes(
            method="GET",
            url="http://example.com",
            client=object(),
            on_response=on_response,
        )
    ]

    assert chunks == [b"one"]
    assert calls["kwargs"]["on_response"] is on_response


@pytest.mark.asyncio
async def test_httpx_adapter_stream_sse_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    context_states = []

    async def fake_stream_sse_httpx(**kwargs):
        context_states.append(
            (
                kwargs["sensitive_observability"],
                hc._SENSITIVE_HTTP_LOG_CONTEXT.get(),
                otel_context.get_value(hc._OTEL_HTTP_SUPPRESSION_KEY),
            )
        )
        yield hc.SSEEvent(event="message", data="hello")

    monkeypatch.setattr(hc, "_astream_sse_httpx", fake_stream_sse_httpx)

    adapter = hc.HttpxAdapter()
    events = [
        ev
        async for ev in adapter.stream_sse(
            url="http://example.com/stream",
            client=object(),
            sensitive_observability=True,
        )
    ]

    assert len(events) == 1
    assert events[0].data == "hello"
    assert context_states == [(True, True, True)]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_httpx_adapter_stream_sse_closes_delegate_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    finalized = []

    async def fake_stream_sse_httpx(**_kwargs):
        try:
            yield hc.SSEEvent(data="one")
            yield hc.SSEEvent(data="two")
        finally:
            finalized.append("closed")

    monkeypatch.setattr(hc, "_astream_sse_httpx", fake_stream_sse_httpx)

    stream = hc.HttpxAdapter().stream_sse(
        url="http://example.com/stream",
        client=object(),
        sensitive_observability=True,
    )
    assert (await stream.__anext__()).data == "one"
    await stream.aclose()

    assert finalized == ["closed"]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


def test_aiohttp_adapter_request_not_supported():
    from tldw_Server_API.app.core import http_client as hc

    adapter = hc.AiohttpAdapter()
    with pytest.raises(NotImplementedError):
        adapter.request(method="GET", url="http://example.com")


@pytest.mark.asyncio
async def test_aiohttp_adapter_arequest_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    calls = {}

    async def fake_afetch_aiohttp(**kwargs):
        calls["kwargs"] = kwargs
        calls["sensitive_context"] = hc._SENSITIVE_HTTP_LOG_CONTEXT.get()
        calls["otel_suppressed"] = otel_context.get_value(
            hc._OTEL_HTTP_SUPPRESSION_KEY
        )
        return DummyAsyncResponse()

    monkeypatch.setattr(hc, "_afetch_aiohttp", fake_afetch_aiohttp)

    adapter = hc.AiohttpAdapter()
    scope = ConfiguredEndpointScope.from_url("http://example.com")
    resp = await adapter.arequest(
        method="GET",
        url="http://example.com",
        client=object(),
        configured_endpoint=scope,
        sensitive_observability=True,
    )

    assert isinstance(resp, DummyAsyncResponse)
    assert calls["kwargs"]["url"] == "http://example.com"
    assert calls["kwargs"]["configured_endpoint"] is scope
    assert calls["sensitive_context"] is True
    assert calls["otel_suppressed"] is True
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


def test_byte_streams_accept_scope_while_sse_signatures_remain_unscoped():
    import inspect

    from tldw_Server_API.app.core import http_client as hc

    assert "configured_endpoint" in inspect.signature(hc.astream_bytes).parameters
    assert "configured_endpoint" not in inspect.signature(hc.astream_sse).parameters
    assert "configured_endpoint" in inspect.signature(hc.HttpxAdapter.stream_bytes).parameters
    assert "configured_endpoint" in inspect.signature(hc.AiohttpAdapter.stream_bytes).parameters
    assert "configured_endpoint" not in inspect.signature(hc.HttpxAdapter.stream_sse).parameters
    assert "sensitive_observability" in inspect.signature(hc.astream_sse).parameters
    assert "sensitive_observability" in inspect.signature(hc.HttpxAdapter.stream_sse).parameters
    assert "sensitive_observability" in inspect.signature(hc.AiohttpAdapter.stream_sse).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
async def test_byte_stream_adapters_forward_configured_endpoint(monkeypatch, backend):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("https://example.com")
    calls = []

    async def delegate(**kwargs):
        calls.append(kwargs)
        yield b"ok"

    monkeypatch.setattr(hc, f"_astream_bytes_{backend}", delegate)
    adapter = hc.HttpxAdapter() if backend == "httpx" else hc.AiohttpAdapter()
    result = [
        chunk
        async for chunk in adapter.stream_bytes(
            method="GET", url="https://example.com/data", client=object(), configured_endpoint=scope
        )
    ]

    assert result == [b"ok"]
    assert calls[0]["configured_endpoint"] is scope


@pytest.mark.asyncio
@pytest.mark.parametrize("scoped", [True, False])
async def test_public_byte_stream_preserves_optional_scope_without_changing_legacy_kwargs(monkeypatch, scoped):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    scope = ConfiguredEndpointScope.from_url("https://example.com") if scoped else None
    calls = []

    class Adapter:
        async def stream_bytes(self, **kwargs):
            calls.append(kwargs)
            yield b"ok"

    monkeypatch.setattr(hc, "_get_transport_adapter", lambda _name: Adapter())
    result = [
        chunk
        async for chunk in hc.astream_bytes(method="GET", url="https://example.com/data", configured_endpoint=scope)
    ]

    assert result == [b"ok"]
    if scoped:
        assert calls[0]["configured_endpoint"] is scope
    else:
        assert "configured_endpoint" not in calls[0]


@pytest.mark.asyncio
async def test_public_byte_stream_budget_yields_only_limit_plus_one_and_closes_delegate(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.exceptions import NetworkError

    closed = []
    visited = []

    class Adapter:
        async def stream_bytes(self, **kwargs):
            assert "max_response_bytes" not in kwargs
            try:
                for index in range(3):
                    visited.append(index)
                    yield b"x" * 65536
            finally:
                closed.append(True)

    monkeypatch.setattr(hc, "_get_transport_adapter", lambda _name: Adapter())
    sizes = []
    with pytest.raises(NetworkError, match="max_response_bytes"):
        async for chunk in hc.astream_bytes(method="GET", url="https://example.com", max_response_bytes=65538):
            sizes.append(len(chunk))

    assert sum(sizes) == 65539
    assert visited == [0, 1]
    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["httpx", "aiohttp"])
@pytest.mark.parametrize("pinned", [False, True])
async def test_native_byte_stream_checks_scope_on_every_egress_validation(monkeypatch, backend, pinned):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    import httpx

    from tldw_Server_API.app.core import http_client as hc
    from tldw_Server_API.app.core.Security.egress import ConfiguredEndpointScope

    url = "https://example.com/data"
    scope = ConfiguredEndpointScope.from_url(url)
    validations = []
    pinning_validations = []

    def check_pin(_host, _port, _pins, _min_version, **kwargs):
        assert kwargs.get("configured_endpoint") is scope
        pinning_validations.append(scope)

    async def validate(_url, **kwargs):
        assert kwargs["configured_endpoint"] is scope
        validations.append(_url)

    async def chunks():
        yield b"ok"

    @asynccontextmanager
    async def io(**kwargs):
        if backend == "httpx":
            response = httpx.Response(200, request=httpx.Request("GET", url))
        else:
            response = SimpleNamespace(status=200, headers={}, url=url)
        yield response, chunks()

    monkeypatch.setattr(hc, "_avalidate_egress_or_raise", validate)
    monkeypatch.setattr(hc, "_check_cert_pinning", check_pin)
    monkeypatch.setattr(hc, f"_{backend}_stream_io", io)
    stream = getattr(hc, f"_astream_bytes_{backend}")(
        method="GET",
        url=url,
        client=object(),
        configured_endpoint=scope,
        retry=hc.RetryPolicy(attempts=1),
        cert_pinning={"example.com": {"fixed-pin"}} if pinned else None,
    )
    assert [chunk async for chunk in stream] == [b"ok"]
    assert validations == [url, url]
    assert pinning_validations == ([scope] if pinned else [])


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [True, -1, 1.5, "10"])
async def test_byte_stream_rejects_malformed_budget_before_admission(monkeypatch, limit):
    from tldw_Server_API.app.core import http_client as hc

    def forbidden_adapter(_name):
        raise AssertionError("Invalid budget must not select or dispatch a transport")

    monkeypatch.setattr(hc, "_get_transport_adapter", forbidden_adapter)
    with pytest.raises(ValueError, match="non-negative integer"):
        await hc.astream_bytes(method="GET", url="https://example.com", max_response_bytes=limit).__anext__()


@pytest.mark.asyncio
async def test_aiohttp_adapter_stream_bytes_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    context_states = []

    async def fake_stream_bytes_aiohttp(**_kwargs):
        context_states.append(
            (
                hc._SENSITIVE_HTTP_LOG_CONTEXT.get(),
                otel_context.get_value(hc._OTEL_HTTP_SUPPRESSION_KEY),
            )
        )
        yield b"alpha"

    monkeypatch.setattr(hc, "_astream_bytes_aiohttp", fake_stream_bytes_aiohttp)

    adapter = hc.AiohttpAdapter()
    chunks = [
        chunk
        async for chunk in adapter.stream_bytes(
            method="GET",
            url="http://example.com",
            client=object(),
            sensitive_observability=True,
        )
    ]

    assert chunks == [b"alpha"]
    assert context_states == [(True, True)]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_aiohttp_adapter_stream_bytes_closes_delegate_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    finalized = []

    async def fake_stream_bytes_aiohttp(**_kwargs):
        try:
            yield b"alpha"
            yield b"beta"
        finally:
            finalized.append("closed")

    monkeypatch.setattr(hc, "_astream_bytes_aiohttp", fake_stream_bytes_aiohttp)

    stream = hc.AiohttpAdapter().stream_bytes(
        method="GET",
        url="http://example.com",
        client=object(),
        sensitive_observability=True,
    )
    assert await stream.__anext__() == b"alpha"
    await stream.aclose()

    assert finalized == ["closed"]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_aiohttp_adapter_stream_response_callback_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {}

    async def fake_stream_bytes_aiohttp(**kwargs):
        calls["kwargs"] = kwargs
        yield b"alpha"

    async def on_response(_status, _headers):
        return None

    monkeypatch.setattr(hc, "_astream_bytes_aiohttp", fake_stream_bytes_aiohttp)

    chunks = [
        chunk
        async for chunk in hc.AiohttpAdapter().stream_bytes(
            method="GET",
            url="http://example.com",
            client=object(),
            on_response=on_response,
        )
    ]

    assert chunks == [b"alpha"]
    assert calls["kwargs"]["on_response"] is on_response


@pytest.mark.asyncio
async def test_public_stream_response_callback_passes_to_transport_adapter(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {}

    class Adapter:
        async def stream_bytes(self, **kwargs):
            calls["kwargs"] = kwargs
            yield b"audio"

    def on_response(_status, _headers):
        return None

    monkeypatch.setattr(hc, "_get_transport_adapter", lambda _name: Adapter())

    chunks = [
        chunk
        async for chunk in hc.astream_bytes(
            method="GET",
            url="http://example.com",
            client=object(),
            on_response=on_response,
        )
    ]

    assert chunks == [b"audio"]
    assert calls["kwargs"]["on_response"] is on_response


@pytest.mark.asyncio
async def test_aiohttp_adapter_stream_sse_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    context_states = []

    async def fake_stream_sse_aiohttp(**kwargs):
        context_states.append(
            (
                kwargs["sensitive_observability"],
                hc._SENSITIVE_HTTP_LOG_CONTEXT.get(),
                otel_context.get_value(hc._OTEL_HTTP_SUPPRESSION_KEY),
            )
        )
        yield hc.SSEEvent(event="message", data="world")

    monkeypatch.setattr(hc, "_astream_sse_aiohttp", fake_stream_sse_aiohttp)

    adapter = hc.AiohttpAdapter()
    events = [
        ev
        async for ev in adapter.stream_sse(
            url="http://example.com/stream",
            client=object(),
            sensitive_observability=True,
        )
    ]

    assert len(events) == 1
    assert events[0].data == "world"
    assert context_states == [(True, True, True)]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False


@pytest.mark.asyncio
async def test_aiohttp_adapter_stream_sse_closes_delegate_on_early_close(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    finalized = []

    async def fake_stream_sse_aiohttp(**_kwargs):
        try:
            yield hc.SSEEvent(data="one")
            yield hc.SSEEvent(data="two")
        finally:
            finalized.append("closed")

    monkeypatch.setattr(hc, "_astream_sse_aiohttp", fake_stream_sse_aiohttp)

    stream = hc.AiohttpAdapter().stream_sse(
        url="http://example.com/stream",
        client=object(),
        sensitive_observability=True,
    )
    assert (await stream.__anext__()).data == "one"
    await stream.aclose()

    assert finalized == ["closed"]
    assert hc._SENSITIVE_HTTP_LOG_CONTEXT.get() is False
