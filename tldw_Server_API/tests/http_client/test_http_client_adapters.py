import pytest


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

    calls = {}

    def fake_fetch_httpx_response(**kwargs):
        calls["kwargs"] = kwargs
        return DummySyncResponse()

    monkeypatch.setattr(hc, "_fetch_httpx_response", fake_fetch_httpx_response)

    adapter = hc.HttpxAdapter()
    resp = adapter.request(method="GET", url="http://example.com", headers={"x": "y"}, client=object())

    assert isinstance(resp, DummySyncResponse)
    assert calls["kwargs"]["method"] == "GET"
    assert calls["kwargs"]["url"] == "http://example.com"
    assert calls["kwargs"]["headers"] == {"x": "y"}


@pytest.mark.asyncio
async def test_httpx_adapter_arequest_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {}

    async def fake_afetch_httpx(**kwargs):
        calls["kwargs"] = kwargs
        return DummyAsyncResponse()

    monkeypatch.setattr(hc, "_afetch_httpx", fake_afetch_httpx)

    adapter = hc.HttpxAdapter()
    resp = await adapter.arequest(method="POST", url="http://example.com", json={"k": "v"}, client=object())

    assert isinstance(resp, DummyAsyncResponse)
    assert calls["kwargs"]["method"] == "POST"
    assert calls["kwargs"]["url"] == "http://example.com"
    assert calls["kwargs"]["json"] == {"k": "v"}


@pytest.mark.asyncio
async def test_httpx_adapter_stream_bytes_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    async def fake_stream_bytes_httpx(**_kwargs):
        yield b"one"
        yield b"two"

    monkeypatch.setattr(hc, "_astream_bytes_httpx", fake_stream_bytes_httpx)

    adapter = hc.HttpxAdapter()
    chunks = [chunk async for chunk in adapter.stream_bytes(method="GET", url="http://example.com", client=object())]

    assert chunks == [b"one", b"two"]


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

    async def fake_stream_sse_httpx(**_kwargs):
        yield hc.SSEEvent(event="message", data="hello")

    monkeypatch.setattr(hc, "_astream_sse_httpx", fake_stream_sse_httpx)

    adapter = hc.HttpxAdapter()
    events = [ev async for ev in adapter.stream_sse(url="http://example.com/stream", client=object())]

    assert len(events) == 1
    assert events[0].data == "hello"


def test_aiohttp_adapter_request_not_supported():
    from tldw_Server_API.app.core import http_client as hc

    adapter = hc.AiohttpAdapter()
    with pytest.raises(NotImplementedError):
        adapter.request(method="GET", url="http://example.com")


@pytest.mark.asyncio
async def test_aiohttp_adapter_arequest_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = {}

    async def fake_afetch_aiohttp(**kwargs):
        calls["kwargs"] = kwargs
        return DummyAsyncResponse()

    monkeypatch.setattr(hc, "_afetch_aiohttp", fake_afetch_aiohttp)

    adapter = hc.AiohttpAdapter()
    resp = await adapter.arequest(method="GET", url="http://example.com", client=object())

    assert isinstance(resp, DummyAsyncResponse)
    assert calls["kwargs"]["url"] == "http://example.com"


@pytest.mark.asyncio
async def test_aiohttp_adapter_stream_bytes_passes_through(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    async def fake_stream_bytes_aiohttp(**_kwargs):
        yield b"alpha"

    monkeypatch.setattr(hc, "_astream_bytes_aiohttp", fake_stream_bytes_aiohttp)

    adapter = hc.AiohttpAdapter()
    chunks = [chunk async for chunk in adapter.stream_bytes(method="GET", url="http://example.com", client=object())]

    assert chunks == [b"alpha"]


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

    async def fake_stream_sse_aiohttp(**_kwargs):
        yield hc.SSEEvent(event="message", data="world")

    monkeypatch.setattr(hc, "_astream_sse_aiohttp", fake_stream_sse_aiohttp)

    adapter = hc.AiohttpAdapter()
    events = [ev async for ev in adapter.stream_sse(url="http://example.com/stream", client=object())]

    assert len(events) == 1
    assert events[0].data == "world"
