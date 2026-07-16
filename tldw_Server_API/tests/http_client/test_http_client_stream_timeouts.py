import asyncio
from collections.abc import Mapping
from contextlib import asynccontextmanager
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

    with pytest.raises(httpx.HTTPStatusError):
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

    class Response(AioStreamResponse):
        async def read(self) -> bytes:
            events.append("read")
            return b"error"

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def body():
            events.append("body")
            yield b"error"

        yield Response("http://93.184.216.34/stream", status=503), body()

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        events.append(f"callback:{status}")

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    with pytest.raises(hc.NetworkError, match="HTTP 503"):
        async for _ in hc._astream_bytes_aiohttp(
            method="GET",
            url="http://93.184.216.34/stream",
            client=object(),
            retry=hc.RetryPolicy(attempts=2),
            on_response=on_response,
        ):
            pass

    assert calls == 2
    assert events == ["callback:503", "read"]


@requires_aiohttp
@pytest.mark.asyncio
async def test_stream_terminal_response_callback_aiohttp_does_not_retry_non_retryable_status(monkeypatch):
    from tldw_Server_API.app.core import http_client as hc

    calls = 0
    statuses: list[int] = []

    class Response(AioStreamResponse):
        async def read(self) -> bytes:
            return b"error"

    @asynccontextmanager
    async def fake_aiohttp_stream_io(**_kwargs):
        nonlocal calls
        calls += 1

        async def body():
            yield b"error"

        yield Response("http://93.184.216.34/stream", status=400), body()

    async def no_sleep(_delay: float) -> None:
        return None

    def on_response(status: int, _headers: Mapping[str, str]) -> None:
        statuses.append(status)

    monkeypatch.setattr(hc, "_aiohttp_stream_io", fake_aiohttp_stream_io)
    monkeypatch.setattr(hc.asyncio, "sleep", no_sleep)

    with pytest.raises(hc.NetworkError, match="HTTP 400"):
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

    with pytest.raises(httpx.HTTPStatusError):
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
