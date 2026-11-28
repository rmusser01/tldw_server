"""Generic response observers stop accumulation without exposing their cause."""

import httpx
import pytest

from tldw_Server_API.app.core import http_client as hc


def test_chunk_observer_runs_before_append():
    seen = []

    def observe(amount):
        seen.append(amount)
        if len(seen) == 2:
            raise RuntimeError("private observer secret")

    with pytest.raises(hc.ResponseObserverAborted, match="^Response observer aborted$") as caught:
        hc._read_bounded_chunks([b"ab", b"cde", b"never"], 100, on_response_chunk=observe)
    assert seen == [2, 3]
    assert str(caught.value._cause) == "private observer secret"


def test_deferred_curl_abort_does_not_retain_over_budget_chunk():
    def observe(amount):
        if amount == 3:
            raise RuntimeError("private")

    collector = hc._BoundedBodyCollector(100, on_response_chunk=observe)
    assert collector(b"ab") == 2
    assert collector(b"cde") == 0
    assert collector(b"never") == 0
    with pytest.raises(hc.ResponseObserverAborted):
        collector.raise_if_aborted()
    assert collector.body == b"ab"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_transport_observer_private_cause_escapes(asynchronous, monkeypatch):
    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    seen = []
    cause = ValueError("secret")

    def observe(amount):
        seen.append(amount)
        raise cause

    def respond(request):
        return httpx.Response(200, content=b"hello", request=request)

    # MockTransport responses are already read; streaming response streams avoid that shortcut.
    class SyncStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"hello"

    class AsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"hello"

    if asynchronous:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=AsyncStream(), request=r))
        ) as client:
            with pytest.raises(hc.ResponseObserverAborted) as caught:
                await hc._httpx_arequest_io(
                    client=client,
                    method="GET",
                    url="https://example.test",
                    max_response_bytes=20,
                    on_response_chunk=observe,
                )
    else:
        with httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=SyncStream(), request=r))
        ) as client:
            with pytest.raises(hc.ResponseObserverAborted) as caught:
                hc._httpx_request_io(
                    client=client,
                    method="GET",
                    url="https://example.test",
                    max_response_bytes=20,
                    on_response_chunk=observe,
                )
    assert caught.value._cause is cause
    assert seen == [5]


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_response_api_abort_is_not_retried_or_exposed(asynchronous, monkeypatch):
    monkeypatch.setattr(hc, "_validate_egress_or_raise", lambda *_args, **_kwargs: None)
    calls = []
    cause = RuntimeError("private-credential")

    def observe(_amount):
        raise cause

    class SyncStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"one"

    class AsyncStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"one"

    def respond(request):
        calls.append(request)
        return httpx.Response(200, stream=AsyncStream() if asynchronous else SyncStream(), request=request)

    if asynchronous:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(hc.ResponseObserverAborted) as caught:
                await hc.afetch(
                    method="GET",
                    url="https://example.test",
                    client=client,
                    max_response_bytes=10,
                    on_response_chunk=observe,
                    retry=hc.RetryPolicy(attempts=3),
                )
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(hc.ResponseObserverAborted) as caught:
                hc.fetch(
                    method="GET",
                    url="https://example.test",
                    client=client,
                    max_response_bytes=10,
                    on_response_chunk=observe,
                    retry=hc.RetryPolicy(attempts=3),
                )
    assert caught.value._cause is cause
    assert "private-credential" not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize("native_abort", [False, True])
def test_simple_curl_deferred_abort_rethrows_private_cause(native_abort, monkeypatch):
    from types import SimpleNamespace

    cause = RuntimeError("private-secret")
    retained = []

    def observe(amount):
        if amount == 3:
            raise cause

    class Session:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

        def get(self, url, **kwargs):
            callback = kwargs["content_callback"]
            assert callback(b"ab") == 2
            assert callback(b"cde") == 0
            retained.append(callback.body)
            if native_abort:
                raise RuntimeError("native curl abort")
            return SimpleNamespace(status_code=200, headers={}, url=url)

    monkeypatch.setattr(hc, "_is_url_allowed", lambda _url: True)
    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: Session)
    with pytest.raises(hc.ResponseObserverAborted) as caught:
        hc.fetch("https://example.test", backend="curl", max_response_bytes=100, on_response_chunk=observe)
    assert caught.value._cause is cause
    assert retained == [b"ab"]
