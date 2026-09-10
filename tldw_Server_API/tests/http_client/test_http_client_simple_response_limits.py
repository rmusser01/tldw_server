from __future__ import annotations

from types import SimpleNamespace

import pytest

import tldw_Server_API.app.core.http_client as hc

URL = "https://example.com/article"


class _StreamingResponse:
    def __init__(
        self,
        url: str,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.url = url
        self.status_code = status_code
        self.headers = headers or {}
        self.encoding = "utf-8"
        self._chunks = chunks
        self.closed = False
        self.callback_results: list[object] = []

    def iter_raw(self):
        yield from self._chunks

    def iter_content(self):
        yield from self._chunks

    def close(self) -> None:
        self.closed = True


class _HTTPXStreamContext:
    def __init__(self, response: _StreamingResponse) -> None:
        self.response = response

    def __enter__(self) -> _StreamingResponse:
        return self.response

    def __exit__(self, *_args: object) -> None:
        self.response.close()


class _StreamingHTTPXClient:
    instances: list[_StreamingHTTPXClient] = []
    responses: list[_StreamingResponse] = []

    def __init__(self, **_kwargs: object) -> None:
        self.stream_calls: list[dict[str, object]] = []
        self.request_called = False
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self) -> _StreamingHTTPXClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def stream(self, method: str, url: str, **kwargs: object) -> _HTTPXStreamContext:
        self.stream_calls.append({"method": method, "url": url, **kwargs})
        return _HTTPXStreamContext(self.__class__.responses.pop(0))

    def request(self, *_args: object, **_kwargs: object) -> object:
        self.request_called = True
        raise AssertionError("bounded fetch must not dispatch through request()")


class _StreamingCurlSession:
    instances: list[_StreamingCurlSession] = []
    responses: list[_StreamingResponse] = []

    def __init__(self, impersonate: str | None = None) -> None:
        self.impersonate = impersonate
        self.get_calls: list[dict[str, object]] = []
        self.closed = False
        self.__class__.instances.append(self)

    def __enter__(self) -> _StreamingCurlSession:
        return self

    def __exit__(self, *_args: object) -> None:
        self.closed = True

    def get(self, url: str, **kwargs: object) -> _StreamingResponse:
        self.get_calls.append({"url": url, **kwargs})
        response = self.__class__.responses.pop(0)
        content_callback = kwargs.get("content_callback")
        if callable(content_callback):
            for chunk in response._chunks:
                response.callback_results.append(content_callback(chunk))
        return response


@pytest.fixture(autouse=True)
def _allow_simple_fetch_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hc, "_is_url_allowed", lambda _url: True)
    monkeypatch.setattr(hc, "_validate_proxies_or_raise", lambda _proxies: None)


@pytest.fixture
def httpx_streaming_backend(monkeypatch: pytest.MonkeyPatch) -> type[_StreamingHTTPXClient]:
    _StreamingHTTPXClient.instances = []
    _StreamingHTTPXClient.responses = []
    monkeypatch.setattr(hc, "_resolve_httpx", lambda: SimpleNamespace(Client=_StreamingHTTPXClient))
    return _StreamingHTTPXClient


@pytest.fixture
def curl_streaming_backend(monkeypatch: pytest.MonkeyPatch) -> type[_StreamingCurlSession]:
    _StreamingCurlSession.instances = []
    _StreamingCurlSession.responses = []
    monkeypatch.setattr(hc, "_resolve_curl_session", lambda: _StreamingCurlSession)
    return _StreamingCurlSession


def test_simple_httpx_fetch_reads_exact_bound_with_identity_encoding(
    httpx_streaming_backend: type[_StreamingHTTPXClient],
) -> None:
    response = _StreamingResponse(URL, [b"ab", b"cde"])
    httpx_streaming_backend.responses = [response]

    result = hc.fetch(URL, backend="httpx", max_response_bytes=5)

    client = httpx_streaming_backend.instances[0]
    assert result["text"] == "abcde"
    assert client.stream_calls[0]["headers"] == {"Accept-Encoding": "identity"}
    assert client.request_called is False
    assert response.closed is True
    assert client.closed is True


@pytest.mark.parametrize("backend", ["httpx", "curl"])
def test_simple_fetch_rejects_response_larger_than_bound_and_closes_response(
    backend: str,
    httpx_streaming_backend: type[_StreamingHTTPXClient],
    curl_streaming_backend: type[_StreamingCurlSession],
) -> None:
    response = _StreamingResponse(URL, [b"abc", b"def"])
    if backend == "httpx":
        httpx_streaming_backend.responses = [response]
    else:
        curl_streaming_backend.responses = [response]

    with pytest.raises(ValueError, match="^Response exceeds max_response_bytes limit$"):
        hc.fetch(URL, backend=backend, max_response_bytes=5)

    assert response.closed is True


def test_simple_httpx_fetch_rejects_non_streaming_backend_before_request_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NonStreamingHTTPXClient:
        instance: _NonStreamingHTTPXClient | None = None

        def __init__(self, **_kwargs: object) -> None:
            self.request_called = False
            self.__class__.instance = self

        def __enter__(self) -> _NonStreamingHTTPXClient:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def request(self, *_args: object, **_kwargs: object) -> object:
            self.request_called = True
            raise AssertionError("bounded fetch must not dispatch through request()")

    monkeypatch.setattr(
        hc,
        "_resolve_httpx",
        lambda: SimpleNamespace(Client=_NonStreamingHTTPXClient),
    )

    with pytest.raises(RuntimeError, match="bounded response streaming"):
        hc.fetch(URL, backend="httpx", max_response_bytes=5)

    assert _NonStreamingHTTPXClient.instance is not None
    assert _NonStreamingHTTPXClient.instance.request_called is False


def test_simple_curl_fetch_reads_bounded_stream_and_closes_session(
    curl_streaming_backend: type[_StreamingCurlSession],
) -> None:
    response = _StreamingResponse(URL, [b"ab", b"cde"])
    curl_streaming_backend.responses = [response]

    result = hc.fetch(URL, backend="curl", max_response_bytes=5)

    session = curl_streaming_backend.instances[0]
    assert result["text"] == "abcde"
    assert session.get_calls[0].get("stream") is not True
    assert callable(session.get_calls[0]["content_callback"])
    assert session.get_calls[0]["accept_encoding"] is None
    assert session.get_calls[0]["headers"] == {"Accept-Encoding": "identity"}
    assert response.callback_results == [2, 3]
    assert response.closed is True
    assert session.closed is True


def test_simple_curl_fetch_follows_compressed_oversized_redirect_before_reading_body(
    curl_streaming_backend: type[_StreamingCurlSession],
) -> None:
    redirect = _StreamingResponse(
        URL,
        [b"redirect body exceeds the bound"],
        status_code=302,
        headers={"Location": "/final", "Content-Encoding": "gzip"},
    )
    final = _StreamingResponse("https://example.com/final", [b"final"])
    curl_streaming_backend.responses = [redirect, final]

    result = hc.fetch(URL, backend="curl", max_response_bytes=5)

    session = curl_streaming_backend.instances[0]
    assert result["url"] == "https://example.com/final"
    assert result["text"] == "final"
    assert [call["url"] for call in session.get_calls] == [URL, "https://example.com/final"]
    assert redirect.closed is True
    assert final.closed is True
    assert session.closed is True


def test_simple_curl_fetch_rejects_compressed_terminal_bounded_response(
    curl_streaming_backend: type[_StreamingCurlSession],
) -> None:
    response = _StreamingResponse(
        URL,
        [b"compressed"],
        headers={"Content-Encoding": "gzip"},
    )
    curl_streaming_backend.responses = [response]

    with pytest.raises(ValueError, match="Compressed responses are not allowed with max_response_bytes"):
        hc.fetch(URL, backend="curl", max_response_bytes=64)

    session = curl_streaming_backend.instances[0]
    assert response.closed is True
    assert session.closed is True


def test_simple_httpx_fetch_rejects_compressed_bounded_response(
    httpx_streaming_backend: type[_StreamingHTTPXClient],
) -> None:
    response = _StreamingResponse(
        URL,
        [b"compressed"],
        headers={"Content-Encoding": "gzip"},
    )
    httpx_streaming_backend.responses = [response]

    with pytest.raises(ValueError, match="Compressed responses are not allowed with max_response_bytes"):
        hc.fetch(URL, backend="httpx", max_response_bytes=64)

    assert response.closed is True


def test_simple_httpx_fetch_streams_final_response_after_redirect(
    httpx_streaming_backend: type[_StreamingHTTPXClient],
) -> None:
    redirect = _StreamingResponse(
        URL,
        [],
        status_code=302,
        headers={"Location": "/final"},
    )
    final = _StreamingResponse("https://example.com/final", [b"final"])
    httpx_streaming_backend.responses = [redirect, final]

    result = hc.fetch(URL, backend="httpx", max_response_bytes=5)

    client = httpx_streaming_backend.instances[0]
    assert result["url"] == "https://example.com/final"
    assert result["text"] == "final"
    assert [call["url"] for call in client.stream_calls] == [URL, "https://example.com/final"]
    assert redirect.closed is True
    assert final.closed is True
