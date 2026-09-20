from __future__ import annotations

import asyncio
import traceback

import httpx
import pytest

from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSNetworkError,
    TTSProviderError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSValidationError,
)


def _assert_exception_graph_is_sanitized(exc: Exception, *sentinels: str) -> None:
    assert exc.__cause__ is None
    assert exc.__context__ is None

    seen: set[int] = set()
    pending = [exc]
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((str(current), repr(current), repr(vars(current))))

        request = getattr(current, "request", None)
        if request is not None:
            rendered.extend(
                (
                    str(request.url),
                    repr(getattr(request.url, "raw_userinfo", b"")),
                    repr(getattr(request.headers, "raw", request.headers)),
                )
            )

        response = getattr(current, "response", None)
        if response is not None:
            rendered.extend((repr(response.headers), repr(response.content)))
            response_request = getattr(response, "request", None)
            if response_request is not None:
                rendered.extend(
                    (
                        str(response_request.url),
                        repr(getattr(response_request.url, "raw_userinfo", b"")),
                        repr(getattr(response_request.headers, "raw", response_request.headers)),
                    )
                )

        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)

    rendered.append("".join(traceback.format_exception(exc)))
    graph_text = "\n".join(rendered)
    for sentinel in sentinels:
        assert sentinel not in graph_text


def _assert_traceback_local_omits(
    exc: Exception,
    *,
    module_name: str,
    local_name: str,
    sentinel: str,
) -> None:
    traceback_frame = exc.__traceback__
    while traceback_frame is not None:
        frame = traceback_frame.tb_frame
        if frame.f_globals.get("__name__") == module_name:
            assert frame.f_locals.get(local_name) != sentinel
        traceback_frame = traceback_frame.tb_next


class _FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        content: bytes = b"audio",
        text: str = "ok",
        json_data=None,
        headers=None,
    ):
        self.status_code = status_code
        self.content = content
        self.text = text
        self._json_data = json_data
        self.headers = headers or {}

    def json(self):
        return self._json_data


@pytest.mark.asyncio
async def test_backend_builds_tts_payload_from_request(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend(
        {"base_url": "http://fish.local", "api_key": "secret", "timeout": 30}
    )

    captured: dict[str, object] = {}

    async def fake_fetch(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    audio = await backend.synthesize(
        text="hello",
        response_format="wav",
        streaming=False,
        reference_id="tldw_u1_vabc",
        extra_params={"chunk_length": 200, "normalize": True},
    )

    assert audio == b"audio"
    assert captured["method"] == "POST"
    assert captured["url"] == "http://fish.local/v1/tts"
    assert captured["json"] == {
        "text": "hello",
        "format": "wav",
        "streaming": False,
        "reference_id": "tldw_u1_vabc",
        "chunk_length": 200,
        "normalize": True,
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 30


@pytest.mark.asyncio
async def test_backend_uses_streaming_helper_for_streaming_requests(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret", "timeout": 12})
    captured: dict[str, object] = {}

    async def fake_stream(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        yield b"chunk1"
        yield b"chunk2"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.astream_bytes",
        fake_stream,
    )

    stream = await backend.synthesize(
        text="hello",
        response_format="wav",
        streaming=True,
        reference_id=None,
        extra_params=None,
    )

    chunks = [chunk async for chunk in stream]
    assert chunks == [b"chunk1", b"chunk2"]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://fish.local/v1/tts"
    assert captured["json"] == {
        "text": "hello",
        "format": "wav",
        "streaming": True,
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 12


@pytest.mark.asyncio
async def test_backend_maps_401_to_authentication_error(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        return _FakeResponse(status_code=401, text="invalid token")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    with pytest.raises(TTSAuthenticationError):
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )


@pytest.mark.asyncio
async def test_backend_maps_non_integer_retry_after_to_rate_limit_error(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        return _FakeResponse(
            status_code=429,
            text="rate limited",
            headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    with pytest.raises(TTSRateLimitError) as exc_info:
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )

    assert exc_info.value.retry_after is None


def test_backend_response_json_logs_parse_failures(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends import fish_s2_native_http
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    class _LoggerStub:
        def __init__(self):
            self.debug_calls = []

        def debug(self, *args, **kwargs):
            self.debug_calls.append((args, kwargs))

    class _InvalidJsonResponse:
        def json(self):
            raise ValueError("not json")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(fish_s2_native_http, "logger", logger_stub)

    assert FishS2NativeHttpBackend._response_json(_InvalidJsonResponse()) is None
    assert logger_stub.debug_calls


@pytest.mark.asyncio
async def test_backend_error_log_omits_upstream_body(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends import fish_s2_native_http
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    class _LoggerStub:
        def __init__(self):
            self.error_calls = []

        def error(self, *args, **kwargs):
            self.error_calls.append((args, kwargs))

    logger_stub = _LoggerStub()
    monkeypatch.setattr(fish_s2_native_http, "logger", logger_stub)

    async def fake_fetch(**_kwargs):
        return _FakeResponse(status_code=500, text="token leaked /private/fish-body.txt")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local"})

    with pytest.raises(TTSProviderError):
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )

    assert logger_stub.error_calls
    rendered_calls = repr(logger_stub.error_calls)
    assert "token leaked" not in rendered_calls
    assert "/private/fish-body.txt" not in rendered_calls


@pytest.mark.asyncio
async def test_backend_add_reference_posts_expected_payload(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret"})

    captured = {}

    async def fake_fetch(*, method, url, json=None, data=None, files=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(json_data={"reference_id": "tldw_u1_voice-1"})

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    result = await backend.add_reference(
        reference_id="tldw_u1_voice-1",
        audio_b64="QUJD",
        reference_text="hello there",
    )

    assert result == {"reference_id": "tldw_u1_voice-1"}
    assert captured["method"] == "POST"
    assert captured["url"] == "http://fish.local/v1/references/add"
    assert captured["json"] is None
    assert captured["data"] == {
        "id": "tldw_u1_voice-1",
        "text": "hello there",
    }
    assert captured["files"] == {
        "audio": ("reference.wav", b"ABC", "audio/wav"),
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == backend.timeout


@pytest.mark.asyncio
async def test_backend_rejects_invalid_reference_audio_without_retaining_input():
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local"})
    sentinel = "fish-native-invalid-base64-secret"

    with pytest.raises(TTSValidationError) as exc_info:
        await backend.add_reference(
            reference_id="voice-1",
            audio_b64=sentinel,
            reference_text="hello there",
        )

    _assert_exception_graph_is_sanitized(exc_info.value, sentinel)
    _assert_traceback_local_omits(
        exc_info.value,
        module_name="tldw_Server_API.app.core.TTS.backends.fish_s2_native_http",
        local_name="audio_b64",
        sentinel=sentinel,
    )


@pytest.mark.asyncio
async def test_backend_delete_reference_posts_expected_payload(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import FishS2NativeHttpBackend

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local", "api_key": "secret", "timeout": 12})

    captured = {}

    async def fake_fetch(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(status_code=204)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    deleted = await backend.delete_reference(reference_id="tldw_u1_voice-1")

    assert deleted is True
    assert captured["method"] == "DELETE"
    assert captured["url"] == "http://fish.local/v1/references/delete"
    assert captured["json"] == {"reference_id": "tldw_u1_voice-1"}
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 12


@pytest.mark.asyncio
async def test_native_backend_does_not_stringify_transport_errors(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    sentinel = "fish-native-hostile-str-secret"
    stringify_calls = 0

    class HostileTransportError(Exception):
        def __str__(self):
            nonlocal stringify_calls
            stringify_calls += 1
            raise RuntimeError(sentinel)

    async def fake_fetch(**_kwargs):
        raise HostileTransportError()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )
    backend = FishS2NativeHttpBackend(
        {"base_url": "https://fish-native.invalid", "api_key": "test-key"}
    )

    with pytest.raises(TTSNetworkError) as exc_info:
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )

    assert stringify_calls == 0
    _assert_exception_graph_is_sanitized(exc_info.value, sentinel)


@pytest.mark.asyncio
async def test_native_backend_preserves_core_transport_timeout_mapping(monkeypatch):
    from tldw_Server_API.app.core.exceptions import NetworkError as CoreNetworkError
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    async def fake_fetch(**_kwargs):
        raise CoreNetworkError("ReadTimeout")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )
    backend = FishS2NativeHttpBackend({"base_url": "https://fish-native.invalid"})

    with pytest.raises(TTSTimeoutError):
        await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["aclose", "cancel"])
async def test_native_backend_stream_deterministically_closes_inner_iterator(
    monkeypatch,
    termination,
):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    backend = FishS2NativeHttpBackend(
        {"base_url": "https://fish-native.invalid", "api_key": "test-key"}
    )
    inner_closed = asyncio.Event()
    inner_streams = []

    async def tracked_inner_stream():
        try:
            yield b"chunk"
            await asyncio.Event().wait()
        finally:
            inner_closed.set()

    def fake_astream_bytes(**_kwargs):
        stream = tracked_inner_stream()
        inner_streams.append(stream)
        return stream

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.astream_bytes",
        fake_astream_bytes,
    )
    stream = await backend.synthesize(
        text="hello",
        response_format="wav",
        streaming=True,
        reference_id=None,
        extra_params=None,
    )

    closed_immediately = False
    try:
        assert await stream.__anext__() == b"chunk"
        if termination == "cancel":
            with pytest.raises(asyncio.CancelledError):
                await stream.athrow(asyncio.CancelledError())
        else:
            await stream.aclose()
        closed_immediately = inner_closed.is_set()
    finally:
        await stream.aclose()
        for inner_stream in inner_streams:
            await inner_stream.aclose()

    assert closed_immediately


@pytest.mark.asyncio
async def test_native_backend_detaches_transport_exception_graph(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    sentinels = (
        "fish-native-caller-context-secret",
        "fish-native-url-secret",
        "fish-native-header-secret",
        "fish-native-body-secret",
    )
    backend = FishS2NativeHttpBackend(
        {
            "base_url": "https://fish-native-url-secret@fish-native.invalid",
            "api_key": "fish-native-header-secret",
        }
    )

    async def fake_fetch(**kwargs):
        raw_request = httpx.Request(
            kwargs["method"],
            kwargs["url"],
            headers=kwargs["headers"],
        )
        response = httpx.Response(
            500,
            request=raw_request,
            content=b"fish-native-body-secret",
        )
        raise httpx.HTTPStatusError(
            "Fish native request failed",
            request=raw_request,
            response=response,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    try:
        raise RuntimeError("fish-native-caller-context-secret")
    except RuntimeError:
        with pytest.raises(TTSNetworkError) as exc_info:
            await backend.synthesize(
                text="hello",
                response_format="wav",
                streaming=False,
                reference_id=None,
                extra_params=None,
            )

    _assert_exception_graph_is_sanitized(exc_info.value, *sentinels)


@pytest.mark.asyncio
async def test_native_backend_rebuilds_typed_transport_error_without_raw_details(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    sentinels = (
        "fish-native-typed-caller-secret",
        "fish-native-typed-url-secret",
        "fish-native-typed-header-secret",
        "fish-native-typed-body-secret",
    )
    backend = FishS2NativeHttpBackend(
        {"base_url": "https://fish-native.invalid", "api_key": "test-key"}
    )
    raw_request = httpx.Request(
        "POST",
        "https://fish-native-typed-url-secret@fish-native.invalid/v1/tts",
        headers={"Authorization": "Bearer fish-native-typed-header-secret"},
    )
    raw_response = httpx.Response(
        500,
        request=raw_request,
        content=b"fish-native-typed-body-secret",
    )
    raw_transport = httpx.HTTPStatusError(
        "fish-native-typed-body-secret",
        request=raw_request,
        response=raw_response,
    )
    raw_error = TTSNetworkError(
        "fish-native-typed-url-secret",
        provider="fish_s2",
        details={"original_error": raw_transport},
    )
    raw_error.__cause__ = raw_transport

    async def fake_fetch(**_kwargs):
        raise raw_error

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    try:
        raise RuntimeError("fish-native-typed-caller-secret")
    except RuntimeError:
        with pytest.raises(TTSNetworkError) as exc_info:
            await backend.synthesize(
                text="hello",
                response_format="wav",
                streaming=False,
                reference_id=None,
                extra_params=None,
            )

    assert exc_info.value is not raw_error
    assert raw_error.__cause__ is raw_transport
    assert raw_error.details == {"original_error": raw_transport}
    _assert_exception_graph_is_sanitized(exc_info.value, *sentinels)


@pytest.mark.asyncio
async def test_native_backend_concurrent_failures_keep_sanitized_errors_request_local(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
        FishS2NativeHttpBackend,
    )

    sentinels = (
        "fish-native-a-url-secret",
        "fish-native-a-header-secret",
        "fish-native-a-body-secret",
        "fish-native-b-url-secret",
        "fish-native-b-header-secret",
        "fish-native-b-body-secret",
    )
    backends = (
        FishS2NativeHttpBackend(
            {
                "base_url": "https://fish-native-a-url-secret@fish-native-a.invalid",
                "api_key": "fish-native-a-header-secret",
            }
        ),
        FishS2NativeHttpBackend(
            {
                "base_url": "https://fish-native-b-url-secret@fish-native-b.invalid",
                "api_key": "fish-native-b-header-secret",
            }
        ),
    )
    both_arrived = asyncio.Event()
    arrivals: list[str] = []

    async def fake_fetch(**kwargs):
        url = str(kwargs["url"])
        arrivals.append(url)
        if len(arrivals) == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=5)

        raw_request = httpx.Request(
            kwargs["method"],
            kwargs["url"],
            headers=kwargs["headers"],
        )
        if "fish-native-a.invalid" in url:
            raise httpx.ReadTimeout(
                "fish-native-a-body-secret",
                request=raw_request,
            )
        response = httpx.Response(
            500,
            request=raw_request,
            content=b"fish-native-b-body-secret",
        )
        raise httpx.HTTPStatusError(
            "Fish native request failed",
            request=raw_request,
            response=response,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_native_http.afetch",
        fake_fetch,
    )

    results = await asyncio.gather(
        *(
            backend.synthesize(
                text="hello",
                response_format="wav",
                streaming=False,
                reference_id=None,
                extra_params=None,
            )
            for backend in backends
        ),
        return_exceptions=True,
    )

    assert len(arrivals) == 2
    assert isinstance(results[0], TTSTimeoutError)
    assert isinstance(results[1], TTSNetworkError)
    for result in results:
        assert isinstance(result, Exception)
        _assert_exception_graph_is_sanitized(result, *sentinels)
