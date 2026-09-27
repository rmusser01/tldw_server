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
        self.headers = headers or {}
        self._json_data = json_data

    def json(self):
        return self._json_data


@pytest.mark.asyncio
async def test_commercial_backend_builds_tts_payload_with_model_header(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend(
        {
            "base_url": "https://api.fish.audio",
            "api_key": "secret",
            "model": "s2-pro",
            "timeout": 30,
        }
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
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    audio = await backend.synthesize(
        text="hello",
        response_format="mp3",
        streaming=False,
        reference_id="voice-model-id",
        extra_params={
            "sample_rate": 44100,
            "mp3_bitrate": 128,
            "latency": "balanced",
            "prosody": {"speed": 1.2, "volume": 0},
            "ignored": "value",
        },
    )

    assert audio == b"audio"
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.fish.audio/v1/tts"
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "model": "s2-pro",
    }
    assert captured["json"] == {
        "text": "hello",
        "format": "mp3",
        "reference_id": "voice-model-id",
        "sample_rate": 44100,
        "mp3_bitrate": 128,
        "latency": "balanced",
        "prosody": {"speed": 1.2, "volume": 0},
    }
    assert captured["timeout"] == 30


@pytest.mark.asyncio
async def test_commercial_backend_uses_streaming_helper(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend(
        {
            "base_url": "https://api.fish.audio",
            "api_key": "secret",
            "model": "s2-pro",
        }
    )
    captured: dict[str, object] = {}

    async def fake_stream(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        yield b"chunk-1"
        yield b"chunk-2"

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.astream_bytes",
        fake_stream,
    )

    stream = await backend.synthesize(
        text="hello",
        response_format="opus",
        streaming=True,
        reference_id=None,
        extra_params={"opus_bitrate": 32000},
    )
    chunks = [chunk async for chunk in stream]

    assert chunks == [b"chunk-1", b"chunk-2"]
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.fish.audio/v1/tts"
    assert captured["json"] == {"text": "hello", "format": "opus", "opus_bitrate": 32000}
    assert captured["headers"] == {
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "model": "s2-pro",
    }


@pytest.mark.asyncio
async def test_commercial_backend_creates_hosted_voice_model(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend(
        {
            "base_url": "https://api.fish.audio",
            "api_key": "secret",
            "timeout": 45,
        }
    )
    captured: dict[str, object] = {}

    async def fake_fetch(*, method, url, data=None, files=None, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["data"] = data
        captured["files"] = files
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(status_code=201, json_data={"_id": "fish-hosted-model-id", "state": "trained"})

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    result = await backend.add_reference(
        reference_id="tldw_u1_voice-1",
        audio_b64="QUJD",
        reference_text="hello there",
        title="Voice One",
        description="private clone",
    )

    assert result == {
        "reference_id": "fish-hosted-model-id",
        "remote_reference_id": "fish-hosted-model-id",
        "state": "trained",
    }
    assert captured["method"] == "POST"
    assert captured["url"] == "https://api.fish.audio/model"
    assert captured["data"] == {
        "type": "tts",
        "title": "Voice One",
        "description": "private clone",
        "train_mode": "fast",
        "visibility": "private",
        "texts": "hello there",
        "enhance_audio_quality": "true",
        "generate_sample": "false",
    }
    assert captured["files"] == {
        "voices": ("reference.wav", b"ABC", "audio/wav"),
    }
    assert captured["headers"] == {"Authorization": "Bearer secret"}
    assert captured["timeout"] == 45


@pytest.mark.asyncio
async def test_commercial_backend_rejects_invalid_reference_audio():
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend({"api_key": "secret"})
    sentinel = "fish-commercial-invalid-base64-secret"

    with pytest.raises(TTSValidationError) as exc:
        await backend.add_reference(
            reference_id="voice-1",
            audio_b64=sentinel,
            reference_text="hello there",
        )

    assert "base64" in str(exc.value).lower()
    _assert_exception_graph_is_sanitized(exc.value, sentinel)
    _assert_traceback_local_omits(
        exc.value,
        module_name="tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api",
        local_name="audio_b64",
        sentinel=sentinel,
    )


@pytest.mark.asyncio
async def test_commercial_backend_deletes_hosted_voice_model(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend({"base_url": "https://api.fish.audio", "api_key": "secret"})
    captured: dict[str, object] = {}

    async def fake_fetch(*, method, url, headers=None, timeout=None, **_kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _FakeResponse(status_code=204, content=b"")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    assert await backend.delete_reference(reference_id="fish-hosted-model-id") is True
    assert captured == {
        "method": "DELETE",
        "url": "https://api.fish.audio/model/fish-hosted-model-id",
        "headers": {"Authorization": "Bearer secret"},
        "timeout": 60,
    }


@pytest.mark.parametrize(
    ("status_code", "expected_exception"),
    [
        (401, TTSAuthenticationError),
        (402, TTSProviderError),
        (429, TTSRateLimitError),
        (422, TTSValidationError),
        (500, TTSProviderError),
    ],
)
@pytest.mark.asyncio
async def test_commercial_backend_maps_error_responses(monkeypatch, status_code, expected_exception):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend({"base_url": "https://api.fish.audio", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        return _FakeResponse(
            status_code=status_code,
            text='{"message":"upstream failed"}',
            headers={"retry-after": "12"},
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    with pytest.raises(expected_exception):
        await backend.synthesize(
            text="hello",
            response_format="mp3",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )


@pytest.mark.asyncio
async def test_commercial_backend_maps_fetch_timeout(monkeypatch):
    from tldw_Server_API.app.core.exceptions import NetworkError as CoreNetworkError
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend({"base_url": "https://api.fish.audio", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        raise CoreNetworkError("ReadTimeout")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    with pytest.raises(TTSTimeoutError):
        await backend.synthesize(
            text="hello",
            response_format="mp3",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )


@pytest.mark.asyncio
async def test_commercial_backend_maps_fetch_network_errors(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend({"base_url": "https://api.fish.audio", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        raise OSError("connection failed")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    with pytest.raises(TTSNetworkError):
        await backend.synthesize(
            text="hello",
            response_format="mp3",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )


@pytest.mark.asyncio
async def test_commercial_backend_does_not_stringify_transport_errors(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
        FishS2CommercialApiBackend,
    )

    sentinel = "fish-commercial-hostile-str-secret"
    stringify_calls = 0

    class HostileTransportError(Exception):
        def __str__(self):
            nonlocal stringify_calls
            stringify_calls += 1
            raise RuntimeError(sentinel)

    async def fake_fetch(**_kwargs):
        raise HostileTransportError()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )
    backend = FishS2CommercialApiBackend(
        {"base_url": "https://fish-commercial.invalid", "api_key": "test-key"}
    )

    with pytest.raises(TTSNetworkError) as exc_info:
        await backend.synthesize(
            text="hello",
            response_format="mp3",
            streaming=False,
            reference_id=None,
            extra_params=None,
        )

    assert stringify_calls == 0
    _assert_exception_graph_is_sanitized(exc_info.value, sentinel)


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["aclose", "cancel"])
async def test_commercial_backend_stream_deterministically_closes_inner_iterator(
    monkeypatch,
    termination,
):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
        FishS2CommercialApiBackend,
    )

    backend = FishS2CommercialApiBackend(
        {"base_url": "https://fish-commercial.invalid", "api_key": "test-key"}
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
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.astream_bytes",
        fake_astream_bytes,
    )
    stream = await backend.synthesize(
        text="hello",
        response_format="mp3",
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
async def test_commercial_backend_detaches_transport_exception_graph(monkeypatch):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
        FishS2CommercialApiBackend,
    )

    sentinels = (
        "fish-commercial-caller-context-secret",
        "fish-commercial-url-secret",
        "fish-commercial-header-secret",
        "fish-commercial-body-secret",
    )
    backend = FishS2CommercialApiBackend(
        {
            "base_url": "https://fish-commercial-url-secret@fish-commercial.invalid",
            "api_key": "fish-commercial-header-secret",
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
            content=b"fish-commercial-body-secret",
        )
        raise httpx.HTTPStatusError(
            "Fish commercial request failed",
            request=raw_request,
            response=response,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    try:
        raise RuntimeError("fish-commercial-caller-context-secret")
    except RuntimeError:
        with pytest.raises(TTSNetworkError) as exc_info:
            await backend.synthesize(
                text="hello",
                response_format="mp3",
                streaming=False,
                reference_id=None,
                extra_params=None,
            )

    _assert_exception_graph_is_sanitized(exc_info.value, *sentinels)


@pytest.mark.asyncio
async def test_commercial_backend_rebuilds_typed_transport_error_without_raw_details(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
        FishS2CommercialApiBackend,
    )

    sentinels = (
        "fish-commercial-typed-caller-secret",
        "fish-commercial-typed-url-secret",
        "fish-commercial-typed-header-secret",
        "fish-commercial-typed-body-secret",
    )
    backend = FishS2CommercialApiBackend(
        {"base_url": "https://fish-commercial.invalid", "api_key": "test-key"}
    )
    raw_request = httpx.Request(
        "POST",
        "https://fish-commercial-typed-url-secret@fish-commercial.invalid/v1/tts",
        headers={"Authorization": "Bearer fish-commercial-typed-header-secret"},
    )
    raw_response = httpx.Response(
        500,
        request=raw_request,
        content=b"fish-commercial-typed-body-secret",
    )
    raw_transport = httpx.HTTPStatusError(
        "fish-commercial-typed-body-secret",
        request=raw_request,
        response=raw_response,
    )
    raw_error = TTSNetworkError(
        "fish-commercial-typed-url-secret",
        provider="fish_s2",
        details={"original_error": raw_transport},
    )
    raw_error.__cause__ = raw_transport

    async def fake_fetch(**_kwargs):
        raise raw_error

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    try:
        raise RuntimeError("fish-commercial-typed-caller-secret")
    except RuntimeError:
        with pytest.raises(TTSNetworkError) as exc_info:
            await backend.synthesize(
                text="hello",
                response_format="mp3",
                streaming=False,
                reference_id=None,
                extra_params=None,
            )

    assert exc_info.value is not raw_error
    assert raw_error.__cause__ is raw_transport
    assert raw_error.details == {"original_error": raw_transport}
    _assert_exception_graph_is_sanitized(exc_info.value, *sentinels)


@pytest.mark.asyncio
async def test_commercial_backend_concurrent_failures_keep_sanitized_errors_request_local(
    monkeypatch,
):
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
        FishS2CommercialApiBackend,
    )

    sentinels = (
        "fish-commercial-a-url-secret",
        "fish-commercial-a-header-secret",
        "fish-commercial-a-body-secret",
        "fish-commercial-b-url-secret",
        "fish-commercial-b-header-secret",
        "fish-commercial-b-body-secret",
    )
    backends = (
        FishS2CommercialApiBackend(
            {
                "base_url": "https://fish-commercial-a-url-secret@fish-commercial-a.invalid",
                "api_key": "fish-commercial-a-header-secret",
            }
        ),
        FishS2CommercialApiBackend(
            {
                "base_url": "https://fish-commercial-b-url-secret@fish-commercial-b.invalid",
                "api_key": "fish-commercial-b-header-secret",
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
        if "fish-commercial-a.invalid" in url:
            raise httpx.ReadTimeout(
                "fish-commercial-a-body-secret",
                request=raw_request,
            )
        response = httpx.Response(
            500,
            request=raw_request,
            content=b"fish-commercial-b-body-secret",
        )
        raise httpx.HTTPStatusError(
            "Fish commercial request failed",
            request=raw_request,
            response=response,
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api.afetch",
        fake_fetch,
    )

    results = await asyncio.gather(
        *(
            backend.synthesize(
                text="hello",
                response_format="mp3",
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
