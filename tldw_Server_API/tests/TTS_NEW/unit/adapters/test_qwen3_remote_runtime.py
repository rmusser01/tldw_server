import asyncio
import traceback
from types import SimpleNamespace

import httpx
import pytest

import tldw_Server_API.app.core.http_client as http_client_mod
import tldw_Server_API.app.core.TTS.adapters.qwen3_runtime_remote as remote_runtime_mod
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.qwen3_runtime_remote import RemoteQwenRuntime
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSNetworkError,
    TTSProviderError,
    TTSRateLimitError,
    TTSTimeoutError,
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


@pytest.mark.asyncio
async def test_remote_runtime_maps_qwen_clone_fields_into_extended_payload():
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(
        text="hello",
        format=AudioFormat.PCM,
        voice_reference=b"VOICE_BYTES",
        extra_params={"reference_text": "ref", "voice_clone_prompt": "UFJPTVBU"},
    )

    payload = runtime._build_payload(
        request,
        resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        mode="voice_clone",
    )

    assert payload["extra_body"]["ref_text"] == "ref"
    assert payload["extra_body"]["voice_clone_prompt"] == "UFJPTVBU"


@pytest.mark.asyncio
async def test_remote_runtime_capabilities_default_to_conservative_values():
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )

    caps = await runtime.get_capabilities()

    assert caps.supports_streaming is False
    assert caps.supports_voice_cloning is False
    assert caps.supports_emotion_control is False
    assert caps.metadata["supported_modes"] == ["custom_voice_preset"]
    assert caps.metadata["supports_uploaded_custom_voices"] is False


@pytest.mark.asyncio
async def test_remote_runtime_does_not_advertise_local_speakers_by_default():
    runtime = RemoteQwenRuntime(
        SimpleNamespace(
            config={"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"},
            PROVIDER_KEY="qwen3_tts",
            provider_name="Qwen3TTS",
            sample_rate=24000,
            SUPPORTED_LANGUAGES={"en"},
            CUSTOMVOICE_SPEAKERS=["Cherry", "Ethan"],
        )
    )

    caps = await runtime.get_capabilities()

    assert caps.supported_voices == []


@pytest.mark.asyncio
async def test_remote_runtime_capabilities_allow_override():
    runtime = RemoteQwenRuntime(
        {
            "base_url": "http://127.0.0.1:8001/v1/audio/speech",
            "api_key": "test-key",
            "capability_override": {
                "supports_streaming": True,
                "supports_voice_cloning": True,
                "supports_emotion_control": True,
                "supported_modes": ["custom_voice_preset", "uploaded_custom_voice"],
                "supported_voices": ["Cherry", "Ethan"],
                "supports_uploaded_custom_voices": True,
            },
        }
    )

    caps = await runtime.get_capabilities()

    assert caps.supports_streaming is True
    assert caps.supports_voice_cloning is True
    assert caps.supports_emotion_control is True
    assert [voice.id for voice in caps.supported_voices] == ["Cherry", "Ethan"]
    assert caps.metadata["supported_modes"] == ["custom_voice_preset", "uploaded_custom_voice"]
    assert caps.metadata["supports_uploaded_custom_voices"] is True


@pytest.mark.asyncio
async def test_remote_runtime_maps_http_401_to_auth_error(monkeypatch):
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)

    async def fake_apost(**_kwargs):
        req = httpx.Request("POST", "http://127.0.0.1:8001/v1/audio/speech")
        return httpx.Response(401, request=req, content=b'{"error":"bad key"}')

    monkeypatch.setattr(remote_runtime_mod, "apost", fake_apost)

    with pytest.raises(TTSAuthenticationError):
        await runtime.generate(request, resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", mode="custom_voice")


@pytest.mark.asyncio
async def test_remote_runtime_maps_timeout_to_tts_timeout_error(monkeypatch):
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)

    async def fake_apost(**_kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(remote_runtime_mod, "apost", fake_apost)

    with pytest.raises(TTSTimeoutError):
        await runtime.generate(request, resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", mode="custom_voice")


@pytest.mark.asyncio
async def test_remote_runtime_stream_maps_http_401_to_auth_error(monkeypatch):
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=True)

    async def fake_astream_bytes(**_kwargs):
        req = httpx.Request("POST", "http://127.0.0.1:8001/v1/audio/speech")
        response = httpx.Response(401, request=req, content=b'{"error":"bad key"}')
        raise httpx.HTTPStatusError("401", request=req, response=response)
        yield b""  # pragma: no cover

    monkeypatch.setattr(remote_runtime_mod, "astream_bytes", fake_astream_bytes)

    response = await runtime.generate(
        request,
        resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        mode="custom_voice",
    )

    with pytest.raises(TTSAuthenticationError):
        [chunk async for chunk in response.audio_stream]


@pytest.mark.asyncio
async def test_remote_runtime_stream_uses_single_attempt_post_policy(monkeypatch):
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=True)
    captured = {}

    async def fake_astream_bytes(**kwargs):
        captured.update(kwargs)
        yield b"chunk-one"

    async def fail_if_apost_used(**_kwargs):
        raise AssertionError("streaming path should use astream_bytes")

    monkeypatch.setattr(remote_runtime_mod, "astream_bytes", fake_astream_bytes, raising=False)
    monkeypatch.setattr(remote_runtime_mod, "apost", fail_if_apost_used)

    response = await runtime.generate(
        request,
        resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        mode="custom_voice",
    )
    chunks = [chunk async for chunk in response.audio_stream]

    assert chunks == [b"chunk-one"]
    assert captured["method"] == "POST"
    assert captured["url"] == "http://127.0.0.1:8001/v1/audio/speech"
    assert captured["retry"].retry_on_unsafe is False
    assert captured["retry"].attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_kind", "expected_error"),
    [
        ("network", TTSNetworkError),
        ("rate_limit", TTSRateLimitError),
    ],
)
async def test_remote_runtime_stream_dispatches_post_once_before_first_byte_failure(
    monkeypatch,
    failure_kind,
    expected_error,
):
    """Ambiguous pre-first-byte failures cannot replay the synthesis POST."""

    dispatches = 0

    def _transport(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        if failure_kind == "network":
            raise httpx.ReadError("pre-first-byte failure", request=request)
        return httpx.Response(
            429,
            request=request,
            headers={"retry-after": "0"},
            content=b'{"error":"rate limited"}',
        )

    async def _allow_test_egress(*_args, **_kwargs) -> None:
        return None

    runtime = RemoteQwenRuntime(
        {"base_url": "https://qwen.invalid/v1/audio/speech", "api_key": "test-key"}
    )
    runtime.client = httpx.AsyncClient(transport=httpx.MockTransport(_transport))
    monkeypatch.setattr(http_client_mod, "_avalidate_egress_or_raise", _allow_test_egress)
    monkeypatch.setattr(
        http_client_mod,
        "_decorrelated_jitter_sleep",
        lambda *_args, **_kwargs: 0.0,
    )

    try:
        response = await runtime.generate(
            TTSRequest(text="hello", format=AudioFormat.PCM, stream=True),
            resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            mode="custom_voice",
        )
        with pytest.raises(expected_error) as exc_info:
            [chunk async for chunk in response.audio_stream]
    finally:
        await runtime.client.aclose()

    assert dispatches == 1
    _assert_exception_graph_is_sanitized(
        exc_info.value,
        "pre-first-byte failure",
        "rate limited",
    )


@pytest.mark.asyncio
async def test_remote_runtime_stream_maps_iterator_timeout_to_tts_timeout_error(monkeypatch):
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=True)

    async def fake_astream_bytes(**_kwargs):
        raise httpx.ReadTimeout("timed out during stream")
        yield b""  # pragma: no cover

    monkeypatch.setattr(remote_runtime_mod, "astream_bytes", fake_astream_bytes)

    response = await runtime.generate(
        request,
        resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        mode="custom_voice",
    )

    with pytest.raises(TTSTimeoutError):
        [chunk async for chunk in response.audio_stream]


@pytest.mark.asyncio
@pytest.mark.parametrize("termination", ["aclose", "cancel"])
async def test_remote_runtime_stream_deterministically_closes_inner_iterator(
    monkeypatch,
    termination,
):
    runtime = RemoteQwenRuntime(
        {"base_url": "https://qwen.invalid/v1/audio/speech", "api_key": "test-key"}
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

    monkeypatch.setattr(remote_runtime_mod, "astream_bytes", fake_astream_bytes)
    response = await runtime.generate(
        TTSRequest(text="hello", format=AudioFormat.PCM, stream=True),
        resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        mode="custom_voice",
    )
    stream = response.audio_stream

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
async def test_remote_runtime_maps_http_429_date_retry_after_to_rate_limit_error(monkeypatch):
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)

    async def fake_apost(**_kwargs):
        req = httpx.Request("POST", "http://127.0.0.1:8001/v1/audio/speech")
        return httpx.Response(
            429,
            request=req,
            headers={"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"},
            content=b'{"error":"rate limited"}',
        )

    monkeypatch.setattr(remote_runtime_mod, "apost", fake_apost)

    with pytest.raises(TTSRateLimitError) as exc:
        await runtime.generate(
            request,
            resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            mode="custom_voice",
        )

    assert exc.value.retry_after is None


def test_remote_runtime_retry_after_parser_ignores_http_date():
    runtime = RemoteQwenRuntime(
        {"base_url": "http://127.0.0.1:8001/v1/audio/speech", "api_key": "test-key"}
    )

    assert runtime._parse_retry_after("120") == 120
    assert runtime._parse_retry_after(" Wed, 21 Oct 2015 07:28:00 GMT ") is None


@pytest.mark.asyncio
async def test_remote_runtime_detaches_http_error_exception_graph(monkeypatch):
    sentinels = (
        "qwen-caller-context-secret",
        "qwen-url-user-secret",
        "qwen-url-password-secret",
        "qwen-query-secret",
        "qwen-header-secret",
        "qwen-response-body-secret",
    )
    runtime = RemoteQwenRuntime(
        {
            "base_url": (
                "https://qwen-url-user-secret:qwen-url-password-secret@qwen.invalid/"
                "v1/audio/speech?access_token=qwen-query-secret"
            ),
            "api_key": "qwen-header-secret",
        }
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)

    async def fake_apost(**kwargs):
        raw_request = httpx.Request(
            "POST",
            kwargs["url"],
            headers=kwargs["headers"],
        )
        return httpx.Response(
            500,
            request=raw_request,
            content=b"qwen-response-body-secret",
        )

    monkeypatch.setattr(remote_runtime_mod, "apost", fake_apost)

    try:
        raise RuntimeError("qwen-caller-context-secret")
    except RuntimeError:
        with pytest.raises(TTSProviderError) as exc_info:
            await runtime.generate(
                request,
                resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                mode="custom_voice",
            )

    _assert_exception_graph_is_sanitized(exc_info.value, *sentinels)


@pytest.mark.asyncio
async def test_remote_runtime_rebuilds_typed_transport_error_without_raw_details(
    monkeypatch,
):
    sentinels = (
        "qwen-typed-caller-secret",
        "qwen-typed-url-secret",
        "qwen-typed-header-secret",
        "qwen-typed-body-secret",
    )
    runtime = RemoteQwenRuntime(
        {"base_url": "https://qwen.invalid/v1/audio/speech", "api_key": "test-key"}
    )
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)
    raw_request = httpx.Request(
        "POST",
        "https://qwen-typed-url-secret@qwen.invalid/v1/audio/speech",
        headers={"Authorization": "Bearer qwen-typed-header-secret"},
    )
    raw_response = httpx.Response(
        500,
        request=raw_request,
        content=b"qwen-typed-body-secret",
    )
    raw_transport = httpx.HTTPStatusError(
        "qwen-typed-body-secret",
        request=raw_request,
        response=raw_response,
    )
    raw_error = TTSNetworkError(
        "qwen-typed-url-secret",
        provider="qwen3_tts",
        details={"original_error": raw_transport},
    )
    raw_error.__cause__ = raw_transport

    async def fake_apost(**_kwargs):
        raise raw_error

    monkeypatch.setattr(remote_runtime_mod, "apost", fake_apost)

    try:
        raise RuntimeError("qwen-typed-caller-secret")
    except RuntimeError:
        with pytest.raises(TTSNetworkError) as exc_info:
            await runtime.generate(
                request,
                resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                mode="custom_voice",
            )

    assert exc_info.value is not raw_error
    assert raw_error.__cause__ is raw_transport
    assert raw_error.details == {"original_error": raw_transport}
    _assert_exception_graph_is_sanitized(exc_info.value, *sentinels)


@pytest.mark.asyncio
async def test_remote_runtime_concurrent_failures_keep_sanitized_errors_request_local(monkeypatch):
    sentinels = (
        "qwen-a-url-secret",
        "qwen-a-query-secret",
        "qwen-a-header-secret",
        "qwen-a-body-secret",
        "qwen-b-url-secret",
        "qwen-b-query-secret",
        "qwen-b-header-secret",
        "qwen-b-body-secret",
    )
    runtimes = (
        RemoteQwenRuntime(
            {
                "base_url": (
                    "https://qwen-a-url-secret@qwen-a.invalid/v1/audio/speech"
                    "?token=qwen-a-query-secret"
                ),
                "api_key": "qwen-a-header-secret",
            }
        ),
        RemoteQwenRuntime(
            {
                "base_url": (
                    "https://qwen-b-url-secret@qwen-b.invalid/v1/audio/speech"
                    "?token=qwen-b-query-secret"
                ),
                "api_key": "qwen-b-header-secret",
            }
        ),
    )
    both_arrived = asyncio.Event()
    arrivals: list[str] = []

    async def fake_apost(**kwargs):
        url = str(kwargs["url"])
        arrivals.append(url)
        if len(arrivals) == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=5)

        raw_request = httpx.Request(
            "POST",
            kwargs["url"],
            headers=kwargs["headers"],
        )
        if "qwen-a.invalid" in url:
            return httpx.Response(
                401,
                request=raw_request,
                content=b"qwen-a-body-secret",
            )
        return httpx.Response(
            429,
            request=raw_request,
            content=b"qwen-b-body-secret",
        )

    monkeypatch.setattr(remote_runtime_mod, "apost", fake_apost)
    request = TTSRequest(text="hello", format=AudioFormat.PCM, stream=False)

    results = await asyncio.gather(
        *(
            runtime.generate(
                request,
                resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                mode="custom_voice",
            )
            for runtime in runtimes
        ),
        return_exceptions=True,
    )

    assert len(arrivals) == 2
    assert isinstance(results[0], TTSAuthenticationError)
    assert isinstance(results[1], TTSRateLimitError)
    for result in results:
        assert isinstance(result, Exception)
        _assert_exception_graph_is_sanitized(result, *sentinels)
