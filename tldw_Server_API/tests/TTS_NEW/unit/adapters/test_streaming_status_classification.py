"""Streaming HTTP status classification at the sanitized transport boundary."""

from __future__ import annotations

import traceback
from collections.abc import AsyncIterator, Callable

import httpx
import pytest

import tldw_Server_API.app.core.http_client as http_client_mod
import tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api as fish_commercial_mod
import tldw_Server_API.app.core.TTS.backends.fish_s2_native_http as fish_native_mod
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.adapters.elevenlabs_adapter import (
    ElevenLabsTTSAdapter,
)
from tldw_Server_API.app.core.TTS.adapters.openai_adapter import OpenAITTSAdapter
from tldw_Server_API.app.core.TTS.adapters.qwen3_runtime_remote import (
    RemoteQwenRuntime,
)
from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import (
    FishS2CommercialApiBackend,
)
from tldw_Server_API.app.core.TTS.backends.fish_s2_native_http import (
    FishS2NativeHttpBackend,
)
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSNetworkError,
    TTSProviderError,
    TTSRateLimitError,
    TTSTimeoutError,
)

_BODY_SENTINEL = "stream-status-private-response-body"
_HEADER_SENTINEL = "stream-status-private-response-header"
_API_KEY_SENTINEL = "stream-status-private-api-key"
_STATUS_CASES = [
    pytest.param(None, TTSNetworkError, id="network"),
    pytest.param(401, TTSAuthenticationError, id="401-auth"),
    pytest.param(403, TTSAuthenticationError, id="403-auth"),
    pytest.param(408, TTSTimeoutError, id="408-timeout"),
    pytest.param(429, TTSRateLimitError, id="429-rate-limit"),
    pytest.param(504, TTSTimeoutError, id="504-timeout"),
]


def _assert_exception_graph_is_sanitized(exc: Exception) -> None:
    """Assert provider-owned response and credential data are unreachable."""

    assert exc.__cause__ is None
    assert exc.__context__ is None
    rendered = repr((str(exc), vars(exc), traceback.format_exception(exc)))
    for sentinel in (_BODY_SENTINEL, _HEADER_SENTINEL, _API_KEY_SENTINEL):
        assert sentinel not in rendered


def _status_client(status_code: int | None) -> tuple[httpx.AsyncClient, Callable[[], int]]:
    dispatches = 0

    def transport(request: httpx.Request) -> httpx.Response:
        nonlocal dispatches
        dispatches += 1
        assert request.method == "POST"
        if status_code is None:
            raise httpx.ReadError(_BODY_SENTINEL, request=request)
        return httpx.Response(
            status_code,
            request=request,
            headers={
                "retry-after": "17",
                "x-private-marker": _HEADER_SENTINEL,
            },
            content=_BODY_SENTINEL.encode(),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
    return client, lambda: dispatches


@pytest.fixture(autouse=True)
def allow_mock_transport_egress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the real streaming helper while replacing only external egress."""

    async def allow(*_args, **_kwargs) -> None:
        return None

    monkeypatch.setattr(http_client_mod, "_avalidate_egress_or_raise", allow)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    _STATUS_CASES,
)
async def test_qwen_stream_classifies_terminal_status_once(
    status_code: int | None,
    expected_error: type[TTSProviderError],
) -> None:
    client, dispatch_count = _status_client(status_code)
    runtime = RemoteQwenRuntime(
        {
            "base_url": "https://qwen-status.invalid/v1/audio/speech",
            "api_key": _API_KEY_SENTINEL,
        }
    )
    runtime.client = client

    try:
        response = await runtime.generate(
            TTSRequest(text="hello", format=AudioFormat.PCM, stream=True),
            resolved_model="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            mode="custom_voice",
        )
        with pytest.raises(expected_error) as exc_info:
            [chunk async for chunk in response.audio_stream]
    finally:
        await client.aclose()

    assert dispatch_count() == 1
    _assert_exception_graph_is_sanitized(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    _STATUS_CASES,
)
async def test_openai_stream_classifies_terminal_status_once(
    status_code: int | None,
    expected_error: type[TTSProviderError],
) -> None:
    client, dispatch_count = _status_client(status_code)
    adapter = OpenAITTSAdapter(
        {
            "api_key": _API_KEY_SENTINEL,
            "base_url": "https://openai-status.invalid/v1",
            "credentials_resolved": True,
        }
    )
    adapter.client = client

    try:
        stream = adapter.generate_stream(
            TTSRequest(text="hello", voice="alloy", format=AudioFormat.MP3)
        )
        with pytest.raises(expected_error) as exc_info:
            [chunk async for chunk in stream]
    finally:
        await client.aclose()

    assert dispatch_count() == 1
    _assert_exception_graph_is_sanitized(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    _STATUS_CASES,
)
async def test_elevenlabs_stream_classifies_terminal_status_once(
    status_code: int | None,
    expected_error: type[TTSProviderError],
) -> None:
    client, dispatch_count = _status_client(status_code)
    adapter = ElevenLabsTTSAdapter(
        {
            "api_key": _API_KEY_SENTINEL,
            "base_url": "https://elevenlabs-status.invalid/v1",
            "credentials_resolved": True,
        }
    )
    adapter.client = client

    try:
        stream = adapter.generate_stream(
            TTSRequest(text="hello", voice="rachel", format=AudioFormat.MP3)
        )
        with pytest.raises(expected_error) as exc_info:
            [chunk async for chunk in stream]
    finally:
        await client.aclose()

    assert dispatch_count() == 1
    _assert_exception_graph_is_sanitized(exc_info.value)
    if status_code in (401, 403, 429):
        assert exc_info.value.provider == "elevenlabs"


def _inject_mock_client(
    client: httpx.AsyncClient,
) -> Callable[..., AsyncIterator[bytes]]:
    def stream(**kwargs) -> AsyncIterator[bytes]:
        return http_client_mod.astream_bytes(client=client, **kwargs)

    return stream


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    _STATUS_CASES,
)
async def test_fish_commercial_stream_classifies_terminal_status_once(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int | None,
    expected_error: type[TTSProviderError],
) -> None:
    client, dispatch_count = _status_client(status_code)
    monkeypatch.setattr(
        fish_commercial_mod,
        "astream_bytes",
        _inject_mock_client(client),
    )
    backend = FishS2CommercialApiBackend(
        {
            "base_url": "https://fish-commercial-status.invalid",
            "api_key": _API_KEY_SENTINEL,
        }
    )

    try:
        stream = await backend.synthesize(
            text="hello",
            response_format="mp3",
            streaming=True,
            reference_id=None,
            extra_params=None,
        )
        with pytest.raises(expected_error) as exc_info:
            [chunk async for chunk in stream]
    finally:
        await client.aclose()

    assert dispatch_count() == 1
    _assert_exception_graph_is_sanitized(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    _STATUS_CASES,
)
async def test_fish_native_stream_classifies_terminal_status_once(
    monkeypatch: pytest.MonkeyPatch,
    status_code: int | None,
    expected_error: type[TTSProviderError],
) -> None:
    client, dispatch_count = _status_client(status_code)
    monkeypatch.setattr(
        fish_native_mod,
        "astream_bytes",
        _inject_mock_client(client),
    )
    backend = FishS2NativeHttpBackend(
        {
            "base_url": "https://fish-native-status.invalid",
            "api_key": _API_KEY_SENTINEL,
        }
    )

    try:
        stream = await backend.synthesize(
            text="hello",
            response_format="wav",
            streaming=True,
            reference_id=None,
            extra_params=None,
        )
        with pytest.raises(expected_error) as exc_info:
            [chunk async for chunk in stream]
    finally:
        await client.aclose()

    assert dispatch_count() == 1
    _assert_exception_graph_is_sanitized(exc_info.value)
