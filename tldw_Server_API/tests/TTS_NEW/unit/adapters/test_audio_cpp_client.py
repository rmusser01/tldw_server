import base64
import json

import httpx
import pytest

from tldw_Server_API.app.core.TTS.adapters.audio_cpp_client import AudioCppClient
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSNetworkError, TTSProviderError

WAV_BYTES = b"RIFF\x24\x00\x00\x00WAVEfmt "


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_cpp_client_health_and_models_use_expected_routes():
    seen: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "pocket-tts"}]})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)

        assert await client.health() == {"status": "ok"}
        assert await client.list_models() == ["pocket-tts"]

    assert seen == [("GET", "/health"), ("GET", "/v1/models")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_cpp_client_speech_returns_wav_bytes():
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(200, content=WAV_BYTES, headers={"content-type": "audio/wav"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080/", http_client=http_client)
        result = await client.speech({"model": "pocket-tts", "input": "hello"})

    assert result.audio_bytes == WAV_BYTES
    assert result.content_type == "audio/wav"
    assert captured_payload == {"model": "pocket-tts", "input": "hello"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_cpp_client_speech_decodes_json_base64_audio():
    encoded = base64.b64encode(WAV_BYTES).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"audio": encoded, "format": "wav"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        result = await client.speech({"model": "pocket-tts", "input": "hello"})

    assert result.audio_bytes == WAV_BYTES
    assert result.content_type == "application/json"
    assert result.metadata["json_format"] == "wav"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_cpp_client_sanitizes_upstream_http_errors():
    request_text = "sensitive request text"
    leaked_path = r"C:\Users\GDesktop-1\Working\tldw\models\audio_cpp\secret.wav"
    long_body = f"{request_text} failed at {leaked_path} " + ("x" * 600)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text=long_body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        with pytest.raises(TTSProviderError) as exc_info:
            await client.speech({"model": "pocket-tts", "input": request_text})

    error = exc_info.value
    assert request_text not in str(error)
    assert leaked_path not in str(error)
    assert error.details["status_code"] == 500
    assert request_text not in error.details["response_text"]
    assert leaked_path not in error.details["response_text"]
    assert len(error.details["response_text"]) <= 320


@pytest.mark.unit
@pytest.mark.asyncio
async def test_audio_cpp_client_maps_transport_errors_to_tts_network_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        with pytest.raises(TTSNetworkError):
            await client.health()
