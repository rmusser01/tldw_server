import base64
import json

import httpx
import pytest

from tldw_Server_API.app.core.http_client import create_async_client
from tldw_Server_API.app.core.TTS.adapters import audio_cpp_client as audio_cpp_client_module
from tldw_Server_API.app.core.TTS.adapters.audio_cpp_client import AudioCppClient
from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSConfigurationError,
    TTSNetworkError,
    TTSProviderError,
    TTSTimeoutError,
)

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


@pytest.mark.unit
@pytest.mark.parametrize("status", [302, 500])
async def test_http_errors_never_expose_arbitrary_upstream_body(status):
    body = 'Authorization: Bearer private-credential; "token": "private-token"'

    def handler(request):
        return httpx.Response(status, text=body)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        with pytest.raises(TTSProviderError) as error:
            await client.speech({"input": "hello"})
    assert "private-credential" not in str(error.value.details)
    assert "private-token" not in str(error.value.details)


@pytest.mark.unit
@pytest.mark.parametrize("body", [b"", b"{}", b"null", b"[]", b'"ok"', b'{"status":"starting"}'])
async def test_health_rejects_missing_or_unrecognized_status(body):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        with pytest.raises(TTSProviderError, match="health"):
            await client.health()


# --- Egress (TASK-13950.11): requests go through core.http_client.afetch ---------------------


@pytest.fixture
def egress_policy(monkeypatch):
    """Production egress policy: private and loopback addresses refused unless scoped.

    A refused request is proven by the MockTransport's record of requests; httpx connects
    through anyio, so no socket patch would catch a real send.
    """
    monkeypatch.setenv("WORKFLOWS_EGRESS_BLOCK_PRIVATE", "true")


def _recording_client(seen: list[httpx.Request], respond):
    """A core.http_client client whose socket layer is an httpx.MockTransport."""

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return respond(request)

    return create_async_client(transport=httpx.MockTransport(handler))


def _healthy(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok"})


@pytest.mark.unit
async def test_owned_client_reaches_the_configured_loopback_server_with_the_same_request(
    monkeypatch, egress_policy
):
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=WAV_BYTES, headers={"content-type": "audio/wav"})

    def factory(**kwargs):
        return create_async_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(audio_cpp_client_module, "create_async_client", factory)
    client = AudioCppClient(base_url="http://127.0.0.1:8080", timeout=12.5)
    try:
        result = await client.speech({"model": "pocket-tts", "input": "hello"})
    finally:
        await client.close()

    assert result.audio_bytes == WAV_BYTES
    [request] = seen
    assert (request.method, str(request.url)) == ("POST", "http://127.0.0.1:8080/v1/audio/speech")
    assert request.headers["host"] == "127.0.0.1:8080"
    assert request.headers["content-type"] == "application/json"
    payload = {"model": "pocket-tts", "input": "hello"}
    assert request.content == httpx.Request("POST", str(request.url), json=payload).content
    assert request.extensions["timeout"] == {"connect": 12.5, "read": 12.5, "write": 12.5, "pool": 12.5}


@pytest.mark.unit
async def test_remote_base_url_on_a_private_address_is_refused_before_sending(egress_policy):
    seen: list[httpx.Request] = []
    async with _recording_client(seen, _healthy) as http_client:
        client = AudioCppClient(
            base_url="http://10.0.0.5:8080",
            http_client=http_client,
            allow_remote_base_url=True,
        )
        with pytest.raises(TTSConfigurationError) as error:
            await client.health()

    assert error.value.error_code == "EGRESS_DENIED"
    assert seen == []


@pytest.mark.unit
async def test_remote_base_url_on_a_public_address_keeps_working(egress_policy):
    seen: list[httpx.Request] = []
    async with _recording_client(seen, _healthy) as http_client:
        client = AudioCppClient(
            base_url="http://93.184.216.34:8080",
            http_client=http_client,
            allow_remote_base_url=True,
        )
        assert await client.health() == {"status": "ok"}

    assert [str(request.url) for request in seen] == ["http://93.184.216.34:8080/health"]


@pytest.mark.unit
async def test_redirect_to_another_private_address_is_not_followed(egress_policy):
    seen: list[httpx.Request] = []

    def redirect(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://10.0.0.9:8080/health"})

    async with _recording_client(seen, redirect) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        with pytest.raises(TTSProviderError) as error:
            await client.health()

    assert error.value.error_code == "HTTP_302"
    assert [request.url.host for request in seen] == ["127.0.0.1"]


@pytest.mark.unit
async def test_transport_timeout_maps_to_tts_timeout_error(egress_policy):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async with create_async_client(transport=httpx.MockTransport(handler)) as http_client:
        client = AudioCppClient(base_url="http://127.0.0.1:8080", http_client=http_client)
        with pytest.raises(TTSTimeoutError):
            await client.health()
