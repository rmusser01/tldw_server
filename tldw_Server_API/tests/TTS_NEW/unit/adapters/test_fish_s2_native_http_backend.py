from __future__ import annotations

import pytest

from tldw_Server_API.app.core.TTS.tts_exceptions import TTSAuthenticationError


class _FakeResponse:
    def __init__(self, status_code: int = 200, content: bytes = b"audio", text: str = "ok"):
        self.status_code = status_code
        self.content = content
        self.text = text


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

    backend = FishS2NativeHttpBackend({"base_url": "http://fish.local"})

    async def fake_stream(*, method, url, json=None, headers=None, timeout=None, **_kwargs):
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
