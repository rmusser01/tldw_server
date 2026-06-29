from __future__ import annotations

import pytest

from tldw_Server_API.app.core.TTS.tts_exceptions import (
    TTSAuthenticationError,
    TTSNetworkError,
    TTSProviderError,
    TTSRateLimitError,
    TTSTimeoutError,
    TTSValidationError,
)


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

    with pytest.raises(TTSValidationError) as exc:
        await backend.add_reference(
            reference_id="voice-1",
            audio_b64="not base64",
            reference_text="hello there",
        )

    assert "base64" in str(exc.value).lower()


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
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = FishS2CommercialApiBackend({"base_url": "https://api.fish.audio", "api_key": "secret"})

    async def fake_fetch(**_kwargs):
        raise TimeoutError("request timed out")

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
