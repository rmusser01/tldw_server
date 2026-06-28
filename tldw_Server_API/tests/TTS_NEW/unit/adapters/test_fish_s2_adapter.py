from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSProviderInitializationError


class _FakeBackend:
    def __init__(self, *, response: bytes = b"audio", stream_chunks: list[bytes] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._response = response
        self._stream_chunks = stream_chunks or [b"chunk1", b"chunk2"]

    async def health_check(self) -> bool:
        return True

    async def synthesize(
        self,
        *,
        text: str,
        response_format: str,
        streaming: bool,
        reference_id: str | list[str] | None,
        extra_params: dict[str, object] | None,
    ) -> bytes | AsyncIterator[bytes]:
        self.calls.append(
            {
                "text": text,
                "response_format": response_format,
                "streaming": streaming,
                "reference_id": reference_id,
                "extra_params": extra_params,
            }
        )
        if streaming:
            async def _gen() -> AsyncIterator[bytes]:
                for chunk in self._stream_chunks:
                    yield chunk

            return _gen()
        return self._response


def test_build_backend_supports_commercial_api():
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import _build_backend
    from tldw_Server_API.app.core.TTS.backends.fish_s2_commercial_api import FishS2CommercialApiBackend

    backend = _build_backend(
        {
            "backend": "commercial_api",
            "base_url": "https://api.fish.audio",
            "api_key": "secret",
            "model": "s2-pro",
        }
    )

    assert isinstance(backend, FishS2CommercialApiBackend)
    assert backend.base_url == "https://api.fish.audio"
    assert backend.api_key == "secret"
    assert backend.model == "s2-pro"


@pytest.mark.asyncio
async def test_adapter_initializes_native_http_backend(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    backend = _FakeBackend()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter._build_backend",
        lambda config: backend,
    )

    adapter = FishS2Adapter({"backend": "native_http", "base_url": "http://fish.local"})

    assert await adapter.ensure_initialized() is True
    assert adapter.capabilities is not None
    assert adapter.capabilities.provider_name == "Fish Audio S2"
    assert adapter.capabilities.supports_streaming is True
    assert adapter.capabilities.supports_voice_cloning is True
    assert adapter.capabilities.supports_multi_speaker is False


@pytest.mark.asyncio
async def test_adapter_capabilities_include_fish_audio_formats(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    backend = _FakeBackend()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter._build_backend",
        lambda config: backend,
    )

    adapter = FishS2Adapter({"backend": "commercial_api", "api_key": "secret"})

    assert await adapter.ensure_initialized() is True
    assert adapter.capabilities is not None
    assert AudioFormat.OPUS in adapter.capabilities.supported_formats
    assert adapter.capabilities.default_format == AudioFormat.WAV


@pytest.mark.asyncio
async def test_adapter_rejects_unimplemented_local_runtime():
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    adapter = FishS2Adapter({"backend": "local_runtime"})

    with pytest.raises(TTSProviderInitializationError):
        await adapter.ensure_initialized()


@pytest.mark.asyncio
async def test_adapter_maps_request_and_merges_defaults(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    backend = _FakeBackend(response=b"fish-audio")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter._build_backend",
        lambda config: backend,
    )

    adapter = FishS2Adapter(
        {
            "backend": "native_http",
            "base_url": "http://fish.local",
            "sample_rate": 24000,
            "extra_params": {
                "default_chunk_length": 180,
                "default_normalize": False,
            },
        }
    )
    await adapter.ensure_initialized()

    response = await adapter.generate(
        TTSRequest(
            text="hello",
            voice="fishref:voice-from-voice",
            format=AudioFormat.WAV,
            stream=False,
            extra_params={
                "reference_id": "voice-explicit",
                "seed": 7,
            },
        )
    )

    assert response.audio_content == b"fish-audio"
    assert response.format == AudioFormat.WAV
    assert response.sample_rate == 24000
    assert backend.calls[-1] == {
        "text": "hello",
        "response_format": "wav",
        "streaming": False,
        "reference_id": "voice-explicit",
        "extra_params": {
            "chunk_length": 180,
            "normalize": False,
            "seed": 7,
        },
    }


@pytest.mark.asyncio
async def test_adapter_preserves_reference_lists_and_commercial_params(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    backend = _FakeBackend(response=b"fish-audio")
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter._build_backend",
        lambda config: backend,
    )

    adapter = FishS2Adapter({"backend": "commercial_api", "api_key": "secret"})
    await adapter.ensure_initialized()

    await adapter.generate(
        TTSRequest(
            text="hello",
            format=AudioFormat.OPUS,
            stream=False,
            extra_params={
                "reference_id": ["voice-a", "voice-b"],
                "sample_rate": 44100,
                "opus_bitrate": 32000,
                "latency": "balanced",
                "prosody": {"speed": 1.1},
                "condition_on_previous_chunks": False,
                "early_stop_threshold": 0.7,
                "max_new_tokens": 1024,
                "min_chunk_length": 80,
                "ignored": "value",
            },
        )
    )

    assert backend.calls[-1] == {
        "text": "hello",
        "response_format": "opus",
        "streaming": False,
        "reference_id": ["voice-a", "voice-b"],
        "extra_params": {
            "sample_rate": 44100,
            "opus_bitrate": 32000,
            "latency": "balanced",
            "prosody": {"speed": 1.1},
            "condition_on_previous_chunks": False,
            "early_stop_threshold": 0.7,
            "max_new_tokens": 1024,
            "min_chunk_length": 80,
        },
    }


@pytest.mark.asyncio
async def test_adapter_returns_streaming_response(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    backend = _FakeBackend(stream_chunks=[b"a", b"b", b"c"])
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter._build_backend",
        lambda config: backend,
    )

    adapter = FishS2Adapter({"backend": "native_http", "base_url": "http://fish.local"})
    await adapter.ensure_initialized()

    response = await adapter.generate(
        TTSRequest(
            text="hello",
            format=AudioFormat.WAV,
            stream=True,
            extra_params={"reference_id": "voice-123"},
        )
    )

    assert response.audio_stream is not None
    chunks = [chunk async for chunk in response.audio_stream]
    assert chunks == [b"a", b"b", b"c"]


@pytest.mark.asyncio
async def test_adapter_add_and_delete_reference_delegate_to_backend(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter import FishS2Adapter

    class _BackendWithRefs(_FakeBackend):
        def __init__(self):
            super().__init__()
            self.add_calls = []
            self.delete_calls = []

        async def add_reference(self, *, reference_id, audio_b64, reference_text, title=None, description=None):
            self.add_calls.append(
                {
                    "reference_id": reference_id,
                    "audio_b64": audio_b64,
                    "reference_text": reference_text,
                    "title": title,
                    "description": description,
                }
            )
            return {"reference_id": reference_id}

        async def delete_reference(self, *, reference_id):
            self.delete_calls.append(reference_id)
            return True

    backend = _BackendWithRefs()
    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.adapters.fish_s2_adapter._build_backend",
        lambda config: backend,
    )

    adapter = FishS2Adapter({"backend": "native_http", "base_url": "http://fish.local"})
    await adapter.ensure_initialized()

    created = await adapter.add_reference(
        reference_id="tldw_u1_voice-1",
        audio_b64="QUJD",
        reference_text="hello there",
        title="Voice One",
        description="private clone",
    )
    deleted = await adapter.delete_reference(reference_id="tldw_u1_voice-1")

    assert created == {"reference_id": "tldw_u1_voice-1"}
    assert deleted is True
    assert backend.add_calls == [
        {
            "reference_id": "tldw_u1_voice-1",
            "audio_b64": "QUJD",
            "reference_text": "hello there",
            "title": "Voice One",
            "description": "private clone",
        }
    ]
    assert backend.delete_calls == ["tldw_u1_voice-1"]
