from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.adapters.audio_cpp_adapter import AudioCppTTSAdapter
from tldw_Server_API.app.core.TTS.adapters.audio_cpp_client import AudioCppSpeechResult
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError

WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt "


class _FakeAudioCppClient:
    def __init__(self, *, audio_bytes: bytes = WAV_BYTES, models: list[str] | None = None) -> None:
        self.audio_bytes = audio_bytes
        self.models = models or ["pocket-tts"]
        self.payloads: list[dict[str, object]] = []
        self.closed = False

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return self.models

    async def speech(self, payload: dict[str, object]) -> AudioCppSpeechResult:
        self.payloads.append(dict(payload))
        return AudioCppSpeechResult(
            audio_bytes=self.audio_bytes,
            content_type="audio/wav",
            metadata={"upstream_latency_ms": 12},
        )

    async def close(self) -> None:
        self.closed = True


def _provider_config(
    *,
    client: _FakeAudioCppClient,
    managed: bool = False,
    external_voice_reference_mode: str = "disabled",
    retain_request_artifacts: bool = False,
    scratch_dir: Path | None = None,
) -> dict[str, object]:
    scratch = scratch_dir or Path("models/audio_cpp/runtime/test_scratch")
    return {
        "enabled": True,
        "base_url": "http://127.0.0.1:8080",
        "model": "audio-cpp/pocket-tts",
        "model_path": "models/audio_cpp/pocket-tts",
        "sample_rate": 24000,
        "timeout": 300,
        "client": client,
        "extra_params": {
            "managed": managed,
            "allow_remote_base_url": False,
            "external_voice_reference_mode": external_voice_reference_mode,
            "retain_request_artifacts": retain_request_artifacts,
            "request_option_allowlist": ["max_tokens", "seed"],
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "models_root": "models/audio_cpp",
                "shared_scratch_dir": str(scratch),
                "model": {
                    "id": "pocket-tts",
                    "family": "pocket_tts",
                    "path": "models/audio_cpp/pocket-tts",
                    "task": "tts",
                    "mode": "offline",
                },
            },
            "voices": {
                "alba": {
                    "name": "Alba",
                    "language": "en",
                    "upstream_value": "alba",
                    "request_field": None,
                }
            },
        },
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_capabilities_advertise_one_shot_streaming_and_verified_formats():
    client = _FakeAudioCppClient()
    adapter = AudioCppTTSAdapter(_provider_config(client=client))

    assert await adapter.ensure_initialized() is True

    capabilities = adapter.capabilities
    assert capabilities is not None
    assert capabilities.supports_streaming is True
    assert capabilities.metadata["incremental_streaming"] is False
    assert {AudioFormat.OGG, AudioFormat.WEBM, AudioFormat.ULAW}.isdisjoint(
        capabilities.supported_formats
    )
    assert [voice.id for voice in capabilities.supported_voices] == ["alba"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_only_request_posts_model_input_and_allowlisted_options():
    client = _FakeAudioCppClient()
    adapter = AudioCppTTSAdapter(_provider_config(client=client))
    await adapter.ensure_initialized()

    response = await adapter.generate(
        TTSRequest(
            text="  hello audio.cpp  ",
            model="audio_cpp:pocket-tts",
            format=AudioFormat.WAV,
            stream=False,
            speed=1.25,
            extra_params={
                "max_tokens": 128,
                "seed": 42,
                "temperature": 0.7,
                "nested": {"ignored": True},
            },
        )
    )

    assert response.audio_data == WAV_BYTES
    assert response.format == AudioFormat.WAV
    assert client.payloads[-1] == {
        "model": "pocket-tts",
        "input": "hello audio.cpp",
        "max_tokens": 128,
        "seed": 42,
    }
    assert response.metadata["ignored_options"] == {
        "temperature": "not_allowlisted",
        "nested": "not_allowlisted",
        "speed": "unsupported",
    }


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_request_returns_full_audio_bytes_for_service_conversion():
    client = _FakeAudioCppClient()
    adapter = AudioCppTTSAdapter(_provider_config(client=client))
    await adapter.ensure_initialized()

    response = await adapter.generate(
        TTSRequest(
            text="hello",
            model="audio-cpp/pocket-tts",
            format=AudioFormat.MP3,
            stream=True,
        )
    )

    assert response.audio_data == WAV_BYTES
    assert response.audio_stream is None
    assert response.format == AudioFormat.WAV
    assert response.metadata["incremental_streaming"] is False


@pytest.mark.unit
@pytest.mark.asyncio
async def test_external_mode_rejects_voice_reference_when_disabled():
    client = _FakeAudioCppClient()
    adapter = AudioCppTTSAdapter(_provider_config(client=client))
    await adapter.ensure_initialized()

    with pytest.raises(TTSValidationError, match="voice_reference"):
        await adapter.generate(
            TTSRequest(
                text="hello",
                model="audio_cpp:pocket-tts",
                format=AudioFormat.WAV,
                stream=False,
                voice_reference=b"RIFF reference audio",
            )
        )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_managed_mode_stages_reference_audio_under_shared_scratch_dir():
    client = _FakeAudioCppClient()
    scratch_dir = Path("models/audio_cpp/runtime/test_scratch")
    adapter = AudioCppTTSAdapter(
        _provider_config(client=client, managed=True, scratch_dir=scratch_dir)
    )
    await adapter.ensure_initialized()

    response = await adapter.generate(
        TTSRequest(
            text="hello",
            model="audio_cpp:pocket-tts",
            format=AudioFormat.WAV,
            stream=False,
            voice_reference=b"RIFF reference audio",
        )
    )

    staged_path = Path(str(client.payloads[-1]["voice_ref"]))
    assert response.audio_data == WAV_BYTES
    assert staged_path.parent == (Path.cwd() / scratch_dir).resolve(strict=False)
    assert staged_path.suffix == ".wav"
    assert not staged_path.exists()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_catalog_only_voice_mapping_requires_reference_audio():
    client = _FakeAudioCppClient()
    adapter = AudioCppTTSAdapter(_provider_config(client=client))
    await adapter.ensure_initialized()

    with pytest.raises(TTSValidationError, match="reference audio"):
        await adapter.generate(
            TTSRequest(
                text="hello",
                voice="alba",
                model="audio_cpp:pocket-tts",
                format=AudioFormat.WAV,
                stream=False,
            )
        )
