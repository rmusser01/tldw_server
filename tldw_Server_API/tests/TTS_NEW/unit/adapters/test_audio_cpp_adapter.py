from __future__ import annotations

import io
import wave
from pathlib import Path

import pytest

from tldw_Server_API.app.core.TTS.adapters.audio_cpp_adapter import AudioCppTTSAdapter
from tldw_Server_API.app.core.TTS.adapters.audio_cpp_client import AudioCppSpeechResult
from tldw_Server_API.app.core.TTS.adapters.base import AudioFormat, TTSRequest
from tldw_Server_API.app.core.TTS.tts_exceptions import TTSValidationError


def _wav_bytes(*, rate=24000, channels=1, frames=b"\x01\x00" * 100):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setparams((channels, 2, rate, 0, "NONE", "not compressed"))
        output.writeframes(frames)
    return buffer.getvalue()


WAV_BYTES = _wav_bytes()


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
    assert {AudioFormat.OGG, AudioFormat.WEBM, AudioFormat.ULAW}.isdisjoint(capabilities.supported_formats)
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

    with pytest.raises(TTSValidationError, match="Voice cloning"):
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
    adapter = AudioCppTTSAdapter(_provider_config(client=client, managed=True, scratch_dir=scratch_dir))
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "managed,mode,expected", [(False, "disabled", False), (False, "shared_path", True), (True, "disabled", True)]
)
async def test_voice_cloning_capability_matches_reference_mode(managed, mode, expected):
    adapter = AudioCppTTSAdapter(
        _provider_config(client=_FakeAudioCppClient(), managed=managed, external_voice_reference_mode=mode)
    )
    assert (await adapter.get_capabilities()).supports_voice_cloning is expected


@pytest.mark.unit
async def test_reference_file_is_private_and_cleaned_after_provider_failure(tmp_path):
    import os
    import stat

    from tldw_Server_API.app.core.TTS.tts_exceptions import TTSGenerationError

    class InspectClient(_FakeAudioCppClient):
        path = None
        permissions = None

        async def speech(self, payload):
            self.path = Path(payload["voice_ref"])
            self.permissions = stat.S_IMODE(self.path.stat().st_mode)
            raise TTSGenerationError("upstream failed")

    client = InspectClient()
    config = _provider_config(client=client, managed=True, scratch_dir=tmp_path / "scratch")
    config["extra_params"]["server"]["models_root"] = str(tmp_path)
    adapter = AudioCppTTSAdapter(config)
    previous = os.umask(0)
    try:
        with pytest.raises(TTSGenerationError):
            await adapter.generate(TTSRequest(text="hello", voice_reference=WAV_BYTES))
    finally:
        os.umask(previous)
    assert not client.path.exists()
    if os.name != "nt":
        assert client.permissions == 0o600


@pytest.mark.unit
async def test_pcm_request_returns_frames_and_actual_wav_metadata():
    frames = b"\x01\x00\x02\x00" * 100
    client = _FakeAudioCppClient(audio_bytes=_wav_bytes(rate=48000, channels=2, frames=frames))
    adapter = AudioCppTTSAdapter(_provider_config(client=client))
    response = await adapter.generate(TTSRequest(text="hello", format=AudioFormat.PCM))
    assert response.audio_data == b"\x01\x00" * 100
    assert (response.format, response.sample_rate, response.channels) == (AudioFormat.PCM, 48000, 1)


@pytest.mark.unit
async def test_default_request_uses_configured_server_model_id():
    client = _FakeAudioCppClient(models=["custom-pocket"])
    config = _provider_config(client=client)
    config["extra_params"]["server"]["model"]["id"] = "custom-pocket"
    adapter = AudioCppTTSAdapter(config)
    await adapter.generate(TTSRequest(text="hello", format=AudioFormat.WAV))
    assert client.payloads[-1]["model"] == "custom-pocket"


@pytest.mark.unit
async def test_cancelled_reference_staging_removes_completed_file(tmp_path, monkeypatch):
    import asyncio
    import threading

    started = threading.Event()
    finish = threading.Event()
    config = _provider_config(client=_FakeAudioCppClient(), managed=True, scratch_dir=tmp_path / "scratch")
    config["extra_params"]["server"]["models_root"] = str(tmp_path)
    adapter = AudioCppTTSAdapter(config)
    write = adapter._write_reference_audio_sync

    def slow_write(path, data):
        started.set()
        finish.wait(timeout=5)
        write(path, data)

    monkeypatch.setattr(adapter, "_write_reference_audio_sync", slow_write)
    task = asyncio.create_task(adapter.generate(TTSRequest(text="hello", voice_reference=WAV_BYTES)))
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        # Drain the staging executor before checking for a file written after cancellation.
        await asyncio.to_thread(lambda: None)
        await asyncio.sleep(0.02)
        assert list((tmp_path / "scratch").glob("*.wav")) == []
    finally:
        finish.set()


@pytest.mark.unit
async def test_managed_generation_restarts_sidecar_and_refreshes_client(monkeypatch):
    from tldw_Server_API.app.core.TTS.adapters import audio_cpp_adapter as module

    class Supervisor:
        url = "http://127.0.0.1:8080"

        async def ensure_started(self):
            return self.url

    clients = []

    class Client(_FakeAudioCppClient):
        def __init__(self, *, base_url, **kwargs):
            super().__init__()
            self.base_url = base_url
            clients.append(self)

    config = _provider_config(client=None, managed=True)
    supervisor = Supervisor()
    config["sidecar_supervisor"] = supervisor
    monkeypatch.setattr(module, "AudioCppClient", Client)
    adapter = AudioCppTTSAdapter(config)
    await adapter.generate(TTSRequest(text="first"))
    supervisor.url = "http://127.0.0.1:8081"
    await adapter.generate(TTSRequest(text="second"))
    assert clients[0].closed
    assert clients[-1].base_url == supervisor.url
    assert clients[-1].payloads[-1]["input"] == "second"
    await adapter.close()


@pytest.mark.unit
async def test_wav_response_reports_actual_rate_and_channels():
    client = _FakeAudioCppClient(audio_bytes=_wav_bytes(rate=48000, channels=2))
    adapter = AudioCppTTSAdapter(_provider_config(client=client))
    response = await adapter.generate(TTSRequest(text="hello", format=AudioFormat.WAV))
    assert (response.sample_rate, response.channels) == (48000, 2)


@pytest.mark.unit
async def test_configured_native_voice_mapping_is_sent_upstream():
    client = _FakeAudioCppClient()
    config = _provider_config(client=client)
    config["extra_params"]["voices"]["alba"]["request_field"] = "voice"
    adapter = AudioCppTTSAdapter(config)
    await adapter.generate(TTSRequest(text="hello", voice="alba", format=AudioFormat.WAV))
    assert client.payloads[-1]["voice"] == "alba"
