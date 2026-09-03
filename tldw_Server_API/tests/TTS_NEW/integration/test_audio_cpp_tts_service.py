from __future__ import annotations

from pathlib import Path

import pytest

from tldw_Server_API.app.api.v1.schemas.audio_schemas import OpenAISpeechRequest
from tldw_Server_API.app.core.TTS import adapter_registry as adapter_registry_module
from tldw_Server_API.app.core.TTS import tts_service_v2 as tts_service_v2_module
from tldw_Server_API.app.core.TTS.adapter_registry import TTSAdapterFactory, TTSProvider
from tldw_Server_API.app.core.TTS.adapters.audio_cpp_adapter import AudioCppTTSAdapter
from tldw_Server_API.app.core.TTS.adapters.audio_cpp_client import AudioCppSpeechResult
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

WAV_BYTES = b"RIFF$\x00\x00\x00WAVEfmt "


class _FakeAudioCppClient:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    async def health(self) -> dict[str, str]:
        return {"status": "ok"}

    async def list_models(self) -> list[str]:
        return ["pocket-tts"]

    async def speech(self, payload: dict[str, object]) -> AudioCppSpeechResult:
        self.payloads.append(dict(payload))
        return AudioCppSpeechResult(
            audio_bytes=WAV_BYTES,
            content_type="audio/wav",
            metadata={"upstream_latency_ms": 5},
        )

    async def close(self) -> None:
        return None


class _FakeMemoryMonitor:
    def is_memory_critical(self) -> bool:
        return False


class _FakeResourceManager:
    def __init__(self) -> None:
        self.memory_monitor = _FakeMemoryMonitor()
        self.touched: list[tuple[str, object]] = []

    def touch_model(self, provider: str, model: object) -> None:
        self.touched.append((provider, model))


async def _fake_resource_manager() -> _FakeResourceManager:
    return _FakeResourceManager()


def _audio_cpp_provider_config(client: _FakeAudioCppClient, *, enabled: bool = True) -> dict[str, object]:
    return {
        "enabled": enabled,
        "base_url": "http://127.0.0.1:8080",
        "model": "audio-cpp/pocket-tts",
        "model_path": "models/audio_cpp/pocket-tts",
        "sample_rate": 24000,
        "timeout": 300,
        "client": client,
        "extra_params": {
            "managed": False,
            "allow_remote_base_url": False,
            "external_voice_reference_mode": "disabled",
            "request_option_allowlist": ["max_tokens", "seed"],
            "server": {
                "host": "127.0.0.1",
                "port": 8080,
                "models_root": "models/audio_cpp",
                "shared_scratch_dir": str(Path("models/audio_cpp/runtime/test_scratch")),
                "model": {
                    "id": "pocket-tts",
                    "family": "pocket_tts",
                    "path": "models/audio_cpp/pocket-tts",
                    "task": "tts",
                    "mode": "offline",
                },
            },
        },
    }


def _factory_config(client: _FakeAudioCppClient, *, enabled: bool = True) -> dict[str, object]:
    return {
        "provider_priority": ["pocket_tts", "audio_cpp"],
        "performance": {
            "token_estimation_enabled": False,
            "max_concurrent_generations": 1,
        },
        "providers": {
            "audio_cpp": _audio_cpp_provider_config(client, enabled=enabled),
            "pocket_tts": {"enabled": False},
        },
    }


@pytest.fixture(autouse=True)
def _stub_resource_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        adapter_registry_module,
        "get_resource_manager",
        _fake_resource_manager,
        raising=True,
    )
    monkeypatch.setattr(
        tts_service_v2_module,
        "get_resource_manager",
        _fake_resource_manager,
        raising=True,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_factory_routes_audio_cpp_explicit_provider_and_namespaced_model():
    client = _FakeAudioCppClient()
    factory = TTSAdapterFactory(_factory_config(client))

    explicit_adapter = await factory.registry.get_adapter("audio_cpp")
    model_adapter = await factory.get_adapter_by_model("audio-cpp/pocket-tts")

    assert isinstance(explicit_adapter, AudioCppTTSAdapter)
    assert model_adapter is explicit_adapter
    assert factory.get_provider_for_model("audio-cpp/pocket-tts") == TTSProvider.AUDIO_CPP
    assert factory.get_provider_for_model("pocket-tts") == TTSProvider.POCKET_TTS


@pytest.mark.integration
@pytest.mark.asyncio
async def test_disabled_audio_cpp_provider_does_not_initialize_by_default():
    client = _FakeAudioCppClient()
    factory = TTSAdapterFactory(_factory_config(client, enabled=False))

    assert await factory.registry.get_adapter("audio_cpp") is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generate_speech_uses_audio_cpp_adapter_for_namespaced_model():
    client = _FakeAudioCppClient()
    factory = TTSAdapterFactory(_factory_config(client))
    service = TTSServiceV2(factory=factory)
    request = OpenAISpeechRequest(
        model="audio-cpp/pocket-tts",
        input="hello from service",
        voice="af_heart",
        response_format="wav",
        stream=True,
        extra_params={"seed": 9},
    )

    try:
        chunks = [chunk async for chunk in service.generate_speech(request, fallback=False)]
    finally:
        await service.shutdown()

    assert chunks == [WAV_BYTES]
    assert client.payloads[-1] == {
        "model": "pocket-tts",
        "input": "hello from service",
        "seed": 9,
    }
    metadata = request._tts_metadata
    assert metadata["provider"] == "audio_cpp"
    assert metadata["model"] == "audio-cpp/pocket-tts"
    assert metadata["incremental_streaming"] is False
