"""Unit tests for Audio Studio provider adapters and registry."""

from __future__ import annotations

import json

import httpx
import pytest

from tldw_Server_API.app.core.Audio_Studio.models import AudioGenerationRequest
from tldw_Server_API.app.core.Audio_Studio.providers.ace_step import AceStepHttpAdapter
from tldw_Server_API.app.core.Audio_Studio.providers.registry import (
    AudioStudioProviderRegistry,
    build_audio_studio_provider_registry,
)
from tldw_Server_API.app.core.Audio_Studio.providers.speech import SpeechTtsAdapter


pytestmark = pytest.mark.unit


def _request(kind: str = "speech") -> AudioGenerationRequest:
    return AudioGenerationRequest(
        workflow="narration",
        kind=kind,
        prompt="ambient theme",
        text="Hello studio",
        provider_options={"voice": "af_heart", "format": "mp3"},
        target_resource_kind="section",
        target_resource_id="sec_001",
        target_revision_id="rev_001",
    )


def test_registry_includes_speech_and_rejects_unsupported_kind() -> None:
    registry = AudioStudioProviderRegistry([SpeechTtsAdapter(tts_service_factory=lambda: None)])

    assert registry.list_providers()[0]["provider_id"] == "tts"
    assert registry.get_adapter("tts", "speech").provider_id == "tts"

    with pytest.raises(ValueError, match="unsupported_audio_generation_kind"):
        registry.get_adapter("tts", "music")
    with pytest.raises(KeyError, match="audio_studio_provider_not_found"):
        registry.get_adapter("missing", "speech")


def test_configured_registry_adds_ace_step_only_when_allowlisted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", raising=False)
    assert [row["provider_id"] for row in build_audio_studio_provider_registry().list_providers()] == ["tts"]

    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "https://ace.example.test")
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://ace.example.test")
    providers = build_audio_studio_provider_registry().list_providers()

    assert [row["provider_id"] for row in providers] == ["tts", "ace_step"]
    assert providers[1]["supported_kinds"] == ["music"]
    assert "api_key" not in json.dumps(providers).lower()


@pytest.mark.asyncio
async def test_speech_adapter_wraps_tts_service_without_real_engine() -> None:
    class _FakeTtsService:
        async def generate_speech(self, request, *, provider, fallback, user_id):
            assert request.input == "Hello studio"
            assert request.voice == "af_heart"
            assert request.response_format == "mp3"
            assert provider == "kokoro"
            assert fallback is True
            assert user_id == 42
            yield b"abc"
            yield b"123"

    adapter = SpeechTtsAdapter(tts_service_factory=lambda: _FakeTtsService())
    result = await adapter.generate(_request(), user_id=42, provider_hint="kokoro")

    assert result.provider == "tts"
    assert result.mime_type == "audio/mpeg"
    assert result.content_bytes == b"abc123"
    assert result.metadata["format"] == "mp3"


@pytest.mark.asyncio
async def test_ace_step_adapter_uses_configured_endpoint_and_runtime_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_BASE_URL", "https://ace.example.test")
    monkeypatch.setenv("AUDIO_STUDIO_EXTERNAL_ENDPOINT_ALLOWLIST", "https://ace.example.test")
    monkeypatch.setenv("AUDIO_STUDIO_ACE_STEP_API_KEY", "secret-runtime-key")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://ace.example.test/generate"
        assert request.headers["authorization"] == "Bearer secret-runtime-key"
        payload = json.loads(request.content.decode("utf-8"))
        assert payload["prompt"] == "ambient theme"
        assert "api_key" not in payload
        return httpx.Response(
            200,
            headers={"content-type": "audio/wav"},
            content=b"RIFF",
        )

    adapter = AceStepHttpAdapter(client_factory=lambda **kwargs: httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs))
    result = await adapter.generate(_request("music"))

    assert result.provider == "ace_step"
    assert result.mime_type == "audio/wav"
    assert result.content_bytes == b"RIFF"
