import os
import sys
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

if sys.version_info < (3, 10):
    pytest.skip("Supertonic2 audio endpoint integration tests require Python 3.10+", allow_module_level=True)

from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.TTS.adapters.base import (
    AudioFormat,
    ProviderStatus,
    TTSAdapter,
    TTSCapabilities,
    TTSRequest,
    TTSResponse,
    VoiceInfo,
)
from tldw_Server_API.app.core.TTS.adapter_registry import TTSProvider
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2

pytestmark = [pytest.mark.integration]


class _MetricsStub:
    def register_metric(self, *args, **kwargs):
        return None

    def set_gauge(self, *args, **kwargs):
        return None

    def increment(self, *args, **kwargs):
        return None

    def observe(self, *args, **kwargs):
        return None

    def gauge_add(self, *args, **kwargs):
        return None


class _RecordingSupertonic2Adapter(TTSAdapter):
    PROVIDER_KEY = "supertonic2"

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.last_language: Optional[str] = None

    async def initialize(self) -> bool:
        self._initialized = True
        self._status = ProviderStatus.AVAILABLE
        self._capabilities = await self.get_capabilities()
        return True

    async def generate(self, request: TTSRequest) -> TTSResponse:
        self.last_language = request.language
        return TTSResponse(audio_data=b"ok", format=request.format, sample_rate=24000)

    async def get_capabilities(self) -> TTSCapabilities:
        return TTSCapabilities(
            provider_name="Supertonic2",
            supports_streaming=True,
            supports_voice_cloning=False,
            supported_languages={"en", "ko", "es", "pt", "fr"},
            supported_formats={AudioFormat.MP3, AudioFormat.WAV},
            max_text_length=15000,
            supported_voices=[
                VoiceInfo(
                    id="supertonic2_m1",
                    name="Supertonic2 M1",
                    language="multi",
                )
            ],
        )


@pytest.fixture
def client(monkeypatch, healthy_no_override_tts_credential_snapshot):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    with TestClient(app) as c:
        yield c


def test_speech_supertonic2_regional_lang_code_normalized(client, monkeypatch):
    adapter = _RecordingSupertonic2Adapter()

    class _FakeFactory:
        def get_provider_for_model(self, _model):
            return TTSProvider.SUPERTONIC2

    service = TTSServiceV2()
    service.metrics = _MetricsStub()
    service._ensure_factory = AsyncMock(return_value=_FakeFactory())
    service._get_adapter = AsyncMock(return_value=adapter)

    async def _fake_get_tts_service():
        return service

    client.app.dependency_overrides[audio_endpoints.get_tts_service] = _fake_get_tts_service
    try:
        payload = {
            "model": "tts-supertonic2-1",
            "input": "Ola mundo",
            "voice": "supertonic2_m1",
            "response_format": "mp3",
            "stream": False,
            "lang_code": "pt-BR",
        }
        response = client.post(
            "/api/v1/audio/speech",
            json=payload,
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
        assert response.status_code == 200, response.text
        assert response.content == b"ok"
        assert adapter.last_language == "pt"
    finally:
        client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)


def test_speech_supertonic2_extra_language_override_normalized(client, monkeypatch):
    adapter = _RecordingSupertonic2Adapter()

    class _FakeFactory:
        def get_provider_for_model(self, _model):
            return TTSProvider.SUPERTONIC2

    service = TTSServiceV2()
    service.metrics = _MetricsStub()
    service._ensure_factory = AsyncMock(return_value=_FakeFactory())
    service._get_adapter = AsyncMock(return_value=adapter)

    async def _fake_get_tts_service():
        return service

    client.app.dependency_overrides[audio_endpoints.get_tts_service] = _fake_get_tts_service
    try:
        payload = {
            "model": "tts-supertonic2-1",
            "input": "Bonjour monde",
            "voice": "supertonic2_m1",
            "response_format": "mp3",
            "stream": False,
            "lang_code": "en-US",
            "extra_params": {"language": "fr-CA"},
        }
        response = client.post(
            "/api/v1/audio/speech",
            json=payload,
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
        assert response.status_code == 200, response.text
        assert response.content == b"ok"
        assert adapter.last_language == "fr"
    finally:
        client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)


def test_voices_catalog_includes_supertonic2_when_available(client):
    class _VoiceCatalogService:
        async def list_voices(self):
            return {
                "supertonic2": [
                    {"id": "supertonic2_m1", "name": "Supertonic2 M1", "language": "multi"}
                ]
            }

    async def _fake_get_tts_service():
        return _VoiceCatalogService()

    client.app.dependency_overrides[audio_endpoints.get_tts_service] = _fake_get_tts_service
    try:
        response = client.get(
            "/api/v1/audio/voices/catalog",
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert "supertonic2" in data
        assert data["supertonic2"]
    finally:
        client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)
