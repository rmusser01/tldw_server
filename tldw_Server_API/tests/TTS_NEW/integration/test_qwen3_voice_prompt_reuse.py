import base64
import io
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
from tldw_Server_API.app.core.TTS.adapters.qwen3_tts_adapter import Qwen3TTSAdapter
from tldw_Server_API.app.core.TTS.tts_service_v2 import TTSServiceV2
from tldw_Server_API.app.core.TTS.voice_manager import VoiceReferenceMetadata


def _make_wav_bytes() -> bytes:
    buf = io.BytesIO()
    audio = np.zeros(int(24000 * 3.1), dtype=np.float32)
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


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


def test_qwen3_custom_voice_reuses_prompt_metadata(client, monkeypatch):
    holder = {"calls": []}

    module = types.ModuleType("qwen_tts")

    class FakeBackend:
        def generate_voice_clone(
            self,
            text,
            language=None,
            ref_audio=None,
            ref_text=None,
            x_vector_only_mode=False,
            voice_clone_prompt=None,
            **_kwargs,
        ):
            holder["calls"].append(
                {
                    "text": text,
                    "language": language,
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                    "x_vector_only_mode": x_vector_only_mode,
                    "voice_clone_prompt": voice_clone_prompt,
                }
            )
            return np.zeros(160, dtype=np.int16)

    class FakeQwen3TTS:
        @classmethod
        def from_pretrained(cls, model_id, **_kwargs):
            return FakeBackend()

    module.Qwen3TTS = FakeQwen3TTS
    monkeypatch.setitem(sys.modules, "qwen_tts", module)

    prompt_bytes = b"PROMPT_BYTES"
    prompt_b64 = base64.b64encode(prompt_bytes).decode("ascii")
    voice_bytes = _make_wav_bytes()

    class _FakeVoiceManager:
        async def load_voice_reference_audio(self, user_id, voice_id):
            assert str(user_id) == "1"
            assert voice_id == "voice-1"
            return voice_bytes

        async def load_reference_metadata(self, user_id, voice_id):
            return VoiceReferenceMetadata(
                voice_id=voice_id,
                reference_text="stored transcript",
                voice_clone_prompt_b64=prompt_b64,
                voice_clone_prompt_format="qwen3_tts_prompt_v1",
            )

    def _fake_get_voice_manager():
        return _FakeVoiceManager()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.voice_manager.get_voice_manager",
        _fake_get_voice_manager,
        raising=True,
    )

    adapter = Qwen3TTSAdapter({"device": "cpu", "model": "auto", "runtime": "upstream"})
    adapter._resolve_torch_dtype = lambda: "float32"

    class _FakeFactory:
        def get_provider_for_model(self, _model):
            return "qwen3_tts"

    service = TTSServiceV2()
    service._ensure_factory = AsyncMock(return_value=_FakeFactory())
    service._get_adapter = AsyncMock(return_value=adapter)
    monkeypatch.setattr(
        service,
        "_get_validation_config",
        lambda: {"providers": {"qwen3_tts": {"runtime": "upstream"}}},
        raising=True,
    )

    async def _fake_get_tts_service_v2():
        return service

    monkeypatch.setattr(
        "tldw_Server_API.app.core.TTS.tts_service_v2.get_tts_service_v2",
        _fake_get_tts_service_v2,
        raising=True,
    )

    client.app.dependency_overrides[audio_endpoints.get_tts_service] = _fake_get_tts_service_v2

    payload = {
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "input": "Hello world",
        "voice": "custom:voice-1",
        "response_format": "pcm",
        "stream": False,
    }
    try:
        r = client.post(
            "/api/v1/audio/speech",
            json=payload,
            headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        )
        assert r.status_code == 200, r.text

        assert holder["calls"]
        call = holder["calls"][-1]
        assert call["ref_audio"] == base64.b64encode(voice_bytes).decode("ascii")
        assert call["voice_clone_prompt"] == prompt_bytes
    finally:
        client.app.dependency_overrides.pop(audio_endpoints.get_tts_service, None)
