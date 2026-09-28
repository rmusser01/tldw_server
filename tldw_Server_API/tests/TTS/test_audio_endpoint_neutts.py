import os
import base64
import io
import wave
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    # Keep errors as HTTP, not embedded in audio
    monkeypatch.setenv("TTS_STREAM_ERRORS_AS_AUDIO", "0")
    # Clear cached TTS config if present
    try:
        from tldw_Server_API.app.core.TTS.tts_config import get_tts_config_manager

        mgr = get_tts_config_manager()
        mgr._config_cache = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
    reset_settings()
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    # /audio/health requires auth since 0fa12b8c26.
    with TestClient(app, headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}) as c:
        yield c


def _small_wav_bytes(duration_sec: float = 0.2, sr: int = 16000) -> bytes:
    """Generate a tiny valid mono 16-bit WAV of silence using stdlib only."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        n = int(sr * duration_sec)
        wf.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


def test_neutts_endpoint_success(client: TestClient):
    # Skip if NeuTTS provider is not available in this runtime
    h = client.get("/api/v1/audio/health")
    assert h.status_code == 200
    details = (h.json().get("providers") or {}).get("details") or {}
    neutts = details.get("neutts") or {}
    if neutts.get("status") != "available":
        pytest.skip("neutts provider not available in this environment")
    payload = {
        "model": "neutts-air",
        "input": "hello world",
        "response_format": "pcm",
        "stream": False,
        "voice_reference": base64.b64encode(_small_wav_bytes()).decode("ascii"),
        # Provide pre-encoded codes to avoid re-encoding path
        # Engine must still be available for inference
        "extra_params": {"reference_text": "hello world"},
    }
    r = client.post(
        "/api/v1/audio/speech",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    assert r.status_code == 200, r.text
    assert r.content and len(r.content) > 0


def test_neutts_endpoint_missing_reference_text(client: TestClient):
    # Skip if NeuTTS provider is not available in this runtime
    h = client.get("/api/v1/audio/health")
    assert h.status_code == 200
    details = (h.json().get("providers") or {}).get("details") or {}
    neutts = details.get("neutts") or {}
    if neutts.get("status") != "available":
        pytest.skip("neutts provider not available in this environment")
    payload = {
        "model": "neutts-air",
        "input": "hello world",
        "response_format": "pcm",
        "stream": False,
        "voice_reference": base64.b64encode(_small_wav_bytes()).decode("ascii"),
        # Missing extra_params.reference_text
        "extra_params": {},
    }
    r = client.post(
        "/api/v1/audio/speech",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
    )
    # Expect 400 on validation error mapping
    assert r.status_code == 400, r.text
    assert "reference_text" in (r.text or "").lower()


def test_audio_health_includes_capabilities_envelope(client: TestClient):
    r = client.get("/api/v1/audio/health")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "capabilities_envelope" in data
    assert isinstance(data["capabilities_envelope"], list)
