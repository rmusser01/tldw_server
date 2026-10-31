import os
import io
import wave
import asyncio
import base64
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


def _small_wav_bytes(duration_sec: float = 0.25, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * int(sr * duration_sec))
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    # Ensure we do not embed errors as audio for assertions
    monkeypatch.setenv("TTS_STREAM_ERRORS_AS_AUDIO", "0")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    # /audio/health requires auth since 0fa12b8c26.
    with TestClient(app, headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}) as c:
        yield c


def _is_neutts_streaming_available(client: TestClient) -> bool:
    # Provider must be available
    h = client.get("/api/v1/audio/health")
    if h.status_code != 200:
        return False
    details = (h.json().get("providers") or {}).get("details") or {}
    val = details.get("neutts")
    if isinstance(val, dict):
        if val.get("status") != "available":
            return False
    else:
        if val != "available":
            return False
    # Adapter must indicate streaming support (GGUF)
    try:
        from tldw_Server_API.app.core.TTS.adapter_registry import get_tts_factory, TTSProvider

        factory = asyncio.run(get_tts_factory())
        adapter = asyncio.run(factory.registry.get_adapter(TTSProvider.NEUTTS))
        return bool(adapter) and bool(getattr(adapter, "_supports_streaming", False))
    except Exception:
        return False


def test_neutts_streaming_when_available(client: TestClient):
    if not _is_neutts_streaming_available(client):
        pytest.skip("NeuTTS streaming (GGUF + llama-cpp) not available in this environment")

    payload = {
        "model": "neutts-air-q8-gguf",  # model hint; adapter selection uses mapping
        "input": "streaming test",
        "response_format": "pcm",
        "stream": True,
        "voice_reference": base64.b64encode(_small_wav_bytes()).decode("ascii"),
        "extra_params": {"reference_text": "streaming test"},
    }
    r = client.post(
        "/api/v1/audio/speech",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        stream=True,
    )
    # TestClient buffers streaming responses; still, body should be non-empty on success
    assert r.status_code == 200, r.text
    body = b"".join(r.iter_bytes()) if hasattr(r, "iter_bytes") else r.content
    assert body and len(body) > 0
    # Spot-check: content-type is audio (PCM expected)
    ct = r.headers.get("content-type", "")
    assert "audio" in ct
