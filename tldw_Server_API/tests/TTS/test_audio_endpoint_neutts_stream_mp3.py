import os
import io
import wave
import base64
import asyncio
import pytest
import numpy as np

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


def _neutts_streaming_available(client: TestClient) -> bool:
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
    try:
        from tldw_Server_API.app.core.TTS.adapter_registry import get_tts_factory, TTSProvider

        factory = asyncio.run(get_tts_factory())
        adapter = asyncio.run(factory.registry.get_adapter(TTSProvider.NEUTTS))
        return bool(adapter) and bool(getattr(adapter, "_supports_streaming", False))
    except Exception:
        return False


def _mp3_supported() -> bool:
    try:
        from tldw_Server_API.app.core.TTS.streaming_audio_writer import StreamingAudioWriter

        writer = StreamingAudioWriter(format="mp3", sample_rate=24000, channels=1)
        # write a tiny chunk and finalize
        writer.write_chunk(np.zeros(2400, dtype=np.int16))
        data = writer.write_chunk(finalize=True)
        writer.close()
        return data is not None
    except Exception:
        return False


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    # Ensure errors are not embedded as audio bytes
    monkeypatch.setenv("TTS_STREAM_ERRORS_AS_AUDIO", "0")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    # /audio/health requires auth since 0fa12b8c26.
    with TestClient(app, headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]}) as c:
        yield c


def test_neutts_streaming_mp3_when_available(client: TestClient):
    if not _neutts_streaming_available(client):
        pytest.skip("NeuTTS streaming (GGUF + llama-cpp) not available in this environment")
    if not _mp3_supported():
        pytest.skip("PyAV/FFmpeg MP3 encoder not available in this environment")

    payload = {
        "model": "neutts-air-q8-gguf",
        "input": "streaming mp3 test",
        "response_format": "mp3",
        "stream": True,
        "voice_reference": base64.b64encode(_small_wav_bytes()).decode("ascii"),
        "extra_params": {"reference_text": "streaming mp3 test"},
    }
    r = client.post(
        "/api/v1/audio/speech",
        json=payload,
        headers={"X-API-KEY": os.environ["SINGLE_USER_API_KEY"]},
        stream=True,
    )
    assert r.status_code == 200, r.text
    body = b"".join(r.iter_bytes()) if hasattr(r, "iter_bytes") else r.content
    assert body and len(body) > 0
    ct = r.headers.get("content-type", "")
    assert "audio/mpeg" in ct or "audio" in ct
