import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio.audio import router as audio_router


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    app = FastAPI()
    app.include_router(audio_router, prefix="/api/v1/audio")
    with TestClient(app) as c:
        yield c


@pytest.mark.unit
def test_transcriptions_health_basic_status(client: TestClient):
    """
    The STT health endpoint should respond with a basic status payload
    even when models are not yet downloaded.
    """
    r = client.get("/api/v1/audio/transcriptions/health")
    assert r.status_code == 200
    data = r.json()
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Transcription_Lib as stt_lib,
    )
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
        resolve_default_transcription_model,
    )

    default_model = resolve_default_transcription_model("whisper-1")
    provider_raw, _, _ = stt_lib.parse_transcription_model(default_model)
    assert data.get("provider") == provider_raw
    assert "model" in data
    assert "available" in data


@pytest.mark.unit
def test_transcriptions_health_warm_uses_whisper_model(monkeypatch, client: TestClient):
    """
    When warm=true and the provider is Whisper, the endpoint should attempt
    to initialize the underlying faster-whisper model via get_whisper_model,
    and report a warm.ok flag without raising even if initialization fails.
    """
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    calls = {}

    def fake_get_whisper_model(model_name, device, check_download_status=False):

        calls["model_name"] = model_name
        calls["device"] = device
        calls["check_download_status"] = check_download_status
        # Return a lightweight sentinel object; STT health does not inspect it.
        return object()

    monkeypatch.setattr(atlib, "get_whisper_model", fake_get_whisper_model)

    r = client.get("/api/v1/audio/transcriptions/health", params={"model": "whisper-1", "warm": "true"})
    assert r.status_code == 200
    data = r.json()

    assert data.get("provider") == "whisper"
    warm = data.get("warm") or {}
    assert warm.get("ok") is True
    assert calls.get("model_name") is not None
    assert data.get("available") is True
    assert data.get("usable") is True


@pytest.mark.unit
def test_transcriptions_health_exposes_usable_for_non_whisper(monkeypatch, client: TestClient):
    """
    Non-Whisper providers can be usable even when they report not-locally-ready.
    Surface `usable` explicitly so clients can avoid hard-disabling dictation.
    """
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

    def fake_status(_model_name: str):
        return {
            "available": False,
            "usable": True,
            "provider": "parakeet",
            "model": "parakeet-mlx",
            "message": "Parakeet can initialize on first use.",
        }

    monkeypatch.setattr(audio_files, "check_transcription_model_status", fake_status)

    r = client.get("/api/v1/audio/transcriptions/health", params={"model": "parakeet-mlx"})
    assert r.status_code == 200
    data = r.json()

    assert data.get("provider") == "parakeet"
    assert data.get("available") is False
    assert data.get("usable") is True


@pytest.mark.unit
def test_transcriptions_health_treats_on_demand_whisper_as_usable(monkeypatch, client: TestClient):
    """Whisper models pending first-use download should still be reported as request-usable."""
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

    def fake_status(_model_name: str):
        return {
            "available": False,
            "usable": True,
            "provider": "whisper",
            "model": "large-v3",
            "on_demand": True,
            "message": "Model large-v3 will download on first use.",
        }

    monkeypatch.setattr(audio_files, "check_transcription_model_status", fake_status)

    r = client.get("/api/v1/audio/transcriptions/health", params={"model": "whisper-large-v3"})
    assert r.status_code == 200
    data = r.json()

    assert data.get("provider") == "whisper"
    assert data.get("available") is False
    assert data.get("usable") is True
    assert data.get("on_demand") is True


@pytest.mark.unit
def test_transcriptions_health_reports_unavailable_when_stt_deps_missing(monkeypatch, client: TestClient):
    """A missing STT/media dependency must yield a well-formed unavailable status, not a 500.

    Audio_Files imports yt_dlp at module level; on an install without it the probe the
    chat page fires on every load used to raise ImportError and return 500.
    """
    import sys

    from tldw_Server_API.app.core.Ingestion_Media_Processing import Audio as audio_pkg

    monkeypatch.delattr(audio_pkg, "Audio_Files", raising=False)
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Files", None)

    r = client.get("/api/v1/audio/transcriptions/health")
    assert r.status_code == 200
    data = r.json()
    assert data["available"] is False
    assert data["usable"] is False
    assert "not available" in data["message"]
