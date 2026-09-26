import pytest
from fastapi import FastAPI, HTTPException, Request
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
def test_transcriptions_health_rejects_anonymous_warm_up(monkeypatch, client: TestClient):
    """An anonymous health probe must not download or initialize a model."""
    import tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib as atlib

    calls = []
    monkeypatch.setattr(atlib, "get_whisper_model", lambda *args, **kwargs: calls.append((args, kwargs)))

    response = client.get("/api/v1/audio/transcriptions/health", params={"model": "whisper-1", "warm": "true"})
    assert response.status_code == 401
    assert calls == []


@pytest.mark.asyncio
async def test_transcriptions_health_rejects_non_admin_warm_up(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_health
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

    async def non_admin(_request):
        return AuthPrincipal(kind="user", user_id=2)

    monkeypatch.setattr(audio_health, "get_auth_principal", non_admin)
    request = Request({"type": "http", "method": "GET", "path": "/api/v1/audio/transcriptions/health", "headers": []})
    with pytest.raises(HTTPException) as exc_info:
        await audio_health._authorize_stt_health_warm(request, warm=True)
    assert exc_info.value.status_code == 403


@pytest.mark.unit
@pytest.mark.parametrize("method,path", [
    ("GET", "/api/v1/audio/health"),
    ("GET", "/api/v1/audio/providers"),
    ("GET", "/api/v1/audio/tts/providers/openai/model-info"),
    ("GET", "/api/v1/audio/voices/catalog"),
    ("POST", "/api/v1/audio/reset-metrics"),
    ("POST", "/api/v1/audio/stream/test"),
])
def test_model_initializing_diagnostics_require_auth(client: TestClient, method: str, path: str):
    response = client.request(method, path)
    assert response.status_code == 401


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

    r = client.get(
        "/api/v1/audio/transcriptions/health",
        params={"model": "whisper-1", "warm": "true"},
        headers={"X-API-KEY": "test-api-key-1234567890"},
    )
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
