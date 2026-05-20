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
        c.headers.update({"X-API-KEY": "test-api-key-1234567890"})
        yield c


@pytest.fixture(autouse=True)
def reset_stt_capabilities_cache():
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_health

    audio_health._STT_CAPABILITIES_CACHE = None
    yield
    audio_health._STT_CAPABILITIES_CACHE = None


@pytest.fixture
def small_catalog(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_health

    monkeypatch.setattr(
        audio_health,
        "get_transcription_models_payload",
        lambda: {
            "categories": {
                "Whisper Models": [
                    {
                        "value": "whisper-small",
                        "label": "Whisper Small",
                        "description": "Balanced speed/accuracy",
                    }
                ],
                "Nemo Models": [
                    {
                        "value": "nemo-canary-1b",
                        "label": "Nemo Canary 1B",
                        "description": "NVIDIA multilingual model",
                    }
                ],
            },
            "all_models": ["whisper-small", "nemo-canary-1b"],
        },
        raising=False,
    )


@pytest.mark.unit
def test_transcription_capabilities_combines_catalog_health_and_provider_sources(
    monkeypatch,
    client: TestClient,
    small_catalog,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import (
        Audio_Transcription_Lib as stt_lib,
    )

    checked_models: list[str] = []

    def fail_if_warmed(*_args, **_kwargs):
        raise AssertionError("capability summary must not warm or load STT models")

    def fake_status(model_name: str):
        checked_models.append(model_name)
        if model_name == "nemo-canary-1b":
            return {
                "available": False,
                "usable": True,
                "on_demand": True,
                "provider": "canary",
                "model": model_name,
                "message": "Canary can initialize on first use.",
            }
        return {
            "available": True,
            "usable": True,
            "on_demand": False,
            "provider": "whisper",
            "model": model_name,
            "message": "Whisper Small is ready.",
        }

    monkeypatch.setattr(stt_lib, "get_whisper_model", fail_if_warmed)
    monkeypatch.setattr(audio_files, "check_transcription_model_status", fake_status)

    response = client.get("/api/v1/audio/transcriptions/capabilities")

    assert response.status_code == 200
    payload = response.json()
    by_id = {model["id"]: model for model in payload["models"]}

    assert checked_models == ["whisper-small", "nemo-canary-1b"]
    assert by_id["whisper-small"] == {
        "id": "whisper-small",
        "label": "Whisper Small",
        "description": "Balanced speed/accuracy",
        "category": "Whisper Models",
        "provider": "faster-whisper",
        "availability": "ready",
        "availability_source": "health",
        "capabilities": {
            "batch": "supported",
            "streaming": "supported",
            "diarization": "supported",
            "timestamps": "supported",
            "segments": "supported",
        },
        "sources": {
            "batch": "provider",
            "streaming": "provider",
            "diarization": "provider",
            "timestamps": "response_schema",
            "segments": "response_schema",
            "label": "static_catalog",
            "description": "static_catalog",
            "availability": "health",
        },
        "message": "Ready",
    }
    assert by_id["nemo-canary-1b"]["provider"] == "canary"
    assert by_id["nemo-canary-1b"]["availability"] == "on_demand"
    assert by_id["nemo-canary-1b"]["capabilities"]["batch"] == "supported"
    assert by_id["nemo-canary-1b"]["capabilities"]["streaming"] == "unsupported"
    assert by_id["nemo-canary-1b"]["capabilities"]["diarization"] == "unsupported"
    assert by_id["nemo-canary-1b"]["sources"]["streaming"] == "provider"


@pytest.mark.unit
def test_transcription_capabilities_use_cache_and_sanitize_status_messages(
    monkeypatch,
    client: TestClient,
    small_catalog,
):
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files

    checked_models: list[str] = []

    def fake_status(model_name: str):
        checked_models.append(model_name)
        return {
            "available": True,
            "usable": False,
            "on_demand": False,
            "provider": "whisper",
            "model": model_name,
            "message": "Traceback from /Users/local/dev/path should stay server-side",
        }

    monkeypatch.setattr(audio_files, "check_transcription_model_status", fake_status)

    first = client.get("/api/v1/audio/transcriptions/capabilities")
    second = client.get("/api/v1/audio/transcriptions/capabilities")

    assert first.status_code == 200
    assert second.status_code == 200
    assert checked_models == ["whisper-small", "nemo-canary-1b"]
    first_model = first.json()["models"][0]
    assert first_model["availability"] == "unavailable"
    assert first_model["message"] == "Unavailable"
    assert "Traceback" not in str(first.json())


@pytest.mark.unit
def test_transcription_capabilities_preserves_unknown_when_provider_metadata_is_missing(
    monkeypatch,
    client: TestClient,
):
    from tldw_Server_API.app.api.v1.endpoints.audio import audio_health
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import Audio_Files as audio_files
    from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio import stt_provider_adapter

    class UnknownRegistry:
        def resolve_provider_for_model(self, _model_name: str):
            return "mystery", "mystery-model", None

        def get_capabilities(self, _provider_name: str):
            raise RuntimeError("provider metadata unavailable")

    monkeypatch.setattr(
        audio_health,
        "get_transcription_models_payload",
        lambda: {
            "categories": {
                "Custom": [
                    {
                        "value": "mystery-model",
                        "label": "Mystery Model",
                    }
                ]
            },
            "all_models": ["mystery-model"],
        },
        raising=False,
    )
    monkeypatch.setattr(stt_provider_adapter, "get_stt_provider_registry", lambda: UnknownRegistry())
    monkeypatch.setattr(audio_files, "check_transcription_model_status", lambda _model_name: {})

    response = client.get("/api/v1/audio/transcriptions/capabilities")

    assert response.status_code == 200
    model = response.json()["models"][0]
    assert model["id"] == "mystery-model"
    assert model["provider"] == "mystery"
    assert model["availability"] == "unknown"
    assert model["capabilities"]["streaming"] == "unknown"
    assert model["capabilities"]["diarization"] == "unknown"
    assert model["sources"]["streaming"] == "unknown"
    assert model["sources"]["diarization"] == "unknown"
