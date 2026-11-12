import time
import traceback
from collections.abc import AsyncIterator, Iterator
from contextlib import suppress
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints import audio as audio_endpoints
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.AuthNZ.settings import get_settings
from tldw_Server_API.app.main import app


pytestmark = pytest.mark.integration


class _RecordingKittenTTSService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.provider_override_load_failures: list[dict[str, object]] = []

    def generate_speech(
        self,
        request_data: Any,
        provider: str | None = None,
        fallback: bool = True,
        provider_overrides: dict[str, Any] | None = None,
        voice_to_voice_start: float | None = None,
        voice_to_voice_route: str = "audio.speech",
        user_id: int | str | None = None,
        request_id: str | None = None,
        metadata_only: bool = False,
    ) -> AsyncIterator[bytes]:
        self.calls.append(
            {
                "request_data": request_data,
                "provider": provider,
                "fallback": fallback,
                "provider_overrides": provider_overrides,
                "voice_to_voice_start": voice_to_voice_start,
                "voice_to_voice_route": voice_to_voice_route,
                "user_id": user_id,
                "request_id": request_id,
                "metadata_only": metadata_only,
            }
        )

        async def _gen() -> AsyncIterator[bytes]:
            yield b"kitten-audio"

        return _gen()


@pytest.fixture()
def _client_with_kitten_overrides(
    bypass_api_limits: Any, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[TestClient, _RecordingKittenTTSService]]:
    async def _override_user() -> User:
        return User(id=1, username="tester", email="tester@example.com", is_active=True)

    service = _RecordingKittenTTSService()

    async def _override_tts_service() -> _RecordingKittenTTSService:
        return service

    from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides

    list_overrides = llm_provider_overrides.AuthnzLLMProviderOverridesRepo.list_overrides

    async def observed_overrides(repo, *args, **kwargs):
        started = time.monotonic()
        try:
            return await list_overrides(repo, *args, **kwargs)
        except BaseException as exc:
            if len(service.provider_override_load_failures) < 3:
                service.provider_override_load_failures.append(
                    {
                        "type": type(exc).__name__,
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "frames": [
                            (frame.name, frame.lineno) for frame in traceback.extract_tb(exc.__traceback__)[-8:]
                        ],
                    }
                )
            raise

    monkeypatch.setattr(llm_provider_overrides.AuthnzLLMProviderOverridesRepo, "list_overrides", observed_overrides)

    original_overrides = dict(app.dependency_overrides)
    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[audio_endpoints.get_tts_service] = _override_tts_service

    try:
        with bypass_api_limits(app), TestClient(app) as client:
            yield client, service
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)


def test_audio_speech_routes_kitten_tts_through_current_backend_path(
    _client_with_kitten_overrides: tuple[TestClient, _RecordingKittenTTSService],
) -> None:
    client, service = _client_with_kitten_overrides
    settings = get_settings()

    response = client.post(
        "/api/v1/audio/speech",
        json={
            "model": "kitten_tts",
            "input": "Hello from kitten TTS",
            "voice": "kitten_voice",
            "response_format": "mp3",
            "stream": False,
        },
        headers={"X-API-KEY": settings.SINGLE_USER_API_KEY},
    )

    if response.status_code != 200:
        # Failure diagnostics must preserve the original response assertion.
        with suppress(Exception):
            from tldw_Server_API.app.core.Provider_Credentials import llm_provider_overrides

            with llm_provider_overrides._OVERRIDE_LOCK:
                print(
                    "Kitten TTS provider readiness:",
                    {
                        "load_failures": service.provider_override_load_failures,
                        "cache_healthy": llm_provider_overrides._OVERRIDE_CACHE_HEALTHY,
                        "refresh_pending": llm_provider_overrides._OVERRIDE_RECOVERY_IN_FLIGHT,
                        "ttl_disabled": llm_provider_overrides._OVERRIDE_CACHE_TTL_DISABLED_FOR_TESTS,
                        "response_status": response.status_code,
                    },
                )
    assert response.status_code == 200, response.text
    assert response.content == b"kitten-audio"
    assert response.headers.get("content-type", "").startswith("audio/mpeg")
    assert len(service.calls) == 1
    call = service.calls[0]
    assert call["provider"] == "kitten_tts"
    assert getattr(call["request_data"], "model", None) == "kitten_tts"
    assert getattr(call["request_data"], "input", None) == "Hello from kitten TTS"
