import pytest
from fastapi import FastAPI, HTTPException
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from tldw_Server_API.app.api.v1.endpoints.audio import audio
from tldw_Server_API.app.api.v1.endpoints.audio import audio_voices
from tldw_Server_API.app.core.AuthNZ.settings import reset_settings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "test-api-key-1234567890")
    monkeypatch.setenv("SINGLE_USER_FIXED_ID", "1")
    reset_settings()
    app = FastAPI()
    app.include_router(audio_voices.router, prefix="/api/v1/audio")

    async def _deny_rate_limit():
        raise HTTPException(status_code=429, detail="rate limited in test")

    app.dependency_overrides[audio_voices.check_rate_limit] = _deny_rate_limit
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.pop(audio_voices.check_rate_limit, None)


def test_voice_list_route_enforces_rate_limit_dependency(client):
    response = client.get(
        "/api/v1/audio/voices",
        headers={"X-API-KEY": "test-api-key-1234567890"},
    )
    assert response.status_code == 429


def _find_route(app: FastAPI, method: str, path: str) -> APIRoute:
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method.upper() in route.methods:
            return route
    raise AssertionError(f"Route not found: {method} {path}")


def _extract_token_scope_dependency(route: APIRoute):
    for dependency in route.dependencies:
        dep_fn = getattr(dependency, "dependency", None)
        if dep_fn is not None and getattr(dep_fn, "_tldw_token_scope", False):
            return dep_fn
    raise AssertionError(f"No token scope dependency found for route: {route.path}")


def test_voice_routes_use_granular_endpoint_ids_and_voice_counter():
    app = FastAPI()
    app.include_router(audio_voices.router, prefix="/api/v1/audio")

    expectations = [
        ("POST", "/api/v1/audio/voices/upload", "audio.voices.upload"),
        ("POST", "/api/v1/audio/voices/encode", "audio.voices.encode"),
        ("GET", "/api/v1/audio/voices", "audio.voices.list"),
        ("GET", "/api/v1/audio/voices/{voice_id}", "audio.voices.get"),
        ("DELETE", "/api/v1/audio/voices/{voice_id}", "audio.voices.delete"),
        ("POST", "/api/v1/audio/voices/{voice_id}/preview", "audio.voices.preview"),
        ("POST", "/api/v1/audio/providers/fish_s2/references", "audio.voices.upload"),
        ("GET", "/api/v1/audio/providers/fish_s2/references", "audio.voices.list"),
        ("DELETE", "/api/v1/audio/providers/fish_s2/references/{reference_id}", "audio.voices.delete"),
    ]

    for method, path, endpoint_id in expectations:
        route = _find_route(app, method, path)
        scope_dep = _extract_token_scope_dependency(route)
        assert getattr(scope_dep, "_tldw_endpoint_id", None) == endpoint_id
        assert getattr(scope_dep, "_tldw_count_as", None) == "voice_call"


def test_audio_aggregate_mounts_presets_and_voice_conversion_routes():
    app = FastAPI()
    app.include_router(audio.router, prefix="/api/v1/audio")

    assert _find_route(app, "GET", "/api/v1/audio/presets")
    assert _find_route(app, "POST", "/api/v1/audio/voice-conversion")
