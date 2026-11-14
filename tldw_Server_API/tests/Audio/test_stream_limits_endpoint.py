from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming as audio_streaming
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_limits_shape(monkeypatch):
    async def _get_limits_for_user(user_id: int):
        _ = user_id
        return {
            "daily_minutes": 30.0,
            "monthly_minutes": 100.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

    async def _get_daily_minutes_used(user_id: int):
        _ = user_id
        return 5.0

    async def _get_monthly_minutes_used(user_id: int):
        _ = user_id
        return 20.0

    async def _get_user_tier(user_id: int):
        _ = user_id
        return "free"

    monkeypatch.setattr(audio_streaming, "_get_limits_for_user", _get_limits_for_user)
    monkeypatch.setattr(audio_streaming, "_get_daily_minutes_used", _get_daily_minutes_used)
    monkeypatch.setattr(audio_streaming, "_get_monthly_minutes_used", _get_monthly_minutes_used)
    monkeypatch.setattr(audio_streaming, "_get_user_tier", _get_user_tier)

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/audio/stream/limits",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    data = await audio_streaming.streaming_limits(
        Request(scope, _receive),
        current_user=SimpleNamespace(id=1),
    )

    # Top-level shape
    assert isinstance(data, dict)
    assert "user_id" in data and isinstance(data["user_id"], int)
    assert "tier" in data and isinstance(data["tier"], str)
    assert "limits" in data and isinstance(data["limits"], dict)
    assert "used_today_minutes" in data
    assert "remaining_minutes" in data  # may be None for unlimited tiers
    assert data["active_streams"] is None
    assert data["used_month_minutes"] == 20.0
    assert data["remaining_month_minutes"] == 80.0
    assert "can_start_stream" in data and isinstance(data["can_start_stream"], bool)

    # Limits structure
    limits = data["limits"]
    for key in ("daily_minutes", "concurrent_streams", "concurrent_jobs", "max_file_size_mb"):
        assert key in limits

    # Value sanity (types only; values are environment/config-dependent)
    if limits["daily_minutes"] is not None:
        assert isinstance(limits["daily_minutes"], (int, float))
    assert isinstance(limits["concurrent_streams"], int)
    assert isinstance(limits["concurrent_jobs"], int)
    assert isinstance(limits["max_file_size_mb"], int)
    assert data["can_start_stream"] is True


@pytest.mark.unit
def test_stream_limits_off_path_through_real_app(monkeypatch):
    """OFF path (spec 2): with usage quotas off, GET /stream/limits returns null limits and can_start_stream=True."""
    monkeypatch.delenv("USAGE_QUOTAS_ENABLED", raising=False)
    monkeypatch.delenv("LIMIT_ENFORCEMENT_ENABLED", raising=False)
    monkeypatch.setattr("tldw_Server_API.app.core.config.load_comprehensive_config", lambda: None)

    async def _get_daily_minutes_used(user_id: int):
        """A daily-minutes-used stand-in reporting none used."""
        _ = user_id
        return 0.0

    async def _get_monthly_minutes_used(user_id: int):
        """A monthly-minutes-used stand-in reporting none used."""
        _ = user_id
        return 0.0

    async def _get_user_tier(user_id: int):
        """A tier-lookup stand-in reporting the free tier."""
        _ = user_id
        return "free"

    monkeypatch.setattr(audio_streaming, "_get_daily_minutes_used", _get_daily_minutes_used)
    monkeypatch.setattr(audio_streaming, "_get_monthly_minutes_used", _get_monthly_minutes_used)
    monkeypatch.setattr(audio_streaming, "_get_user_tier", _get_user_tier)
    # _get_limits_for_user is intentionally left unpatched: it must hit the real
    # get_limits_for_user() and return the quotas-off unlimited dict.

    app = FastAPI()
    app.include_router(audio_streaming.router, prefix="/api/v1/audio")
    app.dependency_overrides[get_request_user] = lambda: SimpleNamespace(id=1)

    with TestClient(app) as client:
        response = client.get("/api/v1/audio/stream/limits")

    assert response.status_code == 200
    data = response.json()
    assert all(value is None for value in data["limits"].values())
    assert data["can_start_stream"] is True
