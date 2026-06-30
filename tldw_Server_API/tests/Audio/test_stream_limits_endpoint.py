from types import SimpleNamespace

import pytest
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_streaming as audio_streaming


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stream_limits_shape(monkeypatch):
    async def _get_limits_for_user(user_id: int):
        _ = user_id
        return {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": 25,
        }

    async def _get_daily_minutes_used(user_id: int):
        _ = user_id
        return 5.0

    async def _active_streams_count(user_id: int):
        _ = user_id
        return 0

    async def _get_user_tier(user_id: int):
        _ = user_id
        return "free"

    monkeypatch.setattr(audio_streaming, "_get_limits_for_user", _get_limits_for_user)
    monkeypatch.setattr(audio_streaming, "_get_daily_minutes_used", _get_daily_minutes_used)
    monkeypatch.setattr(audio_streaming, "_active_streams_count", _active_streams_count)
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
    assert "active_streams" in data and isinstance(data["active_streams"], int)
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
