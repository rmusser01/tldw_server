import io
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from starlette.datastructures import Headers
from starlette.requests import Request

import tldw_Server_API.app.api.v1.endpoints.audio.audio_transcriptions as audio_transcriptions
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal


def _make_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v1/audio/transcriptions",
        "headers": [],
        "query_string": b"",
        "server": ("testserver", 80),
        "client": ("testclient", 12345),
    }

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, _receive)


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(
        io.BytesIO(content),
        filename=filename,
        headers=Headers({"content-type": "audio/wav"}),
    )


def _patch_audio_quota(monkeypatch, *, can_start: bool, max_file_size_mb: int = 1) -> None:
    original_shim = audio_transcriptions._audio_shim_attr

    async def _get_limits_for_user(user_id: int):
        _ = user_id
        return {
            "daily_minutes": 30.0,
            "concurrent_streams": 1,
            "concurrent_jobs": 1,
            "max_file_size_mb": max_file_size_mb,
        }

    async def _can_start_job(user_id: int):
        _ = user_id
        if can_start:
            return True, ""
        return False, "Concurrent job limit reached (1)"

    async def _noop_async(*args, **kwargs):
        _ = args, kwargs

    def _shim_attr(name: str):
        if name == "get_limits_for_user":
            return _get_limits_for_user
        if name == "can_start_job":
            return _can_start_job
        if name in {"increment_jobs_started", "finish_job"}:
            return _noop_async
        if name == "get_job_heartbeat_interval_seconds":
            return lambda: 0
        return original_shim(name)

    monkeypatch.setattr(audio_transcriptions, "_audio_shim_attr", _shim_attr, raising=True)


@pytest.mark.asyncio
async def test_http_file_size_limit_exceeded(monkeypatch):
    """Uploads an oversized file to trigger 413 without invoking ffmpeg."""
    _patch_audio_quota(monkeypatch, can_start=True, max_file_size_mb=1)

    with pytest.raises(HTTPException) as exc_info:
        await audio_transcriptions.create_transcription(
            _make_request(),
            file=_upload("big.wav", b"0" * (2 * 1024 * 1024)),
            model="whisper-1",
            response_format="json",
            current_user=SimpleNamespace(id=1),
            principal=AuthPrincipal(kind="user", user_id=1),
            db=None,
            billing_org_id=None,
        )

    assert exc_info.value.status_code == 413
    assert "exceeds maximum" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_http_concurrent_jobs_cap(monkeypatch):
    """Forces can_start_job to reject to exercise 429 response path."""
    _patch_audio_quota(monkeypatch, can_start=False)

    with pytest.raises(HTTPException) as exc_info:
        await audio_transcriptions.create_transcription(
            _make_request(),
            file=_upload("ok.wav", b"0" * (64 * 1024)),
            model="whisper-1",
            response_format="json",
            current_user=SimpleNamespace(id=1),
            principal=AuthPrincipal(kind="user", user_id=1),
            db=None,
            billing_org_id=None,
        )

    assert exc_info.value.status_code == 429
    assert "Concurrent job limit" in str(exc_info.value.detail)
