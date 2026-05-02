from __future__ import annotations

from typing import Any

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints.audio import audio_jobs


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.infos: list[str] = []

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.errors.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)


class _FailingJobManager:
    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("audio jobs backend exploded at /private/audio-jobs.db")

    def summarize_by_status(self, **kwargs: Any) -> dict[str, int]:
        raise RuntimeError("audio jobs backend exploded at /private/audio-jobs.db")

    def summarize_by_owner_and_status(self, **kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("audio jobs backend exploded at /private/audio-jobs.db")

    def count_jobs(self, **kwargs: Any) -> int:
        raise RuntimeError("audio jobs backend exploded at /private/audio-jobs.db")


@pytest.mark.parametrize(
    ("handler_factory", "expected_detail", "expected_log"),
    [
        (
            lambda: audio_jobs.list_audio_jobs_admin(jm=_FailingJobManager(), limit=50),
            "Failed to list jobs",
            "Failed to list jobs",
        ),
        (
            lambda: audio_jobs.summarize_audio_jobs_admin(jm=_FailingJobManager()),
            "Failed to summarize jobs",
            "Failed to summarize jobs",
        ),
        (
            lambda: audio_jobs.summary_by_owner_admin(jm=_FailingJobManager()),
            "Failed to summarize by owner",
            "Failed to summarize by owner",
        ),
        (
            lambda: audio_jobs.owner_processing_summary(
                owner_user_id=123,
                jm=_FailingJobManager(),
                request=None,
            ),
            "Failed to get owner processing summary",
            "Failed to get owner processing summary",
        ),
    ],
    ids=["list", "summary", "summary_by_owner", "owner_processing"],
)
@pytest.mark.asyncio
async def test_audio_jobs_admin_job_manager_failure_logs_are_sanitized(
    monkeypatch,
    handler_factory,
    expected_detail,
    expected_log,
):
    logger = _LoggerStub()
    monkeypatch.setattr(audio_jobs, "logger", logger)

    with pytest.raises(HTTPException) as exc_info:
        await handler_factory()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert logger.errors == [expected_log]
    error_text = "\n".join(logger.errors)
    assert "audio jobs backend exploded" not in error_text
    assert "/private/audio-jobs.db" not in error_text


@pytest.mark.parametrize(
    ("handler_factory", "expected_detail", "expected_log"),
    [
        (
            lambda: audio_jobs.get_user_tier_admin(user_id=123),
            "Failed to get user tier",
            "Failed to get user tier",
        ),
        (
            lambda: audio_jobs.set_user_tier_admin(
                user_id=123,
                req=audio_jobs.SetUserTierRequest(tier="standard"),
            ),
            "Failed to set user tier",
            "Failed to set user tier",
        ),
    ],
    ids=["get_tier", "set_tier"],
)
@pytest.mark.asyncio
async def test_audio_jobs_admin_tier_failure_logs_are_sanitized(
    monkeypatch,
    handler_factory,
    expected_detail,
    expected_log,
):
    async def _raise_tier_error(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("audio jobs backend exploded at /private/audio-jobs.db")

    logger = _LoggerStub()
    monkeypatch.setattr(audio_jobs, "logger", logger)
    monkeypatch.setattr(audio_jobs, "get_user_tier", _raise_tier_error)
    monkeypatch.setattr(audio_jobs, "set_user_tier", _raise_tier_error)

    with pytest.raises(HTTPException) as exc_info:
        await handler_factory()

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == expected_detail
    assert logger.errors == [expected_log]
    error_text = "\n".join(logger.errors)
    assert "audio jobs backend exploded" not in error_text
    assert "/private/audio-jobs.db" not in error_text
