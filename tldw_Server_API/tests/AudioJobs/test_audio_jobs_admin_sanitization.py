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


class _PagingJobManager:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(kwargs)
        return [
            {
                "id": 3,
                "uuid": "job-3",
                "job_type": "transcribe",
                "status": "queued",
                "priority": 5,
                "retry_count": 0,
                "max_retries": 3,
                "owner_user_id": "owner-1",
                "available_at": None,
                "started_at": None,
                "leased_until": None,
                "created_at": "2026-05-02 13:00:00",
                "updated_at": None,
                "completed_at": None,
            },
            {
                "id": 2,
                "uuid": "job-2",
                "job_type": "transcribe",
                "status": "queued",
                "priority": 5,
                "retry_count": 0,
                "max_retries": 3,
                "owner_user_id": "owner-1",
                "available_at": None,
                "started_at": None,
                "leased_until": None,
                "created_at": "2026-05-02 12:00:00",
                "updated_at": None,
                "completed_at": None,
            },
            {
                "id": 1,
                "uuid": "job-1",
                "job_type": "transcribe",
                "status": "queued",
                "priority": 5,
                "retry_count": 0,
                "max_retries": 3,
                "owner_user_id": "owner-1",
                "available_at": None,
                "started_at": None,
                "leased_until": None,
                "created_at": "2026-05-02 11:00:00",
                "updated_at": None,
                "completed_at": None,
            },
        ]


@pytest.mark.asyncio
async def test_audio_jobs_admin_list_includes_cursor_pagination() -> None:
    jm = _PagingJobManager()

    response = await audio_jobs.list_audio_jobs_admin(jm=jm, limit=2)

    assert len(response.jobs) == 2
    assert response.limit == 2
    assert response.next_cursor
    assert response.pagination.mode == "cursor"
    assert response.pagination.limit == 2
    assert response.pagination.cursor is None
    assert response.pagination.next_cursor == response.next_cursor
    assert response.pagination.has_more is True
    assert jm.calls[0]["limit"] == 3


@pytest.mark.asyncio
async def test_audio_jobs_admin_list_rejects_invalid_cursor() -> None:
    with pytest.raises(HTTPException) as exc_info:
        await audio_jobs.list_audio_jobs_admin(
            jm=_PagingJobManager(),
            limit=2,
            cursor="not-a-valid-cursor",
        )

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == "Invalid cursor"


@pytest.mark.asyncio
async def test_audio_jobs_admin_list_accepts_returned_cursor() -> None:
    jm = _PagingJobManager()
    first_page = await audio_jobs.list_audio_jobs_admin(jm=jm, limit=2)

    second_page = await audio_jobs.list_audio_jobs_admin(
        jm=jm,
        limit=2,
        cursor=first_page.next_cursor,
    )

    assert second_page.pagination.cursor == first_page.next_cursor
    assert jm.calls[1]["created_before"].isoformat() == "2026-05-02T12:00:00"
    assert jm.calls[1]["before_id"] == 2


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
