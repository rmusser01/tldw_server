"""Tests for the Writing scene annotation review Jobs worker service."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest


pytestmark = pytest.mark.unit


def _job(**overrides: Any) -> dict[str, Any]:
    payload = {
        "project_id": "project-1",
        "scene_id": "scene-1",
        "scene_version": 1,
        "provider": "openai",
        "model": "gpt-4o-mini",
        "max_comments": 3,
        "category_filters": [],
    }
    job = {
        "id": 10,
        "job_type": "writing_scene_annotation_review",
        "owner_user_id": "42",
        "payload": payload,
    }
    job.update(overrides)
    return job


@pytest.mark.asyncio
async def test_handle_writing_annotation_review_job_requires_owner_user_id() -> None:
    from tldw_Server_API.app.services.writing_annotation_review_jobs_worker import (
        WritingAnnotationReviewJobError,
        handle_writing_annotation_review_job,
    )

    with pytest.raises(WritingAnnotationReviewJobError) as excinfo:
        await handle_writing_annotation_review_job(_job(owner_user_id=""))

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "invalid_job_payload"


@pytest.mark.asyncio
async def test_handle_writing_annotation_review_job_loads_user_db_and_processes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tldw_Server_API.app.services.writing_annotation_review_jobs_worker as worker

    calls: dict[str, Any] = {}
    fake_db = object()

    async def _get_db(user_id: int, *, client_id: str) -> object:
        calls["db_user_id"] = user_id
        calls["db_client_id"] = client_id
        return fake_db

    class _Helper:
        def __init__(self, db: object) -> None:
            calls["helper_db"] = db

    async def _process(**kwargs: Any) -> dict[str, Any]:
        calls["process_kwargs"] = kwargs
        return {"created_annotation_ids": ["ann-1"], "diagnostics": []}

    monkeypatch.setattr(worker, "get_chacha_db_for_user_id", _get_db)
    monkeypatch.setattr(worker, "ManuscriptDBHelper", _Helper)
    monkeypatch.setattr(worker, "process_scene_annotation_review_job", _process)

    result = await worker.handle_writing_annotation_review_job(_job())

    assert result == {"created_annotation_ids": ["ann-1"], "diagnostics": []}
    assert calls["db_user_id"] == 42
    assert calls["db_client_id"] == "writing-annotation-review-worker-42"
    assert isinstance(calls["process_kwargs"]["manuscript_db"], _Helper)
    assert calls["process_kwargs"]["job_payload"]["scene_id"] == "scene-1"


@pytest.mark.asyncio
async def test_handle_writing_annotation_review_job_rejects_unknown_job_type() -> None:
    from tldw_Server_API.app.services.writing_annotation_review_jobs_worker import (
        WritingAnnotationReviewJobError,
        handle_writing_annotation_review_job,
    )

    with pytest.raises(WritingAnnotationReviewJobError) as excinfo:
        await handle_writing_annotation_review_job(_job(job_type="other"))

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "unsupported_job_type"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("processor_error", "expected_retryable"),
    [
        pytest.param(
            "rate_limit",
            True,
            id="rate-limit",
        ),
        pytest.param(
            "provider_503",
            True,
            id="provider-503",
        ),
        pytest.param(
            "bad_request",
            False,
            id="bad-request",
        ),
        pytest.param(
            "configuration",
            False,
            id="configuration",
        ),
    ],
)
async def test_handle_writing_annotation_review_job_classifies_chat_runtime_retryability(
    processor_error: str,
    expected_retryable: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.Chat.Chat_Deps import (
        ChatBadRequestError,
        ChatConfigurationError,
        ChatProviderError,
        ChatRateLimitError,
    )
    import tldw_Server_API.app.services.writing_annotation_review_jobs_worker as worker

    errors = {
        "rate_limit": ChatRateLimitError("rate limited", provider="openai"),
        "provider_503": ChatProviderError("provider unavailable", status_code=503, provider="openai"),
        "bad_request": ChatBadRequestError("bad request", provider="openai"),
        "configuration": ChatConfigurationError("missing key", provider="openai"),
    }

    async def _process(**_kwargs: Any) -> dict[str, Any]:
        raise errors[processor_error]

    monkeypatch.setattr(worker, "process_scene_annotation_review_job", _process)

    with pytest.raises(worker.WritingAnnotationReviewJobError) as excinfo:
        await worker.handle_writing_annotation_review_job(_job(), manuscript_db=object())

    assert excinfo.value.retryable is expected_retryable
    assert excinfo.value.failure_code == "writing_annotation_review_runtime_failed"


@pytest.mark.asyncio
async def test_run_worker_uses_writing_config_and_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    import tldw_Server_API.app.services.writing_annotation_review_jobs_worker as worker

    captured: dict[str, Any] = {}

    class _SDK:
        def __init__(self, jm: object, cfg: object) -> None:
            captured["jm"] = jm
            captured["cfg"] = cfg
            self.stopped = False

        def stop(self) -> None:
            self.stopped = True
            captured["stopped"] = True

        async def run(self, **kwargs: Any) -> None:
            captured["run_kwargs"] = kwargs

    monkeypatch.setenv("WRITING_ANNOTATION_REVIEW_JOBS_QUEUE", "writing-review")
    monkeypatch.setenv("WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ID", "worker-1")
    monkeypatch.setattr(worker, "_jobs_manager", lambda: "jobs-manager")
    monkeypatch.setattr(worker, "WorkerSDK", _SDK)
    stop_event = asyncio.Event()
    stop_event.set()

    await worker.run_writing_annotation_review_jobs_worker(stop_event)

    cfg = captured["cfg"]
    assert cfg.domain == "writing"
    assert cfg.queue == "writing-review"
    assert cfg.worker_id == "worker-1"
    assert captured["run_kwargs"]["handler"] is worker.handle_writing_annotation_review_job
    assert captured["run_kwargs"]["job_type"] == "writing_scene_annotation_review"
    assert captured["stopped"] is True
