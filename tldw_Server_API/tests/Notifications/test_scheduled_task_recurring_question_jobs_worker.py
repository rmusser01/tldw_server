from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.rag_schemas_unified import UnifiedRAGResponse
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskDefinitionCreateRequest,
    ScheduledTaskPreviewCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import ScheduledTasksDatabase
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_rag_adapter import RecurringQuestionRAGError
from tldw_Server_API.app.services.scheduled_task_recurring_question_service import (
    ScheduledTaskRecurringQuestionService,
)
from tldw_Server_API.app.services.scheduled_task_recurring_question_worker import (
    RecurringQuestionWorkerRetryableError,
    handle_recurring_question_run_job,
    run_recurring_question_jobs_worker,
)

OWNER_ID = 5201
ACTOR = "recurring-question-worker-test"


def _service(tmp_path: Path) -> tuple[ScheduledTaskRecurringQuestionService, ScheduledTasksDatabase]:
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_task_recurring_question_worker.db")
    repo.ensure_schema()
    service = ScheduledTaskRecurringQuestionService(repository=repo)
    service._create_jobs_entry = lambda **_kwargs: {"id": 123}  # type: ignore[method-assign]
    return service, repo


def _payload(*, generation_mode: str = "optional") -> ScheduledTaskPreviewCreateRequest:
    return ScheduledTaskPreviewCreateRequest(
        mode="create",
        family="recurring_question",
        name="Worker question",
        description="Worker test",
        input={"question": "What changed?"},
        config={"generation_mode": generation_mode},
        schedule={"kind": "daily", "time": "09:00", "timezone": "UTC"},
        visibility_policy={"mode": "findings_only"},
        notification_policy={"channels": ["in_app"]},
        approval_policy={"required": False},
    )


def _create_definition_and_run(
    tmp_path: Path,
    *,
    generation_mode: str = "optional",
) -> tuple[ScheduledTasksDatabase, Any, Any]:
    service, repo = _service(tmp_path)
    preview = service.create_preview(owner_id=OWNER_ID, actor=ACTOR, payload=_payload(generation_mode=generation_mode))
    definition = service.create_definition(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskDefinitionCreateRequest(preview_id=preview.id),
    )
    run = service.create_manual_run(owner_id=OWNER_ID, actor=ACTOR, definition_id=definition.id)
    return repo, definition, run


def _job(definition_id: str, run_id: str) -> dict[str, Any]:
    return {
        "id": 123,
        "payload": {
            "owner_user_id": str(OWNER_ID),
            "definition_id": definition_id,
            "run_id": run_id,
        },
    }


@pytest.mark.asyncio
async def test_worker_persists_synthesized_finding(tmp_path):
    repo, definition, run = _create_definition_and_run(tmp_path)

    async def _rag_executor(request):
        return UnifiedRAGResponse(
            query=request.query,
            documents=[{"id": "doc-1", "title": "Doc", "score": 0.91, "snippet": "short"}],
            generated_answer="A useful answer.",
        )

    result = await handle_recurring_question_run_job(
        _job(definition.id, run.id),
        repository=repo,
        rag_executor=_rag_executor,
    )

    stored_run = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    results, total = repo.list_results(owner_id=OWNER_ID, definition_id=definition.id, limit=10, offset=0)
    assert result["status"] == "completed"  # nosec B101
    assert stored_run.status == "completed"  # nosec B101
    assert stored_run.outcome == "finding"  # nosec B101
    assert total == 1  # nosec B101
    assert results[0].answer == "A useful answer."  # nosec B101
    assert results[0].answer_mode == "synthesized"  # nosec B101


@pytest.mark.asyncio
async def test_worker_completes_no_match_without_result(tmp_path):
    repo, definition, run = _create_definition_and_run(tmp_path)

    async def _rag_executor(request):
        return UnifiedRAGResponse(query=request.query, documents=[], generated_answer=None)

    await handle_recurring_question_run_job(_job(definition.id, run.id), repository=repo, rag_executor=_rag_executor)

    stored_run = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    _results, total = repo.list_results(owner_id=OWNER_ID, definition_id=definition.id, limit=10, offset=0)
    assert stored_run.status == "completed"  # nosec B101
    assert stored_run.outcome == "no_match"  # nosec B101
    assert total == 0  # nosec B101


@pytest.mark.asyncio
async def test_worker_fails_generation_required_without_answer(tmp_path):
    repo, definition, run = _create_definition_and_run(tmp_path, generation_mode="required")

    async def _rag_executor(request):
        return UnifiedRAGResponse(
            query=request.query,
            documents=[{"id": "doc-1", "title": "Doc", "score": 0.91, "snippet": "short"}],
            generated_answer=None,
        )

    await handle_recurring_question_run_job(_job(definition.id, run.id), repository=repo, rag_executor=_rag_executor)

    stored_run = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    results, total = repo.list_results(owner_id=OWNER_ID, definition_id=definition.id, limit=10, offset=0)
    assert stored_run.status == "failed"  # nosec B101
    assert stored_run.failure_reason["code"] == "generation_required_unavailable"  # nosec B101
    assert total == 1  # nosec B101
    assert results[0].kind == "failure"  # nosec B101


@pytest.mark.asyncio
async def test_worker_records_non_retryable_rag_unavailable_failure(tmp_path):
    repo, definition, run = _create_definition_and_run(tmp_path)

    async def _rag_executor(_request):
        raise RecurringQuestionRAGError("rag_unavailable", retryable=False)

    await handle_recurring_question_run_job(_job(definition.id, run.id), repository=repo, rag_executor=_rag_executor)

    stored_run = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    results, total = repo.list_results(owner_id=OWNER_ID, definition_id=definition.id, limit=10, offset=0)
    assert stored_run.status == "failed"  # nosec B101
    assert stored_run.failure_reason["code"] == "rag_unavailable"  # nosec B101
    assert total == 1  # nosec B101
    assert results[0].kind == "failure"  # nosec B101


@pytest.mark.asyncio
async def test_worker_reraises_retryable_failure_after_recording_retry_state(tmp_path):
    repo, definition, run = _create_definition_and_run(tmp_path)

    async def _rag_executor(_request):
        raise RecurringQuestionRAGError("quota_exceeded", retryable=True)

    with pytest.raises(RecurringQuestionWorkerRetryableError, match="quota_exceeded"):
        await handle_recurring_question_run_job(_job(definition.id, run.id), repository=repo, rag_executor=_rag_executor)

    stored_run = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    assert stored_run.status == "queued"  # nosec B101
    assert stored_run.outcome == "none"  # nosec B101
    assert stored_run.failure_reason["code"] == "quota_exceeded"  # nosec B101
    assert stored_run.run_summary["retrying"] is True  # nosec B101


@pytest.mark.asyncio
async def test_retryable_failure_does_not_surface_failure_result_and_success_clears_stale_failure(
    tmp_path,
):
    repo, definition, run = _create_definition_and_run(tmp_path)

    async def _retryable_failure(_request):
        raise RecurringQuestionRAGError("quota_exceeded", retryable=True)

    with pytest.raises(RecurringQuestionWorkerRetryableError, match="quota_exceeded"):
        await handle_recurring_question_run_job(
            _job(definition.id, run.id),
            repository=repo,
            rag_executor=_retryable_failure,
        )

    _results, total_after_failure = repo.list_results(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        limit=10,
        offset=0,
    )
    assert total_after_failure == 0  # nosec B101

    async def _successful_retry(request):
        return UnifiedRAGResponse(
            query=request.query,
            documents=[{"id": "doc-1", "title": "Doc", "score": 0.91, "snippet": "short"}],
            generated_answer="A useful answer.",
        )

    await handle_recurring_question_run_job(
        _job(definition.id, run.id),
        repository=repo,
        rag_executor=_successful_retry,
    )

    stored_run = repo.get_run(owner_id=OWNER_ID, run_id=run.id)
    results, total_after_success = repo.list_results(
        owner_id=OWNER_ID,
        definition_id=definition.id,
        limit=10,
        offset=0,
    )
    assert stored_run.status == "completed"  # nosec B101
    assert stored_run.failure_reason is None  # nosec B101
    assert total_after_success == 1  # nosec B101
    assert results[0].kind == "finding"  # nosec B101


@pytest.mark.asyncio
async def test_worker_cancel_check_defers_cancelled_jobs_to_handler(monkeypatch):
    observed: dict[str, bool] = {}

    class FakeWorkerSDK:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        async def run(self, *, handler, cancel_check, **_kwargs) -> None:
            observed["cancel_check"] = await cancel_check({"cancel_requested_at": "2026-07-01T00:00:00Z"})

    monkeypatch.setattr(
        "tldw_Server_API.app.services.scheduled_task_recurring_question_worker.WorkerSDK",
        FakeWorkerSDK,
    )

    await run_recurring_question_jobs_worker()

    assert observed == {"cancel_check": False}  # nosec B101
