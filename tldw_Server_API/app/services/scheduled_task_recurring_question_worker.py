"""Jobs worker for Recurring Question scheduled task runs."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    RunRow,
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.exceptions import (
    RecurringQuestionRAGError,
    RecurringQuestionWorkerRetryableError,
)
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_jobs import (
    RECURRING_QUESTION_QUEUE,
    SCHEDULED_TASKS_DOMAIN,
)
from tldw_Server_API.app.core.Scheduled_Tasks.recurring_question_rag_adapter import (
    RecurringQuestionRAGResult,
    build_rag_request_from_definition,
    execute_recurring_question_rag,
    safe_rag_request_snapshot,
)


async def handle_recurring_question_run_job(
    job: dict[str, Any],
    *,
    repository: ScheduledTasksDatabase | None = None,
    rag_executor: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute one Recurring Question run job."""
    payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
    owner_id = _owner_id_from_payload(job=job, payload=payload)
    definition_id = str(payload.get("definition_id") or "").strip()
    run_id = str(payload.get("run_id") or "").strip()
    if not definition_id or not run_id:
        raise ValueError("missing recurring question job identifiers")
    repo = repository or ScheduledTasksDatabase.for_user(owner_id)
    prepared = await asyncio.to_thread(
        _prepare_run_for_execution,
        repo=repo,
        owner_id=owner_id,
        definition_id=definition_id,
        run_id=run_id,
        job=job,
    )
    if isinstance(prepared, dict):
        return prepared
    definition, running, generation_mode = prepared
    try:
        rag_request = build_rag_request_from_definition(
            definition,
            scope_snapshot=running.scope_snapshot,
            finding_policy=running.finding_policy_snapshot,
            generation_mode=generation_mode,
        )
        adapter_result = await execute_recurring_question_rag(
            rag_request,
            rag_executor=rag_executor,
            generation_mode=generation_mode,
            finding_policy=running.finding_policy_snapshot,
        )
        return await asyncio.to_thread(
            _persist_adapter_result,
            repo=repo,
            owner_id=owner_id,
            definition=definition,
            run=running,
            adapter_result=adapter_result,
            rag_request_snapshot=safe_rag_request_snapshot(rag_request),
        )
    except RecurringQuestionRAGError as exc:
        await asyncio.to_thread(
            _persist_failure,
            repo=repo,
            owner_id=owner_id,
            definition=definition,
            run=running,
            code=exc.code,
            retryable=exc.retryable,
            details=exc.details,
            surface_result=not exc.retryable,
        )
        if exc.retryable:
            raise RecurringQuestionWorkerRetryableError(exc.code) from exc
        return {"status": "failed", "run_id": running.id, "outcome": "degraded", "failure_reason": {"code": exc.code}}
    except Exception as exc:
        await asyncio.to_thread(
            _persist_failure,
            repo=repo,
            owner_id=owner_id,
            definition=definition,
            run=running,
            code="worker_failure",
            retryable=True,
            details={"error_type": type(exc).__name__},
            surface_result=False,
        )
        raise RecurringQuestionWorkerRetryableError("worker_failure") from exc


async def run_recurring_question_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the Recurring Question Jobs worker loop until stopped."""
    worker_id = (
        os.getenv("SCHEDULED_TASKS_RECURRING_QUESTION_WORKER_ID")
        or f"scheduled-tasks-rq-{os.getpid()}"
    ).strip()
    lease_seconds = _coerce_int(
        os.getenv("SCHEDULED_TASKS_RECURRING_QUESTION_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
        60,
    )
    cfg = WorkerConfig(
        domain=SCHEDULED_TASKS_DOMAIN,
        queue=RECURRING_QUESTION_QUEUE,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=_coerce_int(os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=_coerce_int(os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"), 10),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("SCHEDULED_TASKS_RECURRING_QUESTION_RETRY_BACKOFF_SECONDS"), 10),
    )
    sdk = WorkerSDK(_jobs_manager(), cfg)
    stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        async def _watch_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher_task = asyncio.create_task(_watch_stop())

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        return await handle_recurring_question_run_job(job)

    async def _cancel_check(job: dict[str, Any]) -> bool:
        del job
        return False

    logger.info("Scheduled Tasks Recurring Question Jobs worker starting: worker_id={}", worker_id)
    try:
        await sdk.run(handler=_handler, cancel_check=_cancel_check)
    finally:
        if stop_watcher_task is not None and not stop_watcher_task.done():
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher_task


def _persist_adapter_result(
    *,
    repo: ScheduledTasksDatabase,
    owner_id: int,
    definition: DefinitionRow,
    run: RunRow,
    adapter_result: RecurringQuestionRAGResult,
    rag_request_snapshot: dict[str, Any],
) -> dict[str, Any]:
    if adapter_result.outcome == "no_match":
        updated = repo.update_run(
            owner_id=owner_id,
            run_id=run.id,
            patch={
                "status": "completed",
                "outcome": "no_match",
                "ended_at": _utcnow_iso(),
                "failure_reason": None,
                "rag_request_snapshot": rag_request_snapshot,
                "evidence_summary": adapter_result.evidence_summary,
                "run_summary": {"message": adapter_result.summary, "title": adapter_result.title},
            },
        )
        return {"status": updated.status, "run_id": updated.id, "outcome": updated.outcome}
    if adapter_result.failure_reason is not None:
        _persist_failure(
            repo=repo,
            owner_id=owner_id,
            definition=definition,
            run=run,
            code=str(adapter_result.failure_reason.get("code") or "rag_response_error"),
            retryable=False,
            details=adapter_result.failure_reason,
            adapter_result=adapter_result,
            rag_request_snapshot=rag_request_snapshot,
        )
        return {
            "status": "failed",
            "run_id": run.id,
            "outcome": "degraded",
            "failure_reason": adapter_result.failure_reason,
        }
    updated = repo.update_run(
        owner_id=owner_id,
        run_id=run.id,
        patch={
            "status": "completed",
            "outcome": "finding",
            "ended_at": _utcnow_iso(),
            "failure_reason": None,
            "rag_request_snapshot": rag_request_snapshot,
            "evidence_summary": adapter_result.evidence_summary,
            "run_summary": {"message": adapter_result.summary, "title": adapter_result.title},
        },
    )
    result = repo.create_result(
        owner_id=owner_id,
        definition_id=definition.id,
        run_id=run.id,
        kind="finding",
        title=adapter_result.title,
        summary=adapter_result.summary,
        answer=adapter_result.answer,
        answer_mode=adapter_result.answer_mode,
        confidence=adapter_result.confidence,
        source_refs=adapter_result.source_refs,
        dedupe_key=_dedupe_key(definition_id=definition.id, run_id=run.id, kind="finding"),
        visibility_destination={"home": True, "results": True},
    )
    return {"status": updated.status, "run_id": updated.id, "outcome": updated.outcome, "result_id": result.id}


def _persist_failure(
    *,
    repo: ScheduledTasksDatabase,
    owner_id: int,
    definition: DefinitionRow,
    run: RunRow,
    code: str,
    retryable: bool,
    details: dict[str, Any] | None,
    adapter_result: RecurringQuestionRAGResult | None = None,
    rag_request_snapshot: dict[str, Any] | None = None,
    surface_result: bool = True,
) -> None:
    failure_reason = {"code": code, "retryable": retryable, **(details or {})}
    should_finalize = surface_result or not retryable
    repo.update_run(
        owner_id=owner_id,
        run_id=run.id,
        patch={
            "status": "failed" if should_finalize else "queued",
            "outcome": "degraded" if should_finalize else "none",
            "ended_at": _utcnow_iso() if should_finalize else None,
            "failure_reason": failure_reason,
            "rag_request_snapshot": rag_request_snapshot or run.rag_request_snapshot,
            "evidence_summary": adapter_result.evidence_summary if adapter_result is not None else run.evidence_summary,
            "run_summary": {
                "message": adapter_result.summary if adapter_result is not None else "Recurring Question run failed.",
                "failure_code": code,
                "retrying": not should_finalize,
            },
        },
    )
    if not surface_result:
        return
    repo.create_result(
        owner_id=owner_id,
        definition_id=definition.id,
        run_id=run.id,
        kind="failure",
        title=adapter_result.title if adapter_result is not None else "Recurring Question run failed",
        summary=adapter_result.summary if adapter_result is not None else f"Run failed with {code}.",
        answer=None,
        answer_mode="none",
        confidence={"label": "none"},
        source_refs=adapter_result.source_refs if adapter_result is not None else [],
        dedupe_key=_dedupe_key(definition_id=definition.id, run_id=run.id, kind=f"failure:{code}"),
        visibility_destination={"home": True, "results": True},
    )


def _prepare_run_for_execution(
    *,
    repo: ScheduledTasksDatabase,
    owner_id: int,
    definition_id: str,
    run_id: str,
    job: dict[str, Any],
) -> tuple[DefinitionRow, RunRow, str] | dict[str, Any]:
    repo.ensure_schema()
    definition = repo.get_definition(owner_id=owner_id, definition_id=definition_id)
    run = repo.get_run(owner_id=owner_id, run_id=run_id)
    if definition is None:
        raise ValueError(f"definition not found: {definition_id}")
    if run is None:
        raise ValueError(f"run not found: {run_id}")
    if _job_cancelled(job):
        cancelled = repo.update_run(
            owner_id=owner_id,
            run_id=run_id,
            patch={
                "status": "cancelled",
                "outcome": "none",
                "ended_at": _utcnow_iso(),
                "failure_reason": {"code": "job_cancelled", "retryable": False},
                "run_summary": {"message": "Run cancelled before execution."},
            },
        )
        return {"status": cancelled.status, "run_id": cancelled.id, "outcome": cancelled.outcome}
    running = repo.update_run(
        owner_id=owner_id,
        run_id=run_id,
        patch={
            "status": "running",
            "outcome": "none",
            "started_at": _utcnow_iso(),
            "ended_at": None,
            "failure_reason": None,
            "run_summary": {"message": "Running RAG query."},
        },
    )
    config = _definition_config(repo=repo, definition=definition)
    generation_mode = str(config.get("generation_mode") or "optional")
    return definition, running, generation_mode


def _definition_config(*, repo: ScheduledTasksDatabase, definition: DefinitionRow) -> dict[str, Any]:
    preview = repo.get_preview(owner_id=definition.owner_id, preview_id=definition.preview_id)
    if preview is None:
        return {}
    config = preview.normalized_config.get("config", {})
    return dict(config) if isinstance(config, dict) else {}


def _owner_id_from_payload(*, job: dict[str, Any], payload: dict[str, Any]) -> int:
    raw_owner = payload.get("owner_user_id") or job.get("owner_user_id")
    try:
        return int(str(raw_owner))
    except (TypeError, ValueError) as exc:
        raise ValueError("missing recurring question owner_user_id") from exc


def _job_cancelled(job: dict[str, Any]) -> bool:
    return bool(job.get("cancel_requested_at") or job.get("cancelled_at"))


def _dedupe_key(*, definition_id: str, run_id: str, kind: str) -> str:
    payload = json.dumps(
        {"definition_id": definition_id, "run_id": run_id, "kind": kind},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"rq:{hashlib.sha256(payload).hexdigest()}"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
