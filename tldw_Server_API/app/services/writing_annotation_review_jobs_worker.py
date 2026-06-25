"""Worker service for queued manuscript scene annotation reviews."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user_id
from tldw_Server_API.app.core.DB_Management.ManuscriptDB import ManuscriptDBHelper
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int as _coerce_int
from tldw_Server_API.app.core.Jobs.worker_utils import jobs_manager_from_env as _jobs_manager
from tldw_Server_API.app.core.Writing.manuscript_annotation_jobs import (
    WRITING_JOBS_DOMAIN,
    WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE,
    process_scene_annotation_review_job,
    writing_annotation_review_jobs_queue,
)

_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "owner_user_id",
        "user_id",
        "scene_text",
        "content_plain",
        "selected_text",
        "annotation_body",
        "body",
        "suggested_fix",
        "raw_model_output",
    }
)


class WritingAnnotationReviewJobError(RuntimeError):
    """Raised for controlled Writing annotation review worker failures."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        failure_code: str = "writing_annotation_review_job_failed",
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.failure_code = failure_code


async def handle_writing_annotation_review_job(
    job: dict[str, Any],
    *,
    manuscript_db: ManuscriptDBHelper | None = None,
    job_manager: JobManager | None = None,
) -> dict[str, Any]:
    """Handle one leased Writing annotation review Jobs row."""
    payload = _validate_job_payload(job)
    helper = manuscript_db
    loaded_db: Any | None = None
    if helper is None:
        loaded_db = await _get_database_for_job(job)
        helper = ManuscriptDBHelper(loaded_db)
    try:
        return await process_scene_annotation_review_job(
            manuscript_db=helper,
            job_payload=payload,
            job_manager=job_manager,
        )
    except WritingAnnotationReviewJobError:
        raise
    except Exception as exc:
        raise WritingAnnotationReviewJobError(
            "Scene annotation review failed.",
            retryable=_is_retryable_runtime_error(exc),
            failure_code="writing_annotation_review_runtime_failed",
        ) from exc
    finally:
        if loaded_db is not None:
            _close_worker_database(loaded_db)


def _validate_job_payload(job: dict[str, Any]) -> dict[str, Any]:
    job_type = str(job.get("job_type") or "").strip()
    if job_type != WRITING_SCENE_ANNOTATION_REVIEW_JOB_TYPE:
        raise WritingAnnotationReviewJobError(
            f"unsupported job_type: {job_type or '<missing>'}",
            retryable=False,
            failure_code="unsupported_job_type",
        )
    payload = _coerce_payload(job.get("payload"))
    forbidden = _FORBIDDEN_PAYLOAD_KEYS.intersection(payload)
    if forbidden:
        raise WritingAnnotationReviewJobError(
            "payload contains disallowed fields",
            retryable=False,
            failure_code="invalid_job_payload",
        )
    required = ("project_id", "scene_id", "scene_version", "provider", "model", "max_comments")
    for field_name in required:
        if payload.get(field_name) in (None, ""):
            raise WritingAnnotationReviewJobError(
                f"missing {field_name}",
                retryable=False,
                failure_code="invalid_job_payload",
            )
    try:
        payload["scene_version"] = int(payload["scene_version"])
        payload["max_comments"] = int(payload["max_comments"])
    except (TypeError, ValueError) as exc:
        raise WritingAnnotationReviewJobError(
            "invalid numeric payload field",
            retryable=False,
            failure_code="invalid_job_payload",
        ) from exc
    if payload["max_comments"] < 1 or payload["max_comments"] > 10:
        raise WritingAnnotationReviewJobError(
            "invalid max_comments",
            retryable=False,
            failure_code="invalid_job_payload",
        )
    category_filters = payload.get("category_filters")
    if category_filters is None:
        payload["category_filters"] = []
    elif not isinstance(category_filters, list):
        raise WritingAnnotationReviewJobError(
            "invalid category_filters",
            retryable=False,
            failure_code="invalid_job_payload",
        )
    else:
        payload["category_filters"] = [str(category).strip() for category in category_filters if str(category).strip()]
    if payload.get("review_focus") is not None:
        payload["review_focus"] = str(payload["review_focus"]).strip() or None
    for field_name in ("project_id", "scene_id", "provider", "model"):
        payload[field_name] = str(payload[field_name]).strip()
    return payload


def _coerce_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError as exc:
            raise WritingAnnotationReviewJobError(
                "payload must be an object",
                retryable=False,
                failure_code="invalid_job_payload",
            ) from exc
        if isinstance(loaded, dict):
            return dict(loaded)
    raise WritingAnnotationReviewJobError(
        "payload must be an object",
        retryable=False,
        failure_code="invalid_job_payload",
    )


async def _get_database_for_job(job: dict[str, Any]) -> Any:
    owner_user_id = str(job.get("owner_user_id") or "").strip()
    if not owner_user_id:
        raise WritingAnnotationReviewJobError(
            "missing owner_user_id",
            retryable=False,
            failure_code="invalid_job_payload",
        )
    try:
        normalized_user_id = int(owner_user_id)
    except ValueError as exc:
        raise WritingAnnotationReviewJobError(
            "invalid owner_user_id",
            retryable=False,
            failure_code="invalid_job_payload",
        ) from exc
    return await get_chacha_db_for_user_id(
        normalized_user_id,
        client_id=f"writing-annotation-review-worker-{normalized_user_id}",
    )


def _close_worker_database(db: Any) -> None:
    if db is None:
        return
    if hasattr(db, "release_context_connection"):
        db.release_context_connection()
        return
    if hasattr(db, "close_connection"):
        db.close_connection()


def _is_retryable_runtime_error(exc: Exception) -> bool:
    explicit = getattr(exc, "retryable", None)
    if explicit is not None:
        return bool(explicit)
    return isinstance(exc, (ConnectionError, TimeoutError))


async def _should_cancel(job: dict[str, Any], *, job_manager: JobManager) -> bool:
    current = job_manager.get_job(int(job["id"]))
    if not current:
        return False
    if str(current.get("status") or "").strip().lower() == "cancelled":
        return True
    if current.get("cancel_requested_at"):
        job_manager.finalize_cancelled(
            int(job["id"]),
            reason=str(current.get("cancellation_reason") or "requested"),
        )
        return True
    return False


async def run_writing_annotation_review_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the Writing scene annotation review Jobs worker loop."""
    worker_id = (
        os.getenv("WRITING_ANNOTATION_REVIEW_JOBS_WORKER_ID")
        or f"writing-annotation-review-worker-{os.getpid()}"
    ).strip()
    cfg = WorkerConfig(
        domain=WRITING_JOBS_DOMAIN,
        queue=writing_annotation_review_jobs_queue(),
        worker_id=worker_id,
        lease_seconds=_coerce_int(
            os.getenv("WRITING_ANNOTATION_REVIEW_JOBS_LEASE_SECONDS") or os.getenv("JOBS_LEASE_SECONDS"),
            60,
        ),
        renew_jitter_seconds=_coerce_int(
            os.getenv("WRITING_ANNOTATION_REVIEW_JOBS_RENEW_JITTER_SECONDS")
            or os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS"),
            5,
        ),
        renew_threshold_seconds=_coerce_int(
            os.getenv("WRITING_ANNOTATION_REVIEW_JOBS_RENEW_THRESHOLD_SECONDS")
            or os.getenv("JOBS_LEASE_RENEW_THRESHOLD_SECONDS"),
            10,
        ),
    )
    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    stop_watcher_task: asyncio.Task[None] | None = None

    if stop_event is not None:
        if stop_event.is_set():
            sdk.stop()
        else:

            async def _watch_stop() -> None:
                await stop_event.wait()
                sdk.stop()

            stop_watcher_task = asyncio.create_task(_watch_stop())

    logger.info("Writing annotation review Jobs worker starting: queue={} worker_id={}", cfg.queue, worker_id)
    try:
        await sdk.run(
            handler=handle_writing_annotation_review_job,
            cancel_check=lambda job_row: _should_cancel(job_row, job_manager=jm),
        )
    finally:
        if stop_watcher_task is not None and not stop_watcher_task.done():
            stop_watcher_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher_task
