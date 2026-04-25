from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from cachetools import LRUCache
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    check_rate_limit,
    get_auth_principal,
    rbac_rate_limit,
    require_permissions,
)
from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.media_add_deps import get_add_media_form
from tldw_Server_API.app.api.v1.API_Deps.validations_deps import file_validator_instance
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Ingestion_Media_Processing.input_sourcing import (
    TempDirManager,
    save_uploaded_files,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.exceptions import BadRequestError
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services.worker_startup_policy import worker_path_enabled
from tldw_Server_API.app.services.app_lifecycle import assert_may_start_work

router = APIRouter()

MAX_CACHED_JOB_MANAGER_INSTANCES = 4
_job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
_job_manager_lock = threading.Lock()
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})
_TERMINAL_MEDIA_INGEST_JOB_STATUSES = frozenset({"completed", "failed", "cancelled", "quarantined"})


def get_job_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    db_path = (os.getenv("JOBS_DB_PATH") or "").strip()
    cache_key = f"url:{db_url}" if db_url else f"path:{db_path or 'default'}"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached

        if not db_url:
            job_manager = JobManager(db_path=Path(db_path)) if db_path else JobManager()
        else:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)

        _job_manager_cache[cache_key] = job_manager
        return job_manager


class MediaIngestJobItem(BaseModel):
    id: int
    uuid: str | None
    source: str
    source_kind: str
    status: str


class SubmitMediaIngestJobsResponse(BaseModel):
    batch_id: str
    jobs: list[MediaIngestJobItem]
    errors: list[str] = Field(default_factory=list)


class MediaIngestJobStatus(BaseModel):
    id: int
    uuid: str | None
    status: str
    job_type: str
    owner_user_id: str | None
    created_at: str | None
    started_at: str | None
    completed_at: str | None
    cancelled_at: str | None
    cancellation_reason: str | None
    progress_percent: float | None
    progress_message: str | None
    result: dict[str, Any] | None
    error_message: str | None
    media_type: str | None = None
    source: str | None = None
    source_kind: str | None = None
    batch_id: str | None = None


class CancelMediaIngestJobResponse(BaseModel):
    success: bool
    job_id: int
    status: str
    message: str | None = None


class CancelMediaIngestBatchResponse(BaseModel):
    success: bool
    batch_id: str
    requested: int
    cancelled: int
    already_terminal: int
    failed: int = 0
    message: str | None = None


class MediaIngestJobListResponse(BaseModel):
    batch_id: str
    jobs: list[MediaIngestJobStatus]


def _cleanup_dir(path_str: str) -> None:
    try:
        shutil.rmtree(path_str, ignore_errors=True)
    except Exception as exc:
        logger.debug("Failed to cleanup temp dir {}: {}", path_str, exc)


def _validate_submit_inputs(
    media_type: Any,
    urls: list[str] | None,
    files: list[UploadFile] | None,
) -> None:
    if urls or files:
        return

    logger.warning("No URLs or files provided in media ingest job submit request")
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            "No valid media sources supplied. At least one 'url' in the "
            "'urls' list or one 'file' in the 'files' list must be provided."
        ),
    )


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            logger.debug(
                "Failed to parse payload as JSON (length={}, error={})",
                len(payload),
                exc,
            )
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _parse_job_created_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None
    return None


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_heavy_media_ingest_request(form_data: AddMediaForm) -> bool:
    media_type = str(getattr(form_data, "media_type", "") or "").strip().lower()
    if media_type in {"audio", "video"}:
        return True
    if bool(getattr(form_data, "enable_ocr", False)):
        return True
    return False


def _heavy_media_ingest_worker_available() -> bool:
    return worker_path_enabled(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        default_stable=False,
        # Queue routing should still honor explicit route policy in tests so
        # integration coverage can model a deployed heavy-worker path without
        # auto-starting local workers.
        test_mode=False,
    )


def _resolve_media_ingest_queue(form_data: AddMediaForm) -> str:
    default_queue = (os.getenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE") or "default").strip() or "default"
    route_heavy = _is_truthy(os.getenv("MEDIA_INGEST_JOBS_ROUTE_HEAVY", "true"))
    if not route_heavy:
        return default_queue
    if not _is_heavy_media_ingest_request(form_data):
        return default_queue
    if not _heavy_media_ingest_worker_available():
        return default_queue
    # Keep fallback within JobManager standard queue names unless explicitly overridden.
    heavy_queue = (os.getenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE") or "low").strip() or "low"
    return heavy_queue


def _create_media_ingest_job(
    *,
    jm: JobManager,
    selected_queue: str,
    payload: dict[str, Any],
    current_user: User,
    batch_id: str,
    request_id: str | None,
    trace_id: str | None,
) -> dict[str, Any]:
    try:
        return jm.create_job(
            domain="media_ingest",
            queue=selected_queue,
            job_type="media_ingest_item",
            payload=payload,
            batch_group=batch_id,
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=3,
            request_id=request_id,
            trace_id=trace_id,
        )
    except BadRequestError as exc:
        message = str(exc).strip() or "Invalid media ingest job request"
        normalized = message.lower()
        status_code = (
            status.HTTP_429_TOO_MANY_REQUESTS
            if "concurrent job limit" in normalized
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=message) from exc


def _principal_has_admin_claims(principal: AuthPrincipal) -> bool:
    roles = {
        str(role).strip().lower()
        for role in (principal.roles or [])
        if str(role).strip()
    }
    if "admin" in roles:
        return True
    permissions = {
        str(permission).strip().lower()
        for permission in (principal.permissions or [])
        if str(permission).strip()
    }
    return bool(permissions & _ADMIN_CLAIM_PERMISSIONS)


def _job_to_status(job: dict[str, Any]) -> MediaIngestJobStatus:
    payload = _normalize_payload(job.get("payload"))
    id_value = job.get("id")
    if id_value is None:
        raise ValueError(f"Missing job id in job: {job!r}")
    try:
        job_id = int(id_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid job id {id_value!r} in job: {job!r}") from exc
    return MediaIngestJobStatus(
        id=job_id,
        uuid=job.get("uuid"),
        status=job.get("status"),
        job_type=job.get("job_type"),
        owner_user_id=job.get("owner_user_id"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        cancelled_at=job.get("cancelled_at"),
        cancellation_reason=job.get("cancellation_reason"),
        progress_percent=job.get("progress_percent"),
        progress_message=job.get("progress_message"),
        result=job.get("result"),
        error_message=job.get("error_message"),
        media_type=payload.get("media_type"),
        source=payload.get("source"),
        source_kind=payload.get("source_kind"),
        batch_id=payload.get("batch_id"),
    )


def _collect_jobs_for_batch(
    *,
    jm: JobManager,
    batch_id: str,
    owner_filter: str | None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    if not batch_id:
        return []

    matched = jm.list_jobs(
        domain="media_ingest",
        owner_user_id=owner_filter,
        batch_group=batch_id,
        limit=limit,
        sort_by="created_at",
        sort_order="desc",
    )
    if len(matched) >= limit:
        return matched[:limit]

    page_limit = min(500, max(limit, 100))
    seen_ids = {int(job.get("id")) for job in matched if job.get("id") is not None}
    cursor_created_at: datetime | None = None
    cursor_id: int | None = None
    last_cursor: tuple[datetime, int] | None = None

    while len(matched) < limit:
        jobs = jm.list_jobs(
            domain="media_ingest",
            owner_user_id=owner_filter,
            limit=page_limit,
            created_before=cursor_created_at,
            before_id=cursor_id,
            sort_by="created_at",
            sort_order="desc",
        )
        if not jobs:
            break
        for job in jobs:
            raw_job_id = job.get("id")
            if raw_job_id is not None and int(raw_job_id) in seen_ids:
                continue
            payload = _normalize_payload(job.get("payload"))
            if str(payload.get("batch_id") or "") == batch_id:
                matched.append(job)
                if raw_job_id is not None:
                    seen_ids.add(int(raw_job_id))
                if len(matched) >= limit:
                    return matched[:limit]
        if len(jobs) < page_limit:
            break
        last_job = jobs[-1]
        next_created_at = _parse_job_created_at(last_job.get("created_at"))
        next_id_raw = last_job.get("id")
        if next_created_at is None or next_id_raw is None:
            break
        next_cursor = (next_created_at, int(next_id_raw))
        if last_cursor == next_cursor:
            break
        last_cursor = next_cursor
        cursor_created_at, cursor_id = next_cursor

    return matched[:limit]


def _batch_exists_for_any_owner(
    *,
    jm: JobManager,
    batch_id: str,
) -> bool:
    if not batch_id:
        return False
    jobs = _collect_jobs_for_batch(
        jm=jm,
        batch_id=batch_id,
        owner_filter=None,
        limit=1,
    )
    return bool(jobs)


def _resolve_batch_or_session_id(
    *,
    batch_id: str | None,
    session_id: str | None,
) -> str:
    resolved = str(batch_id or session_id or "").strip()
    if resolved:
        return resolved
    raise HTTPException(status_code=400, detail="Either batch_id or session_id is required")


@router.post(
    "/ingest/jobs",
    response_model=SubmitMediaIngestJobsResponse,
    summary="Submit async media ingestion jobs (one job per item)",
    tags=["Media Ingestion Jobs"],
    dependencies=[
        Depends(require_permissions(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
        Depends(guard_storage_quota),
        # Pessimistic pre-check: verifies at least 1 MB of storage quota
        # remains.  Actual size is unknown until after ingestion completes,
        # so the real usage is recorded post-ingestion by the job worker.
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
)
async def submit_media_ingest_jobs(
    request: Request,
    form_data: AddMediaForm = Depends(get_add_media_form),
    files: list[UploadFile] | None = File(None, description="Optional media uploads"),
    current_user: User = Depends(get_request_user),
    jm: JobManager = Depends(get_job_manager),
) -> SubmitMediaIngestJobsResponse:
    rid = ensure_request_id(request) if request is not None else None
    tp = ensure_traceparent(request) if request is not None else ""

    # Normalize sentinel urls=[''] from some clients.
    if form_data.urls and form_data.urls == [""]:
        form_data.urls = None

    _validate_submit_inputs(form_data.media_type, form_data.urls, files)

    options = form_data.model_dump(mode="json")
    options.pop("urls", None)
    options.pop("keywords", None)
    selected_queue = _resolve_media_ingest_queue(form_data)

    batch_id = str(uuid4())
    jobs: list[MediaIngestJobItem] = []
    errors: list[str] = []

    url_list = form_data.urls or []
    for url in url_list:
        if not url or not str(url).strip():
            continue
        payload = {
            "batch_id": batch_id,
            "media_type": str(form_data.media_type),
            "source": str(url).strip(),
            "source_kind": "url",
            "input_ref": str(url).strip(),
            "options": options,
        }
        row = _create_media_ingest_job(
            jm=jm,
            selected_queue=selected_queue,
            payload=payload,
            current_user=current_user,
            batch_id=batch_id,
            request_id=rid,
            trace_id=tp or None,
        )
        row_id = row.get("id")
        if row_id is None:
            raise ValueError(f"Job creation returned no id: {row!r}")
        jobs.append(
            MediaIngestJobItem(
                id=int(row_id),
                uuid=row.get("uuid"),
                source=payload["source"],
                source_kind="url",
                status=row.get("status"),
            )
        )

    if files:
        for upload in files:
            temp_dir_path = None
            try:
                with TempDirManager(prefix="media_ingest_job_", cleanup=False) as temp_dir:
                    temp_dir_path = str(temp_dir)
                    saved_files, file_errors = await save_uploaded_files(
                        [upload],
                        temp_dir=temp_dir,
                        validator=file_validator_instance,
                        expected_media_type_key=str(form_data.media_type),
                    )
                if file_errors:
                    for err in file_errors:
                        msg = err.get("error") or "Failed to stage upload"
                        errors.append(msg)
                    if temp_dir_path:
                        _cleanup_dir(temp_dir_path)
                    continue
                if not saved_files:
                    errors.append("Failed to stage upload")
                    if temp_dir_path:
                        _cleanup_dir(temp_dir_path)
                    continue

                saved = saved_files[0]
                source_path = str(saved.get("path"))
                original_filename = saved.get("original_filename")
                input_ref = saved.get("input_ref") or original_filename or source_path

                payload = {
                    "batch_id": batch_id,
                    "media_type": str(form_data.media_type),
                    "source": source_path,
                    "source_kind": "file",
                    "input_ref": input_ref,
                    "original_filename": original_filename,
                    "temp_dir": temp_dir_path,
                    "cleanup_temp_dir": True,
                    "options": options,
                }
                row = _create_media_ingest_job(
                    jm=jm,
                    selected_queue=selected_queue,
                    payload=payload,
                    current_user=current_user,
                    batch_id=batch_id,
                    request_id=rid,
                    trace_id=tp or None,
                )
                row_id = row.get("id")
                if row_id is None:
                    raise ValueError(f"Job creation returned no id: {row!r}")
                jobs.append(
                    MediaIngestJobItem(
                        id=int(row_id),
                        uuid=row.get("uuid"),
                        source=source_path,
                        source_kind="file",
                        status=row.get("status"),
                    )
                )
            except HTTPException:
                raise
            except Exception as exc:
                logger.warning("Failed to stage upload for ingest jobs: {}", exc)
                errors.append(f"Upload staging failed: {exc}")
                if temp_dir_path:
                    _cleanup_dir(temp_dir_path)

    if not jobs:
        if errors:
            return JSONResponse(
                status_code=status.HTTP_207_MULTI_STATUS,
                content=SubmitMediaIngestJobsResponse(
                    batch_id=batch_id,
                    jobs=[],
                    errors=errors,
                ).model_dump(),
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid media sources supplied.",
        )

    return SubmitMediaIngestJobsResponse(batch_id=batch_id, jobs=jobs, errors=errors)


@router.get(
    "/ingest/jobs/{job_id}",
    response_model=MediaIngestJobStatus,
    summary="Get media ingest job status",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def get_media_ingest_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> MediaIngestJobStatus:
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "media_ingest":
        raise HTTPException(status_code=404, detail="Job not found")

    owner = str(job.get("owner_user_id") or "")
    if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
        raise HTTPException(status_code=403, detail="Not authorized for this job")

    return _job_to_status(job)


@router.get(
    "/ingest/jobs",
    response_model=MediaIngestJobListResponse,
    summary="List media ingest jobs for a batch",
    tags=["Media Ingestion Jobs"],
)
async def list_media_ingest_jobs(
    batch_id: str = Query(..., min_length=1, description="Batch identifier from submit response"),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    _: None = Depends(check_rate_limit),
    jm: JobManager = Depends(get_job_manager),
) -> MediaIngestJobListResponse:
    owner_filter = None if _principal_has_admin_claims(principal) else str(current_user.id)
    indexed_jobs = jm.list_jobs(
        domain="media_ingest",
        owner_user_id=owner_filter,
        batch_group=batch_id,
        limit=limit,
        sort_by="created_at",
        sort_order="desc",
    )
    indexed_statuses = [_job_to_status(job) for job in indexed_jobs[:limit]]
    if len(indexed_statuses) >= limit:
        return MediaIngestJobListResponse(batch_id=batch_id, jobs=indexed_statuses[:limit])

    # Backward compatibility: legacy rows may only have payload.batch_id.
    legacy_jobs = _collect_jobs_for_batch(
        jm=jm,
        batch_id=batch_id,
        owner_filter=owner_filter,
        limit=limit,
    )
    merged: list[MediaIngestJobStatus] = list(indexed_statuses)
    seen_ids = {int(item.id) for item in merged}
    for job in legacy_jobs:
        raw_job_id = job.get("id")
        if raw_job_id is None:
            continue
        job_id = int(raw_job_id)
        if job_id in seen_ids:
            continue
        merged.append(_job_to_status(job))
        seen_ids.add(job_id)
        if len(merged) >= limit:
            break

    return MediaIngestJobListResponse(batch_id=batch_id, jobs=merged[:limit])


@router.get(
    "/ingest/jobs/events/stream",
    summary="Stream media ingest job events (SSE)",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def stream_media_ingest_job_events(
    request: Request,
    batch_id: str | None = Query(
        None,
        min_length=1,
        description="Optional batch identifier to scope events to a single submit response",
    ),
    after_id: int = Query(0, ge=0),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> StreamingResponse:
    assert_may_start_work(request.app, "media.ingest.jobs.events.stream")
    is_admin = _principal_has_admin_claims(principal)
    owner_filter = None if is_admin else str(current_user.id)

    tracked_jobs: list[dict[str, Any]]
    if batch_id:
        tracked_jobs = _collect_jobs_for_batch(
            jm=jm,
            batch_id=batch_id,
            owner_filter=owner_filter,
        )
        if not tracked_jobs and not is_admin and _batch_exists_for_any_owner(jm=jm, batch_id=batch_id):
            raise HTTPException(status_code=403, detail="Not authorized for this batch")
    else:
        tracked_jobs = jm.list_jobs(
            domain="media_ingest",
            owner_user_id=owner_filter,
            limit=200,
            sort_by="created_at",
            sort_order="desc",
        )

    tracked_job_ids = {int(job.get("id")) for job in tracked_jobs if job.get("id") is not None}

    poll_interval = float(os.getenv("JOBS_EVENTS_POLL_INTERVAL", "1.0") or "1.0")
    max_duration_s: float | None = None
    try:
        if is_test_mode():
            max_duration_s = float(os.getenv("JOBS_SSE_TEST_MAX_SECONDS", "1.0") or "1.0")
    except (OSError, ValueError, TypeError):
        max_duration_s = 1.0

    stream = SSEStream(
        heartbeat_interval_s=poll_interval,
        heartbeat_mode="data",
        max_duration_s=max_duration_s,
        labels={"component": "jobs", "endpoint": "media_ingest_events_sse"},
    )

    async def _producer() -> None:
        nonlocal after_id
        snapshot_jobs = [_job_to_status(job).model_dump(mode="json") for job in tracked_jobs]
        await stream.send_event(
            "snapshot",
            {
                "domain": "media_ingest",
                "batch_id": batch_id,
                "jobs": snapshot_jobs,
            },
        )

        while True:
            try:
                if getattr(stream, "_closed", False):
                    break
            except (AttributeError, RuntimeError):
                pass

            conn = jm._connect()
            rows: list[Any] = []
            try:
                if jm.backend == "postgres":
                    with jm._pg_cursor(conn) as cur:
                        query = (
                            "SELECT id, job_id, event_type, attrs_json, owner_user_id "
                            "FROM job_events WHERE id > %s AND domain = %s"
                        )
                        params: list[Any] = [int(after_id), "media_ingest"]
                        if owner_filter is not None:
                            query += " AND owner_user_id = %s"
                            params.append(owner_filter)
                        query += " ORDER BY id ASC LIMIT 500"
                        cur.execute(query, tuple(params))
                        rows = cur.fetchall() or []
                else:
                    query = (
                        "SELECT id, job_id, event_type, attrs_json, owner_user_id "
                        "FROM job_events WHERE id > ? AND domain = ?"
                    )
                    params_sqlite: list[Any] = [int(after_id), "media_ingest"]
                    if owner_filter is not None:
                        query += " AND owner_user_id = ?"
                        params_sqlite.append(owner_filter)
                    query += " ORDER BY id ASC LIMIT 500"
                    rows = conn.execute(query, tuple(params_sqlite)).fetchall() or []
            except (OSError, RuntimeError, TypeError, ValueError):
                rows = []
            finally:
                with contextlib.suppress(OSError, RuntimeError, TypeError, ValueError):
                    conn.close()

            if rows:
                for row in rows:
                    if isinstance(row, dict):
                        event_id = int(row.get("id"))
                        job_id = int(row.get("job_id"))
                        event_type = str(row.get("event_type"))
                        attrs_raw = row.get("attrs_json")
                    else:
                        event_id = int(row[0])
                        job_id = int(row[1])
                        event_type = str(row[2])
                        attrs_raw = row[3]
                    if tracked_job_ids and job_id not in tracked_job_ids:
                        after_id = event_id
                        continue
                    try:
                        attrs = json.loads(attrs_raw) if isinstance(attrs_raw, str) else (attrs_raw or {})
                    except (TypeError, ValueError):
                        attrs = {}
                    await stream.send_event(
                        "job",
                        {
                            "event_id": event_id,
                            "job_id": job_id,
                            "event_type": event_type,
                            "attrs": attrs,
                        },
                        event_id=str(event_id),
                    )
                    after_id = event_id

            if tracked_job_ids:
                refreshed = [jm.get_job(job_id) for job_id in tracked_job_ids]
                if all(
                    (job or {}).get("status") in {"completed", "failed", "cancelled", "quarantined"}
                    for job in refreshed
                ) and not rows:
                    break

            await asyncio.sleep(poll_interval)

    async def _gen():
        producer = asyncio.create_task(_producer())
        try:
            async for line in stream.iter_sse():
                yield line
        finally:
            if not producer.done():
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    producer.cancel()
                with contextlib.suppress(asyncio.CancelledError, RuntimeError, OSError):
                    await producer

    return StreamingResponse(
        _gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete(
    "/ingest/jobs/{job_id}",
    response_model=CancelMediaIngestJobResponse,
    summary="Cancel a media ingest job",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def cancel_media_ingest_job(
    job_id: int,
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
    reason: str | None = Query(None, description="Reason for cancellation"),
) -> CancelMediaIngestJobResponse:
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "media_ingest":
        raise HTTPException(status_code=404, detail="Job not found")

    owner = str(job.get("owner_user_id") or "")
    if not (_principal_has_admin_claims(principal) or owner == str(current_user.id)):
        raise HTTPException(status_code=403, detail="Not authorized for this job")

    status_val = str(job.get("status") or "").lower()
    if status_val in {"completed", "failed", "cancelled", "quarantined"}:
        raise HTTPException(status_code=400, detail="Cannot cancel terminal job")

    ok = jm.cancel_job(int(job_id), reason=reason)
    if not ok:
        raise HTTPException(status_code=400, detail="Cancellation failed")

    return CancelMediaIngestJobResponse(
        success=True,
        job_id=int(job_id),
        status="cancelled",
        message="Job cancellation requested",
    )


@router.post(
    "/ingest/jobs/cancel",
    response_model=CancelMediaIngestBatchResponse,
    summary="Cancel media ingest jobs by batch/session id",
    tags=["Media Ingestion Jobs"],
    dependencies=[Depends(check_rate_limit)],
)
async def cancel_media_ingest_jobs_batch(
    batch_id: str | None = Query(None, min_length=1, description="Batch identifier to cancel"),
    session_id: str | None = Query(
        None,
        min_length=1,
        description="Session identifier alias for batch-level cancellation",
    ),
    reason: str | None = Query(None, description="Reason for cancellation"),
    current_user: User = Depends(get_request_user),
    principal: AuthPrincipal = Depends(get_auth_principal),
    jm: JobManager = Depends(get_job_manager),
) -> CancelMediaIngestBatchResponse:
    resolved_batch_id = _resolve_batch_or_session_id(batch_id=batch_id, session_id=session_id)
    is_admin = _principal_has_admin_claims(principal)
    owner_filter = None if is_admin else str(current_user.id)

    matched_jobs = _collect_jobs_for_batch(
        jm=jm,
        batch_id=resolved_batch_id,
        owner_filter=owner_filter,
        limit=5000,
    )
    if not matched_jobs:
        if not is_admin and _batch_exists_for_any_owner(jm=jm, batch_id=resolved_batch_id):
            raise HTTPException(status_code=403, detail="Not authorized for this batch")
        raise HTTPException(status_code=404, detail="Batch not found")

    requested = len(matched_jobs)
    cancelled = 0
    already_terminal = 0
    failed = 0

    for job in matched_jobs:
        raw_job_id = job.get("id")
        if raw_job_id is None:
            failed += 1
            continue
        job_id = int(raw_job_id)
        status_value = str(job.get("status") or "").lower()
        if status_value in _TERMINAL_MEDIA_INGEST_JOB_STATUSES:
            already_terminal += 1
            continue
        if jm.cancel_job(job_id, reason=reason):
            cancelled += 1
            continue

        refreshed = jm.get_job(job_id) or {}
        refreshed_status = str(refreshed.get("status") or "").lower()
        if refreshed_status in _TERMINAL_MEDIA_INGEST_JOB_STATUSES:
            already_terminal += 1
        else:
            failed += 1

    if cancelled > 0:
        message = f"Cancellation requested for {cancelled} job(s)"
    elif already_terminal == requested:
        message = "All jobs already terminal"
    else:
        message = "No jobs were cancelled"

    return CancelMediaIngestBatchResponse(
        success=failed == 0,
        batch_id=resolved_batch_id,
        requested=requested,
        cancelled=cancelled,
        already_terminal=already_terminal,
        failed=failed,
        message=message,
    )


__all__ = ["router"]
