from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import tempfile
import threading
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse

from cachetools import LRUCache
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

try:
    # Pydantic v2
    from pydantic import model_validator  # type: ignore
except ImportError:  # pragma: no cover - fallback for older environments
    model_validator = None  # type: ignore
import contextlib

from loguru import logger
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal, get_request_user, RequirePermission, RequireRole, User
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_cursor_pagination_meta
from tldw_Server_API.app.api.v1.schemas.pagination import CursorPaginationMeta

from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_MAINTENANCE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.DB_Management.backends.base import (
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.Audio_Transcription_Lib import (
    _get_allowed_media_base_dirs,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.Audio.stt_provider_adapter import (
    resolve_default_transcription_model,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Logging.log_context import ensure_request_id, ensure_traceparent, get_ps_logger
from tldw_Server_API.app.core.Streaming.streams import SSEStream
from tldw_Server_API.app.core.Usage.audio_quota import (
    TIER_LIMITS,
    get_limits_for_user,
    get_user_tier,
    set_user_tier,
)

router = APIRouter()

MAX_CACHED_JOB_MANAGER_INSTANCES = 4
_job_manager_cache: LRUCache = LRUCache(maxsize=MAX_CACHED_JOB_MANAGER_INSTANCES)
_job_manager_lock = threading.Lock()
_ADMIN_CLAIM_PERMISSIONS = frozenset({"*", "system.configure"})

_ADMIN_DEPS = [
    Depends(RequireRole("admin")),
    Depends(RequirePermission(SYSTEM_MAINTENANCE)),
]

_AUDIO_JOBS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
    BackendDatabaseError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    json.JSONDecodeError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)
_AUDIO_JOBS_CURSOR_VERSION = 1


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


def _path_is_within(path: Path, base_dir: Path) -> bool:
    try:
        path.relative_to(base_dir)
        return True
    except ValueError:
        return False


def _validate_audio_job_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("url must use http or https scheme")
    return url


def _validate_audio_job_local_path(local_path: str) -> str:
    path = Path(local_path)
    if not path.is_absolute():
        raise ValueError("local_path must be an absolute path")
    resolved = path.resolve(strict=False)
    allowed_roots = _get_allowed_media_base_dirs()
    if not any(_path_is_within(resolved, root) for root in allowed_roots):
        allowed_str = ", ".join(str(root) for root in allowed_roots) or "<none configured>"
        raise ValueError(f"local_path must resolve under one of the allowed base directories: {allowed_str}")
    return str(resolved)


def _encode_audio_jobs_cursor(created_at: str, job_id: int) -> str:
    datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    payload = {
        "v": _AUDIO_JOBS_CURSOR_VERSION,
        "created_at": created_at,
        "id": int(job_id),
    }
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_audio_jobs_cursor(cursor: str) -> tuple[datetime, int]:
    if not cursor:
        raise ValueError("empty cursor")
    try:
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode((cursor + padding).encode("ascii")).decode("utf-8")
        payload = json.loads(raw)
        if int(payload.get("v", 0)) != _AUDIO_JOBS_CURSOR_VERSION:
            raise ValueError("unsupported cursor version")
        created_at = str(payload["created_at"])
        job_id = int(payload["id"])
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")), job_id
    except (
        AttributeError,
        binascii.Error,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        UnicodeEncodeError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        raise ValueError("invalid cursor") from exc


def get_job_manager() -> JobManager:
    """
    Dependency helper to construct a JobManager using JOBS_DB_URL when set.

    Reuse a cached JobManager per JOBS_DB_URL. When JOBS_DB_URL is a Postgres
    DSN, use the Postgres backend; otherwise fall back to JobManager's default
    backend resolution (typically SQLite).
    """
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    cache_key = db_url or "default"
    with _job_manager_lock:
        cached = _job_manager_cache.get(cache_key)
        if cached is not None:
            return cached

        if not db_url:
            # Backwards-compatible default: rely on JobManager's internal selection
            # (typically a local SQLite database) when JOBS_DB_URL is not provided.
            logger.debug("JOBS_DB_URL not set; using JobManager default backend (likely SQLite).")
            job_manager = JobManager()
        else:
            backend = "postgres" if db_url.startswith("postgres") else None
            job_manager = JobManager(backend=backend, db_url=db_url)

        _job_manager_cache[cache_key] = job_manager
        return job_manager


class SubmitAudioJobRequest(BaseModel):
    url: str | None = Field(None, description="URL to download audio from")
    local_path: str | None = Field(None, description="Server-local path to an existing audio/video file")
    model: str | None = Field(
        None,
        description="Transcription model selector (defaults to config when omitted)",
    )
    hotwords: str | list[str] | None = Field(
        None,
        description="Optional hotwords to guide transcription (CSV/JSON string or list). Primarily used by VibeVoice-ASR.",
    )
    perform_chunking: bool = Field(True, description="Whether to chunk the transcript after STT")
    perform_analysis: bool = Field(False, description="Whether to run LLM analysis after chunking")
    api_name: str | None = Field(None, description="LLM provider key for analysis stage")

    if model_validator is not None:
        @model_validator(mode="before")
        def _validate_inputs(cls, values: Any) -> Any:  # type: ignore[override]
            if not isinstance(values, dict):
                return values
            url = (values.get("url") or "").strip()
            lp = (values.get("local_path") or "").strip()
            if not url and not lp:
                raise ValueError("Either 'url' or 'local_path' must be provided")
            if url and lp:
                raise ValueError("Provide only one of 'url' or 'local_path'")
            if url:
                values["url"] = _validate_audio_job_url(url)
            if lp:
                values["local_path"] = _validate_audio_job_local_path(lp)
            return values
    else:
        # Backward-compatible path for environments still on Pydantic v1
        from pydantic import root_validator as _rv  # type: ignore

        @_rv(pre=True)
        def _validate_inputs(self, values: dict[str, Any]) -> dict[str, Any]:  # type: ignore[no-redef]
            url = (values.get("url") or "").strip()
            lp = (values.get("local_path") or "").strip()
            if not url and not lp:
                raise ValueError("Either 'url' or 'local_path' must be provided")
            if url and lp:
                raise ValueError("Provide only one of 'url' or 'local_path'")
            if url:
                values["url"] = _validate_audio_job_url(url)
            if lp:
                values["local_path"] = _validate_audio_job_local_path(lp)
            return values


class SubmitAudioJobResponse(BaseModel):
    id: int
    uuid: str | None = None
    domain: str
    queue: str
    job_type: str
    status: str


@router.post("/jobs/submit", response_model=SubmitAudioJobResponse, summary="Submit an audio processing job")
async def submit_audio_job(
    req: SubmitAudioJobRequest,
    current_user: Annotated[User, Depends(get_request_user)],
    jm: Annotated[JobManager, Depends(get_job_manager)],
    request: Request,
):
    """
    Create an audio job in the Jobs queue. First stage is determined by input:
    - url → audio_download
    - local_path → audio_convert
    """
    # Correlation IDs
    rid = ensure_request_id(request) if request is not None else None
    tp = ensure_traceparent(request) if request is not None else ""

    try:
        requested_model = (req.model or "").strip()
        if not requested_model:
            requested_model = resolve_default_transcription_model("whisper-1")

        payload: dict[str, Any] = {
            "model": requested_model,
            "hotwords": req.hotwords,
            "perform_chunking": bool(req.perform_chunking),
            "perform_analysis": bool(req.perform_analysis),
            "api_name": req.api_name,
        }
        job_type = "audio_download" if (req.url and req.url.strip()) else "audio_convert"
        if job_type == "audio_download":
            payload["url"] = req.url.strip()
            payload["temp_dir"] = os.getenv("AUDIO_JOBS_TEMP", tempfile.gettempdir())
        else:
            payload["local_path"] = req.local_path.strip()

        row = jm.create_job(
            domain="audio",
            queue="default",
            job_type=job_type,
            payload=payload,
            owner_user_id=str(current_user.id),
            priority=5,
            max_retries=3,
            request_id=rid,
        )
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio", traceparent=tp).info(
            "Submitted audio job: job_id=%s type=%s", row.get("id"), job_type
        )
        return SubmitAudioJobResponse(
            id=int(row.get("id")),
            uuid=row.get("uuid"),
            domain=row.get("domain"),
            queue=row.get("queue"),
            job_type=row.get("job_type"),
            status=row.get("status"),
        )
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        get_ps_logger(request_id=rid, ps_component="endpoint", ps_job_kind="audio", traceparent=tp).error(
            "Failed to submit audio job: %s", e
        )
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to submit job") from e


class AudioJob(BaseModel):
    id: int
    uuid: str | None
    job_type: str
    status: str
    priority: int
    retry_count: int
    max_retries: int
    owner_user_id: str | None
    available_at: str | None
    started_at: str | None
    leased_until: str | None
    created_at: str | None
    updated_at: str | None
    completed_at: str | None


_AUDIO_JOB_FIELD_MAP = getattr(AudioJob, "model_fields", None) or getattr(AudioJob, "__fields__", {})


@router.get("/jobs/{job_id}", response_model=AudioJob, summary="Get audio job status")
async def get_audio_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_request_user)],
    jm: Annotated[JobManager, Depends(get_job_manager)],
    principal: Annotated[AuthPrincipal, Depends(get_auth_principal)],
):
    """
    Return a single audio job if it belongs to the caller (or the caller is admin).
    """
    try:
        d = jm.get_job(int(job_id))
        if not d or str(d.get("domain") or "") != "audio":
            raise HTTPException(status_code=404, detail="Job not found")
        # Owner/admin check
        owner = str(d.get("owner_user_id") or "")
        is_admin = _principal_has_admin_claims(principal)
        if not (is_admin or owner == str(current_user.id)):
            raise HTTPException(status_code=403, detail="Not authorized for this job")
        return AudioJob(**{k: d.get(k) for k in _AUDIO_JOB_FIELD_MAP})
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS:
        logger.exception(
            f"Failed to fetch job: job_id={job_id}, user_id={getattr(current_user, 'id', None)}"
        )
        raise HTTPException(status_code=500, detail="Failed to fetch job") from None


@router.get(
    "/jobs/{job_id}/progress/stream",
    summary="Stream audio job progress (SSE)",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Server-sent events stream of audio job progress",
            "content": {"text/event-stream": {}},
        },
    },
)
async def stream_audio_job_progress(
    job_id: int,
    current_user: Annotated[User, Depends(get_request_user)],
    jm: Annotated[JobManager, Depends(get_job_manager)],
    request: Request,
    after_id: int = 0,
):
    job = jm.get_job(int(job_id))
    if not job or str(job.get("domain") or "") != "audio":
        raise HTTPException(status_code=404, detail="Job not found")
    owner = str(job.get("owner_user_id") or "")
    if owner != str(current_user.id):
        raise HTTPException(status_code=403, detail="Not authorized for this job")

    poll_interval = float(os.getenv("JOBS_EVENTS_POLL_INTERVAL", "1.0") or "1.0")
    stream = SSEStream(
        heartbeat_interval_s=poll_interval,
        heartbeat_mode="data",
        labels={"component": "jobs", "endpoint": "audio_job_progress_sse"},
    )

    async def _producer() -> None:
        nonlocal after_id
        try:
            snapshot = {
                "status": job.get("status"),
                "progress_percent": job.get("progress_percent"),
                "progress_message": job.get("progress_message"),
            }
            await stream.send_event("job", {"event": "job.snapshot", "attrs": snapshot})
        except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS:
            pass

        while True:
            try:
                if getattr(stream, "_closed", False):
                    break
            except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS:
                pass

            conn = jm._connect()
            rows = []
            try:
                if jm.backend == "postgres":
                    with jm._pg_cursor(conn) as cur:
                        cur.execute(
                            "SELECT id, event_type, attrs_json FROM job_events WHERE job_id = %s AND id > %s ORDER BY id ASC LIMIT 200",
                            (int(job_id), int(after_id)),
                        )
                        rows = cur.fetchall() or []
                else:
                    rows = conn.execute(
                        "SELECT id, event_type, attrs_json FROM job_events WHERE job_id = ? AND id > ? ORDER BY id ASC LIMIT 200",
                        (int(job_id), int(after_id)),
                    ).fetchall() or []
            except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS:
                rows = []
            finally:
                with contextlib.suppress(_AUDIO_JOBS_NONCRITICAL_EXCEPTIONS):
                    conn.close()

            if rows:
                for r in rows:
                    if isinstance(r, dict):
                        eid = int(r.get("id"))
                        et = str(r.get("event_type"))
                        attrs = r.get("attrs_json")
                    else:
                        eid = int(r[0])
                        et = str(r[1])
                        attrs = r[2]
                    try:
                        attrs_obj = json.loads(attrs) if isinstance(attrs, str) else (attrs or {})
                    except (TypeError, ValueError):
                        attrs_obj = {}
                    await stream.send_event("job", {"event": et, "attrs": attrs_obj}, event_id=str(eid))
                    after_id = eid

            job_row = jm.get_job(int(job_id))
            status_val = (job_row or {}).get("status")
            if status_val in {"completed", "failed", "cancelled"} and not rows:
                break

            await asyncio.sleep(poll_interval)

    async def _gen():
        prod_task = asyncio.create_task(_producer())
        try:
            async for ln in stream.iter_sse():
                yield ln
        finally:
            if not prod_task.done():
                with contextlib.suppress(_AUDIO_JOBS_NONCRITICAL_EXCEPTIONS):
                    prod_task.cancel()
                with contextlib.suppress(_AUDIO_JOBS_NONCRITICAL_EXCEPTIONS):
                    await prod_task

    sse_headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(_gen(), media_type="text/event-stream", headers=sse_headers)


class ListAudioJobsResponse(BaseModel):
    jobs: list[AudioJob]
    limit: int
    cursor: str | None = None
    next_cursor: str | None = None
    has_more: bool
    pagination: CursorPaginationMeta


@router.get(
    "/jobs/admin/list",
    response_model=ListAudioJobsResponse,
    summary="List audio jobs (admin)",
    dependencies=_ADMIN_DEPS,
)
async def list_audio_jobs_admin(
    jm: Annotated[JobManager, Depends(get_job_manager)],
    status_filter: Annotated[
        str | None,
        Query(description="Filter by status: queued|processing|completed|failed|cancelled"),
    ] = None,
    owner_user_id: Annotated[str | None, Query(description="Filter by owner user id")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    cursor: Annotated[str | None, Query(description="Opaque cursor from the previous page")] = None,
):
    try:
        created_before: datetime | None = None
        before_id: int | None = None
        if cursor is not None:
            try:
                created_before, before_id = _decode_audio_jobs_cursor(cursor)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid cursor") from exc

        requested_limit = int(limit)
        logger.info(
            'Admin list audio jobs: status_filter={}, owner_user_id={}, limit={}, cursor={}',
            status_filter,
            owner_user_id,
            requested_limit,
            bool(cursor),
        )
        jobs = jm.list_jobs(
            domain="audio",
            status=status_filter,
            owner_user_id=str(owner_user_id) if owner_user_id is not None else None,
            limit=requested_limit + 1,
            created_before=created_before,
            before_id=before_id,
            sort_by="created_at",
            sort_order="desc",
        )
        has_more = len(jobs) > requested_limit
        visible_jobs = jobs[:requested_limit]
        next_cursor = None
        if has_more and visible_jobs:
            last_job = visible_jobs[-1]
            next_cursor = _encode_audio_jobs_cursor(
                str(last_job.get("created_at")),
                int(last_job.get("id")),
            )
        # Project to model
        out = [AudioJob(**{k: j.get(k) for k in _AUDIO_JOB_FIELD_MAP}) for j in visible_jobs]
        return ListAudioJobsResponse(
            jobs=out,
            limit=requested_limit,
            cursor=cursor,
            next_cursor=next_cursor,
            has_more=bool(next_cursor),
            pagination=build_cursor_pagination_meta(
                limit=requested_limit,
                cursor=cursor,
                next_cursor=next_cursor,
            ),
        )
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to list jobs")
        raise HTTPException(status_code=500, detail="Failed to list jobs") from e


class AudioJobsSummary(BaseModel):
    counts_by_status: dict[str, int]
    total: int
    owner_user_id: str | None = None


@router.get(
    "/jobs/admin/summary",
    response_model=AudioJobsSummary,
    summary="Summarize audio jobs by status (admin)",
    dependencies=_ADMIN_DEPS,
)
async def summarize_audio_jobs_admin(
    jm: Annotated[JobManager, Depends(get_job_manager)],
    owner_user_id: str | None = Query(None, description="Optional owner filter"),
):
    try:
        logger.info(
            'Admin summarize audio jobs by status: owner_user_id={}',
            owner_user_id,
        )
        by_status = jm.summarize_by_status(
            domain="audio",
            owner_user_id=str(owner_user_id) if owner_user_id is not None else None,
        )
        total = sum(by_status.values())
        return AudioJobsSummary(counts_by_status=by_status, total=total, owner_user_id=owner_user_id)
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to summarize jobs")
        raise HTTPException(status_code=500, detail="Failed to summarize jobs") from e


class AudioJobsOwnerSummaryItem(BaseModel):
    owner_user_id: str | None
    status: str
    count: int


class AudioJobsSummaryByOwner(BaseModel):
    items: list[AudioJobsOwnerSummaryItem]


@router.get(
    "/jobs/admin/summary-by-owner",
    response_model=AudioJobsSummaryByOwner,
    summary="Summarize audio jobs by owner and status (admin)",
    dependencies=_ADMIN_DEPS,
)
async def summary_by_owner_admin(
    jm: Annotated[JobManager, Depends(get_job_manager)],
):
    try:
        items: list[AudioJobsOwnerSummaryItem] = []
        logger.info("Admin summarize audio jobs by owner and status")
        summary = jm.summarize_by_owner_and_status(domain="audio")
        for entry in summary:
            items.append(
                AudioJobsOwnerSummaryItem(
                    owner_user_id=entry.get("owner_user_id"),
                    status=str(entry.get("status") or ""),
                    count=int(entry.get("count") or 0),
                )
            )
        return AudioJobsSummaryByOwner(items=items)
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to summarize by owner")
        raise HTTPException(status_code=500, detail="Failed to summarize by owner") from e


class OwnerProcessingSummary(BaseModel):
    owner_user_id: str
    processing: int
    limit: int | None


@router.get(
    "/jobs/admin/owner/{owner_user_id}/processing",
    response_model=OwnerProcessingSummary,
    summary="Get owner's processing count and limit (admin)",
    dependencies=_ADMIN_DEPS,
)
async def owner_processing_summary(
    owner_user_id: int,
    jm: Annotated[JobManager, Depends(get_job_manager)],
    request: Request,
):
    try:
        logger.info(
            'Admin owner processing summary: owner_user_id={}',
            owner_user_id,
        )
        # Count processing jobs for this owner using JobManager APIs.
        processing = jm.count_jobs(
            domain="audio",
            status="processing",
            owner_user_id=str(owner_user_id),
        )
        # Limit
        # Correlate logs with request_id if available
        rid = ensure_request_id(request) if request is not None else None
        tp = ensure_traceparent(request) if request is not None else ""

        try:
            limits = await get_limits_for_user(owner_user_id)
        except (ValueError, KeyError, RuntimeError) as e:
            get_ps_logger(
                request_id=rid,
                ps_component="endpoint",
                ps_job_kind="audio",
                traceparent=tp,
            ).warning(
                "Failed to get limits for owner %s; returning limit=None: %s", owner_user_id, e
            )
            limits = None
        if limits is None:
            limit = None
        else:
            try:
                limit = int(limits.get("concurrent_jobs") or 0)
            except (ValueError, TypeError) as e:
                get_ps_logger(
                    request_id=rid,
                    ps_component="endpoint",
                    ps_job_kind="audio",
                    traceparent=tp,
                ).warning(
                    "Could not parse concurrent_jobs limit for owner %s; returning limit=None: %s", owner_user_id, e
                )
                limit = None
        return OwnerProcessingSummary(owner_user_id=str(owner_user_id), processing=processing, limit=limit)
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to get owner processing summary")
        raise HTTPException(status_code=500, detail="Failed to get owner processing summary") from e


# --- Admin: user tiers (audio) ---

class UserTierResponse(BaseModel):
    user_id: int
    tier: str


class SetUserTierRequest(BaseModel):
    tier: str = Field(..., description="Tier name (one of: free, standard, premium)")


@router.get(
    "/jobs/admin/tiers/{user_id}",
    response_model=UserTierResponse,
    summary="Get user's audio tier (admin)",
    dependencies=_ADMIN_DEPS,
)
async def get_user_tier_admin(user_id: int):
    try:
        tier = await get_user_tier(int(user_id))
        return UserTierResponse(user_id=int(user_id), tier=tier)
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to get user tier")
        raise HTTPException(status_code=500, detail="Failed to get user tier") from e


@router.put(
    "/jobs/admin/tiers/{user_id}",
    response_model=UserTierResponse,
    summary="Set user's audio tier (admin)",
    dependencies=_ADMIN_DEPS,
)
async def set_user_tier_admin(user_id: int, req: SetUserTierRequest):
    try:
        tier = req.tier.strip().lower()
        allowed = set(TIER_LIMITS.keys())
        if tier not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid tier. Allowed: {', '.join(sorted(allowed))}")
        await set_user_tier(int(user_id), tier)
        return UserTierResponse(user_id=int(user_id), tier=tier)
    except HTTPException:
        raise
    except _AUDIO_JOBS_NONCRITICAL_EXCEPTIONS as e:
        logger.error("Failed to set user tier")
        raise HTTPException(status_code=500, detail="Failed to set user tier") from e
