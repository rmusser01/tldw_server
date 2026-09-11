"""Jobs worker for Presentation Studio video render requests."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Slides.presentation_rendering import (
    PresentationRenderError,
    load_presentation_render_snapshot,
    render_presentation_video,
)
from tldw_Server_API.app.core.Slides.presentation_service import STRUCTURED_SLIDES
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase

_DOMAIN = "presentation_render"
_JOB_TYPE = "presentation_render"


@dataclass
class _ProgressState:
    """Mutable progress snapshot reported back to the Jobs worker SDK."""

    percent: float | None = None
    message: str | None = None


class PresentationRenderJobError(RuntimeError):
    """Normalized worker error with retry metadata for presentation renders."""

    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


def _jobs_manager() -> JobManager:
    """Create a Jobs manager using the configured backend for render jobs."""

    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort integer coercion for worker configuration environment values."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _build_worker_config(*, worker_id: str, queue: str) -> WorkerConfig:
    """Build the worker SDK configuration for presentation render jobs."""

    return WorkerConfig(
        domain=_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=_coerce_int(os.getenv("PRESENTATION_RENDER_JOBS_LEASE_SECONDS"), 180),
        renew_jitter_seconds=_coerce_int(os.getenv("PRESENTATION_RENDER_JOBS_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=_coerce_int(os.getenv("PRESENTATION_RENDER_JOBS_RENEW_THRESHOLD_SECONDS"), 20),
        backoff_base_seconds=_coerce_int(os.getenv("PRESENTATION_RENDER_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("PRESENTATION_RENDER_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("PRESENTATION_RENDER_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )


def _resolve_queue_name() -> str:
    """Normalize the configured render queue name to a supported Jobs queue."""

    configured_queue = (os.getenv("PRESENTATION_RENDER_JOBS_QUEUE") or "").strip().lower()
    if configured_queue in {"default", "high", "low"}:
        return configured_queue
    if configured_queue.endswith("-high"):
        return "high"
    if configured_queue.endswith("-low"):
        return "low"
    if configured_queue.endswith("-default"):
        return "default"
    return "default"


def _normalize_payload(value: Any) -> dict[str, Any]:
    """Normalize stored job payloads into a dictionary for worker processing."""

    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> int:
    """Resolve the owning user ID from the job payload or owner metadata."""

    owner = payload.get("user_id") or job.get("owner_user_id")
    try:
        return int(owner)
    except (TypeError, ValueError) as exc:
        raise PresentationRenderJobError("missing owner_user_id", retryable=False) from exc


def _create_slides_db(user_id: int) -> SlidesDatabase:
    """Open the Slides database for the job owner."""

    return SlidesDatabase(
        db_path=str(DatabasePaths.get_slides_db_path(user_id)),
        client_id=str(user_id),
    )


async def process_presentation_render_job(
    job: dict[str, Any],
    *,
    job_manager: JobManager,
    worker_id: str = "presentation-render-worker",
    progress: _ProgressState | None = None,
) -> dict[str, Any]:
    """Render and persist one presentation render job."""

    payload = _normalize_payload(job.get("payload"))
    if str(job.get("job_type") or "").lower() != _JOB_TYPE:
        raise PresentationRenderJobError("unsupported job_type", retryable=False)

    user_id = _resolve_user_id(job, payload)
    presentation_id = str(payload.get("presentation_id") or "").strip()
    if not presentation_id:
        raise PresentationRenderJobError("presentation_id_required", retryable=False)
    try:
        presentation_version = int(payload.get("presentation_version"))
    except (TypeError, ValueError) as exc:
        raise PresentationRenderJobError("presentation_version_required", retryable=False) from exc
    output_format = str(payload.get("format") or "").strip().lower()
    if not output_format:
        raise PresentationRenderJobError("presentation_render_format_invalid", retryable=False)

    if progress is not None:
        progress.percent = 10.0
        progress.message = "load_snapshot"
    job_manager.update_job_progress(int(job["id"]), progress_percent=10.0, progress_message="load_snapshot")

    slides_db = _create_slides_db(user_id)
    try:
        try:
            kind = slides_db.get_presentation_kind(presentation_id)
        except KeyError as exc:
            raise PresentationRenderJobError(
                "presentation_render_version_not_found",
                retryable=False,
            ) from exc
        if kind.content_kind != STRUCTURED_SLIDES:
            raise PresentationRenderJobError(
                "operation_not_supported_for_content_kind",
                retryable=False,
            )
        snapshot = load_presentation_render_snapshot(
            slides_db,
            presentation_id=presentation_id,
            presentation_version=presentation_version,
        )
    except PresentationRenderError as exc:
        raise PresentationRenderJobError(exc.code, retryable=getattr(exc, "retryable", False)) from exc
    finally:
        slides_db.close_connection()

    if progress is not None:
        progress.percent = 60.0
        progress.message = "render_video"
    job_manager.update_job_progress(int(job["id"]), progress_percent=60.0, progress_message="render_video")

    outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
    try:
        render_result = await asyncio.to_thread(
            render_presentation_video,
            presentation_id=snapshot.presentation_id,
            presentation_version=snapshot.presentation_version,
            title=snapshot.title,
            slides=snapshot.slides,
            output_format=output_format,
            output_dir=outputs_dir,
            user_id=user_id,
        )
    except PresentationRenderError as exc:
        raise PresentationRenderJobError(exc.code, retryable=getattr(exc, "retryable", False)) from exc

    if progress is not None:
        progress.percent = 90.0
        progress.message = "persist_artifact"
    job_manager.update_job_progress(int(job["id"]), progress_percent=90.0, progress_message="persist_artifact")

    metadata_json = json.dumps(
        {
            "origin": "presentation_studio",
            "presentation_id": snapshot.presentation_id,
            "presentation_version": snapshot.presentation_version,
            "output_format": render_result.output_format,
            "slide_count": len(snapshot.slides),
            "theme": snapshot.theme,
            "byte_size": render_result.byte_size,
        },
        ensure_ascii=False,
    )
    with CollectionsDatabase.for_user(user_id=str(user_id)) as collections_db:
        artifact = collections_db.create_output_artifact(
            job_id=int(job["id"]),
            type_="presentation_render",
            title=f"{snapshot.title} ({render_result.output_format})",
            format_=render_result.output_format,
            storage_path=render_result.storage_path,
            metadata_json=metadata_json,
        )

    return {
        "presentation_id": snapshot.presentation_id,
        "presentation_version": snapshot.presentation_version,
        "format": render_result.output_format,
        "output_id": artifact.id,
        "download_url": f"/api/v1/outputs/{artifact.id}/download",
    }


async def run_presentation_render_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Run the long-lived Presentation Studio render worker until stopped."""

    queue_name = _resolve_queue_name()
    worker_id = (os.getenv("PRESENTATION_RENDER_JOBS_WORKER_ID") or "presentation-render-worker").strip()
    cfg = _build_worker_config(worker_id=worker_id, queue=queue_name)
    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    progress = _ProgressState()

    async def _handle_job(job: dict[str, Any]) -> dict[str, Any]:
        return await process_presentation_render_job(
            job,
            job_manager=jm,
            worker_id=worker_id,
            progress=progress,
        )

    def _progress_cb() -> dict[str, Any]:
        update: dict[str, Any] = {}
        if progress.percent is not None:
            update["progress_percent"] = progress.percent
        if progress.message:
            update["progress_message"] = progress.message
        return update

    async def _watch_stop() -> None:
        if stop_event is None:
            return
        await stop_event.wait()
        sdk.stop()

    logger.info("Starting Presentation Render Jobs worker")
    stop_task = asyncio.create_task(_watch_stop())
    try:
        await sdk.run(handler=_handle_job, progress_cb=_progress_cb)
    finally:
        stop_task.cancel()
