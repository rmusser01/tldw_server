from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.core.Chunking.templates import TemplateClassifier
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.DB_Manager import mark_media_as_processed
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import create_media_database
from tldw_Server_API.app.core.DB_Management.media_db.errors import ConflictError
from tldw_Server_API.app.core.Ingestion_Media_Processing.chunking_options import (
    apply_chunking_template_if_any,
    prepare_chunking_options_dict,
)
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import (
    process_batch_media,
    process_document_like_item,
)
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

_MEDIA_DOMAIN = "media_ingest"
_MEDIA_JOB_TYPE = "media_ingest_item"
_MARK_PROCESSED_CONFLICT_RETRIES = 3
_MARK_PROCESSED_CONFLICT_BACKOFF_SECONDS = 0.05


@dataclass
class _ProgressState:
    percent: float | None = None
    message: str | None = None


class MediaIngestJobError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


async def _mark_embeddings_complete_with_retry(*, db: Any, media_id: int, context: str) -> bool:
    for attempt in range(1, _MARK_PROCESSED_CONFLICT_RETRIES + 1):
        try:
            mark_media_as_processed(db_instance=db, media_id=media_id)
            return True
        except ConflictError as exc:
            if attempt >= _MARK_PROCESSED_CONFLICT_RETRIES:
                logger.warning(
                    "Embeddings completed for media {} but the ready-state update still conflicted after {} attempts in {}: {}",
                    media_id,
                    attempt,
                    context,
                    exc,
                )
                return False
            await asyncio.sleep(_MARK_PROCESSED_CONFLICT_BACKOFF_SECONDS * attempt)
    return False


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _build_worker_config(*, worker_id: str, queue: str) -> WorkerConfig:
    lease_seconds = _coerce_int(os.getenv("MEDIA_INGEST_JOBS_LEASE_SECONDS"), 120)
    renew_jitter_seconds = _coerce_int(os.getenv("MEDIA_INGEST_JOBS_RENEW_JITTER_SECONDS"), 5)
    renew_threshold_seconds = _coerce_int(os.getenv("MEDIA_INGEST_JOBS_RENEW_THRESHOLD_SECONDS"), 15)

    return WorkerConfig(
        domain=_MEDIA_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=lease_seconds,
        renew_jitter_seconds=renew_jitter_seconds,
        renew_threshold_seconds=renew_threshold_seconds,
        backoff_base_seconds=_coerce_int(os.getenv("MEDIA_INGEST_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("MEDIA_INGEST_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("MEDIA_INGEST_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )


def _resolve_worker_queue(explicit_queue: str | None = None) -> str:
    if explicit_queue and str(explicit_queue).strip():
        return str(explicit_queue).strip()
    return (
        (os.getenv("MEDIA_INGEST_JOBS_QUEUE") or "").strip()
        or (os.getenv("MEDIA_INGEST_JOBS_DEFAULT_QUEUE") or "").strip()
        or "default"
    )


def _normalize_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> str:
    owner = job.get("owner_user_id") or payload.get("user_id")
    if owner is None or str(owner).strip() == "":
        raise MediaIngestJobError("missing owner_user_id", retryable=False)
    return str(owner)


def _should_cancel(jm: JobManager, job_id: int) -> bool:
    try:
        cur = jm.get_job(int(job_id)) or {}
        status_val = str(cur.get("status") or "").lower()
        return status_val == "cancelled"
    except Exception:
        return False


def _cleanup_temp_dir(temp_dir: str | None) -> None:
    if not temp_dir:
        return
    with contextlib.suppress(Exception):
        shutil.rmtree(temp_dir, ignore_errors=True)


def _build_form_data(payload: dict[str, Any]) -> AddMediaForm:
    options = payload.get("options") or {}
    if not isinstance(options, dict):
        options = {}
    if not options.get("media_type"):
        options["media_type"] = payload.get("media_type")
    options.setdefault("urls", None)
    return AddMediaForm(**options)


def _create_db(user_id: str):
    db_path = DatabasePaths.get_media_db_path(user_id)
    return create_media_database(client_id=f"media_ingest_worker:{user_id}", db_path=str(db_path))


async def _schedule_embeddings(
    *,
    media_id: int,
    user_id: str,
    db,
    form_data: AddMediaForm,
) -> None:
    try:
        from tldw_Server_API.app.api.v1.endpoints.media_embeddings import (
            generate_embeddings_for_media,
            get_media_content,
        )

        media_content = await get_media_content(media_id, db)
        embedding_settings = settings.get("EMBEDDING_CONFIG", {}) or {}
        embedding_model = (
            form_data.embedding_model
            or embedding_settings.get("embedding_model")
            or "sentence-transformers/all-MiniLM-L6-v2"
        )
        embedding_provider = (
            form_data.embedding_provider
            or embedding_settings.get("embedding_provider")
            or "huggingface"
        )

        result = await generate_embeddings_for_media(
            media_id=media_id,
            media_content=media_content,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            chunk_size=form_data.chunk_size or 1000,
            chunk_overlap=getattr(form_data, "overlap", None) or 200,
            user_id=user_id,
        )
        allow_zero = bool(result.get("allow_zero_embeddings"))
        if result.get("status") == "success" or allow_zero:
            await _mark_embeddings_complete_with_retry(
                db=db,
                media_id=media_id,
                context="media_ingest_jobs_worker",
            )
        else:
            db.mark_embeddings_error(
                media_id,
                str(result.get("error") or result.get("message") or "Embedding generation failed"),
            )
    except Exception as exc:
        with contextlib.suppress(Exception):
            db.mark_embeddings_error(media_id, str(exc) or "Embedding generation failed")
        logger.warning("Embedding generation failed for media {}: {}", media_id, exc)


async def _handle_job(job: dict[str, Any], jm: JobManager, progress: _ProgressState) -> dict[str, Any]:
    job_id = int(job.get("id"))
    payload = _normalize_payload(job.get("payload"))
    if str(job.get("job_type") or "").lower() != _MEDIA_JOB_TYPE:
        raise MediaIngestJobError("unsupported job_type", retryable=False)

    user_id = _resolve_user_id(job, payload)
    source = payload.get("source")
    if not source:
        raise MediaIngestJobError("missing source", retryable=False)

    source_kind = str(payload.get("source_kind") or "").lower() or "url"
    input_ref = payload.get("input_ref") or payload.get("original_filename") or source

    form_data = _build_form_data(payload)

    temp_dir = payload.get("temp_dir")
    cleanup_temp_dir = bool(payload.get("cleanup_temp_dir"))
    if not temp_dir:
        temp_dir = tempfile.mkdtemp(prefix="media_ingest_job_")
        cleanup_temp_dir = True

    db = None
    try:
        if _should_cancel(jm, job_id):
            jm.finalize_cancelled(job_id, reason="cancel requested before start")
            return {}

        progress.percent = 5.0
        progress.message = "prepare"
        jm.update_job_progress(job_id, progress_percent=progress.percent, progress_message=progress.message)

        db = _create_db(user_id)
        db_path = getattr(db, "db_path_str", None) or getattr(db, "db_path", None) or ""
        client_id = getattr(db, "client_id", None) or f"media_ingest_worker:{user_id}"
        loop = asyncio.get_running_loop()

        def cancel_check():
            return _should_cancel(jm, job_id)

        chunk_options = prepare_chunking_options_dict(form_data)
        if chunk_options is not None:
            try:
                first_url = source if source_kind == "url" else None
                first_filename = payload.get("original_filename")
                chunk_options = apply_chunking_template_if_any(
                    form_data=form_data,
                    db=db,
                    chunking_options_dict=chunk_options,
                    TemplateClassifier=TemplateClassifier,
                    first_url=first_url,
                    first_filename=first_filename,
                )
            except Exception as exc:
                logger.debug("Chunking template auto-apply failed: {}", exc)

        if str(form_data.media_type) in {"video", "audio"}:
            urls = [source] if source_kind == "url" else []
            uploaded_paths = [source] if source_kind == "file" else []
            source_to_ref_map = {str(source): input_ref}

            progress.percent = 20.0
            progress.message = "process"
            jm.update_job_progress(job_id, progress_percent=progress.percent, progress_message=progress.message)

            results = await process_batch_media(
                media_type=str(form_data.media_type),
                urls=urls,
                uploaded_file_paths=uploaded_paths,
                source_to_ref_map=source_to_ref_map,
                form_data=form_data,
                chunk_options=chunk_options,
                loop=loop,
                db_path=str(db_path),
                client_id=str(client_id),
                temp_dir=temp_dir,
                user_id=int(user_id) if user_id.isdigit() else None,
                cancel_check=cancel_check,
            )
        else:
            progress.percent = 20.0
            progress.message = "process"
            jm.update_job_progress(job_id, progress_percent=progress.percent, progress_message=progress.message)

            result = await process_document_like_item(
                item_input_ref=str(input_ref),
                processing_source=str(source),
                media_type=form_data.media_type,
                is_url=(source_kind == "url"),
                form_data=form_data,
                chunk_options=chunk_options,
                temp_dir=temp_dir,
                loop=loop,
                db_path=str(db_path),
                client_id=str(client_id),
                user_id=int(user_id) if user_id.isdigit() else None,
                cancel_check=cancel_check,
            )
            results = [result]

        if _should_cancel(jm, job_id):
            jm.finalize_cancelled(job_id, reason="cancel requested during processing")
            return {}

        progress.percent = 90.0
        progress.message = "finalize"
        jm.update_job_progress(job_id, progress_percent=progress.percent, progress_message=progress.message)

        result_item = results[0] if results else {}
        media_id = result_item.get("db_id") if isinstance(result_item, dict) else None

        if media_id and getattr(form_data, "generate_embeddings", False):
            asyncio.create_task(
                _schedule_embeddings(
                    media_id=int(media_id),
                    user_id=user_id,
                    db=db,
                    form_data=form_data,
                )
            )

        progress.percent = 100.0
        progress.message = "completed"
        jm.update_job_progress(job_id, progress_percent=progress.percent, progress_message=progress.message)

        if isinstance(result_item, dict):
            return {
                "status": result_item.get("status"),
                "media_id": result_item.get("db_id"),
                "media_uuid": result_item.get("media_uuid"),
                "error": result_item.get("error"),
                "warnings": result_item.get("warnings"),
                "db_message": result_item.get("db_message"),
            }
        return {"status": "Error", "error": "No result produced"}

    finally:
        if db is not None:
            with contextlib.suppress(Exception):
                db.close_connection()
        if cleanup_temp_dir:
            _cleanup_temp_dir(temp_dir)


async def run_media_ingest_jobs_worker(
    stop_event: asyncio.Event | None = None,
    *,
    queue: str | None = None,
    worker_id: str | None = None,
) -> None:
    if JobManager._ACQUIRE_GATE_ENABLED:
        logger.warning(
            "Media Ingest Jobs worker starting with acquire gate enabled; clearing stale gate state"
        )
    JobManager.set_acquire_gate(False)

    jm = _jobs_manager()
    queue_name = _resolve_worker_queue(queue)
    resolved_worker_id = (
        str(worker_id).strip()
        if worker_id and str(worker_id).strip()
        else f"media-ingest-worker-{queue_name}"
    )
    cfg = _build_worker_config(worker_id=resolved_worker_id, queue=queue_name)
    sdk = WorkerSDK(jm, cfg)
    progress_state = _ProgressState()

    async def _cancel_check(job: dict[str, Any]) -> bool:
        job_id = int(job.get("id"))
        return _should_cancel(jm, job_id)

    async def _handler(job: dict[str, Any]) -> dict[str, Any]:
        return await _handle_job(job, jm, progress_state)

    def _progress_cb() -> dict[str, Any]:
        out: dict[str, Any] = {}
        if progress_state.percent is not None:
            out["progress_percent"] = progress_state.percent
        if progress_state.message is not None:
            out["progress_message"] = progress_state.message
        return out

    async def _watch_stop() -> None:
        if stop_event is None:
            return
        try:
            await stop_event.wait()
            sdk.stop()
        except Exception:
            sdk.stop()

    logger.info("Starting Media Ingest Jobs worker (queue={})", queue_name)
    watcher = asyncio.create_task(_watch_stop())
    try:
        await sdk.run(handler=_handler, cancel_check=_cancel_check, progress_cb=_progress_cb)
    finally:
        with contextlib.suppress(Exception):
            watcher.cancel()


async def run_media_ingest_heavy_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    heavy_queue = (
        (os.getenv("MEDIA_INGEST_JOBS_HEAVY_QUEUE") or "").strip()
        or "low"
    )
    await run_media_ingest_jobs_worker(
        stop_event,
        queue=heavy_queue,
        worker_id=f"media-ingest-worker-{heavy_queue}",
    )


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_media_ingest_jobs_worker())
