"""
Chatbooks Jobs worker (Phase 2):

- Consumes core Jobs entries for chatbooks export/import.
- Executes chatbook operations via ChatbookService.
- Updates Jobs status via the core JobManager.

Job contract (domain/queue/job_type):
- domain = "chatbooks"
- queue = os.getenv("CHATBOOKS_JOBS_QUEUE", "default")
- job_type = "export" | "import"

Payload fields:
- action: "export" | "import" (legacy; job_type preferred)
- chatbooks_job_id: str (required)
- name, description, author, tags, categories
- content_selections: {content_type: [ids]}
- include_media, media_quality, include_embeddings, include_generated_content
- file_token (preferred) or file_path (legacy), conflict_resolution, prefix_imported, import_media, import_embeddings

Usage:
  python -m tldw_Server_API.app.core.Chatbooks.services.jobs_worker
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    ConflictResolution,
    ContentType,
    ExportStatus,
    ImportStatus,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

_CHATBOOKS_DOMAIN = "chatbooks"

# Ensure worker runs in core backend mode.
if os.getenv("CHATBOOKS_JOBS_BACKEND") not in {"", "core"}:
    logger.warning("CHATBOOKS_JOBS_BACKEND is not core; forcing core backend for chatbooks jobs worker")
    os.environ["CHATBOOKS_JOBS_BACKEND"] = "core"


class ChatbooksJobError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = False, backoff_seconds: int | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        if backoff_seconds is not None:
            self.backoff_seconds = backoff_seconds


_SERVICE_CACHE: dict[str, ChatbookService] = {}
_DB_CACHE: dict[str, CharactersRAGDB] = {}


def _jobs_manager() -> JobManager:
    db_url = (os.getenv("JOBS_DB_URL") or "").strip()
    if not db_url:
        return JobManager()
    backend = "postgres" if db_url.startswith("postgres") else None
    return JobManager(backend=backend, db_url=db_url)


def _normalize_user_id(value: Any) -> str:
    if value is None or str(value).strip() == "":
        return str(DatabasePaths.get_single_user_id())
    return str(value)


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _get_db(user_id: str) -> CharactersRAGDB:
    cached = _DB_CACHE.get(user_id)
    if cached is not None:
        return cached
    user_id_int: int | None
    try:
        user_id_int = int(user_id)
    except (TypeError, ValueError):
        user_id_int = None
    if user_id_int is None:
        raise ChatbooksJobError("Chatbooks worker requires numeric user_id", retryable=False)
    db_path = DatabasePaths.get_chacha_db_path(user_id_int)
    db = CharactersRAGDB(db_path=str(db_path), client_id=str(user_id))
    _DB_CACHE[user_id] = db
    return db


def _get_service(user_id: str) -> ChatbookService:
    cached = _SERVICE_CACHE.get(user_id)
    if cached is not None:
        return cached
    db = _get_db(user_id)
    user_id_int = _coerce_int(user_id, 0) or None
    service = ChatbookService(user_id, db, user_id_int=user_id_int)
    _SERVICE_CACHE[user_id] = service
    return service


def _map_content_selections(raw: Any) -> dict[ContentType, list]:
    selections: dict[ContentType, list] = {}
    if not isinstance(raw, dict):
        return selections
    for key, value in raw.items():
        try:
            selections[ContentType(key)] = list(value or [])
        except (TypeError, ValueError) as exc:
            logger.debug(f"Ignoring invalid Chatbooks content selection key {key!r}: {exc}")
            continue
    return selections


def _parse_conflict_resolution(raw: Any) -> ConflictResolution:
    if isinstance(raw, ConflictResolution):
        return raw
    try:
        return ConflictResolution(str(raw))
    except Exception:
        return ConflictResolution.SKIP


def _mark_import_job_failed(service: ChatbookService, job_id: str, message: str) -> None:
    """Persist a claimed import job failure before raising a worker error."""
    ij = service._get_import_job(job_id)
    if ij and ij.status != ImportStatus.CANCELLED:
        ij.status = ImportStatus.FAILED
        ij.completed_at = datetime.now(timezone.utc)
        ij.error_message = message
        service._save_import_job(ij)


def _raise_import_job_failed(service: ChatbookService, job_id: str, message: str) -> None:
    _mark_import_job_failed(service, job_id, message)
    raise ChatbooksJobError(message, retryable=False)


async def _handle_export(service: ChatbookService, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
    if not service._claim_export_job(job_id):
        existing = service._get_export_job(job_id)
        if existing and existing.status in {ExportStatus.COMPLETED, ExportStatus.FAILED, ExportStatus.CANCELLED}:
            return {"skipped": True, "status": existing.status.value}
        raise ChatbooksJobError("export job already claimed", retryable=True, backoff_seconds=5)

    selections = _map_content_selections(payload.get("content_selections") or {})
    ok, msg, file_path = await service._create_chatbook_sync_wrapper(
        name=payload.get("name"),
        description=payload.get("description"),
        content_selections=selections,
        author=payload.get("author"),
        include_media=bool(payload.get("include_media")),
        media_quality=str(payload.get("media_quality", "compressed")),
        include_embeddings=bool(payload.get("include_embeddings")),
        include_generated_content=bool(payload.get("include_generated_content", True)),
        tags=payload.get("tags") or [],
        categories=payload.get("categories") or [],
    )

    if not ok:
        ej = service._get_export_job(job_id)
        if ej:
            ej.status = ExportStatus.FAILED
            ej.completed_at = datetime.now(timezone.utc)
            ej.error_message = msg
            service._save_export_job(ej)
        raise ChatbooksJobError(str(msg), retryable=False)

    ej = service._get_export_job(job_id)
    download_url = None
    if ej and ej.status != ExportStatus.CANCELLED:
        ej.status = ExportStatus.COMPLETED
        ej.completed_at = datetime.now(timezone.utc)
        ej.output_path = file_path
        with contextlib.suppress(Exception):
            ej.file_size_bytes = Path(file_path).stat().st_size if file_path else None
        now_utc = datetime.now(timezone.utc)
        ej.expires_at = service._get_export_expiry(now_utc)
        download_expires_at = service._get_download_expiry(now_utc, ej.expires_at)
        download_url = service._build_download_url(ej.job_id, download_expires_at)
        ej.download_url = download_url
        service._save_export_job(ej)

    return {"path": file_path, "download_url": download_url}


async def _handle_import(service: ChatbookService, payload: dict[str, Any], job_id: str) -> dict[str, Any]:
    if not service._claim_import_job(job_id):
        existing = service._get_import_job(job_id)
        if existing and existing.status in {ImportStatus.COMPLETED, ImportStatus.FAILED, ImportStatus.CANCELLED}:
            return {"skipped": True, "status": existing.status.value}
        raise ChatbooksJobError("import job already claimed", retryable=True, backoff_seconds=5)

    selections = _map_content_selections(payload.get("content_selections") or {})
    conflict_resolution = _parse_conflict_resolution(payload.get("conflict_resolution", "skip"))
    source_format = str(payload.get("source_format") or "chatbook").strip().lower()
    file_ref = payload.get("file_token") or payload.get("file_path")
    if not file_ref or not str(file_ref).strip():
        _raise_import_job_failed(service, job_id, "Missing file reference for import job")
    if source_format == "openwebui_json":
        try:
            resolved_path = service._resolve_import_upload_path(file_ref)
        except Exception as exc:
            _mark_import_job_failed(service, job_id, "Invalid or potentially malicious import file")
            raise ChatbooksJobError("Invalid or potentially malicious import file", retryable=False) from exc
        resolved_file_path = str(resolved_path or "").strip()
        if not resolved_file_path:
            _raise_import_job_failed(service, job_id, "Invalid or potentially malicious import file")
        try:
            ok, msg, result = await asyncio.to_thread(
                service.import_openwebui_json,
                resolved_file_path,
                conflict_resolution,
                bool(payload.get("prefix_imported", False)),
            )
        finally:
            try:
                if resolved_path.exists() and resolved_path.is_file():
                    resolved_path.unlink()
            except Exception as cleanup_err:
                logger.debug(f"Chatbooks Jobs worker: failed to remove OpenWebUI import file {resolved_path}: {cleanup_err}")

        ij = service._get_import_job(job_id)
        if ok:
            if ij and ij.status != ImportStatus.CANCELLED:
                ij.status = ImportStatus.COMPLETED
                ij.completed_at = datetime.now(timezone.utc)
                service._save_import_job(ij)
            return {"openwebui_result": result or {}}

        if ij:
            ij.status = ImportStatus.FAILED
            ij.completed_at = datetime.now(timezone.utc)
            ij.error_message = msg
            service._save_import_job(ij)
        raise ChatbooksJobError(str(msg), retryable=False)

    if source_format != "chatbook":
        _raise_import_job_failed(service, job_id, f"Unsupported import source format: {source_format}")
    try:
        resolved_path = service._resolve_import_archive_path(file_ref)
    except Exception as exc:
        _mark_import_job_failed(service, job_id, "Invalid or potentially malicious archive file")
        raise ChatbooksJobError("Invalid or potentially malicious archive file", retryable=False) from exc
    resolved_file_path = str(resolved_path or "").strip()
    if not resolved_file_path:
        _raise_import_job_failed(service, job_id, "Invalid or potentially malicious archive file")
    try:
        ok, msg, result = await asyncio.to_thread(
            service._import_chatbook_sync,
            resolved_file_path,
            selections,
            conflict_resolution,
            bool(payload.get("prefix_imported", False)),
            bool(payload.get("import_media", False)),
            bool(payload.get("import_embeddings", False)),
        )
    finally:
        try:
            if resolved_path.exists() and resolved_path.is_file():
                resolved_path.unlink()
        except Exception as cleanup_err:
            logger.debug(f"Chatbooks Jobs worker: failed to remove import archive {resolved_path}: {cleanup_err}")

    ij = service._get_import_job(job_id)
    if ok:
        if ij and ij.status != ImportStatus.CANCELLED:
            ij.status = ImportStatus.COMPLETED
            ij.completed_at = datetime.now(timezone.utc)
            service._save_import_job(ij)
        if isinstance(result, dict):
            return {
                "imported_items": result.get("imported_items") or {},
                "warnings": result.get("warnings") or [],
            }
        return {"imported_items": {}, "warnings": result or []}

    if ij:
        ij.status = ImportStatus.FAILED
        ij.completed_at = datetime.now(timezone.utc)
        ij.error_message = msg
        service._save_import_job(ij)
    raise ChatbooksJobError(str(msg), retryable=False)


async def _handle_job(job: dict[str, Any]) -> dict[str, Any]:
    payload = job.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}

    user_id = _normalize_user_id(job.get("owner_user_id") or payload.get("user_id"))
    service = _get_service(user_id)

    action = str(payload.get("action") or job.get("job_type") or "").lower()
    chatbooks_job_id = str(payload.get("chatbooks_job_id") or "").strip()
    if not chatbooks_job_id:
        raise ChatbooksJobError("Missing chatbooks_job_id", retryable=False)

    if action == "export":
        return await _handle_export(service, payload, chatbooks_job_id)
    if action == "import":
        return await _handle_import(service, payload, chatbooks_job_id)

    raise ChatbooksJobError(f"Unsupported chatbooks job action: {action}", retryable=False)


async def main() -> None:
    if os.getenv("CHATBOOKS_JOBS_BACKEND") == "prompt_studio":
        logger.warning("CHATBOOKS_JOBS_BACKEND is prompt_studio; chatbooks Jobs worker expects core backend")

    worker_id = (os.getenv("CHATBOOKS_JOBS_WORKER_ID") or f"chatbooks-jobs-{os.getpid()}").strip()
    queue = (os.getenv("CHATBOOKS_JOBS_QUEUE") or "default").strip() or "default"

    cfg = WorkerConfig(
        domain=_CHATBOOKS_DOMAIN,
        queue=queue,
        worker_id=worker_id,
        lease_seconds=_coerce_int(os.getenv("CHATBOOKS_JOBS_LEASE_SECONDS"), 60),
        renew_jitter_seconds=_coerce_int(os.getenv("CHATBOOKS_JOBS_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=_coerce_int(os.getenv("CHATBOOKS_JOBS_RENEW_THRESHOLD_SECONDS"), 10),
        backoff_base_seconds=_coerce_int(os.getenv("CHATBOOKS_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=_coerce_int(os.getenv("CHATBOOKS_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=_coerce_int(os.getenv("CHATBOOKS_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )

    jm = _jobs_manager()
    sdk = WorkerSDK(jm, cfg)
    logger.info(f"Chatbooks Jobs worker starting (queue={queue}, worker_id={worker_id})")
    await sdk.run(handler=_handle_job)


if __name__ == "__main__":
    asyncio.run(main())
