from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.Chatbooks.chatbook_models import (
    FULL_ACCOUNT_EXPORT_MODE,
    ConflictResolution,
    ContentType,
    ExportStatus,
    ImportStatus,
    coerce_chatbook_export_version,
)
from tldw_Server_API.app.core.Chatbooks.chatbook_service import ChatbookService
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Metrics import get_metrics_registry

_CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
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


def _build_chacha_db_for_user(user_id: str) -> CharactersRAGDB:
    # Use the same logic as the dependency util to locate per-user DB
    try:
        db_path = DatabasePaths.get_chacha_db_path(user_id)
        return CharactersRAGDB(db_path=str(db_path), client_id=str(user_id))
    except (TypeError, ValueError) as e:
        logger.debug(f"Core Jobs Worker: invalid user_id {user_id}: {e}")
        try:
            get_metrics_registry().increment(
                "app_warning_events_total",
                labels={"component": "core_jobs_worker", "event": "invalid_user_id"},
            )
        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for invalid_user_id")
        raise


def _count_result_items(counts: object) -> int:
    """Return the non-negative item total from redacted result counts."""
    if not isinstance(counts, dict):
        return 0
    total = 0
    for value in counts.values():
        if isinstance(value, bool):
            continue
        try:
            total += max(0, int(value or 0))
        except (TypeError, ValueError):
            continue
    return total


async def run_chatbooks_core_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    """Shared background worker for Chatbooks when using core Jobs backend.

    Processes jobs for all users by acquiring next job from the global Jobs table.
    """
    logger.info("Starting Core Jobs worker for Chatbooks domain")
    jm = JobManager()
    worker_id = "cb-core-worker"
    poll_sleep = float(os.getenv("JOBS_POLL_INTERVAL_SECONDS", "1.0") or "1.0")
    import random
    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping Core Jobs worker on shutdown signal")
            return
        try:
            lease_seconds = int(os.getenv("JOBS_LEASE_SECONDS", "60") or "60")
            renew_interval = int(os.getenv("JOBS_LEASE_RENEW_SECONDS", "30") or "30")
            renew_jitter = int(os.getenv("JOBS_LEASE_RENEW_JITTER_SECONDS", "5") or "5")
            job = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=lease_seconds, worker_id=worker_id)
            if not job:
                await asyncio.sleep(poll_sleep)
                continue

            owner = job.get("owner_user_id") or (job.get("payload") or {}).get("user_id")
            if not owner:
                jm.fail_job(int(job["id"]), error="missing owner_user_id", retryable=False, worker_id=worker_id, lease_id=str(job.get("lease_id")))
                continue
            lease_id = job.get("lease_id")
            # Build a per-user service
            try:
                db = _build_chacha_db_for_user(str(owner))
            except (TypeError, ValueError) as e:
                err_msg = f"invalid owner_user_id: {e}"
                logger.warning(f"Core Jobs Worker: {err_msg}")
                jm.fail_job(
                    int(job["id"]),
                    error=err_msg,
                    retryable=False,
                    worker_id=worker_id,
                    lease_id=str(lease_id),
                )
                continue
            owner_int: int | None = None
            try:
                owner_int = int(owner)
            except (TypeError, ValueError):
                owner_int = None
            svc = ChatbookService(owner, db, user_id_int=owner_int)

            payload: dict = job.get("payload") or {}
            action = payload.get("action")
            chatbooks_job_id = payload.get("chatbooks_job_id")
            async def _start_renewal(job_id: int, _lease_seconds=lease_seconds, _lease_id=lease_id, _renew_interval=renew_interval, _renew_jitter=renew_jitter):
                async def _loop():
                    while True:
                        try:
                            if stop_event and stop_event.is_set():
                                return
                            jm.renew_job_lease(int(job_id), seconds=_lease_seconds, worker_id=worker_id, lease_id=str(_lease_id))
                        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"Core Jobs Worker: lease renew failed for job {job_id}: {e}")
                            try:
                                get_metrics_registry().increment(
                                    "app_warning_events_total",
                                    labels={"component": "core_jobs_worker", "event": "lease_renew_failed"},
                                )
                            except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                                logger.debug("metrics increment failed for lease_renew_failed")
                        # Apply jitter to renewal interval to avoid thundering herd
                        # Lease jitter only spreads renewal wake-ups; it is not security-sensitive.
                        slp = _renew_interval + random.uniform(-float(_renew_jitter), float(_renew_jitter))  # nosec B311
                        await asyncio.sleep(max(1.0, slp))
                return asyncio.create_task(_loop())

            def _safe_remove_export_file(path_value: str | None) -> None:
                """Best-effort cleanup for export archives created after cancellation."""
                if not path_value:
                    return
                try:
                    export_path = Path(path_value).resolve()
                    export_dir = Path(svc.export_dir).resolve()
                    if os.path.commonpath([str(export_path), str(export_dir)]) != str(export_dir):
                        logger.warning(f"Refusing to delete export outside export dir: {export_path}")
                        return
                    if export_path.exists() and export_path.is_file():
                        export_path.unlink()
                except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as cleanup_err:
                    logger.debug(f"Core Jobs Worker: failed to remove cancelled export file {path_value}: {cleanup_err}")

            if action == "export":
                _renew_task = None
                file_path: str | None = None
                try:
                    ej = svc._get_export_job(chatbooks_job_id)
                except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Core Jobs Worker: get export job failed {chatbooks_job_id}: {e}")
                    try:
                        get_metrics_registry().increment(
                            "app_warning_events_total",
                            labels={"component": "core_jobs_worker", "event": "get_export_job_failed"},
                        )
                    except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                        logger.debug("metrics increment failed for get_export_job_failed")
                    ej = None
                if ej:
                    ej.status = ExportStatus.IN_PROGRESS
                    ej.started_at = datetime.now(timezone.utc)
                    svc._save_export_job(ej)
                # Pre-flight cancel check
                cur = jm.get_job(int(job["id"])) or {}
                if cur.get("cancel_requested_at"):
                    if ej:
                        ej.status = ExportStatus.CANCELLED
                        ej.completed_at = datetime.now(timezone.utc)
                        svc._save_export_job(ej)
                    jm.finalize_cancelled(int(job["id"]), reason="cancel requested before start")
                    continue
                try:
                    # Build selections
                    raw_selections = payload.get("content_selections")
                    selection_mode = str(
                        payload.get("selection_mode")
                        or (FULL_ACCOUNT_EXPORT_MODE if raw_selections is None or raw_selections == {} else "allowlist")
                    )
                    cs = None if selection_mode == FULL_ACCOUNT_EXPORT_MODE else {}
                    if cs is not None:
                        for k, v in (raw_selections or {}).items():
                            with contextlib.suppress(_CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS):
                                cs[ContentType(k)] = v
                        if sum(len(ids or []) for ids in cs.values()) == 0:
                            err_msg = "Export allowlist contains no exportable items."
                            if ej:
                                ej.status = ExportStatus.FAILED
                                ej.completed_at = datetime.now(timezone.utc)
                                ej.error_message = err_msg
                                svc._save_export_job(ej)
                            jm.fail_job(
                                int(job["id"]),
                                error=err_msg,
                                retryable=False,
                                worker_id=worker_id,
                                lease_id=str(lease_id),
                                completion_token=str(lease_id),
                            )
                            continue
                    try:
                        format_version = coerce_chatbook_export_version(payload.get("format_version"))
                    except ValueError as exc:
                        err_msg = str(exc)
                        if ej:
                            ej.status = ExportStatus.FAILED
                            ej.completed_at = datetime.now(timezone.utc)
                            ej.error_message = err_msg
                            svc._save_export_job(ej)
                        jm.fail_job(
                            int(job["id"]),
                            error=err_msg,
                            retryable=False,
                            worker_id=worker_id,
                            lease_id=str(lease_id),
                            completion_token=str(lease_id),
                        )
                        continue
                    # Start periodic lease renewal during heavy processing
                    _renew_task = await _start_renewal(int(job["id"]))
                    ok, msg, file_path = await svc._create_chatbook_sync_wrapper(
                        name=payload.get("name"),
                        description=payload.get("description"),
                        content_selections=cs,
                        author=payload.get("author"),
                        include_media=bool(payload.get("include_media")),
                        media_quality=str(payload.get("media_quality", "compressed")),
                        include_embeddings=bool(payload.get("include_embeddings")),
                        include_generated_content=bool(payload.get("include_generated_content", True)),
                        tags=payload.get("tags") or [],
                        categories=payload.get("categories") or [],
                        format_version=format_version,
                        selection_mode=selection_mode,
                    )
                    if ok:
                        # Mid-flight cancel check (honor cancellation request or terminal state)
                        cur = jm.get_job(int(job["id"])) or {}
                        if cur.get("cancel_requested_at") or (cur.get("status") and str(cur.get("status")).lower() != "processing"):
                            if ej:
                                ej.status = ExportStatus.CANCELLED
                                ej.completed_at = datetime.now(timezone.utc)
                                svc._save_export_job(ej)
                            _safe_remove_export_file(file_path)
                            jm.finalize_cancelled(int(job["id"]), reason="cancel requested during processing")
                            continue
                        try:
                            ej = svc._get_export_job(chatbooks_job_id)
                        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                            ej = None
                        if ej and ej.status != ExportStatus.CANCELLED:
                            ej.status = ExportStatus.COMPLETED
                            ej.completed_at = datetime.now(timezone.utc)
                            ej.output_path = file_path
                            terminal_metadata = svc.build_export_job_metadata(file_path)
                            inventory_summary = terminal_metadata.get("account_inventory_summary")
                            counts = (
                                inventory_summary.get("counts")
                                if isinstance(inventory_summary, dict)
                                else None
                            )
                            total_items = _count_result_items(counts)
                            ej.progress_percentage = 100
                            ej.total_items = total_items
                            ej.processed_items = total_items
                            ej.metadata = {
                                **(ej.metadata if isinstance(ej.metadata, dict) else {}),
                                **terminal_metadata,
                            }
                            try:
                                ej.file_size_bytes = Path(file_path).stat().st_size if file_path else None
                            except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"Core Jobs Worker: stat exported file failed: {e}")
                                try:
                                    get_metrics_registry().increment(
                                        "app_warning_events_total",
                                        labels={"component": "core_jobs_worker", "event": "export_stat_failed"},
                                    )
                                except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                                    logger.debug("metrics increment failed for export_stat_failed")
                            now_utc = datetime.now(timezone.utc)
                            ej.expires_at = svc._get_export_expiry(now_utc)
                            download_expires_at = svc._get_download_expiry(now_utc, ej.expires_at)
                            ej.download_url = svc._build_download_url(ej.job_id, download_expires_at)
                            svc._save_export_job(ej)
                        jm.complete_job(int(job["id"]), result={"path": file_path}, worker_id=worker_id, lease_id=str(lease_id), completion_token=str(lease_id))
                    else:
                        try:
                            ej = svc._get_export_job(chatbooks_job_id)
                        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"Core Jobs Worker: get export job (post-fail) failed {chatbooks_job_id}: {e}")
                            ej = None
                        if ej:
                            ej.status = ExportStatus.FAILED
                            ej.completed_at = datetime.now(timezone.utc)
                            ej.error_message = msg
                            svc._save_export_job(ej)
                        jm.fail_job(int(job["id"]), error=str(msg), retryable=False, worker_id=worker_id, lease_id=str(lease_id), completion_token=str(lease_id))
                finally:
                    if _renew_task is not None:
                        try:
                            _renew_task.cancel()
                        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"Core Jobs Worker: failed to cancel renew task: {e}")
                            try:
                                get_metrics_registry().increment(
                                    "app_warning_events_total",
                                    labels={"component": "core_jobs_worker", "event": "renew_task_cancel_failed"},
                                )
                            except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                                logger.debug("metrics increment failed for renew_task_cancel_failed")
            elif action == "import":
                file_ref = payload.get("file_token") or payload.get("file_path")
                _renew_task = None
                try:
                    ij = svc._get_import_job(chatbooks_job_id)
                    if ij:
                        ij.status = ImportStatus.IN_PROGRESS
                        ij.started_at = datetime.now(timezone.utc)
                        svc._save_import_job(ij)
                    # Pre-flight cancel check
                    cur = jm.get_job(int(job["id"])) or {}
                    if cur.get("cancel_requested_at"):
                        if ij:
                            ij.status = ImportStatus.CANCELLED
                            ij.completed_at = datetime.now(timezone.utc)
                            svc._save_import_job(ij)
                        jm.finalize_cancelled(int(job["id"]), reason="cancel requested before start")
                        continue
                    raw_content_selections = payload.get("content_selections")
                    cs = None if raw_content_selections is None else {}
                    if cs is not None:
                        for k, v in raw_content_selections.items():
                            try:
                                cs[ContentType(k)] = v
                            except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                                logger.debug(f"Core Jobs Worker: invalid content type {k}: {e}")
                                try:
                                    get_metrics_registry().increment(
                                        "app_warning_events_total",
                                        labels={"component": "core_jobs_worker", "event": "invalid_content_type"},
                                    )
                                except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                                    logger.debug("metrics increment failed for invalid_content_type")
                    conf_val = payload.get("conflict_resolution", "skip")
                    try:
                        conf = ConflictResolution(conf_val)
                    except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Core Jobs Worker: invalid conflict_resolution {conf_val}: {e}; defaulting to SKIP")
                        conf = ConflictResolution.SKIP
                    import_media = bool(payload.get("import_media", False))
                    import_embeddings = bool(payload.get("import_embeddings", False))
                    unsupported_conflicts = {ConflictResolution.OVERWRITE, ConflictResolution.MERGE, ConflictResolution.ASK}
                    if conf in unsupported_conflicts:
                        ok = False
                        msg = (
                            f"Conflict resolution '{conf.value}' is not supported yet. "
                            "Use 'skip' or 'rename'."
                        )
                    else:
                        _renew_task = await _start_renewal(int(job["id"]))
                        ok, msg, result = await asyncio.to_thread(
                            svc._import_chatbook_sync,
                            file_ref, cs,
                            conf,
                            bool(payload.get("prefix_imported", False)),
                            import_media,
                            import_embeddings,
                        )
                    ij = svc._get_import_job(chatbooks_job_id)
                    if ok:
                        # Mid-flight cancel check (honor cancellation request or terminal state)
                        cur = jm.get_job(int(job["id"])) or {}
                        if cur.get("cancel_requested_at") or (cur.get("status") and str(cur.get("status")).lower() != "processing"):
                            if ij:
                                ij.status = ImportStatus.CANCELLED
                                ij.completed_at = datetime.now(timezone.utc)
                                svc._save_import_job(ij)
                            jm.finalize_cancelled(int(job["id"]), reason="cancel requested during processing")
                            continue
                        if ij and ij.status != ImportStatus.CANCELLED:
                            ij.status = ImportStatus.COMPLETED
                            ij.completed_at = datetime.now(timezone.utc)
                            ij.progress_percentage = 100
                            if isinstance(result, dict):
                                imported_items = (
                                    result.get("imported_items")
                                    if isinstance(result.get("imported_items"), dict)
                                    else {}
                                )
                                skipped_non_restorable = (
                                    result.get("skipped_non_restorable")
                                    if isinstance(result.get("skipped_non_restorable"), dict)
                                    else {}
                                )
                                successful_count = _count_result_items(imported_items)
                                skipped_count = _count_result_items(skipped_non_restorable)
                                total_items = successful_count + skipped_count
                                ij.total_items = total_items
                                ij.processed_items = total_items
                                ij.successful_items = successful_count
                                ij.skipped_items = skipped_count
                                result_warnings = (
                                    result.get("warnings")
                                    if isinstance(result.get("warnings"), list)
                                    else []
                                )
                                ij.warnings = list(ij.warnings or []) + [
                                    warning
                                    for warning in result_warnings
                                    if warning not in (ij.warnings or [])
                                ]
                                ij.metadata = {
                                    **(ij.metadata if isinstance(ij.metadata, dict) else {}),
                                    "imported_items": imported_items,
                                    "inventory_summary": result.get("inventory_summary"),
                                    "skipped_non_restorable": skipped_non_restorable,
                                }
                            else:
                                ij.total_items = 0
                                ij.processed_items = 0
                            svc._save_import_job(ij)
                        jm.complete_job(int(job["id"]), worker_id=worker_id, lease_id=str(lease_id), completion_token=str(lease_id))
                    else:
                        if ij:
                            ij.status = ImportStatus.FAILED
                            ij.completed_at = datetime.now(timezone.utc)
                            ij.error_message = msg
                            svc._save_import_job(ij)
                        jm.fail_job(int(job["id"]), error=str(msg), retryable=False, worker_id=worker_id, lease_id=str(lease_id), completion_token=str(lease_id))
                finally:
                    if _renew_task is not None:
                        try:
                            _renew_task.cancel()
                        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as e:
                            logger.debug(f"Core Jobs Worker: failed to cancel renew task (import): {e}")
                            try:
                                get_metrics_registry().increment(
                                    "app_warning_events_total",
                                    labels={"component": "core_jobs_worker", "event": "renew_task_cancel_failed"},
                                )
                            except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                                logger.debug("metrics increment failed for renew_task_cancel_failed")
                    if file_ref:
                        try:
                            cleanup_path = svc._resolve_import_archive_path(file_ref)
                            if cleanup_path.exists() and cleanup_path.is_file():
                                cleanup_path.unlink()
                        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS as cleanup_err:
                            logger.debug(f"Core Jobs Worker: failed to remove import archive {file_ref}: {cleanup_err}")
            else:
                jm.fail_job(int(job["id"]), error="unknown action", retryable=False, worker_id=worker_id, lease_id=str(lease_id), completion_token=str(lease_id))
        except asyncio.CancelledError:
            logger.info("Core Jobs worker cancellation received; exiting worker loop")
            raise
        except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
            logger.exception("Core Jobs worker loop error")
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "core_jobs_worker", "event": "worker_loop_error"},
                )
            except _CORE_JOBS_WORKER_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for worker_loop_error")
            await asyncio.sleep(poll_sleep)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(run_chatbooks_core_jobs_worker())
