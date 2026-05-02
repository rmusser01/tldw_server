"""Storage cleanup background service.

Handles periodic cleanup of:
- Expired transient files (with quota updates)
- Old soft-deleted files (trash purge)
- Temporary file cleanup
"""
from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import (
    FILE_CATEGORY_TTS_AUDIO,
    FILE_CATEGORY_VOICE_CLONE,
    AuthnzGeneratedFilesRepo,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.services.storage_quota_service import get_storage_service

try:  # pragma: no cover - optional dependency guard for static tooling
    import asyncpg as _asyncpg
except Exception:  # noqa: BLE001 - fallback when asyncpg is unavailable
    _ASYNC_DB_EXCEPTIONS: tuple[type[BaseException], ...] = ()
else:
    _ASYNC_DB_EXCEPTIONS = (_asyncpg.PostgresError,)

# Default configuration
DEFAULT_CLEANUP_INTERVAL_SEC = 3600  # 1 hour
DEFAULT_TRASH_RETENTION_DAYS = 30
DEFAULT_BATCH_SIZE = 100

_STORAGE_CLEANUP_EXCEPTIONS = (
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
if _ASYNC_DB_EXCEPTIONS:
    _STORAGE_CLEANUP_EXCEPTIONS = _STORAGE_CLEANUP_EXCEPTIONS + _ASYNC_DB_EXCEPTIONS


def _safe_resolve_user_path(
    user_id: int | None,
    storage_path: str,
    file_category: str | None = None,
) -> Path | None:
    """Resolve a stored path and ensure it stays within the user's allowed directory."""
    if not user_id or not storage_path:
        return None

    if file_category == FILE_CATEGORY_VOICE_CLONE:
        base_dir = DatabasePaths.get_user_voices_dir(user_id)
    else:
        base_dir = DatabasePaths.get_user_outputs_dir(user_id)
    full_path = base_dir / storage_path
    try:
        resolved_path = full_path.resolve()
        if resolved_path.is_relative_to(base_dir.resolve()):
            return resolved_path
    except ValueError:
        return None
    return None


def _mark_tts_history_artifact_deleted(user_id: int | None, file_id: int | None) -> None:
    if not user_id or not file_id:
        return
    try:
        db_path = DatabasePaths.get_media_db_path(user_id)
        with managed_media_database(
            "storage_cleanup",
            db_path=str(db_path),
            initialize=False,
        ) as db:
            db.mark_tts_history_artifacts_deleted_for_file_id(
                user_id=str(user_id),
                file_id=int(file_id),
            )
    except _STORAGE_CLEANUP_EXCEPTIONS as exc:
        logger.debug(f"storage_cleanup: failed to update tts_history for file {file_id}: {exc}")


async def cleanup_expired_files(
    storage_service,
    files_repo: AuthnzGeneratedFilesRepo,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """
    Delete files that have passed their expiration date.

    Args:
        files_repo: Generated files repository
        batch_size: Maximum files to process per batch

    Returns:
        Number of files cleaned up
    """
    total_deleted = 0

    try:
        expired_files = await files_repo.get_expired_files(limit=batch_size)

        for file_record in expired_files:
            file_id = file_record.get("id")
            user_id = file_record.get("user_id")
            storage_path = file_record.get("storage_path", "")
            file_category = file_record.get("file_category")

            try:
                # Update usage counters and hard-delete record first
                deleted = False
                if file_id:
                    deleted = await storage_service.unregister_generated_file(file_id, hard_delete=True)
                    if deleted:
                        total_deleted += 1

                if deleted and file_category == FILE_CATEGORY_TTS_AUDIO:
                    _mark_tts_history_artifact_deleted(user_id, file_id)

                # Delete physical file if it exists (with path validation)
                resolved_path = _safe_resolve_user_path(user_id, storage_path, file_category=file_category)
                if deleted and resolved_path and resolved_path.exists():
                    resolved_path.unlink()
                    logger.debug(f"Deleted expired file: {resolved_path}")

            except _STORAGE_CLEANUP_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).warning("Failed to cleanup expired file")

    except _STORAGE_CLEANUP_EXCEPTIONS as exc:
        logger.error(f"cleanup_expired_files failed: {exc}")

    return total_deleted


async def purge_old_trashed_files(
    files_repo: AuthnzGeneratedFilesRepo,
    retention_days: int = DEFAULT_TRASH_RETENTION_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> int:
    """
    Permanently delete files that have been in trash for too long.

    Args:
        files_repo: Generated files repository
        retention_days: Days to retain files in trash
        batch_size: Maximum files to process per batch

    Returns:
        Number of files purged
    """
    total_purged = 0

    try:
        old_files = await files_repo.get_old_trashed_files(
            days_old=retention_days,
            limit=batch_size,
        )

        for file_record in old_files:
            file_id = file_record.get("id")
            user_id = file_record.get("user_id")
            storage_path = file_record.get("storage_path", "")
            file_category = file_record.get("file_category")

            try:
                # Delete physical file if it exists (with path validation)
                resolved_path = _safe_resolve_user_path(user_id, storage_path, file_category=file_category)
                if resolved_path and resolved_path.exists():
                    resolved_path.unlink()
                    logger.debug(f"Purged trashed file: {resolved_path}")

                # Hard delete the database record
                await files_repo.hard_delete_file(file_id)
                total_purged += 1

                if file_category == FILE_CATEGORY_TTS_AUDIO:
                    _mark_tts_history_artifact_deleted(user_id, file_id)

            except _STORAGE_CLEANUP_EXCEPTIONS as exc:
                logger.warning(f"Failed to purge trashed file {file_id}: {exc}")

    except _STORAGE_CLEANUP_EXCEPTIONS as exc:
        logger.error(f"purge_old_trashed_files failed: {exc}")

    return total_purged


async def recalculate_user_usage(
    files_repo: AuthnzGeneratedFilesRepo,
    user_id: int,
) -> dict:
    """
    Recalculate storage usage for a user.

    Args:
        files_repo: Generated files repository
        user_id: User ID to recalculate

    Returns:
        Usage statistics dict
    """
    try:
        usage = await files_repo.get_user_storage_usage(user_id)
        return usage
    except _STORAGE_CLEANUP_EXCEPTIONS as exc:
        logger.error(f"recalculate_user_usage failed for user {user_id}: {exc}")
        return {}


async def run_storage_cleanup_cycle(
    retention_days: int = DEFAULT_TRASH_RETENTION_DAYS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    temp_retention_hours: int = 24,
) -> dict:
    """
    Run a single cleanup cycle.

    Args:
        retention_days: Days to retain files in trash
        batch_size: Maximum files to process per batch

    Returns:
        Statistics dict with counts of cleaned up items
    """
    stats = {
        "expired_deleted": 0,
        "trash_purged": 0,
        "errors": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    }

    try:
        storage_service = await get_storage_service()
        files_repo = await storage_service.get_generated_files_repo()

        # Cleanup expired files
        stats["expired_deleted"] = await cleanup_expired_files(
            storage_service,
            files_repo,
            batch_size=batch_size,
        )

        # Purge old trashed files
        stats["trash_purged"] = await purge_old_trashed_files(
            files_repo,
            retention_days=retention_days,
            batch_size=batch_size,
        )

        # Cleanup temp directories
        temp_stats = await storage_service.cleanup_temp_files(older_than_hours=temp_retention_hours)
        stats["temp_deleted"] = temp_stats.get("files_deleted", 0)
        stats["temp_bytes_freed"] = temp_stats.get("bytes_freed", 0)

        stats["completed_at"] = datetime.now(timezone.utc).isoformat()

        if stats["expired_deleted"] > 0 or stats["trash_purged"] > 0:
            logger.info(
                f"Storage cleanup: deleted {stats['expired_deleted']} expired files, "
                f"purged {stats['trash_purged']} from trash"
            )

    except asyncio.CancelledError:
        raise
    except _STORAGE_CLEANUP_EXCEPTIONS as exc:
        logger.error(f"Storage cleanup cycle failed: {exc}")
        stats["errors"] += 1
        stats["completed_at"] = datetime.now(timezone.utc).isoformat()

    return stats


async def run_storage_cleanup_loop(
    stop_event: asyncio.Event | None = None,
    interval_seconds: int | None = None,
    temp_retention_hours: int = 24,
) -> None:
    """
    Run scheduled cleanup of expired and trashed files.

    Configuration via environment variables:
    - STORAGE_CLEANUP_INTERVAL_SEC: Interval between cleanup cycles (default: 3600)
    - STORAGE_TRASH_RETENTION_DAYS: Days to keep files in trash (default: 30)
    - STORAGE_CLEANUP_BATCH_SIZE: Files to process per batch (default: 100)
    """
    if interval_seconds is None:
        interval_sec = int(os.getenv("STORAGE_CLEANUP_INTERVAL_SEC", str(DEFAULT_CLEANUP_INTERVAL_SEC)))
    else:
        interval_sec = int(interval_seconds)
    if interval_sec <= 0:
        logger.info("Storage cleanup scheduler disabled by STORAGE_CLEANUP_INTERVAL_SEC=0")
        return

    retention_days = int(os.getenv("STORAGE_TRASH_RETENTION_DAYS", str(DEFAULT_TRASH_RETENTION_DAYS)))
    batch_size = int(os.getenv("STORAGE_CLEANUP_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))

    logger.info(
        f"Starting storage cleanup worker (every {interval_sec}s, "
        f"trash retention: {retention_days}d, batch: {batch_size})"
    )

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stopping storage cleanup worker on shutdown signal")
            return

        try:
            await run_storage_cleanup_cycle(
                retention_days=retention_days,
                batch_size=batch_size,
                temp_retention_hours=temp_retention_hours,
            )
        except asyncio.CancelledError:
            raise
        except _STORAGE_CLEANUP_EXCEPTIONS as exc:
            logger.warning(f"Storage cleanup loop error: {exc}")

        # Wait for next cycle or stop event
        try:
            if stop_event:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=interval_sec,
                )
                logger.info("Stopping storage cleanup worker on shutdown signal")
                return
            else:
                await asyncio.sleep(interval_sec)
        except asyncio.TimeoutError:
            # Timeout means we should continue the loop
            continue


class StorageCleanupService:
    """
    Class-based wrapper for storage cleanup with lifecycle management.

    Provides start/stop methods for integration with FastAPI lifespan.
    """

    def __init__(self, interval_seconds: int | None = None, temp_retention_hours: int = 24):
        """
        Initialize the cleanup service.

        Args:
            interval_seconds: Time between cleanup runs
            temp_retention_hours: Age threshold for temp cleanup
        """
        self.interval = interval_seconds
        self.temp_retention_hours = temp_retention_hours
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._running = False

    async def start(self):
        """Start the cleanup background task."""
        if self._running:
            return
        self._running = True
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            run_storage_cleanup_loop(
                stop_event=self._stop_event,
                interval_seconds=self.interval,
                temp_retention_hours=self.temp_retention_hours,
            ),
            name="storage_cleanup_service",
        )
        logger.info(f"StorageCleanupService started (interval: {self.interval}s)")

    async def stop(self):
        """Stop the cleanup background task."""
        self._running = False
        stop_event = self._stop_event
        task = self._task
        self._stop_event = None
        self._task = None
        if stop_event:
            stop_event.set()
        if task:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except asyncio.TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            except RuntimeError as exc:
                # Happens when shutdown races event-loop teardown in tests.
                logger.debug(f"StorageCleanupService stop skipped due to runtime teardown: {exc}")
        logger.info("StorageCleanupService stopped")


# Singleton
_cleanup_service: StorageCleanupService | None = None


def get_cleanup_service(
    interval_seconds: int | None = None,
    temp_retention_hours: int = 24,
) -> StorageCleanupService:
    """Get or create the storage cleanup service singleton."""
    global _cleanup_service
    if _cleanup_service is None:
        _cleanup_service = StorageCleanupService(
            interval_seconds=interval_seconds,
            temp_retention_hours=temp_retention_hours,
        )
    return _cleanup_service


async def reset_cleanup_service() -> None:
    """Stop and clear the storage cleanup singleton (best-effort)."""
    global _cleanup_service
    service = _cleanup_service
    _cleanup_service = None
    if service is None:
        return
    with contextlib.suppress(asyncio.CancelledError, RuntimeError, TimeoutError):
        await service.stop()
