# media_files_cleanup_service.py
"""
Background service to clean up orphaned media files.

Runs periodically to find files on disk that have no corresponding database record,
and removes files that have been orphaned for longer than the grace period.

Enable via env:
  - MEDIA_FILES_CLEANUP_ENABLED=true (default: false)
  - MEDIA_FILES_CLEANUP_INTERVAL_SEC=86400 (default: daily)
  - MEDIA_FILES_CLEANUP_GRACE_DAYS=7 (default: 7 days)

This helps reclaim disk space from files that were:
  - Left behind after failed uploads
  - Orphaned by database corruption/restore
  - Created but never linked to a media record
"""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path
from sqlite3 import Error as SQLiteError

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.Metrics import get_metrics_registry

_MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    ImportError,
    LookupError,
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
)

# Configuration from environment
CLEANUP_ENABLED = os.environ.get("MEDIA_FILES_CLEANUP_ENABLED", "false").lower() == "true"
CLEANUP_INTERVAL_SEC = int(os.environ.get("MEDIA_FILES_CLEANUP_INTERVAL_SEC", "86400"))
GRACE_PERIOD_DAYS = int(os.environ.get("MEDIA_FILES_CLEANUP_GRACE_DAYS", "7"))

# Module-level task reference
_cleanup_task: asyncio.Task | None = None


def _get_storage_base_path() -> Path | None:
    """Get the storage base path from config."""
    try:
        from tldw_Server_API.app.core.config import load_comprehensive_config
        config = load_comprehensive_config()
        storage_path = config.get("media_storage_path") or config.get("storage_path")
        if storage_path:
            return Path(storage_path)
    except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "media_files_cleanup: failed to read storage path from config"
        )

    # Fallback to default location
    project_root = Path(__file__).resolve().parents[3]
    return project_root / "Databases" / "media_storage"


def _enumerate_user_ids() -> list[int]:
    """Get list of user IDs from user database directories."""
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "media_files_cleanup: failed to resolve user db base dir"
        )
        return []

    uids: list[int] = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                uids.append(int(p.name))
            except (TypeError, ValueError):
                continue

    if not uids:
        try:
            uids = [DatabasePaths.get_single_user_id()]
        except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS:
            uids = []

    return sorted(set(uids))


def _collect_known_storage_paths(user_id: int) -> set[str]:
    """Collect all storage_path values from a user's MediaFiles table."""
    known_paths: set[str] = set()
    try:
        db_path = DatabasePaths.get_media_db_path(user_id)
        if not Path(db_path).exists():
            return known_paths

        with managed_media_database(
            "cleanup_service",
            db_path=str(db_path),
            initialize=False,
        ) as db:
            # Query all MediaFiles records (including soft-deleted)
            conn = db.get_connection()
            cursor = conn.execute("SELECT storage_path FROM MediaFiles WHERE storage_path IS NOT NULL")
            for row in cursor.fetchall():
                path = row[0] if isinstance(row, (list, tuple)) else row.get("storage_path")
                if path:
                    known_paths.add(path)
    except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).warning(
            "media_files_cleanup: failed to query MediaFiles for user {user_id}",
            user_id=user_id,
        )

    return known_paths


def _find_orphaned_files(storage_base: Path, known_paths: set[str], grace_days: int) -> list[Path]:
    """
    Find files on disk that are not in the known paths set
    and are older than the grace period.
    """
    orphaned: list[Path] = []
    grace_seconds = grace_days * 24 * 60 * 60
    now = time.time()

    if not storage_base.exists():
        return orphaned

    # Walk through storage structure: {base}/{user_id}/media/{media_id}/{filename}
    for user_dir in storage_base.iterdir():
        if not user_dir.is_dir():
            continue

        media_dir = user_dir / "media"
        if not media_dir.exists():
            continue

        for media_id_dir in media_dir.iterdir():
            if not media_id_dir.is_dir():
                continue

            for file_path in media_id_dir.iterdir():
                if not file_path.is_file():
                    continue

                # Build relative path as stored in database
                try:
                    rel_path = str(file_path.relative_to(storage_base))
                except ValueError:
                    continue

                # Check if path is known
                if rel_path in known_paths:
                    continue

                # Check grace period (file age)
                try:
                    mtime = file_path.stat().st_mtime
                    age_seconds = now - mtime
                    if age_seconds < grace_seconds:
                        continue
                except OSError:
                    continue

                orphaned.append(file_path)

    return orphaned


async def cleanup_orphaned_files() -> dict:
    """
    Main cleanup routine. Finds and removes orphaned media files.

    Returns:
        Dict with cleanup statistics
    """
    storage_base = _get_storage_base_path()
    if not storage_base or not storage_base.exists():
        logger.debug("media_files_cleanup: storage base path does not exist, skipping")
        return {"status": "skipped", "reason": "storage_path_not_found"}

    # Collect all known paths from all user databases
    all_known_paths: set[str] = set()
    user_ids = _enumerate_user_ids()

    for user_id in user_ids:
        paths = _collect_known_storage_paths(user_id)
        all_known_paths.update(paths)

    logger.info(
        f"media_files_cleanup: collected {len(all_known_paths)} known paths "
        f"from {len(user_ids)} users"
    )

    # Find orphaned files (run in thread pool to avoid blocking)
    orphaned = await asyncio.to_thread(
        _find_orphaned_files,
        storage_base,
        all_known_paths,
        GRACE_PERIOD_DAYS
    )

    if not orphaned:
        logger.info("media_files_cleanup: no orphaned files found")
        return {"status": "completed", "files_removed": 0, "bytes_freed": 0}

    # Remove orphaned files
    files_removed = 0
    bytes_freed = 0
    errors = []

    for file_path in orphaned:
        try:
            file_size = file_path.stat().st_size
            file_path.unlink()
            files_removed += 1
            bytes_freed += file_size
            logger.debug(f"media_files_cleanup: removed orphan {file_path}")

            # Try to clean up empty parent directories
            try:
                parent = file_path.parent
                while parent != storage_base:
                    if parent.exists() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
            except OSError:
                pass

        except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).warning(
                "media_files_cleanup: failed to remove orphaned media file"
            )
            errors.append(str(file_path))

    # Record metrics
    try:
        metrics = get_metrics_registry()
        metrics.increment(
            "media_files_cleanup_runs_total",
            labels={"status": "success"}
        )
        metrics.observe(
            "media_files_cleanup_files_removed",
            files_removed
        )
        metrics.observe(
            "media_files_cleanup_bytes_freed",
            bytes_freed
        )
    except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS:
        pass

    logger.info(
        f"media_files_cleanup: removed {files_removed} orphaned files, "
        f"freed {bytes_freed / (1024*1024):.2f} MB"
    )

    return {
        "status": "completed",
        "files_removed": files_removed,
        "bytes_freed": bytes_freed,
        "errors": errors if errors else None
    }


async def _cleanup_loop():
    """Background loop that runs cleanup periodically."""
    # Initial delay to let the app fully start
    await asyncio.sleep(60)

    while True:
        try:
            result = await cleanup_orphaned_files()
            logger.debug(
                "media_files_cleanup: cycle completed "
                f"status={result.get('status')} "
                f"files_removed={result.get('files_removed')} "
                f"bytes_freed={result.get('bytes_freed')}"
            )
        except asyncio.CancelledError:
            logger.info("media_files_cleanup: task cancelled")
            break
        except _MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(error_type=type(e).__name__).error(
                "media_files_cleanup: error in cleanup cycle"
            )
            with contextlib.suppress(_MEDIA_CLEANUP_NONCRITICAL_EXCEPTIONS):
                get_metrics_registry().increment(
                    "media_files_cleanup_runs_total",
                    labels={"status": "error"}
                )

        await asyncio.sleep(CLEANUP_INTERVAL_SEC)


def start_cleanup_scheduler() -> asyncio.Task | None:
    """
    Start the background cleanup scheduler.

    Returns:
        The asyncio Task if started, None if disabled
    """
    global _cleanup_task

    if not CLEANUP_ENABLED:
        logger.debug("media_files_cleanup: disabled via MEDIA_FILES_CLEANUP_ENABLED")
        return None

    if _cleanup_task and not _cleanup_task.done():
        logger.debug("media_files_cleanup: scheduler already running")
        return _cleanup_task

    logger.info(
        f"media_files_cleanup: starting scheduler "
        f"(interval={CLEANUP_INTERVAL_SEC}s, grace={GRACE_PERIOD_DAYS}d)"
    )
    _cleanup_task = asyncio.create_task(_cleanup_loop())
    return _cleanup_task


def stop_cleanup_scheduler():
    """Stop the background cleanup scheduler."""
    global _cleanup_task

    if _cleanup_task and not _cleanup_task.done():
        _cleanup_task.cancel()
        logger.info("media_files_cleanup: scheduler stopped")

    _cleanup_task = None


def is_cleanup_running() -> bool:
    """Check if the cleanup scheduler is running."""
    return _cleanup_task is not None and not _cleanup_task.done()


__all__ = [
    "cleanup_orphaned_files",
    "start_cleanup_scheduler",
    "stop_cleanup_scheduler",
    "is_cleanup_running",
    "CLEANUP_ENABLED",
    "CLEANUP_INTERVAL_SEC",
    "GRACE_PERIOD_DAYS",
]
