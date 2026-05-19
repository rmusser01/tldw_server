"""
Daily outputs purge scheduler.

Runs a daily cleanup to remove outputs according to retention policy and
aged soft-deleted rows. Optionally deletes files before removing DB rows.

Enable via env:
  - OUTPUTS_PURGE_ENABLED=true
  - OUTPUTS_PURGE_INTERVAL_SEC=86400 (default daily)
  - OUTPUTS_PURGE_DELETE_FILES=false
  - OUTPUTS_PURGE_GRACE_DAYS=30

This is a simple asyncio interval job (keeps latency minimal). For cron-timezone
exactness, integrate with the APScheduler service later.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3

from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.exceptions import StoragePathValidationError
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.outputs_service import normalize_output_storage_path

_OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS = (
    AssertionError,
    AttributeError,
    BackendDatabaseError,
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
    sqlite3.Error,
)


def _enumerate_user_ids() -> list[int]:
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
        try:
            get_metrics_registry().increment(
                "app_warning_events_total",
                labels={"component": "outputs_purge", "event": "settings_user_db_dir_read_failed"},
            )
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for outputs_purge settings read failure")
        logger.bind(error_type=type(e).__name__).debug(
            "outputs_purge: failed to resolve user db base dir"
        )
        return []
    uids: list[int] = []
    for p in base.iterdir():
        if p.is_dir():
            try:
                uids.append(int(p.name))
            except (TypeError, ValueError) as e:
                logger.debug("outputs_purge: skipping non-int user dir")
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "outputs_purge", "event": "invalid_user_dir_name"},
                    )
                except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for invalid_user_dir_name")
    if not uids:
        try:
            uids = [DatabasePaths.get_single_user_id()]
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(error_type=type(e).__name__).debug(
                "outputs_purge: failed to derive single_user_id"
            )
            try:
                get_metrics_registry().increment(
                    "app_warning_events_total",
                    labels={"component": "outputs_purge", "event": "single_user_id_fallback_failed"},
                )
            except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for single_user_id_fallback_failed")
            uids = []
    return sorted(set(uids))


async def _purge_for_user(user_id: int, delete_files: bool, grace_days: int) -> tuple[int, int]:
    """Return (removed, files_deleted)."""
    cdb = CollectionsDatabase.for_user(user_id)
    media_db = None
    # Build candidate set similar to /outputs/purge endpoint
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    ids: set[int] = set()
    paths: dict[int, str] = {}
    try:
        cur = cdb.backend.execute(
            "SELECT id, storage_path FROM outputs WHERE user_id = ? AND retention_until IS NOT NULL AND retention_until <= ?",
            (str(user_id), now),
        )
        for row in cur.rows:
            rid = int(row["id"]) if isinstance(row, dict) else int(row[0])
            ids.add(rid)
            paths[rid] = row["storage_path"] if isinstance(row, dict) else row[1]
    except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).warning(
            f"outputs_purge: error selecting retention candidates for user {user_id}"
        )
        try:
            get_metrics_registry().increment(
                "app_exception_events_total",
                labels={"component": "outputs_purge", "event": "select_retention_candidates_failed"},
            )
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for select_retention_candidates_failed")
    try:
        cur2 = cdb.backend.execute(
            "SELECT id, storage_path FROM outputs WHERE user_id = ? AND deleted = 1 AND deleted_at IS NOT NULL AND julianday(?) - julianday(deleted_at) >= ?",
            (str(user_id), now, grace_days),
        )
        for row in cur2.rows:
            rid = int(row["id"]) if isinstance(row, dict) else int(row[0])
            ids.add(rid)
            paths[rid] = row["storage_path"] if isinstance(row, dict) else row[1]
    except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).warning(
            f"outputs_purge: error selecting deleted candidates for user {user_id}"
        )
        try:
            get_metrics_registry().increment(
                "app_exception_events_total",
                labels={"component": "outputs_purge", "event": "select_deleted_candidates_failed"},
            )
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for select_deleted_candidates_failed")
    files_deleted = 0
    if delete_files and ids:
        for rid, pth in list(paths.items()):
            try:
                relative_name = normalize_output_storage_path(user_id, pth)
                outputs_dir = DatabasePaths.get_user_outputs_dir(user_id)
                p = outputs_dir / relative_name
                if p.exists():
                    p.unlink()
                    files_deleted += 1
            except StoragePathValidationError as e:
                logger.bind(error_type=type(e).__name__).warning(
                    f"outputs_purge: invalid output path for output {rid}"
                )
            except (OSError, PermissionError) as e:
                logger.bind(error_type=type(e).__name__).warning(
                    f"outputs_purge: failed to delete file for output {rid}"
                )
                try:
                    get_metrics_registry().increment(
                        "app_warning_events_total",
                        labels={"component": "outputs_purge", "event": "file_delete_failed"},
                    )
                except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for file_delete_failed")
    if ids:
        try:
            with managed_media_database(
                "outputs_purge",
                db_path=str(DatabasePaths.get_media_db_path(user_id)),
                initialize=False,
            ) as media_db:
                for rid in ids:
                    try:
                        media_db.mark_tts_history_artifacts_deleted_for_output(
                            user_id=str(user_id),
                            output_id=int(rid),
                        )
                    except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as exc:
                        logger.bind(error_type=type(exc).__name__).debug(
                            f"outputs_purge: failed to update tts_history for output {rid}"
                        )
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                "outputs_purge: failed to open Media DB for history update"
            )
    removed = 0
    if ids:
        placeholders = ",".join(["?"] * len(ids))
        try:
            output_ids_clause = f"({placeholders})"
            purge_sql_template = "DELETE FROM outputs WHERE user_id = ? AND id IN {output_ids_clause}"
            purge_sql = purge_sql_template.format_map(locals())  # nosec B608
            cdb.backend.execute(
                purge_sql,
                tuple([str(user_id)] + list(ids)),
            )
            removed = len(ids)
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
            logger.bind(error_type=type(e).__name__).warning(
                f"outputs_purge: DB delete failed for user {user_id}"
            )
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "outputs_purge", "event": "db_delete_failed"},
                )
            except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for db_delete_failed")
            removed = 0
    return removed, files_deleted


async def start_outputs_purge_scheduler() -> asyncio.Task | None:
    enabled = env_flag_enabled("OUTPUTS_PURGE_ENABLED")
    if not enabled:
        return None
    try:
        interval = int(os.getenv("OUTPUTS_PURGE_INTERVAL_SEC", "86400"))
    except (TypeError, ValueError) as e:
        logger.bind(error_type=type(e).__name__).debug(
            "outputs_purge: invalid OUTPUTS_PURGE_INTERVAL_SEC; using default"
        )
        interval = 86400
    delete_files = env_flag_enabled("OUTPUTS_PURGE_DELETE_FILES")
    try:
        grace_days = int(os.getenv("OUTPUTS_PURGE_GRACE_DAYS", "30"))
    except (TypeError, ValueError) as e:
        logger.bind(error_type=type(e).__name__).debug(
            "outputs_purge: invalid OUTPUTS_PURGE_GRACE_DAYS; using default"
        )
        grace_days = 30

    async def _runner():
        await asyncio.sleep(min(interval, 60))
        while True:
            try:
                uids = _enumerate_user_ids()
                total_removed = 0
                total_files = 0
                for uid in uids:
                    r, f = await _purge_for_user(uid, delete_files, grace_days)
                    total_removed += r
                    total_files += f
                if total_removed or total_files:
                    logger.info(f"Outputs purge: removed={total_removed} files_deleted={total_files}")
            except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
                logger.bind(error_type=type(e).__name__).debug("Outputs purge run failed")
                try:
                    get_metrics_registry().increment(
                        "app_exception_events_total",
                        labels={"component": "outputs_purge", "event": "purge_run_failed"},
                    )
                except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
                    logger.debug("metrics increment failed for purge_run_failed")
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="outputs_purge_scheduler")
    logger.info(f"Started outputs purge scheduler: interval={interval}s delete_files={delete_files} grace_days={grace_days}")
    return task
