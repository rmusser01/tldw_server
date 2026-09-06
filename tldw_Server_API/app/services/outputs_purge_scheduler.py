"""
Daily outputs purge scheduler.

Runs a daily cleanup to remove outputs according to retention policy and
aged soft-deleted rows. Managed files use durable post-commit cleanup; optional
unowned file removal follows successful database deletion.

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

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseError as BackendDatabaseError
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ReadingFileDeletionRequired
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.outputs_service import delete_output_with_file, find_outputs_to_purge

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
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        paths = find_outputs_to_purge(cdb=cdb, now_iso=now, soft_deleted_grace_days=grace_days, include_retention=True)
    except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS as e:
        logger.bind(error_type=type(e).__name__).warning("outputs_purge: error selecting purge candidates")
        try:
            get_metrics_registry().increment(
                "app_exception_events_total",
                labels={"component": "outputs_purge", "event": "select_purge_candidates_failed"},
            )
        except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
            logger.debug("metrics increment failed for select_purge_candidates_failed")
        return 0, 0

    removed_ids = []
    legacy_history_ids = []
    files_deleted = 0
    for output_id in paths:
        try:
            deleted, file_deleted, legacy_history = await delete_output_with_file(
                cdb,
                user_id,
                output_id,
                hard=True,
                delete_file=delete_files,
                purge_before=now,
                soft_deleted_grace_days=grace_days,
            )
            if deleted:
                removed_ids.append(output_id)
                if legacy_history:
                    legacy_history_ids.append(output_id)
            files_deleted += int(file_deleted)
        except ReadingFileDeletionRequired:
            continue
        except (*_OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS, HTTPException) as e:
            logger.bind(error_type=type(e).__name__).warning("outputs_purge: DB delete failed")
            try:
                get_metrics_registry().increment(
                    "app_exception_events_total",
                    labels={"component": "outputs_purge", "event": "db_delete_failed"},
                )
            except _OUTPUTS_PURGE_NONCRITICAL_EXCEPTIONS:
                logger.debug("metrics increment failed for db_delete_failed")
    if legacy_history_ids:
        try:
            with managed_media_database(
                "outputs_purge",
                db_path=str(DatabasePaths.get_media_db_path(user_id)),
                initialize=False,
            ) as media_db:
                for rid in legacy_history_ids:
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
    return len(removed_ids), files_deleted


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
