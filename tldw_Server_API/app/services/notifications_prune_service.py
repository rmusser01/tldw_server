"""
Notification retention prune scheduler.

Archives expired inbox notifications and hard-deletes archived rows after a grace window.

Env:
  NOTIFICATIONS_PRUNE_ENABLED=true|false
  NOTIFICATIONS_PRUNE_INTERVAL_SEC=3600
  NOTIFICATIONS_PRUNE_READ_DISMISSED_DAYS=30
  NOTIFICATIONS_PRUNE_ARCHIVE_GRACE_DAYS=7
  NOTIFICATIONS_RETENTION_DAYS_REMINDER_DUE=90
  NOTIFICATIONS_RETENTION_DAYS_REMINDER_FAILED=90
  NOTIFICATIONS_RETENTION_DAYS_JOB_COMPLETED=30
  NOTIFICATIONS_RETENTION_DAYS_JOB_FAILED=60
"""

from __future__ import annotations

import asyncio
import os
from sqlite3 import Error as SQLiteError

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Metrics import get_metrics_registry
from tldw_Server_API.app.core.testing import env_flag_enabled

DEFAULT_RETENTION_DAYS: dict[str, int] = {
    "reminder_due": 90,
    "reminder_failed": 90,
    "job_completed": 30,
    "job_failed": 60,
}
DEFAULT_READ_DISMISSED_DAYS = 30
DEFAULT_ARCHIVE_GRACE_DAYS = 7
MIN_INTERVAL_SECONDS = 60

_NOTIFICATIONS_PRUNE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    SQLiteError,
    TimeoutError,
    TypeError,
    ValueError,
)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError) as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            f"notifications_prune: invalid {name}; defaulting to {default}"
        )
        return default


def _enumerate_user_ids() -> list[int]:
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _NOTIFICATIONS_PRUNE_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "notifications_prune: failed to resolve user db base dir"
        )
        return []

    user_ids: list[int] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        try:
            user_ids.append(int(entry.name))
        except (TypeError, ValueError):
            continue

    if not user_ids:
        try:
            user_ids = [DatabasePaths.get_single_user_id()]
        except _NOTIFICATIONS_PRUNE_NONCRITICAL_EXCEPTIONS as exc:
            logger.bind(error_type=type(exc).__name__).debug(
                "notifications_prune: failed to derive single user id"
            )
            user_ids = []
    return sorted(set(user_ids))


class NotificationsPruneService:
    def __init__(
        self,
        *,
        retention_days_by_kind: dict[str, int] | None = None,
        read_dismissed_grace_days: int = DEFAULT_READ_DISMISSED_DAYS,
        archive_grace_days: int = DEFAULT_ARCHIVE_GRACE_DAYS,
    ) -> None:
        self.retention_days_by_kind = dict(DEFAULT_RETENTION_DAYS)
        if retention_days_by_kind:
            self.retention_days_by_kind.update(retention_days_by_kind)
        self.read_dismissed_grace_days = int(read_dismissed_grace_days)
        self.archive_grace_days = int(archive_grace_days)

    async def run_once_for_user(self, *, user_id: int) -> dict[str, int]:
        with CollectionsDatabase.for_user(user_id=user_id) as cdb:
            archived, deleted = cdb.prune_user_notifications(
                retention_days_by_kind=self.retention_days_by_kind,
                read_dismissed_grace_days=self.read_dismissed_grace_days,
                archive_grace_days=self.archive_grace_days,
            )
        if archived:
            get_metrics_registry().increment(
                "app_warning_events_total",
                value=float(archived),
                labels={"component": "notifications_prune", "event": "archived"},
            )
        if deleted:
            get_metrics_registry().increment(
                "app_warning_events_total",
                value=float(deleted),
                labels={"component": "notifications_prune", "event": "deleted"},
            )
        return {"archived": archived, "deleted": deleted}

    async def run_once(self, *, user_ids: list[int] | None = None) -> dict[str, int]:
        total_archived = 0
        total_deleted = 0
        for user_id in user_ids or _enumerate_user_ids():
            summary = await self.run_once_for_user(user_id=user_id)
            total_archived += int(summary["archived"])
            total_deleted += int(summary["deleted"])
        return {"archived": total_archived, "deleted": total_deleted}


async def start_notifications_prune_scheduler() -> asyncio.Task | None:
    if not env_flag_enabled("NOTIFICATIONS_PRUNE_ENABLED"):
        return None

    interval = max(MIN_INTERVAL_SECONDS, _int_env("NOTIFICATIONS_PRUNE_INTERVAL_SEC", 3600))
    read_dismiss_days = max(0, _int_env("NOTIFICATIONS_PRUNE_READ_DISMISSED_DAYS", DEFAULT_READ_DISMISSED_DAYS))
    archive_grace_days = max(0, _int_env("NOTIFICATIONS_PRUNE_ARCHIVE_GRACE_DAYS", DEFAULT_ARCHIVE_GRACE_DAYS))
    retention_by_kind = {
        "reminder_due": max(0, _int_env("NOTIFICATIONS_RETENTION_DAYS_REMINDER_DUE", DEFAULT_RETENTION_DAYS["reminder_due"])),
        "reminder_failed": max(0, _int_env("NOTIFICATIONS_RETENTION_DAYS_REMINDER_FAILED", DEFAULT_RETENTION_DAYS["reminder_failed"])),
        "job_completed": max(0, _int_env("NOTIFICATIONS_RETENTION_DAYS_JOB_COMPLETED", DEFAULT_RETENTION_DAYS["job_completed"])),
        "job_failed": max(0, _int_env("NOTIFICATIONS_RETENTION_DAYS_JOB_FAILED", DEFAULT_RETENTION_DAYS["job_failed"])),
    }
    service = NotificationsPruneService(
        retention_days_by_kind=retention_by_kind,
        read_dismissed_grace_days=read_dismiss_days,
        archive_grace_days=archive_grace_days,
    )

    async def _runner() -> None:
        await asyncio.sleep(min(interval, 60))
        while True:
            try:
                summary = await service.run_once()
                if summary["archived"] or summary["deleted"]:
                    logger.info(
                        "Notifications prune run complete: archived={} deleted={}",
                        summary["archived"],
                        summary["deleted"],
                    )
            except _NOTIFICATIONS_PRUNE_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).warning(
                    "Notifications prune run failed"
                )
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="notifications_prune_scheduler")
    logger.info(
        "Started notifications prune scheduler: interval={}s read_dismissed_days={} archive_grace_days={}",
        interval,
        read_dismiss_days,
        archive_grace_days,
    )
    return task
