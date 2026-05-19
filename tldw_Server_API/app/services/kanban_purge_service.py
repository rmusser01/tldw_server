"""
Kanban soft-delete purge scheduler.

Permanently removes soft-deleted Kanban items after a grace period.

Enable via env:
  - KANBAN_PURGE_ENABLED=true
  - KANBAN_PURGE_INTERVAL_SEC=86400 (default daily)
  - KANBAN_PURGE_GRACE_DAYS=30
"""

from __future__ import annotations

import asyncio
import contextlib
import os

from loguru import logger

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.Kanban_DB import KanbanDB
from tldw_Server_API.app.core.testing import env_flag_enabled

_KANBAN_PURGE_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    KeyError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _enumerate_user_ids() -> list[int]:
    """Get list of user IDs from user database directories."""
    try:
        base = DatabasePaths.get_user_db_base_dir()
    except _KANBAN_PURGE_NONCRITICAL_EXCEPTIONS as exc:
        logger.bind(error_type=type(exc).__name__).debug(
            "kanban_purge: failed to resolve user db base dir"
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
        except _KANBAN_PURGE_NONCRITICAL_EXCEPTIONS:
            uids = []

    return sorted(set(uids))


def _purge_for_user(user_id: int, grace_days: int) -> dict:
    """Purge soft-deleted kanban items for a user and return counts."""
    db_path = DatabasePaths.get_kanban_db_path(user_id)
    db = KanbanDB(db_path=str(db_path), user_id=str(user_id))
    try:
        return db.purge_deleted_items(days_old=grace_days)
    finally:
        with contextlib.suppress(_KANBAN_PURGE_NONCRITICAL_EXCEPTIONS):
            db.close()


async def start_kanban_purge_scheduler() -> asyncio.Task | None:
    enabled = env_flag_enabled("KANBAN_PURGE_ENABLED")
    if not enabled:
        return None
    try:
        interval = int(os.getenv("KANBAN_PURGE_INTERVAL_SEC", "86400"))
    except (TypeError, ValueError):
        interval = 86400
    try:
        grace_days = int(os.getenv("KANBAN_PURGE_GRACE_DAYS", "30"))
    except (TypeError, ValueError):
        grace_days = 30

    async def _runner() -> None:
        await asyncio.sleep(min(interval, 60))
        while True:
            try:
                totals = {"boards": 0, "lists": 0, "cards": 0}
                for user_id in _enumerate_user_ids():
                    counts = _purge_for_user(user_id, grace_days)
                    for key in totals:
                        totals[key] += int(counts.get(key, 0))
                if any(totals.values()):
                    logger.info(
                        "Kanban purge removed boards={boards} lists={lists} cards={cards}",
                        **totals,
                    )
            except _KANBAN_PURGE_NONCRITICAL_EXCEPTIONS as exc:
                logger.bind(error_type=type(exc).__name__).debug(
                    "kanban_purge: purge run failed"
                )
            await asyncio.sleep(interval)

    task = asyncio.create_task(_runner(), name="kanban_purge_scheduler")
    logger.info(f"Started Kanban purge scheduler: interval={interval}s grace_days={grace_days}")
    return task


__all__ = ["start_kanban_purge_scheduler"]
