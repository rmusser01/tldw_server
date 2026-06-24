"""
Authoritative backup scheduler service.

Registers one APScheduler date job per active backup schedule row and enqueues
Jobs-backed backup work for each claimed fire slot.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.date import DateTrigger
from loguru import logger

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Admin_Backups.backup_schedule_jobs import (
    BACKUP_SCHEDULE_DOMAIN,
    BACKUP_SCHEDULE_JOB_TYPE,
    backup_schedule_queue,
    build_backup_schedule_idempotency_key,
    build_backup_schedule_job_payload,
    build_backup_schedule_run_slot_key,
    normalize_backup_schedule_slot,
)
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.admin_backup_schedules_service import (
    AdminBackupSchedulesService,
)

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
        AuthnzBackupSchedulesRepo,
    )


_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_MIN_RESCAN_SECONDS = 30


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp into a timezone-aware UTC-capable datetime."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class _AdminBackupScheduler:
    def __init__(self, *, repo=None, jobs: JobManager | None = None) -> None:
        self._repo = repo
        self._jobs = jobs or JobManager()
        self._aps: AsyncIOScheduler | None = None
        self._lock = asyncio.Lock()
        self._started = False
        self._rescan_task: asyncio.Task | None = None
        self._service = AdminBackupSchedulesService(repo=repo)

    async def _ensure_repo(self) -> AuthnzBackupSchedulesRepo:
        """Return the schedules repository, initializing it on first use."""
        if self._repo is not None:
            return self._repo
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
        from tldw_Server_API.app.core.AuthNZ.repos.backup_schedules_repo import (
            AuthnzBackupSchedulesRepo,
        )

        pool = await get_db_pool()
        repo = AuthnzBackupSchedulesRepo(pool)
        await repo.ensure_schema()
        self._repo = repo
        self._service = AdminBackupSchedulesService(repo=repo)
        return repo

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            tz = os.getenv("ADMIN_BACKUP_SCHEDULER_TZ", "UTC")
            self._aps = AsyncIOScheduler(timezone=tz)
            self._aps.start()
            await self._load_all()
            try:
                interval = int(os.getenv("ADMIN_BACKUP_SCHEDULER_RESCAN_SEC", "300") or 300)
            except _NONCRITICAL_EXCEPTIONS:
                interval = 300
            interval = max(_MIN_RESCAN_SECONDS, interval)

            async def _rescan_loop() -> None:
                while True:
                    try:
                        await asyncio.sleep(interval)
                        await self._rescan_once()
                    except asyncio.CancelledError:
                        break
                    except _NONCRITICAL_EXCEPTIONS:
                        logger.debug("Admin backup scheduler rescan error")

            self._rescan_task = asyncio.create_task(_rescan_loop(), name="admin_backup_scheduler_rescan")
            self._started = True
            logger.info("Admin backup scheduler started")

    async def stop(self) -> None:
        async with self._lock:
            try:
                if self._aps:
                    self._aps.shutdown(wait=False)
            except _NONCRITICAL_EXCEPTIONS:
                pass
            self._aps = None
            try:
                if self._rescan_task:
                    self._rescan_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._rescan_task
            except _NONCRITICAL_EXCEPTIONS:
                pass
            self._rescan_task = None
            self._started = False
            logger.info("Admin backup scheduler stopped")

    async def _list_all_schedules(self) -> list[dict[str, Any]]:
        repo = await self._ensure_repo()
        items: list[dict[str, Any]] = []
        offset = 0
        page_size = 200
        while True:
            page, total = await repo.list_schedules(limit=page_size, offset=offset, include_deleted=False)
            items.extend(page)
            offset += len(page)
            if not page or offset >= int(total):
                break
        return items

    async def _load_all(self) -> None:
        loaded = 0
        for item in await self._list_all_schedules():
            if item.get("deleted_at") is None and not bool(item.get("is_paused")):
                self._add_job(item)
                loaded += 1
        if loaded:
            logger.info("Admin backup scheduler registered {} schedule(s)", loaded)

    async def _rescan_once(self) -> None:
        if not self._aps:
            return
        desired: set[str] = set()
        for item in await self._list_all_schedules():
            if item.get("deleted_at") is None and not bool(item.get("is_paused")):
                desired.add(str(item["id"]))
                self._add_job(item)
        try:
            current_ids = {job.id for job in (self._aps.get_jobs() or [])}
            for stale_id in list(current_ids - desired):
                with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                    self._aps.remove_job(stale_id)
        except _NONCRITICAL_EXCEPTIONS:
            pass

    async def reconcile_schedule(self, *, schedule_id: str) -> None:
        async with self._lock:
            if not self._started or not self._aps:
                return
            repo = await self._ensure_repo()
            item = await repo.get_schedule(schedule_id, include_deleted=True)
            if not item or item.get("deleted_at") is not None or bool(item.get("is_paused")):
                with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                    self._aps.remove_job(schedule_id)
                return
            self._add_job(item)

    async def unschedule_schedule(self, *, schedule_id: str) -> None:
        async with self._lock:
            if not self._started or not self._aps:
                return
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(schedule_id)

    def _add_job(self, item: dict[str, Any]) -> None:
        if not self._aps:
            return
        next_run_raw = item.get("next_run_at")
        if not next_run_raw:
            return
        run_dt = _parse_datetime(str(next_run_raw))
        if run_dt is None:
            logger.warning("Admin backup scheduler invalid next_run_at for {}", item.get("id"))
            return
        try:
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(str(item["id"]))
            self._aps.add_job(
                self._run_schedule,
                trigger=DateTrigger(run_date=run_dt),
                id=str(item["id"]),
                args=[str(item["id"])],
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
            )
        except _NONCRITICAL_EXCEPTIONS:
            logger.warning("Admin backup scheduler failed to add job {}", item.get("id"))

    async def _run_schedule(self, schedule_id: str) -> None:
        repo = await self._ensure_repo()
        schedule = await repo.get_schedule(schedule_id, include_deleted=True)
        if not schedule or schedule.get("deleted_at") is not None or bool(schedule.get("is_paused")):
            return

        scheduled_raw = str(schedule.get("next_run_at") or "").strip()
        scheduled_dt = _parse_datetime(scheduled_raw)
        if scheduled_dt is None:
            logger.warning("Admin backup scheduler missing next_run_at for {}", schedule_id)
            return

        now_utc = datetime.now(timezone.utc)
        if scheduled_dt > (now_utc + timedelta(minutes=1)):
            logger.debug("Admin backup scheduler skipping early trigger for {}", schedule_id)
            return

        run_slot_utc = normalize_backup_schedule_slot(scheduled_dt)
        claim = await repo.claim_run_slot(
            schedule_id=schedule_id,
            scheduled_for=run_slot_utc,
            run_slot_key=build_backup_schedule_run_slot_key(schedule_id=schedule_id, scheduled_for=run_slot_utc),
            enqueued_at=now_utc.isoformat(),
        )
        if not claim:
            logger.info("Admin backup schedule already claimed: schedule_id={}", schedule_id)
            return

        next_from = max(now_utc, scheduled_dt + timedelta(seconds=1))
        next_run_at = self._service.compute_next_run_at(schedule, from_time=next_from)
        payload = build_backup_schedule_job_payload(
            schedule_id=schedule_id,
            run_id=str(claim["id"]),
            scheduled_for=run_slot_utc,
            dataset=str(schedule["dataset"]),
            target_user_id=schedule.get("target_user_id"),
            retention_count=int(schedule["retention_count"]),
        )
        owner_user_id = schedule.get("target_user_id")

        try:
            job = self._jobs.create_job(
                domain=BACKUP_SCHEDULE_DOMAIN,
                queue=backup_schedule_queue(),
                job_type=BACKUP_SCHEDULE_JOB_TYPE,
                payload=payload,
                owner_user_id=owner_user_id,
                idempotency_key=build_backup_schedule_idempotency_key(
                    schedule_id=schedule_id,
                    scheduled_for=run_slot_utc,
                ),
            )
            job_id = str(job.get("id"))
            await repo.mark_run_queued(
                run_id=str(claim["id"]),
                job_id=job_id,
                next_run_at=next_run_at,
                last_run_at=run_slot_utc,
            )
            logger.info("Admin backup schedule queued: schedule_id={} job_id={}", schedule_id, job_id)
            updated = await repo.get_schedule(schedule_id, include_deleted=True)
            if updated and updated.get("deleted_at") is None and not bool(updated.get("is_paused")):
                self._add_job(updated)
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Admin backup schedule enqueue failed for {}", schedule_id)
            await repo.mark_run_failed(
                run_id=str(claim["id"]),
                error=str(exc),
                last_status="error",
                next_run_at=next_run_at,
                last_run_at=run_slot_utc,
            )
            updated = await repo.get_schedule(schedule_id, include_deleted=True)
            if updated and updated.get("deleted_at") is None and not bool(updated.get("is_paused")):
                self._add_job(updated)


_INSTANCE: _AdminBackupScheduler | None = None


def get_admin_backup_scheduler() -> _AdminBackupScheduler:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _AdminBackupScheduler()
    return _INSTANCE


async def start_admin_backup_scheduler(enabled: bool | None = None) -> asyncio.Task | None:
    if enabled is None:
        enabled = env_flag_enabled("ADMIN_BACKUP_SCHEDULER_ENABLED")
    if not enabled:
        return None
    scheduler = get_admin_backup_scheduler()
    await scheduler.start()

    async def _noop() -> None:
        while True:
            await asyncio.sleep(60)

    return asyncio.create_task(_noop(), name="admin_backup_scheduler")


async def stop_admin_backup_scheduler(task: asyncio.Task | None) -> None:
    try:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except _NONCRITICAL_EXCEPTIONS:
        pass
    with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
        await get_admin_backup_scheduler().stop()


__all__ = [
    "get_admin_backup_scheduler",
    "start_admin_backup_scheduler",
    "stop_admin_backup_scheduler",
    "_AdminBackupScheduler",
]
