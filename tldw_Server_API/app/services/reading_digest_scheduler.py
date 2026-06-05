"""
Reading Digest Scheduler service.

Cron-based scheduler that enqueues reading digest jobs into the core Jobs
pipeline and persists schedules in the per-user Collections database.

Env:
  READING_DIGEST_SCHEDULER_ENABLED=true -> start service at app startup
  READING_DIGEST_SCHEDULER_TZ=<IANA>    -> default timezone (e.g., UTC)
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.Collections.reading_digest_jobs import (
    READING_DIGEST_DOMAIN,
    READING_DIGEST_JOB_TYPE,
    reading_digest_queue,
)
from tldw_Server_API.app.core.config import settings as core_settings
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ReadingDigestScheduleRow
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import env_flag_enabled

_READING_DIGEST_NONCRITICAL_EXCEPTIONS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    AttributeError,
    ConnectionError,
    TimeoutError,
)
_READING_DIGEST_RUN_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _get_reading_digest_run_lock(schedule_id: str, user_id: int | str | None) -> asyncio.Lock:
    """Return a process-local lock for one user/schedule run slot."""
    key = (str(user_id or ""), str(schedule_id))
    lock = _READING_DIGEST_RUN_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _READING_DIGEST_RUN_LOCKS[key] = lock
    return lock


class _ReadingDigestScheduler:
    def __init__(self) -> None:
        self._aps: AsyncIOScheduler | None = None
        self._db_cache: dict[int, CollectionsDatabase] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._rescan_task: asyncio.Task | None = None
        self._jobs = JobManager()

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            tz = os.getenv("READING_DIGEST_SCHEDULER_TZ", "UTC")
            self._aps = AsyncIOScheduler(timezone=tz)
            self._aps.start()
            await self._load_all()
            try:
                interval = int(os.getenv("READING_DIGEST_SCHEDULER_RESCAN_SEC", "600") or 600)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                interval = 600

            async def _rescan_loop() -> None:
                while True:
                    try:
                        await asyncio.sleep(interval)
                        await self._rescan_once()
                    except asyncio.CancelledError:
                        break
                    except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug(f"Reading digest scheduler rescan error: {exc}")

            self._rescan_task = asyncio.create_task(_rescan_loop(), name="reading_digest_scheduler_rescan")
            self._started = True
            logger.info("Reading digest scheduler started")

    async def stop(self) -> None:
        async with self._lock:
            try:
                if self._aps:
                    self._aps.shutdown(wait=False)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                pass
            self._aps = None
            try:
                if self._rescan_task:
                    self._rescan_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._rescan_task
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                pass
            self._rescan_task = None
            self._started = False
            logger.info("Reading digest scheduler stopped")

    def _get_db(self, user_id: int) -> CollectionsDatabase:
        if user_id not in self._db_cache:
            self._db_cache[user_id] = CollectionsDatabase.for_user(user_id)
        return self._db_cache[user_id]

    async def _load_all(self) -> None:
        loaded = 0
        user_ids = self._enumerate_user_ids()
        for uid in sorted(user_ids):
            try:
                db = self._get_db(uid)
                rows, _total = db.list_reading_digest_schedules(tenant_id="default", limit=1000, offset=0)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                rows = []
            for s in rows:
                if s.enabled:
                    self._add_job(s, uid)
                    loaded += 1
        if loaded:
            logger.info("Reading digest scheduler registered {} schedule(s)", loaded)

    async def _rescan_once(self) -> None:
        if not self._aps:
            return
        desired: set[str] = set()
        user_ids = self._enumerate_user_ids()
        for uid in sorted(user_ids):
            try:
                db = self._get_db(uid)
                rows, _total = db.list_reading_digest_schedules(tenant_id="default", limit=1000, offset=0)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                rows = []
            for s in rows:
                if s.enabled:
                    desired.add(s.id)
                    self._add_job(s, uid)
        try:
            current_ids = {j.id for j in (self._aps.get_jobs() or [])}
            for jid in list(current_ids - desired):
                with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
                    self._aps.remove_job(jid)
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
            pass

    def _enumerate_user_ids(self) -> set[int]:
        user_ids: set[int] = set()
        try:
            base = DatabasePaths.get_user_db_base_dir()
            for p in base.iterdir():
                if p.is_dir():
                    try:
                        user_ids.add(int(p.name))
                    except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                        continue
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Reading digest scheduler: failed to enumerate user dirs: {exc}")
        with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
            user_ids.add(int(core_settings.get("SINGLE_USER_FIXED_ID", 1)))
        return user_ids

    def _add_job(self, schedule: ReadingDigestScheduleRow, user_id: int | None = None) -> None:
        if not self._aps:
            return
        try:
            with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(schedule.id)
            tz = schedule.timezone or os.getenv("READING_DIGEST_SCHEDULER_TZ", "UTC")
            try:
                trigger = CronTrigger.from_crontab(schedule.cron, timezone=tz)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
                logger.warning("Reading digest scheduler invalid cron for {}: {}", schedule.id, exc)
                return
            effective_uid = user_id if user_id is not None else int(schedule.user_id)
            self._aps.add_job(
                self._run_schedule,
                trigger=trigger,
                id=schedule.id,
                args=[schedule.id, effective_uid],
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
            )
            try:
                now = datetime.now(trigger.timezone)
                nxt = trigger.get_next_fire_time(None, now)
                next_iso = nxt.isoformat() if nxt else None
                self._get_db(effective_uid).set_reading_digest_history(schedule.id, next_run_at=next_iso)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                pass
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Reading digest scheduler failed to add job {}: {}", schedule.id, exc)

    async def _run_schedule(self, schedule_id: str, user_id: int | None = None) -> None:
        lock = _get_reading_digest_run_lock(schedule_id, user_id)
        async with lock:
            await self._run_schedule_locked(schedule_id, user_id)

    async def _run_schedule_locked(self, schedule_id: str, user_id: int | None = None) -> None:
        db = None
        schedule = None
        if user_id is not None:
            db = self._get_db(int(user_id))
            try:
                schedule = db.get_reading_digest_schedule(schedule_id)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                schedule = None
        if schedule is None:
            try:
                db = self._get_db(int(core_settings.get("SINGLE_USER_FIXED_ID", 1)))
                schedule = db.get_reading_digest_schedule(schedule_id)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                schedule = None
        if not schedule or not schedule.enabled:
            return

        try:
            tz = schedule.timezone or os.getenv("READING_DIGEST_SCHEDULER_TZ", "UTC")
            trigger = CronTrigger.from_crontab(schedule.cron, timezone=tz)
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Reading digest scheduler invalid cron for {}: {}", schedule_id, exc)
            return

        now = datetime.now(trigger.timezone)
        expected_next_raw = schedule.next_run_at or None
        scheduled_dt = None
        if expected_next_raw:
            try:
                scheduled_dt = datetime.fromisoformat(expected_next_raw)
                if scheduled_dt.tzinfo is None:
                    scheduled_dt = scheduled_dt.replace(tzinfo=trigger.timezone)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                scheduled_dt = None
        if scheduled_dt is None:
            scheduled_dt = trigger.get_next_fire_time(None, now)
        if scheduled_dt is None:
            logger.warning("Reading digest scheduler could not determine run slot for {}", schedule_id)
            return

        now_for_next = max(now, scheduled_dt + timedelta(seconds=1))
        next_dt = trigger.get_next_fire_time(scheduled_dt, now_for_next)
        next_iso = next_dt.isoformat() if next_dt else None
        claimed = False
        try:
            claimed = db.try_claim_reading_digest_run(
                schedule_id,
                expected_next_run_at=expected_next_raw,
                next_run_at=next_iso,
                last_run_at=datetime.now(timezone.utc).isoformat(),
                last_status="pending",
                disallow_statuses=("pending", "queued", "running"),
            )
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Reading digest scheduler claim failed for {}: {}", schedule_id, exc)
        if not claimed:
            logger.info("Reading digest schedule already claimed: schedule_id={}", schedule_id)
            return

        try:
            latest = db.get_reading_digest_schedule(schedule_id)
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
            latest = None
        if not latest or not latest.enabled:
            with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
                db.set_reading_digest_history(schedule_id, last_status="skipped_disabled")
            return
        schedule = latest

        try:
            if schedule.require_online:
                sm = await get_session_manager()
                sessions = await sm.get_active_sessions(int(schedule.user_id))
                if not sessions:
                    try:
                        db.set_reading_digest_history(
                            schedule_id,
                            last_status="skipped_offline",
                            next_run_at=next_iso,
                        )
                    except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                        db.set_reading_digest_history(schedule_id, last_status="skipped_offline")
                    return
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Reading digest presence gating failed for {}: {}", schedule_id, exc)

        payload = {
            "schedule_id": schedule.id,
            "user_id": schedule.user_id,
            "tenant_id": schedule.tenant_id,
        }
        try:
            run_slot = expected_next_raw or scheduled_dt.isoformat()
            job = self._jobs.create_job(
                domain=READING_DIGEST_DOMAIN,
                queue=reading_digest_queue(),
                job_type=READING_DIGEST_JOB_TYPE,
                payload=payload,
                owner_user_id=int(schedule.user_id),
                idempotency_key=f"reading_digest:{schedule.id}:{run_slot}",
            )
            logger.info("Reading digest schedule queued: schedule_id={} job_id={}", schedule_id, job.get("id"))
            with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
                db.set_reading_digest_history(schedule_id, last_status="queued")
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning("Reading digest schedule enqueue failed: {}", exc)
            with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
                db.set_reading_digest_history(schedule_id, last_status="error")

    # CRUD wrappers
    def create(
        self,
        *,
        tenant_id: str,
        user_id: str,
        name: str | None,
        cron: str,
        timezone: str | None,
        enabled: bool,
        require_online: bool,
        filters: dict[str, Any],
        template_id: int | None,
        template_name: str | None,
        format: str,
        retention_days: int | None,
    ) -> str:
        sid = __import__("uuid").uuid4().hex
        db = self._get_db(int(user_id))
        db.create_reading_digest_schedule(
            id=sid,
            tenant_id=tenant_id,
            name=name,
            cron=cron,
            timezone=timezone,
            enabled=enabled,
            require_online=require_online,
            filters=filters,
            template_id=template_id,
            template_name=template_name,
            format=format,
            retention_days=retention_days,
        )
        schedule = db.get_reading_digest_schedule(sid)
        if schedule and schedule.enabled:
            self._add_job(schedule, int(user_id))
        return sid

    def update(self, schedule_id: str, patch: dict[str, Any]) -> bool:
        schedule = self.get(schedule_id)
        if not schedule:
            return False
        db = self._get_db(int(schedule.user_id))
        db.update_reading_digest_schedule(schedule_id, patch)
        schedule = db.get_reading_digest_schedule(schedule_id)
        if schedule.enabled:
            self._add_job(schedule, int(schedule.user_id))
        else:
            try:
                if self._aps:
                    self._aps.remove_job(schedule_id)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                pass
        return True

    def delete(self, schedule_id: str) -> bool:
        schedule = self.get(schedule_id)
        if not schedule:
            return False
        try:
            if self._aps:
                self._aps.remove_job(schedule_id)
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
            pass
        db = self._get_db(int(schedule.user_id))
        return db.delete_reading_digest_schedule(schedule_id)

    def get(self, schedule_id: str) -> ReadingDigestScheduleRow | None:
        user_ids = self._enumerate_user_ids()
        for uid in sorted(user_ids):
            try:
                db = self._get_db(uid)
                return db.get_reading_digest_schedule(schedule_id)
            except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
                continue
        return None

    def list(self, *, tenant_id: str, user_id: str, limit: int = 50, offset: int = 0) -> list[ReadingDigestScheduleRow]:
        try:
            db = self._get_db(int(user_id))
            rows, _total = db.list_reading_digest_schedules(tenant_id=tenant_id, limit=limit, offset=offset)
            return rows
        except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
            return []


_INSTANCE: _ReadingDigestScheduler | None = None


def get_reading_digest_scheduler() -> _ReadingDigestScheduler:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _ReadingDigestScheduler()
    return _INSTANCE


async def start_reading_digest_scheduler(enabled: bool | None = None) -> asyncio.Task | None:
    if enabled is None:
        enabled = env_flag_enabled("READING_DIGEST_SCHEDULER_ENABLED")
    if not enabled:
        return None
    svc = get_reading_digest_scheduler()
    await svc.start()

    async def _noop() -> None:
        while True:
            await asyncio.sleep(60)

    task = asyncio.create_task(_noop(), name="reading_digest_scheduler")
    return task


async def stop_reading_digest_scheduler(task: asyncio.Task | None) -> None:
    try:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except _READING_DIGEST_NONCRITICAL_EXCEPTIONS:
        pass
    with contextlib.suppress(_READING_DIGEST_NONCRITICAL_EXCEPTIONS):
        await get_reading_digest_scheduler().stop()
