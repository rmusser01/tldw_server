"""Scheduler feed that arms automation definitions into the Jobs pipeline.

Implements the server-side dispatch half of the server-offload execution
seam (TASK-13020; the governing contract is tldw_chatbook ADR-077,
"Server-offloaded scheduled agent tasks", accepted 2026-08-21). Automation
definitions were previously modeled, validated, and audited but never
dispatched -- ``DEFAULT_DEFINITION_HEALTH = "execution_unavailable"`` was
permanent. This module mirrors the proven reminders pattern
(``reminders_scheduler.py``): an env-gated APScheduler service that arms
due work into the Jobs pipeline with idempotent run-slot enqueueing.

Schedule dict conventions (the ``schedule`` field the automation service
validates only by ``kind``; per-kind fields are defined here):

- ``{"kind": "one_time", "run_at": ISO-8601}``
- ``{"kind": "interval", "seconds": int > 0, "start_at": ISO-8601?}``
- ``{"kind": "daily", "at": "HH:MM", "timezone": IANA?}``
- ``{"kind": "weekly", "weekday": 0-6 (Mon=0) or name, "at": "HH:MM", "timezone": IANA?}``
- ``{"kind": "cron", "cron": five-field expression, "timezone": IANA?}``

Anything unusable degrades honestly: the definition is skipped with a
warning (mirroring the reminders scheduler's invalid-cron path), never
armed, never executed.

Env:
  SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED  -> start service at app startup
  SCHEDULED_TASKS_AUTOMATION_SCHEDULER_TZ       -> default timezone (UTC)
  SCHEDULED_TASKS_AUTOMATION_RESCAN_SEC         -> rescan interval (>= 30)
  AUTOMATION_JOBS_QUEUE                         -> queue name (default: default)

The gate ships OFF: arming before the ``agent_task_run`` consumer
(TASK-13021) is deployed only queues jobs nobody executes, so enable it
together with the consumer.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timedelta, timezone

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.config import settings as core_settings
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.reminders_scheduler import (
    _NONCRITICAL_EXCEPTIONS,
    _normalize_slot_to_utc_iso,
    _parse_iso_datetime,
)

AUTOMATION_DOMAIN = "scheduled_tasks"
AUTOMATION_JOB_TYPE = "agent_task_run"
ARMED_HEALTH = "ready"
SCHEDULER_ACTOR = "automation-scheduler"
_MIN_RESCAN_SECONDS = 30

_SUPPORTED_SCHEDULE_KINDS = ("one_time", "interval", "daily", "weekly", "cron")


def automation_jobs_queue() -> str:
    queue = (os.getenv("AUTOMATION_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def _scheduler_timezone() -> str:
    return os.getenv("SCHEDULED_TASKS_AUTOMATION_SCHEDULER_TZ", "UTC") or "UTC"


def build_trigger(schedule: dict, *, default_timezone: str | None = None):
    """Build an APScheduler trigger from a definition's ``schedule`` dict.

    Returns ``(trigger, None)`` on success or ``(None, reason)`` when the
    schedule cannot arm -- the caller skips the definition honestly rather
    than guessing. Field expectations are the module's documented
    conventions; the automation service validates only ``kind``.
    """
    tz_name = str(schedule.get("timezone") or default_timezone or _scheduler_timezone())
    try:
        tz = CronTrigger.from_crontab("* * * * *", timezone=tz_name).timezone
    except _NONCRITICAL_EXCEPTIONS:
        tz = timezone.utc

    kind = schedule.get("kind")
    if kind == "one_time":
        run_dt = _parse_iso_datetime(str(schedule.get("run_at") or ""), timezone_name=tz_name)
        if run_dt is None:
            return None, "one_time schedule missing/unparsable run_at"
        return DateTrigger(run_date=run_dt), None

    if kind == "interval":
        try:
            seconds = float(schedule["seconds"])
        except (KeyError, TypeError, ValueError):
            return None, "interval schedule missing/invalid seconds"
        if seconds <= 0:
            return None, "interval schedule seconds must be > 0"
        start_at = _parse_iso_datetime(str(schedule.get("start_at") or ""), timezone_name=tz_name)
        return IntervalTrigger(seconds=seconds, start_date=start_at), None

    if kind in ("daily", "weekly"):
        at = str(schedule.get("at") or "").strip()
        parts = at.split(":")
        if len(parts) != 2 or not all(p.isdigit() for p in parts):
            return None, f"{kind} schedule missing/invalid 'at' (expected HH:MM)"
        hour, minute = int(parts[0]), int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None, f"{kind} schedule 'at' out of range"
        if kind == "daily":
            return CronTrigger(hour=hour, minute=minute, timezone=tz), None
        weekday = schedule.get("weekday", 0)
        try:
            return (
                CronTrigger(day_of_week=str(weekday), hour=hour, minute=minute, timezone=tz),
                None,
            )
        except _NONCRITICAL_EXCEPTIONS:
            return None, f"weekly schedule invalid weekday {weekday!r}"

    if kind == "cron":
        expr = str(schedule.get("cron") or "").strip()
        if not expr:
            return None, "cron schedule missing expression"
        try:
            return CronTrigger.from_crontab(expr, timezone=tz), None
        except _NONCRITICAL_EXCEPTIONS as exc:
            return None, f"cron schedule invalid expression: {exc}"

    return None, f"unsupported schedule kind: {kind!r}"


def compute_run_slot(schedule: dict, trigger, *, now: datetime | None = None):
    """Return the UTC slot this fire belongs to, or ``None``.

    ``one_time`` slots are the ``run_at`` value directly. Periodic slots
    are re-derived at fire time the way the reminders scheduler does: the
    next fire time strictly after ``now - 1s`` -- for a trigger firing at
    ``now``, that is ``now``'s own slot (schedules with a period longer
    than one second, which interval validation floors well above).
    """
    if schedule.get("kind") == "one_time":
        tz_name = str(schedule.get("timezone") or _scheduler_timezone())
        return _parse_iso_datetime(str(schedule.get("run_at") or ""), timezone_name=tz_name)
    now = now or datetime.now(timezone.utc)
    try:
        return trigger.get_next_fire_time(None, now - timedelta(seconds=1))
    except _NONCRITICAL_EXCEPTIONS:
        return None


class _AutomationScheduler:
    """Arms configured automation definitions into the Jobs pipeline."""

    def __init__(self) -> None:
        self._aps: AsyncIOScheduler | None = None
        self._db_cache: dict[int, ScheduledTasksDatabase] = {}
        self._lock = asyncio.Lock()
        self._started = False
        self._rescan_task: asyncio.Task | None = None
        self._jobs = JobManager()

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            self._aps = AsyncIOScheduler(timezone=_scheduler_timezone())
            self._aps.start()
            await self._load_all()
            try:
                interval = int(
                    os.getenv("SCHEDULED_TASKS_AUTOMATION_RESCAN_SEC", "300") or 300
                )
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
                    except _NONCRITICAL_EXCEPTIONS as exc:
                        logger.debug("Automation scheduler rescan error: {}", exc)

            self._rescan_task = asyncio.create_task(
                _rescan_loop(), name="automation_scheduler_rescan"
            )
            self._started = True
            logger.info("Automation definition scheduler started")

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
            self._db_cache.clear()
            logger.info("Automation definition scheduler stopped")

    def _get_db(self, user_id: int) -> ScheduledTasksDatabase:
        if user_id not in self._db_cache:
            db = ScheduledTasksDatabase.for_user(user_id)
            db.ensure_schema()
            self._db_cache[user_id] = db
        return self._db_cache[user_id]

    def _enumerate_user_ids(self) -> set[int]:
        user_ids: set[int] = set()
        try:
            base = DatabasePaths.get_user_db_base_dir()
            for entry in base.iterdir():
                if entry.is_dir():
                    with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                        user_ids.add(int(entry.name))
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Automation scheduler: failed to enumerate user dirs: {}", exc)
        with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
            user_ids.add(int(core_settings.get("SINGLE_USER_FIXED_ID", 1)))
        return user_ids

    def _configured_definitions(self, user_id: int) -> list[DefinitionRow]:
        try:
            return self._get_db(user_id).list_definitions(
                owner_id=user_id, lifecycle="configured", limit=1000
            )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Automation scheduler: list failed for user {}: {}", user_id, exc)
            return []

    async def _load_all(self) -> None:
        armed = 0
        for uid in sorted(self._enumerate_user_ids()):
            for definition in self._configured_definitions(uid):
                if self._arm(definition, uid):
                    armed += 1
        if armed:
            logger.info("Automation scheduler armed {} definition(s)", armed)

    async def _rescan_once(self) -> None:
        if not self._aps:
            return
        desired: set[str] = set()
        for uid in sorted(self._enumerate_user_ids()):
            for definition in self._configured_definitions(uid):
                if self._arm(definition, uid):
                    desired.add(_job_id(definition.id))
        try:
            current_ids = {job.id for job in (self._aps.get_jobs() or [])}
            for stale_id in list(current_ids - desired):
                with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                    self._aps.remove_job(stale_id)
        except _NONCRITICAL_EXCEPTIONS:
            pass

    async def reconcile_definition(self, *, definition_id: str, user_id: int) -> None:
        """Immediately sync one definition into scheduler state."""
        async with self._lock:
            if not self._started or not self._aps:
                return
            definition = None
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                definition = self._get_db(int(user_id)).get_definition(
                    owner_id=int(user_id), definition_id=definition_id
                )
            if definition is None or definition.lifecycle != "configured":
                with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                    self._aps.remove_job(_job_id(definition_id))
                return
            self._arm(definition, int(user_id))

    async def unschedule_definition(self, *, definition_id: str) -> None:
        """Immediately unschedule one definition."""
        async with self._lock:
            if not self._started or not self._aps:
                return
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(_job_id(definition_id))

    def _arm(self, definition: DefinitionRow, user_id: int) -> bool:
        if not self._aps:
            return False
        trigger, reason = build_trigger(definition.schedule)
        if trigger is None:
            logger.warning(
                "Automation scheduler cannot arm definition {} (user {}): {}",
                definition.id,
                user_id,
                reason,
            )
            return False
        try:
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(_job_id(definition.id))
            self._aps.add_job(
                self._run_definition_schedule,
                trigger=trigger,
                id=_job_id(definition.id),
                args=[definition.id, user_id],
                max_instances=1,
                coalesce=True,
                misfire_grace_time=300,
            )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Automation scheduler failed to arm definition {}: {}", definition.id, exc
            )
            return False
        self._mark_ready(definition, user_id)
        return True

    def _mark_ready(self, definition: DefinitionRow, user_id: int) -> None:
        """Health honesty (TASK-13020 AC#4): armed definitions read ``ready``.

        Written only on change so rescans do not churn the definition's
        version column; audited through the standard trail.
        """
        if definition.health == ARMED_HEALTH:
            return
        db = self._get_db(user_id)
        try:
            updated = db.update_definition(
                owner_id=user_id,
                definition_id=definition.id,
                patch={"health": ARMED_HEALTH, "updated_by": SCHEDULER_ACTOR},
            )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Automation scheduler health update failed for {}: {}", definition.id, exc
            )
            return
        with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
            db.create_audit_event(
                owner_id=user_id,
                definition_id=definition.id,
                event_type="scheduler_armed",
                actor=SCHEDULER_ACTOR,
                summary=(
                    f"Definition armed by scheduler (schedule kind "
                    f"'{definition.schedule.get('kind')}'); health -> ready."
                ),
                before={"health": definition.health},
                after={"health": updated.health},
            )

    async def _run_definition_schedule(
        self, definition_id: str, user_id: int | None = None
    ) -> None:
        if user_id is None:
            return
        db = self._get_db(int(user_id))
        try:
            definition = db.get_definition(owner_id=int(user_id), definition_id=definition_id)
        except KeyError:
            return
        if definition is None or definition.lifecycle != "configured":
            return

        trigger, reason = build_trigger(definition.schedule)
        if trigger is None:
            logger.warning(
                "Automation scheduler fire with unusable schedule for {}: {}",
                definition_id,
                reason,
            )
            return

        slot = compute_run_slot(definition.schedule, trigger)
        if slot is None:
            logger.warning(
                "Automation scheduler could not determine run slot for {}", definition_id
            )
            return
        slot_utc = _parse_iso_datetime(_normalize_slot_to_utc_iso(slot))
        now_utc = datetime.now(timezone.utc)
        # The early-skip guard catches a wrongly-scheduled future trigger
        # (e.g. a stale APS job), not sub-minute clock skew between the
        # trigger's boundary and this callback -- the fire path can lag its
        # boundary by the misfire grace, so the tolerance here sits above
        # that, mirroring the reminders scheduler's intent.
        if slot_utc and slot_utc > (now_utc + timedelta(minutes=2)):
            logger.debug("Automation scheduler skipping early trigger for {}", definition_id)
            return

        payload = {
            "definition_id": definition.id,
            "user_id": int(user_id),
            "family": definition.family,
            "scheduled_for": _normalize_slot_to_utc_iso(slot),
        }
        idempotency_key = f"definition:{definition.id}:{payload['scheduled_for']}"
        try:
            job = self._jobs.create_job(
                domain=AUTOMATION_DOMAIN,
                queue=automation_jobs_queue(),
                job_type=AUTOMATION_JOB_TYPE,
                payload=payload,
                owner_user_id=int(user_id),
                idempotency_key=idempotency_key,
            )
            logger.info(
                "Automation definition queued: definition_id={} job_id={}",
                definition_id,
                job.get("id"),
            )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Automation definition enqueue failed for {}: {}", definition_id, exc
            )


def _job_id(definition_id: str) -> str:
    return f"automation:{definition_id}"


_INSTANCE: _AutomationScheduler | None = None


def get_automation_scheduler() -> _AutomationScheduler:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _AutomationScheduler()
    return _INSTANCE


async def start_automation_scheduler(enabled: bool | None = None) -> asyncio.Task | None:
    if enabled is None:
        enabled = env_flag_enabled("SCHEDULED_TASKS_AUTOMATION_SCHEDULER_ENABLED")
    if not enabled:
        return None
    scheduler = get_automation_scheduler()
    await scheduler.start()

    async def _noop() -> None:
        while True:
            await asyncio.sleep(60)

    return asyncio.create_task(_noop(), name="automation_scheduler")


async def stop_automation_scheduler(task: asyncio.Task | None) -> None:
    try:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except _NONCRITICAL_EXCEPTIONS:
        pass
    with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
        await get_automation_scheduler().stop()
