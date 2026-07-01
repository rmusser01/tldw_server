"""APScheduler bridge for Recurring Question scheduled task execution."""

from __future__ import annotations

import asyncio
import contextlib
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from tldw_Server_API.app.core.config import settings as core_settings
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import DefinitionRow, ScheduledTasksDatabase
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.services.scheduled_task_automation_service import ScheduledTaskAutomationError
from tldw_Server_API.app.services.scheduled_task_recurring_question_service import (
    ScheduledTaskRecurringQuestionService,
)

_MIN_RESCAN_SECONDS = 30
_MIN_RECONCILE_SECONDS = 60
_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_SCHEDULER_ACTOR = "scheduled-task-recurring-question-scheduler"


def recurring_question_scheduler_timezone() -> str:
    return (os.getenv("SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_TZ") or "UTC").strip() or "UTC"


def _normalize_slot_to_utc_iso(slot: datetime, *, timezone_name: str | None = None) -> str:
    if slot.tzinfo is None:
        if timezone_name:
            try:
                from zoneinfo import ZoneInfo

                slot = slot.replace(tzinfo=ZoneInfo(timezone_name))
            except _NONCRITICAL_EXCEPTIONS:
                slot = slot.replace(tzinfo=timezone.utc)
        else:
            slot = slot.replace(tzinfo=timezone.utc)
    return slot.astimezone(timezone.utc).isoformat()


class _RecurringQuestionScheduler:
    def __init__(self) -> None:
        self._aps: AsyncIOScheduler | None = None
        self._lock = asyncio.Lock()
        self._started = False
        self._rescan_task: asyncio.Task | None = None
        self._repo_cache: dict[int, ScheduledTasksDatabase] = {}
        self._service_cache: dict[int, ScheduledTaskRecurringQuestionService] = {}

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            tz = recurring_question_scheduler_timezone()
            self._aps = AsyncIOScheduler(timezone=tz)
            self._aps.start()
            await self._load_all()
            self._rescan_task = asyncio.create_task(
                self._rescan_loop(),
                name="scheduled_tasks_recurring_question_scheduler_rescan",
            )
            self._started = True
            logger.info("Scheduled Tasks Recurring Question scheduler started")

    async def stop(self) -> None:
        async with self._lock:
            if self._aps:
                with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                    self._aps.shutdown(wait=False)
            self._aps = None
            if self._rescan_task:
                self._rescan_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._rescan_task
            self._rescan_task = None
            self._repo_cache.clear()
            self._service_cache.clear()
            self._started = False
            logger.info("Scheduled Tasks Recurring Question scheduler stopped")

    async def rescan(self) -> None:
        async with self._lock:
            await self._rescan_once()

    async def _rescan_loop(self) -> None:
        rescan_interval = _env_int(
            "SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_RESCAN_SEC",
            300,
            minimum=_MIN_RESCAN_SECONDS,
        )
        reconcile_interval = _env_int(
            "SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_RECONCILE_SEC",
            300,
            minimum=_MIN_RECONCILE_SECONDS,
        )
        last_reconcile = datetime.now(timezone.utc)
        while True:
            try:
                await asyncio.sleep(rescan_interval)
                await self.rescan()
                now = datetime.now(timezone.utc)
                if (now - last_reconcile).total_seconds() >= reconcile_interval:
                    await self.reconcile_all_stale_runs()
                    last_reconcile = now
            except asyncio.CancelledError:
                break
            except _NONCRITICAL_EXCEPTIONS as exc:
                logger.debug("Recurring Question scheduler rescan error: {}", exc)

    async def _load_all(self) -> None:
        await self._rescan_once()

    async def _rescan_once(self) -> None:
        if not self._aps:
            return
        desired: set[str] = set()
        for owner_id in sorted(self._enumerate_user_ids()):
            repo = self._get_repo(owner_id)
            for definition in _iter_recurring_question_definitions(repo, owner_id=owner_id):
                job_id = self._job_id(owner_id, definition.id)
                if self._should_register(definition):
                    desired.add(job_id)
                    self._add_job(definition)
        current_ids = {job.id for job in (self._aps.get_jobs() or [])}
        for stale_id in sorted(current_ids - desired):
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(stale_id)

    async def reconcile_all_stale_runs(self) -> dict[int, list[str]]:
        repaired: dict[int, list[str]] = {}
        stale_after_seconds = _env_int(
            "SCHEDULED_TASKS_RECURRING_QUESTION_STALE_RUN_SEC",
            7200,
            minimum=60,
        )
        for owner_id in sorted(self._enumerate_user_ids()):
            service = self._get_service(owner_id)
            repaired_ids = service.reconcile_stale_runs(
                owner_id=owner_id,
                actor=_SCHEDULER_ACTOR,
                stale_after=timedelta(seconds=stale_after_seconds),
            )
            if repaired_ids:
                repaired[owner_id] = repaired_ids
        return repaired

    def _get_service(self, owner_id: int) -> ScheduledTaskRecurringQuestionService:
        if owner_id not in self._service_cache:
            repo = self._get_repo(owner_id)
            self._service_cache[owner_id] = ScheduledTaskRecurringQuestionService(
                repository=repo,
                job_manager=JobManager(),
            )
        return self._service_cache[owner_id]

    def _get_repo(self, owner_id: int) -> ScheduledTasksDatabase:
        if owner_id not in self._repo_cache:
            repo = ScheduledTasksDatabase.for_user(owner_id)
            repo.ensure_schema()
            self._repo_cache[owner_id] = repo
        return self._repo_cache[owner_id]

    def _enumerate_user_ids(self) -> set[int]:
        user_ids: set[int] = set()
        try:
            base = DatabasePaths.get_user_db_base_dir()
            for entry in base.iterdir():
                if entry.is_dir():
                    with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                        user_ids.add(int(entry.name))
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Recurring Question scheduler failed to enumerate user dirs: {}", exc)
        with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
            user_ids.add(int(core_settings.get("SINGLE_USER_FIXED_ID", 1)))
        return user_ids

    @staticmethod
    def _should_register(definition: DefinitionRow) -> bool:
        return (
            definition.family == "recurring_question"
            and definition.lifecycle == "configured"
            and definition.resolution_state == "open"
            and str(definition.schedule.get("kind") or "") != "one_time"
        )

    def _add_job(self, definition: DefinitionRow) -> None:
        if not self._aps:
            return
        job_id = self._job_id(definition.owner_id, definition.id)
        try:
            trigger = _build_trigger(definition.schedule)
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Recurring Question scheduler invalid schedule owner_id={} definition_id={} error={}",
                definition.owner_id,
                definition.id,
                exc,
            )
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(job_id)
            return
        if trigger is None:
            with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
                self._aps.remove_job(job_id)
            return
        with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
            self._aps.remove_job(job_id)
        self._aps.add_job(
            self._run_definition_schedule,
            trigger=trigger,
            id=job_id,
            args=[definition.owner_id, definition.id, definition.version],
            max_instances=1,
            coalesce=True,
            misfire_grace_time=_env_int(
                "SCHEDULED_TASKS_RECURRING_QUESTION_MISFIRE_GRACE_SEC",
                300,
                minimum=1,
            ),
        )

    async def _run_definition_schedule(
        self,
        owner_id: int,
        definition_id: str,
        definition_version: int,
    ) -> None:
        slot = datetime.now(timezone.utc).replace(microsecond=0)
        try:
            await self._enqueue_due_slot(
                owner_id=owner_id,
                definition_id=definition_id,
                definition_version=definition_version,
                schedule_slot=slot,
            )
        except ScheduledTaskAutomationError as exc:
            if exc.code != "run_in_progress":
                logger.info(
                    "Recurring Question schedule skipped owner_id={} definition_id={} reason={}",
                    owner_id,
                    definition_id,
                    exc.code,
                )
        except _NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Recurring Question scheduler enqueue failed owner_id={} definition_id={} error={}",
                owner_id,
                definition_id,
                exc,
            )

    async def _enqueue_due_slot(
        self,
        *,
        owner_id: int,
        definition_id: str,
        definition_version: int,
        schedule_slot: datetime,
    ):
        slot = _normalize_slot_to_utc_iso(schedule_slot)
        return self._get_service(owner_id).create_scheduled_run(
            owner_id=owner_id,
            actor=_SCHEDULER_ACTOR,
            definition_id=definition_id,
            definition_version=definition_version,
            schedule_slot=slot,
        )

    @staticmethod
    def _job_id(owner_id: int, definition_id: str) -> str:
        return f"scheduled-task-rq:{owner_id}:{definition_id}"


def _iter_recurring_question_definitions(
    repo: ScheduledTasksDatabase,
    *,
    owner_id: int,
) -> list[DefinitionRow]:
    rows: list[DefinitionRow] = []
    offset = 0
    page_size = 200
    while True:
        page, _total = repo.list_definitions(
            owner_id=owner_id,
            family="recurring_question",
            limit=page_size,
            offset=offset,
        )
        rows.extend(page)
        if len(page) < page_size:
            return rows
        offset += len(page)


def _build_trigger(schedule: dict[str, Any]) -> Any | None:
    kind = str(schedule.get("kind") or "").strip()
    if not kind or kind == "one_time":
        return None
    timezone_name = str(schedule.get("timezone") or recurring_question_scheduler_timezone())
    if kind == "cron":
        expression = str(schedule.get("cron") or schedule.get("crontab") or schedule.get("expression") or "").strip()
        if not expression:
            return None
        return CronTrigger.from_crontab(expression, timezone=timezone_name)
    if kind == "daily":
        hour, minute, second = _parse_time_fields(schedule)
        return CronTrigger(hour=hour, minute=minute, second=second, timezone=timezone_name)
    if kind == "weekly":
        hour, minute, second = _parse_time_fields(schedule)
        day_of_week = _normalize_day_of_week(
            schedule.get("days") or schedule.get("weekdays") or schedule.get("day_of_week")
        )
        if not day_of_week:
            return None
        return CronTrigger(day_of_week=day_of_week, hour=hour, minute=minute, second=second, timezone=timezone_name)
    if kind == "interval":
        seconds = _interval_seconds(schedule)
        if seconds is None:
            return None
        return IntervalTrigger(seconds=seconds, timezone=timezone_name)
    return None


def _parse_time_fields(schedule: dict[str, Any]) -> tuple[int, int, int]:
    raw_time = str(schedule.get("time") or "00:00:00").strip()
    parts = raw_time.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        second = int(parts[2]) if len(parts) > 2 else 0
    except (TypeError, ValueError, IndexError):
        return 0, 0, 0
    return hour, minute, second


def _normalize_day_of_week(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, list):
        return None
    normalized: list[str] = []
    for item in value:
        if isinstance(item, int):
            normalized.append(str(item))
        elif isinstance(item, str) and item.strip():
            normalized.append(item.strip().lower()[:3])
    return ",".join(normalized) if normalized else None


def _interval_seconds(schedule: dict[str, Any]) -> int | None:
    for key, multiplier in (
        ("seconds", 1),
        ("every_seconds", 1),
        ("minutes", 60),
        ("every_minutes", 60),
        ("hours", 3600),
        ("every_hours", 3600),
    ):
        value = schedule.get(key)
        if value is None:
            continue
        try:
            seconds = int(value) * multiplier
        except (TypeError, ValueError):
            return None
        return seconds if seconds > 0 else None
    return None


def _env_int(key: str, default: int, *, minimum: int) -> int:
    try:
        value = int(os.getenv(key, str(default)) or default)
    except _NONCRITICAL_EXCEPTIONS:
        value = default
    return max(minimum, value)


_INSTANCE: _RecurringQuestionScheduler | None = None


def get_scheduled_task_recurring_question_scheduler() -> _RecurringQuestionScheduler:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _RecurringQuestionScheduler()
    return _INSTANCE


async def start_scheduled_task_recurring_question_scheduler(enabled: bool | None = None) -> asyncio.Task | None:
    if enabled is None:
        enabled = env_flag_enabled("SCHEDULED_TASKS_RECURRING_QUESTION_SCHEDULER_ENABLED")
    if not enabled:
        return None
    scheduler = get_scheduled_task_recurring_question_scheduler()
    await scheduler.start()

    async def _noop() -> None:
        while True:
            await asyncio.sleep(60)

    return asyncio.create_task(_noop(), name="scheduled_tasks_recurring_question_scheduler")


async def stop_scheduled_task_recurring_question_scheduler(task: asyncio.Task | None) -> None:
    if task:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    with contextlib.suppress(_NONCRITICAL_EXCEPTIONS):
        await get_scheduled_task_recurring_question_scheduler().stop()


__all__ = [
    "_RecurringQuestionScheduler",
    "_normalize_slot_to_utc_iso",
    "get_scheduled_task_recurring_question_scheduler",
    "recurring_question_scheduler_timezone",
    "start_scheduled_task_recurring_question_scheduler",
    "stop_scheduled_task_recurring_question_scheduler",
]
