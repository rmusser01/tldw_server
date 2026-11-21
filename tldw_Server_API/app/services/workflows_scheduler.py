"""
Workflows Scheduler service

Provides a lightweight recurring scheduler (cron-based) that enqueues
`workflow_run` tasks into the core Scheduler and persists definitions
in the Workflows Scheduler DB.

Env:
  WORKFLOWS_SCHEDULER_ENABLED=false  -> opt out of startup (enabled by default)
  WORKFLOWS_SCHEDULER_TZ=<IANA>      -> default timezone (e.g., UTC)
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
import os
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.schedulers.base import STATE_PAUSED, STATE_RUNNING
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from tldw_Server_API.app.core.AuthNZ.session_manager import get_session_manager
from tldw_Server_API.app.core.config import settings as core_settings
from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseError as BackendDatabaseError,
)
from tldw_Server_API.app.core.DB_Management.Workflows_Scheduler_DB import (
    WorkflowSchedule,
    WorkflowsSchedulerDB,
)
from tldw_Server_API.app.core.Scheduler import Scheduler, SchedulerError, get_global_scheduler
from tldw_Server_API.app.core.Scheduler.handlers import (
    watchlists as _ensure_watchlists,  # noqa: F401  # register watchlist_run
)
from tldw_Server_API.app.core.Scheduler.handlers import (
    acp as _ensure_acp_handlers,  # noqa: F401  # register acp_run
)
from tldw_Server_API.app.core.Scheduler.handlers import (
    workflows as _ensure_handlers,  # noqa: F401  # register workflow_run
)
from tldw_Server_API.app.core.testing import env_flag_enabled

_WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS = (
    asyncio.TimeoutError,
    AssertionError,
    AttributeError,
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
    builtins.TimeoutError,
    TypeError,
    ValueError,
    UnicodeDecodeError,
    BackendDatabaseError,
    SchedulerError,
)


def build_schedule_payload(schedule: WorkflowSchedule) -> dict[str, Any]:
    """Build the Scheduler payload for a recurring workflow schedule.

    The returned payload preserves the workflow id, user/tenant routing fields,
    execution mode, validation mode, and decoded inputs. Malformed persisted
    ``inputs_json`` is treated as an empty dict so one corrupt schedule row does
    not abort a scheduler fire.
    """
    try:
        inputs = json.loads(schedule.inputs_json or "{}")
    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as exc:
        logger.warning(
            "Workflows scheduler: malformed inputs_json for schedule {}: {}",
            schedule.id,
            exc,
        )
        inputs = {}
    return {
        "workflow_id": schedule.workflow_id,
        "inputs": inputs,
        "user_id": schedule.user_id,
        "tenant_id": schedule.tenant_id,
        "mode": schedule.run_mode,
        "validation_mode": schedule.validation_mode,
    }


def resolve_schedule_submission_target(payload: dict[str, Any]) -> tuple[str, str]:
    """Return the Scheduler handler and queue for a schedule payload.

    Watchlist-backed schedules are identified by ``inputs.watchlist_job_id`` and
    route to the watchlists queue; all other schedules route to the workflows
    queue as standard workflow runs.
    """
    inputs = payload.get("inputs")
    if isinstance(inputs, dict) and inputs.get("watchlist_job_id"):
        return "watchlist_run", "watchlists"
    return "workflow_run", "workflows"


async def ensure_schedule_queue_worker(core: Scheduler, queue_name: str) -> None:
    """Require real queue capacity without downscaling existing workers."""
    pool = core.worker_pool
    workers = pool.get_status().get("workers_by_queue", {}) if pool else {}
    if workers.get(queue_name, 0) < 1:
        if await core.scale_workers(1, queue_name) < 1:
            raise SchedulerError(f"No worker capacity for queue {queue_name}")


_CRON_LEASE = "workflows:recurring-cron"


class _WFRecurringScheduler:
    def __init__(self) -> None:
        self._core_scheduler: Scheduler | None = None
        self._aps: AsyncIOScheduler | None = None
        self._db = WorkflowsSchedulerDB()
        self._lock = asyncio.Lock()
        self._started = False
        # Cache of per-user scheduler DB handles
        self._db_cache: dict[int, WorkflowsSchedulerDB] = {}
        self._rescan_task: asyncio.Task | None = None

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return
            # Rebuild DB handles on each cold start to pick up env/path changes.
            self._db_cache.clear()
            previous_db = self._db
            self._db = WorkflowsSchedulerDB()
            if previous_db.backend.config.backend_type == BackendType.POSTGRESQL:
                previous_db.backend.get_pool().close_all()
            # Start or reuse the global core job scheduler (workers)
            self._core_scheduler = await get_global_scheduler()
            # Start APScheduler for cron
            tz = os.getenv("WORKFLOWS_SCHEDULER_TZ", "UTC")
            self._aps = AsyncIOScheduler(timezone=tz)
            self._aps.start(paused=True)
            await self._refresh_cron_owner()
            # Periodic rescan to pick up new/removed schedules
            try:
                interval = max(1, int(os.getenv("WORKFLOWS_SCHEDULER_RESCAN_SEC", "30") or 30))
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
                interval = 30
            leadership = self._core_scheduler.leader_election
            heartbeat = max(1, min(5, leadership.config.leader_ttl_seconds // 3)) if leadership else 5

            async def _rescan_loop():
                loop = asyncio.get_running_loop()
                next_scan = loop.time() + interval
                while True:
                    try:
                        await asyncio.sleep(min(interval, heartbeat))
                        if await self._refresh_cron_owner() and loop.time() >= next_scan:
                            await self._rescan_once()
                            next_scan = loop.time() + interval
                    except asyncio.CancelledError:
                        break
                    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Workflows scheduler: rescan error: {e}")
            self._rescan_task = asyncio.create_task(_rescan_loop(), name="workflows_scheduler_rescan")
            self._started = True
            logger.info("Workflows recurring scheduler started")

    async def _refresh_cron_owner(self) -> bool:
        """Resume only the APScheduler holding the shared recurring-dispatch lease."""
        leadership = self._core_scheduler.leader_election if self._core_scheduler else None
        if self._aps is None or leadership is None:
            return False
        if self._aps.state == STATE_RUNNING:
            if await leadership.is_leader(_CRON_LEASE):
                return True
            self._aps.pause()
        if not await leadership.acquire_leadership(_CRON_LEASE):
            return False
        if self._aps.state == STATE_PAUSED:
            self._aps.remove_all_jobs()
            await self._load_all()
            self._aps.resume()
        return True

    async def _may_submit_cron(self) -> bool:
        if self._aps is None or self._aps.state != STATE_RUNNING:
            return False
        leadership = self._core_scheduler.leader_election if self._core_scheduler else None
        return leadership is not None and await leadership.is_leader(_CRON_LEASE)

    async def stop(self) -> None:
        async with self._lock:
            try:
                if self._rescan_task:
                    self._rescan_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._rescan_task
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: rescan task cancel failed: {e}")
            self._rescan_task = None
            try:
                if self._aps:
                    self._aps.pause()
                    self._aps.shutdown(wait=False)
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: APS shutdown failed: {e}")
            try:
                if self._core_scheduler:
                    leadership = self._core_scheduler.leader_election
                    if leadership is not None:
                        await leadership.release_leadership(_CRON_LEASE)
                    await self._core_scheduler.stop()
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: core scheduler stop failed: {e}")
            self._aps = None
            self._core_scheduler = None
            # Ensure per-user DB handles are rebuilt on next start so tests/env
            # changes (e.g. USER_DB_BASE_DIR) do not leak stale scheduler paths.
            self._db_cache.clear()
            self._started = False
            logger.info("Workflows recurring scheduler stopped")

    async def _load_all(self) -> None:
        """Scan all user directories and register their schedules."""
        loaded = 0
        try:
            seen_schedule_ids: set[str] = set()
            for uid in self._schedule_scan_user_ids():
                try:
                    items = self._list_registered_schedules(uid)
                except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Workflows scheduler: list_schedules failed for user {uid}: {e}")
                    items = []
                for s in items:
                    if not s.enabled or s.id in seen_schedule_ids:
                        continue
                    seen_schedule_ids.add(s.id)
                    effective_uid = self._resolve_schedule_owner_id(s, fallback_user_id=uid)
                    acp_cfg = getattr(s, "acp_config_json", None)
                    if acp_cfg:
                        self._add_acp_job(s, effective_uid)
                    else:
                        self._add_job(s, effective_uid)
                    loaded += 1
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler load_all failed: {e}")
        if loaded:
            logger.info(f"Workflows scheduler: registered {loaded} schedule(s)")

    def _schedule_scan_user_ids(self) -> list[int]:
        fallback_user_id = DatabasePaths.get_single_user_id()
        if self._db.backend.config.backend_type == BackendType.POSTGRESQL:
            return [fallback_user_id]
        user_ids = {fallback_user_id}
        try:
            base = DatabasePaths.get_user_db_base_dir()
            for p in base.iterdir():
                if p.is_dir():
                    try:
                        user_ids.add(int(p.name))
                    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
                        continue
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Workflows scheduler: failed to enumerate user dirs: {exc}")
        return sorted(user_ids)

    def _get_db(self, user_id: int) -> WorkflowsSchedulerDB:
        """Reuse the shared PostgreSQL database or open the user’s SQLite database."""
        if self._db.backend.config.backend_type == BackendType.POSTGRESQL:
            return self._db
        if user_id not in self._db_cache:
            self._db_cache[user_id] = WorkflowsSchedulerDB(user_id=user_id)
        return self._db_cache[user_id]

    def _list_registered_schedules(self, user_id: int) -> list[WorkflowSchedule]:
        """Return every persisted schedule visible in a user's scheduler DB.

        Shared backends can expose schedules for more than one owner through a
        single DB handle, so this intentionally uses ``user_id=None`` and pages
        until the DB returns a short page.
        """
        db = self._get_db(user_id)
        page_size = 1000
        offset = 0
        schedules: list[WorkflowSchedule] = []
        while True:
            if hasattr(db, "list_all_schedules"):
                page = db.list_all_schedules(user_id=None, limit=page_size, offset=offset)
            else:
                page = db.list_schedules(
                    tenant_id="default",
                    user_id=None,
                    limit=page_size,
                    offset=offset,
                )
            schedules.extend(page)
            if len(page) < page_size:
                return schedules
            offset += len(page)

    @staticmethod
    def _resolve_schedule_owner_id(schedule: WorkflowSchedule, *, fallback_user_id: int) -> int:
        """Resolve the owner user id stored on a schedule.

        Older rows can contain malformed owner values; in that case the caller's
        enumerated user id is used so scheduling can continue.
        """
        try:
            return int(schedule.user_id)
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
            return fallback_user_id

    async def _rescan_once(self) -> None:
        if not self._aps:
            return
        # Collect desired enabled schedule IDs from all users
        desired: set[str] = set()
        seen_schedule_ids: set[str] = set()
        for uid in self._schedule_scan_user_ids():
            try:
                items = self._list_registered_schedules(uid)
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: list_schedules failed for user {uid}: {e}")
                items = []
            for s in items:
                if not s.enabled or s.id in seen_schedule_ids:
                    continue
                seen_schedule_ids.add(s.id)
                desired.add(s.id)
                effective_uid = self._resolve_schedule_owner_id(s, fallback_user_id=uid)
                acp_cfg = getattr(s, "acp_config_json", None)
                if acp_cfg:
                    self._add_acp_job(s, effective_uid)
                else:
                    self._add_job(s, effective_uid)
        # Remove jobs that no longer exist or are disabled
        try:
            current_ids = {j.id for j in (self._aps.get_jobs() or [])}
            for jid in list(current_ids - desired):
                try:
                    self._aps.remove_job(jid)
                except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Workflows scheduler: failed to remove job {jid}: {e}")
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: failed to reconcile jobs: {e}")

    def _add_job(self, schedule: WorkflowSchedule, user_id: int | None = None) -> None:
        """Register an APScheduler job for a recurring workflow schedule."""
        if not self._aps:
            return
        try:
            tz = schedule.timezone or os.getenv("WORKFLOWS_SCHEDULER_TZ", "UTC")
            # Validate cron; provide feedback via logs on errors
            try:
                trigger = CronTrigger.from_crontab(schedule.cron, timezone=tz)
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Invalid cron for schedule {schedule.id}: {e}")
                if self._aps.get_job(schedule.id) is not None:
                    self._aps.remove_job(schedule.id)
                return

            # Per-job concurrency: skip vs queue
            # - skip: max_instances=1, coalesce=True
            # - queue: allow overlap (max_instances>1), coalesce=False
            if (schedule.concurrency_mode or "skip").lower() == "queue":
                max_instances = 3
                coalesce = False if schedule.coalesce is None else bool(schedule.coalesce)
            else:
                max_instances = 1
                coalesce = True if schedule.coalesce is None else bool(schedule.coalesce)

            misfire_grace_time = 300 if schedule.misfire_grace_sec is None else int(schedule.misfire_grace_sec)
            # Pass user_id so run handler can pick correct per-user DB
            effective_uid = user_id if user_id is not None else int(schedule.user_id)
            # Determine jitter: prefer enabling for watchlist jobs to avoid the "on the hour" thundering herd
            jitter_sec = 0
            try:
                raw_inputs = __import__("json").loads(schedule.inputs_json or "{}")
                is_watchlist = isinstance(raw_inputs, dict) and bool(raw_inputs.get("watchlist_job_id"))
                if is_watchlist:
                    try:
                        jitter_env = os.getenv("WATCHLISTS_SCHEDULER_JITTER_SEC", "90")
                        jitter_sec = int(jitter_env) if str(jitter_env).strip() else 90
                        if jitter_sec < 0:
                            jitter_sec = 0
                    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                        logger.debug(f"Workflows scheduler: invalid WATCHLISTS_SCHEDULER_JITTER_SEC: {e}")
                        jitter_sec = 90
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: watchlist jitter parse failed: {e}")
                jitter_sec = 0

            # Persist jitter metadata for watchlist schedules
            try:
                if jitter_sec != schedule.jitter_sec:
                    self._get_db(effective_uid).update_schedule(schedule.id, {"jitter_sec": jitter_sec})
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: failed to persist jitter for {schedule.id}: {e}")

            trigger.jitter = jitter_sec or None
            self._register_cron_job(
                schedule, effective_uid, self._run_schedule, trigger,
                max_instances=max_instances, coalesce=coalesce, misfire_grace_time=misfire_grace_time,
            )
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to add schedule job {schedule.id}: {e}")

    def _add_acp_job(self, schedule: WorkflowSchedule, user_id: int | None = None) -> None:
        """Register an APScheduler job for an ACP agent schedule.

        Similar to ``_add_job`` but submits an ``acp_run`` task instead of
        ``workflow_run``, using the ACP-specific configuration stored in
        ``acp_config_json``.
        """
        import json as _json

        if not self._aps:
            return
        try:

            tz = schedule.timezone or os.getenv("WORKFLOWS_SCHEDULER_TZ", "UTC")
            try:
                trigger = CronTrigger.from_crontab(schedule.cron, timezone=tz)
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.warning(f"Invalid cron for ACP schedule {schedule.id}: {e}")
                if self._aps.get_job(schedule.id) is not None:
                    self._aps.remove_job(schedule.id)
                return

            # Concurrency settings -- ACP jobs default to skip
            if (schedule.concurrency_mode or "skip").lower() == "queue":
                max_instances = 3
                coalesce = False if schedule.coalesce is None else bool(schedule.coalesce)
            else:
                max_instances = 1
                coalesce = True if schedule.coalesce is None else bool(schedule.coalesce)

            misfire_grace_time = 300 if schedule.misfire_grace_sec is None else int(schedule.misfire_grace_sec)
            effective_uid = user_id if user_id is not None else int(schedule.user_id)

            self._register_cron_job(
                schedule, effective_uid, self._run_acp_schedule, trigger,
                max_instances=max_instances, coalesce=coalesce, misfire_grace_time=misfire_grace_time,
            )
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to add ACP schedule job {schedule.id}: {e}")

    @staticmethod
    def _cron_definition(schedule: WorkflowSchedule) -> tuple[str, str | None, bool]:
        return schedule.cron, schedule.timezone, bool(schedule.acp_config_json)

    def _register_cron_job(
        self, schedule: WorkflowSchedule, user_id: int, callback: Callable[..., Awaitable[None]],
        trigger: CronTrigger, *, max_instances: int, coalesce: bool, misfire_grace_time: int,
    ) -> None:
        if self._aps is None:
            return
        job = self._aps.get_job(schedule.id)
        args = (schedule.id, user_id)
        kwargs = {"expected_cron": self._cron_definition(schedule)}
        options = {"max_instances": max_instances, "coalesce": coalesce, "misfire_grace_time": misfire_grace_time}
        if (
            job is not None
            and job.args == args
            and job.kwargs == kwargs
            and isinstance(job.trigger, CronTrigger)
            and job.trigger.timezone == trigger.timezone
            and job.trigger.jitter == trigger.jitter
        ):
            # Preserve the pending occurrence whenever the trigger is unchanged.
            if any(getattr(job, key) != value for key, value in options.items()):
                job.modify(**options)
        else:
            job = self._aps.add_job(
                callback, trigger=trigger, id=schedule.id, args=args, kwargs=kwargs,
                replace_existing=True, **options,
            )
        try:
            next_run = job.next_run_time
            stored_next = schedule.next_run_at
            if stored_next is not None:
                with contextlib.suppress(ValueError):
                    stored_next = datetime.fromisoformat(stored_next)
            if next_run is not None and next_run != stored_next:
                self._get_db(user_id).set_history(schedule.id, next_run_at=next_run.isoformat())
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: failed to set next_run_at for {schedule.id}: {e}")

    async def _run_acp_schedule(
        self, schedule_id: str, user_id: int, *, expected_cron: tuple[str, str | None, bool] | None = None
    ) -> None:
        """Execute an ACP schedule by submitting an ``acp_run`` task."""
        import json as _json

        db = self._get_db(user_id)
        s = db.get_schedule(schedule_id)
        if not s or (expected_cron is not None and expected_cron != self._cron_definition(s)):
            return
        if not s.enabled:
            try:
                db.set_history(schedule_id, last_status="skipped_disabled")
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: failed to set skipped status for ACP schedule {schedule_id}: {e}")
            return

        # Record last_run_at and pending status
        try:
            from datetime import timezone as _tz
            db.set_history(schedule_id, last_run_at=datetime.now(_tz.utc).isoformat(), last_status="pending")
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: failed to set pending status for ACP schedule {schedule_id}: {e}")

        # Parse ACP config
        try:
            acp_config = _json.loads(s.acp_config_json) if isinstance(s.acp_config_json, str) else (s.acp_config_json or {})
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as exc:
            logger.warning(
                "Workflows scheduler: malformed acp_config_json for ACP schedule {}: {}",
                s.id,
                exc,
            )
            acp_config = {}

        payload = {
            "user_id": user_id,
            "prompt": acp_config.get("prompt", ""),
            "cwd": acp_config.get("cwd", "."),
            "agent_type": acp_config.get("agent_type"),
            "model": acp_config.get("model"),
            "token_budget": acp_config.get("token_budget"),
            "persona_id": acp_config.get("persona_id"),
            "workspace_id": acp_config.get("workspace_id"),
            "sandbox_enabled": acp_config.get("sandbox_enabled", False),
        }

        try:
            if self._core_scheduler is None:
                logger.warning("Core Scheduler not initialized; skipping ACP schedule run")
                return
            if expected_cron is not None and not await self._may_submit_cron():
                return
            task_id = await self._core_scheduler.submit(
                handler="acp_run",
                payload=payload,
                queue_name="acp",
                metadata={"user_id": str(user_id)},
            )
            logger.info(f"Scheduled acp_run submitted: task_id={task_id} schedule_id={s.id}")
            try:
                db.set_history(schedule_id, last_status="queued")
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: failed to set queued status for ACP schedule {schedule_id}: {e}")
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to submit scheduled acp_run: {e}")
            try:
                db.set_history(schedule_id, last_status="error")
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e2:
                logger.debug(f"Workflows scheduler: failed to set error status for ACP schedule {schedule_id}: {e2}")

    async def _run_schedule(
        self, schedule_id: str, user_id: int | None = None, *, expected_cron: tuple[str, str | None, bool] | None = None
    ) -> None:
        # Fetch latest schedule in case it was modified
        # Backward compatibility: determine user_id from stored schedule when not provided
        db = None
        if user_id is not None:
            db = self._get_db(int(user_id))
            s = db.get_schedule(schedule_id)
        else:
            s = self.get(schedule_id)
            if s is None:
                return
            try:
                db = self._get_db(int(s.user_id))
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
                db = self._db
        if not s or not s.enabled or (expected_cron is not None and expected_cron != self._cron_definition(s)):
            return
        try:
            fallback_owner_id = (
                int(user_id)
                if user_id is not None
                else int(core_settings.get("SINGLE_USER_FIXED_ID", 1))
            )
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
            fallback_owner_id = 1
        owner_user_id = self._resolve_schedule_owner_id(s, fallback_user_id=fallback_owner_id)
        # Record last_run_at and pending status
        try:
            from datetime import timezone
            db.set_history(schedule_id, last_run_at=datetime.now(timezone.utc).isoformat(), last_status="pending")
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: failed to set pending status for {schedule_id}: {e}")
        payload = build_schedule_payload(s)
        payload["user_id"] = str(owner_user_id)
        # Presence gating: optionally skip when user is offline
        try:
            if getattr(s, "require_online", False):
                sm = await get_session_manager()
                sessions = await sm.get_active_sessions(owner_user_id)
                if not sessions:
                    # mark skipped and compute next run time
                    try:
                        tz = s.timezone or os.getenv("WORKFLOWS_SCHEDULER_TZ", "UTC")
                        trigger = CronTrigger.from_crontab(s.cron, timezone=tz)
                        now = datetime.now(trigger.timezone)
                        nxt = trigger.get_next_fire_time(None, now)
                        db.set_history(schedule_id, last_status="skipped_offline", next_run_at=(nxt.isoformat() if nxt else None))
                    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
                        db.set_history(schedule_id, last_status="skipped_offline")
                    return
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as _e:
            logger.debug(f"Presence gating check failed for schedule {schedule_id}: {_e}")
        # Optionally mint a short-lived, scoped bearer token and inject into run secrets
        try:
            use_vk = env_flag_enabled("WORKFLOWS_MINT_VIRTUAL_KEYS")
            if use_vk:
                from tldw_Server_API.app.core.AuthNZ.jwt_service import JWTService
                from tldw_Server_API.app.core.AuthNZ.settings import get_settings as _get_settings
                settings = _get_settings()
                jwt_svc = JWTService(settings)
                ttl = int(os.getenv("WORKFLOWS_VIRTUAL_KEY_TTL_MIN", "15") or 15)
                token = jwt_svc.create_virtual_access_token(
                    user_id=owner_user_id,
                    username=str(owner_user_id),
                    role="user",
                    scope="workflows",
                    ttl_minutes=ttl,
                    schedule_id=str(schedule_id),
                )
                payload["secrets"] = {"jwt": token}
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as _vk_e:
            logger.debug(f"Scheduler: virtual-key minting disabled/failed: {_vk_e}")
        try:
            if self._core_scheduler is None:
                logger.warning("Core Scheduler not initialized; skipping schedule run")
                return
            handler_name, queue_name = resolve_schedule_submission_target(payload)
            await ensure_schedule_queue_worker(self._core_scheduler, queue_name)
            if expected_cron is not None and not await self._may_submit_cron():
                return
            task_id = await self._core_scheduler.submit(
                handler=handler_name,
                payload=payload,
                queue_name=queue_name,
                metadata={"user_id": str(owner_user_id)},
            )
            logger.info(f"Scheduled {handler_name} submitted: task_id={task_id} schedule_id={s.id}")
            try:
                db.set_history(schedule_id, last_status="queued")
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: failed to set queued status for {schedule_id}: {e}")
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.warning(f"Failed to submit scheduled workflow: {e}")
            try:
                db.set_history(schedule_id, last_status="error")
            except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                logger.debug(f"Workflows scheduler: failed to set error status for {schedule_id}: {e}")

    # CRUD wrappers
    def create(self, *, tenant_id: str, user_id: str, workflow_id: int | None, name: str | None, cron: str, timezone: str | None, inputs: dict[str, Any], run_mode: str, validation_mode: str, enabled: bool, concurrency_mode: str = "skip", misfire_grace_sec: int = 300, coalesce: bool = True, require_online: bool = False, acp_config_json: str | None = None) -> str:
        sid = __import__("uuid").uuid4().hex
        db = self._get_db(int(user_id))
        db.create_schedule(
            id=sid,
            tenant_id=tenant_id,
            user_id=user_id,
            workflow_id=workflow_id,
            name=name,
            cron=cron,
            timezone=timezone,
            inputs=inputs,
            run_mode=run_mode,
            validation_mode=validation_mode,
            enabled=enabled,
            require_online=require_online,
            concurrency_mode=concurrency_mode,
            misfire_grace_sec=int(misfire_grace_sec),
            coalesce=bool(coalesce),
            acp_config_json=acp_config_json,
        )
        s = db.get_schedule(sid)
        if s and s.enabled:
            if s.acp_config_json:
                self._add_acp_job(s, int(user_id))
            else:
                self._add_job(s, int(user_id))
        return sid

    def update(self, schedule_id: str, update: dict[str, Any]) -> bool:
        # Resolve correct DB by locating the schedule first
        s = self.get(schedule_id)
        if not s:
            return False
        db = self._get_db(int(s.user_id))
        ok = db.update_schedule(schedule_id, update)
        s = db.get_schedule(schedule_id)
        if s:
            if s.enabled:
                if s.acp_config_json:
                    self._add_acp_job(s, int(s.user_id))
                else:
                    self._add_job(s, int(s.user_id))
            else:
                try:
                    if self._aps:
                        self._aps.remove_job(schedule_id)
                except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(f"Workflows scheduler: failed to remove job {schedule_id}: {e}")
        return ok

    def delete(self, schedule_id: str) -> bool:
        s = self.get(schedule_id)
        if not s:
            return False
        try:
            if self._aps:
                self._aps.remove_job(schedule_id)
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: failed to remove job {schedule_id}: {e}")
        db = self._get_db(int(s.user_id))
        return db.delete_schedule(schedule_id)

    def get(self, schedule_id: str) -> WorkflowSchedule | None:
        # Check default DB first
        try:
            found = self._db.get_schedule(schedule_id)
            if found or self._db.backend.config.backend_type == BackendType.POSTGRESQL:
                return found
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(
                f"Workflows scheduler: default DB lookup failed for schedule {schedule_id}: {e}"
            )
        # Scan per-user DBs
        try:
            base = DatabasePaths.get_user_db_base_dir()
            for p in base.iterdir():
                if not p.is_dir():
                    continue
                try:
                    uid = int(p.name)
                except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS:
                    continue
                try:
                    db = self._get_db(uid)
                    s = db.get_schedule(schedule_id)
                except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
                    logger.debug(
                        f"Workflows scheduler: user DB lookup failed for schedule {schedule_id} in user {uid}: {e}"
                    )
                    continue
                if s:
                    return s
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: failed to locate schedule {schedule_id}: {e}")
        return None

    def list(self, *, tenant_id: str, user_id: str | None = None, limit: int = 50, offset: int = 0) -> builtins.list[WorkflowSchedule]:
        # Require user_id to select correct per-user DB
        if not user_id:
            return []
        try:
            db = self._get_db(int(user_id))
            return db.list_schedules(tenant_id=tenant_id, user_id=user_id, limit=limit, offset=offset)
        except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
            logger.debug(f"Workflows scheduler: list failed for user {user_id}: {e}")
            return []


_INSTANCE: _WFRecurringScheduler | None = None


def get_workflows_scheduler() -> _WFRecurringScheduler:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = _WFRecurringScheduler()
    return _INSTANCE


async def start_workflows_scheduler() -> asyncio.Task | None:
    if not env_flag_enabled("WORKFLOWS_SCHEDULER_ENABLED", default=True):
        return None
    svc = get_workflows_scheduler()
    await svc.start()
    # return a dummy task to integrate with lifespan management
    async def _noop():
        while True:
            await asyncio.sleep(60)
    task = asyncio.create_task(_noop(), name="workflows_recurring_scheduler")
    return task


async def stop_workflows_scheduler(task: asyncio.Task | None) -> None:
    try:
        if task:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows scheduler: stop task cancel failed: {e}")
    try:
        await get_workflows_scheduler().stop()
    except _WORKFLOWS_SCHED_NONCRITICAL_EXCEPTIONS as e:
        logger.debug(f"Workflows scheduler: stop failed: {e}")
