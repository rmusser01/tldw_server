"""Control-plane service for unified scheduled-task management."""

from __future__ import annotations

import json
from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.schemas.reminders_schemas import (
    ReminderTaskCreateRequest,
    ReminderTaskUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_control_plane_schemas import (
    ScheduledTask,
    ScheduledTaskDeleteResponse,
    ScheduledTaskListResponse,
)
from tldw_Server_API.app.core.Personalization.companion_activity import (
    record_reminder_task_created,
    record_reminder_task_deleted,
    record_reminder_task_updated,
)
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase, ReminderTaskRow
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import DefinitionRow, ScheduledTasksDatabase
from tldw_Server_API.app.core.DB_Management.Watchlists_DB import JobRow, WatchlistsDatabase
from tldw_Server_API.app.services.reminders_scheduler import get_reminders_scheduler

_NONCRITICAL_SCHEDULER_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)
_NONCRITICAL_READ_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)

_REMINDER_PREFIX = "reminder_task"
_WATCHLIST_PREFIX = "watchlist_job"
_AUTOMATION_DEFINITION_PREFIX = "automation_definition"


def _load_json_object(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _reminder_schedule_summary(row: ReminderTaskRow) -> str | None:
    if row.schedule_kind == "one_time":
        return row.run_at
    if row.cron:
        if row.timezone:
            return f"{row.cron} ({row.timezone})"
        return row.cron
    return None


def _watchlist_schedule_summary(row: JobRow) -> str | None:
    if row.schedule_expr:
        if row.schedule_timezone:
            return f"{row.schedule_expr} ({row.schedule_timezone})"
        return row.schedule_expr
    return "Manual"


def _automation_schedule_summary(row: DefinitionRow) -> str | None:
    kind = row.schedule.get("kind")
    if not isinstance(kind, str) or not kind:
        return None
    timezone = row.schedule.get("timezone")
    if kind == "daily":
        time = row.schedule.get("time")
        summary = f"Daily at {time}" if isinstance(time, str) and time else "Daily"
    elif kind == "weekly":
        day = row.schedule.get("day")
        time = row.schedule.get("time")
        parts = ["Weekly"]
        if isinstance(day, str) and day:
            parts.append(day)
        if isinstance(time, str) and time:
            parts.append(f"at {time}")
        summary = " ".join(parts)
    elif kind == "cron":
        expr = row.schedule.get("cron") or row.schedule.get("expr")
        summary = str(expr) if expr else "Cron"
    elif kind == "one_time":
        run_at = row.schedule.get("run_at")
        summary = str(run_at) if run_at else "One time"
    elif kind == "interval":
        every = row.schedule.get("every")
        unit = row.schedule.get("unit")
        summary = f"Every {every} {unit}" if every and unit else "Interval"
    else:
        summary = kind
    if isinstance(timezone, str) and timezone:
        return f"{summary} ({timezone})"
    return summary


def _automation_definition_status(row: DefinitionRow) -> tuple[bool, str]:
    if row.lifecycle == "configured":
        if row.health == "execution_unavailable":
            return True, "configured_execution_unavailable"
        if row.health == "capability_unavailable":
            return True, "blocked_capability_unavailable"
        if row.health == "permission_required":
            return True, "blocked_permission_required"
        return True, row.health
    if row.lifecycle == "paused":
        return False, "paused"
    if row.lifecycle == "archived":
        return False, "archived"
    if row.lifecycle == "disabled":
        return False, "disabled"
    return False, "needs_attention"


def _reminder_row_to_activity_payload(row: ReminderTaskRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "title": row.title,
        "body": row.body,
        "schedule_kind": row.schedule_kind,
        "enabled": row.enabled,
        "link_type": row.link_type,
        "link_id": row.link_id,
        "link_url": row.link_url,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


class ScheduledTasksControlPlaneService:
    """Builds the unified scheduled-task read model and reminder-native mutations."""

    @staticmethod
    def _collections_db(user_id: int) -> CollectionsDatabase:
        return CollectionsDatabase.for_user(user_id=user_id)

    @staticmethod
    def _watchlists_db(user_id: int) -> WatchlistsDatabase:
        db = WatchlistsDatabase.for_user(user_id=user_id)
        with suppress(Exception):
            db.ensure_schema()
        return db

    @staticmethod
    def _scheduled_tasks_db(user_id: int) -> ScheduledTasksDatabase:
        db = ScheduledTasksDatabase.for_user(user_id=user_id)
        db.ensure_schema()
        return db

    @staticmethod
    def _normalize_reminder(row: ReminderTaskRow) -> ScheduledTask:
        status = row.last_status or ("scheduled" if row.enabled else "disabled")
        return ScheduledTask(
            id=f"{_REMINDER_PREFIX}:{row.id}",
            primitive="reminder_task",
            title=row.title,
            description=row.body,
            status=status,
            enabled=bool(row.enabled),
            schedule_summary=_reminder_schedule_summary(row),
            timezone=row.timezone,
            next_run_at=row.next_run_at,
            last_run_at=row.last_run_at,
            edit_mode="native",
            manage_url=None,
            source_ref={
                "task_id": row.id,
                "schedule_kind": row.schedule_kind,
                "run_at": row.run_at,
                "cron": row.cron,
                "timezone": row.timezone,
                "link_type": row.link_type,
                "link_id": row.link_id,
                "link_url": row.link_url,
            },
        )

    @staticmethod
    def _normalize_watchlist_job(row: JobRow) -> ScheduledTask:
        status = "scheduled" if row.active else "disabled"
        return ScheduledTask(
            id=f"{_WATCHLIST_PREFIX}:{row.id}",
            primitive="watchlist_job",
            title=row.name,
            description=row.description,
            status=status,
            enabled=bool(row.active),
            schedule_summary=_watchlist_schedule_summary(row),
            timezone=row.schedule_timezone,
            next_run_at=row.next_run_at,
            last_run_at=row.last_run_at,
            edit_mode="external",
            manage_url="/watchlists?tab=jobs",
            source_ref={
                "job_id": row.id,
                "scope": _load_json_object(row.scope_json),
                "schedule_expr": row.schedule_expr,
                "timezone": row.schedule_timezone,
            },
        )

    @staticmethod
    def _normalize_automation_definition(row: DefinitionRow) -> ScheduledTask:
        enabled, status = _automation_definition_status(row)
        timezone = row.schedule.get("timezone")
        return ScheduledTask(
            id=f"{_AUTOMATION_DEFINITION_PREFIX}:{row.id}",
            primitive="automation_definition",
            title=row.name,
            description=row.description,
            status=status,
            enabled=enabled,
            schedule_summary=_automation_schedule_summary(row),
            timezone=timezone if isinstance(timezone, str) else None,
            next_run_at=None,
            last_run_at=None,
            edit_mode="native",
            manage_url=None,
            source_ref={
                "definition_id": row.id,
                "family": row.family,
                "lifecycle": row.lifecycle,
                "health": row.health,
                "version": row.version,
                "visibility_policy": row.visibility_policy,
                "execution_available": False,
                "disabled_lock_kind": row.disabled_lock_kind,
                "disabled_reason": row.disabled_reason,
            },
        )

    async def _reconcile_reminder_task_best_effort(self, *, task_id: str, user_id: int) -> None:
        try:
            await get_reminders_scheduler().reconcile_task(task_id=task_id, user_id=user_id)
        except _NONCRITICAL_SCHEDULER_EXCEPTIONS:
            return

    async def _unschedule_reminder_task_best_effort(self, *, task_id: str) -> None:
        try:
            await get_reminders_scheduler().unschedule_task(task_id=task_id)
        except _NONCRITICAL_SCHEDULER_EXCEPTIONS:
            return

    async def list_tasks(self, *, user_id: int) -> ScheduledTaskListResponse:
        items: list[ScheduledTask] = []
        errors: list[str] = []

        try:
            reminder_rows = self._collections_db(user_id).list_reminder_tasks()
            items.extend(self._normalize_reminder(row) for row in reminder_rows)
        except _NONCRITICAL_READ_EXCEPTIONS as exc:
            logger.warning(
                "scheduled tasks control plane could not list reminder tasks user_id={} error={}",
                user_id,
                exc,
            )
            errors.append("reminder_tasks_unavailable")

        try:
            watchlist_rows, _ = self._watchlists_db(user_id).list_jobs(q=None, limit=500, offset=0)
            items.extend(self._normalize_watchlist_job(row) for row in watchlist_rows)
        except _NONCRITICAL_READ_EXCEPTIONS as exc:
            logger.warning(
                "scheduled tasks control plane could not list watchlist jobs user_id={} error={}",
                user_id,
                exc,
            )
            errors.append("watchlist_jobs_unavailable")

        try:
            definition_rows, _ = self._scheduled_tasks_db(user_id).list_definitions(
                owner_id=user_id,
                limit=500,
                offset=0,
            )
            items.extend(self._normalize_automation_definition(row) for row in definition_rows)
        except _NONCRITICAL_READ_EXCEPTIONS as exc:
            logger.warning(
                "scheduled tasks control plane could not list automation definitions user_id={} error={}",
                user_id,
                exc,
            )
            errors.append("automation_definitions_unavailable")

        return ScheduledTaskListResponse(
            items=items,
            total=len(items),
            partial=bool(errors),
            errors=errors,
        )

    async def get_task(self, *, user_id: int, task_id: str) -> ScheduledTask:
        if task_id.startswith(f"{_REMINDER_PREFIX}:"):
            raw_task_id = task_id.removeprefix(f"{_REMINDER_PREFIX}:")
            row = self._collections_db(user_id).get_reminder_task(raw_task_id)
            return self._normalize_reminder(row)

        if task_id.startswith(f"{_WATCHLIST_PREFIX}:"):
            raw_task_id = task_id.removeprefix(f"{_WATCHLIST_PREFIX}:")
            row = self._watchlists_db(user_id).get_job(int(raw_task_id))
            return self._normalize_watchlist_job(row)

        if task_id.startswith(f"{_AUTOMATION_DEFINITION_PREFIX}:"):
            raw_task_id = task_id.removeprefix(f"{_AUTOMATION_DEFINITION_PREFIX}:")
            row = self._scheduled_tasks_db(user_id).get_definition(owner_id=user_id, definition_id=raw_task_id)
            if row is None:
                raise KeyError("scheduled_task_not_found")
            return self._normalize_automation_definition(row)

        raise KeyError("scheduled_task_not_found")

    async def create_reminder(self, *, user_id: int, payload: ReminderTaskCreateRequest) -> ScheduledTask:
        reminders_db = self._collections_db(user_id)
        task_id = reminders_db.create_reminder_task(
            title=payload.title,
            body=payload.body,
            schedule_kind=payload.schedule_kind,
            run_at=payload.run_at,
            cron=payload.cron,
            timezone=payload.timezone,
            enabled=payload.enabled,
            link_type=payload.link_type,
            link_id=payload.link_id,
            link_url=payload.link_url,
        )
        await self._reconcile_reminder_task_best_effort(task_id=task_id, user_id=user_id)
        row = reminders_db.get_reminder_task(task_id)
        record_reminder_task_created(user_id=user_id, task=_reminder_row_to_activity_payload(row))
        return self._normalize_reminder(row)

    async def update_reminder(
        self,
        *,
        user_id: int,
        task_id: str,
        payload: ReminderTaskUpdateRequest,
    ) -> ScheduledTask:
        reminders_db = self._collections_db(user_id)
        patch = payload.model_dump(exclude_unset=True)
        row = reminders_db.update_reminder_task(task_id, patch)
        await self._reconcile_reminder_task_best_effort(task_id=task_id, user_id=user_id)
        if patch:
            record_reminder_task_updated(
                user_id=user_id,
                task=_reminder_row_to_activity_payload(row),
                patch=patch,
            )
        return self._normalize_reminder(row)

    async def delete_reminder(self, *, user_id: int, task_id: str) -> ScheduledTaskDeleteResponse:
        reminders_db = self._collections_db(user_id)
        existing_row = None
        with suppress(KeyError):
            existing_row = reminders_db.get_reminder_task(task_id)
        deleted = reminders_db.delete_reminder_task(task_id)
        if deleted:
            await self._unschedule_reminder_task_best_effort(task_id=task_id)
            if existing_row is not None:
                record_reminder_task_deleted(
                    user_id=user_id,
                    task=_reminder_row_to_activity_payload(existing_row),
                )
        return ScheduledTaskDeleteResponse(deleted=deleted)
