from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase

REMINDERS_DOMAIN = "notifications"
REMINDER_JOB_TYPE = "reminder_due"


def _payload_as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    return {}


def _normalize_iso_utc(value: str | None) -> str:
    if not value:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_user_id(job: dict[str, Any], payload: dict[str, Any]) -> int:
    owner = job.get("owner_user_id") or payload.get("user_id")
    if owner is None or str(owner).strip() == "":
        raise ValueError("missing owner_user_id")
    return int(owner)


def _task_run_completion_patch(task: Any, *, now_iso: str, last_status: str) -> dict[str, Any]:
    """Build the task state patch shared by completed and skipped runs."""
    patch: dict[str, Any] = {
        "last_run_at": now_iso,
        "last_status": last_status,
    }
    if task.schedule_kind == "one_time":
        patch["enabled"] = False
        patch["next_run_at"] = None
    return patch


async def handle_reminder_job(
    job: dict[str, Any],
    *,
    collections_db: CollectionsDatabase | None = None,
) -> dict[str, Any]:
    payload = _payload_as_dict(job.get("payload"))
    task_id = str(payload.get("task_id") or "").strip()
    if not task_id:
        raise ValueError("missing task_id")
    user_id = _resolve_user_id(job, payload)
    run_slot_utc = _normalize_iso_utc(payload.get("scheduled_for"))
    run_slot_key = run_slot_utc
    now_iso = datetime.now(timezone.utc).isoformat()

    cdb = collections_db or CollectionsDatabase.for_user(user_id=user_id)
    run = cdb.create_reminder_task_run(
        task_id=task_id,
        scheduled_for=payload.get("scheduled_for"),
        job_id=str(job.get("id")) if job.get("id") is not None else None,
        run_slot_utc=run_slot_utc,
        run_slot_key=run_slot_key,
        status="running",
        error=None,
        started_at=now_iso,
    )

    if run.status == "succeeded":
        return {"status": "succeeded", "task_id": task_id, "run_id": run.id, "deduped": True}
    if run.status == "skipped":
        return {"status": "skipped", "task_id": task_id, "run_id": run.id, "deduped": True}

    try:
        task = cdb.get_reminder_task(task_id)
    except KeyError:
        completed_at = datetime.now(timezone.utc).isoformat()
        cdb.update_reminder_task_run_status(
            run_id=run.id,
            status="skipped",
            error="task_missing",
            completed_at=completed_at,
        )
        logger.info("Reminder job skipped for missing task: task_id={} run_id={}", task_id, run.id)
        return {
            "status": "skipped",
            "reason": "task_missing",
            "task_id": task_id,
            "run_id": run.id,
        }

    if not task.enabled:
        completed_at = datetime.now(timezone.utc).isoformat()
        cdb.update_reminder_task_run_status(
            run_id=run.id,
            status="skipped",
            error="task_disabled",
            completed_at=completed_at,
        )
        cdb.update_reminder_task(
            task_id,
            _task_run_completion_patch(task, now_iso=now_iso, last_status="skipped"),
        )
        logger.info("Reminder job skipped for disabled task: task_id={} run_id={}", task_id, run.id)
        return {
            "status": "skipped",
            "reason": "task_disabled",
            "task_id": task_id,
            "run_id": run.id,
        }

    prefs = cdb.get_notification_preferences()
    if not prefs.reminder_enabled:
        completed_at = datetime.now(timezone.utc).isoformat()
        cdb.update_reminder_task_run_status(
            run_id=run.id,
            status="skipped",
            error="reminder_notifications_disabled",
            completed_at=completed_at,
        )
        cdb.update_reminder_task(
            task_id,
            _task_run_completion_patch(task, now_iso=now_iso, last_status="skipped"),
        )
        logger.info("Reminder job skipped by notification preferences: task_id={} run_id={}", task_id, run.id)
        return {
            "status": "skipped",
            "reason": "reminder_notifications_disabled",
            "task_id": task_id,
            "run_id": run.id,
        }

    notification = cdb.create_user_notification(
        kind="reminder_due",
        title=task.title,
        message=task.body or "",
        severity="info",
        source_task_id=task.id,
        source_task_run_id=run.id,
        source_job_id=str(job.get("id")) if job.get("id") is not None else None,
        source_domain=REMINDERS_DOMAIN,
        source_job_type=REMINDER_JOB_TYPE,
        link_type=task.link_type,
        link_id=task.link_id,
        link_url=task.link_url,
        dedupe_key=f"task:{task_id}:{run_slot_utc}",
    )

    cdb.update_reminder_task_run_status(
        run_id=run.id,
        status="succeeded",
        error=None,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    cdb.update_reminder_task(
        task_id,
        _task_run_completion_patch(task, now_iso=now_iso, last_status="succeeded"),
    )
    logger.info("Reminder job handled successfully: task_id={} run_id={}", task_id, run.id)
    return {
        "status": "succeeded",
        "task_id": task_id,
        "run_id": run.id,
        "notification_id": notification.id,
    }


__all__ = [
    "REMINDERS_DOMAIN",
    "REMINDER_JOB_TYPE",
    "handle_reminder_job",
]
