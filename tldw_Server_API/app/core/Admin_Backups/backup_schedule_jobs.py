from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any


BACKUP_SCHEDULE_DOMAIN = "admin_backups"
BACKUP_SCHEDULE_JOB_TYPE = "scheduled_backup"


def backup_schedule_queue() -> str:
    """Return the queue name for scheduled backup jobs."""
    queue = (os.getenv("ADMIN_BACKUP_JOBS_QUEUE") or "default").strip()
    return queue or "default"


def normalize_backup_schedule_slot(value: str | datetime) -> str:
    """Normalize a fire slot value to an ISO8601 UTC timestamp."""
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def build_backup_schedule_run_slot_key(*, schedule_id: str, scheduled_for: str | datetime) -> str:
    """Build the unique DB claim key for one schedule fire slot."""
    return f"{schedule_id}:{normalize_backup_schedule_slot(scheduled_for)}"


def build_backup_schedule_idempotency_key(*, schedule_id: str, scheduled_for: str | datetime) -> str:
    """Build the Jobs idempotency key for one schedule fire slot."""
    return f"backup_schedule:{build_backup_schedule_run_slot_key(schedule_id=schedule_id, scheduled_for=scheduled_for)}"


def build_backup_schedule_job_payload(
    *,
    schedule_id: str,
    run_id: str,
    scheduled_for: str | datetime,
    dataset: str,
    target_user_id: int | None,
    retention_count: int,
) -> dict[str, Any]:
    """Build the queued Jobs payload for a scheduled backup run."""
    return {
        "schedule_id": str(schedule_id),
        "run_id": str(run_id),
        "scheduled_for": normalize_backup_schedule_slot(scheduled_for),
        "dataset": str(dataset),
        "target_user_id": int(target_user_id) if target_user_id is not None else None,
        "retention_count": int(retention_count),
    }


def parse_backup_schedule_job_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a scheduled backup Jobs payload."""
    schedule_id = str(payload.get("schedule_id") or "").strip()
    run_id = str(payload.get("run_id") or "").strip()
    dataset = str(payload.get("dataset") or "").strip().lower()
    if not schedule_id:
        raise ValueError("missing_schedule_id")
    if not run_id:
        raise ValueError("missing_run_id")
    if not dataset:
        raise ValueError("missing_dataset")
    scheduled_for = normalize_backup_schedule_slot(payload.get("scheduled_for"))
    target_user_id = payload.get("target_user_id")
    if target_user_id is not None:
        target_user_id = int(target_user_id)
    retention_count = int(payload.get("retention_count") or 0)
    return {
        "schedule_id": schedule_id,
        "run_id": run_id,
        "scheduled_for": scheduled_for,
        "dataset": dataset,
        "target_user_id": target_user_id,
        "retention_count": retention_count,
    }


__all__ = [
    "BACKUP_SCHEDULE_DOMAIN",
    "BACKUP_SCHEDULE_JOB_TYPE",
    "backup_schedule_queue",
    "build_backup_schedule_idempotency_key",
    "build_backup_schedule_job_payload",
    "build_backup_schedule_run_slot_key",
    "normalize_backup_schedule_slot",
    "parse_backup_schedule_job_payload",
]
