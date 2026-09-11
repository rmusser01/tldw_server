"""Jobs helpers for Recurring Question scheduled task execution."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import RunRow

SCHEDULED_TASKS_DOMAIN = "scheduled_tasks"
RECURRING_QUESTION_JOB_TYPE = "recurring_question_run"
RECURRING_QUESTION_QUEUE = "scheduled-tasks"


def build_manual_run_idempotency_payload(*, definition_id: str) -> dict[str, str]:
    """Return the idempotency payload for manual Recurring Question run creation."""
    return {
        "action": "create_manual_run",
        "definition_id": definition_id,
        "trigger_reason": "manual",
    }


def build_scheduled_run_idempotency_key(
    *,
    definition_id: str,
    definition_version: int,
    schedule_slot: str,
) -> str:
    """Return a deterministic Jobs idempotency key for one scheduled slot."""
    payload = {
        "definition_id": definition_id,
        "definition_version": definition_version,
        "schedule_slot": schedule_slot,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"scheduled-task-rq:{hashlib.sha256(encoded).hexdigest()}"


def build_recurring_question_run_job_payload(
    *,
    run: RunRow,
    owner_user_id: str,
) -> dict[str, Any]:
    """Return the opaque Jobs payload for a Recurring Question run."""
    return {
        "run_id": run.id,
        "definition_id": run.definition_id,
        "definition_version": run.definition_version,
        "owner_user_id": owner_user_id,
        "trigger_reason": run.trigger_reason,
        "schedule_slot": run.schedule_slot,
    }
