from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

TERMINAL_JOB_STATUSES: frozenset[str] = frozenset(
    {"completed", "failed", "cancelled", "quarantined"}
)


def is_terminal_job_status(value: Any) -> bool:
    """Return whether a value is a canonical terminal Jobs status."""
    return isinstance(value, str) and value in TERMINAL_JOB_STATUSES


@dataclass
class JobRecord:
    id: int
    uuid: str | None
    domain: str
    queue: str
    job_type: str
    owner_user_id: str | None
    project_id: int | None
    payload: dict[str, Any]
    result: dict[str, Any] | None
    status: str
    priority: int
    max_retries: int
    retry_count: int
    available_at: datetime | None
    leased_until: datetime | None
    lease_id: str | None
    worker_id: str | None
    acquired_at: datetime | None
    error_message: str | None
    error_code: str | None
    error_class: str | None
    error_stack: dict[str, Any] | None
    last_error: str | None
    cancel_requested_at: datetime | None
    cancelled_at: datetime | None
    cancellation_reason: str | None
    created_at: datetime | None
    updated_at: datetime | None
    completed_at: datetime | None
