"""Legacy-compatible job values and trusted internal queue safety limits."""

from __future__ import annotations

import contextvars
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import Enum, auto
from typing import Any

from ..runtime.controls import positive_integer


class JobStatus(Enum):
    """Job status enumeration; values match the legacy export."""

    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class JobPriority(Enum):
    """Job priority levels, highest value first."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScrapingJob:
    """Public job constructor retained until the Task 9 compatibility re-export."""

    job_id: str
    url: str
    method: str
    user_id: str | None = None
    priority: JobPriority = JobPriority.NORMAL
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    retries: int = 0
    max_retries: int = 3
    result: dict[str, Any] | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cancel_requested: bool = False
    context: contextvars.Context | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the unchanged public shape, not private execution bindings."""
        return {
            "job_id": self.job_id,
            "url": self.url,
            "method": self.method,
            "user_id": self.user_id,
            "priority": self.priority.value,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retries": self.retries,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
            "cancel_requested": self.cancel_requested,
        }


@dataclass(frozen=True)
class QueueLimits:
    """Composition-time bounds shared by all callers and priorities."""

    max_pending: int = 128
    max_live: int = 256
    max_payload_bytes: int = 262144
    max_total_payload_bytes: int = 16777216
    max_history_bytes: int = 67108864

    def __post_init__(self) -> None:
        for item in fields(self):
            positive_integer(getattr(self, item.name), item.name)


@dataclass(frozen=True)
class QueueStopReport:
    """Logical shutdown is not a claim that synchronous execution was killed."""

    state: str
    draining_jobs: int
