"""Validated private payloads and immediate event-loop-local admission leases."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from .models import JobPriority, JobStatus, QueueLimits, ScrapingJob


class QueueRejected(Exception):
    """Typed rejection before the queue retains any job, future or binding."""

    def __init__(self, code: Literal["queue_overloaded", "queue_stopping", "job_payload_too_large"]) -> None:
        self.code = code
        super().__init__(code)


class SnapshotTooLarge(ValueError):
    """The incremental JSON representation exhausted its byte allowance."""


PUBLIC_DIAGNOSTICS = frozenset(
    {
        "execution_failed",
        "source_extraction_failed",
        "deadline_exhausted",
        "budget_exhausted",
        "Job cancelled",
        "Job cancelled due to shutdown",
    }
)
DIAGNOSTIC_KEYS = frozenset(
    {
        "error",
        "errors",
        "errormessage",
        "errordetail",
        "diagnostic",
        "diagnostics",
        "exception",
        "exceptions",
        "exceptionmessage",
        "traceback",
        "message",
        "detail",
        "details",
        "reason",
    }
)


def public_diagnostic(value: Any) -> Any:
    """Keep known public codes, never arbitrary provider/transport diagnostics."""
    if value is None or value is False:
        return value
    if type(value) is str and (not value or value in PUBLIC_DIAGNOSTICS):
        return value
    return "execution_failed"


def bounded_snapshot(
    value: Any, limit: int, *, redact: bool = False, article_path: tuple[str, ...] | None = None
) -> tuple[Any, int]:
    """Copy strict JSON data while sizing UTF-8 incrementally, without coercion.

    JSON string escaping is chunked, so oversized strings never create a second
    unbounded serialized result. A depth/cycle guard rejects hostile containers.
    Only direct prose fields at the trusted article_path bypass URL redaction.
    """
    from .projections import PUBLIC_CONTENT_KEYS, private_state_key, public_state_string

    size = 0
    ancestors: set[int] = set()

    def charge(amount: int) -> None:
        nonlocal size
        size += amount
        if size > limit:
            raise SnapshotTooLarge("JSON snapshot exceeds byte allowance")

    def string(text: str) -> None:
        charge(2)
        for start in range(0, len(text), 256):
            try:
                charge(len(json.dumps(text[start : start + 256], ensure_ascii=False)[1:-1].encode("utf-8")))
            except UnicodeError:
                raise ValueError("invalid JSON string encoding") from None

    def visit(item: Any, depth: int, path: tuple[str | int, ...]) -> Any:
        if depth > 64:
            raise ValueError("JSON nesting exceeds allowance")
        kind = type(item)
        if kind is str:
            public_content = (
                article_path is not None
                and bool(path)
                and path[:-1] == article_path
                and path[-1] in PUBLIC_CONTENT_KEYS
            )
            if redact and not public_content:
                item = public_state_string(item)
            string(item)
            return item
        if item is None or kind is bool:
            charge(4 if item is None or item is True else 5)
            return item
        if kind in (int, float):
            if kind is float and not math.isfinite(item):
                raise ValueError("non-finite JSON number")
            # Huge integers cannot bypass the bounded encoder or Python's digit cap.
            if kind is int and item.bit_length() > 14000:
                raise ValueError("JSON integer exceeds allowance")
            charge(len(json.dumps(item)))
            return item
        if kind not in (dict, list, tuple):
            raise ValueError("unsupported JSON payload value")
        identity = id(item)
        if identity in ancestors:
            raise ValueError("cyclic JSON payload")
        ancestors.add(identity)
        try:
            charge(2)
            if kind is dict:
                result = {}
                for key, child in item.items():
                    if type(key) is not str:
                        raise ValueError("JSON object keys must be strings")
                    if redact and private_state_key(key):
                        continue
                    if redact and "".join(char for char in key.lower() if char.isalnum()) in DIAGNOSTIC_KEYS:
                        child = public_diagnostic(child)
                    if result:
                        charge(1)
                    string(key)
                    charge(1)
                    result[key] = visit(child, depth + 1, path + (key,))
                return result
            result_list = []
            for index, child in enumerate(item):
                if result_list:
                    charge(1)
                result_list.append(visit(child, depth + 1, path + (index,)))
            return result_list
        finally:
            ancestors.remove(identity)

    snapshot = visit(value, 0, ())
    return snapshot, size


@dataclass(frozen=True, repr=False)
class JobPayload:
    """Private normalized execution snapshot; never exposed in polling/history."""

    job_id: str
    url: str
    operation: str
    owner: str | None
    options: dict[str, Any] = field(repr=False)
    size_bytes: int

    @classmethod
    def from_job(cls, job: ScrapingJob, limit: int = 262144) -> JobPayload:
        """Validate before allocating retained queue state; discard sizing bytes."""
        if any(type(value) is not str or not value for value in (job.job_id, job.url, job.method)):
            raise ValueError("job identity, URL and method must be nonempty strings")
        if job.user_id is not None and type(job.user_id) is not str:
            raise ValueError("job owner must be a string or None")
        if not isinstance(job.priority, JobPriority) or job.status is not JobStatus.PENDING:
            raise ValueError("invalid job priority or initial status")
        if type(job.retries) is not int or type(job.max_retries) is not int or not 0 <= job.retries <= job.max_retries:
            raise ValueError("invalid retry counts")
        if type(job.metadata) is not dict:
            raise ValueError("execution options must be a JSON object")
        if not isinstance(job.created_at, datetime) or any(
            value is not None and not isinstance(value, datetime) for value in (job.started_at, job.completed_at)
        ):
            raise ValueError("job timestamps must be datetime values")
        if type(job.cancel_requested) is not bool:
            raise ValueError("cancel_requested must be a boolean")
        normalized, size = bounded_snapshot(
            {
                "job_id": job.job_id,
                "url": job.url,
                "operation": job.method,
                "owner": job.user_id,
                "options": job.metadata,
            },
            limit,
        )
        return cls(**normalized, size_bytes=size)


@dataclass
class Admission:
    """No waits: these counters are mutated only on the queue's event loop."""

    limits: QueueLimits
    live: int = 0
    pending: int = 0
    payload_bytes: int = 0

    def acquire(self, size: int) -> AdmissionLease:
        """Charge all aggregate dimensions atomically or change nothing."""
        if (
            self.live >= self.limits.max_live
            or self.pending >= self.limits.max_pending
            or self.payload_bytes + size > self.limits.max_total_payload_bytes
        ):
            raise QueueRejected("queue_overloaded")
        self.live += 1
        self.pending += 1
        self.payload_bytes += size
        return AdmissionLease(self, size)


@dataclass
class AdmissionLease:
    """An accepted job's idempotent capacity release, independent of result state."""

    owner: Admission
    size: int
    pending: bool = True
    released: bool = False

    def release_pending(self) -> None:
        if self.pending:
            self.pending = False
            self.owner.pending -= 1

    def release(self) -> None:
        if not self.released:
            self.released = True
            self.release_pending()
            self.owner.live -= 1
            self.owner.payload_bytes -= self.size
