from __future__ import annotations

"""Compute the next retry delay.

Given an attempt number or the previous delay, say how long to wait; and, for
outbound HTTP, whether an attempt is worth retrying at all
(:func:`classify_http_retry`, :func:`is_dns_resolution_error`). DB contention
classification lives in core/DB_Management/retry_policy.py (ADR-047 follow-up).

Two schedules, because the codebase retries two different things and one algorithm
does not serve both:

``decorrelated_jitter_delay`` -- for OUTBOUND HTTP. ``min(cap, uniform(base, prev*3))``
is the decorrelated-jitter form, chosen because it de-synchronises a fleet of clients
retrying the same failing endpoint. The symmetric +/-25% jitter it replaces kept
clients clustered in a narrow band, which is the failure mode jitter exists to prevent.
Pair it with :func:`parse_retry_after_seconds`, which honours both delta-seconds and
HTTP-date forms.

``capped_exponential_delay`` -- for IN-PROCESS CONTENTION, principally SQLite
``database is locked``. Short, bounded, and multiplicatively jittered. Decorrelated
jitter is wrong here: ``prev*3`` grows faster than capped exponential, so a lock retry
would sleep considerably longer than the 0.05-1.2s these call sites use today, for a
contention window that is typically milliseconds and entirely local.

Both were previously inlined: decorrelated jitter only inside core/http_client.py
(6,600 LOC), and the contention schedule copy-pasted 28 times in one file plus two
module-level variants. Those copies had drifted three ways -- one tested
``"database is locked"`` without ``.lower()``, one hardcoded ``attempt < 4`` inside a
``range(5)`` loop, and 5 of 28 omitted the jitter term the other 23 had.
"""

import random
import socket
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

__all__ = [
    "capped_exponential_delay",
    "classify_http_retry",
    "decorrelated_jitter_delay",
    "is_dns_resolution_error",
    "is_sqlite_locked_error",
    "parse_retry_after_seconds",
]

_NONCRITICAL = (AttributeError, LookupError, TypeError, ValueError, OSError)


def decorrelated_jitter_delay(prev: float, base_ms: int, cap_s: int) -> float:
    """Return the next delay for an outbound HTTP retry, in seconds.

    ``prev`` is the previous delay (0 on the first retry). Growth is ``prev*3``
    bounded by ``cap_s``, with the lower bound at ``base_ms``.
    """
    base = max(0.001, base_ms / 1000.0)
    cap = max(base, float(cap_s))
    return base if prev <= 0 else min(cap, random.uniform(base, prev * 3))  # nosec B311


def capped_exponential_delay(
    attempt: int,
    *,
    base_s: float = 0.05,
    cap_s: float = 2.0,
    jitter: bool = True,
) -> float:
    """Return the next delay for in-process contention, in seconds.

    ``attempt`` is zero-based. The default ``base_s`` and the multiplicative
    ``0.5 + random()`` jitter reproduce the schedule the SQLite retry loops already
    used; ``cap_s`` bounds it if a caller raises its attempt count.
    """
    delay = min(float(cap_s), float(base_s) * (2 ** max(0, int(attempt))))
    if jitter:
        delay *= 0.5 + random.random()  # nosec B311
    return delay


def is_sqlite_locked_error(exc: BaseException) -> bool:
    """Whether ``exc`` is SQLite reporting lock contention.

    Case-insensitive: one of the 28 inlined copies compared without ``.lower()`` and
    so missed a differently-cased message.
    """
    return "database is locked" in str(exc).lower()


def parse_retry_after_seconds(
    retry_after: str | None,
    *,
    now: datetime | None = None,
) -> float | None:
    """Parse a ``Retry-After`` header as delta-seconds or HTTP-date, in seconds."""
    if not retry_after:
        return None
    try:
        return max(0.0, float(retry_after.strip()))
    except _NONCRITICAL:
        pass
    try:
        parsed = parsedate_to_datetime(retry_after.strip())
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        current = now or datetime.now(timezone.utc)
        return max(0.0, (parsed - current).total_seconds())
    except _NONCRITICAL:
        return None


# The builtin members of http_client's _HTTPCLIENT_NONCRITICAL_EXCEPTIONS, which is what
# these classifiers caught before they moved (its custom members cannot be raised by
# getattr/str on an exception and would be a circular import here).
_CLASSIFY_NONCRITICAL = (AttributeError, OSError, RuntimeError, TypeError, ValueError)

_DNS_FAILURE_MARKERS = (
    "nodename nor servname provided",
    "Name or service not known",
    "Temporary failure in name resolution",
    "Host could not be resolved",
    "DNSResolutionError",
)


def is_dns_resolution_error(exc: BaseException) -> bool:
    """Best-effort detection of DNS resolution / unknown-host failures.

    Looks for ``socket.gaierror`` in the exception chain, for common
    platform-specific substrings in the message, and for the explicit
    ``_tldw_dns_resolution`` flag / "DNSResolutionError" sentinel http_client sets.
    """
    try:
        if getattr(exc, "_tldw_dns_resolution", False):
            return True
    except _CLASSIFY_NONCRITICAL:
        pass
    try:
        seen_ids: set[int] = set()
        cur: BaseException | None = exc
        while cur is not None and id(cur) not in seen_ids:
            seen_ids.add(id(cur))
            if isinstance(cur, socket.gaierror):
                return True
            msg = str(cur)
            if any(m in msg for m in _DNS_FAILURE_MARKERS):
                return True
            next_exc = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
            if not isinstance(next_exc, BaseException):
                break
            cur = next_exc
    except _CLASSIFY_NONCRITICAL:
        return False
    return False


def classify_http_retry(
    method: str,
    status: int | None,
    exc: BaseException | None,
    policy: Any,
) -> tuple[bool, str]:
    """Whether an outbound HTTP attempt should be retried, and a short reason.

    ``policy`` supplies ``retry_on_methods``, ``retry_on_status`` and
    ``retry_on_unsafe`` (http_client.RetryPolicy). Network exceptions are retried
    for retriable methods, except DNS failures, which are treated as permanent.
    """
    m = method.upper()
    if exc is not None:
        if m not in policy.retry_on_methods and not policy.retry_on_unsafe:
            return False, "method_not_retriable"
        try:
            if is_dns_resolution_error(exc):
                return False, exc.__class__.__name__
        except _CLASSIFY_NONCRITICAL:
            pass
        return True, exc.__class__.__name__
    if status is None:
        return False, "no_status"
    if status in policy.retry_on_status and (m in policy.retry_on_methods or policy.retry_on_unsafe):
        return True, f"{status}"
    return False, "status_not_retriable"
