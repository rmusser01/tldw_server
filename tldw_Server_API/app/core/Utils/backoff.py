from __future__ import annotations

"""Compute the next retry delay.

One job: given an attempt number or the previous delay, say how long to wait.
Deliberately knows nothing about *what* is being retried.

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
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

__all__ = [
    "capped_exponential_delay",
    "decorrelated_jitter_delay",
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
