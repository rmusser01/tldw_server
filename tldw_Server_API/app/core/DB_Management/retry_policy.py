"""One contention-retry policy for both database backends (TASK-13319, ADR-047).

SQLite reports contention as "database is locked"; PostgreSQL as SQLSTATE 40001/40P01/
55P03, which the backends surface as ``TransientContentionError``. Either way the whole
transaction must be re-run, so ``fn`` passed to ``run_with_contention_retry`` must own its
transaction: a PostgreSQL transaction is already aborted once the error is raised.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Callable, TypeVar

from tldw_Server_API.app.core.DB_Management.backends.base import TransientContentionError
from tldw_Server_API.app.core.Utils.backoff import capped_exponential_delay, is_sqlite_locked_error

T = TypeVar("T")

DEFAULT_ATTEMPTS = 5


def is_retryable_contention(exc: BaseException) -> bool:
    """Whether ``exc``, or anything it was raised from, is transient contention."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, TransientContentionError):
            return True
        if isinstance(current, sqlite3.OperationalError) and is_sqlite_locked_error(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def run_with_contention_retry(
    fn: Callable[[], T],
    *,
    attempts: int = DEFAULT_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn``, re-running it after transient contention up to ``attempts`` times."""
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if attempt == attempts - 1 or not is_retryable_contention(exc):
                raise
            sleep(capped_exponential_delay(attempt))
    raise AssertionError("unreachable")  # pragma: no cover
