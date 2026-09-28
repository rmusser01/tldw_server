"""The shared contention-retry policy (TASK-13319)."""

from __future__ import annotations

import sqlite3

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import (
    ConstraintViolationError,
    DatabaseError,
    TransientContentionError,
)
from tldw_Server_API.app.core.DB_Management.retry_policy import (
    is_retryable_contention,
    run_with_contention_retry,
)


def _wrapped(inner: BaseException) -> DatabaseError:
    """Prompt Studio's cursor wrapper re-raises backend errors `from` the original."""
    try:
        raise inner
    except BaseException as exc:
        try:
            raise DatabaseError("Backend query execution failed") from exc
        except DatabaseError as outer:
            return outer


@pytest.mark.parametrize(
    ("exc", "retryable"),
    [
        (TransientContentionError("x"), True),
        (_wrapped(TransientContentionError("x")), True),
        (sqlite3.OperationalError("Database is LOCKED"), True),
        (_wrapped(sqlite3.OperationalError("database is locked")), True),
        (sqlite3.OperationalError("no such table: t"), False),
        (ConstraintViolationError("x"), False),
        (DatabaseError("PostgreSQL query execution failed"), False),
    ],
)
def test_classification(exc, retryable):
    assert is_retryable_contention(exc) is retryable


def test_retries_contention_then_returns():
    calls, sleeps = [], []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise TransientContentionError("busy")
        return "ok"

    assert run_with_contention_retry(fn, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert len(sleeps) == 2


def test_gives_up_after_the_attempt_budget():
    calls = []

    def fn():
        calls.append(1)
        raise TransientContentionError("busy")

    with pytest.raises(TransientContentionError):
        run_with_contention_retry(fn, attempts=4, sleep=lambda _s: None)
    assert len(calls) == 4


def test_does_not_retry_other_failures():
    calls = []

    def fn():
        calls.append(1)
        raise ConstraintViolationError("dup")

    with pytest.raises(ConstraintViolationError):
        run_with_contention_retry(fn, sleep=lambda _s: None)
    assert calls == [1]
