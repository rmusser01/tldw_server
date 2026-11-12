"""Typed retry-index coordination failures preserve native database errors."""

from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core import exceptions
from tldw_Server_API.app.core.DB_Management import jobs_failed_requeue as retry_db

pytestmark = pytest.mark.unit


class IndexExecutor:
    """Supply catalog/lock observations without a database or native lifecycle."""

    def __init__(self, failure: str) -> None:
        """Select a failed admission phase and record the last command."""
        self.failure = failure
        self.sql = ""

    def execute(self, sql: str, _params: tuple[Any, ...] = ()) -> None:
        """Record commands and optionally raise the same native error instance."""
        self.sql = sql

    def fetchone(self) -> tuple[Any, ...] | None:
        """Return deterministic timeout, lock and index-state observations."""
        if "pg_settings" in self.sql:
            return (1,)
        if "pg_try_advisory_lock" in self.sql:
            return (self.failure != "timeout",)
        if "pg_class" in self.sql:
            return (False, True, True) if self.failure == "collision" else None
        return None


@pytest.mark.parametrize("failure,message", [
    ("timeout", "Jobs retry-admission index advisory lock timeout"),
    ("collision", "Jobs retry-admission index definition collision; refusing replacement"),
    ("verification", "Jobs retry-admission index verification failed"),
])
def test_retry_index_named_failure_is_central_runtime_error(
    failure: str, message: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All three safe admission failures share a centrally catchable subtype."""
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(retry_db, "monotonic", ticks.__next__)
    with pytest.raises(RuntimeError) as caught:
        retry_db.ensure_retry_admission_index(IndexExecutor(failure), backend="postgres")
    assert str(caught.value) == message
    assert type(caught.value) is getattr(exceptions, "JobsRetryAdmissionIndexError", None)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_retry_index_native_executor_error_propagates_unchanged(backend: str) -> None:
    """Native execute failures are not reclassified as coordination failures."""
    import sqlite3

    import psycopg

    error = sqlite3.OperationalError("native failure") if backend == "sqlite" else psycopg.errors.InsufficientPrivilege("native failure")

    class NativeFailure:
        """Raise an existing native error directly from the executor boundary."""

        def execute(self, _sql: str, _params: tuple[Any, ...] = ()) -> None:
            """Propagate the captured instance before index admission."""
            raise error

    with pytest.raises(type(error)) as caught:
        retry_db.ensure_retry_admission_index(NativeFailure(), backend=backend)
    assert caught.value is error
