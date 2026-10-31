"""Routing tests for JobManager batch lease renewal."""

from __future__ import annotations

import os
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Any

import pytest

import tldw_Server_API.app.core.Jobs.manager as manager_module
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.operations.contracts import BatchRenewLeasesResult

pytestmark = pytest.mark.unit

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=timezone.utc)


class FakeConnection:
    """Record connection closure without providing transaction internals."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FixedClock:
    """Supply a deterministic clock callable to backend operations."""

    def now_utc(self) -> datetime:
        return NOW


class ExplodingClock:
    """Record an unexpected clock read before backend dispatch."""

    def __init__(self) -> None:
        self.calls = 0

    def now_utc(self) -> datetime:
        self.calls += 1
        raise RuntimeError("clock must not run")


def _minimal_manager(backend: str) -> tuple[JobManager, FakeConnection]:
    manager = object.__new__(JobManager)
    connection = FakeConnection()
    manager.backend = backend
    manager._clock = FixedClock()
    manager._connect = lambda: connection
    manager._pg_cursor = lambda conn: nullcontext(conn)
    manager._should_enforce_ack = lambda: True
    return manager, connection


@pytest.mark.parametrize(
    ("backend", "backend_name"),
    [("sqlite", "_sqlite_renew_leases_batch"), ("postgres", "_postgres_renew_leases_batch")],
)
def test_batch_renew_routes_normalized_ordered_items_to_backend(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    backend_name: str,
) -> None:
    """A facade call dispatches one immutable normalized command to its backend."""

    manager, connection = _minimal_manager(backend)
    dispatched: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    lease_max_reads = 0
    enforcement_resolutions = 0
    real_getenv = os.getenv

    def backend_operation(*args: Any, **kwargs: Any) -> BatchRenewLeasesResult:
        dispatched.append((args, kwargs))
        return BatchRenewLeasesResult(requested_count=3, applied_count=2)

    def getenv(name: str, default: str | None = None) -> str | None:
        nonlocal lease_max_reads
        if name == "JOBS_LEASE_MAX_SECONDS":
            lease_max_reads += 1
            return "60"
        return real_getenv(name, default)

    def should_enforce_ack() -> bool:
        nonlocal enforcement_resolutions
        enforcement_resolutions += 1
        return True

    monkeypatch.setattr(manager_module, backend_name, backend_operation, raising=False)
    monkeypatch.setattr(manager_module.os, "getenv", getenv)
    monkeypatch.setattr(manager, "_should_enforce_ack", should_enforce_ack)

    applied_count = manager.batch_renew_leases(
        [
            {"job_id": 3, "seconds": 0, "worker_id": "worker-a", "lease_id": "lease-a"},
            {"job_id": 2, "seconds": 30, "worker_id": "worker-b", "lease_id": "lease-b"},
            {"job_id": 1, "seconds": 120, "worker_id": "worker-c", "lease_id": "lease-c"},
        ]
    )

    assert applied_count == 2
    assert len(dispatched) == 1
    args, kwargs = dispatched[0]
    expected_args = (connection, manager._pg_cursor) if backend == "postgres" else (connection,)
    assert args == expected_args
    command = kwargs["command"]
    assert tuple(item.job_id for item in command.items) == (3, 2, 1)
    assert tuple(item.seconds for item in command.items) == (1, 30, 60)
    assert tuple(item.worker_id for item in command.items) == ("worker-a", "worker-b", "worker-c")
    assert tuple(item.lease_id for item in command.items) == ("lease-a", "lease-b", "lease-c")
    assert command.enforce is True
    assert kwargs["clock"] == manager._clock.now_utc
    assert lease_max_reads == 3
    assert enforcement_resolutions == 1
    assert connection.closed is True


def test_batch_renew_opens_connection_before_normalizing_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connection failures win over malformed batch items."""

    manager, _ = _minimal_manager("sqlite")
    monkeypatch.setattr(
        manager,
        "_connect",
        lambda: (_ for _ in ()).throw(ConnectionError("offline")),
    )

    with pytest.raises(ConnectionError, match="offline"):
        manager.batch_renew_leases(
            [{"job_id": "invalid", "seconds": 30}],
            enforce=False,
        )


@pytest.mark.parametrize(
    ("backend", "backend_name"),
    [("sqlite", "_sqlite_renew_leases_batch"), ("postgres", "_postgres_renew_leases_batch")],
)
def test_batch_renew_normalizes_before_clock_cursor_or_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    backend_name: str,
) -> None:
    """A later malformed item wins before clock, cursor, or backend work."""

    manager, connection = _minimal_manager(backend)
    clock = ExplodingClock()
    manager._clock = clock
    called = False
    cursor_calls = 0

    def backend_operation(*args: Any, **kwargs: Any) -> BatchRenewLeasesResult:
        nonlocal called
        called = True
        raise AssertionError("backend must not run")

    def cursor(*args: Any, **kwargs: Any) -> Any:
        nonlocal cursor_calls
        cursor_calls += 1
        raise AssertionError("cursor must not open")

    manager._pg_cursor = cursor
    monkeypatch.setattr(manager_module, backend_name, backend_operation, raising=False)

    with pytest.raises(ValueError):
        manager.batch_renew_leases(
            [
                {"job_id": 1, "seconds": 30},
                {"job_id": "invalid", "seconds": 30},
            ],
            enforce=False,
        )

    assert called is False
    assert clock.calls == 0
    assert cursor_calls == 0
    assert connection.closed is True


def test_batch_renew_routes_empty_command_to_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty batch still reaches the backend transaction boundary."""

    manager, connection = _minimal_manager("sqlite")
    dispatched: list[dict[str, Any]] = []

    def backend(*args: Any, **kwargs: Any) -> BatchRenewLeasesResult:
        dispatched.append(kwargs)
        return BatchRenewLeasesResult(requested_count=0, applied_count=0)

    monkeypatch.setattr(manager_module, "_sqlite_renew_leases_batch", backend, raising=False)

    assert manager.batch_renew_leases([], enforce=False) == 0
    assert len(dispatched) == 1
    assert dispatched[0]["command"].items == ()
    assert dispatched[0]["command"].enforce is False
    assert connection.closed is True
