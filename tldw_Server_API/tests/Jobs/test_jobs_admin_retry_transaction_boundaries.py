from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from tldw_Server_API.app.api.v1.endpoints import jobs_admin
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.Jobs.manager import JobManager

pytestmark = pytest.mark.unit

_BACKENDS = [
    pytest.param("sqlite", id="sqlite"),
    pytest.param("postgres", marks=pytest.mark.pg_jobs, id="postgres"),
]
_ORIGINAL_CONNECT = JobManager._connect
_RETRY_RESET_FIELDS = (
    "result",
    "completed_at",
    "started_at",
    "acquired_at",
    "last_error",
    "error_message",
    "error_code",
    "error_class",
    "error_stack",
    "completion_token",
)


class _CursorProxy:
    """Delegate a DB-API cursor while exposing each statement to a hook."""

    def __init__(self, inner: Any, before_execute: Callable[[Any], None]) -> None:
        self._inner = inner
        self._before_execute = before_execute

    def __enter__(self) -> _CursorProxy:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = None) -> Any:
        self._before_execute(sql)
        if params is None:
            return self._inner.execute(sql)
        return self._inner.execute(sql, params)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ConnectionProxy:
    """Delegate a connection with statement and commit failure hooks."""

    def __init__(
        self,
        inner: Any,
        *,
        before_execute: Callable[[Any], None] | None = None,
        fail_commit: bool = False,
    ) -> None:
        self._inner = inner
        self._before_execute = before_execute or (lambda _sql: None)
        self._fail_commit = fail_commit

    def __enter__(self) -> _ConnectionProxy:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
        if exc_type is None and self._fail_commit:
            self._inner.rollback()
            raise RuntimeError("forced admin commit failure")
        return self._inner.__exit__(exc_type, exc, tb)

    def execute(self, sql: Any, params: Any = ()) -> Any:
        self._before_execute(sql)
        return self._inner.execute(sql, params)

    def cursor(self, *args: Any, **kwargs: Any) -> _CursorProxy:
        return _CursorProxy(
            self._inner.cursor(*args, **kwargs),
            self._before_execute,
        )

    def commit(self) -> None:
        if self._fail_commit:
            self._inner.rollback()
            raise RuntimeError("forced admin commit failure")
        self._inner.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _normalize_sql(sql: Any) -> str:
    return " ".join(str(sql).lower().split())


def _admin_principal() -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        subject="jobs-admin",
        roles=["admin"],
        is_admin=True,
    )


def _confirmed_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/jobs/batch",
            "headers": [(b"x-confirm", b"true")],
        }
    )


def _manager(
    backend: str,
    *,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> JobManager:
    monkeypatch.setenv("JOBS_COUNTERS_ENABLED", "true")
    monkeypatch.setenv("JOBS_GAUGES_DEBOUNCE_MS", "0")
    monkeypatch.delenv("JOBS_DOMAIN_SCOPED_RBAC", raising=False)
    monkeypatch.delenv("JOBS_DOMAIN_RBAC_PRINCIPAL", raising=False)
    if backend == "postgres":
        dsn = str(request.getfixturevalue("jobs_pg_dsn"))
        monkeypatch.setenv("JOBS_DB_URL", dsn)
        monkeypatch.setenv("JOBS_PG_SKIP_SCHEMA_INIT", "true")
        return JobManager(backend="postgres", db_url=dsn)

    monkeypatch.delenv("JOBS_DB_URL", raising=False)
    db_path = tmp_path / f"{name}.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    return JobManager(db_path)


def _bind_endpoint_manager(
    monkeypatch: pytest.MonkeyPatch,
    manager: JobManager,
) -> None:
    monkeypatch.setattr(
        jobs_admin,
        "JobManager",
        lambda backend=None, db_url=None: manager,
    )


def _raw_execute(
    manager: JobManager,
    sqlite_sql: str,
    params: tuple[Any, ...],
    *,
    postgres_sql: str | None = None,
) -> None:
    conn = _ORIGINAL_CONNECT(manager)
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(postgres_sql or sqlite_sql, params)
        else:
            conn.execute(sqlite_sql, params)
        conn.commit()
    finally:
        conn.close()


def _raw_fetchall(
    manager: JobManager,
    sqlite_sql: str,
    params: tuple[Any, ...],
    *,
    postgres_sql: str | None = None,
) -> list[Any]:
    conn = _ORIGINAL_CONNECT(manager)
    try:
        if manager.backend == "postgres":
            with manager._pg_cursor(conn) as cur:
                cur.execute(postgres_sql or sqlite_sql, params)
                return list(cur.fetchall() or [])
        return list(conn.execute(sqlite_sql, params).fetchall() or [])
    finally:
        conn.close()


def _value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return row[index]


def _counter(
    manager: JobManager,
    *,
    domain: str,
    job_type: str = "work",
) -> tuple[int, int, int, int] | None:
    rows = _raw_fetchall(
        manager,
        (
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
            "FROM job_counters WHERE domain=? AND queue=? AND job_type=?"
        ),
        (domain, "default", job_type),
        postgres_sql=(
            "SELECT ready_count, scheduled_count, processing_count, quarantined_count "
            "FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s"
        ),
    )
    if not rows:
        return None
    row = rows[0]
    return tuple(
        int(_value(row, key, index) or 0)
        for index, key in enumerate(
            (
                "ready_count",
                "scheduled_count",
                "processing_count",
                "quarantined_count",
            )
        )
    )


def _job_snapshots(manager: JobManager, job_ids: list[int]) -> list[tuple[str, bool, Any]]:
    if manager.backend == "postgres":
        rows = _raw_fetchall(
            manager,
            "",
            (job_ids,),
            postgres_sql=(
                "SELECT status, available_at, completion_token FROM jobs "
                "WHERE id = ANY(%s) ORDER BY id"
            ),
        )
    else:
        placeholders = ",".join("?" for _ in job_ids)
        rows = _raw_fetchall(
            manager,
            (
                "SELECT status, available_at, completion_token FROM jobs "
                f"WHERE id IN ({placeholders}) ORDER BY id"  # nosec B608
            ),
            tuple(job_ids),
        )
    return [
        (
            str(_value(row, "status", 0)),
            _value(row, "available_at", 1) is None,
            _value(row, "completion_token", 2),
        )
        for row in rows
    ]


def _retry_snapshots(
    manager: JobManager,
    job_ids: list[int],
) -> list[dict[str, Any]]:
    columns = ("status", "available_at", *_RETRY_RESET_FIELDS)
    if manager.backend == "postgres":
        rows = _raw_fetchall(
            manager,
            "",
            (job_ids,),
            postgres_sql=(
                f"SELECT {','.join(columns)} FROM jobs "  # nosec B608
                "WHERE id = ANY(%s) ORDER BY id"
            ),
        )
    else:
        placeholders = ",".join("?" for _ in job_ids)
        rows = _raw_fetchall(
            manager,
            (
                f"SELECT {','.join(columns)} FROM jobs "  # nosec B608
                f"WHERE id IN ({placeholders}) ORDER BY id"  # nosec B608
            ),
            tuple(job_ids),
        )
    return [
        {
            column: _value(row, column, index)
            for index, column in enumerate(columns)
        }
        for row in rows
    ]


def _seed_ready(
    manager: JobManager,
    *,
    domain: str,
    count: int,
) -> list[int]:
    return [
        int(
            manager.create_job(
                domain=domain,
                queue="default",
                job_type="work",
                payload={},
                owner_user_id="owner",
            )["id"]
        )
        for _ in range(count)
    ]


def _seed_terminal(
    manager: JobManager,
    *,
    domain: str,
    status: str,
    count: int = 2,
) -> list[int]:
    if status == "failed":
        _seed_ready(manager, domain=domain, count=count)
        job_ids = []
        for attempt in range(count):
            worker_id = f"terminal-seed-{attempt}"
            acquired = manager.acquire_next_job(
                domain=domain,
                queue="default",
                job_type="work",
                lease_seconds=30,
                worker_id=worker_id,
            )
            assert acquired is not None
            job_id = int(acquired["id"])
            _raw_execute(
                manager,
                "UPDATE jobs SET result=? WHERE id=?",
                ('{"stale":"result"}', job_id),
                postgres_sql="UPDATE jobs SET result=%s::jsonb WHERE id=%s",
            )
            assert manager.fail_job(
                job_id,
                error="stale error message",
                retryable=False,
                worker_id=worker_id,
                lease_id=str(acquired["lease_id"]),
                error_code="stale_error",
                error_class="StaleFailure",
                error_stack={"trace": "stale"},
                completion_token=f"token-{job_id}",
                enforce=True,
            )
            job_ids.append(job_id)
    else:
        job_ids = _seed_ready(manager, domain=domain, count=count)
        for job_id in job_ids:
            _raw_execute(
                manager,
                (
                    "UPDATE jobs SET status=?, completion_token=?, retry_count=0, "
                    "available_at=NULL, quarantined_at=DATETIME('now') WHERE id=?"
                ),
                (status, f"token-{job_id}", job_id),
                postgres_sql=(
                    "UPDATE jobs SET status=%s, completion_token=%s, retry_count=0, "
                    "available_at=NULL, quarantined_at=NOW() WHERE id=%s"
                ),
            )
    _raw_execute(
        manager,
        (
            "UPDATE job_counters SET ready_count=0, scheduled_count=0, "
            "processing_count=0, quarantined_count=? "
            "WHERE domain=? AND queue=? AND job_type=?"
        ),
        (count if status == "quarantined" else 0, domain, "default", "work"),
        postgres_sql=(
            "UPDATE job_counters SET ready_count=0, scheduled_count=0, "
            "processing_count=0, quarantined_count=%s "
            "WHERE domain=%s AND queue=%s AND job_type=%s"
        ),
    )
    return job_ids


def _seed_ready_and_due_scheduled(
    manager: JobManager,
    *,
    domain: str,
) -> list[int]:
    ready = _seed_ready(manager, domain=domain, count=1)[0]
    scheduled = int(
        manager.create_job(
            domain=domain,
            queue="default",
            job_type="work",
            payload={},
            owner_user_id="owner",
            available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
        )["id"]
    )
    _raw_execute(
        manager,
        "UPDATE jobs SET available_at=DATETIME('now', '-1 hour') WHERE id=?",
        (scheduled,),
        postgres_sql="UPDATE jobs SET available_at=NOW() - INTERVAL '1 hour' WHERE id=%s",
    )
    return [ready, scheduled]


def _invoke(
    operation: str,
    manager: JobManager,
    *,
    domain: str,
    delay_seconds: int = 30,
    job_id: int | None = None,
    only_failed: bool = True,
) -> int:
    if operation == "retry":
        return manager.retry_now_jobs(
            job_id=job_id,
            domain=domain,
            queue="default",
            job_type="work",
            only_failed=only_failed,
        )
    if operation == "cancel":
        response = asyncio.run(
            jobs_admin.batch_cancel_endpoint(
                jobs_admin.BatchCancelRequest(
                    domain=domain,
                    queue="default",
                    job_type="work",
                    job_id=job_id,
                ),
                _confirmed_request(),
                _admin_principal(),
            )
        )
        return int(response.affected)
    if operation == "reschedule":
        assert job_id is None
        response = asyncio.run(
            jobs_admin.batch_reschedule_endpoint(
                jobs_admin.BatchRescheduleRequest(
                    domain=domain,
                    queue="default",
                    job_type="work",
                    delay_seconds=delay_seconds,
                ),
                _confirmed_request(),
                _admin_principal(),
            )
        )
        return int(response.affected)
    if operation == "requeue":
        response = asyncio.run(
            jobs_admin.batch_requeue_quarantined_endpoint(
                jobs_admin.BatchRequeueQuarantinedRequest(
                    domain=domain,
                    queue="default",
                    job_type="work",
                    job_id=job_id,
                ),
                _confirmed_request(),
                _admin_principal(),
            )
        )
        return int(response.affected)
    raise AssertionError(f"unsupported operation: {operation}")


def _seed_for_operation(
    operation: str,
    manager: JobManager,
    *,
    domain: str,
    count: int = 2,
) -> list[int]:
    if operation == "retry":
        return _seed_terminal(manager, domain=domain, status="failed", count=count)
    if operation == "requeue":
        return _seed_terminal(manager, domain=domain, status="quarantined", count=count)
    return _seed_ready(manager, domain=domain, count=count)


def _patch_connections(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[Any], Any],
) -> None:
    def _connect(manager: JobManager) -> Any:
        return factory(_ORIGINAL_CONNECT(manager))

    monkeypatch.setattr(JobManager, "_connect", _connect)


def _capture_failure(callback: Callable[[], Any]) -> BaseException | None:
    try:
        callback()
    except (HTTPException, RuntimeError) as exc:
        return exc
    return None


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("operation", ["retry", "cancel", "reschedule", "requeue"])
def test_duplicate_admin_mutations_use_only_rows_updated_by_each_transaction(
    backend: str,
    operation: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"duplicate-{operation}-{backend}",
    )
    domain = f"duplicate-{operation}-{backend}"
    job_ids = _seed_for_operation(operation, manager, domain=domain)
    _bind_endpoint_manager(monkeypatch, manager)

    select_barrier = threading.Barrier(2)
    start_barrier = threading.Barrier(2)
    counter_writes: list[str] = []

    def _wrap(inner: Any) -> _ConnectionProxy:
        claimed = False
        waited = False

        def _before_execute(sql: Any) -> None:
            nonlocal claimed, waited
            normalized = _normalize_sql(sql)
            if normalized == "begin immediate":
                claimed = True
            if "job_counters" in normalized and normalized.startswith(("insert", "update")):
                counter_writes.append(normalized)
            is_legacy_preaggregation = (
                normalized.startswith("select domain, queue, job_type, count(")
                and " from jobs " in f" {normalized} "
            )
            if not claimed and not waited and is_legacy_preaggregation:
                waited = True
                select_barrier.wait(timeout=10)

        return _ConnectionProxy(inner, before_execute=_before_execute)

    _patch_connections(monkeypatch, _wrap)

    def _run() -> int:
        start_barrier.wait(timeout=10)
        return _invoke(operation, manager, domain=domain)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(_run) for _ in range(2)]
        results = [future.result(timeout=20) for future in futures]

    if operation == "retry":
        assert sorted(results) == [0, len(job_ids)]
    else:
        assert all(result in {0, len(job_ids)} for result in results)
    assert len(counter_writes) == 1

    snapshots = _job_snapshots(manager, job_ids)
    expected_status = {
        "retry": "queued",
        "cancel": "cancelled",
        "reschedule": "queued",
        "requeue": "queued",
    }[operation]
    assert all(snapshot[0] == expected_status for snapshot in snapshots)
    if operation in {"retry", "requeue"}:
        assert all(snapshot[2] is None for snapshot in snapshots)
    if operation == "retry":
        retry_snapshots = _retry_snapshots(manager, job_ids)
        assert all(
            snapshot[field] is None
            for snapshot in retry_snapshots
            for field in _RETRY_RESET_FIELDS
        )

    expected_counter = {
        "retry": (len(job_ids), 0, 0, 0),
        "cancel": (0, 0, 0, 0),
        "reschedule": (0, len(job_ids), 0, 0),
        "requeue": (len(job_ids), 0, 0, 0),
    }[operation]
    assert _counter(manager, domain=domain) == expected_counter


@pytest.mark.parametrize("backend", _BACKENDS)
def test_retry_failed_jobs_atomically_clear_terminal_and_attempt_state(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"retry-state-{backend}",
    )
    domain = f"retry-state-{backend}"
    job_ids = _seed_terminal(
        manager,
        domain=domain,
        status="failed",
        count=2,
    )

    before = _retry_snapshots(manager, job_ids)
    assert all(
        snapshot[field] is not None
        for snapshot in before
        for field in _RETRY_RESET_FIELDS
    )
    assert _counter(manager, domain=domain) == (0, 0, 0, 0)

    assert _invoke("retry", manager, domain=domain) == len(job_ids)

    after = _retry_snapshots(manager, job_ids)
    assert all(snapshot["status"] == "queued" for snapshot in after)
    assert all(snapshot["available_at"] is None for snapshot in after)
    assert all(
        snapshot[field] is None
        for snapshot in after
        for field in _RETRY_RESET_FIELDS
    )
    assert _counter(manager, domain=domain) == (len(job_ids), 0, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_batch_cancel_uses_nullness_as_the_durable_queue_bucket(
    backend: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"cancel-bucket-{backend}",
    )
    domain = f"cancel-bucket-{backend}"
    job_ids = _seed_ready_and_due_scheduled(manager, domain=domain)
    _bind_endpoint_manager(monkeypatch, manager)

    assert _counter(manager, domain=domain) == (1, 1, 0, 0)
    assert _invoke("cancel", manager, domain=domain) == len(job_ids)
    assert _counter(manager, domain=domain) == (0, 0, 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("delay_seconds", [0, 30], ids=["ready", "scheduled"])
def test_batch_reschedule_preserves_canonical_ready_and_scheduled_buckets(
    backend: str,
    delay_seconds: int,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"reschedule-bucket-{backend}-{delay_seconds}",
    )
    domain = f"reschedule-bucket-{backend}-{delay_seconds}"
    job_ids = _seed_ready_and_due_scheduled(manager, domain=domain)
    _bind_endpoint_manager(monkeypatch, manager)

    assert _counter(manager, domain=domain) == (1, 1, 0, 0)
    assert _invoke(
        "reschedule",
        manager,
        domain=domain,
        delay_seconds=delay_seconds,
    ) == len(job_ids)

    snapshots = _job_snapshots(manager, job_ids)
    if delay_seconds == 0:
        assert all(snapshot[1] for snapshot in snapshots)
        assert _counter(manager, domain=domain) == (2, 0, 0, 0)
    else:
        assert all(not snapshot[1] for snapshot in snapshots)
        assert _counter(manager, domain=domain) == (0, 2, 0, 0)


@pytest.mark.pg_jobs
def test_postgres_batch_reschedule_reports_the_jobs_update_rowcount(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        "postgres",
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name="reschedule-rowcount-postgres",
    )
    domain = "reschedule-rowcount-postgres"
    job_ids = _seed_ready(manager, domain=domain, count=3)
    _bind_endpoint_manager(monkeypatch, manager)

    assert _invoke("reschedule", manager, domain=domain) == len(job_ids)
    assert _counter(manager, domain=domain) == (0, len(job_ids), 0, 0)


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize(
    "scenario",
    [
        "retry_failed",
        "retry_scheduled",
        "cancel",
        "reschedule",
        "requeue",
    ],
)
def test_missing_counter_reconciles_exact_same_key_lifecycle_state(
    backend: str,
    scenario: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"missing-counter-{scenario}-{backend}",
    )
    domain = f"missing-counter-{scenario}-{backend}"
    operation = scenario.split("_", 1)[0]
    only_failed = True

    if scenario == "retry_failed":
        target = _seed_terminal(
            manager,
            domain=domain,
            status="failed",
            count=1,
        )[0]
        _seed_ready(manager, domain=domain, count=1)
        expected_counter = (2, 0, 0, 0)
    elif scenario == "retry_scheduled":
        _seed_ready(manager, domain=domain, count=1)
        target = int(
            manager.create_job(
                domain=domain,
                queue="default",
                job_type="work",
                payload={},
                owner_user_id="owner",
                available_at=datetime.now(tz=timezone.utc) + timedelta(days=1),
            )["id"]
        )
        _raw_execute(
            manager,
            "UPDATE jobs SET completion_token=? WHERE id=?",
            ("stale-scheduled-token", target),
            postgres_sql="UPDATE jobs SET completion_token=%s WHERE id=%s",
        )
        expected_counter = (2, 0, 0, 0)
        only_failed = False
    elif scenario == "cancel":
        target = _seed_ready(manager, domain=domain, count=2)[0]
        expected_counter = (1, 0, 0, 0)
    elif scenario == "reschedule":
        processing = _seed_ready(manager, domain=domain, count=1)[0]
        acquired = manager.acquire_next_job(
            domain=domain,
            queue="default",
            lease_seconds=30,
            worker_id="worker",
        )
        assert acquired is not None
        assert int(acquired["id"]) == processing
        target = _seed_ready(manager, domain=domain, count=1)[0]
        expected_counter = (0, 1, 1, 0)
    else:
        target = _seed_terminal(
            manager,
            domain=domain,
            status="quarantined",
            count=1,
        )[0]
        _seed_ready(manager, domain=domain, count=1)
        expected_counter = (2, 0, 0, 0)

    _raw_execute(
        manager,
        (
            "DELETE FROM job_counters WHERE domain=? AND queue=? AND job_type=?"
        ),
        (domain, "default", "work"),
        postgres_sql=(
            "DELETE FROM job_counters WHERE domain=%s AND queue=%s AND job_type=%s"
        ),
    )
    assert _counter(manager, domain=domain) is None
    _bind_endpoint_manager(monkeypatch, manager)

    assert (
        _invoke(
            operation,
            manager,
            domain=domain,
            job_id=None if scenario == "reschedule" else target,
            only_failed=only_failed,
        )
        == 1
    )
    assert _counter(manager, domain=domain) == expected_counter

    if scenario in {"retry_failed", "retry_scheduled", "requeue"}:
        assert _job_snapshots(manager, [target])[0][2] is None


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("operation", ["retry", "cancel", "reschedule", "requeue"])
def test_counter_write_failure_rolls_back_admin_state_and_suppresses_gauges(
    backend: str,
    operation: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"counter-failure-{operation}-{backend}",
    )
    domain = f"counter-failure-{operation}-{backend}"
    job_ids = _seed_for_operation(operation, manager, domain=domain, count=1)
    before_jobs = _job_snapshots(manager, job_ids)
    before_retry_state = (
        _retry_snapshots(manager, job_ids)
        if operation == "retry"
        else None
    )
    before_counter = _counter(manager, domain=domain)
    _bind_endpoint_manager(monkeypatch, manager)

    gauge_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        JobManager,
        "_update_gauges",
        lambda self, **kwargs: gauge_calls.append(kwargs),
    )

    def _wrap(inner: Any) -> _ConnectionProxy:
        failed = False

        def _before_execute(sql: Any) -> None:
            nonlocal failed
            normalized = _normalize_sql(sql)
            if not failed and "job_counters" in normalized and normalized.startswith(("insert", "update")):
                failed = True
                raise RuntimeError("forced admin counter failure")

        return _ConnectionProxy(inner, before_execute=_before_execute)

    _patch_connections(monkeypatch, _wrap)

    failure = _capture_failure(
        lambda: _invoke(operation, manager, domain=domain)
    )

    assert failure is not None
    if isinstance(failure, HTTPException):
        assert failure.status_code == 500
    assert _job_snapshots(manager, job_ids) == before_jobs
    if operation == "retry":
        assert _retry_snapshots(manager, job_ids) == before_retry_state
    assert _counter(manager, domain=domain) == before_counter
    assert gauge_calls == []


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("operation", ["cancel", "reschedule", "requeue"])
def test_commit_failure_rolls_back_admin_endpoint_and_suppresses_gauges(
    backend: str,
    operation: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        backend,
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"commit-failure-{operation}-{backend}",
    )
    domain = f"commit-failure-{operation}-{backend}"
    job_ids = _seed_for_operation(operation, manager, domain=domain, count=1)
    before_jobs = _job_snapshots(manager, job_ids)
    before_counter = _counter(manager, domain=domain)
    _bind_endpoint_manager(monkeypatch, manager)

    gauge_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        JobManager,
        "_update_gauges",
        lambda self, **kwargs: gauge_calls.append(kwargs),
    )
    _patch_connections(
        monkeypatch,
        lambda inner: _ConnectionProxy(inner, fail_commit=True),
    )

    failure = _capture_failure(
        lambda: _invoke(operation, manager, domain=domain)
    )

    assert isinstance(failure, HTTPException)
    assert failure.status_code == 500
    assert _job_snapshots(manager, job_ids) == before_jobs
    assert _counter(manager, domain=domain) == before_counter
    assert gauge_calls == []


@pytest.mark.parametrize("operation", ["retry", "cancel", "reschedule", "requeue"])
def test_sqlite_admin_preaggregation_starts_with_begin_immediate(
    operation: str,
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = _manager(
        "sqlite",
        request=request,
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        name=f"begin-immediate-{operation}",
    )
    domain = f"begin-immediate-{operation}"
    _seed_for_operation(operation, manager, domain=domain, count=1)
    _bind_endpoint_manager(monkeypatch, manager)

    statements: list[str] = []
    _patch_connections(
        monkeypatch,
        lambda inner: _ConnectionProxy(
            inner,
            before_execute=lambda sql: statements.append(_normalize_sql(sql)),
        ),
    )

    assert _invoke(operation, manager, domain=domain) == 1

    begin_index = statements.index("begin immediate")
    critical_index = next(
        index
        for index, statement in enumerate(statements)
        if (
            (
                statement.startswith("select domain, queue, job_type, count(")
                and " from jobs " in f" {statement} "
            )
            or statement.startswith("update jobs set")
        )
    )
    assert begin_index < critical_index
