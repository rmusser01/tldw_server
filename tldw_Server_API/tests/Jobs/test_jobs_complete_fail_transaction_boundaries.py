from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager


class _FailCommit:
    """Roll back and fail the transaction's commit boundary."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._manual_commit_attempted = False

    def __enter__(self) -> _FailCommit:
        self._inner.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if exc_type is not None:
            self._inner.rollback()
            return False
        self._inner.rollback()
        if self._manual_commit_attempted:
            return False
        raise RuntimeError("forced commit failure")

    def commit(self) -> None:
        self._manual_commit_attempted = True
        self._inner.rollback()
        raise RuntimeError("forced commit failure")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _manager(tmp_path: Any, jobs_pg_dsn: str | None) -> JobManager:
    if jobs_pg_dsn is not None:
        return JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    return JobManager(tmp_path / "finalize-boundary.db")


def _processing_job(jm: JobManager, *, suffix: str = "") -> dict[str, Any]:
    created = jm.create_job(
        domain=f"finalize-boundary{suffix}",
        queue="default",
        job_type="work",
        payload={},
        owner_user_id="owner-1",
        request_id="request-1",
        trace_id="trace-1",
        max_retries=2,
    )
    acquired = jm.acquire_next_job(
        domain=str(created["domain"]),
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
    )
    assert acquired is not None
    assert int(acquired["id"]) == int(created["id"])
    return acquired


def _patch_observers(monkeypatch: pytest.MonkeyPatch, jm: JobManager) -> dict[str, list[Any]]:
    import tldw_Server_API.app.core.Jobs.manager as manager_module
    import tldw_Server_API.app.core.Jobs.metrics as metrics_module

    calls: dict[str, list[Any]] = {
        "completed": [],
        "duration": [],
        "failed": [],
        "retried": [],
        "retry_after": [],
        "event": [],
        "gauge": [],
        "cascade": [],
    }
    monkeypatch.setattr(manager_module, "increment_completed", lambda labels: calls["completed"].append(labels))
    monkeypatch.setattr(manager_module, "observe_duration", lambda *args: calls["duration"].append(args))
    monkeypatch.setattr(manager_module, "increment_failures", lambda *args, **kwargs: calls["failed"].append((args, kwargs)))
    monkeypatch.setattr(manager_module, "increment_retries", lambda labels: calls["retried"].append(labels))
    monkeypatch.setattr(metrics_module, "observe_retry_after", lambda *args: calls["retry_after"].append(args))
    def event_observer(event_type: str, **kwargs: Any) -> None:
        calls["event"].append((event_type, kwargs))

    monkeypatch.setattr(manager_module, "emit_job_event", event_observer)
    monkeypatch.setattr(manager_module, "observe_job_event", event_observer)
    monkeypatch.setattr(jm, "_update_gauges", lambda **kwargs: calls["gauge"].append(kwargs))
    monkeypatch.setattr(
        jm,
        "_cancel_dependent_jobs",
        lambda job_uuid, **kwargs: calls["cascade"].append((job_uuid, kwargs)),
    )
    return calls


def _assert_no_observers(calls: dict[str, list[Any]]) -> None:
    assert calls == {
        "completed": [],
        "duration": [],
        "failed": [],
        "retried": [],
        "retry_after": [],
        "event": [],
        "gauge": [],
        "cascade": [],
    }


def _assert_complete_commit_failure(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquired = _processing_job(jm, suffix="-complete-rollback")
    job_id = int(acquired["id"])
    calls = _patch_observers(monkeypatch, jm)
    original_connect = jm._connect
    monkeypatch.setattr(jm, "_connect", lambda: _FailCommit(original_connect()))

    with pytest.raises(RuntimeError, match="forced commit failure"):
        jm.complete_job(
            job_id,
            result={"ok": True},
            worker_id=str(acquired["worker_id"]),
            lease_id=str(acquired["lease_id"]),
        )

    persisted = reader.get_job(job_id)
    assert persisted is not None
    assert persisted["status"] == "processing"
    assert persisted["worker_id"] == "worker-1"
    assert persisted["lease_id"] == acquired["lease_id"]
    _assert_no_observers(calls)


def _assert_fail_commit_failure(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retryable: bool,
) -> None:
    acquired = _processing_job(jm, suffix=f"-fail-rollback-{retryable}")
    job_id = int(acquired["id"])
    calls = _patch_observers(monkeypatch, jm)
    original_connect = jm._connect
    monkeypatch.setattr(jm, "_connect", lambda: _FailCommit(original_connect()))

    with pytest.raises(RuntimeError, match="forced commit failure"):
        jm.fail_job(
            job_id,
            error="boom",
            retryable=retryable,
            backoff_seconds=1,
            worker_id=str(acquired["worker_id"]),
            lease_id=str(acquired["lease_id"]),
        )

    persisted = reader.get_job(job_id)
    assert persisted is not None
    assert persisted["status"] == "processing"
    assert persisted["worker_id"] == "worker-1"
    assert persisted["lease_id"] == acquired["lease_id"]
    _assert_no_observers(calls)


def _assert_success_observers_see_committed_state(
    jm: JobManager,
    reader: JobManager,
    monkeypatch: pytest.MonkeyPatch,
    *,
    operation: str,
) -> None:
    acquired = _processing_job(jm, suffix=f"-{operation}-success")
    job_id = int(acquired["id"])
    snapshots: list[tuple[str, str, Any, Any]] = []

    def snapshot(observer: str):
        def record(*_args: Any, **_kwargs: Any) -> None:
            persisted = reader.get_job(job_id)
            assert persisted is not None
            snapshots.append(
                (
                    observer,
                    persisted["status"],
                    persisted["worker_id"],
                    persisted["lease_id"],
                )
            )

        return record

    import tldw_Server_API.app.core.Jobs.manager as manager_module

    monkeypatch.setattr(jm, "_update_gauges", snapshot("gauge"))
    event_snapshot = snapshot("event")
    monkeypatch.setattr(manager_module, "emit_job_event", event_snapshot)
    monkeypatch.setattr(manager_module, "observe_job_event", event_snapshot)
    if operation == "complete":
        monkeypatch.setattr(manager_module, "increment_completed", snapshot("completed"))
        assert jm.complete_job(
            job_id,
            result={"ok": True},
            worker_id=str(acquired["worker_id"]),
            lease_id=str(acquired["lease_id"]),
        )
        expected_status = "completed"
    else:
        monkeypatch.setattr(manager_module, "increment_failures", snapshot("failed"))
        monkeypatch.setattr(jm, "_cancel_dependent_jobs", snapshot("cascade"))
        assert jm.fail_job(
            job_id,
            error="boom",
            retryable=False,
            worker_id=str(acquired["worker_id"]),
            lease_id=str(acquired["lease_id"]),
        )
        expected_status = "failed"

    assert snapshots
    final = reader.get_job(job_id)
    assert final is not None
    assert all(
        (status, worker_id, lease_id) == (expected_status, None, None)
        for _, status, worker_id, lease_id in snapshots
    ), {
        "snapshots": snapshots,
        "final": (final["status"], final["worker_id"], final["lease_id"]),
    }


@pytest.mark.unit
def test_complete_sqlite_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = _manager(tmp_path, None)
    _assert_complete_commit_failure(jm, _manager(tmp_path, None), monkeypatch)


@pytest.mark.pg_jobs
def test_complete_postgres_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_complete_commit_failure(jm, _manager(tmp_path, jobs_pg_dsn), monkeypatch)


@pytest.mark.unit
@pytest.mark.parametrize("retryable", [False, True], ids=["terminal", "retry"])
def test_fail_sqlite_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    retryable: bool,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "99")
    jm = _manager(tmp_path, None)
    _assert_fail_commit_failure(jm, _manager(tmp_path, None), monkeypatch, retryable=retryable)


@pytest.mark.pg_jobs
@pytest.mark.parametrize("retryable", [False, True], ids=["terminal", "retry"])
def test_fail_postgres_commit_failure_rolls_back_without_observers(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    retryable: bool,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "99")
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_fail_commit_failure(jm, _manager(tmp_path, jobs_pg_dsn), monkeypatch, retryable=retryable)


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_finalize_sqlite_observers_see_committed_state_and_cleared_lease(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = _manager(tmp_path, None)
    _assert_success_observers_see_committed_state(
        jm,
        _manager(tmp_path, None),
        monkeypatch,
        operation=operation,
    )


@pytest.mark.pg_jobs
@pytest.mark.parametrize("operation", ["complete", "fail"])
def test_finalize_postgres_observers_see_committed_state_and_cleared_lease(
    tmp_path: Any,
    jobs_pg_dsn: str,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "false")
    jm = _manager(tmp_path, jobs_pg_dsn)
    _assert_success_observers_see_committed_state(
        jm,
        _manager(tmp_path, jobs_pg_dsn),
        monkeypatch,
        operation=operation,
    )
