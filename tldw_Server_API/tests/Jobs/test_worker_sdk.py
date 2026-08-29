import asyncio
import gc
import sqlite3

import pytest

from tldw_Server_API.app.core.Jobs import worker_sdk as worker_sdk_module
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK

try:
    import psycopg
except ImportError:  # pragma: no cover - exercised in minimal SQLite installs
    psycopg = None


_BACKEND_ADAPTER_ERRORS = [
    pytest.param(sqlite3.OperationalError, id="sqlite"),
]
if psycopg is not None:
    _BACKEND_ADAPTER_ERRORS.append(
        pytest.param(psycopg.OperationalError, id="postgres"),
    )


class DummySleep:
    """Async sleep stub that records durations and yields via original sleep.

    Important: pass the original asyncio.sleep to avoid recursive self-calls
    when tests monkeypatch asyncio.sleep to this stub.
    """
    def __init__(self, orig_sleep):
        self.calls = []
        self._orig_sleep = orig_sleep

    async def __call__(self, seconds: float):
        self.calls.append(seconds)
        # Yield control using the original sleep to avoid recursion
        await self._orig_sleep(0)


class _TerminalWorkerError(RuntimeError):
    retryable = False
    failure_code = "clone_failed"


class _RetryableWorkerError(RuntimeError):
    retryable = True


@pytest.mark.asyncio
@pytest.mark.parametrize(("bind_completion_token", "fails"), [(False, False), (True, False), (True, True)])
async def test_completion_token_binding_is_opt_in_for_success_and_failure(
    tmp_path,
    bind_completion_token,
    fails,
):
    manager = JobManager(tmp_path / f"completion-token-{bind_completion_token}-{fails}.db")
    job = manager.create_job(
        domain="chatbooks",
        queue="default",
        job_type="completion-token-contract",
        payload={},
        owner_user_id="owner-1",
        max_retries=0,
    )
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="worker-token-contract",
            bind_completion_token=bind_completion_token,
            retry_on_exception=False,
        ),
    )

    async def handler(acquired):
        sdk.stop()
        if fails:
            raise _TerminalWorkerError("bounded failure")
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler), timeout=1)

    stored = manager.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == ("failed" if fails else "completed")
    assert bool(stored["completion_token"]) is bind_completion_token


def _slides_jobs_key(character: str) -> str:
    return "slides:v1:" + character * 64


def _failure_test_worker(tmp_path, name: str, *, max_retries: int = 0):
    db_path = tmp_path / f"{name}.db"
    ensure_jobs_tables(db_path)
    manager = JobManager(db_path)
    job = manager.create_job(
        domain="chatbooks",
        queue="default",
        job_type="failure-callback",
        payload={},
        owner_user_id="u",
        max_retries=max_retries,
    )
    sdk = WorkerSDK(
        manager,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id=f"worker-{name}",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
            retry_on_exception=max_retries > 0,
        ),
    )
    return manager, job, sdk


@pytest.mark.asyncio
async def test_run_with_job_type_filter_only_handles_matching_jobs(tmp_path):
    db_path = tmp_path / "jobs_wsdk_job_type.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    unwanted = jm.create_job(
        domain="writing",
        queue="default",
        job_type="unwanted",
        payload={"name": "skip"},
        owner_user_id="u",
    )
    wanted = jm.create_job(
        domain="writing",
        queue="default",
        job_type="wanted",
        payload={"name": "handle"},
        owner_user_id="u",
    )

    cfg = WorkerConfig(
        domain="writing",
        queue="default",
        worker_id="w-job-type",
        lease_seconds=5,
        renew_threshold_seconds=1,
        renew_jitter_seconds=0,
    )
    sdk = WorkerSDK(jm, cfg)
    sdk._sleep = DummySleep(asyncio.sleep)
    handled: list[str] = []

    async def handler(job_row):
        handled.append(str(job_row["job_type"]))
        sdk.stop()
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler, job_type="wanted"), timeout=1)

    assert handled == ["wanted"]
    assert jm.get_job(int(wanted["id"]))["status"] == "completed"
    assert jm.get_job(int(unwanted["id"]))["status"] == "queued"


@pytest.mark.asyncio
async def test_auto_renew_jitter_and_progress(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=20, worker_id="w1")
    assert acq is not None

    # Configure worker with no jitter for deterministic sleep
    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w1", lease_seconds=20, renew_threshold_seconds=5, renew_jitter_seconds=0)
    sdk = WorkerSDK(jm, cfg)

    # Capture renew calls and progress fields
    calls = []
    renewed = asyncio.Event()

    def fake_renew(**kwargs):
        calls.append(kwargs)
        renewed.set()
        return True

    # Capture original sleep; use it inside the stub and assign to sdk._sleep
    _orig_sleep = asyncio.sleep
    sleep_stub = DummySleep(_orig_sleep)
    monkeypatch.setattr(jm, "renew_job_lease", lambda **kwargs: fake_renew(**kwargs))
    sdk._sleep = sleep_stub

    # Provide a progress callback
    def progress_cb():
        return {"progress_percent": 12.5, "progress_message": "tick"}

    task = asyncio.create_task(sdk._auto_renew(acq, progress_cb=progress_cb))
    await asyncio.wait_for(renewed.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    # Verify sleep durations are lease - threshold (no jitter)
    assert any(abs(s - 15) < 0.1 for s in sleep_stub.calls)
    # Verify renew_job_lease received progress args
    assert any("progress_percent" in c and c.get("progress_percent") == 12.5 for c in calls)
    assert any("progress_message" in c and c.get("progress_message") == "tick" for c in calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_error", _BACKEND_ADAPTER_ERRORS)
async def test_run_treats_backend_acquire_errors_as_transient(adapter_error):
    class FailingManager:
        def should_enforce_leases(self):
            return True

        def acquire_next_job(self, **_kwargs):
            sdk.stop()
            raise adapter_error("adapter unavailable")

    sdk = WorkerSDK(
        FailingManager(),
        WorkerConfig(domain="jobs", queue="default", worker_id="worker"),
    )

    async def handler(_job):
        pytest.fail("handler must not run after an acquire adapter error")

    await asyncio.wait_for(sdk.run(handler=handler), timeout=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter_error", _BACKEND_ADAPTER_ERRORS)
async def test_auto_renew_contains_backend_adapter_errors(adapter_error):
    class FailingManager:
        def renew_job_lease(self, **_kwargs):
            raise adapter_error("adapter unavailable")

    sdk = WorkerSDK(
        FailingManager(),
        WorkerConfig(
            domain="jobs",
            queue="default",
            worker_id="worker",
            lease_seconds=2,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    sdk._sleep = DummySleep(asyncio.sleep)

    await asyncio.wait_for(
        sdk._auto_renew(
            {
                "id": 7,
                "lease_id": "lease-7",
            }
        ),
        timeout=1,
    )


@pytest.mark.asyncio
async def test_run_contains_finished_renew_task_adapter_error(tmp_path):
    jm = JobManager(tmp_path / "jobs-renew-task-error.db")
    created = jm.create_job(
        domain="jobs",
        queue="default",
        job_type="renew-error",
        payload={},
        owner_user_id="owner",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(domain="jobs", queue="default", worker_id="worker"),
    )

    async def failing_renew(_job, progress_cb=None):
        del progress_cb
        raise sqlite3.OperationalError("renew adapter unavailable")

    sdk._auto_renew = failing_renew

    async def handler(_job):
        await asyncio.sleep(0)
        sdk.stop()
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler), timeout=1)

    assert jm.get_job(int(created["id"]))["status"] == "completed"


@pytest.mark.asyncio
async def test_run_retryable_exception_and_backoff(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk2.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")

    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w2")
    assert acq is not None

    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w2", lease_seconds=10, backoff_base_seconds=2, backoff_max_seconds=8)
    sdk = WorkerSDK(jm, cfg)

    # Acquire once then no more jobs
    acquires = {"count": 0}
    def fake_acquire(**kwargs):
        if acquires["count"] == 0:
            acquires["count"] += 1
            return acq
        return None

    class RetryErr(Exception):
        retryable = True
        backoff_seconds = 7

    fail_calls = []
    def fake_fail(job_id, **kwargs):
        fail_calls.append({"job_id": job_id, **kwargs})

    # Capture and use original sleep inside the stub
    _orig_sleep = asyncio.sleep
    sleep_stub = DummySleep(_orig_sleep)
    backoff_sleep_started = asyncio.Event()

    async def tracked_sleep(seconds: float) -> None:
        if int(seconds) in (2, 4, 8):
            backoff_sleep_started.set()
        await sleep_stub(seconds)

    monkeypatch.setattr(jm, "acquire_next_job", lambda **kwargs: fake_acquire(**kwargs))
    monkeypatch.setattr(jm, "fail_job", lambda job_id, **kwargs: fake_fail(job_id, **kwargs))
    sdk._sleep = tracked_sleep

    async def handler(job):
        raise RetryErr("boom")

    run_task = asyncio.create_task(sdk.run(handler=handler))
    await asyncio.wait_for(backoff_sleep_started.wait(), timeout=1)
    sdk.stop()
    await asyncio.wait_for(run_task, timeout=1)

    # Verify fail_job was called with retryable True and backoff_seconds from exception
    assert any(c.get("retryable") is True and int(c.get("backoff_seconds")) == 7 for c in fail_calls)
    # Verify backoff sleeps used exponential sequence up to max
    # After job handled and no further jobs, loop should sleep at least base once
    assert any(int(s) in (2, 4, 8) for s in sleep_stub.calls)


@pytest.mark.asyncio
async def test_run_stop_interrupts_idle_backoff_sleep(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_stop.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)

    cfg = WorkerConfig(
        domain="chatbooks",
        queue="default",
        worker_id="w-stop",
        backoff_base_seconds=30,
        backoff_max_seconds=30,
    )
    sdk = WorkerSDK(jm, cfg)
    sleep_started = asyncio.Event()

    async def blocking_sleep(_seconds: float) -> None:
        sleep_started.set()
        await asyncio.Future()

    async def handler(_job):
        pytest.fail("Handler should not run when there are no jobs")

    monkeypatch.setattr(jm, "acquire_next_job", lambda **kwargs: None)
    sdk._sleep = blocking_sleep

    task = asyncio.create_task(sdk.run(handler=handler))
    await asyncio.wait_for(sleep_started.wait(), timeout=1)
    sdk.stop()
    await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_stop_keeps_active_job_renewed_until_handler_finishes(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "jobs_wsdk_stop_active.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="w-stop-active",
            lease_seconds=30,
            renew_threshold_seconds=29,
            renew_jitter_seconds=0,
        ),
    )
    handler_started = asyncio.Event()
    release_handler = asyncio.Event()
    renew_sleep_started = asyncio.Event()
    allow_renew_interval = asyncio.Event()
    renewed = asyncio.Event()
    next_renew_sleep_started = asyncio.Event()
    renew_cancel_received = asyncio.Event()
    allow_renew_cleanup = asyncio.Event()
    completion_succeeded = asyncio.Event()
    worker_acquisitions: list[str] = []
    sleep_calls = 0
    original_acquire = jm.acquire_next_job
    original_renew = jm.renew_job_lease
    original_complete = jm.complete_job

    def tracked_acquire(**kwargs):
        worker_acquisitions.append(str(kwargs["worker_id"]))
        return original_acquire(**kwargs)

    def tracked_renew(**kwargs):
        result = original_renew(**kwargs)
        if result:
            renewed.set()
        return result

    def tracked_complete(job_id, **kwargs):
        result = original_complete(job_id, **kwargs)
        if result:
            completion_succeeded.set()
        return result

    async def controlled_sleep(_seconds: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls == 1:
            renew_sleep_started.set()
            await allow_renew_interval.wait()
            return
        next_renew_sleep_started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            renew_cancel_received.set()
            await allow_renew_cleanup.wait()
            raise

    async def handler(_job_row):
        handler_started.set()
        await release_handler.wait()
        return {"ok": True}

    monkeypatch.setattr(jm, "acquire_next_job", tracked_acquire)
    monkeypatch.setattr(jm, "renew_job_lease", tracked_renew)
    monkeypatch.setattr(jm, "complete_job", tracked_complete)
    sdk._sleep = controlled_sleep

    run_task = asyncio.create_task(sdk.run(handler=handler))
    try:
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        await asyncio.wait_for(renew_sleep_started.wait(), timeout=1)
        sdk.stop()

        connection = jm._connect()
        try:
            with connection:
                connection.execute(
                    "UPDATE jobs SET leased_until=DATETIME('now', '-10 seconds') WHERE id=?",
                    (int(job["id"]),),
                )
        finally:
            connection.close()

        allow_renew_interval.set()
        await asyncio.wait_for(renewed.wait(), timeout=1)
        await asyncio.wait_for(next_renew_sleep_started.wait(), timeout=1)

        competitor = JobManager(db_path)
        assert competitor.acquire_next_job(
            domain="chatbooks",
            queue="default",
            lease_seconds=30,
            worker_id="w-competitor",
        ) is None

        release_handler.set()
        await asyncio.wait_for(completion_succeeded.wait(), timeout=1)
        await asyncio.wait_for(renew_cancel_received.wait(), timeout=1)
        assert not run_task.done()

        allow_renew_cleanup.set()
        await asyncio.wait_for(run_task, timeout=1)
    finally:
        release_handler.set()
        allow_renew_interval.set()
        allow_renew_cleanup.set()
        if not run_task.done():
            run_task.cancel()
        await asyncio.gather(run_task, return_exceptions=True)

    assert worker_acquisitions == ["w-stop-active"]
    assert (jm.get_job(int(job["id"])) or {})["status"] == "completed"


@pytest.mark.asyncio
async def test_run_cancellation_check(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk3.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w3")
    assert acq is not None

    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w3")
    sdk = WorkerSDK(jm, cfg)

    finalize_calls = []

    def fake_finalize(job_id, **kwargs):
        finalize_calls.append({"job_id": job_id, **kwargs})
        sdk.stop()
        return True

    monkeypatch.setattr(jm, "acquire_next_job", lambda **kwargs: acq)
    monkeypatch.setattr(jm, "finalize_cancelled", fake_finalize)

    async def handler(job):
        pytest.fail("Handler should not run when cancel_check returns True")

    async def cancel_check(job):
        return True

    await asyncio.wait_for(
        sdk.run(handler=handler, cancel_check=cancel_check),
        timeout=1,
    )

    assert finalize_calls == [
        {
            "job_id": int(acq["id"]),
            "reason": "requested",
            "expected_uuid": acq["uuid"],
            "worker_id": "w3",
            "lease_id": acq["lease_id"],
        }
    ]


@pytest.mark.asyncio
async def test_run_success_completes_job(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_success.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")

    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w-success", lease_seconds=5, renew_threshold_seconds=1, renew_jitter_seconds=0)
    sdk = WorkerSDK(jm, cfg)

    # Capture completion and still allow real completion side effects
    calls = []
    orig_complete = jm.complete_job

    def spy_complete(job_id, **kwargs):

        calls.append({"job_id": job_id, **kwargs})
        return orig_complete(job_id, **kwargs)

    monkeypatch.setattr(jm, "complete_job", spy_complete)

    # Use stub sleep for fast loop exit
    _orig_sleep = asyncio.sleep
    sdk._sleep = DummySleep(_orig_sleep)

    async def handler(job_row):
        sdk.stop()
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler), timeout=1)

    assert calls and int(calls[0]["job_id"]) == int(job["id"])
    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "completed"


@pytest.mark.asyncio
async def test_run_calls_on_completed_after_durable_completion(tmp_path):
    db_path = tmp_path / "jobs_wsdk_post_complete.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="w-post-complete",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    observed: list[tuple[int, dict[str, object], str]] = []

    async def handler(_job_row):
        return {"ok": True}

    async def on_completed(job_row, result):
        stored = jm.get_job(int(job_row["id"])) or {}
        observed.append((int(job_row["id"]), result, str(stored.get("status"))))
        sdk.stop()

    await asyncio.wait_for(
        sdk.run(handler=handler, on_completed=on_completed),
        timeout=1,
    )

    assert observed == [(int(job["id"]), {"ok": True}, "completed")]


@pytest.mark.asyncio
async def test_run_calls_on_failed_only_after_durable_terminal_failure(tmp_path):
    manager, job, sdk = _failure_test_worker(tmp_path, "terminal-failure")
    observed: list[tuple[int, str, str]] = []

    async def handler(_job_row):
        raise _TerminalWorkerError("bounded failure")

    async def on_failed(job_row, exc):
        stored = manager.get_job(int(job_row["id"])) or {}
        observed.append(
            (int(job_row["id"]), type(exc).__name__, str(stored.get("status")))
        )
        sdk.stop()

    await asyncio.wait_for(
        sdk.run(handler=handler, on_failed=on_failed),
        timeout=1,
    )

    assert observed == [(int(job["id"]), "_TerminalWorkerError", "failed")]


@pytest.mark.asyncio
async def test_on_failed_uses_acquired_identity_snapshot_when_handler_mutates_row(
    tmp_path,
):
    _manager, job, sdk = _failure_test_worker(tmp_path, "mutated-row")
    observed_uuids: list[str] = []

    async def handler(job_row):
        job_row["uuid"] = "mutated"
        raise _TerminalWorkerError("bounded failure")

    async def on_failed(job_row, _exc):
        observed_uuids.append(str(job_row["uuid"]))
        sdk.stop()

    await asyncio.wait_for(
        sdk.run(handler=handler, on_failed=on_failed),
        timeout=1,
    )

    assert observed_uuids == [str(job["uuid"])]


@pytest.mark.asyncio
async def test_run_does_not_call_on_failed_when_retry_is_scheduled(
    monkeypatch,
    tmp_path,
):
    manager, job, sdk = _failure_test_worker(
        tmp_path,
        "retry-failure",
        max_retries=1,
    )
    observed: list[int] = []
    original_fail = manager.fail_job

    def stop_after_fail(*args, **kwargs):
        result = original_fail(*args, **kwargs)
        sdk.stop()
        return result

    monkeypatch.setattr(manager, "fail_job", stop_after_fail)

    async def handler(_job_row):
        raise _RetryableWorkerError("retry this")

    async def on_failed(job_row, _exc):
        observed.append(int(job_row["id"]))

    await asyncio.wait_for(
        sdk.run(handler=handler, on_failed=on_failed),
        timeout=1,
    )

    assert observed == []
    assert (manager.get_job(int(job["id"])) or {})["status"] == "queued"


@pytest.mark.asyncio
async def test_run_does_not_call_on_failed_when_terminalization_is_rejected(
    monkeypatch,
    tmp_path,
):
    _manager, _job, sdk = _failure_test_worker(tmp_path, "rejected-failure")
    observed: list[int] = []

    def reject_failure(*_args, **_kwargs):
        sdk.stop()
        return False

    monkeypatch.setattr(sdk.jm, "fail_job", reject_failure)

    async def handler(_job_row):
        raise _TerminalWorkerError("stale lease")

    async def on_failed(job_row, _exc):
        observed.append(int(job_row["id"]))

    await asyncio.wait_for(
        sdk.run(handler=handler, on_failed=on_failed),
        timeout=1,
    )

    assert observed == []


@pytest.mark.asyncio
async def test_on_failed_error_is_isolated_without_refinalizing_job(
    monkeypatch,
    tmp_path,
):
    manager, job, sdk = _failure_test_worker(tmp_path, "callback-error")
    fail_calls: list[int] = []
    original_fail = manager.fail_job

    def spy_fail(job_id, **kwargs):
        fail_calls.append(int(job_id))
        return original_fail(job_id, **kwargs)

    monkeypatch.setattr(manager, "fail_job", spy_fail)

    async def handler(_job_row):
        raise _TerminalWorkerError("bounded failure")

    async def on_failed(_job_row, _exc):
        sdk.stop()
        raise RuntimeError("audit sink unavailable")

    await asyncio.wait_for(
        sdk.run(handler=handler, on_failed=on_failed),
        timeout=1,
    )

    assert fail_calls == [int(job["id"])]
    assert (manager.get_job(int(job["id"])) or {})["status"] == "failed"


@pytest.mark.asyncio
async def test_on_failed_does_not_suppress_callback_cancellation(tmp_path):
    _manager, _job, sdk = _failure_test_worker(tmp_path, "callback-cancel")

    async def handler(_job_row):
        raise _TerminalWorkerError("bounded failure")

    async def on_failed(_job_row, _exc):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            sdk.run(handler=handler, on_failed=on_failed),
            timeout=1,
        )


@pytest.mark.asyncio
async def test_run_calls_rejection_callback_when_completion_cas_loses(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "jobs_wsdk_completion_rejected.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="w-completion-rejected",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    completed: list[int] = []
    rejected: list[tuple[int, dict[str, object]]] = []

    monkeypatch.setattr(jm, "complete_job", lambda *_args, **_kwargs: False)

    async def handler(_job_row):
        return {"ok": True}

    async def on_completed(job_row, _result):
        completed.append(int(job_row["id"]))

    async def on_completion_rejected(job_row, result):
        rejected.append((int(job_row["id"]), result))
        sdk.stop()

    await asyncio.wait_for(
        sdk.run(
            handler=handler,
            on_completed=on_completed,
            on_completion_rejected=on_completion_rejected,
        ),
        timeout=1,
    )

    assert completed == []
    assert rejected == [(int(job["id"]), {"ok": True})]


@pytest.mark.asyncio
async def test_post_completion_callback_error_does_not_refinalize_job(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "jobs_wsdk_post_complete_callback_error.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="w-post-complete-error",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    fail_calls: list[int] = []
    original_fail = jm.fail_job

    def spy_fail(job_id, **kwargs):
        fail_calls.append(int(job_id))
        return original_fail(job_id, **kwargs)

    monkeypatch.setattr(jm, "fail_job", spy_fail)

    async def handler(_job_row):
        return {"ok": True}

    async def on_completed(_job_row, _result):
        sdk.stop()
        raise RuntimeError("event sink unavailable")

    await asyncio.wait_for(
        sdk.run(handler=handler, on_completed=on_completed),
        timeout=1,
    )

    assert fail_calls == []
    assert (jm.get_job(int(job["id"])) or {})["status"] == "completed"


@pytest.mark.asyncio
async def test_stuck_completion_callback_does_not_block_later_jobs(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "jobs_wsdk_post_complete_callback_timeout.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    first = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    second = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="w-post-complete-timeout",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
            completion_callback_timeout_seconds=0.01,
            completion_callback_max_detached_tasks=1,
        ),
    )
    fail_calls: list[int] = []
    completion_callback_job_ids: list[int] = []
    completed_job_ids: list[int] = []
    callback_started = asyncio.Event()
    callback_cancelled = asyncio.Event()
    callback_finished = asyncio.Event()
    later_job_completed = asyncio.Event()
    release_callback = asyncio.Event()
    loop_errors: list[dict[str, object]] = []
    blocked_job_id: int | None = None
    original_fail = jm.fail_job
    original_complete = jm.complete_job
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    loop.set_exception_handler(
        lambda _loop, context: loop_errors.append(context)
    )

    def spy_fail(job_id, **kwargs):
        fail_calls.append(int(job_id))
        return original_fail(job_id, **kwargs)

    def spy_complete(job_id, **kwargs):
        completed = original_complete(job_id, **kwargs)
        if completed:
            completed_job_ids.append(int(job_id))
            if len(completed_job_ids) == 2:
                later_job_completed.set()
                sdk.stop()
        return completed

    monkeypatch.setattr(jm, "fail_job", spy_fail)
    monkeypatch.setattr(jm, "complete_job", spy_complete)

    async def handler(job_row):
        return {"job_id": int(job_row["id"])}

    async def on_completed(job_row, _result):
        nonlocal blocked_job_id
        completion_callback_job_ids.append(int(job_row["id"]))
        if blocked_job_id is None:
            blocked_job_id = int(job_row["id"])
            callback_started.set()
            try:
                await release_callback.wait()
            except asyncio.CancelledError:
                callback_cancelled.set()
                await release_callback.wait()
            finally:
                callback_finished.set()
            raise RuntimeError("late completion callback failure")

    run_task = asyncio.create_task(
        sdk.run(handler=handler, on_completed=on_completed)
    )
    later_job_waiter = asyncio.create_task(later_job_completed.wait())
    try:
        await asyncio.wait_for(callback_cancelled.wait(), timeout=1)
        done, _pending = await asyncio.wait({later_job_waiter}, timeout=0.5)

        assert later_job_waiter in done
        assert not release_callback.is_set()
        assert (jm.get_job(int(first["id"])) or {})["status"] == "completed"
        assert (jm.get_job(int(second["id"])) or {})["status"] == "completed"
    finally:
        release_callback.set()
        try:
            await asyncio.wait_for(callback_finished.wait(), timeout=1)
            await asyncio.wait_for(run_task, timeout=1)
            await asyncio.sleep(0)
            assert not sdk._detached_completion_callbacks
            gc.collect()
            await asyncio.sleep(0)
        finally:
            loop.set_exception_handler(previous_exception_handler)
            if not later_job_waiter.done():
                later_job_waiter.cancel()
            await asyncio.gather(later_job_waiter, return_exceptions=True)

    assert callback_started.is_set()
    assert completion_callback_job_ids == [blocked_job_id]
    assert not any(
        context.get("message") == "Task exception was never retrieved"
        for context in loop_errors
    )
    assert fail_calls == []
    assert (jm.get_job(int(first["id"])) or {})["status"] == "completed"
    assert (jm.get_job(int(second["id"])) or {})["status"] == "completed"


@pytest.mark.asyncio
async def test_rejection_callback_error_does_not_fail_unowned_job(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "jobs_wsdk_rejection_callback_error.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="t",
        payload={},
        owner_user_id="u",
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="chatbooks",
            queue="default",
            worker_id="w-rejection-error",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    fail_calls: list[int] = []
    original_fail = jm.fail_job

    monkeypatch.setattr(jm, "complete_job", lambda *_args, **_kwargs: False)

    def spy_fail(job_id, **kwargs):
        fail_calls.append(int(job_id))
        return original_fail(job_id, **kwargs)

    monkeypatch.setattr(jm, "fail_job", spy_fail)

    async def handler(_job_row):
        return {"ok": True}

    async def on_completion_rejected(_job_row, _result):
        sdk.stop()
        raise RuntimeError("reconciliation sink unavailable")

    await asyncio.wait_for(
        sdk.run(
            handler=handler,
            on_completion_rejected=on_completion_rejected,
        ),
        timeout=1,
    )

    assert fail_calls == []
    assert (jm.get_job(int(job["id"])) or {})["status"] == "processing"


@pytest.mark.asyncio
async def test_run_non_retryable_failure_marks_failed(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_fail.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")

    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w-fail", lease_seconds=5, renew_threshold_seconds=1, renew_jitter_seconds=0)
    sdk = WorkerSDK(jm, cfg)

    # Capture failure args and signal completion
    calls = []
    done = asyncio.Event()
    orig_fail = jm.fail_job

    def spy_fail(job_id, **kwargs):

        calls.append({"job_id": job_id, **kwargs})
        ok = orig_fail(job_id, **kwargs)
        done.set()
        return ok

    monkeypatch.setattr(jm, "fail_job", spy_fail)

    # Use stub sleep for fast loop exit
    _orig_sleep = asyncio.sleep
    sdk._sleep = DummySleep(_orig_sleep)

    class NonRetryableErr(Exception):
        retryable = False
        failure_code = "prototype_runtime_terminal"

    async def handler(job_row):
        raise NonRetryableErr("boom")

    task = asyncio.create_task(sdk.run(handler=handler))
    await asyncio.wait_for(done.wait(), timeout=1)
    sdk.stop()
    await asyncio.wait_for(task, timeout=1)

    assert any(c.get("retryable") is False for c in calls)
    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "failed"
    assert stored["error_code"] == "prototype_runtime_terminal"


@pytest.mark.asyncio
async def test_run_handler_cancelled_error_propagates(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_cancelled.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")

    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w-cancelled", lease_seconds=5, renew_threshold_seconds=1, renew_jitter_seconds=0)
    sdk = WorkerSDK(jm, cfg)

    fail_calls = []

    def spy_fail(job_id, **kwargs):
        fail_calls.append({"job_id": job_id, **kwargs})
        return True

    monkeypatch.setattr(jm, "fail_job", spy_fail)

    _orig_sleep = asyncio.sleep
    sdk._sleep = DummySleep(_orig_sleep)

    async def handler(_job_row):
        raise asyncio.CancelledError()

    task = asyncio.create_task(sdk.run(handler=handler))
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert fail_calls == []
    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "processing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("terminal_status", "error_code"),
    [("failed", "slides_render_failed"), ("cancelled", "slides_render_cancelled")],
)
async def test_run_terminal_outcome_uses_exact_terminalizer_without_complete_or_generic_fail(
    monkeypatch,
    tmp_path,
    terminal_status,
    error_code,
):
    db_path = tmp_path / f"jobs_wsdk_terminal_{terminal_status}.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=_slides_jobs_key("a" if terminal_status == "failed" else "b"),
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(
            domain="slides",
            queue="default",
            worker_id="slides-worker",
            lease_seconds=5,
            renew_threshold_seconds=1,
            renew_jitter_seconds=0,
        ),
    )
    sdk._sleep = DummySleep(asyncio.sleep)
    complete_calls = []
    fail_calls = []
    terminal_calls = []
    original_terminalize = jm.terminalize_job_from_worker

    monkeypatch.setattr(jm, "complete_job", lambda *args, **kwargs: complete_calls.append((args, kwargs)))
    monkeypatch.setattr(jm, "fail_job", lambda *args, **kwargs: fail_calls.append((args, kwargs)))

    def spy_terminalize(**kwargs):
        terminal_calls.append(kwargs)
        return original_terminalize(**kwargs)

    monkeypatch.setattr(jm, "terminalize_job_from_worker", spy_terminalize)

    async def handler(job_row):
        sdk.stop()
        return worker_sdk_module.WorkerTerminalOutcome(
            status=terminal_status,
            error_code=error_code,
            message="bounded worker-safe detail",
        )

    await asyncio.wait_for(sdk.run(handler=handler, job_type="presentation.generate"), timeout=1)

    assert complete_calls == []
    assert fail_calls == []
    assert len(terminal_calls) == 1
    terminal_call = terminal_calls[0]
    assert terminal_call["job_uuid"] == job["uuid"]
    assert terminal_call["owner_user_id"] == "owner-1"
    assert terminal_call["domain"] == "slides"
    assert terminal_call["queue"] == "default"
    assert terminal_call["job_type"] == "presentation.generate"
    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == terminal_status
    assert stored["error_code"] == error_code
    assert stored["error_message"] == "bounded worker-safe detail"


@pytest.mark.asyncio
async def test_terminal_outcome_accepts_exact_already_terminal_race(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_terminal_race.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=_slides_jobs_key("c"),
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(domain="slides", queue="default", worker_id="slides-worker"),
    )
    sdk._sleep = DummySleep(asyncio.sleep)
    terminal_calls = 0

    def race_winner(**_kwargs):
        nonlocal terminal_calls
        terminal_calls += 1
        return "ALREADY_TERMINAL"

    monkeypatch.setattr(jm, "terminalize_job_from_worker", race_winner)

    async def handler(_job_row):
        sdk.stop()
        return worker_sdk_module.WorkerTerminalOutcome(
            status="failed",
            error_code="slides_render_failed",
            message="safe",
        )

    await asyncio.wait_for(
        sdk.run(handler=handler, job_type="presentation.generate"),
        timeout=1,
    )
    assert terminal_calls == 1


@pytest.mark.asyncio
async def test_terminal_outcome_accepts_real_reconciler_cas_winner(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_reconciler_terminal_race.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=_slides_jobs_key("e"),
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(domain="slides", queue="default", worker_id="slides-worker"),
    )
    sdk._sleep = DummySleep(asyncio.sleep)
    original_terminalize = jm.terminalize_job_from_worker
    reconciler_results: list[str] = []

    def reconciler_wins_after_sdk_preread(**kwargs):
        reconciler_results.append(
            jm.terminalize_slides_generation_job_from_reconciler(
                job_uuid=kwargs["job_uuid"],
                owner_user_id=kwargs["owner_user_id"],
                expected_status="processing",
                status="failed",
                error_code="generation_expired",
                error_message="Generation input expired.",
                completion_token="reconciler:expiry:v1",
                job_id=kwargs["job_id"],
            )
        )
        return original_terminalize(**kwargs)

    monkeypatch.setattr(
        jm,
        "terminalize_job_from_worker",
        reconciler_wins_after_sdk_preread,
    )

    async def handler(_job_row):
        sdk.stop()
        return worker_sdk_module.WorkerTerminalOutcome(
            status="failed",
            error_code="slides_render_failed",
            message="bounded worker-safe detail",
        )

    await asyncio.wait_for(
        sdk.run(handler=handler, job_type="presentation.generate"),
        timeout=1,
    )

    assert reconciler_results == ["APPLIED"]
    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "failed"
    assert stored["error_code"] == "generation_expired"


@pytest.mark.asyncio
async def test_terminal_outcome_observes_uuid_authoritative_compressed_archive(
    monkeypatch,
    tmp_path,
):
    db_path = tmp_path / "jobs_wsdk_terminal_archive.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    job = jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": "receipt-archive"},
        owner_user_id="owner-1",
        idempotency_key=_slides_jobs_key("d"),
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(domain="slides", queue="default", worker_id="slides-worker"),
    )
    sdk._sleep = DummySleep(asyncio.sleep)
    terminal_calls = 0
    original_terminalize = jm.terminalize_job_from_worker

    def record_terminalize(**kwargs):
        nonlocal terminal_calls
        terminal_calls += 1
        return original_terminalize(**kwargs)

    monkeypatch.setattr(jm, "terminalize_job_from_worker", record_terminalize)
    monkeypatch.setenv("JOBS_ARCHIVE_BEFORE_DELETE", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS", "true")
    monkeypatch.setenv("JOBS_ARCHIVE_COMPRESS_DROP_JSON", "true")

    async def handler(job_row):
        lease_id = str(job_row["lease_id"])
        assert jm.fail_job(
            int(job_row["id"]),
            error="generic archived winner",
            retryable=False,
            worker_id="slides-worker",
            lease_id=lease_id,
            completion_token=lease_id,
            enforce=True,
            error_code="archived_terminal_winner",
            error_class="ArchivedTerminalWinner",
        )
        connection = jm._connect()
        try:
            with connection:
                connection.execute(
                    "UPDATE jobs SET completed_at='2000-01-01 00:00:00' WHERE id=?",
                    (int(job_row["id"]),),
                )
        finally:
            connection.close()
        assert (
            jm.prune_jobs(
                statuses=["failed"],
                older_than_days=0,
                domain="slides",
                queue="default",
                job_type="presentation.generate",
            )
            == 1
        )
        connection = jm._connect()
        try:
            with connection:
                connection.execute(
                    "UPDATE jobs_archive SET id=NULL WHERE uuid=?",
                    (str(job_row["uuid"]),),
                )
        finally:
            connection.close()
        sdk.stop()
        return worker_sdk_module.WorkerTerminalOutcome(
            status="failed",
            error_code="handler_terminal_outcome",
            message="safe",
        )

    await asyncio.wait_for(
        sdk.run(handler=handler, job_type="presentation.generate"),
        timeout=1,
    )
    assert terminal_calls == 0
    archived = jm.resolve_slides_generation_job(
        job_uuid=str(job["uuid"]),
        owner_user_id="owner-1",
        idempotency_key=_slides_jobs_key("d"),
    )
    assert archived is not None
    assert archived["archived"] is True
    assert archived["id"] is None
    assert archived["status"] == "failed"
    assert archived["payload"] == {"receipt_id": "receipt-archive"}


@pytest.mark.asyncio
async def test_terminal_cas_conflict_raises_dedicated_error_without_numeric_fallback(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk_terminal_conflict.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={},
        owner_user_id="owner-1",
        idempotency_key=_slides_jobs_key("e"),
    )
    sdk = WorkerSDK(
        jm,
        WorkerConfig(domain="slides", queue="default", worker_id="slides-worker"),
    )
    sdk._sleep = DummySleep(asyncio.sleep)
    complete_calls = []
    fail_calls = []
    monkeypatch.setattr(jm, "complete_job", lambda *args, **kwargs: complete_calls.append((args, kwargs)))
    monkeypatch.setattr(jm, "fail_job", lambda *args, **kwargs: fail_calls.append((args, kwargs)))
    monkeypatch.setattr(jm, "terminalize_job_from_worker", lambda **_kwargs: "CONFLICT")

    async def handler(_job_row):
        return worker_sdk_module.WorkerTerminalOutcome(
            status="failed",
            error_code="slides_render_failed",
            message="safe",
        )

    with pytest.raises(worker_sdk_module.WorkerTerminalizationConflict):
        await asyncio.wait_for(sdk.run(handler=handler, job_type="presentation.generate"), timeout=1)

    assert complete_calls == []
    assert fail_calls == []


@pytest.mark.parametrize("status", ["completed", "queued", "processing", "retry"])
def test_worker_terminal_outcome_rejects_open_ended_statuses(status):
    with pytest.raises(ValueError):
        worker_sdk_module.WorkerTerminalOutcome(
            status=status,
            error_code="slides_render_failed",
            message="safe",
        )


def test_worker_terminal_outcome_rejects_unbounded_or_unsafe_detail():
    with pytest.raises(ValueError):
        worker_sdk_module.WorkerTerminalOutcome(
            status="failed",
            error_code="slides_render_failed",
            message="x" * 1025,
        )
    with pytest.raises(ValueError):
        worker_sdk_module.WorkerTerminalOutcome(
            status="failed",
            error_code="bad code with spaces",
            message="safe",
        )
