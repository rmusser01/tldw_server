import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerSDK, WorkerConfig


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


@pytest.mark.asyncio
async def test_auto_renew_jitter_and_progress(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    j = jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")
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
    sdk.stop()
    try:
        await asyncio.wait_for(task, timeout=1)
    except asyncio.TimeoutError:
        task.cancel()
        raise

    # Verify sleep durations are lease - threshold (no jitter)
    assert any(abs(s - 15) < 0.1 for s in sleep_stub.calls)
    # Verify renew_job_lease received progress args
    assert any("progress_percent" in c and c.get("progress_percent") == 12.5 for c in calls)
    assert any("progress_message" in c and c.get("progress_message") == "tick" for c in calls)


@pytest.mark.asyncio
async def test_run_retryable_exception_and_backoff(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk2.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    j = jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")

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
    monkeypatch.setattr(jm, "acquire_next_job", lambda **kwargs: fake_acquire(**kwargs))
    monkeypatch.setattr(jm, "fail_job", lambda job_id, **kwargs: fake_fail(job_id, **kwargs))
    sdk._sleep = sleep_stub

    async def handler(job):
        raise RetryErr("boom")

    run_task = asyncio.create_task(sdk.run(handler=handler))
    # Allow a few loop iterations then stop
    await _orig_sleep(0)
    await _orig_sleep(0)
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
async def test_run_cancellation_check(monkeypatch, tmp_path):
    db_path = tmp_path / "jobs_wsdk3.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    j = jm.create_job(domain="chatbooks", queue="default", job_type="t", payload={}, owner_user_id="u")
    acq = jm.acquire_next_job(domain="chatbooks", queue="default", lease_seconds=10, worker_id="w3")
    assert acq is not None

    cfg = WorkerConfig(domain="chatbooks", queue="default", worker_id="w3")
    sdk = WorkerSDK(jm, cfg)

    cancel_called = {"count": 0}
    def fake_cancel(job_id, **kwargs):
        cancel_called["count"] += 1

    monkeypatch.setattr(jm, "acquire_next_job", lambda **kwargs: acq)
    monkeypatch.setattr(jm, "cancel_job", lambda job_id, **kwargs: fake_cancel(job_id, **kwargs))

    async def handler(job):
        pytest.fail("Handler should not run when cancel_check returns True")

    async def cancel_check(job):
        return True

    run_task = asyncio.create_task(sdk.run(handler=handler, cancel_check=cancel_check))
    await asyncio.sleep(0)
    sdk.stop()
    await asyncio.wait_for(run_task, timeout=1)

    assert cancel_called["count"] >= 1


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

    async def handler(job_row):
        raise NonRetryableErr("boom")

    task = asyncio.create_task(sdk.run(handler=handler))
    await asyncio.wait_for(done.wait(), timeout=1)
    sdk.stop()
    await asyncio.wait_for(task, timeout=1)

    assert any(c.get("retryable") is False for c in calls)
    stored = jm.get_job(int(job["id"]))
    assert stored["status"] == "failed"


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
