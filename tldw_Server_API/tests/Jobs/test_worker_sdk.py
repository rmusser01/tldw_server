import asyncio
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.Jobs import worker_sdk as worker_sdk_module
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK


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


def _slides_jobs_key(character: str) -> str:
    return "slides:v1:" + character * 64


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
