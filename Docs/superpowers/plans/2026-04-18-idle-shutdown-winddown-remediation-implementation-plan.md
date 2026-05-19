# Idle Shutdown Wind-Down Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce idle local `uvicorn` and Docker/container shutdown latency by removing the fixed Evaluations shutdown stall, explicitly stopping every owned in-process job poller at the right point in teardown, and emitting timing data that makes the remaining shutdown cost attributable.

**Architecture:** Execute this remediation in three narrow code batches. First make the Evaluations connection-pool maintenance thread wakeable so idle shutdown stops paying a synthetic five-second join tax. Then close the current shutdown-ownership gaps for helper-started job pollers and register all owned pollers in one inventory that `main.py` can quiesce immediately after the drain handoff on the zero-active path while preserving the existing bounded lease-wait branch when active processing jobs exist. Finally add lightweight timing helpers in `main.py` so the idle-shutdown path records the required segment durations and total teardown time without introducing a broader shutdown refactor.

**Tech Stack:** Python 3, FastAPI lifespan, asyncio, threading, Loguru, pytest, SQLite, Bandit, Markdown

---

## Current-Tree Evidence

- Captured idle shutdown trace from `2026-04-18 12:13:11.551` to `12:13:16.556` shows the Evaluations connection-pool shutdown alone consuming about five seconds before the rest of teardown resumes.
- `tldw_Server_API/app/core/Evaluations/connection_pool.py` currently runs the maintenance worker with `time.sleep(60)` and joins it with `join(timeout=5)`, which matches the observed fixed stall.
- `tldw_Server_API/app/main.py` applies the drain handoff first, then optionally waits for leases, but only reaches the per-worker stop block much later, which matches the repeated `Jobs acquire gate enabled; declining new acquisition` log spam during idle shutdown.
- `tldw_Server_API/app/main.py` starts helper-owned pollers that do not all have explicit teardown ownership today:
  - `audiobook_jobs_task` is started with a stop event but is not stopped in the current shutdown block.
  - `start_connectors_worker()` currently creates its own stop event internally and returns only a task.
  - `start_reminder_jobs_worker()` returns only a task and is cancelled late instead of being stop-signaled.
  - `start_admin_backup_jobs_worker()` wraps `WorkerSDK` but exposes no stop signal.
  - `start_admin_byok_validation_jobs_worker()` wraps `WorkerSDK` and is started in `main.py`, but currently has no matching shutdown path in `main.py`.
- The app already stores shutdown coordinator artifacts on `app.state`, so exposing a job-poller inventory and shutdown timing summary there is consistent with current lifecycle observability patterns and avoids brittle full-log assertions in tests.

## Initial Poller Inventory To Preserve Or Fix

- Explicit stop-event pollers already started in `main.py` and expected to remain explicitly owned:
  - `core_jobs_task`
  - `files_jobs_task`
  - `data_tables_jobs_task`
  - `prompt_studio_jobs_task`
  - `privilege_snapshot_task`
  - `audio_jobs_task`
  - `audiobook_jobs_task`
  - `presentation_render_jobs_task`
  - `media_ingest_jobs_task`
  - `media_ingest_heavy_jobs_task`
  - `reading_digest_jobs_task`
  - `study_pack_jobs_task`
  - `study_suggestions_jobs_task`
  - `companion_reflection_jobs_task`
  - `evals_abtest_jobs_task`
- Helper-started or currently underspecified pollers that must gain explicit ownership in this slice:
  - `reminder_jobs_task`
  - `admin_backup_jobs_task`
  - `admin_byok_validation_jobs_task`
  - connectors worker task currently stored as `_conn_task`
- Non-poller background tasks, schedulers, and the larger shutdown coordinator remain out of scope except where the timing logs need to wrap their existing teardown blocks.

## Implementation File Map

**Create:**
- `tldw_Server_API/tests/Evaluations/unit/test_connection_pool_shutdown.py`: Focused unit coverage for wakeable maintenance-thread shutdown and the shorter defensive join fallback.
- `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`: Focused lifecycle tests for job-poller inventory capture, zero-active versus active-processing sequencing, and required timing-segment recording.

**Modify:**
- `tldw_Server_API/app/core/Evaluations/connection_pool.py`: Replace the fixed-sleep maintenance loop with a shutdown-wakeable wait and reduce the defensive join fallback.
- `tldw_Server_API/app/main.py`: Build and expose an owned job-poller inventory, stop pollers immediately after the drain handoff on the zero-active path, preserve the bounded lease-wait branch for active processing jobs, remove duplicated late stop handling for moved pollers, and emit shutdown timing summaries.
- `tldw_Server_API/app/services/connectors_worker.py`: Accept a caller-supplied stop event so `main.py` owns shutdown for the helper-started connectors poller.
- `tldw_Server_API/app/services/reminder_jobs_worker.py`: Accept a caller-supplied stop event so `main.py` can stop the reminder worker cleanly instead of relying on late cancellation.
- `tldw_Server_API/app/services/admin_backup_jobs_worker.py`: Add stop-event-aware `WorkerSDK` shutdown wiring and expose that stop path to `main.py`.
- `tldw_Server_API/app/services/admin_byok_validation_jobs_worker.py`: Add stop-event-aware `WorkerSDK` shutdown wiring and expose that stop path to `main.py`.
- `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`: Keep the helper start-function truthiness coverage aligned with the new optional `stop_event` signature.

## Implementation Notes

- Keep this slice narrow. Do not move the broader legacy teardown tail into the shutdown coordinator and do not redesign busy-shutdown semantics beyond preserving the existing `JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC` contract.
- Treat the numeric target as a manual acceptance gate, not a brittle CI runtime assertion. Automated tests should assert ordering, ownership, and bounded behavior rather than wall-clock totals across machines.
- Use `time.monotonic()` for shutdown timing segments. Record each segment to `app.state._tldw_shutdown_timing_segments` as a list of dictionaries with at least `segment` and `duration_ms`.
- Use a consistent timing log format so grep-friendly manual validation is possible:
  - `App Shutdown Timing: segment=<segment_name> duration_ms=<ms> ...`
  - `App Shutdown Timing: total duration_ms=<ms> slowest_segment=<segment_name> slowest_duration_ms=<ms>`
- Record the owned poller inventory on `app.state._tldw_shutdown_job_poller_inventory` as dictionaries with at least `name`, `task_name`, `has_stop_event`, and `timeout_sec`. Tests should assert against this state instead of relying on log text.
- For the optional lease-wait timing segment, record a zero-duration or skipped entry when no wait is applied so the required segment set is always visible in the summary.
- Keep Evaluations shutdown synchronous in this slice. Only move it behind `asyncio.to_thread(...)` in a follow-up if the new timing summary still shows a material blocking cost after the wakeable maintenance change lands.

### Task 1: Make Evaluations Maintenance Shutdown Wakeable

**Files:**
- Create: `tldw_Server_API/tests/Evaluations/unit/test_connection_pool_shutdown.py`
- Modify: `tldw_Server_API/app/core/Evaluations/connection_pool.py`

- [ ] **Step 1: Write the failing Evaluations shutdown tests**

```python
import threading
import time

from tldw_Server_API.app.core.Evaluations.connection_pool import ConnectionPool


def test_shutdown_sets_maintenance_wakeup_before_join(temp_db_path):
    pool = ConnectionPool(str(temp_db_path), enable_monitoring=False)
    pool._maintenance_shutdown_event = threading.Event()

    observed: dict[str, object] = {}

    class _FakeThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            observed["timeout"] = timeout
            observed["event_is_set"] = pool._maintenance_shutdown_event.is_set()

    pool._maintenance_task = _FakeThread()

    pool.shutdown()

    assert observed == {"timeout": 1.0, "event_is_set": True}


def test_shutdown_no_longer_pays_fixed_five_second_join(monkeypatch, temp_db_path):
    pool = ConnectionPool(str(temp_db_path), enable_monitoring=False)
    pool._maintenance_shutdown_event = threading.Event()

    class _FastExitThread:
        def is_alive(self) -> bool:
            return True

        def join(self, timeout: float | None = None) -> None:
            return None

    pool._maintenance_task = _FastExitThread()

    started = time.perf_counter()
    pool.shutdown()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.25
```

- [ ] **Step 2: Run the new shutdown tests to confirm the current implementation fails**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_connection_pool_shutdown.py
```

Expected: FAIL because `ConnectionPool` does not yet create a maintenance shutdown event and still joins the maintenance thread with the existing five-second fallback.

- [ ] **Step 3: Implement the wakeable maintenance shutdown**

Update `tldw_Server_API/app/core/Evaluations/connection_pool.py` so the maintenance worker waits on an event instead of sleeping blindly:

```python
import threading


class ConnectionPool:
    def __init__(...):
        ...
        self._maintenance_shutdown_event = threading.Event()

    def _start_maintenance(self):
        if self._maintenance_task is not None:
            return

        def maintenance_worker() -> None:
            while not self._shutdown:
                try:
                    self._perform_maintenance()
                    if self._maintenance_shutdown_event.wait(timeout=60.0):
                        return
                except Exception as exc:
                    logger.error(f"Pool maintenance error: {exc}")
                    if self._maintenance_shutdown_event.wait(timeout=30.0):
                        return

        self._maintenance_task = threading.Thread(target=maintenance_worker, daemon=True)
        self._maintenance_task.start()

    def shutdown(self):
        logger.info("Shutting down connection pool")
        self._shutdown = True
        self._maintenance_shutdown_event.set()
        ...
        if self._maintenance_task and self._maintenance_task.is_alive():
            self._maintenance_task.join(timeout=1.0)
```

Keep the join fallback bounded and defensive. Do not remove the join entirely.

- [ ] **Step 4: Re-run the focused Evaluations shutdown tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_connection_pool_shutdown.py tldw_Server_API/tests/Evaluations/unit/test_connection_pool.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/core/Evaluations/connection_pool.py tldw_Server_API/tests/Evaluations/unit/test_connection_pool_shutdown.py
git commit -m "fix: wake evaluations pool shutdown promptly"
```

### Task 2: Give Helper-Started Job Pollers Explicit Stop Ownership

**Files:**
- Modify: `tldw_Server_API/app/services/connectors_worker.py`
- Modify: `tldw_Server_API/app/services/reminder_jobs_worker.py`
- Modify: `tldw_Server_API/app/services/admin_backup_jobs_worker.py`
- Modify: `tldw_Server_API/app/services/admin_byok_validation_jobs_worker.py`
- Modify: `tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py`
- Test: `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`

- [ ] **Step 1: Write the failing helper-start ownership tests**

Add focused tests in `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py` for helper-started workers:

```python
import asyncio

import pytest

from tldw_Server_API.app.services import connectors_worker
from tldw_Server_API.app.services import reminder_jobs_worker
from tldw_Server_API.app.services import admin_backup_jobs_worker
from tldw_Server_API.app.services import admin_byok_validation_jobs_worker


@pytest.mark.asyncio
async def test_start_connectors_worker_uses_caller_stop_event(monkeypatch):
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def _fake_run(stop_event: asyncio.Event | None = None) -> None:
        started.set()
        await stop_event.wait()
        stopped.set()

    monkeypatch.setenv("CONNECTORS_WORKER_ENABLED", "true")
    monkeypatch.setattr(connectors_worker, "run_connectors_worker", _fake_run)

    stop_event = asyncio.Event()
    task = await connectors_worker.start_connectors_worker(stop_event=stop_event)
    await asyncio.wait_for(started.wait(), timeout=1.0)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert stopped.is_set()


@pytest.mark.asyncio
async def test_admin_backup_worker_stops_worker_sdk_when_stop_event_is_set(monkeypatch):
    stop_called = asyncio.Event()

    class _FakeSDK:
        def __init__(self, *_args, **_kwargs) -> None:
            self._stopped = asyncio.Event()

        def stop(self) -> None:
            stop_called.set()
            self._stopped.set()

        async def run(self, **_kwargs) -> None:
            await self._stopped.wait()

    monkeypatch.setenv("ADMIN_BACKUP_JOBS_WORKER_ENABLED", "true")
    monkeypatch.setattr(admin_backup_jobs_worker, "WorkerSDK", _FakeSDK)

    stop_event = asyncio.Event()
    task = await admin_backup_jobs_worker.start_admin_backup_jobs_worker(stop_event=stop_event)
    stop_event.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert stop_called.is_set()
```

- [ ] **Step 2: Run the helper-start ownership tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py::test_start_connectors_worker_uses_caller_stop_event tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py::test_admin_backup_worker_stops_worker_sdk_when_stop_event_is_set tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py
```

Expected: FAIL because the helper start functions do not yet accept a caller-supplied stop event and the `WorkerSDK`-based helpers do not yet wire `stop_event` to `sdk.stop()`.

- [ ] **Step 3: Implement caller-owned stop events for helper-started pollers**

Thread a caller-supplied `stop_event` through the helper start functions. For simple loop-based workers:

```python
async def start_reminder_jobs_worker(stop_event: asyncio.Event | None = None) -> asyncio.Task | None:
    if not env_flag_enabled("REMINDER_JOBS_WORKER_ENABLED"):
        return None
    managed_stop_event = stop_event or asyncio.Event()
    return asyncio.create_task(
        run_reminder_jobs_worker(managed_stop_event),
        name="reminder_jobs_worker",
    )
```

Apply the same pattern to `start_connectors_worker()`.

For `WorkerSDK`-based helpers, add a stop watcher inside the run wrapper:

```python
async def run_admin_backup_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    sdk = WorkerSDK(jm, cfg)
    stop_watcher = None
    if stop_event is not None:
        async def _watch_for_stop() -> None:
            await stop_event.wait()
            sdk.stop()

        stop_watcher = asyncio.create_task(_watch_for_stop(), name="admin_backup_jobs_worker_stop_watch")

    try:
        await sdk.run(handler=handle_backup_schedule_job)
    finally:
        if stop_watcher is not None:
            stop_watcher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_watcher
```

Mirror the same `stop_event` pattern in `admin_byok_validation_jobs_worker.py`.

Keep the existing no-argument call contract intact so current startup truthiness tests and any external callers remain valid.

- [ ] **Step 4: Re-run the helper-start ownership tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/services/connectors_worker.py tldw_Server_API/app/services/reminder_jobs_worker.py tldw_Server_API/app/services/admin_backup_jobs_worker.py tldw_Server_API/app/services/admin_byok_validation_jobs_worker.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py
git commit -m "fix: expose explicit stop ownership for jobs pollers"
```

### Task 3: Reorder Idle Shutdown Poller Quiesce And Add Timed Segments

**Files:**
- Modify: `tldw_Server_API/app/main.py`
- Test: `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`

- [ ] **Step 1: Write the failing shutdown sequencing and timing tests**

Add focused helper-level tests in `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`:

```python
import asyncio
from fastapi import FastAPI

from tldw_Server_API.app import main as main_module


@pytest.mark.asyncio
async def test_zero_active_processing_quiesces_job_pollers_without_lease_wait(monkeypatch):
    app = FastAPI()
    handles = [
        main_module._ManagedJobPoller(
            name="audiobook_jobs_task",
            task=asyncio.create_task(asyncio.sleep(3600)),
            stop_event=asyncio.Event(),
        )
    ]
    stop_calls: list[str] = []

    async def _fake_stop_pollers(_app, _handles):
        stop_calls.append("stop")

    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", _fake_stop_pollers)

    await main_module._quiesce_owned_job_pollers_for_shutdown(
        app,
        handles,
        wait_for_leases_sec=30,
        count_active_processing=lambda: 0,
    )

    assert stop_calls == ["stop"]
    segments = getattr(app.state, "_tldw_shutdown_timing_segments")
    assert [entry["segment"] for entry in segments][:2] == ["optional_lease_wait", "job_poller_quiesce"]
    assert segments[0]["duration_ms"] == 0


@pytest.mark.asyncio
async def test_active_processing_preserves_bounded_lease_wait_before_quiesce(monkeypatch):
    app = FastAPI()
    handles: list[main_module._ManagedJobPoller] = []
    counts = iter([2, 1, 0])
    observed_sleeps: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        observed_sleeps.append(delay)

    monkeypatch.setattr(main_module._asyncio, "sleep", _fake_sleep)
    monkeypatch.setattr(main_module, "_stop_registered_job_pollers", lambda *_args, **_kwargs: asyncio.sleep(0))

    await main_module._quiesce_owned_job_pollers_for_shutdown(
        app,
        handles,
        wait_for_leases_sec=5,
        count_active_processing=lambda: next(counts),
    )

    assert observed_sleeps != []
```

- [ ] **Step 2: Run the sequencing tests to confirm the current lifecycle path does not satisfy them**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py
```

Expected: FAIL because `main.py` does not yet expose a unified owned-poller inventory, does not quiesce pollers immediately after the zero-active drain handoff, and does not yet record the required timing segments.

- [ ] **Step 3: Implement the owned-poller inventory, early quiesce helper, and timing logs**

Add narrow lifecycle helpers in `tldw_Server_API/app/main.py`:

```python
from contextlib import contextmanager
from dataclasses import dataclass
import time


@dataclass
class _ManagedJobPoller:
    name: str
    task: asyncio.Task
    stop_event: asyncio.Event | None = None
    timeout_sec: float = 5.0


def _record_shutdown_timing_segment(app: FastAPI, segment: str, duration_ms: int, **extra: object) -> None:
    segments = getattr(app.state, "_tldw_shutdown_timing_segments", None)
    if not isinstance(segments, list):
        segments = []
        app.state._tldw_shutdown_timing_segments = segments
    payload = {"segment": segment, "duration_ms": duration_ms, **extra}
    segments.append(payload)
    logger.info("App Shutdown Timing: segment={} duration_ms={} {}", segment, duration_ms, extra)


@contextmanager
def _timed_shutdown_segment(app: FastAPI, segment: str, **extra: object):
    started = time.monotonic()
    try:
        yield
    finally:
        duration_ms = int((time.monotonic() - started) * 1000)
        _record_shutdown_timing_segment(app, segment, duration_ms, **extra)


async def _stop_registered_job_pollers(app: FastAPI, handles: list[_ManagedJobPoller]) -> None:
    for handle in handles:
        if handle.stop_event is not None:
            handle.stop_event.set()
        try:
            await asyncio.wait_for(handle.task, timeout=handle.timeout_sec)
        except asyncio.TimeoutError:
            handle.task.cancel()
```

Then wire startup registration in `main.py` so every owned in-process poller is appended to one `owned_job_pollers` list and mirrored onto `app.state._tldw_shutdown_job_poller_inventory`. This registration must include the previously missed or underspecified pollers:

```python
owned_job_pollers: list[_ManagedJobPoller] = []

if _enabled:
    audiobook_jobs_stop_event = _asyncio.Event()
    audiobook_jobs_task = _asyncio.create_task(_run_audiobook_jobs(audiobook_jobs_stop_event))
    owned_job_pollers.append(
        _ManagedJobPoller(
            name="audiobook_jobs_task",
            task=audiobook_jobs_task,
            stop_event=audiobook_jobs_stop_event,
        )
    )

reminder_jobs_stop_event = _asyncio.Event()
reminder_jobs_task = await start_reminder_jobs_worker(reminder_jobs_stop_event)
...
admin_backup_jobs_stop_event = _asyncio.Event()
admin_backup_jobs_task = await start_admin_backup_jobs_worker(admin_backup_jobs_stop_event)
...
admin_byok_validation_jobs_stop_event = _asyncio.Event()
admin_byok_validation_jobs_task = await start_admin_byok_validation_jobs_worker(admin_byok_validation_jobs_stop_event)
...
connectors_jobs_stop_event = _asyncio.Event()
connectors_jobs_task = await start_connectors_worker(connectors_jobs_stop_event)
```

Move poller quiesce to immediately after the drain handoff:

```python
with _timed_shutdown_segment(app, "transition_handoff"):
    ...

await _quiesce_owned_job_pollers_for_shutdown(
    app,
    owned_job_pollers,
    wait_for_leases_sec=_max_wait,
    count_active_processing=jm_chk.count_active_processing,
)
```

Required behavior of `_quiesce_owned_job_pollers_for_shutdown(...)`:

- Record the `optional_lease_wait` segment every time.
- If `wait_for_leases_sec <= 0`, or if `count_active_processing()` is already `0`, do not sleep; record a zero-duration or skipped lease-wait segment and immediately stop pollers.
- If active processing jobs exist and `wait_for_leases_sec > 0`, preserve the existing bounded 0.5-second polling lease-wait loop before stopping pollers.
- Record `job_poller_quiesce` around the actual stop phase.

After the early quiesce path is in place, delete or guard the duplicated late-stop branches for those moved pollers so shutdown does not double-await the same task later in teardown.

Wrap the remaining required blocks with timing helpers:

- `evaluations_pool_shutdown`
- `unified_audit_and_executor_shutdown`
- `telemetry_shutdown`
- total app teardown

Store the final total summary on `app.state._tldw_shutdown_timing_total` with at least `duration_ms`, `slowest_segment`, and `slowest_duration_ms`.

- [ ] **Step 4: Re-run the shutdown sequencing tests**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/main.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py
git commit -m "fix: quiesce idle jobs pollers earlier on shutdown"
```

### Task 4: Final Verification And Manual Acceptance

**Files:**
- Modify only if verification exposes a real defect. Do not widen scope for style-only cleanups.

- [ ] **Step 1: Run the focused automated regression suite**

Run:
```bash
source .venv/bin/activate && python -m pytest -v tldw_Server_API/tests/Evaluations/unit/test_connection_pool_shutdown.py tldw_Server_API/tests/Evaluations/unit/test_connection_pool.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py tldw_Server_API/tests/Services/test_service_startup_truthiness_batch2.py
```

Expected: PASS

- [ ] **Step 2: Run Bandit on the touched Python scope**

Run:
```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/main.py tldw_Server_API/app/core/Evaluations/connection_pool.py tldw_Server_API/app/services/connectors_worker.py tldw_Server_API/app/services/reminder_jobs_worker.py tldw_Server_API/app/services/admin_backup_jobs_worker.py tldw_Server_API/app/services/admin_byok_validation_jobs_worker.py -f json -o /tmp/bandit_idle_shutdown_winddown.json
```

Expected: `No issues identified` in the CLI summary or only pre-existing accepted findings outside the touched logic.

- [ ] **Step 3: Manual idle `uvicorn` acceptance**

Run the same local path already reproducing the issue:

```bash
source .venv/bin/activate && python -m uvicorn tldw_Server_API.app.main:app
```

Then trigger shutdown with `Ctrl-C` while the server is idle and confirm all of the following in the logs:

- `App Shutdown Timing: total duration_ms=<ms>` is `<= 8000`
- `App Shutdown Timing: segment=evaluations_pool_shutdown duration_ms=<ms>` is `<= 1000`
- `App Shutdown Timing: segment=optional_lease_wait duration_ms=0` or equivalent skipped metadata appears on the zero-active path
- repeated `Jobs acquire gate enabled; declining new acquisition` log spam does not continue once shutdown begins
- the owned poller inventory on `app.state._tldw_shutdown_job_poller_inventory` includes the previously missing `audiobook`, `connectors`, `admin_backup`, and `admin_byok_validation` workers when they are enabled

- [ ] **Step 4: Manual idle Docker/container-stop acceptance**

Use the same local container launch path already reproducing the issue, then stop that container while the server is idle:

```bash
docker stop <your-local-tldw-container-name>
```

Confirm the container logs show the same acceptance criteria as the local `uvicorn` path and that total idle shutdown also completes within `<= 8000 ms` from the first shutdown transition log to `Application shutdown complete.`

- [ ] **Step 5: Commit only if verification required a real code change**

```bash
git add <verification-fix-paths>
git commit -m "fix: address idle shutdown verification regressions"
```
