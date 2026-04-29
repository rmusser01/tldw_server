# Worker Lifecycle Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract shared lifecycle ownership helpers from `tldw_Server_API/app/main.py` and migrate the first Jobs-domain stop-event worker batch into the shared background-worker shutdown path.

**Architecture:** Add a neutral lifecycle exception module and a focused lifecycle worker module that owns `ManagedWorker`, phased app-state inventory publication, task startup, and bounded stop behavior. Keep `main.py` as the policy owner for environment flags and startup ordering, while using compatibility wrappers so current job-poller tests and app-state inventory consumers keep working during migration. Use `job_poller_quiesce` only for existing early Jobs acquire-gate workers and `background_worker_shutdown` for the first migrated metrics/crypto/integrity batch.

**Tech Stack:** Python 3, FastAPI lifespan, asyncio tasks/events, pytest, TestClient, Bandit.

---

## File Structure

- Create `tldw_Server_API/app/services/lifecycle_exceptions.py`
  - Owns shared lifecycle guard exception tuples.
  - Must not import from `tldw_Server_API.app.main`.

- Create `tldw_Server_API/app/services/lifecycle_workers.py`
  - Owns `ShutdownPhase`, `ManagedWorker`, `WorkerInventory`, `start_stop_event_worker`, and `stop_registered_workers`.
  - Publishes full inventory to `app.state._tldw_shutdown_worker_inventory`.
  - Publishes filtered job-poller compatibility inventory to `app.state._tldw_shutdown_job_poller_inventory`.

- Modify `tldw_Server_API/app/main.py`
  - Imports lifecycle guard exceptions and worker helpers.
  - Keeps private compatibility aliases and wrappers for existing tests.
  - Registers the first background-worker batch through `start_stop_event_worker`.
  - Stops background-phase workers before their legacy direct shutdown branches.

- Create `tldw_Server_API/tests/Services/test_lifecycle_workers.py`
  - Unit tests for inventory publication, task naming/startup, concurrent shutdown, timeout cancellation, and stopped-name recording.

- Modify `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`
  - Keep current job-poller compatibility assertions.
  - Add full worker inventory assertions for the migrated background-worker batch.
  - Add regression coverage proving background-phase workers do not enter the early job-poller quiesce path.

---

### Task 1: Extract Lifecycle Modules

**Files:**
- Create: `tldw_Server_API/app/services/lifecycle_exceptions.py`
- Create: `tldw_Server_API/app/services/lifecycle_workers.py`
- Create: `tldw_Server_API/tests/Services/test_lifecycle_workers.py`

- [ ] **Step 1: Write failing inventory publication tests**

Add tests that create one job-poller worker and one background worker, publish them, and assert:

```python
assert app.state._tldw_shutdown_worker_inventory == [
    {
        "name": "job_worker",
        "task_name": "job-task",
        "has_stop_event": True,
        "timeout_sec": 5.0,
        "category": "jobs",
        "shutdown_phase": "job_poller_quiesce",
    },
    {
        "name": "background_worker",
        "task_name": "background-task",
        "has_stop_event": True,
        "timeout_sec": 2.0,
        "category": "jobs",
        "shutdown_phase": "background_worker_shutdown",
    },
]
assert app.state._tldw_shutdown_job_poller_inventory == [
    {
        "name": "job_worker",
        "task_name": "job-task",
        "has_stop_event": True,
        "timeout_sec": 5.0,
    }
]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_lifecycle_workers.py::test_worker_inventory_publishes_full_and_filtered_views -v
```

Expected: FAIL because `lifecycle_workers.py` does not exist.

- [ ] **Step 3: Write failing startup and shutdown behavior tests**

Add tests for:

- `start_stop_event_worker(...)` creates an `asyncio.Event`, names the task with the supplied stable task name, registers the worker, and returns `(task, stop_event)`.
- `stop_registered_workers(...)` sets stop events, waits concurrently, records stopped names to the supplied app-state attribute, and cancels a stubborn worker after timeout without blocking cooperative workers.

- [ ] **Step 4: Implement `lifecycle_exceptions.py`**

Add:

```python
from __future__ import annotations

LIFECYCLE_GUARD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)
```

Do not include `asyncio.CancelledError`.

- [ ] **Step 5: Implement `lifecycle_workers.py` minimally**

Implement:

```python
class ShutdownPhase(str, Enum):
    JOB_POLLER_QUIESCE = "job_poller_quiesce"
    BACKGROUND_WORKER_SHUTDOWN = "background_worker_shutdown"


@dataclass
class ManagedWorker:
    name: str
    task: asyncio.Task[Any]
    stop_event: asyncio.Event | None = None
    timeout_sec: float = 5.0
    category: str | None = None
    shutdown_phase: ShutdownPhase = ShutdownPhase.JOB_POLLER_QUIESCE
```

`WorkerInventory` should expose `handles`, `register(...)`, `replace_phase(...)`, `handles_for_phase(...)`, and `publish()` methods. Also provide small module-level helpers if they make `main.py` compatibility wrappers thinner:

```python
def publish_worker_inventory(app: Any, handles: Sequence[ManagedWorker]) -> None: ...
async def stop_registered_workers(app: Any, handles: Sequence[ManagedWorker], *, stopped_names_attr: str, log_label: str) -> None: ...
async def start_stop_event_worker(inventory: WorkerInventory, *, name: str, task_name: str, coroutine_factory: Callable[[asyncio.Event], Awaitable[Any]], timeout_sec: float = 5.0, category: str | None = None, shutdown_phase: ShutdownPhase = ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN) -> tuple[asyncio.Task[Any], asyncio.Event]: ...
```

- [ ] **Step 6: Run lifecycle helper tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_lifecycle_workers.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add tldw_Server_API/app/services/lifecycle_exceptions.py tldw_Server_API/app/services/lifecycle_workers.py tldw_Server_API/tests/Services/test_lifecycle_workers.py
git commit -m "feat: add lifecycle worker primitives"
```

---

### Task 2: Wire Main Compatibility Wrappers To Lifecycle Helpers

**Files:**
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`

- [ ] **Step 1: Write failing compatibility assertions**

Extend `test_publish_shutdown_job_poller_inventory_captures_registered_metadata` or add a nearby unit test proving:

- `_ManagedJobPoller(...)` still defaults to `job_poller_quiesce`.
- `_publish_shutdown_job_poller_inventory(app, handles)` still writes the old filtered inventory shape.
- `_publish_shutdown_job_poller_inventory(app, handles)` also writes `_tldw_shutdown_worker_inventory` with `category` and `shutdown_phase`.

- [ ] **Step 2: Run the focused test**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py::test_publish_shutdown_job_poller_inventory_captures_registered_metadata -v
```

Expected: FAIL on the new full-inventory assertion.

- [ ] **Step 3: Import lifecycle helpers in `main.py`**

Near the existing imports and private guard definitions, import:

```python
from tldw_Server_API.app.services.lifecycle_exceptions import LIFECYCLE_GUARD_EXCEPTIONS
from tldw_Server_API.app.services.lifecycle_workers import (
    ManagedWorker,
    ShutdownPhase as WorkerShutdownPhase,
    publish_worker_inventory,
    stop_registered_workers,
)
```

Set `_STARTUP_GUARD_EXCEPTIONS = LIFECYCLE_GUARD_EXCEPTIONS` so the existing private name remains stable.

- [ ] **Step 4: Replace private worker mechanics with compatibility wrappers**

Replace `_ManagedJobPoller` with an alias or subclass compatible with existing tests:

```python
_ManagedJobPoller = ManagedWorker
```

Then update:

- `_publish_shutdown_job_poller_inventory(...)` to delegate to `publish_worker_inventory(...)`
- `_register_owned_job_poller(...)` to append a `ManagedWorker(..., shutdown_phase=WorkerShutdownPhase.JOB_POLLER_QUIESCE)`
- `_replace_owned_job_poller_inventory(...)` to replace only `WorkerShutdownPhase.JOB_POLLER_QUIESCE` handles, preserving any already-registered non-job phases, then publish both inventory views
- `_stop_registered_job_pollers(...)` to filter `handles` to `WorkerShutdownPhase.JOB_POLLER_QUIESCE` and call `stop_registered_workers(..., stopped_names_attr="_tldw_shutdown_quiesced_job_poller_names", log_label="job poller")`

Use the `WorkerShutdownPhase` alias consistently in `main.py`. Do not import lifecycle `ShutdownPhase` under the bare name because the lifespan shutdown block already imports `ShutdownPhase` from `shutdown_coordinator`, which would make `ShutdownPhase` a local name and break earlier startup references.

Do not import `WorkerInventory` or `start_stop_event_worker` in Task 2. They are not used until Task 3, and importing them early creates lint failures for unused imports.

- [ ] **Step 5: Run current shutdown job-poller tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/main.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py
git commit -m "refactor: wire main lifecycle wrappers"
```

---

### Task 3: Migrate First Jobs-Domain Background Worker Batch

**Files:**
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`

- [ ] **Step 1: Write failing full-inventory integration test**

Add a test near `test_lifespan_shutdown_stops_jobs_metrics_reconcile_worker` that:

- enables `JOBS_METRICS_GAUGES_ENABLED`, `JOBS_METRICS_RECONCILE_ENABLE`, `JOBS_CRYPTO_ROTATE_SERVICE_ENABLED`, and `JOBS_INTEGRITY_SWEEP_ENABLED`
- patches `jobs_metrics_service.run_jobs_metrics_gauges`, `jobs_metrics_service.run_jobs_metrics_reconcile`, `jobs_crypto_rotate_service.run_jobs_crypto_rotate`, and `jobs_integrity_service.run_jobs_integrity_sweeper` with fake coroutines that wait for their stop event
- opens `TestClient(app)`
- asserts `_tldw_shutdown_worker_inventory` contains `jobs_metrics_task`, `jobs_metrics_reconcile_task`, `jobs_crypto_rotate_task`, and `jobs_integrity_task` with `shutdown_phase == "background_worker_shutdown"`
- asserts `_tldw_shutdown_job_poller_inventory` does not contain those four names

- [ ] **Step 2: Write failing shutdown phase regression test**

Add a test proving background workers are not stopped by early job-poller quiesce:

- monkeypatch `main_module._stop_registered_job_pollers` with a spy that records names it receives
- enable and patch the four background workers
- open and close `TestClient(app)`
- assert the spy did not receive any of the four background worker names
- assert all four fake worker stop events were set by shutdown
- assert each fake worker observed exactly one stop signal or completion path, so idempotent stop events cannot hide duplicate ownership by legacy late-stop branches

- [ ] **Step 3: Run new tests to verify failure**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py -k "background_worker_inventory or background_workers_do_not_enter_job_poller_quiesce" -v
```

Expected: FAIL because the workers are still directly created and not inventory-owned.

- [ ] **Step 4: Add background worker inventory setup in `main.py`**

Near the existing `owned_job_pollers` initialization, create one worker inventory and preserve the current local name for compatibility with existing helpers and tests:

```python
from tldw_Server_API.app.services.lifecycle_workers import WorkerInventory

worker_inventory = WorkerInventory(app)
owned_job_pollers: list[_ManagedJobPoller] = worker_inventory.handles
worker_inventory.publish()
```

The `owned_job_pollers` name is transitional; the underlying list contains all managed phases. `_quiesce_owned_job_pollers_for_shutdown(...)` and `_stop_registered_job_pollers(...)` must filter to `WorkerShutdownPhase.JOB_POLLER_QUIESCE`, while background shutdown uses `worker_inventory.handles_for_phase(WorkerShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)`. This avoids losing background workers when `_replace_owned_job_poller_inventory(...)` refreshes the job-poller slice after startup.

- [ ] **Step 5: Register the four startup blocks through `start_stop_event_worker`**

Replace direct `asyncio.Event()` plus `asyncio.create_task(...)` in the startup blocks for:

- `jobs_metrics_task`
- `jobs_metrics_reconcile_task`
- `jobs_crypto_rotate_task`
- `jobs_integrity_task`

Use:

```python
from tldw_Server_API.app.services.lifecycle_workers import start_stop_event_worker

jobs_metrics_task, jobs_metrics_stop_event = await start_stop_event_worker(
    worker_inventory,
    name="jobs_metrics_task",
    task_name="jobs_metrics_task",
    coroutine_factory=_run_jobs_metrics,
    timeout_sec=5.0,
    category="jobs",
    shutdown_phase=WorkerShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
)
```

Use equivalent names for the other three workers. Keep the existing environment flag checks and log messages. Use `WorkerShutdownPhase.BACKGROUND_WORKER_SHUTDOWN` everywhere in `main.py`.

- [ ] **Step 6: Stop background-phase workers in the existing shutdown window**

After `_quiesce_owned_job_pollers_for_shutdown(...)` and before the direct legacy worker shutdown branches, add a focused background stop call:

```python
await stop_registered_workers(
    app,
    worker_inventory.handles_for_phase(WorkerShutdownPhase.BACKGROUND_WORKER_SHUTDOWN),
    stopped_names_attr="_tldw_shutdown_stopped_background_worker_names",
    log_label="background worker",
)
stopped_background_worker_names = set(
    getattr(app.state, "_tldw_shutdown_stopped_background_worker_names", [])
)
```

Use a helper such as:

```python
def _should_run_late_background_stop(task_name: str, task: Any) -> bool:
    return bool(task) and task_name not in stopped_background_worker_names
```

- [ ] **Step 7: Remove or guard duplicate late-stop branches**

For the four migrated workers, either remove their direct late-stop blocks or guard them with `_should_run_late_background_stop(...)` during transition. Prefer removing the duplicate block once tests prove the inventory path owns shutdown.

- [ ] **Step 8: Run targeted integration tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py -v
```

Expected: PASS.

- [ ] **Step 9: Run lifecycle contract tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit Task 3**

```bash
git add tldw_Server_API/app/main.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py
git commit -m "refactor: migrate jobs background workers to lifecycle inventory"
```

---

### Task 4: Final Verification And Security Check

**Files:**
- Verify touched implementation and tests.

- [ ] **Step 1: Run focused lifecycle tests**

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_lifecycle_workers.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -v
```

Expected: PASS.

- [ ] **Step 2: Run Bandit on touched scope**

```bash
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/main.py tldw_Server_API/app/services/lifecycle_exceptions.py tldw_Server_API/app/services/lifecycle_workers.py tldw_Server_API/tests/Services/test_lifecycle_workers.py tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py -f json -o /tmp/bandit_worker_lifecycle_consolidation.json
```

Expected: exits 0 or reports no new findings in touched code. Fix any new finding before continuing.

- [ ] **Step 3: Inspect final diff**

```bash
git diff --stat HEAD~3..HEAD
git status --short
```

Expected: only intended files are committed for this task; unrelated dirty worktree entries remain untouched.

- [ ] **Step 4: Final commit if verification fixes were needed**

Only if Task 4 required code or test edits:

```bash
git add <touched files>
git commit -m "test: verify worker lifecycle consolidation"
```
