# LLM Usage Accounting PR 1b (the remaining roots) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LLM chat calls made by background jobs, workflow runs, scheduled tasks and audio WebSocket sessions are recorded for their user, and a call a request hands to an executor thread keeps its user.

**Architecture:** PR 1a records a call when the user is known at the point of the call. This PR gives each remaining root a user with one helper, `llm_call_root`, and makes the event loop's default executor carry the LLM call state, so `loop.run_in_executor(None, ...)` no longer drops it.

**Tech Stack:** Python 3.12, asyncio, `contextvars`, `concurrent.futures`, pytest (asyncio mode auto).

**Spec:** `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1, Delivery PR 1b). Task 7 corrects §1 where the survey for this plan found it wrong. Read the spec and PR 1a's ADR, `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`, before any task.

## Global Constraints

- **No new refusals.** This PR adds no quota check.
- **Accounting never breaks the work it accounts for.** A root that cannot work out its user runs exactly as it does today, unrecorded.
- **Carry only the LLM call state** across a thread hand-off: `capture_llm_call_state()` and `run_with_llm_call_state(...)` from PR 1a. Never `contextvars.copy_context()` for this.
- **One helper for roots.** Every job, run and scheduled-task root uses `llm_call_root(owner)`. A task that serves one user for its whole life (a WebSocket handler) uses `bind_llm_call_user(user_id)`.
- **An owner that is not a number is no user.** Job owners are strings and can be empty or absent. `as_quota_user_id` (`core/Usage/quota_checks.py`, the helper the spec names) maps anything non-numeric to `None`, and a `None` user never overrides a user that is already set. Do not add a second coercion helper.
- **The server runs on uvloop** (it comes with `uvicorn[standard]`). A uvloop loop does not expose `_default_executor`, so app code never reads that attribute. Tests run on the stdlib loop unless they build a uvloop loop themselves.
- **Every code block and test in this plan was dry-run** in a scratch checkout of this branch, and an independent reviewer applied all eight tasks there. If an edit does not match the file, or a test does not behave as a step says, stop and report it. Do not improvise.
- **Every edit at a call site wraps one call.** No argument, return value or exception handling changes.
- **Test-first.** Each task's tests are written and seen failing before its code.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit ids, `git add -A backlog/tasks`.
- **Commits** end with exactly `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `git stash`. Never `--no-verify`.
- **Python:** `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python` (written `$PY` below). Tests run with `TLDW_TEST_NO_DOCKER=1`.
- **Three test files flake under parallel load on this machine with none of these changes:** `tests/Admin/test_admin_llm_provider_test_runtime.py`, `tests/LLM_Calls/test_mlx_provider.py`, `tests/Web_Scraping/test_phase4_safe_regex.py`. Re-run one alone before reporting it.
- **One test fails on `dev` in this environment:** `tests/Jobs/test_retry_admission_fixture_registration.py::test_bridge_resolution_preserves_jobs_loop_and_isolation`, with `ValueError: Pytest terminal summary report not found`. It fails the same way with and without this plan's changes.
- **`tests/Jobs` cannot run under xdist as a whole.** Two files put random UUIDs in their parametrized ids. Task 2 Step 5 gives the commands that work.

## Review Focus

Inputs the spec implies and a person would hit. Each has a test in the task named.

1. **A job whose owner is missing, empty or not a number.** Expected: the job runs as today and nothing is recorded; no exception. (Tasks 1 and 2)
2. **A child workflow run awaited inside a parent's step.** Expected: when the child finishes, the parent's user is back in effect. (Task 3)
3. **Two jobs for different owners on the same worker, one after the other.** Expected: the second job never sees the first job's user. (Task 2)
4. **An executor hand-off made with no user set.** Expected: the callee sees no user, and a later hand-off from another task does not inherit one. (Task 1)
5. **A WebSocket that authenticates with no user id.** Expected: the session works as today and nothing is recorded. (Task 5)

## File Structure

| File | Responsibility |
|---|---|
| `tldw_Server_API/app/core/Usage/llm_call_context.py` | Adds `bind_llm_call_user`. |
| `tldw_Server_API/app/core/Usage/llm_roots.py` (new) | `llm_call_root`, and the default executor that carries the LLM call state. |
| `tldw_Server_API/app/main.py` | Installs the executor on the server loop at startup. |
| `tldw_Server_API/app/core/Jobs/worker_sdk.py` | Both handler calls run under `llm_call_root`. |
| `tldw_Server_API/app/core/Workflows/engine.py` | `start_run` and `continue_run` run under `llm_call_root`. |
| `tldw_Server_API/app/services/agent_task_jobs_worker.py`, `tldw_Server_API/app/services/audio_jobs_worker.py`, `tldw_Server_API/app/core/Scheduler/handlers/watchlists.py` | The roots that do not go through `WorkerSDK`. |
| `tldw_Server_API/app/core/Audio/streaming_service.py` | The shared audio WebSocket authentication binds the user. |
| `tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py`, `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py` | Two private thread pools carry the state. |

---

### Task 1: The root helper and an executor that carries the call state

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/llm_call_context.py` (add one function at the end)
- Create: `tldw_Server_API/app/core/Usage/llm_roots.py`
- Modify: `tldw_Server_API/app/main.py` (the `lifespan` function, directly after the `set_accounting_loop(asyncio.get_running_loop())` line PR 1a added)
- Test: `tldw_Server_API/tests/Usage/test_llm_roots.py`; add one test to `tldw_Server_API/tests/Usage/test_llm_accounting_lifespan.py`

**Interfaces:**
- Consumes, from PR 1a: `llm_call_context`, `current_llm_user_id`, `current_llm_call_context`, `capture_llm_call_state`, `run_with_llm_call_state`, and the module variable `_CTX` in `llm_call_context.py`. From spec 2: `as_quota_user_id(value: Any) -> int | None` in `core/Usage/quota_checks.py`.
- Produces:
  - `bind_llm_call_user(user_id: int | None) -> None`
  - `llm_call_root(user_id: Any, *, operation: str | None = None)` (context manager)
  - `ensure_llm_state_executor(loop: AbstractEventLoop | None = None) -> bool`
  - `LLMStateThreadPoolExecutor`

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_roots.py`:

```python
"""Background roots get a user, and an executor hand-off keeps it (LLM usage accounting spec §1)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Usage import llm_roots
from tldw_Server_API.app.core.Usage.llm_call_context import (
    bind_llm_call_user,
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)
from tldw_Server_API.app.core.Usage.llm_roots import (
    LLMStateThreadPoolExecutor,
    ensure_llm_state_executor,
    llm_call_root,
)

pytestmark = pytest.mark.unit


def _probe() -> tuple[int | None, str]:
    """What the current thread sees."""
    return current_llm_user_id(), current_llm_call_context().operation


async def test_a_root_sets_its_user_and_operation_and_restores_them() -> None:
    """Inside the root the owner is the user; afterwards the previous state is back."""
    with llm_call_context(user_id=3):
        with llm_call_root("7", operation="workflow"):
            assert _probe() == (7, "workflow")
        assert _probe() == (3, "llm_call")


async def test_a_root_with_no_usable_owner_keeps_the_user_already_set() -> None:
    """A missing or non-numeric owner never clears a user set by an outer root."""
    with llm_call_context(user_id=3):
        for owner in ("", None, "u"):
            with llm_call_root(owner):
                assert current_llm_user_id() == 3
    with llm_call_root(""):
        assert current_llm_user_id() is None


async def test_binding_a_user_lasts_for_the_rest_of_the_task_only() -> None:
    """A bound user stays for the task that bound it and does not reach a sibling task."""

    async def _session(user_id: int | None) -> int | None:
        """Bind, yield to the loop, then report."""
        bind_llm_call_user(user_id)
        await asyncio.sleep(0)
        return current_llm_user_id()

    assert await asyncio.gather(_session(5), _session(None), _session(9)) == [5, None, 9]
    assert current_llm_user_id() is None


async def test_run_in_executor_drops_the_user_until_the_executor_is_installed() -> None:
    """The default executor loses the user; the installed one carries it, for this hand-off only."""
    loop = asyncio.get_running_loop()
    # The first assertion needs a loop nothing has installed on: pytest-asyncio gives each test its own.
    with llm_call_context(user_id=7):
        assert await loop.run_in_executor(None, _probe) == (None, "llm_call")
    assert ensure_llm_state_executor() is True
    with llm_call_context(user_id=7, operation="rag"):
        assert await loop.run_in_executor(None, _probe) == (7, "rag")
    assert await loop.run_in_executor(None, _probe) == (None, "llm_call")


async def test_installing_the_executor_is_idempotent_and_keeps_to_thread_working() -> None:
    """A second install keeps the same executor, and asyncio.to_thread still carries the user."""
    loop = asyncio.get_running_loop()
    ensure_llm_state_executor()
    first = loop._default_executor  # noqa: SLF001 - the stdlib loop's only way to read the default executor
    assert isinstance(first, LLMStateThreadPoolExecutor)
    ensure_llm_state_executor(loop)
    assert loop._default_executor is first  # noqa: SLF001
    with llm_call_context(user_id=4):
        assert await asyncio.to_thread(_probe) == (4, "llm_call")


def test_on_uvloop_the_executor_is_installed_once_and_carries_the_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The server runs on uvloop, whose loop does not expose its default executor to Python."""
    uvloop = pytest.importorskip("uvloop")
    created: list[LLMStateThreadPoolExecutor] = []

    class _Counting(LLMStateThreadPoolExecutor):
        """Counts how many executors get built."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(llm_roots, "LLMStateThreadPoolExecutor", _Counting)

    async def _main() -> tuple[int | None, str]:
        """Enter three roots on one loop, then hand a call to the executor."""
        for owner in ("1", "2", "3"):
            with llm_call_root(owner):
                pass
        with llm_call_context(user_id=7):
            return await asyncio.get_running_loop().run_in_executor(None, _probe)

    loop = uvloop.new_event_loop()
    try:
        assert not hasattr(loop, "_default_executor")
        assert loop.run_until_complete(_main()) == (7, "llm_call")
        loop.run_until_complete(loop.shutdown_default_executor())
    finally:
        loop.close()
    assert len(created) == 1


def test_with_no_running_loop_nothing_is_installed() -> None:
    """Outside an event loop there is nothing to install on, and nothing raises."""
    assert ensure_llm_state_executor() is False


async def test_a_root_installs_the_executor_on_its_own_loop() -> None:
    """Entering a root is enough for an executor hand-off inside it to keep the user."""
    loop = asyncio.get_running_loop()
    with llm_call_root(8):
        assert await loop.run_in_executor(None, _probe) == (8, "llm_call")
```

Add to `tldw_Server_API/tests/Usage/test_llm_accounting_lifespan.py`:

```python
def test_the_app_lifespan_installs_the_state_carrying_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """While the app is up, the server loop's default executor carries the LLM call state."""
    from tldw_Server_API.app.core.Usage.llm_roots import LLMStateThreadPoolExecutor
    from tldw_Server_API.app.main import app

    monkeypatch.setattr(llm_accounting, "_accounting_loop", None)
    with TestClient(app):
        loop = llm_accounting.get_accounting_loop()
        assert loop is not None
        assert isinstance(loop._default_executor, LLMStateThreadPoolExecutor)  # noqa: SLF001
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_roots.py tldw_Server_API/tests/Usage/test_llm_accounting_lifespan.py`
Expected: `test_llm_roots.py` fails to collect with an `ImportError` (`llm_roots` does not exist yet), and the new lifespan test fails on its import.

- [ ] **Step 3: Add `bind_llm_call_user` to `llm_call_context.py`**

Append at the end of `tldw_Server_API/app/core/Usage/llm_call_context.py`:

```python


def bind_llm_call_user(user_id: int | None) -> None:
    """Set the user for the rest of the current task; ``None`` binds nothing.

    For a task that serves one user for its whole life, such as a WebSocket
    handler. Everything else uses ``llm_call_context`` or ``llm_call_root``,
    which restore the previous user when they exit.
    """
    if user_id is not None:
        _CTX.set(replace(_CTX.get(), user_id=user_id))
```

- [ ] **Step 4: Write `llm_roots.py`**

```python
"""Give background work a user for LLM usage accounting, and keep it across executor hand-offs (spec §1).

A job, a workflow run and a scheduled task start with no HTTP request, so
nothing tells the recorder who they are for. ``llm_call_root`` does. An
executor hand-off made with ``loop.run_in_executor(None, ...)`` does not copy
context variables, so the default executor installed here carries the LLM call
state itself.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

from tldw_Server_API.app.core.Usage.llm_call_context import (
    capture_llm_call_state,
    llm_call_context,
    run_with_llm_call_state,
)
from tldw_Server_API.app.core.Usage.quota_checks import as_quota_user_id

# Loops whose default executor carries the LLM call state. asyncio has no getter for the default
# executor, and uvloop, which the server runs on, keeps it where Python cannot read it.
# ponytail: a later set_default_executor on a listed loop goes unnoticed; nothing in the app calls it.
_INSTALLED: weakref.WeakSet[asyncio.AbstractEventLoop] = weakref.WeakSet()


class LLMStateThreadPoolExecutor(ThreadPoolExecutor):
    """A default executor whose worker threads see the submitter's LLM call state, and nothing else of its context."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        """Capture the state where the work is submitted and apply it where the work runs."""
        return super().submit(run_with_llm_call_state, capture_llm_call_state(), fn, *args, **kwargs)


def ensure_llm_state_executor(loop: asyncio.AbstractEventLoop | None = None) -> bool:
    """Make ``loop``'s default executor carry the LLM call state. Idempotent; False with no running loop."""
    try:
        target = loop or asyncio.get_running_loop()
    except RuntimeError:
        return False
    if target in _INSTALLED:
        return True
    # Same sizing and thread names as the executor asyncio creates on first use.
    target.set_default_executor(LLMStateThreadPoolExecutor(thread_name_prefix="asyncio"))
    _INSTALLED.add(target)
    return True


@contextmanager
def llm_call_root(user_id: Any, *, operation: str | None = None) -> Iterator[None]:
    """Run one unit of background work for ``user_id``: a job, a workflow run, a scheduled task.

    An owner that is not a number sets no user, and never clears one that is
    already set.
    """
    overrides: dict[str, Any] = {}
    resolved = as_quota_user_id(user_id)
    if resolved is not None:
        overrides["user_id"] = resolved
    if operation:
        overrides["operation"] = operation
    ensure_llm_state_executor()
    with llm_call_context(**overrides):
        yield
```

Three things here are deliberate:
- `as_quota_user_id` is imported in this module and not in `llm_call_context.py`. That module sits under `providers/base.py`'s import chain and must stay free of the quota modules.
- The `WeakSet` is how the helper knows it has already installed on a loop. Reading `loop._default_executor` would work on the stdlib loop only: on uvloop the attribute does not exist, the check would always be false, and every job and workflow run would replace the server loop's thread pool.
- A replaced default executor needs no shutdown. Once nothing refers to it, its idle threads exit by themselves.

- [ ] **Step 5: Install the executor at startup**

In `tldw_Server_API/app/main.py`, in `lifespan`, directly after the line `set_accounting_loop(asyncio.get_running_loop())`, add:

```python
    from tldw_Server_API.app.core.Usage.llm_roots import ensure_llm_state_executor

    ensure_llm_state_executor(asyncio.get_running_loop())
```

`services/shutdown_executor_resources.py` already shuts the default executor down with `loop.shutdown_default_executor()`, which works for this subclass.

- [ ] **Step 6: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_roots.py tldw_Server_API/tests/Usage/test_llm_accounting_lifespan.py`
Expected: 10 passed (8 in `test_llm_roots.py` and 2 in the lifespan file). The uvloop test must run, not skip: `uvloop` is installed in this venv.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/app/core/Usage/llm_roots.py \
  tldw_Server_API/app/main.py tldw_Server_API/tests/Usage/test_llm_roots.py \
  tldw_Server_API/tests/Usage/test_llm_accounting_lifespan.py
git commit -m "feat(usage): a root helper and a default executor that carries the LLM call state (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Jobs run for their owner

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py` (`run_prepared`, the handler call near line 719; `run`, the handler call near line 955)
- Test: `tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py`

**Interfaces:**
- Consumes: `llm_call_root` from Task 1.
- Produces: a handler run by `WorkerSDK.run` or `run_prepared` sees `current_llm_user_id()` equal to the job's numeric `owner_user_id`. This covers every worker built on the SDK.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py`, using the same harness as `tests/Jobs/test_worker_sdk.py::test_run_with_job_type_filter_only_handles_matching_jobs`:

```python
"""A job handler run by WorkerSDK sees the job's owner as the LLM usage user (spec §1)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

pytestmark = pytest.mark.unit


def _sdk(jm: JobManager) -> WorkerSDK:
    """A worker on the test queue."""
    return WorkerSDK(
        jm,
        WorkerConfig(
            domain="writing", queue="default", worker_id="w-llm-user",
            lease_seconds=5, renew_threshold_seconds=1, renew_jitter_seconds=0,
        ),
    )


async def test_each_job_runs_for_its_own_owner_and_none_leaks(tmp_path) -> None:
    """Owners "7", "u" and "9" are seen as 7, no user and 9, in order, and no user remains afterwards."""
    db_path = tmp_path / "jobs_llm_user.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    for owner in ("7", "u", "9"):
        jm.create_job(domain="writing", queue="default", job_type="t", payload={"owner": owner}, owner_user_id=owner)
    sdk = _sdk(jm)
    seen: list[tuple[str, int | None]] = []

    async def handler(job_row):
        """Record the user this job sees."""
        seen.append((str(job_row["payload"]["owner"]), current_llm_user_id()))
        if len(seen) == 3:
            sdk.stop()
        return {"ok": True}

    await asyncio.wait_for(sdk.run(handler=handler, job_type="t"), timeout=5)

    assert sorted(seen) == [("7", 7), ("9", 9), ("u", None)]
    assert current_llm_user_id() is None


async def test_a_failing_handler_still_releases_the_user(tmp_path) -> None:
    """A handler that raises does not leave its owner set for the next job."""
    db_path = tmp_path / "jobs_llm_user_fail.db"
    ensure_jobs_tables(db_path)
    jm = JobManager(db_path)
    jm.create_job(domain="writing", queue="default", job_type="t", payload={"n": 1}, owner_user_id="7")
    jm.create_job(domain="writing", queue="default", job_type="t", payload={"n": 2}, owner_user_id="u")
    sdk = _sdk(jm)
    seen: list[int | None] = []

    async def handler(job_row):
        """Fail on the owned job, then report on the unowned one."""
        seen.append(current_llm_user_id())
        if len(seen) == 2:
            sdk.stop()
            return {"ok": True}
        raise RuntimeError("handler failed")

    await asyncio.wait_for(sdk.run(handler=handler, job_type="t"), timeout=5)

    assert sorted(seen, key=lambda value: -1 if value is None else value) == [None, 7]
```

The job row's `payload` reaches the handler as a dict in this harness; a dry run of this file confirmed it.

For `run_prepared`, append this to `tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py`. It uses that file's `PreparedManager`, `_sdk`, `_run_once` and `_complete` helpers. The fixture's default job has no `owner_user_id` key, which is the second case:

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(("owner", "expected"), [("7", 7), (None, None)], ids=["owned", "no-owner"])
async def test_prepared_handler_sees_the_job_owner_as_the_llm_user(owner, expected):
    """LLM usage accounting: a prepared job runs for its owner; a job with none runs with no user."""
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    manager = PreparedManager()
    if owner is not None:
        manager.jobs[0]["owner_user_id"] = owner
    sdk = _sdk(manager)
    seen: list[int | None] = []

    async def handler(_job_row, _context):
        seen.append(current_llm_user_id())
        return _complete()

    await _run_once(sdk, handler)

    assert seen == [expected]
    assert len(manager.apply_calls) == 1
    assert current_llm_user_id() is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py`
Expected: the new tests fail with `None` where `7` and `9` are expected.

- [ ] **Step 3: Run both handler calls under the root**

In `tldw_Server_API/app/core/Jobs/worker_sdk.py`, add the import next to the other `tldw_Server_API` imports:

```python
from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root
```

In `run_prepared`, change:

```python
                    disposition = await handler(copy.deepcopy(acquired_job), context)
```

to:

```python
                    with llm_call_root(acquired_job.get("owner_user_id")):
                        disposition = await handler(copy.deepcopy(acquired_job), context)
```

In `run`, change:

```python
                    result = await handler(job)
```

to:

```python
                    with llm_call_root(job.get("owner_user_id")):
                        result = await handler(job)
```

Each `with` sits inside the existing `try`, so the existing `except` clauses still see the handler's exception.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py tldw_Server_API/tests/Jobs/test_worker_sdk.py`
Expected: 136 passed, and no existing test in the two existing files changes result.

- [ ] **Step 5: Run the Jobs suite for regressions**

Two files in `tests/Jobs` put random UUIDs in their parametrized test ids, so xdist workers collect different ids and a parallel run of the directory aborts with "Different tests were collected". This is the case on `dev`, with none of these changes. Run the directory in parallel without those two files, then run the two serially:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Jobs \
  --ignore=tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  --ignore=tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py
```

Expected: the same pass and fail counts as on the task's base commit.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Jobs/worker_sdk.py tldw_Server_API/tests/Jobs/test_worker_sdk_llm_user.py \
  tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py
git commit -m "feat(usage): a WorkerSDK job runs for its owner in LLM usage accounting (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Workflow runs run for their owner

**Files:**
- Modify: `tldw_Server_API/app/core/Workflows/engine.py` (`start_run` near line 530, `continue_run` near line 1107)
- Test: `tldw_Server_API/tests/Workflows/test_engine_llm_user.py`

**Interfaces:**
- Consumes: `llm_call_root`.
- Produces: every step of a run sees the run's owner as the user, with `operation == "workflow"`, whichever thread or loop runs it. `start_run` and `continue_run` keep their names, signatures and behaviour.

`start_run` runs on a new thread with its own event loop when it goes through `WorkflowScheduler._spawn`, and in the caller's task when it is awaited directly. Both entries are wrapped, so both cases are covered.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Workflows/test_engine_llm_user.py`, using the harness of `tests/Workflows/test_engine_state_contracts.py::test_adapter_exception_outside_noncritical_tuple_fails_run`:

```python
"""A workflow step sees its run's owner as the LLM usage user (spec §1)."""

import asyncio
import threading
import uuid

import pytest

from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase
from tldw_Server_API.app.core.Usage.llm_call_context import (
    current_llm_call_context,
    current_llm_user_id,
    llm_call_context,
)
from tldw_Server_API.app.core.Workflows import engine as engine_mod

pytestmark = pytest.mark.unit


def _create_run(db: WorkflowsDatabase, user_id: str) -> str:
    """A one-step run owned by ``user_id``."""
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id, tenant_id="default", user_id=user_id, inputs={}, workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "probe", "version": 1, "steps": [{"id": "s1", "type": "provider_step", "config": {}}]},
    )
    return run_id


async def test_a_step_awaited_directly_sees_the_run_owner_and_the_caller_keeps_its_own(tmp_path, monkeypatch) -> None:
    """Inside the run the owner is the user; afterwards the caller's user is back."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "7")
    seen: list[tuple[int | None, str]] = []

    async def _adapter(_config, _context):
        """Record what the step sees."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    with llm_call_context(user_id=3):
        await engine_mod.WorkflowEngine(db).start_run(run_id)
        assert current_llm_user_id() == 3
    assert seen == [(7, "workflow")]


async def test_a_spawned_run_on_its_own_thread_sees_the_run_owner(tmp_path, monkeypatch) -> None:
    """A run started through the scheduler's thread still sees its owner."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "9")
    seen: list[tuple[int | None, str]] = []
    done = threading.Event()

    async def _adapter(_config, _context):
        """Record what the step sees, then signal."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        done.set()
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    engine = engine_mod.WorkflowEngine(db)
    engine_mod.WorkflowScheduler._spawn(engine.start_run(run_id))  # noqa: SLF001 - the thread hand-off under test
    assert await asyncio.to_thread(done.wait, 10)
    assert seen == [(9, "workflow")]


async def test_a_run_with_a_non_numeric_owner_runs_with_no_user(tmp_path, monkeypatch) -> None:
    """An owner that is not a number sets no user, and the run still executes."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = _create_run(db, "service-account")
    seen: list[int | None] = []

    async def _adapter(_config, _context):
        """Record what the step sees."""
        seen.append(current_llm_user_id())
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    await engine_mod.WorkflowEngine(db).start_run(run_id)
    assert seen == [None]
```

A dry run of these three tests against the Step 3 code passed as written: the stub adapter's `{"ok": True}` return is enough for the run to complete.

Add a fourth test to the same file, for `continue_run`. It resumes a two-step run after its first step, and it passed in the same dry run:

```python
async def test_a_resumed_run_sees_the_run_owner(tmp_path, monkeypatch) -> None:
    """continue_run is its own entry: the step it resumes into sees the owner too."""
    db = WorkflowsDatabase(str(tmp_path / "wf.db"))
    run_id = f"run-{uuid.uuid4().hex}"
    db.create_run(
        run_id=run_id, tenant_id="default", user_id="7", inputs={}, workflow_id=None,
        definition_version=1,
        definition_snapshot={
            "name": "probe", "version": 1,
            "steps": [
                {"id": "s1", "type": "provider_step", "config": {}},
                {"id": "s2", "type": "provider_step", "config": {}},
            ],
        },
    )
    seen: list[tuple[int | None, str]] = []

    async def _adapter(_config, _context):
        """Record what the resumed step sees."""
        seen.append((current_llm_user_id(), current_llm_call_context().operation))
        return {"ok": True}

    monkeypatch.setattr(engine_mod, "get_adapter", lambda _step_type: _adapter)
    with llm_call_context(user_id=3):
        await engine_mod.WorkflowEngine(db).continue_run(run_id, after_step_id="s1", last_outputs={})
        assert current_llm_user_id() == 3
    assert seen == [(7, "workflow")]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Workflows/test_engine_llm_user.py`
Expected: the first two tests fail with `(None, 'llm_call')` or `(3, 'llm_call')` where the owner and `workflow` are expected.

- [ ] **Step 3: Wrap both entries**

In `tldw_Server_API/app/core/Workflows/engine.py`, add the import next to the other `tldw_Server_API` imports:

```python
from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root
```

Rename the existing `async def start_run(` to `async def _start_run_impl(`, and the existing `async def continue_run(` to `async def _continue_run_impl(`. Change nothing else in either body. Directly above `_start_run_impl`, add:

```python
    def _llm_user_for_run(self, run_id: str) -> Any:
        """The run's owner, for LLM usage accounting; None when the run cannot be read."""
        try:
            return getattr(self.db.get_run(run_id), "user_id", None)
        except _WF_NONCRITICAL_EXCEPTIONS:
            return None

    async def start_run(
        self,
        run_id: str,
        mode: RunMode = RunMode.ASYNC,
        *,
        scheduler_for_notify: WorkflowScheduler | None = None,
    ) -> None:
        """Execute a run, with its owner as the user for LLM usage accounting."""
        with llm_call_root(self._llm_user_for_run(run_id), operation="workflow"):
            await self._start_run_impl(run_id, mode, scheduler_for_notify=scheduler_for_notify)
```

Directly above `_continue_run_impl`, add:

```python
    async def continue_run(
        self,
        run_id: str,
        after_step_id: str,
        last_outputs: dict | None = None,
        next_step_id: str | None = None,
        retry_resume: bool = False,
    ) -> None:
        """Resume a run, with its owner as the user for LLM usage accounting."""
        with llm_call_root(self._llm_user_for_run(run_id), operation="workflow"):
            await self._continue_run_impl(
                run_id, after_step_id, last_outputs=last_outputs, next_step_id=next_step_id, retry_resume=retry_resume
            )
```

Neither `_start_run_impl` nor `_continue_run_impl` exists anywhere in the repo today, and `Any` is already imported in `engine.py`. The engine's own re-entries (`_spawn(self.start_run(...))`, `self.continue_run(...)`) go through the public names, so each sets the same owner again. Leave them as they are.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Workflows/test_engine_llm_user.py`
Expected: 4 passed.

- [ ] **Step 5: Run the Workflows suite for regressions**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Workflows`
Expected: the same pass and fail counts as on the task's base commit. `tests/Workflows/adapters/test_control_adapters.py` replaces `start_run` on a mock engine and must still pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Workflows/engine.py tldw_Server_API/tests/Workflows/test_engine_llm_user.py
git commit -m "feat(usage): a workflow run and its resume run for the run owner in LLM usage accounting (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The roots that do not go through WorkerSDK

**Files:**
- Modify: `tldw_Server_API/app/services/agent_task_jobs_worker.py` (the `handle_agent_task_job` call near line 98)
- Modify: `tldw_Server_API/app/services/audio_jobs_worker.py` (the `analyze` call near line 352)
- Modify: `tldw_Server_API/app/core/Scheduler/handlers/watchlists.py` (`watchlists_enrich_output`, near line 98)
- Test: `tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py`

**Interfaces:**
- Consumes: `llm_call_root`.
- Produces: scheduled agent tasks, audio analysis jobs and watchlist output enrichment each run for their owner.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py`:

```python
"""Roots with their own loop give their work a user (LLM usage accounting spec §1)."""

import asyncio

import pytest

from tldw_Server_API.app.core.Scheduler.handlers import watchlists as watchlists_handlers
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

pytestmark = pytest.mark.unit


async def test_watchlist_output_enrichment_runs_for_the_payload_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The enrichment handler sees the payload's user, and none remains afterwards."""
    seen: list[int | None] = []

    async def _enrich(**kwargs: object) -> None:
        """Record the user the enrichment sees."""
        seen.append(current_llm_user_id())

    monkeypatch.setattr(watchlists_handlers.output_enrichment_handler, "enrich_output", _enrich)
    result = await watchlists_handlers.watchlists_enrich_output({"output_id": 5, "user_id": "7"})
    assert result == {"output_id": 5, "status": "completed"}
    assert seen == [7]
    assert current_llm_user_id() is None


async def test_the_agent_task_worker_runs_a_job_for_its_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """One iteration of the agent-task worker loop: the handler sees the job's owner."""
    from tldw_Server_API.app.core.Scheduled_Tasks import automation_executors
    from tldw_Server_API.app.services import agent_task_jobs_worker as worker

    stop = asyncio.Event()
    seen: list[int | None] = []

    class Manager:
        """Hands out one job, then stops the loop when it completes."""

        def acquire_next_job(self, **kwargs: object) -> dict:
            return {"id": 77, "lease_id": "lease", "job_type": "agent_task_run", "owner_user_id": "7", "payload": {}}

        def complete_job(self, *args: object, **kwargs: object) -> None:
            stop.set()

        def fail_job(self, *args: object, **kwargs: object) -> None:
            stop.set()
            raise AssertionError("the job must complete, not fail")

    async def handler(job: dict, **kwargs: object) -> dict:
        """Record the user the job runs as."""
        seen.append(current_llm_user_id())
        return {"status": "completed"}

    monkeypatch.setattr(worker, "jobs_manager_from_env", Manager)
    monkeypatch.setattr(worker, "handle_agent_task_job", handler)
    monkeypatch.setattr(automation_executors, "register_automation_executors", lambda: None)
    await asyncio.wait_for(worker.run_agent_task_jobs_worker(stop), timeout=5)
    assert seen == [7]
    assert current_llm_user_id() is None


async def test_the_audio_jobs_worker_runs_analysis_for_the_job_owner(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """An audio_analyze job: the analysis call, on its thread, sees the job's owner."""
    monkeypatch.setenv("JOBS_DB_PATH", str(tmp_path / "jobs_llm_user.db"))
    monkeypatch.setenv("JOBS_POLL_INTERVAL_SECONDS", "0.05")

    import tldw_Server_API.app.core.LLM_Calls.Summarization_General_Lib as summarization
    import tldw_Server_API.app.services.audio_jobs_worker as worker
    from tldw_Server_API.app.core.Jobs.manager import JobManager

    async def _unlimited(user_id: int) -> dict:
        return {"daily_minutes": None, "concurrent_streams": None, "concurrent_jobs": None, "max_file_size_mb": None}

    async def _yes(user_id: int) -> tuple[bool, str]:
        return True, ""

    async def _noop(user_id: int) -> None:
        return None

    monkeypatch.setattr(worker, "get_limits_for_user", _unlimited)
    monkeypatch.setattr(worker, "can_start_job", _yes)
    monkeypatch.setattr(worker, "increment_jobs_started", _noop)
    monkeypatch.setattr(worker, "finish_job", _noop)

    seen: list[int | None] = []

    def _analyze(text: str, api_name: str, *args: object, **kwargs: object) -> str:
        """Record the user the analysis runs as."""
        seen.append(current_llm_user_id())
        return "analysis"

    monkeypatch.setattr(summarization, "analyze", _analyze)

    jm = JobManager()
    jm.create_job(
        domain="audio", queue="default", job_type="audio_analyze",
        payload={"text": "hello worker", "perform_analysis": True}, owner_user_id="7",
    )
    stop = asyncio.Event()
    task = asyncio.create_task(worker.run_audio_jobs_worker(stop))
    try:
        for _ in range(100):
            await asyncio.sleep(0.05)
            if task.done():
                pytest.fail(f"audio worker crashed: {task.exception()!r}")
            if seen:
                break
    finally:
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)
    assert seen == [7]
```

Each test reuses an existing harness:
- **Watchlists:** `tests/Watchlists/test_watchlists_scheduler_handler.py::test_watchlists_enrich_output_calls_core_handler`. The `@task` decorator returns an awaitable wrapper, so the handler is awaited directly.
- **Agent-task worker:** `tests/Notifications/test_agent_task_jobs_consumer.py::test_worker_does_not_ack_or_fail_a_busy_execution_claim`. A stub manager hands out one job and stops the loop when it completes.
- **Audio jobs worker:** `tests/AudioJobs/test_audio_worker_smoke.py::test_audio_worker_owner_at_limit_runs_the_job`. A real `JobManager` under `JOBS_DB_PATH`. The worker imports `analyze` inside the loop iteration, so patching the module attribute works. The stub runs on a thread through `asyncio.to_thread`, which copies the context.

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py`
Expected: 3 failed, each with `assert [None] == [7]`.

- [ ] **Step 3: Wrap the three calls**

Add `from tldw_Server_API.app.core.Usage.llm_roots import llm_call_root` to each of the three files.

`tldw_Server_API/app/services/agent_task_jobs_worker.py`, change:

```python
                result = await handle_agent_task_job(job, jobs_manager=jm)
```

to:

```python
                _payload = job.get("payload")
                _owner = job.get("owner_user_id") or (_payload.get("user_id") if isinstance(_payload, dict) else None)
                with llm_call_root(_owner):
                    result = await handle_agent_task_job(job, jobs_manager=jm)
```

That is the same owner `handle_agent_task_job` resolves for itself (`agent_task_jobs.py:190`).

`tldw_Server_API/app/services/audio_jobs_worker.py`, change:

```python
                    analysis = await asyncio.to_thread(analyze, text, payload.get("api_name") or "openai")
```

to:

```python
                    with llm_call_root(owner):
                        analysis = await asyncio.to_thread(analyze, text, payload.get("api_name") or "openai")
```

`owner` is the job's `owner_user_id`, read near line 195 in the same loop iteration.

`tldw_Server_API/app/core/Scheduler/handlers/watchlists.py`, in `watchlists_enrich_output`, change:

```python
    await output_enrichment_handler.enrich_output(
        output_id=output_id_int,
        user_id=user_id_int,
        grouping_config=grouping_config,
        summary_config=summary_config,
    )
```

to:

```python
    with llm_call_root(user_id_int):
        await output_enrichment_handler.enrich_output(
            output_id=output_id_int,
            user_id=user_id_int,
            grouping_config=grouping_config,
            summary_config=summary_config,
        )
```

The enrichment's LLM calls go through `loop.run_in_executor(None, ...)` in `services/outputs_service.py`. `llm_call_root` installs the state-carrying executor on whichever loop runs the handler, so those calls keep the user.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py`
Expected: 3 passed. Then run it with the three harness files:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_scheduler_handler.py \
  tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py \
  tldw_Server_API/tests/AudioJobs/test_audio_worker_smoke.py
```

Expected: 39 passed, 1 skipped. The skip is an opt-in heavy smoke test that is skipped on `dev` too.

- [ ] **Step 5: Commit**

```bash
git add tldw_Server_API/app/services/agent_task_jobs_worker.py tldw_Server_API/app/services/audio_jobs_worker.py \
  tldw_Server_API/app/core/Scheduler/handlers/watchlists.py tldw_Server_API/tests/Usage/test_llm_roots_non_sdk.py
git commit -m "feat(usage): scheduled agent tasks, audio analysis jobs and watchlist enrichment run for their owner (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Audio and realtime WebSocket sessions

**Files:**
- Modify: `tldw_Server_API/app/core/Audio/streaming_service.py` (`_audio_ws_authenticate`, near line 382)
- Test: add to `tldw_Server_API/tests/Audio/test_realtime_auth.py`

**Interfaces:**
- Consumes: `bind_llm_call_user`.
- Produces: after `_audio_ws_authenticate` returns `(True, user_id)` with a user id, the handler's task has that user bound. This covers the four routes in `audio_streaming.py` and the two realtime routes, which authenticate through `core/Audio/Realtime/auth.py`.

The helper is awaited inside the WebSocket handler's task, so a value it binds stays for that task and for the tasks the handler creates afterwards. It has nine `return` statements, so it is wrapped, not edited at each one.

- [ ] **Step 1: Write the failing tests**

Append these two tests to `tldw_Server_API/tests/Audio/test_realtime_auth.py`. They use names the file already has: `DummyWebSocket`, `_single_user_settings` and `streaming_service`.

```python
@pytest.mark.asyncio
async def test_realtime_auth_binds_the_user_for_the_handler_task_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A task that authenticates sees the user for the rest of its life; the parent task does not."""
    import asyncio

    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    ws = DummyWebSocket(first_message='{"type":"session.update","session":{"type":"realtime"}}')
    ws.state.single_user_session_id = "validated-cookie-session"
    ws.state.user_id = 1
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(auth_settings, "get_settings", _single_user_settings)
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    async def _handler() -> tuple[tuple[bool, int | None], int | None]:
        """Authenticate, then report the user this task sees."""
        result = await authenticate_realtime_websocket(ws, "native")
        return result, current_llm_user_id()

    assert await asyncio.create_task(_handler()) == ((True, 1), 1)
    assert current_llm_user_id() is None


@pytest.mark.asyncio
async def test_failed_realtime_auth_binds_no_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The openai-compat route with no credentials is refused and no user is bound."""
    from tldw_Server_API.app.core.Audio.Realtime.auth import authenticate_realtime_websocket
    from tldw_Server_API.app.core.AuthNZ import ip_allowlist
    from tldw_Server_API.app.core.AuthNZ import settings as auth_settings
    from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id

    ws = DummyWebSocket(first_message='{"type":"session.update","session":{"type":"realtime"}}')
    monkeypatch.setattr(streaming_service, "is_multi_user_mode", lambda: False)
    monkeypatch.setattr(auth_settings, "get_settings", _single_user_settings)
    monkeypatch.setattr(ip_allowlist, "resolve_client_ip", lambda *_args, **_kwargs: "127.0.0.1")

    assert await authenticate_realtime_websocket(ws, route_kind="openai_compat") == (False, None)
    assert current_llm_user_id() is None
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Audio/test_realtime_auth.py`
Expected: 1 failed, 4 passed. The first new test fails with `assert ((True, 1), None) == ((True, 1), 1)`. The second passes before and after; it guards the no-user case.

- [ ] **Step 3: Wrap the helper**

In `tldw_Server_API/app/core/Audio/streaming_service.py`, add the import:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import bind_llm_call_user
```

Rename the existing `async def _audio_ws_authenticate(` to `async def _audio_ws_authenticate_impl(`, changing nothing else in it. It is the last definition in the file, so add the wrapper at the end of the file:

```python
async def _audio_ws_authenticate(
    websocket: WebSocket,
    outer_stream: Optional[Any],
    *,
    endpoint_id: str,
    ws_path: str,
    allow_initial_auth_message: bool = True,
) -> tuple[bool, Optional[int]]:
    """Authenticate an audio WebSocket, and bind its user for LLM usage accounting.

    The handler awaits this in its own task, so the bound user stays for the
    session and for the tasks the handler starts afterwards.
    """
    authenticated, user_id = await _audio_ws_authenticate_impl(
        websocket,
        outer_stream,
        endpoint_id=endpoint_id,
        ws_path=ws_path,
        allow_initial_auth_message=allow_initial_auth_message,
    )
    if authenticated:
        bind_llm_call_user(user_id)
    return authenticated, user_id
```

`_audio_ws_authenticate` is imported by name in `api/v1/endpoints/audio/__init__.py`, `audio.py` and `audio_streaming.py`, and looked up by name through `_audio_shim_attr`. Keeping the public name on the wrapper keeps all of those working. Confirm nothing else refers to the old function object: `git grep -n "_audio_ws_authenticate_impl" -- tldw_Server_API` prints only the two lines you wrote.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Audio/test_realtime_auth.py`
Expected: all pass.

- [ ] **Step 5: Run the audio WebSocket tests for regressions**

Run, serially: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Audio/test_ws_audio_chat_stream.py tldw_Server_API/tests/Audio/test_ws_concurrent_streams.py`
Then every test file that names the helper: `git grep -l "_audio_ws_authenticate" -- 'tldw_Server_API/tests/**/test_*.py'`.
Expected: the same results as on the task's base commit.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Audio/streaming_service.py tldw_Server_API/tests/Audio/test_realtime_auth.py
git commit -m "feat(usage): audio and realtime WebSocket sessions bind their user for LLM usage accounting (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Two private thread pools carry the state

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py` (the `_exec.submit(_call_provider)` near line 421)
- Modify: `tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py` (the `executor.submit(run_async_scrape)` near line 1205)
- Regenerate: `Docs/Design/web_scraping_refactor_import_inventory.json`, `Docs/Design/WebScraping_Refactor_Import_Inventory.md` (Step 5)
- Test: `tldw_Server_API/tests/Usage/test_llm_state_private_pools.py`

**Interfaces:**
- Consumes: `capture_llm_call_state`, `run_with_llm_call_state` from PR 1a.
- Produces: work these two pools run sees the submitter's LLM call state.

These pools are created at the call site, not through the loop's default executor, so Task 1 does not cover them.

- [ ] **Step 1: Write the failing tests**

`tldw_Server_API/tests/Usage/test_llm_state_private_pools.py`. Each test calls the smallest function that contains the changed line, with the pool's target replaced by a probe. The harnesses are those of `tests/Claims/test_ingestion_llm_json_fence_parsing.py` and `tests/WebScraping/test_extractor_env_overrides.py::test_extractor_max_workers_env`. Neither needs the network.

```python
"""Two thread pools created at their call site carry the LLM call state (LLM usage accounting spec §1)."""

import json

import pytest

from tldw_Server_API.app.core.Claims_Extraction import ingestion_claims
from tldw_Server_API.app.core.Usage.llm_call_context import current_llm_user_id, llm_call_context
from tldw_Server_API.app.core.Web_Scraping import Article_Extractor_Lib as extractor_lib

pytestmark = pytest.mark.unit


def test_the_claims_extraction_pool_sees_the_submitters_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The provider call runs on a one-thread pool; it sees the user set where the pool was fed."""
    seen: list[int | None] = []

    def _probe(**kwargs: object) -> str:
        """Record the user and return one claim."""
        seen.append(current_llm_user_id())
        return json.dumps({"claims": [{"text": "Claim A."}]})

    monkeypatch.setattr(ingestion_claims, "chat_api_call", _probe, raising=False)
    chunks = [{"text": "irrelevant", "metadata": {"chunk_index": 0}}]
    with llm_call_context(user_id=7):
        claims = ingestion_claims.extract_claims_for_chunks(chunks, extractor_mode="openai", max_per_chunk=5)
    assert [c.get("claim_text") for c in claims] == ["Claim A."]
    assert seen == [7]


def test_the_article_extractor_pool_sees_the_submitters_user(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scrape runs on its own loop in a pool thread; the task on that loop sees the user."""
    seen: list[int | None] = []

    async def _probe(*_args: object, **_kwargs: object) -> list:
        """Record the user and return no pages."""
        seen.append(current_llm_user_id())
        return []

    monkeypatch.setattr(extractor_lib, "recursive_scrape", _probe)
    with llm_call_context(user_id=7):
        assert extractor_lib.sync_recursive_scrape("https://example.com", 1, 1) == []
    assert seen == [7]
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_state_private_pools.py`
Expected: 2 failed, each with `assert [None] == [7]`.

- [ ] **Step 3: Carry the state at both submits**

Add to both files:

```python
from tldw_Server_API.app.core.Usage.llm_call_context import capture_llm_call_state, run_with_llm_call_state
```

`ingestion_claims.py`, change:

```python
                fut = _exec.submit(_call_provider)
```

to:

```python
                _llm_state = capture_llm_call_state()
                fut = _exec.submit(lambda: run_with_llm_call_state(_llm_state, _call_provider))
```

`Article_Extractor_Lib.py`, change:

```python
        future = executor.submit(run_async_scrape)
```

to:

```python
        _llm_state = capture_llm_call_state()
        future = executor.submit(lambda: run_with_llm_call_state(_llm_state, run_async_scrape))
```

Both submits stay one-argument on purpose. Two existing tests replace the pool with a fake whose `submit(self, fn)` takes one callable: `tests/Claims/test_claims_cancellation_and_timeout.py:49` and `tests/WebScraping/test_extractor_env_overrides.py:58`. Passing the state as extra positional arguments breaks both, and the Claims one fails quietly, because the extractor swallows the `TypeError` as a provider failure. Capture the state outside the lambda, where the pool is fed. Inside the lambda the code is already on the pool thread.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_state_private_pools.py`
Expected: 2 passed. Then run it with the existing tests around both sites:

```bash
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_state_private_pools.py \
  tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py \
  tldw_Server_API/tests/WebScraping/test_extractor_env_overrides.py \
  tldw_Server_API/tests/Claims/test_ingestion_llm_json_fence_parsing.py
```

Expected: 10 passed.

- [ ] **Step 5: Regenerate the web-scraping import inventory**

The repo keeps a checked-in inventory of every import of the Web_Scraping modules, and a test compares it with the code. The new test file imports `Article_Extractor_Lib`, so the inventory changes. Without this step `tests/WebScraping/test_refactor_import_inventory.py::test_import_inventory_artifact_matches_current_import_surface` fails.

```bash
$PY Helper_Scripts/web_scraping_refactor_inventory.py --root . \
  --json Docs/Design/web_scraping_refactor_import_inventory.json \
  --markdown Docs/Design/WebScraping_Refactor_Import_Inventory.md
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/WebScraping/test_refactor_import_inventory.py
```

Expected: the two inventory files change by about a dozen lines between them, and the test file gives 10 passed.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py \
  tldw_Server_API/app/core/Web_Scraping/Article_Extractor_Lib.py \
  tldw_Server_API/tests/Usage/test_llm_state_private_pools.py \
  Docs/Design/web_scraping_refactor_import_inventory.json Docs/Design/WebScraping_Refactor_Import_Inventory.md
git commit -m "feat(usage): the claims and article-extractor thread pools carry the LLM call state (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: An end-to-end row through an executor, the spec corrections and the docs

**Files:**
- Test: `tldw_Server_API/tests/Usage/test_llm_usage_recorded_through_executor.py`
- Modify: `Docs/Design/2026-10-09-llm-usage-accounting-design.md` (§1 and the Status line)
- Modify: `Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md` (Consequences), then refresh `Docs/Published`
- Modify: `Docs/Operations/Usage_Quotas.md`
- Backlog: TASK-13437 notes; one new follow-up task.

- [ ] **Step 1: Write the end-to-end test**

`tldw_Server_API/tests/Usage/test_llm_usage_recorded_through_executor.py`, modelled on PR 1a's `test_llm_usage_recorded_end_to_end.py`: a real SQLite AuthNZ pool, a fake provider and real rows.

```python
"""A call handed to an executor thread, and a call under a job root, each write one real llm_usage_log row (spec §1)."""

import asyncio
import uuid
from collections.abc import Iterable
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat import chat_service
from tldw_Server_API.app.core.DB_Management.scope_context import scoped_context
from tldw_Server_API.app.core.LLM_Calls.adapter_registry import ChatProviderRegistry
from tldw_Server_API.app.core.LLM_Calls.providers.base import ChatProvider
from tldw_Server_API.app.core.Usage import llm_accounting
from tldw_Server_API.app.core.Usage.llm_roots import ensure_llm_state_executor, llm_call_root
from tldw_Server_API.tests.provider_credential_test_helpers import resolved_request_fields_async
from tldw_Server_API.tests.Usage.test_usage_tracker_sqlite import _ensure_llm_cache_columns, _ensure_llm_tables

pytestmark = pytest.mark.integration


class _ExecProvider(ChatProvider):
    """A sync-only provider, like most production adapters."""

    name = "exec-fake"

    def capabilities(self) -> dict[str, Any]:
        """Streaming is supported."""
        return {"supports_streaming": True}

    def chat(self, request: dict[str, Any], *, timeout: float | None = None) -> dict[str, Any]:
        """Return a completion with usage."""
        return {
            "choices": [{"message": {"content": "ok"}}],
            "usage": {"prompt_tokens": 30, "completion_tokens": 12, "total_tokens": 42},
        }

    def stream(self, request: dict[str, Any], *, timeout: float | None = None) -> Iterable[str]:
        """Unused."""
        return iter(())


async def _fresh_pool(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A real SQLite AuthNZ pool with the usage tables, bound to this test's loop."""
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("SINGLE_USER_API_KEY", "ut-key-" + uuid.uuid4().hex)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'users.sqlite'}")

    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.session_manager import reset_session_manager
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    reset_settings()
    await reset_db_pool()
    await reset_session_manager()
    pool = await get_db_pool()
    await _ensure_llm_tables(pool)
    await _ensure_llm_cache_columns(pool)
    monkeypatch.setattr(llm_accounting, "_accounting_loop", asyncio.get_running_loop())
    return pool


async def _rows(pool) -> list[dict[str, Any]]:
    rows = await pool.fetchall("SELECT user_id, operation FROM llm_usage_log WHERE provider = ?", "exec-fake")
    return [dict(r) if not isinstance(r, dict) else r for r in rows]


async def test_a_request_style_executor_hand_off_records_the_scope_user(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """The scope is set in the request task; the provider call runs on the default executor; one row for user 41."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    provider = _ExecProvider()
    request = {"messages": [{"role": "user", "content": "hi"}], "model": "exec-model"}

    with scoped_context(user_id=41):
        ensure_llm_state_executor()
        result = await asyncio.get_running_loop().run_in_executor(None, provider.chat, request)
    assert result["choices"][0]["message"]["content"] == "ok"
    await llm_accounting.drain_accounting()

    rows = await _rows(pool)
    assert len(rows) == 1
    assert (rows[0]["user_id"], rows[0]["operation"]) == (41, "llm_call")


async def test_a_job_root_records_the_job_owner(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No scope; a job root for owner "42"; the shared-layer call writes one row for user 42."""
    pool = await _fresh_pool(monkeypatch, tmp_path)
    registry = ChatProviderRegistry(include_defaults=False)
    registry.register_adapter("exec-fake", _ExecProvider)
    monkeypatch.setattr(chat_service, "_get_llm_registry", lambda: registry)

    fields = await resolved_request_fields_async("exec-fake", api_key="k", app_config={}, model="exec-model")
    with llm_call_root("42"):
        result = await chat_service.perform_chat_api_call_async(
            api_endpoint="exec-fake", **fields, messages_payload=[{"role": "user", "content": "hi"}],
            model="exec-model", streaming=False,
        )
    assert result["choices"][0]["message"]["content"] == "ok"
    await llm_accounting.drain_accounting()

    rows = await _rows(pool)
    assert len(rows) == 1
    assert (rows[0]["user_id"], rows[0]["operation"]) == (42, "llm_call")
```

Both tests pass as soon as they are written, because Tasks 1 to 6 are already in. Show that the first can fail: remove its `ensure_llm_state_executor()` line, run it, confirm it fails with `assert 0 == 1` (no row was written), then restore the line.

- [ ] **Step 2: Run the test and confirm it passes**

Run: `TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -p no:cacheprovider tldw_Server_API/tests/Usage/test_llm_usage_recorded_through_executor.py`
Expected: 2 passed.

- [ ] **Step 3: Correct the spec**

In `Docs/Design/2026-10-09-llm-usage-accounting-design.md`:
- Status line: `- **Status:** Approved; PR 1a merged (#3222), PR 1b in progress`.
- §1, "Roots that need an explicit set point": state what this PR does.
  - Jobs workers on `WorkerSDK` run each handler under `llm_call_root(owner)`.
  - Scheduled automations do not run under `WorkerSDK`: `services/agent_task_jobs_worker.py` has its own loop and sets the root itself. So does `services/audio_jobs_worker.py`.
  - The workflow engine wraps both `start_run` and `continue_run`.
  - The `watchlists_enrich_output` Scheduler task sets the root from its payload's user.
  - The audio and realtime WebSockets bind the user in `_audio_ws_authenticate`. The other sockets (ACP, prompt studio, persona, meetings, sandbox, watchlists, workflows) make no chat LLM call.
- §1, "Places that lose context today": replace the `loop.run_in_executor` bullet. It named two sites. The survey found the pattern at many more: document insights, quiz generation, reading summaries, character-memory extraction, watchlist output summaries, the media processors, live meeting insights, Data Tables and G-Eval. They all pass `None` as the executor, so one default executor that carries the LLM call state covers them. It is installed on the server loop at startup and by `llm_call_root` on any other loop. The server runs on uvloop (`uvicorn[standard]`), whose loop does not expose its default executor, so the helper remembers the loops it has installed on. The Scheduler registry's `run_in_executor` path is used only by tests.
- Add to "Review changes" a dated entry for this revision.

- [ ] **Step 4: Update the ADR and the operator page**

`Docs/ADR/067-llm-calls-accounted-in-provider-base-class.md`, Consequences, add:
- Background work gets its user from `llm_call_root`. The loop's default executor carries the LLM call state, so `loop.run_in_executor(None, ...)` keeps the user. A thread pool created at a call site must carry the state itself.
- Still unrecorded, each with its reason:
  - MCP tool calls: MCP authentication sets no content scope.
  - The in-process claims rebuild threads: their tasks carry no owner.
  - System jobs with no user, such as the scheduled quality evaluation.

`Docs/Operations/Usage_Quotas.md`: replace the bullets PR 1a added about what is and is not yet recorded with the current statement. Recorded: calls made during a request, including those handed to an executor thread; jobs run by the Jobs workers; workflow runs; scheduled agent tasks; watchlist output enrichment; audio and realtime WebSocket sessions. Not recorded: the three items above.

Then:

```bash
bash Helper_Scripts/refresh_docs_published.sh
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider tldw_Server_API/tests/Docs
```

Expected: `tests/Docs` passes. Commit only the mirror changes your edits caused.

- [ ] **Step 5: Update the backlog**

Find a free task id first: take the highest id on `origin/dev` and in every open PR, and leave a gap of at least 20, because other sessions allocate ids in sequence and may hold unpushed ones.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13437 --append-notes "PR 1b: background roots get a user through llm_call_root (WorkerSDK run and run_prepared, the workflow engine's start_run and continue_run, the agent-task and audio jobs workers, watchlist output enrichment), audio and realtime WebSockets bind theirs in _audio_ws_authenticate, and the loop's default executor carries the LLM call state so run_in_executor(None, ...) keeps the user. Still unrecorded: MCP tool calls, the in-process claims rebuild threads, system jobs with no user. Next: PR 2 (the check and the cost keys)."
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task create "Record LLM usage for MCP tool calls and the in-process claims rebuild" --id TASK-<free id> --labels quotas,backend --description "Two roots still make chat LLM calls with no known user after the LLM usage accounting PR 1b. MCP: tool calls over /ws and HTTP reach perform_chat_api_call_async (core/MCP_unified quizzes module), and MCP authentication sets no content scope; the tool-execution entry has the user in its RequestContext. Claims rebuild: core/Claims_Extraction/claims_rebuild_service.py runs daemon threads whose ClaimsRebuildTask carries only media_id and db_path, so there is no owner to attribute to. Parent: TASK-13437." --ac "An MCP tool call that makes an LLM call records one row for the calling user" --ac "A claims rebuild task carries its owner and its LLM calls are recorded for that user"
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task normalize --check backlog/tasks/task-13437*.md backlog/tasks/task-<free id>*.md
```

Expected: `normalize --check` exits 0.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/tests/Usage/test_llm_usage_recorded_through_executor.py Docs/ADR Docs/Published \
  Docs/Operations/Usage_Quotas.md Docs/Design/2026-10-09-llm-usage-accounting-design.md
git add -A backlog/tasks
git commit -m "docs(usage): the roots PR 1b covers, what is still unrecorded, and an end-to-end test through an executor (TASK-13437)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Ship

- [ ] **Step 1: Rebase and re-check**

```bash
git fetch origin dev
git rebase origin/dev
$PY Helper_Scripts/export_openapi_schema.py --check apps/tldw-frontend/lib/api/openapi.fingerprint.json
```

This branch starts on PR 1a's head. Rebase only after PR 1a (#3222) has merged, so PR 1a's commits drop out and only this PR's commits remain. Expected: the fingerprint matches; this PR changes no route or schema.

- [ ] **Step 2: Run the sweep**

Every test file that references a changed symbol, plus the suites around the changed code:

```bash
git grep -l -e WorkerSDK -e worker_sdk -e WorkflowEngine -e WorkflowScheduler -e _audio_ws_authenticate \
  -e audio_jobs_worker -e agent_task_jobs_worker -e watchlists_enrich_output -e ingestion_claims \
  -e Article_Extractor_Lib -e llm_call_root -e run_in_executor -- 'tldw_Server_API/tests/**/test_*.py' > <scratchpad>/pr1b_tests.txt
TLDW_TEST_NO_DOCKER=1 $PY -m pytest -q -n 4 -p no:cacheprovider --timeout=600 $(cat <scratchpad>/pr1b_tests.txt) \
  tldw_Server_API/tests/Usage tldw_Server_API/tests/Jobs tldw_Server_API/tests/Workflows \
  tldw_Server_API/tests/LLM_Calls tldw_Server_API/tests/Docs tldw_Server_API/tests/lint \
  --ignore=tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_postgres.py \
  --ignore=tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_sqlite.py
```

Drop those two files from the contract list as well if `git grep` put them there, and run them serially afterwards. Their parametrized ids hold random UUIDs, so a parallel run that includes them aborts at collection (see Task 2 Step 5).

`start_run` is left out of the pattern on purpose: the Sandbox runners have a `start_run` of their own, which would pull in about forty unrelated test files, and the whole `tests/Workflows` directory is in the run anyway.

Every failure needs evidence before it is called pre-existing: it fails the same way at `origin/dev` in a detached worktree, or its execution path touches no changed file. A failure that passes alone is a load flake only if it also passes alone at `origin/dev`.

- [ ] **Step 3: Static checks**

Bandit `-ll` on the changed app files: 0 issues. Ruff per-rule counts on the changed app files must not rise against `origin/dev`.

- [ ] **Step 4: Open the PR against `dev`**

The body covers: which roots now have a user; the default executor and why it carries only the LLM call state; what is still unrecorded and why; that there are no new refusals; the review focus items and their tests; the sweep results with each failure's evidence; and the `## Change summary` waiver section. Queue it alone. After the merge, delete the branch.
