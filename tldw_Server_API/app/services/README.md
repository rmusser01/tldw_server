# Services Module

This package contains application-layer services used by the FastAPI app:
domain services, background workers, schedulers, and lifecycle helpers. The
current lifecycle pattern keeps `tldw_Server_API/app/main.py` as the FastAPI
entry point while moving startup and shutdown work into focused, testable
service modules.

## What Belongs Here

- Application services used by API endpoints or background jobs.
- Background worker and scheduler entry points.
- Startup helpers named `startup_*.py`.
- Shutdown helpers named `shutdown_*.py`.
- Lifespan orchestration helpers named `lifespan_*.py`.

Keep lower-level database, ingestion, RAG, LLM, and AuthNZ implementation details
in their existing `app/core/` modules. Service modules can call those core
modules, but should not duplicate their business logic.

## Lifecycle Pattern

`main.py` should own wiring, FastAPI app construction, and global adapter
callbacks. It should not grow large blocks of inline startup or shutdown logic.
The lifecycle service pattern is:

1. `main.py` calls `run_lifespan_startup_sequence(...)`.
2. Startup helpers initialize core resources, validation, runtime services,
   workers, schedulers, and observability in the legacy order.
3. Startup helpers return explicit handle dataclasses.
4. `LifespanWorkerRuntimeState` stores long-lived task, stop-event, scheduler,
   and resource handles that shutdown will need.
5. `main.py` yields to FastAPI.
6. `main.py` calls `run_lifespan_shutdown_sequence(...)`.
7. Shutdown helpers drain admission gates, quiesce owned pollers, stop workers,
   run coordinated legacy shutdown, and clean up resources.

This keeps ordering visible in `lifespan_startup_sequence.py` and
`lifespan_shutdown_sequence.py`, while individual modules own narrowly scoped
behavior.

## Module Families

- `lifespan_startup_sequence.py` and `lifespan_shutdown_sequence.py`:
  top-level lifecycle orchestration.
- `lifespan_worker_runtime_state.py`: mutable handle container shared between
  startup and shutdown.
- `startup_pre_core.py`, `startup_core_initialization.py`,
  `startup_worker_bootstrap.py`, `startup_worker_groups.py`,
  `startup_service_tail.py`: startup phase grouping.
- `startup_*.py`: focused startup slices such as auth, validation, telemetry,
  resource governor, worker groups, and scheduler setup.
- `shutdown_*.py`: focused shutdown slices such as worker late-stop helpers,
  coordinated shutdown, resource cleanup, and telemetry cleanup.
- `*_jobs_worker.py`, `*_scheduler.py`, and `*_service.py`: runtime services
  invoked by endpoints, startup, or jobs.

## Handles And Ownership

Use explicit dataclasses for values that must survive beyond the helper call.
Common examples:

- `*_task`
- `*_stop_event`
- scheduler handles
- database/session manager handles
- service singleton handles

Startup helpers should return handles rather than mutating module globals. When
shutdown needs a handle, copy it into `LifespanWorkerRuntimeState` through the
existing `apply_*_handles(...)` flow.

For job pollers that shutdown owns, register them with
`register_owned_job_poller(...)`. If task creation succeeds but ownership
registration fails, cancel the task before returning or re-raising; otherwise the
worker can keep running without a shutdown owner.

## Worker Shutdown Contract

Background workers should follow the same cooperative stop pattern:

1. Startup creates an `asyncio.Event` stop signal and an `asyncio.Task`.
2. Startup stores or registers both handles.
3. Shutdown sets the stop event.
4. Shutdown waits for the task with a bounded timeout, usually about 5 seconds.
5. On timeout or guarded shutdown failure, shutdown cancels the task.
6. After cancelling, shutdown waits briefly and suppresses expected
   `asyncio.CancelledError`, timeout, and configured guard exceptions.
7. Shutdown clears stale task handles in returned handle dataclasses.

Cancel-only helpers are acceptable for legacy tasks that do not support a stop
event yet, but new workers should prefer cooperative stop events.

## Exception Policy

Lifecycle helpers should make failure semantics explicit.

- Critical startup resources should fail fast. Auth DB pool initialization is a
  critical startup dependency and raises `AuthStartupError` on failure.
- Optional startup services may log and return `None` when disabled or
  unavailable.
- Reporting-only paths may degrade, but should log at an operator-visible level
  when the loss matters.
- Avoid `except Exception` in new lifecycle code. Use the exception tuples
  supplied by the orchestrator, or define a narrow local tuple.
- Do not silently swallow shutdown failures. If a helper catches and continues,
  log enough context to diagnose what was skipped.
- Do not include `{exc}` in `logger.exception(...)` messages unless the duplicate
  exception text is intentional.

## Testing Expectations

Tests should mirror the service module name under
`tldw_Server_API/tests/Services/`.

Use focused unit tests for lifecycle helpers:

- Patch lazy imports with `monkeypatch` and `sys.modules`.
- Assert task/stop-event handles are returned and cleared correctly.
- Assert timeout and cancel fallbacks do not abort the full shutdown sequence.
- Assert critical startup failures propagate.
- Assert optional startup failures degrade in the documented way.

When changing lifecycle behavior, run the touched service tests plus a compile
check for the touched modules. For security-sensitive code, run Bandit on the
touched service scope.

## Adding A New Lifecycle Slice

1. Pick a narrow module name such as `startup_<area>.py` or
   `shutdown_<area>.py`.
2. Move one coherent block from `main.py` or an orchestrator into the helper.
3. Pass dependencies explicitly instead of importing app globals.
4. Return a dataclass when shutdown needs handles from startup.
5. Preserve the existing startup or shutdown order in the sequence module.
6. Add tests under `tests/Services/`.
7. Update this README when the new slice changes the lifecycle contract.

## Common Pitfalls

- Creating a task before ownership registration and not cancelling it on
  registration failure.
- Catching `asyncio.TimeoutError` too late, allowing shutdown to abort.
- Returning `None` for critical startup resources that the app cannot run
  without.
- Resetting service singletons while coordinated shutdown still owns the
  corresponding service.
- Mixing transport component names into legacy component state fields.
- Treating `bool("false")` or `bool("0")` as a feature flag parser.
