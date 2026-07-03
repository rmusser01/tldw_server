# ADR-021: Services Lifecycle Startup and Shutdown

**Status:** Accepted
**Date:** 2026-06-05
**Backfilled from:** `tldw_Server_API/app/services/README.md`
**Decision owner:** TASK-2259 confirmation and TASK-2260 backfill scope
**Related task:** TASK-2260
**Related spec/plan:** `Docs/ADR/inventory/2026-06-04-services-lifecycle-confirmation-audit.md`

## Decision

FastAPI lifespan startup and shutdown orchestration is owned by focused Services helpers, and lifecycle-managed workers are owned through the worker lifecycle session with cooperative stop-event workers, explicit shutdown phases, bounded timeout/cancel fallback, and compatibility caveats for callback-only workers and legacy shutdown adapters.

## Context

The FastAPI application has many startup and shutdown responsibilities: core resource initialization, validation, background worker startup, scheduler setup, request/job drain gates, job-poller quiesce, worker shutdown, legacy component coordination, and final resource cleanup. Keeping all of that inline in `main.py` made ordering hard to test and made long-lived worker handles easy to scatter across module globals.

TASK-2259 confirmed the current Services lifecycle behavior that bounds this ADR:

- `main.py` constructs `LifespanWorkerRuntimeState`, passes it into `run_lifespan_startup_sequence(...)`, then passes the same object into `run_lifespan_shutdown_sequence(...)`.
- Startup helpers return explicit handles. The worker-bootstrap path returns a `WorkerLifecycleSession`, and `LifespanWorkerRuntimeState` stores that session for shutdown.
- Lifecycle-managed workers are declared as `WorkerSpec` values and started through `LifecycleWorkerEngine`, with `WorkerLifecycleSession` tracking handles, enabled/disabled state, stopped or quiesced workers, and diagnostic inventory.
- Stop-event task workers are the default strategy for new lifecycle-managed workers. Callback-only workers remain supported for components that expose shutdown callbacks instead of task handles.
- Shutdown runs in staged order: transition drain/gate, job-poller handoff and bounded quiesce, background-worker shutdown, coordinated legacy components, pre-worker cleanup, post-worker phase, post-worker services cleanup, and final resource cleanup.
- Within a shutdown phase, the lifecycle engine stops dependent workers before their dependencies and stops independent workers concurrently.
- Shutdown uses bounded waits and cancellation fallback so a stuck worker does not block the full application teardown indefinitely.

This ADR is intentionally bounded. It covers Services lifecycle-managed workers and the FastAPI lifespan startup/shutdown sequence. It does not replace ADR-003's Jobs-vs-Scheduler ownership rule, does not claim that every background operation in the repository is Services-managed, and does not claim that all legacy shutdown adapters have been removed.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep startup and shutdown logic inline in `main.py` | Inline lifecycle logic makes ordering, ownership, and failure semantics harder to test and turns the FastAPI entry point into a catch-all orchestration module. |
| Store worker tasks and stop events as scattered module globals | Scattered handle ownership makes shutdown incomplete by default and makes it difficult to know which worker owns which stop path. |
| Use direct task cancellation as the default worker shutdown strategy | Cancel-only shutdown gives workers no cooperative stop signal and makes graceful lease drain or task-local cleanup less reliable. |
| Stop every worker in one unphased global loop | A single loop obscures the required ordering between request/job drain, job-poller quiesce, background workers, legacy components, post-worker cleanup, and final resource cleanup. |
| Require all lifecycle workers to be stop-event task workers immediately | Some current components expose shutdown callbacks or still participate in legacy shutdown adapters; forcing immediate conversion would overstate the migration and risk breaking existing teardown paths. |
| Treat this as a general background-work ownership ADR | The confirmed evidence only supports Services lifespan-managed workers. Jobs/Scheduler defaults, external worker processes, and module-specific runtime ownership remain separate decisions. |

## Consequences

New FastAPI lifespan startup or shutdown work should be added through focused `startup_*.py`, `shutdown_*.py`, or `lifespan_*.py` Services helpers rather than growing large inline blocks in `main.py`.

New long-running workers started by the Services lifespan path should prefer declarative `WorkerSpec` registration and stop-event task ownership. If a worker cannot expose a stop event yet, it should use an explicit callback-only strategy or a clearly caveated legacy adapter rather than an unowned task.

Startup helpers should return explicit handle dataclasses or worker lifecycle sessions. Shutdown-needed worker handles should flow through `LifespanWorkerRuntimeState` by storing the `WorkerLifecycleSession`, not by adding ad hoc module globals.

Shutdown ordering should preserve the staged sequence confirmed by TASK-2259: transition drain/gate first, job-poller quiesce before background worker shutdown, coordinated legacy component shutdown before later cleanup, and final resource cleanup last.

Worker shutdown should remain bounded. Stop events and callbacks get a configured timeout; unresponsive tasks are cancelled and awaited briefly so one stuck worker does not block teardown indefinitely. Bounded lease drain is not a guarantee that every external job finishes before shutdown proceeds.

Legacy shutdown adapters and callback-only workers remain compatibility paths. Future cleanup can migrate more components into declarative lifecycle worker specs, but this ADR does not require removing all legacy paths in one change.

## Follow-up

- Use this ADR as the covering record for the Services lifecycle portion of INV-031.
- Keep `Docs/ADR/inventory/2026-06-04-services-lifecycle-confirmation-audit.md` as the evidence record and caveat boundary for this backfill.
- Create separate ADRs if a future change materially changes Jobs-vs-Scheduler ownership, external worker-process ownership, or legacy shutdown adapter removal.
