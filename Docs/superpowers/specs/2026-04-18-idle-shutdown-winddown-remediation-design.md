# Idle Shutdown Wind-Down Remediation Design

Date: 2026-04-18
Topic: Idle shutdown latency for local `uvicorn` and Docker/container stop
Status: Approved design

## Goal

Reduce the long wind-down time during idle shutdown for the local plain `uvicorn` path and Docker/container stop path.

This slice is specifically about shutdowns where the server is effectively idle. The target is to remove fixed or avoidable teardown waits so the process exits promptly when there is no meaningful work left to drain.

## Observed Behavior And Evidence

The current code and the captured shutdown log point to two concrete contributors:

1. The Evaluations connection pool shutdown blocks for about five seconds even on an idle shutdown.
   - In the captured log, `tldw_Server_API.app.core.Evaluations.connection_pool:shutdown` starts at `2026-04-18 12:13:11.551` and completes at `2026-04-18 12:13:16.556`.
   - This aligns with `tldw_Server_API/app/core/Evaluations/connection_pool.py`, where shutdown sets `_shutdown = True` and then calls `self._maintenance_task.join(timeout=5)`.
   - The maintenance worker itself sleeps in a loop for up to 60 seconds between maintenance passes, which strongly suggests the five-second stall is synthetic waiting rather than real cleanup work.

2. Jobs workers continue polling during shutdown after the acquire gate has already closed.
   - The captured log shows repeated `Jobs acquire gate enabled; declining new acquisition` messages at one-second intervals during shutdown.
   - The same span also shows repeated `Ensured user directory exists` log lines, which indicates worker activity is still happening even though the app is already draining.
   - This means the drain gate is being enabled earlier than worker stop signaling, so idle workers continue their polling loops until a later shutdown step finally stops them.

The captured shutdown also shows that the app eventually exits immediately after `Application shutdown complete.` from uvicorn, so in this trace the dominant delay is inside the application lifespan teardown rather than uvicorn's outer graceful-shutdown timer.

## Scope

This design covers:

- idle shutdown for local plain `uvicorn`
- idle shutdown for Docker/container stop
- the FastAPI lifespan teardown path in `tldw_Server_API/app/main.py`
- the Evaluations connection pool maintenance-thread shutdown path in `tldw_Server_API/app/core/Evaluations/connection_pool.py`
- shutdown timing logs needed to identify the next slow teardown block quickly

This design includes:

- removing the synthetic Evaluations pool shutdown wait
- inventorying started in-process job pollers and closing shutdown-ownership gaps
- stopping jobs workers earlier in the shutdown sequence
- adding lightweight shutdown duration logging around major teardown blocks

## Out Of Scope

This design does not include:

- a full rewrite of the shutdown coordinator model
- reworking busy shutdown semantics for active requests, active jobs, or active WebSocket sessions
- converting every legacy teardown step in `main.py` into a coordinated shutdown component
- general performance cleanup outside the shutdown path

## Approaches Considered

### Recommended: Targeted remediation with targeted instrumentation

Fix the proven idle-shutdown hotspots first:

- make the Evaluations pool maintenance thread interruptible on shutdown
- inventory all started `acquire_next_job()` pollers and close missing stop ownership
- move jobs-worker stop signaling earlier in `main.py` for the idle path while preserving lease-drain behavior
- add timing logs around major teardown blocks

Why this is preferred:

- directly addresses the two concrete issues proven by the captured log
- keeps the write scope narrow and lower risk
- improves both behavior and future diagnosis
- avoids turning a bounded remediation into a large shutdown architecture project

### Alternative: Instrumentation-first, behavior changes later

Add detailed shutdown timing logs first, ship that alone, and defer behavioral changes until after another data-gathering round.

Trade-offs:

- lower initial behavior risk
- slower to deliver relief for the already-proven five-second stall
- leaves the repeated worker polling noise in place

### Alternative: Full shutdown refactor

Move the long direct teardown tail in `main.py` into the shutdown coordinator so all shutdown work becomes centrally budgeted and summarized.

Trade-offs:

- best long-term architecture
- much larger change surface
- higher regression risk
- not necessary to solve the concrete idle-shutdown problem already isolated

## Chosen Design

Use the recommended targeted remediation with targeted instrumentation.

The design is intentionally narrow:

- eliminate the fixed Evaluations pool shutdown wait
- make shutdown ownership explicit for started in-process job pollers
- quiesce jobs workers earlier on the idle path so they stop polling once draining begins
- make the remaining shutdown path legible through per-block duration logs

## Architecture And Change Boundaries

### 1. Evaluations Pool Shutdown

Change `tldw_Server_API/app/core/Evaluations/connection_pool.py` so the maintenance thread is wakeable on shutdown rather than relying on a long sleep and a fixed `join(timeout=5)`.

The desired shape is:

- the maintenance worker waits on a shutdown signal instead of only `time.sleep(...)`
- `shutdown()` triggers that signal before joining the thread
- the normal idle-shutdown path exits near-immediately after signaling
- a bounded join remains as a defensive fallback in case the thread still does not exit cleanly

The critical behavioral change is that shutdown no longer pays a fixed five-second tax just because the maintenance thread happens to be in its sleep window.

For this slice, the Evaluations shutdown invocation should remain synchronous in the lifespan path after the synthetic join wait is removed. The remaining work is a bounded close of SQLite connections and maintenance-thread teardown, which should be measured directly via shutdown timing logs. Moving this shutdown behind `asyncio.to_thread(...)` is not part of this slice unless post-fix timing still shows material event-loop blocking.

### 2. Worker Quiescing Order

Change the shutdown order in `tldw_Server_API/app/main.py` so the idle-shutdown path reaches worker stop signaling much earlier than it does today, while preserving the current bounded lease-drain behavior for non-idle shutdowns.

Before changing order, build an inventory of all started in-process `acquire_next_job()` pollers and ensure each one has explicit shutdown ownership from `main.py` or an equivalent registered stop hook. This includes:

- workers started directly with an explicit stop event
- workers started through helper functions that currently return only a task or hide the stop signal internally

The remediation is not just a reorder of the existing stop block. It must also close ownership gaps so that every started poller is either:

- explicitly stop-signaled,
- explicitly cancelled as documented fallback behavior, or
- intentionally excluded with a written rationale

Examples already visible in the current code include:

- `audiobook_jobs_task`, which is started with a stop event but is not currently handled in the existing teardown block
- helper-started workers such as reminder and connectors workers, where shutdown ownership is not exposed in the same way as the directly managed pollers

The desired shape is:

- mark lifecycle as draining
- enable the Jobs acquire gate
- inspect whether active processing jobs exist
- if `JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC > 0` and active processing jobs exist, preserve the existing bounded lease wait
- once the lease wait completes, or immediately when there are no active processing jobs, signal or stop all inventoried job pollers
- only then continue through the rest of teardown

This should prevent idle workers from continuing their one-second polling loops during shutdown and reduce log noise such as repeated `Jobs acquire gate enabled; declining new acquisition` messages.

This is a sequencing and ownership-hardening change, not a change to the logical shutdown contract. The idle path should stop pollers immediately after the drain handoff, while the non-idle path should preserve the existing lease-drain semantics when configured.

### 3. Shutdown Observability

Add lightweight duration logging around the major teardown blocks that remain in `main.py`.

This does not need a new metrics system or a full tracing model. The requirement is simpler:

- log start and end durations for the major shutdown blocks
- surface any materially slow block in a single pass through the logs
- record total app-teardown wall time

At minimum, the logged timing segments should cover:

- transition handoff / drain gate
- optional lease wait
- job poller quiesce
- Evaluations pool shutdown
- unified audit plus executor shutdown
- telemetry shutdown
- total app teardown

This should make future shutdown regressions attributable without needing to manually reconstruct timing from raw log timestamps.

## Expected Behavior

After this change, the idle shutdown sequence should behave like:

`mark draining` -> `set acquire gate` -> `observe zero active processing jobs` -> `stop all owned job pollers` -> `run remaining teardown` -> `emit per-block durations` -> `complete uvicorn shutdown`

The important difference is that idle workers should stop because they were explicitly told to stop, not because they keep waking up, discovering the gate is closed, and polling again until a later shutdown phase reaches them.

For the Evaluations pool, the maintenance-thread shutdown path should become signal-driven and near-immediate in the normal case.

When active processing jobs exist and `JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC` is configured, the expected sequence should remain:

`mark draining` -> `set acquire gate` -> `perform bounded lease wait` -> `stop all owned job pollers` -> `run remaining teardown` -> `emit per-block durations` -> `complete uvicorn shutdown`

## Failure Handling

Shutdown must remain best-effort and should not become more brittle because of these changes.

Required behavior:

- if early worker stop signaling fails for a specific worker, log the failure and continue shutdown
- if a started poller lacks clean shutdown ownership today, this slice must either add that ownership explicitly or document and use an explicit cancel fallback
- if the Evaluations maintenance thread does not exit promptly after being signaled, retain a bounded join fallback
- if duration logging itself fails, it must not affect shutdown correctness

The overall principle is that observability and optimization must not be allowed to block process exit or mask the original shutdown flow.

## Testing Strategy

The implementation plan should include both focused tests and manual validation.

### Automated Tests

Add focused tests for:

- Evaluations pool shutdown:
  - verify shutdown does not spend about five seconds waiting on a sleeping maintenance thread in the normal case
  - verify the maintenance worker exits after receiving the shutdown signal
- lifespan shutdown sequencing:
  - verify the zero-active-processing path reaches poller quiesce before the long resource-cleanup tail
  - verify the active-processing path preserves bounded lease-wait behavior when `JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC` is configured
  - verify idle workers do not continue repeated acquire-gate polling after shutdown begins
  - verify all started in-process `acquire_next_job()` pollers have explicit shutdown ownership or a documented intentional exclusion

Tests should stay narrow and behavior-oriented rather than trying to assert every log line in the full lifespan teardown.

### Manual Validation

Run manual verification for:

- local plain `uvicorn`
- Docker/container stop

The validation should confirm:

- idle shutdown completes within the numeric target defined below
- idle shutdown no longer spends about five seconds in Evaluations pool shutdown
- jobs workers stop promptly after drain begins on the zero-active-processing path
- non-idle shutdown still preserves bounded lease-wait behavior when configured
- repeated acquire-gate polling log spam disappears or is substantially reduced
- the required timing segments appear in the logs and clearly identify the slowest remaining shutdown block, if any

## Acceptance Criteria

This design is complete when the implementation achieves all of the following:

- idle local plain `uvicorn` shutdown completes in 8 seconds or less from the first shutdown transition log to `Application shutdown complete.` under default config with zero active processing jobs
- idle Docker/container stop completes in 8 seconds or less under the same zero-active-processing conditions
- the Evaluations pool shutdown timing segment completes in 1 second or less in the normal idle case
- the zero-active-processing path reaches poller quiesce immediately after the drain handoff without first spending time in a lease-wait loop
- the active-processing path preserves the existing bounded lease-wait semantics when `JOBS_SHUTDOWN_WAIT_FOR_LEASES_SEC` is configured
- every started in-process `acquire_next_job()` poller has explicit shutdown ownership or a documented intentional exclusion with rationale
- shutdown logs include durations for transition handoff, optional lease wait, job poller quiesce, Evaluations pool shutdown, unified audit plus executor shutdown, telemetry shutdown, and total app teardown

## Risks

The main risks are:

- changing shutdown ordering could surface latent assumptions about workers still running deeper into teardown
- the Evaluations pool maintenance thread may have hidden coupling that assumes periodic sleep instead of signal-driven wakeup
- overly chatty duration logging could reduce readability if applied too broadly

These risks are contained by keeping the scope narrow, preserving bounded fallbacks, and instrumenting only the major teardown blocks rather than every line of shutdown code.

## Non-Goals

This design is not a mandate to:

- coordinator-ize all legacy shutdown logic
- optimize every teardown block in one pass
- tune outer uvicorn graceful-shutdown settings
- change busy-shutdown semantics for in-flight work

## Next Step

The next step after spec approval is to write a concrete implementation plan for the remediation work itself. No implementation changes are authorized by this design document alone.
