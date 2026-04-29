# Worker Lifecycle Consolidation Design

- Date: 2026-04-29
- Issue: #1114
- Topic: Continue Phase 2.1 lifecycle cleanup by consolidating remaining in-process worker ownership on the current shutdown model
- Status: Approved design

## Goal

Reduce lifecycle coupling in `tldw_Server_API/app/main.py` by consolidating remaining stop-event-owned background workers under the current managed-poller and shutdown-coordinator seams.

This design intentionally adapts issue #1114 to the current codebase. The historical issue text references a `WorkerRegistry`, but the current branch no longer contains `tldw_Server_API/app/services/worker_registry.py`, and the recovery plan records that the old WorkerRegistry slice was rejected as stale. The current architecture already has `_ManagedJobPoller`, app-state worker inventory, early job-poller quiescing, and a shutdown coordinator. The next slice should extend those seams rather than resurrect an obsolete abstraction.

## Context

Issue #1114 tracks the remaining Phase 2.1 worker migration work after earlier startup extraction. The current `main.py` still contains many inline `create_task(...)`, stop-event, and late shutdown branches. Some workers are already inventoried and stopped through `_register_owned_job_poller(...)` and `_quiesce_owned_job_pollers_for_shutdown(...)`; others still have bespoke startup and teardown logic.

The highest leverage path is to make lifecycle ownership explicit and reusable for workers that already follow the same shape:

- create an `asyncio.Event`
- start an async loop task
- publish lifecycle inventory
- on shutdown, set the stop event, wait with a bounded timeout, and cancel if stubborn

This work is architectural cleanup, not feature development. Its value is smaller `main.py`, more consistent shutdown behavior, easier test coverage, and fewer future service-specific lifecycle copies.

## Scope

In scope:

- stop-event-owned background loops started inside the FastAPI lifespan
- app-state worker inventory compatibility
- concurrent stop-event shutdown behavior
- targeted extraction of lifecycle mechanics from `main.py`
- regression tests for inventory, stop behavior, and migrated worker startup/shutdown

Initial migration candidates include workers with current direct task ownership such as:

- `embeddings_compactor`
- `jobs_metrics`
- `jobs_metrics_reconcile`
- `jobs_crypto_rotate`
- `jobs_webhooks`
- `meetings_webhook_dlq`
- `workflows_dlq`
- `workflows_gc`
- `workflows_maint`
- `jobs_integrity`
- `tts_history_cleanup`

Out of scope:

- bringing back the stale `WorkerRegistry` design
- redesigning cron-style schedulers in the first slice
- ResourceGovernor extraction
- route registration refactors
- Auth dependency refactors from #1016
- broader audit unification from #1053
- changing worker enablement flags or default enablement semantics

## Approaches Considered

### Recommended: Extend The Current Managed-Worker Model

Keep the current shutdown architecture and extract only the repeated lifecycle mechanics into a small service module. `main.py` still owns service-specific enablement decisions and startup ordering for the first slice.

Pros:

- fits the current codebase
- keeps behavior compatible
- avoids reintroducing stale abstractions
- reduces duplicated teardown paths incrementally
- directly supports #1114 while respecting the Phase 2 recovery notes

Cons:

- the first slice still leaves scheduler-style services in `main.py`
- the final module shape may need another pass after several migrations

### Alternative: Reintroduce WorkerRegistry

Create a new `WorkerRegistry` with `register()` and `register_custom()` and move all worker startup into it.

Pros:

- clean conceptual model
- aligns with the original wording of #1114

Cons:

- conflicts with the current recovery plan
- duplicates existing managed-poller and shutdown-coordinator responsibilities
- higher chance of broad startup/shutdown regressions

### Alternative: Scheduler-Only Cleanup

Leave poller-style workers alone and migrate only cron or cleanup schedulers.

Pros:

- narrower initial behavior risk

Cons:

- misses the highest-duplication lifecycle surface
- leaves direct stop-event worker handling scattered across `main.py`
- does not build on the existing early quiesce model

## Chosen Design

Use the recommended current-model extension.

Create `tldw_Server_API/app/services/lifecycle_workers.py` to hold reusable lifecycle primitives currently embedded in `main.py`. The first implementation should keep startup policy in `main.py` and extract mechanics only. That means service-specific imports, environment flag checks, sidecar-mode decisions, and route-key enablement checks remain close to their existing code until tests prove a broader extraction is safe.

## Components

### ManagedWorker

Public dataclass replacing the private `_ManagedJobPoller` shape over time.

Fields:

- `name`
- `task`
- `stop_event`
- `timeout_sec`
- `category`

`category` should be optional and used for diagnostics only. Existing app-state inventory consumers must continue to see `name`, `task_name`, `has_stop_event`, and `timeout_sec`.

### WorkerInventory

Small owner for started worker handles.

Responsibilities:

- hold `ManagedWorker` records
- register workers after successful startup
- publish inventory to `app.state._tldw_shutdown_job_poller_inventory`
- expose handles to the shutdown path
- tolerate app-state publication failures as best-effort metadata failures

### start_stop_event_worker

Helper for repeated startup mechanics.

Responsibilities:

- accept a worker name, coroutine factory, timeout, and logger labels
- create the stop event
- create the task
- register the task in `WorkerInventory`
- return the task and stop event to callers that still need local references during transition

The helper must not hide enablement policy in the first slice. It should be called only after `main.py` has decided a worker should start.

### stop_registered_workers

Extract the current concurrent stop-event shutdown behavior.

Responsibilities:

- set each worker stop event when present
- wait up to each worker timeout
- cancel workers that do not exit
- continue stopping remaining workers after individual failures
- record quiesced worker names for compatibility with existing late-stop fallback logic

This should preserve the current behavior of `_stop_registered_job_pollers(...)` before the old private helper is removed or renamed.

## Startup Flow

1. `main.py` creates a `WorkerInventory(app)` near the existing `owned_job_pollers` initialization point.
2. `main.py` evaluates the same environment flags and route defaults it evaluates today.
3. For compatible enabled workers, `main.py` calls `start_stop_event_worker(...)`.
4. The helper creates an `asyncio.Event`, starts the task, registers the handle, and republishes app-state inventory.
5. Disabled workers and startup failures retain equivalent logging to the current implementation.

For the first migration batch, `main.py` should keep enough returned local task variables to avoid a risky all-at-once rewrite. Once a worker is fully owned by inventory shutdown, its duplicated late-stop branch can be removed.

## Shutdown Flow

1. The existing transition handoff still marks lifecycle shutdown and closes the Jobs acquire gate.
2. The early worker quiesce path calls the extracted `stop_registered_workers(...)` through the current `_quiesce_owned_job_pollers_for_shutdown(...)` behavior or a compatible renamed wrapper.
3. Registered workers are stopped concurrently.
4. App-state quiesced names are published so legacy fallback logic can skip already-stopped workers.
5. Direct late-stop branches remain only for non-migrated workers or as explicit fallback during transition.
6. Once tests prove a worker is inventory-owned, its duplicate manual teardown block should be removed.

The core invariant is one owner per worker. A migrated worker should not have both an inventory-owned stop path and an unguarded duplicate late-stop path.

## Error Handling

Startup:

- optional worker startup failures remain non-fatal
- the same guarded exception posture should be preserved
- disabled workers continue to log why they did not start

Shutdown:

- stop events are set before cancellation
- each worker keeps a bounded timeout
- stubborn workers are cancelled
- failures for one worker do not block remaining shutdown
- metadata publication failures do not fail shutdown

`asyncio.CancelledError` handling must preserve existing special cases where the current code deliberately re-raises it.

## Migration Strategy

### Stage 1: Extract Lifecycle Primitives

Create `lifecycle_workers.py` with `ManagedWorker`, `WorkerInventory`, and extracted stop behavior. Keep compatibility wrappers or aliases in `main.py` if needed so existing tests can be moved incrementally.

Success criteria:

- existing shutdown job-poller tests still pass
- app-state inventory shape is unchanged
- zero behavior change for started workers

### Stage 2: Migrate Compatible Stop-Event Workers

Move the simplest direct stop-event workers first. Good first candidates are workers that already expose a single stop event and a direct task:

- jobs metrics gauges
- jobs metrics reconcile
- jobs crypto rotate
- jobs webhooks
- meetings webhook DLQ
- workflows DLQ
- workflows artifact GC
- workflows DB maintenance
- jobs integrity
- TTS history cleanup

Success criteria:

- each migrated worker appears in inventory when enabled
- each migrated worker stops through the inventory path
- duplicate late-stop branches are removed or explicitly guarded as fallback

### Stage 3: Reassess Custom Loops And Schedulers

After the first migration batch, classify remaining lifecycle entries:

- stop-event worker
- service object with `start()` / `stop()`
- cron scheduler
- helper-started worker with hidden stop ownership
- intentionally excluded

Scheduler-style services should get a separate design or implementation slice if their lifecycle shape differs enough from stop-event workers.

## Testing Strategy

Automated tests:

- unit tests for `WorkerInventory` registration and inventory publication
- unit tests for concurrent stop behavior, timeout fallback, cancellation, and quiesced-name recording
- integration tests extending `test_main_shutdown_job_pollers.py` so migrated workers appear in inventory when their flags are enabled
- lifespan smoke coverage showing startup and shutdown remain reentrant
- regression tests proving already-quiesced workers are not stopped twice by unguarded late branches

Verification commands for implementation planning should include:

- targeted pytest for the new lifecycle helper tests
- targeted pytest for `tldw_Server_API/tests/Services/test_main_shutdown_job_pollers.py`
- targeted pytest for `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`
- Bandit on `tldw_Server_API/app/main.py`, `tldw_Server_API/app/services/lifecycle_workers.py`, and touched tests

## Acceptance Criteria

- A current-code lifecycle helper exists and does not depend on the stale WorkerRegistry design.
- App-state worker inventory remains backward compatible.
- At least one coherent batch of remaining stop-event workers is migrated into the shared inventory and shutdown path.
- Migrated workers no longer require duplicated unguarded late-stop code in `main.py`.
- Startup and shutdown behavior remains compatible with current environment flags and sidecar mode.
- Targeted service lifecycle tests pass.
- Bandit reports no new findings in touched scope.

## Follow-Up Work

After this slice, revisit issue #1114 for the remaining categories:

- service-object lifecycle entries such as storage cleanup and personalization consolidation
- helper-started workers with hidden stop ownership
- cron scheduler services
- lightweight retention cleanup schedulers

Those should be planned as follow-up batches based on the classification produced in Stage 3, not folded into the first stop-event worker migration.
