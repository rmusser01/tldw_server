# Worker Lifecycle Deprecated Code Removal Design

- Date: 2026-05-03
- Issue: #1114
- Topic: Complete the WorkerRegistry migration by removing deprecated lifecycle ownership paths
- Status: Ready for user review

## Goal

Close issue #1114 cleanly by turning the current partial WorkerRegistry migration into a single-owner lifecycle model and deleting deprecated fallback code that is no longer needed.

The current issue title still says "migrate remaining inline workers", but the repo has moved past that point. Most workers listed in the issue are already registered through `WorkerRegistry` or the underlying `lifecycle_workers.py` inventory model. The remaining work is now:

- prove each migrated worker has exactly one shutdown owner
- remove duplicate legacy stop paths after ownership is proven
- decide whether scheduler-style services need a small lifecycle adapter
- prune compatibility state and handle plumbing that no longer drives shutdown
- update #1114 with a final migration/deprecation map

This is a cleanup and completion plan, not a new worker feature.

## Current State

The active lifecycle implementation is `tldw_Server_API/app/services/lifecycle_workers.py`. `tldw_Server_API/app/services/worker_registry.py` is now a compatibility export for `ManagedWorker`, `ShutdownPhase`, `WorkerRegistry`, and `start_stop_event_worker`.

Already lifecycle-managed workers include:

- cleanup workers in `startup_cleanup_workers.py`: `ephemeral_cleanup_task`, `chatbooks_cleanup`, `storage_cleanup_service`
- claims rebuild in `startup_claims_rebuild.py`: `claims_rebuild`
- auxiliary services in `startup_auxiliary_services.py`: `usage_task`, `llm_usage_task`, claims alerts, claims review metrics
- compactor and WebSub workers in `startup_compactor_websub_workers.py`: `embeddings_compactor_task`, `websub_renewal_task`
- recurring schedulers in `startup_recurring_schedulers.py`: AuthNZ, workflows, reading digest, admin backup, companion reflection, reminders, connectors sync
- maintenance schedulers in `startup_maintenance_schedulers.py`: RAG quality eval, outputs purge, kanban activity cleanup, ingestion source archive cleanup, kanban purge, file artifacts export GC, notifications prune, jobs prune

The remaining risk is not startup registration coverage. The remaining risk is duplicated shutdown ownership: registered workers may still have explicit task/stop-event handles flowing through legacy shutdown helpers, with skip checks based on `stopped_background_worker_names` and coordinated legacy suppression.

## Non-Goals

- Reintroducing the stale standalone WorkerRegistry implementation.
- Rewriting startup ordering.
- Changing worker enablement flags or default enablement behavior.
- Changing job lease semantics for `job_poller_quiesce` workers.
- Removing diagnostics that are still used by tests, operators, or issue closeout evidence.
- Solving unrelated shutdown coordinator work outside issue #1114.

## Design Principles

### One Shutdown Owner Per Worker

A migrated worker should be stopped by exactly one path:

- `WorkerRegistry` / `stop_registered_workers` for lifecycle-managed workers
- a legacy direct stop only while that worker has not yet been proven registry-owned
- a domain-specific service shutdown path only when the service cannot be represented safely by `ManagedWorker`

Skip checks such as `stopped_background_worker_names` are acceptable during transition, but they should not become the permanent architecture for workers that are fully registry-owned.

### Delete Compatibility Code Only After a Testable Ownership Proof

Removal should be gated by tests that show:

- the worker is registered with the expected `name`
- it is assigned to the expected shutdown phase
- the registry stop path signals the expected stop event or shutdown callback
- the legacy direct-stop helper no longer receives that worker's handles
- shutdown remains best-effort and bounded when a worker exits, raises, or ignores cancellation

### Scheduler Abstraction Must Earn Its Place

Schedulers should not get a new abstraction only for naming symmetry. Add a helper such as `register_scheduler(...)` only if it removes repeated rollback/stopper code or makes ownership materially clearer.

`ManagedWorker` with `task=None` and `shutdown_callback=...` is acceptable for callback-only schedulers when it is simple and testable.

## Approach Considered

### Recommended: Ownership-First Cleanup

Start by proving current registry ownership, then delete deprecated direct-stop paths in phased batches. Scheduler normalization happens only after the direct-stop cleanup shows what duplication remains.

Pros:

- directly supports issue completion
- prioritizes deleted code over additional framework
- lowers double-stop risk
- preserves current startup behavior
- keeps each PR reviewable

Cons:

- requires careful shutdown tests before deletion
- may expose old compatibility assumptions in tests

### Alternative: Worker-Batch Cleanup

Group workers by domain and clean each group independently.

Pros:

- lower local risk per batch
- easy to map to service ownership

Cons:

- leaves duplicate shutdown concepts around longer
- weaker route to issue closure because compatibility cleanup remains scattered

### Alternative: Scheduler-First Abstraction

Build a `SchedulerRegistry` or scheduler-specific facade first, then move recurring and maintenance schedulers onto it.

Pros:

- cleaner taxonomy if scheduler behavior keeps diverging

Cons:

- likely premature
- adds code before deprecated paths are removed
- may obscure the simpler `ManagedWorker` callback model already present

## Chosen Design

Use ownership-first cleanup.

The implementation series should be a set of small PRs, each with a clear deletion target. A phase is complete only when it removes deprecated path usage or documents why a path must remain.

## Phase 1: Ownership Audit And Contract Tests

Goal: establish a test-backed inventory of what `WorkerRegistry` owns today.

Work:

- create a canonical ownership matrix for every #1114 worker with these columns:
  - issue checklist name
  - `ManagedWorker.name`
  - task name
  - runtime-state field
  - stopped-name key
  - legacy helper or direct-stop path
  - final target state
- create or update tests that enumerate full worker inventory from startup helpers
- assert each worker's `name`, `category`, and `shutdown_phase`
- assert background-phase workers do not appear in `_tldw_shutdown_job_poller_inventory`
- assert `stop_registered_workers` publishes stopped names for successful stop-event, callback-only, and task-cancel cases
- identify each remaining legacy direct-stop helper that still receives handles for already registry-owned workers

Primary files:

- `tldw_Server_API/app/services/lifecycle_workers.py`
- `tldw_Server_API/app/services/lifespan_shutdown_sequence.py`
- `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`
- focused service tests under `tldw_Server_API/tests/Services/`

Success criteria:

- every worker targeted for deletion has a row in the ownership matrix before code is removed
- registry-owned workers have explicit shutdown ownership in tests
- duplicate-stop candidates are listed in the test or design follow-up notes
- no production code deletion happens until these assertions are in place

## Phase 2: Remove Deprecated Direct Stops For Custom Loops

Goal: delete the oldest compatibility code for workers that now fit the registry stop-event model.

Initial deletion candidates:

- `ephemeral_cleanup_task`
- `chatbooks_cleanup`
- `storage_cleanup_service`
- `claims_rebuild`
- `embeddings_compactor_task`
- `websub_renewal_task`
- `usage_aggregator`
- `llm_usage_aggregator`

Work:

- stop passing obsolete task/stop-event handles into legacy shutdown helpers where registry ownership is sufficient
- remove or shrink direct-stop branches in `shutdown_pre_worker_cleanup.py`, `shutdown_post_worker_services.py`, `shutdown_usage_aggregators.py`, and related helper modules
- keep service-specific shutdown calls only when the service has state outside the registered task
- preserve non-stop finalizers and singleton resets even when direct worker stops are removed
- update `LifespanWorkerRuntimeState` only after downstream shutdown helpers no longer need a handle

Important boundary:

Removing a deprecated direct stop does not mean deleting all code in the same helper. For example, `storage_cleanup_service` worker stop ownership can move fully to `WorkerRegistry`, while storage cleanup singleton resets and AuthNZ limiter resets still need an explicit finalizer path unless a separate test proves they are obsolete.

Success criteria:

- each targeted worker is stopped through `background_worker_shutdown`
- direct legacy shutdown helpers no longer stop targeted workers
- non-stop finalizers still run when required
- tests prove no double stop and no lost shutdown
- deleted code exceeds added code for the phase

## Phase 3: Normalize Scheduler Ownership

Goal: decide whether scheduler services should remain plain `ManagedWorker` entries or receive a small helper.

Work:

- review recurring and maintenance scheduler registration patterns after Phase 2
- keep current direct `ManagedWorker` registrations if duplication is low
- add a minimal `register_scheduler(...)` helper only if it removes repeated task/callback/rollback code across scheduler modules
- do not create a separate registry object unless a scheduler-specific inventory or shutdown phase is needed
- retire duplicate scheduler stop-helper plumbing when a scheduler is proven to be registry callback-owned

Candidate scheduler groups:

- callback-only AuthNZ scheduler
- recurring scheduler tasks in `startup_recurring_schedulers.py`
- maintenance scheduler tasks in `startup_maintenance_schedulers.py`

Success criteria:

- scheduler shutdown ownership is explicit and tested
- callback-owned schedulers no longer flow through duplicate direct-stop helpers
- no new abstraction exists unless it removes duplicated code
- scheduler tests cover startup rollback and shutdown callback behavior

## Phase 4: Prune Compatibility State And Runtime Handles

Goal: remove stale bridge state once active shutdown no longer uses it.

Work:

- audit `LifespanWorkerRuntimeState` fields that only exist for deprecated direct-stop helpers
- remove fields after their consumers are gone
- reduce app-state compatibility inventories to what is still needed:
  - keep `_tldw_shutdown_worker_inventory` for diagnostics
  - keep `_tldw_shutdown_job_poller_inventory` only for job-poller compatibility consumers
  - remove internal stopped-name or legacy-plan state only after tests prove it is not part of active shutdown behavior
- update or remove tests that assert deprecated implementation details instead of behavior

Success criteria:

- lifecycle runtime state is smaller
- compatibility state has documented consumers
- legacy handle plumbing is absent for fully registry-owned background workers

## Phase 5: Issue Closeout

Goal: make #1114 closable with evidence.

Work:

- update the issue with a final table of original checklist entries:
  - migrated and deprecated path removed
  - migrated but intentionally retaining compatibility state
  - moved to a separate issue
  - no longer applicable
- link the PRs or commits for each phase
- add a short note explaining that the active registry is the `lifecycle_workers.py` facade, not the stale standalone design

Success criteria:

- #1114 has no ambiguous remaining checklist entries
- any intentionally retained compatibility path has a named reason and owner
- no broad "remaining workers" work is left hidden inside the issue

## Testing Strategy

Each implementation phase should run focused service tests before broader test selection.

Required focused tests:

- lifecycle worker registration and publication tests
- shutdown sequence tests that prove background workers stop before legacy fallbacks
- startup helper tests for each touched worker group
- rollback tests for registration failures when a task has already been started

Suggested commands:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_lifecycle_workers.py -q
python -m pytest tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -q
python -m pytest tldw_Server_API/tests/Services/test_startup_cleanup_workers.py -q
python -m pytest tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py -q
python -m pytest tldw_Server_API/tests/Services/test_startup_maintenance_schedulers.py -q
```

Before each PR is considered complete:

```bash
source .venv/bin/activate
python -m ruff check <touched-python-files>
python -m bandit -r <touched-production-paths> -f json -o /tmp/bandit_worker_lifecycle_<phase>.json
git diff --check
```

## Review Risks

- Removing a legacy fallback too early could drop shutdown for a service object whose task handle does not fully represent the service's state.
- Scheduler tasks may need stop callbacks even when their task is registry-owned.
- Existing tests may assert old handle plumbing rather than user-visible shutdown behavior.
- The app currently has a dirty local `dev` worktree in some sessions; implementation should happen in an isolated worktree or a clean branch before code changes.

## Completion Definition

This work is done when:

- every worker originally listed in #1114 has an explicit final state
- fully migrated workers have one shutdown owner
- deprecated direct-stop branches for those workers are removed
- scheduler ownership is either normalized or explicitly retained as-is
- compatibility state is reduced to documented diagnostics and active consumers
- focused tests cover ownership and shutdown behavior
- #1114 is updated with the final migration/deprecation map
