# Worker Lifecycle Deprecated Code Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close issue #1114 by proving WorkerRegistry ownership, removing deprecated duplicate shutdown paths, and documenting the final state of every originally listed worker.

**Architecture:** Keep `tldw_Server_API/app/services/lifecycle_workers.py` as the lifecycle ownership center. Move workers toward one shutdown owner by deleting legacy direct-stop paths only after tests prove the corresponding `ManagedWorker.name`, shutdown phase, and finalizer behavior. Avoid new scheduler abstractions unless they remove repeated callback/rollback code.

**Tech Stack:** FastAPI lifespan helpers, asyncio tasks/events, pytest, loguru, ruff, bandit.

---

## Starting Conditions

Work from an isolated worktree because the main `dev` checkout may have unrelated local changes.

Recommended setup:

```bash
git fetch origin
git worktree add .worktrees/worker-lifecycle-cleanup-1114 -b codex/worker-lifecycle-cleanup-1114 origin/dev
cd .worktrees/worker-lifecycle-cleanup-1114
source .venv/bin/activate
```

Spec:

- `Docs/superpowers/specs/2026-05-03-worker-lifecycle-deprecated-code-removal-design.md`

Primary production files:

- `tldw_Server_API/app/services/lifecycle_workers.py`
- `tldw_Server_API/app/services/lifespan_shutdown_sequence.py`
- `tldw_Server_API/app/services/lifespan_worker_runtime_state.py`
- `tldw_Server_API/app/services/shutdown_pre_worker_cleanup.py`
- `tldw_Server_API/app/services/shutdown_post_worker_services.py`
- `tldw_Server_API/app/services/shutdown_usage_aggregators.py`
- `tldw_Server_API/app/services/shutdown_claims_maintenance_tasks.py`
- `tldw_Server_API/app/services/shutdown_notifications_compactor_websub_workers.py`
- `tldw_Server_API/app/services/shutdown_recurring_schedulers.py`
- `tldw_Server_API/app/services/startup_cleanup_workers.py`
- `tldw_Server_API/app/services/startup_claims_rebuild.py`
- `tldw_Server_API/app/services/startup_auxiliary_services.py`
- `tldw_Server_API/app/services/startup_compactor_websub_workers.py`
- `tldw_Server_API/app/services/startup_recurring_schedulers.py`
- `tldw_Server_API/app/services/startup_maintenance_schedulers.py`

Primary test files:

- Create: `tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py`
- Modify: `tldw_Server_API/tests/Services/test_lifecycle_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_usage_aggregators.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_claims_maintenance_tasks.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_notifications_compactor_websub_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_recurring_schedulers.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_cleanup_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_auxiliary_services.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_maintenance_schedulers.py`
- Modify: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`

## Task 1: Add The Ownership Matrix And Baseline Contract Tests

**Files:**

- Create: `tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py`
- Modify: `tldw_Server_API/tests/Services/test_lifecycle_workers.py`
- Modify: startup helper tests for any matrix entry whose inventory coverage is missing

- [x] **Step 1: Write the ownership matrix test file**

Create a test-owned matrix. Keep it in tests so implementation and closeout can import or inspect one canonical source without adding production surface.

```python
from __future__ import annotations

from dataclasses import dataclass

import pytest


@dataclass(frozen=True)
class WorkerOwnershipRow:
    issue_name: str
    managed_name: str
    task_name: str | None
    runtime_field: str | None
    stopped_name_key: str
    legacy_helper: str | None
    target_state: str


WORKER_OWNERSHIP_MATRIX = (
    WorkerOwnershipRow(
        issue_name="Ephemeral cleanup loop",
        managed_name="ephemeral_cleanup_task",
        task_name="ephemeral_cleanup_task",
        runtime_field="cleanup_task",
        stopped_name_key="ephemeral_cleanup_task",
        legacy_helper="shutdown_pre_worker_cleanup",
        target_state="registry-owned; direct cancel removed; finalizers retained",
    ),
    WorkerOwnershipRow(
        issue_name="Chatbooks cleanup",
        managed_name="chatbooks_cleanup",
        task_name="chatbooks_cleanup_task",
        runtime_field="chatbooks_cleanup_task",
        stopped_name_key="chatbooks_cleanup",
        legacy_helper="shutdown_pre_worker_cleanup",
        target_state="registry-owned; direct stop removed",
    ),
    WorkerOwnershipRow(
        issue_name="Storage cleanup service",
        managed_name="storage_cleanup_service",
        task_name=None,
        runtime_field="storage_cleanup_service",
        stopped_name_key="storage_cleanup_service",
        legacy_helper="shutdown_pre_worker_cleanup",
        target_state="registry-owned stop callback; singleton reset finalizer retained",
    ),
)


@pytest.mark.unit
def test_worker_ownership_matrix_has_unique_managed_names() -> None:
    names = [row.managed_name for row in WORKER_OWNERSHIP_MATRIX]
    assert len(names) == len(set(names))
```

Start with the first three rows, then add rows for `claims_rebuild`, `embeddings_compactor_task`, `websub_renewal_task`, `usage_aggregator`, `llm_usage_aggregator`, recurring schedulers, and maintenance schedulers before deleting code for each group.

- [x] **Step 2: Run the new matrix test**

Run:

```bash
python -m pytest tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py -q
```

Expected: pass after the matrix file is created.

- [x] **Step 3: Add registry stop contract coverage if missing**

In `test_lifecycle_workers.py`, verify these cases already exist or add them:

- stop-event workers publish stopped names after graceful event stop
- callback-only workers publish stopped names after callback success
- workers cancelled after timeout still publish stopped names
- background workers are omitted from `_tldw_shutdown_job_poller_inventory`

Use existing tests where possible instead of duplicating.

- [x] **Step 4: Add startup inventory assertions for the Phase 2 deletion candidates**

Update focused startup helper tests so each Phase 2 worker proves its managed name and phase:

- `test_startup_cleanup_workers.py`: `ephemeral_cleanup_task`, `chatbooks_cleanup`, `storage_cleanup_service`
- `test_startup_claims_rebuild.py`: `claims_rebuild`
- `test_startup_compactor_websub_workers.py`: `embeddings_compactor_task`, `websub_renewal_task`
- `test_startup_auxiliary_services.py`: `usage_aggregator`, `llm_usage_aggregator`

Expected assertions:

```python
assert handle.name == "chatbooks_cleanup"
assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
assert app.state._tldw_shutdown_job_poller_inventory == []
```

- [x] **Step 5: Run Phase 1 focused tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  tldw_Server_API/tests/Services/test_lifecycle_workers.py \
  tldw_Server_API/tests/Services/test_startup_cleanup_workers.py \
  tldw_Server_API/tests/Services/test_startup_auxiliary_services.py \
  tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py \
  -q
```

Expected: pass.

- [x] **Step 6: Commit Task 1**

```bash
git add tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  tldw_Server_API/tests/Services/test_lifecycle_workers.py \
  tldw_Server_API/tests/Services/test_startup_cleanup_workers.py \
  tldw_Server_API/tests/Services/test_startup_auxiliary_services.py \
  tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py
git commit -m "test: document worker lifecycle ownership"
```

## Task 2: Remove Deprecated Direct Stops For Cleanup Workers

**Files:**

- Modify: `tldw_Server_API/app/services/shutdown_pre_worker_cleanup.py`
- Modify: `tldw_Server_API/app/services/lifespan_shutdown_sequence.py`
- Modify: `tldw_Server_API/app/services/lifespan_worker_runtime_state.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py`
- Modify: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`
- Modify: `tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py`

- [x] **Step 1: Write failing tests for cleanup direct-stop removal**

Change or add tests that assert `shutdown_pre_worker_cleanup` no longer directly cancels or stops registry-owned worker handles:

```python
@pytest.mark.asyncio
async def test_shutdown_pre_worker_cleanup_does_not_direct_stop_registry_owned_cleanup_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_cleanup = _import_shutdown_pre_worker_cleanup()
    cleanup_task = _FakeTask()
    chatbooks_task = _FakeTask()
    chatbooks_stop = _FakeStopEvent()
    storage_cleanup_service = _FakeStorageCleanupService()
    reset_calls: list[str] = []

    async def _record_cleanup() -> None:
        reset_calls.append("cleanup")

    async def _record_storage() -> None:
        reset_calls.append("storage")

    async def _record_auth() -> None:
        reset_calls.append("auth")

    monkeypatch.setattr(shutdown_cleanup, "_reset_cleanup_service", _record_cleanup)
    monkeypatch.setattr(shutdown_cleanup, "_reset_storage_service", _record_storage)
    monkeypatch.setattr(shutdown_cleanup, "_reset_authnz_rate_limiter", _record_auth)

    await shutdown_cleanup._shutdown_pre_worker_cleanup(
        app=SimpleNamespace(state=SimpleNamespace()),
        cleanup_task=cleanup_task,
        chatbooks_cleanup_task=chatbooks_task,
        chatbooks_cleanup_stop_event=chatbooks_stop,
        storage_cleanup_service=storage_cleanup_service,
        coordinated_legacy_component_names=set(),
        stopped_background_worker_names={
            "ephemeral_cleanup_task",
            "chatbooks_cleanup",
            "storage_cleanup_service",
        },
        guard_exceptions=(RuntimeError,),
    )

    assert cleanup_task.cancelled is False
    assert chatbooks_stop.is_set is False
    assert chatbooks_task.cancelled is False
    assert storage_cleanup_service.stopped is False
    assert reset_calls == ["cleanup", "storage", "auth"]
```

Keep or add a separate test proving finalizers still run:

```python
assert reset_calls == ["cleanup", "storage", "auth"]
```

- [x] **Step 2: Run tests and verify the new deletion expectation fails**

Run:

```bash
python -m pytest tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py -q
```

Expected before implementation: at least one failure showing the helper still performs a direct stop or returns obsolete handles.

- [x] **Step 3: Remove direct worker stop responsibility from `shutdown_pre_worker_cleanup.py`**

Keep:

- deferred startup cancellation
- storage cleanup singleton reset
- storage quota singleton reset
- AuthNZ limiter reset

Remove or bypass:

- direct `cleanup_task.cancel()`
- direct `chatbooks_cleanup_stop_event.set()`
- direct `chatbooks_cleanup_task.cancel()`
- direct `storage_cleanup_service.stop()`

Only keep a direct stop path if the ownership matrix marks that worker as not registry-owned.

- [x] **Step 4: Stop passing cleanup worker handles once consumers are gone**

After tests prove finalizers no longer need worker handles, update:

- `lifespan_shutdown_sequence.py`
- `lifespan_worker_runtime_state.py`
- any dataclasses that only exist to carry cleanup worker stop handles

Do not remove a field until `rg` shows no active shutdown consumer besides tests and startup return compatibility.

- [x] **Step 5: Run cleanup shutdown and lifecycle contract tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_startup_cleanup_workers.py \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  -q
```

Expected: pass.

- [x] **Step 6: Commit Task 2**

```bash
git add tldw_Server_API/app/services/shutdown_pre_worker_cleanup.py \
  tldw_Server_API/app/services/lifespan_shutdown_sequence.py \
  tldw_Server_API/app/services/lifespan_worker_runtime_state.py \
  tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py
git commit -m "refactor: remove legacy cleanup worker stops"
```

## Task 3: Remove Deprecated Direct Stops For Custom Background Workers

**Files:**

- Modify: `tldw_Server_API/app/services/shutdown_post_worker_services.py`
- Modify: `tldw_Server_API/app/services/shutdown_claims_maintenance_tasks.py`
- Modify: `tldw_Server_API/app/services/shutdown_notifications_compactor_websub_workers.py`
- Modify: `tldw_Server_API/app/services/shutdown_usage_aggregators.py`
- Modify: `tldw_Server_API/app/services/lifespan_shutdown_sequence.py`
- Modify: `tldw_Server_API/app/services/lifespan_worker_runtime_state.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_claims_maintenance_tasks.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_notifications_compactor_websub_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_usage_aggregators.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_claims_rebuild.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_auxiliary_services.py`
- Modify: `tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py`

- [x] **Step 1: Write failing tests for claims/compactor/WebSub/usage direct-stop removal**

Add tests that pass these workers as already stopped through background registry:

- `claims_rebuild`
- `embeddings_compactor_task`
- `websub_renewal_task`
- `usage_aggregator`
- `llm_usage_aggregator`

Expected assertions:

```python
assert calls == []
assert handles.claims_task is None
assert handles.embeddings_compactor_task is None
assert handles.websub_renewal_task is None
assert handles.usage_task is None
assert handles.llm_usage_task is None
```

- [x] **Step 2: Run targeted shutdown tests and verify failures**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py \
  tldw_Server_API/tests/Services/test_shutdown_claims_maintenance_tasks.py \
  tldw_Server_API/tests/Services/test_shutdown_notifications_compactor_websub_workers.py \
  tldw_Server_API/tests/Services/test_shutdown_usage_aggregators.py \
  -q
```

Expected before implementation: failures for legacy direct-stop behavior that still runs without registry ownership suppression.

- [x] **Step 3: Remove direct-stop calls for registry-owned custom background workers**

Implementation targets:

- `shutdown_claims_maintenance_tasks.py`: remove direct claims rebuild stop once `claims_rebuild` is registry-owned
- `shutdown_notifications_compactor_websub_workers.py`: remove direct compactor/WebSub stops once registry-owned
- `shutdown_usage_aggregators.py`: remove direct usage and LLM usage stops once registry-owned
- `shutdown_post_worker_services.py`: stop passing these handles into legacy helper calls once the helper no longer owns them

Keep unrelated shutdown behavior for workers not yet proven registry-owned.

- [x] **Step 4: Remove obsolete runtime fields in a second pass**

After helper signatures no longer use the handles, remove corresponding `LifespanWorkerRuntimeState` fields and startup-tail handle pass-through only when `rg` proves they are no longer read by active shutdown:

```bash
rg -n "claims_task|embeddings_compactor_task|websub_renewal_task|usage_task|llm_usage_task" \
  tldw_Server_API/app/services tldw_Server_API/tests/Services
```

Do not remove fields still needed for startup tests, diagnostics, or issue closeout until the consuming test is updated to behavior-based assertions.

- [x] **Step 5: Run custom-worker focused tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py \
  tldw_Server_API/tests/Services/test_shutdown_claims_maintenance_tasks.py \
  tldw_Server_API/tests/Services/test_shutdown_notifications_compactor_websub_workers.py \
  tldw_Server_API/tests/Services/test_shutdown_usage_aggregators.py \
  tldw_Server_API/tests/Services/test_startup_claims_rebuild.py \
  tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py \
  tldw_Server_API/tests/Services/test_startup_auxiliary_services.py \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  -q
```

Expected: pass.

- [x] **Step 6: Commit Task 3**

```bash
git add tldw_Server_API/app/services/shutdown_post_worker_services.py \
  tldw_Server_API/app/services/shutdown_claims_maintenance_tasks.py \
  tldw_Server_API/app/services/shutdown_notifications_compactor_websub_workers.py \
  tldw_Server_API/app/services/shutdown_usage_aggregators.py \
  tldw_Server_API/app/services/lifespan_shutdown_sequence.py \
  tldw_Server_API/app/services/lifespan_worker_runtime_state.py \
  tldw_Server_API/tests/Services
git commit -m "refactor: remove legacy custom worker stops"
```

## Task 4: Normalize Scheduler Ownership And Remove Duplicate Scheduler Stops

**Files:**

- Modify: `tldw_Server_API/app/services/startup_recurring_schedulers.py`
- Modify: `tldw_Server_API/app/services/startup_maintenance_schedulers.py`
- Modify: `tldw_Server_API/app/services/shutdown_recurring_schedulers.py`
- Modify: `tldw_Server_API/app/services/shutdown_post_worker_services.py`
- Modify: `tldw_Server_API/app/services/shutdown_authnz_scheduler.py`
- Modify: `tldw_Server_API/app/services/shutdown_telemetry_services.py`
- Modify: `tldw_Server_API/app/services/shutdown_final_cleanup_tail.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_maintenance_schedulers.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_recurring_schedulers.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_telemetry_services.py`
- Modify: `tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py`

- [x] **Step 1: Add scheduler rows to the ownership matrix**

Rows should cover:

- `authnz_scheduler`
- `workflows_sched_task`
- `reading_digest_sched_task`
- `admin_backup_sched_task`
- `companion_reflection_sched_task`
- `reminders_sched_task`
- `connectors_sync_sched_task`
- maintenance scheduler task names currently registered in `startup_maintenance_schedulers.py`

- [x] **Step 2: Write failing tests for duplicate scheduler stop removal**

In `test_shutdown_recurring_schedulers.py` and related shutdown tests, prove registry callback-owned schedulers are not sent through duplicate direct-stop helpers after `background_worker_shutdown` has already stopped them.

Expected pattern:

```python
await run_shutdown_post_worker_services(
    workflows_sched_task="workflow-task",
    stopped_background_worker_names={"workflows_sched_task"},
    # Supply the existing required keyword arguments with None or empty sets.
    # The assertion target is that the workflow scheduler stopper is not called.
)
assert stop_calls == []
```

- [x] **Step 3: Decide whether `register_scheduler(...)` is needed**

Before implementation, inspect duplication in recurring and maintenance scheduler registration.

Use this decision rule:

- If all scheduler registrations are readable and have different stopper needs, keep direct `ManagedWorker(...)`.
- If at least three registrations repeat the same task/callback/rollback pattern, add a small helper in `lifecycle_workers.py` or a scheduler startup helper module.
- Do not add a new registry class.

- [x] **Step 4: Remove duplicate scheduler direct-stop plumbing**

For callback-owned recurring schedulers, prefer registry callback ownership. Remove or bypass direct stop calls in:

- `shutdown_recurring_schedulers.py`
- `shutdown_post_worker_services.py`
- `shutdown_authnz_scheduler.py` only after AuthNZ callback ownership is proven

Keep final cleanup behavior that is not a scheduler task stop.

- [x] **Step 5: Run scheduler tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py \
  tldw_Server_API/tests/Services/test_startup_maintenance_schedulers.py \
  tldw_Server_API/tests/Services/test_shutdown_recurring_schedulers.py \
  tldw_Server_API/tests/Services/test_shutdown_telemetry_services.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  -q
```

Expected: pass.

- [x] **Step 6: Commit Task 4**

```bash
git add tldw_Server_API/app/services/startup_recurring_schedulers.py \
  tldw_Server_API/app/services/startup_maintenance_schedulers.py \
  tldw_Server_API/app/services/shutdown_recurring_schedulers.py \
  tldw_Server_API/app/services/shutdown_post_worker_services.py \
  tldw_Server_API/app/services/shutdown_authnz_scheduler.py \
  tldw_Server_API/app/services/shutdown_telemetry_services.py \
  tldw_Server_API/app/services/shutdown_final_cleanup_tail.py \
  tldw_Server_API/tests/Services
git commit -m "refactor: consolidate scheduler shutdown ownership"
```

## Task 5: Prune Compatibility State And Runtime Handle Plumbing

**Files:**

- Modify: `tldw_Server_API/app/services/lifespan_worker_runtime_state.py`
- Modify: startup handle dataclasses that only carry removed shutdown handles
- Modify: shutdown helper dataclasses that now only return removed handles
- Modify: behavior tests that asserted implementation detail handles
- Modify: `tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py`

- [x] **Step 1: Audit remaining runtime-state fields**

Run:

```bash
rg -n "cleanup_task|chatbooks_cleanup_task|chatbooks_cleanup_stop_event|storage_cleanup_service|claims_task|usage_task|llm_usage_task|workflows_sched_task|reading_digest_sched_task|admin_backup_sched_task|companion_reflection_sched_task|reminders_sched_task|connectors_sync_sched_task" \
  tldw_Server_API/app/services tldw_Server_API/tests/Services
```

Mark each field as:

- active startup return compatibility
- active shutdown input
- diagnostics only
- removable

- [x] **Step 2: Write tests that assert behavior instead of handle plumbing**

Where tests currently assert specific handle pass-through, replace with:

- inventory contains the worker
- shutdown stopped-name list contains the worker
- finalizer still ran
- direct-stop helper was not called

- [x] **Step 3: Remove removable fields and dataclass members**

Remove fields only when Step 1 marks them removable and tests no longer depend on them.

Likely targets after Tasks 2-4:

- cleanup direct-stop handles
- usage aggregator direct-stop handles
- recurring scheduler direct-stop handles
- maintenance scheduler direct-stop handles

- [x] **Step 4: Preserve documented diagnostics**

Keep:

- `_tldw_shutdown_worker_inventory`
- `_tldw_shutdown_job_poller_inventory` for job-poller compatibility consumers

Do not remove stopped-name state until all duplicate fallback logic has been removed or a test proves it is only diagnostic.

- [x] **Step 5: Run lifecycle service test set**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_lifecycle_workers.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py \
  tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py \
  -q
```

Expected: pass.

- [x] **Step 6: Commit Task 5**

```bash
git add tldw_Server_API/app/services tldw_Server_API/tests/Services
git commit -m "refactor: prune legacy lifecycle handle plumbing"
```

## Task 6: Final Verification And Issue Closeout

**Files:**

- Modify: `Docs/superpowers/specs/2026-05-03-worker-lifecycle-deprecated-code-removal-design.md` only if implementation discovers a deliberate scope change
- Modify: `Docs/superpowers/plans/2026-05-03-worker-lifecycle-deprecated-code-removal-implementation-plan.md` only for checklist status if this plan is used as execution tracker
- GitHub issue: `https://github.com/rmusser01/tldw_server/issues/1114`

- [x] **Step 1: Run focused lifecycle tests**

Run:

```bash
python -m pytest \
  tldw_Server_API/tests/Services/test_lifecycle_workers.py \
  tldw_Server_API/tests/Services/test_worker_lifecycle_ownership_matrix.py \
  tldw_Server_API/tests/Services/test_main_lifecycle_contract.py \
  tldw_Server_API/tests/Services/test_startup_cleanup_workers.py \
  tldw_Server_API/tests/Services/test_startup_auxiliary_services.py \
  tldw_Server_API/tests/Services/test_startup_compactor_websub_workers.py \
  tldw_Server_API/tests/Services/test_startup_recurring_schedulers.py \
  tldw_Server_API/tests/Services/test_startup_maintenance_schedulers.py \
  tldw_Server_API/tests/Services/test_shutdown_pre_worker_cleanup.py \
  tldw_Server_API/tests/Services/test_shutdown_post_worker_services.py \
  tldw_Server_API/tests/Services/test_shutdown_usage_aggregators.py \
  tldw_Server_API/tests/Services/test_shutdown_recurring_schedulers.py \
  -q
```

Expected: pass.

- [x] **Step 2: Run lint and security checks on touched Python files**

Run:

```bash
python -m ruff check tldw_Server_API/app/services tldw_Server_API/tests/Services
python -m bandit -r tldw_Server_API/app/services -f json -o /tmp/bandit_worker_lifecycle_1114.json
git diff --check
```

Expected:

- Ruff passes for touched scope, or any pre-existing unrelated findings are documented with narrower touched-file rerun.
- Bandit JSON has no new findings in touched production files.
- `git diff --check` passes.

- [x] **Step 3: Update issue #1114 with final migration/deprecation table**

Use the ownership matrix to post a concise closeout comment:

```bash
gh issue comment 1114 --repo rmusser01/tldw_server --body-file /tmp/issue-1114-closeout.md
```

The comment should include:

- original issue checklist name
- final `ManagedWorker.name`
- final shutdown owner
- deprecated path removed
- retained compatibility state, if any
- follow-up issue link, if any

- [x] **Step 4: Commit final docs or closeout updates**

If any docs or plan checklist statuses changed:

```bash
git add Docs/superpowers/specs/2026-05-03-worker-lifecycle-deprecated-code-removal-design.md \
  Docs/superpowers/plans/2026-05-03-worker-lifecycle-deprecated-code-removal-implementation-plan.md
git commit -m "docs: close worker lifecycle cleanup plan"
```

- [x] **Step 5: Open or update PR**

```bash
git push -u origin codex/worker-lifecycle-cleanup-1114
gh pr create --repo rmusser01/tldw_server --base dev --head codex/worker-lifecycle-cleanup-1114 --title "refactor: complete worker lifecycle cleanup" --body-file /tmp/pr-worker-lifecycle-cleanup-1114.md
```

PR body must include a human-editable `Change summary` section explaining what changed and why, per the AI-generated PR policy.

## Final Acceptance Criteria

- Every original #1114 checklist item has a row in the ownership matrix.
- Fully migrated workers have exactly one shutdown owner.
- Deprecated direct-stop branches for registry-owned workers are removed.
- Non-stop finalizers and singleton resets still run where required.
- Scheduler duplicate direct-stop helpers are removed or explicitly retained with a reason.
- Runtime handle plumbing is smaller and has no obsolete active shutdown consumers.
- Focused lifecycle tests pass.
- Ruff, Bandit, and `git diff --check` pass on touched scope.
- Issue #1114 is updated with the final migration/deprecation map.
