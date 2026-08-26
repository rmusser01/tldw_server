# Shared Workspace Clone Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the in-memory shared-workspace clone background task with one durable, owner-scoped Jobs operation exposed through canonical POST and GET envelopes and executed by a lifecycle-managed worker.

**Architecture:** The recipient API validates and fingerprints a bounded request, replays or atomically admits it through the generic Jobs receipt primitive, and projects only the canonical Workspace operation contract. A dedicated Sharing worker revalidates current authorization, runs the synchronous clone service in a worker thread, and completes the fenced Job while the copy remains hidden. Its post-completion finalizer uses an exact terminal-result CAS to select a durable publication or compensation checkpoint before exposing or deleting resources, then confirms the public result after media and the target Workspace are visible. A periodic bounded reconciliation pass resumes checkpointed work and scans active/archive Jobs with runtime-owned keyset cursors; exact terminal-result changes use an owner/domain/queue/type/status-scoped CAS that works against active or archived Jobs.

**Tech Stack:** FastAPI, Pydantic v2, Python asyncio and `asyncio.to_thread`, Jobs `WorkerSDK`, SQLite/PostgreSQL Jobs backends, AuthNZ shared-workspace repository, CharactersRAGDB, MediaDatabase, pytest.

**Spec:** `Docs/superpowers/specs/2026-08-25-shared-workspace-clone-jobs-design.md`

## Global Constraints

- Preserve `/research-workspace` and Shared With Me as separate product surfaces.
- Expose only `POST /api/v1/sharing/shared-with-me/{share_id}/clone` and `GET /api/v1/sharing/shared-with-me/{share_id}/clone/{operation_id}`; add no aliases or redirects.
- Use Jobs domain `sharing`, queue `workspace-clone`, job type `workspace_clone`, recipient ownership, and `max_retries=0`.
- Require an exact 16-200 character `Idempotency-Key` containing only `[A-Za-z0-9._~-]`; never persist the raw key.
- Keep Job payloads and public progress content-free and bounded; never persist source titles, URLs, paths, credentials, or authorization claims.
- Revalidate canonical active access and `allow_clone` before owner data access, at every CloneService cancellation boundary, and immediately before the durable publication checkpoint.
- Keep the clone target and operation-owned media hidden until durable fenced Job completion; expose media before the Workspace.
- Treat missing, foreign, malformed, ambiguous, and wrong-scope operation rows as neutral `404` or typed `503` exactly as the approved contract requires.
- Do not add vector generation in this task; report `needs_indexing` when applicable.
- Register the worker only through `provide_primary_jobs_worker_specs()` in shutdown phase `JOB_POLLER_QUIESCE`; do not extend `PrimaryJobsPollerHandles` or legacy startup paths.
- Frontend persistence, polling UI, and live CDP acceptance remain in TASK-12020.49 and TASK-12020.50.

## File Structure

- `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`: shared serialized Workspace operation base.
- `tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py`: strict clone request, progress, result, error, and operation envelopes.
- `tldw_Server_API/app/core/Sharing/shared_workspace_clone_operations.py`: request normalization, digest/fingerprint/target identity, admission command construction, strict Job projection, and safe error types.
- `tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py`: authorization bridge, thread-owned clone execution, publication finalization, reconciliation, audit, and the single application/standalone runner.
- `tldw_Server_API/app/core/Jobs/operations/contracts.py`: backend-neutral scoped terminal-result patch command/result contracts.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/terminal_result.py`: SQLite active/archive terminal-result CAS.
- `tldw_Server_API/app/core/Jobs/operations/postgres/terminal_result.py`: PostgreSQL parity for the same CAS.
- `tldw_Server_API/app/core/Jobs/manager.py`: sharing queue registration and public delegation to the backend-neutral terminal patch.
- `tldw_Server_API/app/core/Jobs/worker_sdk.py`: optional post-failure callback invoked only after a durable failure transition.
- `tldw_Server_API/app/api/v1/endpoints/sharing.py`: canonical POST/GET routes and removal of the FastAPI background task.
- `tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py`: authoritative clone capability.
- `tldw_Server_API/app/core/Sharing/share_audit_service.py`: requested/completed/failed event constants.
- `tldw_Server_API/app/services/startup_primary_jobs_pollers.py`: one declarative clone worker spec and dedicated route/flag/sidecar predicate.

---

### Task 1: Canonical Clone Contracts And Projection

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/workspace_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py`
- Create: `tldw_Server_API/app/core/Sharing/shared_workspace_clone_operations.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_clone_operations.py`

**Interfaces:**
- Consumes: `CreateJobCommand`, `IdempotentOperationCommand`, `WorkspaceOperationStatus`, and immutable clone result dataclasses.
- Produces: `normalize_clone_name(name: str | None) -> str | None`, `validate_idempotency_key(value: str) -> str`, `build_clone_admission_command(...) -> IdempotentOperationCommand`, `target_workspace_id(operation_id: str) -> str`, and `project_clone_operation(job: Mapping[str, Any], *, share_id: int, recipient_user_id: int) -> SharedWorkspaceCloneOperationResponse`.

- [x] **Step 1: Write failing contract tests**

Cover exact idempotency-key bounds/alphabet/no normalization, whitespace-normalized optional names, deterministic SHA-256 fingerprint and UUIDv5 target identity, deep-copy-safe bounded schemas, queued/running/completed/failed mappings, partial readiness, safe cleanup states, malformed result fail-closed behavior, wrong owner/domain/queue/type/share neutral not-found behavior, and POST/GET serialization parity.

- [x] **Step 2: Run the contract tests red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_shared_workspace_clone_operations.py`

Expected: collection fails because the clone contracts and operation module do not exist.

- [x] **Step 3: Add the shared Workspace operation base and strict clone models**

Make `WorkspaceOperationResponse` inherit a new base without changing its existing JSON fields. Define clone progress phases, count/readiness/warning result types, safe failure error, and `SharedWorkspaceCloneOperationResponse` with `extra="forbid"`, bounded strings/counts, `schema_version=1`, `command="shared_workspace_clone"`, and status-dependent result/error validation.

- [x] **Step 4: Implement deterministic normalization, admission construction, and projection**

Use canonical compact/sorted JSON for fingerprints, `hmac.compare_digest` where persisted digests are compared, a fixed documented UUIDv5 namespace for target IDs, a 31-day receipt expiry, and a strict projection that never forwards arbitrary payload/result/diagnostics fields.

- [x] **Step 5: Run the contract tests green**

Run the Step 2 command and require all tests to pass.

- [x] **Step 6: Commit the contract slice**

Commit: `feat(sharing): define canonical clone operations`

### Task 2: Scoped Terminal Result CAS And Worker Failure Hook

**Files:**
- Modify: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Create: `tldw_Server_API/app/core/Jobs/operations/sqlite/terminal_result.py`
- Create: `tldw_Server_API/app/core/Jobs/operations/postgres/terminal_result.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/app/core/Jobs/worker_sdk.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_terminal_operation_result_sqlite.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_terminal_operation_result_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_worker_sdk.py`

**Interfaces:**
- Consumes: exact Job UUID, recipient owner, domain, queue, job type, allowed terminal statuses, expected result fingerprint, and bounded replacement result.
- Produces: `JobManager.patch_terminal_operation_result(command: TerminalOperationResultPatchCommand) -> TerminalOperationResultPatchOutcome` with `APPLIED`, `IDEMPOTENT`, `MISSING`, and `CONFLICT`; `WorkerSDK.run(..., on_failed: FailureCallback | None = None)`.

- [x] **Step 1: Write failing SQLite/PostgreSQL CAS tests**

Verify active and archived updates, `updated_at` advancement, idempotent replay, duplicate active/archive authority rejection, wrong owner/scope/status/result rejection, concurrent winner behavior, JSON size enforcement, and no mutation on malformed correlation.

- [x] **Step 2: Write failing WorkerSDK callback tests**

Prove `on_failed(job, exc)` runs after and only after a durable terminal failure, is not called for retry scheduling or rejected terminalization, is bounded/isolated like completion callbacks, and does not suppress cancellation.

- [x] **Step 3: Run the new tests red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_terminal_operation_result_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_terminal_operation_result_postgres.py tldw_Server_API/tests/Jobs/test_worker_sdk.py`

Expected: missing terminal patch contracts/backends and unsupported `on_failed` argument.

- [x] **Step 4: Implement backend-neutral terminal patching**

Perform one transactionally consistent active/archive authority read, require one exact correlation and terminal status, compare the expected persisted result digest, then replace only that exact row. Keep backend SQL in the backend modules and expose one JobManager delegation method.

- [x] **Step 5: Add the post-failure callback without changing default WorkerSDK behavior**

Make failure finalization awaitable, inspect the exact owner-scoped Job after `fail_job`, invoke `on_failed` only when the durable state is terminal, and retain existing behavior when no callback is supplied.

- [x] **Step 6: Run parity and WorkerSDK tests green**

Run the Step 3 command. PostgreSQL may skip only when its canonical fixture explicitly reports unavailable; record that condition.

- [x] **Step 7: Commit the Jobs integration primitive**

Commit: `feat(jobs): add scoped terminal operation updates`

### Task 3: Recipient Clone Admission And Status APIs

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/sharing_schemas.py`
- Modify: `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_clone_endpoints.py`

**Interfaces:**
- Consumes: Task 1 admission/projection helpers, `try_get_job_manager`, canonical access service, and the generic receipt API.
- Produces: canonical POST and GET routes returning `SharedWorkspaceCloneOperationResponse`; no `BackgroundTasks`, old `CloneWorkspaceResponse`, alias, or redirect.

- [x] **Step 1: Write failing API tests**

Cover required header validation, allow-clone authorization, no Jobs typed `503`, created/converged/replayed admission, same-key replay after revocation, mismatch and same-share active conflicts, terminal `200` versus active `202`, owner isolation, neutral malformed/wrong-scope `404`, exact poll href, bounded typed errors, and identical POST/GET envelopes.

- [x] **Step 2: Write failing OpenAPI and route-ownership tests**

Assert the exact request/response models and statuses, one POST plus one GET path, no old `CloneWorkspaceResponse`, no `BackgroundTasks` clone runner, and recipient route-class error normalization.

- [x] **Step 3: Run the API tests red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_shared_workspace_clone_endpoints.py tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py`

- [x] **Step 4: Replace the endpoint atomically**

Check a receipt replay before current share resolution, resolve access only for new admission, require `allow_clone`, call `admit_idempotent_operation`, emit `share.clone_requested` only for `CREATED`, and map all known conflicts/unavailability to stable typed responses. Status reads use only owner-scoped Jobs data and remain available after revocation.

- [x] **Step 5: Remove obsolete clone schemas and background execution**

Delete `CloneWorkspaceRequest`, `CloneWorkspaceResponse`, `_run_clone_task`, and their imports. Do not preserve compatibility paths.

- [x] **Step 6: Run API/OpenAPI tests green**

Run the Step 3 command and require all tests to pass.

- [x] **Step 7: Commit the API slice**

Commit: `feat(sharing): expose durable clone operations`

### Task 4: Clone Worker, Publication, And Reconciliation

**Files:**
- Create: `tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_clone_jobs_worker.py`
- Create: `tldw_Server_API/tests/Sharing/test_shared_workspace_clone_reconciliation.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`

**Interfaces:**
- Consumes: Task 1 payload/result contracts, Task 2 terminal patch and failure callback, CloneService, canonical share repository, user-scoped DB loaders, and operation-owned publication/cleanup methods.
- Produces: `handle_shared_workspace_clone_job(job, *, runtime)`, `finalize_shared_workspace_clone(job, result, *, runtime)`, `reconcile_shared_workspace_clone_jobs(*, jobs, limit=100)`, and `run_shared_workspace_clone_jobs_worker(stop_event=None)`.

- [x] **Step 1: Write failing payload, authorization, and thread-boundary tests**

Reject malformed payloads and Job scope, revalidate active membership/`allow_clone` before owner DB resolution and through a thread-to-event-loop authorization bridge at CloneService boundaries, derive the target from Job UUID, pass progress/cancellation callbacks, use `asyncio.to_thread`, and close thread-local ChaCha/Media connections in the worker thread.

- [x] **Step 2: Write failing publication and hard-exit reconciliation tests**

Prove durable completion precedes exposure, media is exposed before Workspace, callback rejection cleans staged data, completed hidden copies are finalized, failed/cancelled/quarantined copies are discarded, unrelated media is untouched, cleanup result CAS is scoped, archived Jobs reconcile, all passes are bounded, and GET remains side-effect free.

- [x] **Step 3: Run worker tests red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_shared_workspace_clone_jobs_worker.py tldw_Server_API/tests/Sharing/test_shared_workspace_clone_reconciliation.py`

- [x] **Step 4: Implement the thread-owned clone boundary**

Resolve/cache DB objects only after initial authorization, open their thread-local connections inside `to_thread`, bridge each synchronous `should_cancel()` call to the event-loop share repository with a bounded timeout, classify only stable clone errors, disable retries, and serialize a validated terminal result with `publication_confirmed=false` so post-completion finalization owns authorization and exposure.

- [x] **Step 5: Implement post-completion publication**

After WorkerSDK fenced completion, keep the public operation in `running/finalizing`, reauthorize, and CAS the unconfirmed terminal result to `authorized` before mutation. Enumerate and confirm exact operation-owned media, clear the exact Workspace publication marker, then CAS to `publication_confirmed=true`. Revocation must win an `aborting` CAS before cleanup and records `aborted` only after cleanup succeeds. Treat zero-row idempotent replays as success only after re-reading deterministic state; never expose the Workspace before all owned media are active.

- [x] **Step 6: Implement bounded periodic reconciliation**

Run a scoped Jobs integrity sweep, scan at most 100 active/archive clone Jobs per pass with keyset cursor progression, resume authorized publication and aborting cleanup, finalize completed hidden targets, clean terminal failed targets/media, patch cleanup metadata through Task 2's CAS, and sleep interruptibly. Run this loop and WorkerSDK under the single runner and stop both from the same stop event.

- [x] **Step 7: Run worker/reconciliation and foundation regression tests green**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_shared_workspace_clone_jobs_worker.py tldw_Server_API/tests/Sharing/test_shared_workspace_clone_reconciliation.py tldw_Server_API/tests/Sharing/test_clone_service.py tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py`

- [x] **Step 8: Commit the worker slice**

Commit: `feat(sharing): execute clone jobs safely`

### Task 5: Capability, Audit, And Lifecycle Ownership

**Files:**
- Modify: `tldw_Server_API/app/core/Sharing/shared_workspace_access_service.py`
- Modify: `tldw_Server_API/app/core/Sharing/share_audit_service.py`
- Modify: `tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/sharing.py`
- Modify: `tldw_Server_API/app/core/Jobs/manager.py`
- Modify: `tldw_Server_API/app/services/startup_primary_jobs_pollers.py`
- Modify: `tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py`
- Modify: `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`
- Modify: `tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py`
- Modify: `tldw_Server_API/tests/Sharing/test_shared_workspace_clone_endpoints.py`
- Modify: `tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py`
- Modify: `tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py`
- Modify: `apps/tldw-frontend/e2e/workflows/research-workspace.shared-recipient.spec.ts`

**Interfaces:**
- Consumes: Task 4 single worker runner.
- Produces: authoritative `clone_workspace` capability, `SHARE_CLONE_REQUESTED`/`SHARE_CLONED`/`SHARE_CLONE_FAILED` constants, and one `shared_workspace_clone_jobs_task` worker spec.

- [x] **Step 1: Write failing capability and audit-order tests**

Expect `allowed=true` when `allow_clone=true`, `owner_disabled` when false, no `clone_deferred`, requested audit only after created admission, cloned only after successful finalization, failed only after durable failure, and bounded audit metadata without content/error text.

- [x] **Step 2: Write failing lifecycle tests**

Assert the sharing queue is allowed without an env override, worker flag defaults true, explicit false disables, Sharing route disabled prevents startup, sidecar mode prevents application ownership, phase is `JOB_POLLER_QUIESCE`, the spec appears exactly once in the catalog, and legacy handles/start functions do not own it.

- [x] **Step 3: Run the focused tests red**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing/test_shared_workspace_access_service.py tldw_Server_API/tests/Sharing/test_shared_workspace_recipient_endpoints.py tldw_Server_API/tests/Services/test_startup_primary_jobs_pollers.py tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py`

- [x] **Step 4: Implement policy, audit constants, queue, and declarative worker spec**

Use a dedicated predicate combining `SHARED_WORKSPACE_CLONE_JOBS_WORKER_ENABLED`, route key `sharing`, and `not context.sidecar_mode`; add only the lazy service delegate required by `stop_event_worker_spec`.

- [x] **Step 5: Run capability/audit/lifecycle tests green**

Run the Step 3 command plus `tldw_Server_API/tests/Sharing/test_sharing_endpoints.py`.

- [x] **Step 6: Commit the integration slice**

Commit: `feat(sharing): register clone worker lifecycle`

### Task 6: End-To-End Backend Parity, Documentation, And Security Gates

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `Docs/User_Guides/Server/Organizations_and_Sharing.md`
- Modify: `Docs/Code_Documentation/Jobs_Module.md`
- Modify: `backlog/tasks/task-12020.48 - Expose-and-execute-canonical-shared-workspace-clone-operations.md` through Backlog MCP only
- Create or modify: backend integration tests under `tldw_Server_API/tests/Sharing/` selected by the preceding slices

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verified SQLite/PostgreSQL API-worker flow, CI shard assignment, operational environment documentation, and completed TASK-12020.48 record.

- [ ] **Step 1: Add full API-to-worker acceptance tests before documentation**

Exercise create, response-loss replay, worker completion, partial vector readiness, status, archived replay, revocation before execution, fatal cleanup, owner isolation, and hard-exit reconciliation using real SQLite stores and the canonical PostgreSQL fixtures.

- [ ] **Step 2: Run the integration matrix**

Run: `source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest -q tldw_Server_API/tests/Sharing tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_postgres.py tldw_Server_API/tests/Jobs/test_jobs_terminal_operation_result_sqlite.py tldw_Server_API/tests/Jobs/test_jobs_terminal_operation_result_postgres.py tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle.py tldw_Server_API/tests/Workspaces/test_workspace_clone_target_lifecycle_postgres.py tldw_Server_API/tests/Media_DB/test_media_clone_snapshot_repository.py`

- [ ] **Step 3: Document operations and assign every new test to a CI shard**

Document the exact routes, header contract, queue/domain/type, 31-day receipt expiry, no automatic retry, worker flag/default, sidecar ownership rule, bounded reconciliation, and vector-indexing limitation. Update `.github/workflows/ci.yml` for every new test path.

- [ ] **Step 4: Run static and repository gates**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m compileall -q tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/app/core/Sharing tldw_Server_API/app/core/Jobs/operations
python -m ruff check tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/workspace_schemas.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/app/core/Sharing/shared_workspace_clone_operations.py tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py tldw_Server_API/app/core/Jobs/operations tldw_Server_API/app/services/startup_primary_jobs_pollers.py
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/sharing.py tldw_Server_API/app/api/v1/schemas/shared_workspace_recipient_schemas.py tldw_Server_API/app/core/Sharing/shared_workspace_clone_operations.py tldw_Server_API/app/core/Sharing/shared_workspace_clone_jobs_worker.py tldw_Server_API/app/core/Jobs/operations -f json -o /tmp/bandit_task_12020_48.json
python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml
git diff --check
```

- [ ] **Step 5: Self-review against the approved spec**

Confirm every public field/error/status is bounded, no raw key or owner content enters Jobs/audit/progress, no route aliases or redirects exist, no clone succeeds before durable completion, and frontend/CDP/vector indexing remain out of scope.

- [ ] **Step 6: Finalize the Backlog task and commit**

Record exact pass/skip counts, PostgreSQL fixture availability, Bandit result, touched files, residual risks, and commits through Backlog MCP. Commit: `docs(sharing): close clone operations task`

## Plan Self-Review

- **Spec coverage:** Tasks 1-6 cover the canonical envelope, replay/admission ordering, authorization, WorkerSDK execution, publication/reconciliation, capability policy, audit semantics, lifecycle ownership, SQLite/PostgreSQL parity, OpenAPI, security, documentation, and CI sharding. TASK-12020.49, TASK-12020.50, and TASK-12020.45 remain explicitly out of scope.
- **Publication crash safety:** Durable Job completion occurs while the target is still hidden. The awaited finalizer activates operation-owned media before clearing the Workspace marker. Hard exits can delay discoverability but cannot expose an incomplete clone; reconciliation repairs the hidden target.
- **Terminal metadata safety:** Cleanup/result repair does not use `update_job_result()`. It uses an exact owner/domain/queue/type/status/result CAS across active/archive storage so a stale reconciler cannot overwrite another operation.
- **Placeholder scan:** No TBD/TODO/follow-up implementation placeholders remain; each deferred product area has an existing task ID.
- **Type consistency:** API, admission, worker, finalizer, and reconciler all use Job UUID as `operation_id`, UUIDv5 as `workspace_id`, integer `share_id`, recipient ID serialized as the Jobs owner string, and one strict clone response model.
