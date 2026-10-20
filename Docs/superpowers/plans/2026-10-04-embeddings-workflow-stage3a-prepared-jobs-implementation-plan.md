# Embeddings Stage 3A Prepared Jobs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a dormant, domain-neutral Jobs capability that admits exact owned roots, fences enlisted domain transactions, retains logical-effect proofs, and bounds retry-neutral recovery.

**Architecture:** Add a narrow capability beside the existing Admin Webhook prepared path, not an expansion of its DTOs or origins. Jobs owns capability metadata and effect receipts on its configured backend; new SQL and migrations live behind DB_Management abstractions. Domain repositories enlist in the same synchronous transaction and validate their own facts after the Jobs row is locked.

**Tech Stack:** Existing Python/dataclasses, SQLite, PostgreSQL/psycopg, pytest, Hypothesis, Ruff, and Bandit; no new dependency.

**Spec:** [Approved Stage 3 design](../specs/2026-10-02-embeddings-workflow-stage3-durable-store-jobs-runner-design.md).

**Tracking:** Planning TASK-13902; approved design TASK-13901; implementation TASK-13900. Historical TASK-13421/13450/13451 records are archived because upstream reused those numbers for unrelated work.

**ADR check:** ADR required: yes; [accepted ADR-060](../../ADR/060-embeddings-durable-workflow-foundation.md) already governs these persistence and authority rules. Preserve ADR-022 and the upstream Jobs completion decision ADR-058. No additional architecture decision is proposed here.

## Global Constraints

- One root for `embeddings/workflow/embedding_workflow`; its payload contains only `workflow_id`. Jobs code receives a trusted scope/payload-field policy and never imports Embeddings.
- Owner identity is nonblank and exact, at most 512 UTF-8 bytes. Reference and Jobs UUIDs are canonical UUIDv4; integer Jobs IDs are hints only.
- Explicit `ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT`; persist and validate retry/quarantine controls and a no-progress limit of at most 5 on replay, binding, and resume.
- Jobs alone owns lifecycle/quarantine. No provider calls, domain DLQ, terminal revival, or replacement of a previously bound root.
- Logical facts and acknowledgements are at most 4 KiB. Fixed reason codes only; no arbitrary exception text, input, credentials, provider bodies, paths, URLs, or metric identifiers.
- Strict unexpired lease fencing even when legacy enforcement flags are off. Database time governs leases and schedules; local monotonic time governs caller deadlines.
- Lock Jobs first, then capability metadata and domain workflow/generation/attempt/output/operations. No adapter call or `await` inside a transaction.
- Explicit migrations only. Ordinary Jobs construction, imports, application startup, and legacy migrations do not install this support schema or admit roots.
- PostgreSQL runtime is non-superuser/non-BYPASSRLS with transaction-local owner scope. No missing-owner or cross-owner access, including receipt cleanup.
- SQLite/PostgreSQL must agree on outcomes and crash windows. Use existing `jobs_pg_dsn`/`pg_temp_db`; fixture-declared unavailability is not backend certification.
- Queue configuration, admission quotas, pause/drain, counters/events, legacy roots, Admin Webhook validation, and completion UUID fencing remain intact.

## Review Focus

1. Same numeric ID reused for another UUID: never mutate or acknowledge the replacement root (Tasks 2/3).
2. Accepted admission replay after queue policy becomes restrictive: resolve its exact retained authority before rejecting new work; never create another root (Task 2).
3. Commit succeeds but acknowledgement is lost and history moves/prunes: exact proof remains replayable without spending another retry (Task 3).
4. Lease expires before the handler starts, or acquire and sweep race: one episode charge, finite termination, no provider retry consumption (Task 4).
5. Owner changes on a reused PostgreSQL connection, or RLS setup fails: fail closed rather than retaining prior/admin authority (Task 5).

## Execution Preflight

Planning used checkout `1077b51a45` and compared relevant Jobs boundaries with local `origin/dev` at `75ab224081bf140ef52017c1a9b0a04f6878d488`. No rebase or runtime change occurred while writing this plan.

- [x] Use an isolated execution checkout based on then-current `dev`, carrying only this workstream's reviewed docs/tasks. Rebased onto `454974e812` on 2026-10-20 (first parent `df7c290fbf`, the initially observed tip); shared checkout untouched. Current instructions use backlog-py.
- [x] Recheck task IDs and ADR numbering against current `dev` and open PR inventories; preserve accepted ADRs and the archived design history. Fresh tasks TASK-13900/13901/13902; ADR-060 remains available.
- [x] Confirm `JobManager.complete_job` and WorkerSDK preserve the newer `expected_uuid` completion fence. Current-dev code retained; completion baseline included below.
- [x] Search for concurrent neutral-capability work before creating files. No matching implementation on current dev; open Jobs changes #2862/#2738 are unrelated.
- [x] Create collision-free Stage 3A implementation Backlog task TASK-13900 before runtime edits. TASK-13902 records planning only.
- [x] Activate the project virtual environment before all Python/test/lint/security commands. Fresh inline/types/completion baseline: 63 passed, 9 `RUN_JOBS` gate skips, 144 warnings. Enabled PostgreSQL completion baseline: 9 passed, 15 deselected, 30 warnings. Existing SQLite/WorkerSDK baseline: 273 passed; CI/task-format checks: 64 passed. New capability certification remains required.

## File Map

Create these focused modules:

- `tldw_Server_API/app/core/Jobs/operations/workflow_contracts.py`: frozen commands, identity/control snapshots, bounded receipts, typed outcomes, and fingerprints.
- `tldw_Server_API/app/core/Jobs/workflow_capability.py`: capability facade, owner-scoped transaction lifetime, and fail-closed readiness; no SQL or application wiring.
- `tldw_Server_API/app/core/DB_Management/jobs_workflow_migrations.py`: explicit version-1 support migration and structural/security readiness checks.
- `tldw_Server_API/app/core/DB_Management/jobs_workflow_store.py`: transaction/store protocols and backend dispatch; no domain knowledge.
- `tldw_Server_API/app/core/DB_Management/jobs_workflow_sqlite.py`: transaction-neutral SQLite admission, lookup, fences, receipts, and recovery SQL.
- `tldw_Server_API/app/core/DB_Management/jobs_workflow_postgres.py`: matching PostgreSQL SQL with row/advisory locks and owner RLS.
- `tldw_Server_API/tests/Jobs/workflow_test_support.py`: backend-neutral test case, explicit schema installation, synthetic enlisted domain row, and controlled crash/race helpers.
- `tldw_Server_API/tests/Jobs/test_workflow_contracts.py`, `test_workflow_schema.py`, `test_workflow_admission_fencing.py`, `test_workflow_effect_receipts.py`, `test_workflow_expiry_progress.py`, `test_workflow_security_compatibility.py`: each named behavior boundary.
- `tldw_Server_API/tests/Jobs/property/test_workflow_capability_properties.py`: bounded model versus real backend sequences.

Modify only the required integration points:

- `tldw_Server_API/app/core/Jobs/manager.py`: additive capability factory/admission policy preparation and enrolled-root expiry/prune dispatch; no broad extraction.
- `tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py` and `operations/postgres/lifecycle.py`: record an enrolled acquisition atomically before returning a lease.
- `tldw_Server_API/tests/Jobs/conftest.py`: add explicitly requested backend fixtures; do not change existing autouse defaults.
- `Docs/Development/Embeddings_Stage3A_Jobs_Capability.md`: internal contract and explicit install/rollback instructions, added with Task 5.
- `.github/workflows/ci.yml`: only if the existing Jobs shard coverage contract reports the new tests uncovered; use its current shard mechanism.

No Embeddings production module, endpoint, worker registration, application lifespan, or generic Jobs migration entrypoint is edited.

## Stage 1: Immutable Contracts And Explicit Schema

**Goal:** Validate capability facts and install an inert, owner-scoped support schema.
**Success Criteria:** Invalid identities/facts reject before I/O; repeat migration is safe; readiness rejects incomplete or insecure schema without repair.
**Tests:** Contract boundary/property cases, fresh/repeat/rollback migration, dormancy.
**Status:** Complete

### Task 1: Contracts, Store Protocol, And Schema

**Files:** Create the contracts, capability skeleton, migration/store/backend modules, test support, contract/schema tests; add only the new fixtures to Jobs `conftest.py`.

**Interfaces:** All following names live in `workflow_contracts.py` unless stated otherwise. Frozen dataclasses defensively copy nested values; no mutable dictionaries escape.

```text
WorkflowScope(domain: str, queue: str, job_type: str)
WorkflowPolicy(scope: WorkflowScope, payload_field: str, reason_codes: frozenset[str])
WorkflowControls(max_retries: int, quarantine_threshold: int | None,
                 expired_lease_policy: ExpiredLeasePolicy,
                 no_progress_limit: int = 5)
WorkflowAdmissionCommand(owner_user_id: str, reference_id: str,
                         idempotency_key: str, controls: WorkflowControls)
WorkflowRoot(job_uuid: str, owner_user_id: str, scope: WorkflowScope,
             reference_id: str, idempotency_key: str, controls: WorkflowControls,
             job_id_hint: int | None = None)
WorkflowLease(worker_id: str, lease_id: str)
WorkflowEffect(effect_key: str, action: WorkflowAction,
               guard_fingerprint: str, reason_code: str | None = None,
               not_before_at: datetime | None = None)
WorkflowProgress(proof_key: str, kind: WorkflowProgressKind, facts_fingerprint: str)
WorkflowReceipt(effect_key: str, facts_fingerprint: str, job_uuid: str,
                action: WorkflowAction, resulting_status: str,
                applied_at: datetime, released: bool = False)
workflow_effect_fingerprint(root: WorkflowRoot, effect: WorkflowEffect) -> str
bounded_workflow_json(raw: bytes) -> object
```

Closed enums: `WorkflowAction` is `continue`, `retry`, `complete`, `fail`, `cancel`; `WorkflowProgressKind` is `root_binding`, `output_receipt`, `cursor`, `accounting`, `sink_lifecycle`. Result enum `WorkflowOutcome` is `applied`, `already_applied`, `active`, `archived`, `missing`, `conflict`, `stale_lease`, `terminal`, `insufficient_horizon`, `rejected`, `unavailable`, `exhausted`. Define `WorkflowAdmissionResult(outcome, root | None)`, `WorkflowLookupResult(outcome, root | None, status | None)`, `WorkflowEffectResult(outcome, receipt | None)`, `WorkflowHorizonResult(outcome, leased_until | None, guaranteed_seconds)`, and `WorkflowProgressResult(outcome, reset)` with only valid outcome/field combinations.

Enum member names are uppercase versions of these values. Policy reasons must include the fixed Jobs-owned `no_progress_exhausted` code; callers cannot replace that infrastructure outcome with arbitrary text.

Scope components/payload field/reasons use existing fixed-code syntax (1-64 ASCII characters); policy reason membership is required. Reject reserved Admin Webhook scope. Keys/fingerprints are 64 lowercase hex characters; idempotency key is nonblank, at most 200 UTF-8 bytes. Reject booleans for integer controls; max retries is a finite nonnegative integer, quarantine threshold is null or positive, expiry is exactly `REQUEUE_NO_ATTEMPT`, and no-progress limit is 1-5. Aware datetimes normalize to UTC. Continue/retry require a schedule; retry/fail require allowlisted reasons; complete/cancel reject irrelevant retry facts. Fingerprints exclude numeric hints, worker/lease, applied time, and delivery claims, but include owner, root UUID, scope/payload/key, controls, action, guard, original schedule, and reason. Test byte limits after canonical encoding, not character counts.

Capability migration/store entrypoints:

```text
install_workflow_support(manager: JobManager) -> None  # workflow_capability.py
inspect_workflow_support(manager: JobManager) -> WorkflowOutcome
PreparedWorkflowCapability(manager: JobManager, policy: WorkflowPolicy)
install_workflow_support_in_transaction(connection: Any, *, backend: str) -> None
inspect_workflow_support_in_transaction(connection: Any, *, backend: str) -> WorkflowOutcome
```

Migration functions are in `jobs_workflow_migrations.py`; they do not commit or open another connection. `install_workflow_support` explicitly owns the transaction. Backend implementations remain SQL-only leaves. No default schema installation call is added to `JobManager`.

- [x] **Step 1: Write contract and migration tests.** Test support produces `WorkflowCase(manager, capability, command)` with `from_manager(manager)`, `install()`, `table_names()`, and `support_version()` methods; backend fixtures use temporary SQLite or `jobs_pg_dsn`, never a bespoke Postgres cluster. `uninstalled_workflow_case` creates only the ordinary Jobs schema; `workflow_case` explicitly installs support before behavior tests. Both are parameterized by backend, marking PostgreSQL cases `pg_jobs`; `workflow_pg_case` requests the installed PostgreSQL variant. Its installer uses the privileged fixture connection, while its capability manager uses a non-superuser/non-BYPASSRLS role provisioned in that same per-test database with the existing Jobs RLS test pattern. Test-only configuration permits queue `workflow` through `JOBS_ALLOWED_QUEUES_EMBEDDINGS`, and restores it via monkeypatch. The 600-second rollback test sets the trusted lease cap to at least 600; the insufficient-cap test sets it below 300. Use deterministic UUIDv4 literals and the exact tests:

```python
def test_controls_reject_retry_consuming_expiry():
    with pytest.raises(ValueError):
        WorkflowControls(3, None, ExpiredLeasePolicy.CONSUME_RETRY, 5)

@pytest.mark.parametrize("limit", [0, 6, True])
def test_controls_bound_no_progress(limit):
    with pytest.raises(ValueError):
        WorkflowControls(3, None, ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT, limit)

def test_owner_limit_is_utf8_bytes():
    controls = WorkflowControls(3, None, ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT)
    WorkflowAdmissionCommand("\u00e9" * 256, ROOT_REFERENCE, "key", controls)
    with pytest.raises(ValueError):
        WorkflowAdmissionCommand("\u00e9" * 257, ROOT_REFERENCE, "key", controls)

def test_support_requires_explicit_install(uninstalled_workflow_case):
    case = uninstalled_workflow_case
    assert inspect_workflow_support(case.manager) is WorkflowOutcome.UNAVAILABLE
    assert "job_workflow_roots" not in case.table_names()
    case.install()
    case.install()
    assert case.support_version() == 1
```

Define `ROOT_REFERENCE = "00000000-0000-4000-8000-000000000001"` in test support. Add `test_effect_fingerprint_ignores_delivery_lease_and_hint`, `test_effect_facts_accept_4096_reject_4097_bytes`, `test_dto_copy_cannot_mutate_identity`, `test_migration_rolls_back_partial_install`, and `test_missing_constraint_is_unavailable_not_repaired`, with assertions of unchanged digest, exact size boundary, defensive immutability, zero partial tables/version, and unchanged malformed schema respectively.

`bounded_workflow_json` checks raw byte size before UTF-8/JSON decoding and returns parsed data for subsequent closed-schema validation, never execution authority. Its byte-boundary test uses `b"{}" + b" " * 4094` versus `b"{}" + b" " * 4095`; the first parses as an empty object and the second raises `ValueError`. Malformed/nested/unknown receipt facts must still be rejected by the typed receipt decoder, including bounded parse errors without logging source data.

- [x] **Step 2: Run red tests.**

Run: `source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_workflow_contracts.py tldw_Server_API/tests/Jobs/test_workflow_schema.py -m 'not pg_jobs'`

Expected: imports/new behavior fail, not an unrelated dependency failure. Record the red evidence.

- [x] **Step 3: Implement the stated contracts and version-1 migration.** Add three Jobs-owned support tables: `job_workflow_schema` (singleton version), `job_workflow_roots` (immutable owner/scope/key/payload/control snapshot, root UUID/hint, current acquisition epoch/lease digest/progress flag, last closed epoch, no-progress count), and `job_workflow_effect_receipts` (owner/root/effect key, action or progress-proof purpose, fingerprint, bounded facts, result, release marker). Unique admission identity is owner/scope/idempotency key; root UUID is unique; receipt uniqueness is owner/root/key with composite owner/root FK to the support root, never to active `jobs`. Progress and action keys share a namespace: conflicting purpose/facts is conflict. Add an explicit-install-only nullable `workflow_capability_version` column to active `jobs`; only capability admissions write `1`. Legacy rows stay null.

Enable foreign keys for SQLite. PostgreSQL forces owner RLS on roots and receipts with no `app.is_admin` bypass; schema version is readable readiness metadata only. Persist encoded bytes as bounded canonical text; check nonnegative counters, 1-5 no-progress limit, purpose/action vocabularies, UUID/digest shape, and owner/field byte bounds. Bound each root to 32,768 action receipts plus 32,768 progress receipts; action releases retain their slot/fingerprint, not another receipt row. This Jobs support bound is separate from the spec's six domain-table row limits. Admission must reject unsupported shape; runtime never adds columns or repairs constraints.

- [x] **Step 4: Run green contract/schema tests** with the Step 2 command, then with `-m pg_jobs` for backend migration/security parity. Only the existing fixture may skip PostgreSQL; record incomplete certification if it does.
- [x] **Step 5: Commit** contracts/schema/fixtures/tests only: `git commit -m "feat(jobs): define inert workflow capability schema"`. Stage the exact Task 1 file list, not the whole checkout.

## Stage 2: Exact Admission And Enlisted Live Fences

**Goal:** Create one exact root/control authority and validate its live lease inside a caller-enlisted transaction.
**Success Criteria:** Concurrent creates converge; conflicting or missing authority never recreates work; domain rollback rolls back capability mutations.
**Tests:** Replay, controls, queue-policy changes, identity reuse, cancellation and expiry races.
**Status:** Complete

### Task 2: Admission, Lookup, And Transaction Lifetime

**Files:** Modify capability/store/backend modules and `Jobs/manager.py`; create `test_workflow_admission_fencing.py` and extend test support.

**Interfaces:** `JobManager.workflow_capability(policy: WorkflowPolicy) -> PreparedWorkflowCapability` only constructs a facade. `JobManager._prepare_workflow_admission(command, policy) -> WorkflowAdmissionInputs` reuses current queue/fair-share/secret/size/encryption policy and quota inputs without admitting work. Define frozen `WorkflowAdmissionInputs(job: CreateJobCommand, max_queued_quota: int, submits_per_minute_quota: int, counters_enabled: bool, outbox_enabled: bool)` in contracts. Validate exact reference-only payload before and after policy transformations; reject truncation or a changed logical payload. Replay precedes mutable policy checks but not immutable identity/control checks.

```text
PreparedWorkflowCapability.admit(command: WorkflowAdmissionCommand) -> WorkflowAdmissionResult
PreparedWorkflowCapability.lookup(root: WorkflowRoot) -> WorkflowLookupResult
PreparedWorkflowCapability.transaction(owner_user_id: str) -> ContextManager[WorkflowTransaction]
WorkflowTransaction.lock_live_root(root: WorkflowRoot, lease: WorkflowLease) -> WorkflowFence
WorkflowTransaction.ensure_horizon(fence: WorkflowFence, minimum_seconds: int) -> WorkflowHorizonResult
```

`WorkflowTransaction` protocol lives in `jobs_workflow_store.py`. It exposes the same native `connection` and `backend` to a DB_Management repository, not a second manager read. `WorkflowFence` is a transaction-bound opaque token minted by `lock_live_root`; define it in store, reject manually constructed, cross-transaction, expired, or post-commit tokens. Invalid fences raise `WorkflowFenceError(outcome: WorkflowOutcome)` from contracts. Lease checks compare the exact root UUID/owner/scope/payload/key/controls, processing status, worker, lease ID, and database expiry. Revalidate at every mutating use, not only token construction; use PostgreSQL `clock_timestamp()` rather than transaction-frozen `NOW()` and fractional SQLite database time. PostgreSQL locks the Jobs row first, then support row; SQLite starts `BEGIN IMMEDIATE`. Horizon exceeding the configured cap returns insufficient, not a silently smaller guarantee.

- [x] **Step 1: Write failing behavior tests.** Test support adds `admit(**control_overrides)`, `admit_new_reference()`, `acquire() -> WorkflowLease`, `root -> WorkflowRoot`, `counters() -> tuple[int, int]`, `job_count()`, `set_queue_allowed(bool)`, `replace_numeric_id_with_other_uuid()`, `write_domain(tx, value)`, `read_domain()`, and `lease_deadline() -> datetime`. Acquire a 120-second lease by default.

```python
def test_duplicate_admission_reuses_exact_root(workflow_case):
    first = workflow_case.admit()
    second = workflow_case.admit()
    assert first.root.job_uuid == second.root.job_uuid
    assert second.outcome is WorkflowOutcome.ALREADY_APPLIED
    assert workflow_case.job_count() == 1

def test_replay_rejects_changed_controls(workflow_case):
    workflow_case.admit()
    assert workflow_case.admit(no_progress_limit=4).outcome is WorkflowOutcome.CONFLICT
    assert workflow_case.job_count() == 1

def test_exact_replay_precedes_new_queue_rejection(workflow_case):
    root = workflow_case.admit().root
    workflow_case.set_queue_allowed(False)
    assert workflow_case.admit().root == root
    assert workflow_case.admit_new_reference().outcome is WorkflowOutcome.REJECTED

def test_enlisted_domain_error_rolls_back_lease_change(workflow_case):
    workflow_case.admit()
    lease = workflow_case.acquire()
    before = workflow_case.lease_deadline()
    with pytest.raises(RuntimeError):
        with workflow_case.capability.transaction("owner-a") as tx:
            fence = tx.lock_live_root(workflow_case.root, lease)
            workflow_case.write_domain(tx, "not-committed")
            assert tx.ensure_horizon(fence, 600).outcome is WorkflowOutcome.APPLIED
            raise RuntimeError("controlled test rollback")
    assert workflow_case.read_domain() is None
    assert workflow_case.lease_deadline() == before
```

Add parameterized `test_replay_rejects_owner_payload_key_scope_or_retry_control_conflict`, `test_concurrent_creates_publish_one_root_and_one_created_event`, `test_support_insert_failure_leaves_no_root`, `test_numeric_hint_reuse_cannot_select_replacement`, `test_cancel_or_expire_between_read_and_write_rejects_checkpoint`, `test_cap_below_300_seconds_returns_insufficient`, and `test_legacy_enforcement_disable_does_not_weaken_fence`. Assert one UUID/event, zero rows after rollback, untouched replacement/cursor, exact insufficient result, and strict stale lease respectively. Use transaction barriers and controlled DB timestamps, not sleeps.

`test_fence_rechecks_expiry_inside_transaction` mints a fence, uses the controlled test helper to set the locked row's deadline to current fractional database time, then attempts a horizon/domain mutation. Assert `WorkflowFenceError(STALE_LEASE)` and unchanged synthetic domain data; a transaction-start timestamp must not keep that lease valid.

- [x] **Step 2: Run red tests:** `source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_workflow_admission_fencing.py -m 'not pg_jobs'`.
- [x] **Step 3: Implement the interfaces on both backends.** Admission serializes owner/scope/key correlation (SQLite write transaction; PostgreSQL stable transaction advisory lock), reads retained support authority and both active/archive UUID matches, validates all controls, then creates root/support/event/counter atomically. Reuse transaction-neutral admission leaf helpers such as `_quota_rejection`, `_insert_job`, `_bump_counters`, and `_insert_created_event`; do not call the committing `create_job`/`admit_job` and append support afterward. Retained support pointing to a missing root returns missing/unavailable, never new admission. Conflicting legacy correlation is not adopted. Bounded archive normalization/decryption must preserve exact payload and detect active/archive duplicates even after integer reuse.

The facade opens one connection, begins once, establishes exact transaction-local owner context, verifies the effective PostgreSQL role, commits once, and closes/rolls back on all exits. Do not call the best-effort `_pg_cursor` RLS setup as the security gate: a failed setup must raise, never rollback and continue under old/admin scope. Restore caller ContextVars on exit. Root locks remain held while the synthetic/domain repository checks its version/generation/claim/cursor by CAS; the capability cannot verify domain facts on another connection.

- [x] **Step 4: Run green tests** with the Step 2 command and `-m pg_jobs`; also run existing admission/idempotency tests for each backend.
- [x] **Step 5: Commit:** `git commit -m "feat(jobs): admit and fence exact owned workflow roots"` after staging only Task 2 files.

## Stage 3: Lease-Independent Effects And Retained Proofs

**Goal:** Apply transitions once, retain proof after later transitions/archive/prune, and release proof recoverably.
**Success Criteria:** Old applied effects replay without mutation; unapplied effects need current authority plus domain revalidation; changed facts fail closed.
**Tests:** Lost acknowledgement, retry accounting, terminal races, archive compression, release/replay.
**Status:** Complete

### Task 3: Logical Effects, Proof Lookup, And Release

**Files:** Modify contracts/capability/store/backend modules and enrolled-root pruning integration in `Jobs/manager.py`; create `test_workflow_effect_receipts.py`.

**Interfaces:** Extend Task 2's transaction/facade:

```text
PreparedWorkflowCapability.lookup_effect(root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult
PreparedWorkflowCapability.release_effect(root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult
WorkflowTransaction.apply_effect(fence: WorkflowFence, effect: WorkflowEffect,
                                 verified_guard_fingerprint: str) -> WorkflowEffectResult
WorkflowTransaction.lookup_effect(root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult
WorkflowTransaction.release_effect(root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult
```

Domain revalidation occurs after the Jobs lock and before `apply_effect` on the same transaction. The verified guard must equal the immutable intent's guard; neither digest alone nor a stale token proves domain validity. The caller aborts if its locked domain CAS no longer supports the intent. Jobs does not query domain tables. No-lease reconciliation may only read/release existing exact receipts; it cannot apply an unapplied transition. The domain caller decides supersession of obsolete domain intents; Jobs returns terminal/stale/conflict/missing evidence, never applies a different action under the key.

- [x] **Step 1: Write failing receipt tests.** Support adds `apply(effect, lease, verified_guard=None)`, `effect(key, action, reason=None)`, `archive_root(compressed=False)`, `prune_root_history()`, `status()`, `receipt_count()`, `retry_count()`, `effect_receipt(root, key)`, and explicit synthetic guard CAS.

```python
def test_earlier_effect_survives_later_transition_and_archive(workflow_case):
    workflow_case.admit()
    first = workflow_case.effect("a" * 64, WorkflowAction.CONTINUE)
    workflow_case.apply(first, workflow_case.acquire())
    last = workflow_case.effect("b" * 64, WorkflowAction.COMPLETE)
    workflow_case.apply(last, workflow_case.acquire())
    workflow_case.archive_root(compressed=True)
    replay = workflow_case.capability.lookup_effect(workflow_case.root, first)
    assert replay.outcome is WorkflowOutcome.ALREADY_APPLIED
    assert replay.receipt.resulting_status == "queued"
    assert workflow_case.status() == "completed"

def test_retry_lost_ack_does_not_spend_twice(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("c" * 64, WorkflowAction.RETRY, "provider_transient")
    lease = workflow_case.acquire()
    workflow_case.apply(effect, lease)
    workflow_case.apply(effect, lease)
    assert workflow_case.retry_count() == 1
    assert workflow_case.receipt_count() == 1

def test_pruned_history_does_not_erase_unreleased_proof(workflow_case):
    workflow_case.admit()
    effect = workflow_case.effect("d" * 64, WorkflowAction.COMPLETE)
    workflow_case.apply(effect, workflow_case.acquire())
    workflow_case.prune_root_history()
    assert workflow_case.capability.lookup_effect(workflow_case.root, effect).outcome is WorkflowOutcome.ALREADY_APPLIED
    assert workflow_case.admit().outcome in {WorkflowOutcome.MISSING, WorkflowOutcome.UNAVAILABLE}
```

Add `test_effect_insert_failure_rolls_back_status_counter_and_event`, `test_new_lease_delivers_same_intent_after_locked_guard_revalidation`, `test_stale_domain_guard_aborts_without_receipt`, `test_same_key_changed_facts_conflict_even_after_release`, `test_release_lost_ack_replays_without_reapplying`, `test_terminal_or_reused_uuid_cannot_receive_unapplied_effect`, and `test_receipt_capacity_exhaustion_precedes_transition`. Assert complete rollback, unchanged key/fingerprint, zero receipt after failed CAS, fixed conflict, absorbing release marker, terminal/replacement untouched, and zero transition at one-over capacity.

- [x] **Step 2: Run red tests:** `source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_workflow_effect_receipts.py -m 'not pg_jobs'`.
- [x] **Step 3: Implement the interfaces.** Receipt lookup by owner/root/key and fingerprint precedes lease validation only for already-applied exact effects. Unapplied effects lock/revalidate the exact live root and fenced domain facts. Apply lifecycle mutation, durable counters/event, and receipt in one transaction. Continue clears the lease and queues at the original scheduled instant without retry/streak increments. Retry charges once and honors max retries/quarantine; complete/fail/cancel are terminal and clear authority. Use existing Jobs transition/counter/event semantics; do not call a committing legacy prepared operation or bypass a current UUID fence.

Release atomically marks the exact receipt released and returns its proof; repeat release is acknowledgement-neutral. Keep a compact immutable key/fingerprint/outcome tombstone for the support-root lifetime so delayed duplicate delivery cannot reapply after release. Unreleased proof and minimal root support survive ordinary Jobs active/archive pruning. A missing root is not missing receipt authority. No support-root deletion is exposed in this unit: eventual owner-scoped domain retention release belongs to Stage 3B/3D after every delivery horizon is proven closed. Capacity exhaustion leaves existing obligations inspectable and rejects new effects before any transition.

- [x] **Step 4: Run green tests** with the Step 2 command and `-m pg_jobs`; run existing prepared-disposition/terminal-result/prune tests. Verify no new observer replay: only the first applied effect emits post-commit best-effort metrics, using immutable UUID-bound facts, never a later `get_job(id)` result.
- [x] **Step 5: Commit:** `git commit -m "feat(jobs): retain workflow disposition receipts across recovery"` after staging exact Task 3 files.

## Stage 4: Bounded Retry-Neutral Acquisition Recovery

**Goal:** Bound no-progress cycles even before handler start while preserving provider retry accounting.
**Success Criteria:** Acquire/sweep close one episode once; fifth no-progress episode fails through Jobs; new semantic proof resets once.
**Tests:** All acquisition modes, fractional expiry, race/restart, progress replay, cancellation.
**Status:** Complete

### Task 4: Acquisition Episodes And Semantic Progress

**Files:** Modify capability/store/backend modules, both Jobs `operations/*/lifecycle.py` acquisition leaves, manager expired recovery dispatch; create `test_workflow_expiry_progress.py`.

**Interfaces:** Add `WorkflowTransaction.record_progress(fence: WorkflowFence, proof: WorkflowProgress) -> WorkflowProgressResult`. Backend leaves also produce:

```text
record_workflow_acquisition_in_transaction(connection: Any, *, backend: str,
                                            acquired: Mapping[str, object]) -> None
recover_workflow_expiry_in_transaction(connection: Any, *, backend: str,
                                       locked_root: Mapping[str, object]) -> WorkflowEffectResult
```

Put these SQL leaves in DB_Management backend modules, expose dispatch through `jobs_workflow_store.py`. They never begin, commit, or await; the existing Jobs acquire/recovery transaction owns the lifetime. A null/absent enrollment marker returns immediately without touching support tables. Marker `1` requires valid schema and exact support/control authority; missing/corrupt support must roll back acquisition, not fall through to legacy retry-consuming behavior. Current epoch increases atomically at acquisition, with a digest of the lease and a fresh progress flag; last closed epoch prevents duplicate episode charging.

- [x] **Step 1: Write failing expiry/progress tests.** Support adds `expire_current_lease(fractional=False)`, `recover(via)`, `no_progress_count()`, `failure_reason()`, `restart()`, `close_empty_episode()`, and `progress(proof_key, kind, facts_fingerprint)`; `close_empty_episode` acquires once and applies a continue intent. `recover` calls public acquisition or `integrity_sweep`, not a hand-built replacement reclaimer.

```python
@pytest.mark.parametrize("via", ["acquire", "sweep"])
def test_pre_handler_expiry_fails_at_fifth_episode_without_retry(via, workflow_case):
    workflow_case.admit()
    for episode in range(1, 6):
        workflow_case.acquire()
        workflow_case.expire_current_lease()
        workflow_case.recover(via)
        assert workflow_case.no_progress_count() == episode
        assert workflow_case.retry_count() == 0
    assert workflow_case.status() == "failed"
    assert workflow_case.failure_reason() == "no_progress_exhausted"

def test_deferral_ack_cannot_reset_no_progress(workflow_case):
    workflow_case.admit()
    for episode in range(1, 6):
        effect = workflow_case.effect(f"{episode:064x}", WorkflowAction.CONTINUE)
        workflow_case.apply(effect, workflow_case.acquire())
    assert workflow_case.status() == "failed"
    assert workflow_case.no_progress_count() == 5

def test_progress_proof_resets_only_once_across_restart(workflow_case):
    workflow_case.admit()
    workflow_case.close_empty_episode()
    workflow_case.acquire()
    workflow_case.progress("a" * 64, WorkflowProgressKind.OUTPUT_RECEIPT, "b" * 64)
    assert workflow_case.no_progress_count() == 0
    workflow_case.expire_current_lease()
    workflow_case.recover("sweep")
    workflow_case.restart()
    workflow_case.close_empty_episode()
    workflow_case.acquire()
    workflow_case.progress("a" * 64, WorkflowProgressKind.OUTPUT_RECEIPT, "b" * 64)
    assert workflow_case.no_progress_count() == 1
```

Define `recover("acquire")` to finish the expired episode through acquisition's recovery path without leaving a second untracked lease in this loop: return/store the lease acquired by that call and have the next `acquire()` consume it once. For sweep, `acquire()` gets a fresh lease normally. Add `test_acquire_and_sweep_race_charge_same_epoch_once`, `test_fractional_deadline_is_not_expired_early`, `test_same_progress_key_changed_facts_conflicts`, `test_late_proof_from_prior_epoch_cannot_reset_current_count`, `test_cancel_wins_over_expiry_terminal_disposition`, `test_schema_loss_blocks_enrolled_acquire`, and `test_legacy_and_admin_webhook_expiry_behavior_unchanged`. Assert once-only episode/event, unchanged before-expiry state, conflict/no reset, cancellation preserved, no lease publication, and existing retry budgets respectively. Parameterize both `single_update` modes on each backend. No provider mock should be invoked in any test.

- [x] **Step 2: Run red tests:** `source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_workflow_expiry_progress.py -m 'not pg_jobs'`.
- [x] **Step 3: Implement the episode algorithm.** On each enrolled acquisition, record epoch/lease digest in the same transaction that publishes processing state. On continue or locked expiry, close that epoch once: if new semantic progress was committed in it, retain/reset zero; otherwise increment the independent no-progress count. On the pinned limit, apply non-retryable Jobs failure with `no_progress_exhausted` and retained proof rather than queuing again. Provider retries/streaks remain untouched by expiry/no-progress accounting; existing provider-origin retry effects still use Jobs quarantine policy. Terminal/cancelled roots take precedence. Expiry maintenance authority is the locked exact old lease plus DB expiry, not a fabricated worker lease.

Use a separate deterministic failure key, SHA-256 of the versioned tuple `(jobs_workflow_no_progress_v1, root_uuid, acquisition_epoch)`, when exhaustion stops a continuation. Return `EXHAUSTED` without applying that continuation; never store a fail action under the continuation's existing key. Expiry-triggered failure uses the same exact episode identity, so competing reclaim paths converge on one failure receipt/event.

Only a new, transaction-fenced root-binding/output/cursor/accounting/sink-lifecycle proof can reset; insert/check its immutable receipt and reset in one transaction. Replay, lease renewal, new claim, deferral acknowledgement, proof release, or time cannot reset. A token from an old epoch cannot reset a later one; newly recovered evidence may be validated and recorded under current live authority once. Claim epoch/proof state survives process restart. Close progressed expired episodes without charging; recovery reads of existing effects never authorize paid work.

Dispatch from the existing `_recover_expired_processing_jobs` path used by acquisition and integrity sweep, not a new loop/service. Lock Jobs before support; set explicit per-root owner scope for enrolled recovery while preserving the surrounding caller context. Do not scan owner-hidden support under an admin bypass. Bound recovery pages with existing Jobs recovery batch controls and preserve domain/queue/type filters. A support error stops enrolled recovery with typed unavailable evidence; it never changes legacy root behavior.

- [x] **Step 4: Run green tests** with the Step 2 command and `-m pg_jobs`, plus both existing lease-reclaim-budget suites and WorkerSDK prepared tests.
- [x] **Step 5: Commit:** `git commit -m "feat(jobs): bound retry-neutral workflow recovery episodes"` after staging exact Task 4 files.

## Stage 5: Backend, Security, And Compatibility Certification

**Goal:** Certify this capability independently before any Embeddings repository/runner uses it.
**Success Criteria:** Both backend matrices pass; owner/role/transaction tests prove fail-closed isolation; legacy and startup behavior remain unchanged; no new Bandit finding.
**Tests:** Real transaction races, RLS roles, model sequences, privacy, dormant imports.
**Status:** In Progress

### Task 5: Certification And Internal Documentation

**Files:** Create security/compatibility and property tests plus internal contract doc; extend fixtures only as needed; modify CI Jobs shard membership only when required by its coverage check.

**Interfaces:** Consume Tasks 1-4 only. Extend Task 1's runtime-role fixture; add test-support `runtime_role_flags() -> tuple[bool, bool]`, `set_owner(str | None)`, `force_rls_setup_failure()`, `direct_support_crud(owner, verb)`, `serialized_support() -> str`, `complete_for_owner(owner)`, `visible_receipt_count()`, and `visible_root_count()`. Do not introduce database/container fixtures. Grant only required Jobs/support table/sequence privileges, not `BYPASSRLS`, blanket schema DDL, or domain-owner access. State-machine observation is a test-only `WorkflowObservation(root_uuid, retry_count, no_progress_count, effect_fingerprints, status)` snapshot from the store case, never a new production telemetry API.

- [x] **Step 1: Write security and dormancy tests.**

```python
def test_rls_setup_failure_never_continues_as_admin(workflow_pg_case):
    assert workflow_pg_case.runtime_role_flags() == (False, False)
    workflow_pg_case.force_rls_setup_failure()
    assert workflow_pg_case.admit().outcome is WorkflowOutcome.UNAVAILABLE
    assert workflow_pg_case.job_count() == 0

def test_connection_reuse_cannot_read_previous_owner_receipt(workflow_pg_case):
    workflow_pg_case.complete_for_owner("owner-a")
    workflow_pg_case.set_owner("owner-b")
    assert workflow_pg_case.visible_receipt_count() == 0
    workflow_pg_case.set_owner(None)
    assert workflow_pg_case.visible_root_count() == 0

def test_normal_startup_does_not_install_support(tmp_path):
    manager = JobManager(tmp_path / "dormant.db")
    case = WorkflowCase.from_manager(manager)
    assert "job_workflow_roots" not in case.table_names()
    assert case.job_count() == 0
```

Also assert denied cross-owner select/insert/update/delete/release, FK owner mismatch, FORCE RLS flags on both owned support tables, owner reset after rollback, unsupported version rejection, compressed archive bounds, and no secret sentinel in durable receipts/logs. Test facts over 4 KiB and malformed timestamp/digest/UUID before backend materialization. Confirm constructor/import never schedules a worker or starts a reconciler and legacy `find_job_by_identity`/prepared origins continue rejecting noncanonical Admin Webhook calls.

- [x] **Step 2: Add property-based real-store sequences.** `test_workflow_capability_properties.py` defines a Hypothesis `RuleBasedStateMachine` whose rules are admit/replay/acquire/renew/expire/continue/retry/cancel/complete/progress/release/archive/restart. The model tracks exact root, controls, current lease/epoch, progress-proof identities, no-progress count, retry count, terminal state, and receipts. After each rule assert:

```python
assert observed.root_uuid == model.root_uuid
assert observed.retry_count == model.retry_count
assert observed.no_progress_count == model.no_progress_count
assert observed.effect_fingerprints == model.effect_fingerprints
assert not model.terminal or observed.status == model.terminal_status
```

Use bounded examples and the existing CI Hypothesis profile. Include changed-fact and stale-lease rejection commands, not just valid paths. Match SQLite and PostgreSQL outcomes with the same traces; concurrent barrier tests additionally cover effects versus cancel/prune and acquire versus sweep.

- [x] **Step 3: Run the complete matrix.** Commands below are execution gates, not assertions that this planning turn implemented or passed the new tests.

```bash
source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_workflow_*.py tldw_Server_API/tests/Jobs/property/test_workflow_capability_properties.py -m 'not pg_jobs'
source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_workflow_*.py tldw_Server_API/tests/Jobs/property/test_workflow_capability_properties.py -m pg_jobs
source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_admission_operations_* tldw_Server_API/tests/Jobs/test_jobs_idempotency_receipts_* tldw_Server_API/tests/Jobs/test_jobs_prepared_disposition_operations_* tldw_Server_API/tests/Jobs/test_jobs_lease_reclaim_budget_* tldw_Server_API/tests/Jobs/test_jobs_prune_* tldw_Server_API/tests/Jobs/test_worker_sdk_prepared.py
source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_row_identity_atomicity.py tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py
source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Embeddings_isolated/test_workflow_runner.py tldw_Server_API/tests/Embeddings_isolated/test_workflow_types.py
```

Expected: all selected tests pass, PostgreSQL runtime roles are actually non-bypass, and no unexpected deselection/skip hides a capability case. Record fixture-declared PostgreSQL unavailability as incomplete certification; no merge/activation claim until CI supplies the missing evidence. `test_jobs_completion_row_identity_atomicity.py` comes from current `dev`, so an absent file means preflight was not completed. Run endpoint/legacy parity explicitly:

```bash
source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Embeddings/test_embeddings_orchestrator_endpoint_parity.py tldw_Server_API/tests/Embeddings/test_embeddings_jobs_adapter.py tldw_Server_API/tests/Embeddings/test_embeddings_redis_worker.py tldw_Server_API/tests/Embeddings/test_media_embeddings_storage_scope.py
```

- [x] **Step 4: Run changed-scope quality/security and CI membership checks.**

```bash
source .venv/bin/activate && python -m ruff check tldw_Server_API/app/core/Jobs/operations/workflow_contracts.py tldw_Server_API/app/core/Jobs/workflow_capability.py tldw_Server_API/app/core/DB_Management/jobs_workflow*.py tldw_Server_API/tests/Jobs/workflow_test_support.py tldw_Server_API/tests/Jobs/test_workflow_*.py tldw_Server_API/tests/Jobs/property/test_workflow_capability_properties.py
source .venv/bin/activate && python -m compileall -q tldw_Server_API/app/core/Jobs/operations/workflow_contracts.py tldw_Server_API/app/core/Jobs/workflow_capability.py tldw_Server_API/app/core/DB_Management/jobs_workflow_migrations.py tldw_Server_API/app/core/DB_Management/jobs_workflow_store.py tldw_Server_API/app/core/DB_Management/jobs_workflow_sqlite.py tldw_Server_API/app/core/DB_Management/jobs_workflow_postgres.py
source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Jobs/operations/workflow_contracts.py tldw_Server_API/app/core/Jobs/workflow_capability.py tldw_Server_API/app/core/Jobs/manager.py tldw_Server_API/app/core/Jobs/operations/sqlite/lifecycle.py tldw_Server_API/app/core/Jobs/operations/postgres/lifecycle.py tldw_Server_API/app/core/DB_Management/jobs_workflow_migrations.py tldw_Server_API/app/core/DB_Management/jobs_workflow_store.py tldw_Server_API/app/core/DB_Management/jobs_workflow_sqlite.py tldw_Server_API/app/core/DB_Management/jobs_workflow_postgres.py -f json -o /tmp/bandit_embeddings_stage3a.json
git diff --check
```

Include the modified manager/lifecycle/conftest files in Ruff and formatter verification, comparing pre-existing findings with the recorded baseline rather than ignoring changed lines. Do not rewrite whole legacy files for formatting. Run the existing required-workflow/Jobs-shard coverage guard, adding exact new test paths only if uncovered; do not relax that guard. Expected: no new lint/format/security findings, compile success, clean diff, complete shard membership. Capture warnings and genuine skips.

Run the existing guard with `source .venv/bin/activate && RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/CI/test_required_workflow_contracts.py`; all assertions must pass. Use `python -m ruff format --check` on the new modules/test files listed in the Ruff command, and add the modified legacy files to lint/compile verification while preserving their established formatting.

- [ ] **Step 5: Document, review, and commit.** Write the internal doc with actual API examples, explicit migration invocation, owner/lease/proof semantics, independent budgets, dormant rollback (stop callers; preserve proof/support obligations; no destructive down migration), and deferred support/domain cleanup. Update this plan's stage statuses and the implementation Backlog task incrementally. Request a whole-branch review against the approved spec and address validated findings before integration. Stage exact verified Task 5 files and `git commit -m "test(jobs): certify prepared workflow recovery contracts"`. PR creation/merge is a later explicit instruction; any materially AI-authored PR requires the requester's own Change summary.

## Coverage And Next Units

| Spec Requirement | Owner |
| --- | --- |
| Exact owner/root identity and immutable controls; no recreation | Tasks 1/2 |
| Explicit migrations, strict live transaction fencing and lock order | Tasks 1/2/5 |
| Logical intent independent of delivery lease; applied proof after later transitions/archive/prune | Task 3 |
| Retry-neutral continuation/expiry, once-only provider retry, terminal precedence | Tasks 3/4 |
| Bounded durable no-progress count and replay-safe semantic reset | Task 4 |
| Role/owner isolation, bounded facts, privacy, Admin Webhook/Jobs compatibility, dormancy | Tasks 1/5 |
| Six Embeddings tables, domain CAS, manifests/generations, aggregate retention/capacity | Stage 3B, separate plan after 3A interfaces are reviewed |
| Atomic global capacity primitive without runtime RLS bypass | Stage 3B must select/test a concrete primitive, not sum owner-visible rows |
| Resolver/provider/sink/accounting ports, 295-second normal plan, runner/reconciler | Stage 3C, separate plan; no real adapters or production billing |
| Private durable-phase/terminal mapping to unchanged trace literals | Stage 3C must enumerate/test the complete mapping |
| Domain crash/property/backend/endpoint parity and full-foundation certification | Stage 3D; Task 5 certifies only the Jobs substrate |

The follow-on repository receives the transaction/fence/receipt APIs, not access to mutable manager internals. The runner receives a certified repository and fake adapter contracts, not this Jobs store directly. Stage 3A can be reviewed and shipped inert without the other units, but it cannot claim completed durable Embeddings execution or production activation.

## Plan Review Gate

Requester reviewed the committed plan and selected subagent-driven execution (option 1) on 2026-10-20. Runtime implementation proceeds under TASK-13900 with task-scoped reviews and final verification; stage statuses above record actual progress.
