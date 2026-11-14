# Jobs Completion Row-Identity Atomicity Design

Date: 2026-09-07
Topic: Atomic row identity and bookkeeping for `JobManager.complete_job`
Status: Approved design
Tracking: TASK-13215
Base: `origin/dev` at `e3174f1ad9f6dd0b11e4ecb20d48c1c4090d3bfe`

ADR check (2026-10-02): required yes. Governing decision:
`Docs/ADR/058-jobs-completion-row-identity.md`. The optional acquired UUID
precondition is a durable completion API contract; the ADR records the already
approved decision.

## Objective

Fix the validated race in `JobManager.complete_job` before resuming the strict
completion-operation extraction. A completion must transition only the exact
durable job row whose facts were loaded for that operation, and its enabled
lifecycle counter and outbox bookkeeping must commit atomically with the
terminal state.

This is a focused defect remediation. `JobManager` remains the public facade
and continues to own completion in this change. Backend extraction resumes in
a separate reviewable change after this blocker merges.

## Validated Defect

The current method first reads a job by numeric `id`, then later updates by
that same `id`. When the first read returns no row, another connection can
insert a queued row with the reused numeric ID before the update runs. With
acknowledgement enforcement disabled and the row's domain allowlisted, the
later queued update can complete the new row.

The mutation succeeds without authoritative base facts. The method therefore
returns `True`, but skips the bookkeeping and post-commit behavior guarded by
the missing base row.

The interleaving was reproduced deterministically with real databases:

- SQLite completed the concurrently inserted queued row while retaining only
  `job.created` and leaving `ready_count` at one.
- PostgreSQL 18 under the configured `READ COMMITTED` behavior produced the
  same completed row, missing `job.completed` event, and stale ready counter.
- PostgreSQL with forced RLS and a visible `chatbooks`/`u1` row produced the
  same incorrect transition. RLS controls visibility but does not make the
  separate statements one row-identity operation.

The defect violates the transaction contract and is a blocker for moving the
method into backend operation modules.

## Goals

- Bind one invocation to the visible row incarnation loaded at its transaction
  boundary.
- Return `False` when the initial locked lookup sees no visible row.
- Prevent an insert or delete/reinsert after the authoritative locked lookup
  from redirecting that invocation to the same numeric ID's new row.
- Preserve normal processing completion, explicitly allowed queued completion,
  and exact completion-token replay behavior.
- Let UUID-aware callers reject a stale row before mutation while preserving
  compatibility for direct callers that do not yet supply a UUID.
- Support legacy rows whose stored UUID is null, empty, or noncanonical.
- Commit the terminal mutation, enabled lifecycle-counter update or repair,
  and enabled `job.completed` outbox event as one transaction.
- Preserve RLS visibility by continuing to use the established PostgreSQL
  cursor path.
- Keep externally visible error precedence and noncritical post-commit failure
  handling stable unless the atomicity fix requires otherwise.

## Non-Goals

- Do not extract completion into `operations/postgres` or `operations/sqlite`.
- Do not add a schema migration, advisory lock, isolation-level change, or
  feature flag.
- Do not require UUIDs for every historical row or direct completion caller.
- Do not backfill, canonicalize, or reject legacy UUID values.
- Do not migrate every direct `complete_job` caller to UUID-aware completion.
- Do not change the queued-completion domain allowlist or acknowledgement
  enforcement policy.
- Do not redesign SLA policy lookup or make SLA-breach persistence mandatory.
- Do not redesign result serialization. In particular, this change does not
  add tests that make the current nonserializable-result fallback a permanent
  supported contract.
- Do not fold unrelated Jobs defects or the larger completion refactor into
  this pull request.

## Approaches Considered

### Recommended: Lock, capture identity, guard every mutation

Load the visible row at the start of the transaction while taking the backend's
established write lock: `SELECT ... FOR UPDATE` on PostgreSQL and
`BEGIN IMMEDIATE` before lookup on SQLite. Capture the raw stored UUID and use
`id` plus null-safe UUID equality in every mutation and replay lookup.

Benefits:

- follows existing Jobs cancellation and lifecycle locking patterns
- keeps row facts, state classification, transition, counters, and outbox in
  one explicit transaction
- works with nullable legacy UUIDs
- fixes both initial-miss and row-replacement races without schema work
- remains small enough to review before extraction

Tradeoff: SQLite takes its database write reservation before reading, so even a
missing-row completion briefly blocks competing writers. Completion already
performs a write and the bounded lock is the pragmatic correctness boundary.

### Alternative: UUID-guarded update with `RETURNING`

An update could return the transitioned row and drive bookkeeping from those
facts. This reduces the need for a preliminary read in some states, but still
needs separate handling for terminal replay, queued policy, legacy null UUIDs,
and backend SQL differences. It also expands the change while completion is
still embedded in `JobManager`.

### Alternative: Shared advisory locks

A lock keyed by job ID could serialize absent-row operations on PostgreSQL.
That would introduce a new cross-process locking protocol, require every
inserter and terminalizer to participate, and increase deadlock and rollout
risk. PostgreSQL does not need an absent-row lock when the method returns
immediately after the missing lookup.

### Alternative: Route ordinary success through the existing terminal-result operation

The extracted terminal-result path has stronger correlation semantics, but its
current command and resolver are intentionally specialized for standalone
slides generation and terminal error outcomes. It does not persist the normal
completion result payload or preserve the general queued-completion contract.
Expanding it here would combine a defect fix with a cross-domain API redesign.

## Public Contract

The method adds one optional keyword-only argument:

```python
expected_uuid: str | None = None
```

Existing direct callers remain source-compatible. `WorkerSDK` supplies the UUID
from the acquired job for ordinary successful completion.

`WorkerSDK` must call the expanded contract directly. It must not catch a
`TypeError` and retry without `expected_uuid`, because such a compatibility
fallback would silently remove the intended stale-worker precondition. Strict
test doubles used with `WorkerSDK` are updated to accept and assert the new
keyword.

When `expected_uuid` is not `None`, the locked row's external UUID
representation must match before state classification or mutation. A mismatch
returns `False`. For legacy rows, a null stored UUID has the same external
representation as the empty UUID already exposed by worker acquisition. The
raw stored value is still retained for the SQL identity predicate.

This creates two intentionally different guarantees:

- UUID-aware callers receive a pre-mutation stale-incarnation rejection.
- All callers receive in-operation replacement protection from the transaction
  lock and stored-identity predicates.

Callers that omit `expected_uuid` do not gain protection against choosing an
already-replaced row before their call begins. A legacy null or empty UUID also
cannot distinguish two pre-call row incarnations that both lack usable UUIDs;
its guarantee is limited to the in-operation lock boundary. Migrating callers
and legacy data policy are separate follow-ups.

The temporal boundary is explicit: the locked lookup, not receipt of a numeric
ID by some earlier caller, starts this operation's row-incarnation guarantee.
An insert after a locked miss cannot be mutated because the method returns
immediately. A replacement before the locked lookup is distinguishable only
when the caller supplies a usable expected UUID. This task does not claim a
stronger guarantee for numeric-ID-only or null-UUID callers.

## Identity Boundary

The initial locked, visible row is authoritative for the complete operation.
The locked projection must include at least:

- `id`, `uuid`, and `status`
- `completion_token`, `worker_id`, and `lease_id`
- `domain`, `queue`, `job_type`, and `available_at`
- `started_at`, `acquired_at`, `trace_id`, and `request_id`
- `owner_user_id`

The raw UUID may be null or noncanonical. It is not validated or rewritten.
Every completion update and same-transaction replay read uses both the numeric
ID and null-safe equality against that captured raw UUID:

```text
PostgreSQL: uuid IS NOT DISTINCT FROM %s
SQLite:     uuid IS ?
```

The UUID predicate is defense in depth around the row lock. PostgreSQL prevents
deletion or modification of a selected row until the transaction exits;
SQLite's immediate transaction prevents a competing writer from deleting or
replacing it. If a guarded update still affects zero rows, completion may only
return `True` for an exact completed-token replay found with the same `id` and
captured UUID.

## Transaction Flow

Result preparation that does not require row facts remains before connection
acquisition and locking. This preserves the current ordering of:

1. required completion-token validation
2. default enforcement resolution
3. result-size configuration parsing
4. initial JSON serialization and size/truncation handling
5. connection acquisition

The current handling of a nonserializable result is left untouched by this
focused fix but is not elevated into a documented long-term API guarantee.

Inside the transaction:

1. PostgreSQL opens the established RLS-aware cursor and selects the row with
   `FOR UPDATE`. SQLite issues `BEGIN IMMEDIATE`, then selects the row.
2. If no visible row exists, return `False` without issuing a completion update
   or replay lookup.
3. If `expected_uuid` is supplied and does not match, return `False`.
4. Capture all row facts and apply optional result encryption using the locked
   row's domain. Preserve the existing noncritical encryption fallback.
5. Classify the transition directly from the locked status and policy; do not
   attempt one update, query the domain again, and then attempt another update.
6. Execute one identity-guarded completion update for the selected branch.
7. If the update unexpectedly affects zero rows, permit only an exact
   completed-token replay scoped to the same captured row identity.
8. For an applied transition, perform enabled lifecycle-counter bookkeeping,
   stage savepoint-scoped SLA handling, persist the enabled `job.completed`
   outbox event, and queue post-commit callbacks.
9. Commit the transaction.
10. Run metrics, tracing, observers, non-outbox event emission, and audit work
    only after a successful commit, retaining existing nonfatal handling.

`True` means either the exact row transition committed with mandatory enabled
bookkeeping or the same completion token is an exact replay of that row's
already-completed state. Every other expected no-transition returns `False`.

## State Decisions

| Locked state | Conditions | Result |
| --- | --- | --- |
| Missing or RLS-hidden | No visible row at initial lookup | `False`; no mutation |
| `completed` | Nonempty supplied token equals stored token | `True` replay; no duplicate bookkeeping |
| `completed` | Token absent or different | `False` |
| `failed`, `cancelled`, `quarantined` | Any token | `False` |
| `processing` | Enforcement enabled and worker, lease, and token guards match | Apply completion |
| `processing` | Enforcement enabled and an ownership guard fails | `False` |
| `processing` | Enforcement disabled and token guard permits | Apply completion |
| `queued` | Enforcement disabled, locked domain is allowlisted, and token guard permits | Apply completion |
| `queued` | Enforcement enabled, domain disallowed, or token guard rejects | `False` |
| Any other state | Any inputs | `False` |

The allowlist is computed from the existing environment setting and fallback.
Its decision uses the already locked `domain`; there is no secondary lookup
that could observe another row incarnation.

## Atomic Bookkeeping

When an applied completion changes `processing` to `completed`, the matching
processing counter is decremented. When explicitly allowed queued completion
changes `queued` to `completed`, the ready or scheduled counter is decremented
according to the locked `available_at` value. If the scoped counter row is
missing, the existing reconciliation helper repairs it in the same
transaction.

When `JOBS_EVENTS_OUTBOX` is enabled, insertion of `job.completed` is mandatory
and occurs in the same transaction. A counter, reconciliation, outbox, or
commit failure rolls back the job transition and propagates the original
failure. No queued post-commit side effect runs after a failed commit.

SLA-breach attachment and event writes retain the current savepoint-backed,
best-effort behavior after the savepoint has been established. Their handled
statement failures roll back to that savepoint and do not invalidate an
otherwise sound completion. Failure to create, roll back, or release the
savepoint itself is a transaction-control failure and must propagate rather
than risk committing an unknown transaction state.

## Backend Details

### PostgreSQL

- Use the established `_pg_cursor` path so RLS context remains authoritative.
- Lock the visible row with `SELECT ... FOR UPDATE` under the configured
  `READ COMMITTED` transaction.
- An absent row has no row lock. A concurrent insert may therefore succeed, but
  the completion invocation has already classified the target as missing and
  must return `False` without another ID-only query or update.
- Use `IS NOT DISTINCT FROM` for nullable stored UUID guards.
- Preserve the established explicit commit error mapping used by completion.

### SQLite

- Enter `BEGIN IMMEDIATE` before the initial lookup, matching existing Jobs
  lifecycle operations.
- The immediate transaction reserves the writer boundary even when the row is
  absent, so a competing insert/delete cannot enter during classification and
  mutation.
- Use `IS ?` for nullable stored UUID guards.
- Keep all state, counters, SLA savepoint work, and outbox persistence inside
  the one transaction context.
- The stronger boundary means missing-row and terminal-replay calls also
  request the SQLite writer reservation. They use the configured connection
  timeout and may surface the existing SQLite busy/locked database exception
  under contention. This task adds no hidden retry loop; contention remains
  visible to callers and the lock is released on every return path.

## Lock Ordering

The operation locks the job row or SQLite writer boundary before touching
`job_counters` and `job_events`. That order matches current acquisition,
release, prepared-disposition, cancellation, and pruning paths reviewed on the
target base. No new table-first-then-job lock order or nested write connection
is introduced.

The existing SLA policy read may use a separate connection, but it is
read-only. Redesigning that lookup is outside this defect fix.

## Deterministic Test Strategy

### SQLite regression harness

Instrument the connection/cursor boundary to pause immediately after the
target initial `fetchone()` rather than relying on sleeps. Coordination uses
bounded events/barriers, and teardown releases every paused worker in a
`finally` block so a failed assertion cannot hang the suite.

For an initial miss:

1. Start completion and pause after the missing lookup.
2. Attempt the competing insert through a second connection with `timeout=0`.
3. Demonstrate that the pre-fix path permits the insert and can transition it.
4. On the fixed path, assert the insert receives a SQLite busy/locked error
   code while completion returns `False`; do not bind the test to one localized
   exception message.
5. Retry the insert after completion exits and verify the new job remains
   queued with correct creation bookkeeping.

For an existing row, pause after the locked fetch and attempt delete/reinsert
from a zero-timeout second connection. The fixed path must block the delete;
the completed row and bookkeeping must belong to the originally locked
incarnation.

### PostgreSQL regression harness

Use isolated roles/databases and the real production cursor behavior.

- Initial absent row: pause after the locked lookup; allow a concurrent insert
  to succeed; verify completion returns `False` and the inserted row remains
  queued.
- Existing visible row: prove the completion transaction owns the row lock by
  asserting a competing `SELECT ... FOR UPDATE NOWAIT` raises psycopg's
  lock-not-available error immediately.
- Repeat the relevant cases under normal `READ COMMITTED` and forced RLS with a
  visible scoped row.
- Verify an RLS-hidden row remains unchanged and completion returns `False`.
- Always clear `JobManager` RLS context during test cleanup.

### Contract and atomicity coverage

Add or strengthen tests for:

- expected UUID match and mismatch
- legacy null and noncanonical stored UUIDs
- enforced processing completion with matching and stale worker/lease identity
- enforcement-disabled processing completion
- allowlisted and disallowed queued completion
- exact completion-token replay and mismatched-token rejection
- concurrent same-token finalizers producing one mutation, one counter delta,
  and one durable completion event
- enabled counter and outbox success in the same commit
- counter, reconciliation, outbox, and commit fault rollback
- no post-commit callback after rollback
- existing best-effort SLA failure behavior
- `WorkerSDK` forwarding the acquired job UUID
- strict `WorkerSDK` doubles accepting the new keyword without a retry that
  drops it
- required-token, result-size, and malformed size-configuration precedence on
  missing and stale-UUID rows

Do not add a nonserializable-result characterization to this PR.

## Verification And Delivery

Before implementation, rerun the focused completion/finalization/lifecycle
baseline and both deterministic database reproductions on this exact `dev`
base. Implement test-first, proving the regression against the unchanged
method before applying the fix.

Local verification includes the focused SQLite Jobs matrix with `RUN_JOBS=1`,
the required real-PostgreSQL matrix with
`TLDW_TEST_POSTGRES_REQUIRED=1`, broader relevant Jobs regressions, project
formatting/lint checks, and Bandit over `manager.py` and `worker_sdk.py`.
Broader or stress validation belongs in the dedicated mandatory Jobs CI path.

No migration, rollout flag, or data rewrite is required for prevention. The
change is revertible as one facade-level defect fix.

## Historical Data Boundary

This preventive fix does not prove whether the race occurred in an existing
deployment and does not rewrite historical rows. A completed job without a
`job.completed` event is not sufficient evidence because the outbox may have
been disabled at transition time. Synthesizing events would therefore create
false history. Counter values can be rebuilt from durable current state, but a
global repair has different locking and operator-impact concerns from the
runtime fix.

TASK-13217 owns the bounded assessment, idempotent counter reconciliation, and
operator guidance for possible pre-fix drift. This PR records the residual
risk without performing an unsafe automatic repair.

## Follow-Up Work

After this remediation merges:

1. Resume the approved strict extraction of completion into backend-specific
   operation modules without changing behavior.
2. TASK-13216 inventories direct `complete_job` callers and migrates appropriate
   acquired-job paths to pass `expected_uuid` in a compatibility-focused PR.
3. TASK-13217 assesses possible historical counter and event drift without
   inventing unverifiable outbox history.
4. Unrelated serialization or SLA-connection concerns require independent
   characterization and a dedicated Backlog task before implementation.
