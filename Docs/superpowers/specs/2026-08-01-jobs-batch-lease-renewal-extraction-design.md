# Jobs Batch Lease Renewal Extraction Design

Date: 2026-08-01
Topic: Atomic Jobs batch lease renewal extraction
Status: Approved design
Tracking: TASK-12989

## Objective

Extract `JobManager.batch_renew_leases` into dedicated SQLite and PostgreSQL
lifecycle operations without changing its public signature or integer return
contract. Make the existing whole-batch transaction boundary explicit and
testable while preserving input-order processing, backend-specific clock
timing, duration clamping, lease-identity enforcement, and row-count behavior.

This is the first deferred lifecycle slice after single-job acquisition,
renewal, and release extraction. It is intentionally limited to batch lease
renewal. Batch completion/failure and terminal transition families remain in
`JobManager`.

## Goals

- Keep `JobManager.batch_renew_leases(items, *, enforce=None) -> int` stable.
- Preserve one atomic transaction for the complete batch.
- Treat missing, non-processing, and stale-lease rows as non-fatal no-ops.
- Roll back every earlier renewal when a later clock or database operation
  fails unexpectedly.
- Preserve SQLite and PostgreSQL timing and SQL behavior where they currently
  differ.
- Introduce immutable typed batch commands and results.
- Keep backend SQL and transaction details out of `JobManager`.
- Add durable SQLite and required real-PostgreSQL evidence before routing the
  facade through the new operations.

## Non-Goals

- Do not change the public method signature, accepted item keys, or return
  shape.
- Do not extract `batch_complete_jobs`, `batch_fail_jobs`, terminal
  transitions, retry, quarantine, pruning, or admin-owned SQL.
- Do not introduce a batch-size limit, sorting, deduplication, or new lock
  ordering.
- Do not add schema or migration changes.
- Do not integrate the existing `JobsSettings` scaffold.
- Do not add events, counters, metrics, or other renewal side effects.
- Do not replace the ordered loop with backend-specific set-based bulk SQL.
- Do not unify SQLite and PostgreSQL clock sampling in this extraction.

## Existing Behavior

The current facade opens one connection and one backend transaction, then
updates each item in input order. It sums statement row counts and returns the
sum as an integer.

Expected no-transition cases do not raise. An update contributes zero when the
job is missing, not processing, or has a stale worker/lease identity while
enforcement is enabled. Other valid updates in the same batch still commit.

Unexpected exceptions leave the transaction context and roll back all earlier
updates. Duplicate job identifiers are separate ordered attempts and may each
contribute to the returned count. Renewal uses `MAX`/`GREATEST`, so a shorter
requested extension does not shorten an existing longer lease.

PostgreSQL samples `self._clock.now_utc()` once per batch. SQLite samples it
once per item. `JOBS_LEASE_MAX_SECONDS` is currently read while processing each
item. Empty batches still acquire a connection and enter the backend operation
path but do not read the lease maximum.

## Approaches Considered

### Recommended: Dedicated backend batch operations

Add one typed batch operation to each lifecycle backend. Each operation owns
the complete atomic scope and iterates its immutable command items in order.
The normal `JobManager` path supplies a fresh connection, so that scope is a
native transaction. A direct SQLite call inside a caller-owned transaction
uses a savepoint and leaves the outer transaction open.
Single and batch renewal may share only transaction-neutral SQL construction
through a pure statement-and-parameter builder. Execution and result handling
remain separate.

Benefits:

- transaction or savepoint ownership is obvious and directly testable
- no caller-transaction commit or per-item commit ambiguity
- backend differences remain explicit
- the public facade becomes smaller without changing callers
- the change remains independently reviewable and revertible

### Alternative: Loop over the single-job operation

This would reduce visible SQL but requires making the single-job operation
transaction-aware. Its current transaction wrappers could commit per item or
create backend-specific nested-context behavior. The abstraction would hide
rather than clarify the batch boundary.

### Alternative: Set-based bulk SQL

A set-based statement could reduce round trips, but per-item durations,
optional lease enforcement, duplicate attempts, ordered clock sampling, and
exact row-count compatibility make the SQL substantially more complex. That
optimization is not justified without measured batch-size or contention data.

## Architecture

`JobManager` remains the public facade and owns:

- default `enforce` resolution
- item field extraction and existing `int(...)` conversions
- per-item duration clamping using `JOBS_LEASE_MAX_SECONDS`
- immutable command construction
- backend selection and connection setup
- mapping `BatchRenewLeasesResult.applied_count` to the public integer
- non-fatal connection cleanup

Backend lifecycle operations own:

- backend-specific fixed SQL variants and bound parameters
- backend-specific clock sampling
- one atomic scope around the complete ordered item tuple
- a native transaction on fresh connections and a savepoint for direct SQLite
  calls made inside a caller-owned transaction
- update execution and exact applied-attempt counting
- rollback through the native transaction or savepoint on unexpected errors
- result construction before the atomic context exits

Backend operation modules must not import or reference `JobManager`.

## Typed Contracts

Add immutable contracts in `operations/contracts.py`:

```python
@dataclass(frozen=True)
class BatchRenewLeaseItem:
    job_id: int
    seconds: int
    worker_id: str | None = None
    lease_id: str | None = None


@dataclass(frozen=True)
class BatchRenewLeasesCommand:
    items: tuple[BatchRenewLeaseItem, ...]
    enforce: bool


@dataclass(frozen=True)
class BatchRenewLeasesResult:
    requested_count: int
    applied_count: int
```

`BatchRenewLeasesCommand.__post_init__` snapshots any supplied sequence as an
immutable tuple, and each item rejects non-positive normalized durations.
`BatchRenewLeaseItem.seconds` is a facade-normalized value that is already
clamped to the applicable per-item maximum. Backend operations neither read
`JOBS_LEASE_MAX_SECONDS` nor clamp item durations again. The result enforces:

```text
0 <= applied_count <= requested_count
```

`applied_count` means successful update attempts, not distinct job rows.
Duplicate job IDs therefore retain current counting semantics.

## Data Flow

1. A caller invokes `JobManager.batch_renew_leases` with the existing item
   dictionaries and optional `enforce` override.
2. The facade resolves default enforcement exactly once.
3. The facade opens the existing backend connection before consuming items,
   preserving current connection-failure precedence, and supplies the
   established PostgreSQL cursor factory where applicable.
4. Before dispatching any backend mutation, the facade consumes the declared
   list in order and extracts only `job_id`, `seconds`, `worker_id`, and
   `lease_id` into immutable items.
5. For each item, the facade preserves the existing conversions and clamps
   seconds to `[1, current JOBS_LEASE_MAX_SECONDS]`. The environment value
   remains an operation-time per-item read; this task does not adopt
   `JobsSettings` snapshot behavior.
6. The selected backend operation enters one atomic scope, opens its backend
   cursor where applicable (thereby preserving PostgreSQL RLS setup), and
   processes every command item in input order. Direct SQLite calls inside an
   existing transaction use a savepoint without committing or rolling back the
   caller's surrounding work.
7. PostgreSQL samples the provided clock once before its loop. SQLite samples
   the provided clock before each item update.
8. Matching updates increment `applied_count`; expected no-transitions add
   zero and processing continues.
9. The operation constructs a valid result before leaving the atomic context.
   Normal exit commits or releases its savepoint; unexpected exit rolls back
   the owned transaction or only the batch savepoint.
10. The facade returns `result.applied_count` and closes the connection through
    the established non-fatal cleanup path.

An empty item tuple still follows the connection and backend operation path.
It returns zero and does not read `JOBS_LEASE_MAX_SECONDS`. PostgreSQL retains
its current once-per-batch clock sampling; SQLite performs no per-item clock
calls for an empty tuple.

## SQL Reuse Boundary

Single and batch renewal may share only a private, pure helper that selects the
fixed SQL variant and builds its bound parameters. The helper must not execute
SQL, inspect row counts, classify outcomes, or own a cursor, connection, or
transaction. Single and batch operations retain separate execution and result
handling. They must not share:

- connection acquisition
- transaction contexts
- no-transition classification reads
- public result mapping
- post-operation side effects

The single-job operation continues to return `LifecycleResult` and classify a
zero-row update. The batch operation only needs exact applied-attempt counts;
it must not issue extra classification queries for expected no-ops.

## Error Handling

- Missing, wrong-status, and stale enforced identities are expected zero-row
  outcomes and do not abort the batch.
- Existing `int(...)` conversion failures retain their built-in exception
  classes. Exact interpreter-specific messages are not part of the contract.
- The facade opens the connection before item normalization, preserving current
  precedence when both connection setup and input conversion would fail. It
  completes normalization before dispatch, so malformed input cannot occur after
  a backend operation starts mutating rows.
- Clock failures and unexpected SQLite/PostgreSQL errors propagate unchanged.
- The backend atomic context rolls back every earlier batch update before an
  unexpected exception reaches the facade without closing a caller-owned
  SQLite transaction.
- No error is logged and suppressed in the backend operation.
- Connection-close failures retain the existing non-fatal cleanup behavior.

## Test Strategy

### Public characterization before extraction

Add or strengthen SQLite and required real-PostgreSQL coverage for:

- exact count for several valid items
- mixed valid, missing, wrong-status, and stale-identity items
- duplicate job IDs as separate update attempts
- lease non-shortening
- duration clamping below, within, and above the configured maximum
- representative malformed inputs with matching exception classes and no
  durable mutation
- empty-batch behavior
- whole-batch rollback after a later database-triggered failure, verified from
  a fresh connection

These tests must pass against the inline implementation before routing changes.

### Typed contracts and direct operations

Contract and direct-operation tests begin red because the new types and entry
points do not exist. They cover:

- immutable command item snapshots
- requested/applied count invariants, including a narrow property test
- import boundaries that exclude `JobManager`
- exact result counts and durable lease values
- logical no-ops alongside committed valid renewals
- duplicate attempts and non-shortening behavior
- rollback after a later database-triggered failure
- rollback after a later SQLite clock failure
- PostgreSQL once-per-batch and SQLite per-item clock sampling at the operation
  boundary

Database-triggered and clock-failure rollback tests are the behavioral evidence
that the complete mutation loop remains inside one transaction. Tests must not
spy on result construction or transaction-wrapper internals.

PostgreSQL trigger and function names must be unique. Cleanup runs in `finally`
through a fresh connection so failed transactions cannot leak test objects.
SQLite uses a disposable per-test database.

### Facade routing

Small routing tests verify typed command dispatch to each backend and mapping
of `applied_count` to the public integer. They do not assert transaction-wrapper
internals, introspect the Python signature, or add side-effect spies.

### Verification gates

- New focused contract, facade, SQLite, and required real-PostgreSQL tests with
  zero PostgreSQL skips.
- Existing single-job renewal/release operation and parity suites.
- The established neighboring Jobs regression matrix.
- Ruff and compileall on touched Python paths.
- Bandit on every touched production path.
- `git diff --check` and operation-module `JobManager` boundary scans.
- Independent whole-branch review before pull request preparation.

Each new test module carries exactly one accepted test-type marker. PostgreSQL
modules additionally use `pg_jobs` as an infrastructure marker.

## Risks And Mitigations

### Accidental per-item commits

Mitigation: backend operations own one atomic scope, and durable
database-trigger tests verify rollback after an earlier successful update.
Direct SQLite tests additionally prove a savepoint preserves the caller's
transaction and unrelated uncommitted work.

### Single-job regression from helper reuse

Mitigation: share only pure statement-and-parameter construction and rerun
existing single-job renewal/release suites unchanged.

### Settings semantics drift

Mitigation: preserve per-item environment reads and defer any snapshot policy
to the dedicated `JobsSettings` adoption slice.

### Backend timing drift

Mitigation: inject the existing manager clock and preserve PostgreSQL batch
sampling versus SQLite per-item sampling in direct-operation tests.

### Deadlocks or oversized transactions

The extraction preserves current input ordering and unbounded batch size. A
future change may add deterministic lock ordering, chunking, or limits only
after production batch-size and contention measurements establish the need.

## Delivery Boundary

The implementation should remain one reviewable PR after this design and its
implementation plan are approved. It may contain separate commits for public
characterization, contracts, SQLite extraction, PostgreSQL extraction, facade
routing, and final tracking. No terminal or other batch lifecycle family may
be added to the PR.
