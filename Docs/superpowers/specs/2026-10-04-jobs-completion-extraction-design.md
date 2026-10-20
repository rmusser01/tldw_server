# Jobs Completion Extraction Design

Date: 2026-10-04
Status: Written specification approved 2026-10-20; foundation implementation plan awaiting human review
Tracking: TASK-13421 (ownership assessment; implementation tasks are separate)
Baseline: `origin/dev` at `c95e41fc62a07e55fd74052023826c71b2a2c789`
Planning revalidation: `origin/dev` at `454974e812f8faf8532c931f3f7c8e0bf8d9445c` on 2026-10-20

## Objective

Extract single-job completion persistence from `JobManager.complete_job` without
changing the behavior established by merged PR #3092. Put completion SQL under
`core/DB_Management`, keep public policy and observers in `JobManager`, and retain
one transaction for the locked row, enabled counters, and enabled outbox records.

Deliver two independently reviewable PRs: foundation first, backend routing
second. Neither PR fixes unrelated defects or introduces a generic terminal
transition engine.

## Context and Approval

PR #3092 merged as `6c5da178252d14dc0d997d97988ccaefdf86bc89`. It repaired a
missing-row/concurrent-insert completion race and established the lock, stored
UUID, optional expected UUID, and atomic bookkeeping contract in ADR-058.
Jobs runtime and tests are unchanged between that merge and this baseline.

The historical 2026-08-23 extraction design on the retained
`codex/jobs-completion-foundation` branch predates that fix. This specification
replaces its extraction proposal, not its historical evidence. In particular:

- `expected_uuid` is now part of the public completion API.
- Missing and stale-identity attempts return without transformation or mutation.
- Queued completion uses the authoritative locked row, with no second domain
  lookup or processing-update fallback.
- Final serialization occurs once after transformation, not once per update.
- SQLite starts `BEGIN IMMEDIATE`; PostgreSQL locks the visible row with
  `FOR UPDATE` through the existing RLS-aware cursor.

The requester approved option A on 2026-10-04: completion-specific atomic
operations and SQL helpers under `DB_Management`, preserving facade-owned
connection/RLS contexts. The requester approved this revised written specification
on 2026-10-20. The separate foundation implementation plan still requires human
review and execution selection before implementation.

At the planning revalidation, all 16 relevant completion, policy, context, and
shared-helper function ASTs match the original baseline. Incoming admission,
failure-policy, WorkerSDK usage-accounting, and PostgreSQL fixture changes remain
intact. The unpublished complementary ADR was renumbered from 060 to 078 because
other open PRs now use 060; its accepted ownership rationale is unchanged.

## Scope

In scope are completion characterization, completion-only contracts, mechanical
helper relocation, and the subsequent SQLite/PostgreSQL completion extraction.
The public signature, boolean result, accepted input behavior, validation order,
environment-read timing, error precedence, and observer behavior remain stable.

Out of scope:

- `fail_job`, cancellation, retry, quarantine, dependencies, and batch completion
- a generic terminal engine or migration of existing admission/lease operations
- schema, API schemas, configuration, environment variables, and settings rollout
- UUID normalization, broad acquired-caller migration (TASK-13216), and historical
  bookkeeping repair (TASK-13217)
- query optimization, backend unification, broader error suppression, or fixes
  for observer/cleanup and encryption/truncation quirks

The obsolete foundation task ID TASK-13112 belongs to unrelated current work.
Do not reuse it or its historical child IDs. Create uniquely identified tasks
for the foundation, routing, and any newly validated follow-up defects.

## ADR Check

ADR required: yes. [ADR-078](../../ADR/078-jobs-completion-persistence-boundary.md)
records the durable completion persistence boundary approved here.
[ADR-058](../../ADR/058-jobs-completion-row-identity.md) continues to govern locking,
row identity, and atomic bookkeeping. ADR-078 supplements it; neither accepted
rationale nor ADR-058's status changes.

## Alternatives

| Approach | Assessment |
| --- | --- |
| A: Whole completion persistence operation in DB_Management | Selected. Places SQL in the approved layer without splitting the atomic protocol. |
| B: Jobs orchestration over granular database methods | Keeps orchestration near older operations, but spreads ordering and atomicity obligations across many calls. |
| C: SQL in Jobs operations | Matches older extraction locations but retains SQL ownership debt and needs a policy exception. |

Do not generalize failure/cancellation semantics or relocate all existing Jobs
operations to make completion look uniform.

## Ownership and Modules

`JobManager` remains the only public interface for endpoints, workers, and domain
services. It retains preflight preparation, connection creation, test-mode
capture, current connection/cursor context managers, public boolean mapping,
non-durable callback dispatch, and `_close_connection_nonfatal`.

The proposed completion modules are private implementation boundaries, not new
public database APIs:

| Module under core/DB_Management | Responsibility |
| --- | --- |
| `jobs_completion_contracts.py` | Completion command, lazy policy interfaces, dispositions, and typed effect facts; no SQL or Jobs imports. |
| `jobs_completion_sqlite.py` | Full SQLite completion persistence protocol on the supplied connection. |
| `jobs_completion_postgres.py` | Full PostgreSQL completion persistence protocol on the supplied connection and RLS-aware cursor. |
| `jobs_lifecycle_persistence.py` | Mechanical relocation of existing event insertion and missing-counter reconciliation helpers. |
| `jobs_transaction_helpers.py` | Mechanical relocation of the existing PostgreSQL aborted-transaction commit guard. |
| `jobs_exception_policy.py` | The exact existing noncritical exception tuple and optional PostgreSQL exception construction. |

Database modules must not import `JobManager`, Jobs observers/metrics/tracing, or
the existing Jobs operation contracts at runtime. They may depend on the new
completion contracts, shared core exceptions, standard libraries, and backend
drivers. Avoid a wrapper under Jobs that merely forwards to the database module.

The facade passes the existing executor and connection. Operations never open,
close, replace, or pool a connection; create an independent transaction; or
establish a second RLS cursor. Existing implicit context-exit behavior remains
with the facade. Applied operations perform the same explicit commit as today.
Shared SQL helpers never commit or roll back the owning transaction themselves;
the PostgreSQL commit guard retains its existing aborted-transaction rollback.

## Completion Contracts

### Command and Policy Inputs

`CompleteJobCommand` is a frozen, shallow shell containing the original job ID,
prepared result reference, worker ID, lease ID, completion token, optional
`expected_uuid`, effective enforcement flag, and captured outbox flag.

Construction adds no validation, coercion, deep copy, recursive freezing, or
serialization. It must not normalize the expected UUID or move `int(job_id)` or
`str(expected_uuid)` ahead of their existing backend execution points.

Separate lazy policy callbacks provide domain-aware transformation, queued-domain
allowlist evaluation, counter enablement, duration facts, and SLA evaluation.
They receive facts, never executors, and do not commit or dispatch observers.
Reads of dynamic settings and samples of clocks occur at the current points.
Duration facts preserve PostgreSQL's `self._clock.now_utc()` and SQLite's separate
naive `datetime.utcnow()` behavior. SLA evaluation retains its later independent
clock sample. Existing parsing, conversion, and suppression boundaries stay
intact; no single settings snapshot or unified clock is introduced.

The existing `_duration_sla_breach` callback calls `_get_sla_policy`, which opens
and closes its own read-only policy connection and, on PostgreSQL, establishes
that lookup's RLS cursor. Preserve this existing behavior and cleanup/error
precedence. Do not move the policy lookup onto the locked completion executor,
cache it eagerly, or add a new connection. The prohibition on operation-owned
connections concerns the completion write protocol, not removal of this
pre-existing policy lookup. Other SLA-policy SQL ownership is outside this slice.

### Result and Effects

`CompleteJobResult` distinguishes `APPLIED`, `IDEMPOTENT_REPLAY`, and
`NO_TRANSITION`. Applied and matching-token replay map to public `True`; every
other no-transition maps to `False`. Only applied results carry effects.

An applied result carries processing/queued origin, the original locked-row
projection needed by effects, ordered typed effect facts, and inserted durable
event facts. Preserve all currently supplied tracing/event fields, including
raw database values; do not refetch or normalize them.

Use completion-specific reasons for missing row, identity mismatch, terminal
state, or guard rejection only where existing facts distinguish them. Do not
perform diagnostic reads to create finer reasons. Replay/no-transition results
are payload-free beyond their disposition/reason.

Effect facts contain data, not callbacks or a manager reference. Optional metric
preparation can currently fail inside a suppression boundary while completion
still commits. Represent that absence or partial ordered preparation explicitly;
do not require every metric/trace fact, retry its preparation, or broaden the
exception tuple. The transformed result's `_truncated` marker remains the
criterion for its optional metric, even when encryption hides that marker.

Callback-reference binding is also part of preparation, not post-commit dispatch.
Executor-free facade preparation callbacks capture the current callable at each
original preparation point in a facade-local, per-attempt binding registry and
associate it with the corresponding typed effect fact. The database operation
receives only data and opaque binding identifiers, never those callables or a
manager reference. Do not bind every observer eagerly at call entry or resolve
it again after commit. Preserve callback/argument evaluation order and the
original suppression boundaries, including their partial-preparation behavior.

In particular, a suppressed gauge-binding failure preserves already prepared
duration/completed metrics but prevents the later trace fact from being prepared;
completion/SLA processing continues as today. A callback replaced during commit
does not replace the callable already captured for this attempt. Lifecycle-event
and SLA-warning bindings retain their own current preparation points and failure
behavior; do not wrap all binding in the optional metrics suppression boundary.
Dispatch uses the captured bindings and prepared arguments exactly once, with
no observer rebinding, payload rebuilding, or resampling of already prepared
policy facts after commit. Existing dynamic reads inside observer bodies retain
their current post-commit timing.

Construct and validate applied results before explicit commit. Replay and
no-transition results are built before implicit context exit. Validation checks
internal disposition/fact consistency, not new restrictions on accepted caller
inputs or stored values. No arbitrary copying, serialization, parsing, coercion,
or fallible callback-argument conversion may first occur after commit. Facade
dispatch is mechanical translation of prepared facts with current suppression.

## Transaction and Compatibility Protocol

1. Require a completion token when configured; resolve omitted enforcement;
   read result-size/truncation/outbox settings; perform the current best-effort
   size serialization and size rejection/truncation.
2. Open the connection and capture the existing test-mode flag. Enter the same
   connection context and, on PostgreSQL, the same `_pg_cursor` context.
3. SQLite executes `BEGIN IMMEDIATE` before the authoritative lookup. PostgreSQL
   executes the current visible-row `SELECT ... FOR UPDATE`. Preserve current
   test-mode entry-log ordering relative to these statements.
4. A locked miss returns `False`. Compare `expected_uuid`, when supplied, using
   the existing string conversion and null/empty interpretation. A mismatch
   returns `False` before terminal replay or transformation.
5. Classify the locked status and token. Only a completed row with matching
   present stored/supplied tokens replays successfully. Failed, cancelled, and
   quarantined rows do not complete or replay.
6. Transform the result under the existing noncritical suppression boundary,
   then serialize it once. For nonterminal rows, retain serialization even when
   the subsequent status/lease/allowlist guards reject the transition.
7. Dispatch by locked status: enforced processing update, unenforced processing
   update, or permitted unenforced queued update. Every mutation binds numeric
   ID and the raw stored UUID with the current null-safe predicate (`IS ?` on
   SQLite; `IS NOT DISTINCT FROM` on PostgreSQL). Preserve token semantics,
   backend conversion points, `SELECT changes()`/rowcount handling, and SQL
   timestamps. No normalization, extra read, or RETURNING rewrite is allowed.
8. If no mutation applies and a token is supplied, retain the existing defensive
   replay query with the same ID/raw-UUID predicate and replay conditions.
9. Preserve test-mode diagnostic query/log timing and exact suppression. Prepare
   optional truncation, duration, completed-metric, gauge, and trace facts under
   their current exception boundaries before counter handling, capturing their
   callback references at the original preparation points in the facade registry.
10. When enabled, decrement the correct processing, ready, or scheduled counter.
    Reconcile missing counter rows from transaction-local state using unchanged
    helper SQL. Counter failures remain transaction-fatal.
11. Evaluate SLA policy lazily. For a breach, establish the existing savepoint,
    insert the attachment and optional SLA outbox event, then release it.
    Recover an allowed write failure only through successful rollback-to and
    release of that established savepoint. Savepoint creation/rollback/release
    failures remain fatal. Return either SLA observer facts or the existing
    deferred warning facts, not both.
12. Insert enabled `job.completed` outbox data and prepare its observer facts.
    These writes and counters must identify the same locked row incarnation.
13. Build the applied result, perform the original explicit commit (including
    the PostgreSQL INERROR guard), and return the prepared result only if commit
    succeeds. Replay/no-transition retain current implicit context-exit behavior.
14. Run applied effects after explicit commit, before cursor/connection context
    exit: truncation metric, duration observation, completed metric, gauges,
    trace, SLA warning or SLA observer then SLA metric, completion observer.
    Use the pre-commit captured bindings, preserving partial preparation and
    exact exception suppression/propagation.
15. Exit the original contexts and close in the current `finally` boundary.
    Context-exit failures can still prevent a public result. Cleanup exceptions
    outside the existing suppression tuple retain their current masking behavior.

An initial nonserializable result can pass the size probe and fail only in final
serialization after connection creation. Locked missing/stale/terminal paths can
return without that final serialization. Connection/conversion/serialization
error precedence is characterized, not simplified. No effects run after failed
commit, rollback, replay, or no transition.

## Delivery and Verification

### PR 1: Foundation

Create a uniquely identified foundation task and an approved test-first plan.
Add paired SQLite/real-PostgreSQL characterization, completion contracts, and
mechanical relocations of the three helper responsibilities above plus the
exception policy. Import moved helpers back into `manager.py` under their
existing private names for remaining callers; preserve signatures, SQL,
exception-class identity, and call order. Do not introduce a new module-level
driver import that changes optional-driver behavior.

`complete_job` remains the production implementation. Its control flow, SQL,
SLA staging, explicit commit, and observer routing remain unchanged except for
using mechanically relocated helper definitions. Do not add or route the new
backend completion operations in this PR. Verify all other lifecycle callers
affected by shared helper relocation, not just completion.

Characterization covers successful processing/queued ready/queued scheduled
completion; preflight/connection/final-serialization precedence; missing/stale
UUID/raw-UUID/token cases; contention and concurrent replay; enabled/disabled and
missing counters; RLS filtering; event attribution; SLA savepoint success/write
recovery/control failure; commit/rollback failure; and dynamic policy timing.

Test effect presence, cardinality, order, optional preparation failure, and
callback exceptions inside/outside the exact tuple. Assert explicit commit,
context exit, and close ordering, including early returns and primary failures.
Characterize callback replacement during commit and binding failures before
commit on both backends: the original captured callback still runs, and a
suppressed gauge-binding failure preserves the metric prefix but omits tracing.
No observer binding or dispatch runs for replay/no-transition paths. These are
preservation tests against existing completion, not behavioral corrections.
Add direct command/result invariant tests without imposing new input validation.
Use existing real Jobs PostgreSQL fixtures, not a parallel database provisioner.

#### Shared-Helper Verification Matrix

The foundation plan must cover every current caller of each moved helper on its
affected backend, including branches that only invoke it after an initial
counter update finds no row. Preserve manager aliases for remaining callers.
The existing test modules below are starting coverage, not proof that every
branch has already been characterized; add focused paired coverage where needed.
All test-module names in this table are under `tldw_Server_API/tests/Jobs/`.

| Moved responsibility | Current callers | Required verification |
| --- | --- | --- |
| `_insert_lifecycle_event` | `_stage_completion_sla_breach`; `complete_job`; `fail_job` terminal, retry, and quarantine branches | SQLite and real PostgreSQL event attribution, outbox flag behavior, SLA savepoints, write/commit failures, and observer cardinality; `test_jobs_lifecycle_event_atomicity.py` and `test_jobs_lifecycle_hardening_regressions.py`. |
| `_reconcile_lifecycle_counter_row` | `complete_job`; `fail_job` terminal branch; `reschedule_jobs`; `retry_now_jobs` | SQLite and real PostgreSQL missing-counter and reconciliation-failure paths; terminal original bucket, reschedule `set_now`/delay branches, and immediate retry failed/queued-scheduled branches. Start with `test_jobs_lifecycle_hardening_regressions.py`, `test_jobs_dependency_acquire_counter_regressions.py`, and `test_jobs_admin_retry_transaction_boundaries.py`. |
| `_commit_postgres_transaction` | `complete_job`; `fail_job` retry/quarantine and terminal branches | Real PostgreSQL successful commit, INERROR rollback/rejection, commit/rollback failures, and no pre-commit observers; lifecycle atomicity/hardening suites plus direct guard characterization. |
| `_JOB_NONCRITICAL_EXCEPTIONS` and optional driver construction | Existing manager-wide suppression boundaries and the relocated helpers | Exact ordered exception-class identity/membership and optional-driver import behavior; completion binding/preparation and observer/cleanup failpoints, affected lifecycle/admin suites, and the full non-PG Jobs suite. Do not restrict validation to the completion paths or introduce eager driver imports. |

The plan records exact test selections for this matrix. PostgreSQL lifecycle and
admin missing-counter cases are mandatory in addition to completion/finalize/
parity/RLS gates; a completion-only run cannot qualify shared-helper relocation.
Refresh the caller inventory against the implementation baseline before moving
helpers, without expanding the extraction into those callers' behavior changes.

### PR 2: Backend Routing

Start from then-current live dev after the foundation merges, with a separate
task and reviewed routing plan. Move the complete backend persistence protocols
and completion-only SLA persistence into the database modules. Retain lazy SLA
policy evaluation and existing facade observer handling; do not pass
`_stage_completion_sla_breach` as an executor-bearing callback into the operation.

Add direct backend operation tests, facade dispatch tests, import/dependency
checks, and rerun every foundation characterization against routed completion.
The extraction removes completion-owned persistence SQL from the facade, not
RLS setup or unrelated lifecycle SQL. Existing migrations and acquisition/lease
operations remain where they are.

### Gates for Both PRs

- Focused red/green coverage before each production move, full non-PG Jobs suite,
  and required real-PostgreSQL Jobs completion/finalize/parity/RLS coverage.
- No substantive PostgreSQL skips. Explicitly enable relevant opt-in outbox/SSE
  cases; describe unrelated existing skips without counting them as passing.
- SQL/parameter and dependency-diff review for mechanical moves; no broad SQL
  snapshots. Existing narrow semantic failpoints may prove failure ordering.
- Touched-scope Ruff, scoped Black, compilation, whitespace, and project-venv
  production/test Bandit compared with baseline: no new findings or errors.
- Canonical docs/published parity and strict documentation build for ADR changes.
- Current-head review and repository-required hosted gates, plus Jobs suites.
- A new requester-written Change summary before each materially AI-authored PR
  merges. PR #3092's human summary is not reused as approval of extraction.

The assessment baseline ran 86 existing SQLite/required real-PostgreSQL tests
with no skips (XML `/tmp/task13421_completion_baseline.xml`). It is initial
evidence only, not full qualification of either future extraction PR.

## Defects, Risks, and Rollback

Ordinary validated defects are recorded in uniquely identified follow-up tasks
and separate PRs. Characterization preserves current behavior with references
to those tasks. Stop the extraction for a newly validated security/cross-tenant
failure, irreversible data loss, broken atomicity, or success before commit;
obtain a separate remediation decision rather than freezing that defect.

The main risks are split transaction ownership, changed error precedence,
optional facts becoming mandatory, driver/import cycles, shared-helper blast
radius, and stale task/ADR identifiers. Mitigate them with one operation on the
existing executor, exact lazy timing, dependency checks, full lifecycle coverage,
and fresh live-index checks before assigning identifiers.

Fault-injection coverage must also preserve the existing read-only SLA policy
connection lifetime while the completion lock is held, including lookup/close
failure suppression. It must not treat that policy connection as a second owner
of completion writes.

Neither PR changes schema or configuration. Reverting routing restores the
characterized facade implementation while leaving foundation contracts/helpers
available. Reverting foundation while routing depends on it is not safe; revert
routing first or revert both in dependency order. Each PR must stand alone and
be revertible at its own merge point.

## Next Approval Gate

Written-specification review is complete. Review the
[foundation implementation plan](../plans/2026-10-20-jobs-completion-foundation-implementation-plan.md)
and select execution before implementation. TASK-13871 covers PR 1 only; PR 2
gets a separate task and plan after the foundation merges. TASK-13421 records the
completed ownership assessment and written-design approval. No completion
implementation is included here.
