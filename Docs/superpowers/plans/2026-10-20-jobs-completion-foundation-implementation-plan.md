# Jobs Completion Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the strict-extraction foundation without routing production completion through a new operation.

**Architecture:** Characterize the existing facade, introduce completion-only shallow contracts, and move existing shared persistence/commit/exception helpers mechanically into DB_Management. JobManager retains its existing completion SQL, transaction contexts, policy timing, callback binding, and observer dispatch. Backend operations and facade binding adapters belong to the later routing PR, not this foundation.

**Tech Stack:** Existing Python/dataclasses/typing, pytest/Hypothesis, SQLite, psycopg, repository PostgreSQL fixtures, Ruff, Black, Bandit, MkDocs. No new dependencies.

**Spec:** [Approved completion extraction design](../specs/2026-10-04-jobs-completion-extraction-design.md), approved 2026-10-20.

**Tracking:** TASK-13871; assessment TASK-13421. Status: plan awaiting human review/execution selection; all implementation stages Not Started.

**Baseline:** Live dev `454974e812f8faf8532c931f3f7c8e0bf8d9445c`; this documentation branch is rebased onto it. The 16 completion/policy/context/helper function ASTs are unchanged from the approved October 4 baseline. Fresh existing completion/event/RLS baseline: 86 passed, 227 warnings, no skips; `/tmp/jobs_completion_plan_baseline_20261020.xml`. This is not foundation qualification.

**ADR Check:** No additional decision required. [ADR-058](../../ADR/058-jobs-completion-row-identity.md) governs row identity/atomicity; [ADR-078](../../ADR/078-jobs-completion-persistence-boundary.md) governs the approved persistence boundary. ADR-078 is the collision-free renumbering of this branch's unpublished ADR-060, with unchanged rationale. Never rewrite accepted rationale or include an unrelated ADR from another PR.

## Global Constraints

- "The public signature, boolean result, accepted input behavior, validation order, environment-read timing, error precedence, and observer behavior remain stable."
- "Effect facts contain data, not callbacks or a manager reference."
- "Construction adds no validation, coercion, deep copy, recursive freezing, or serialization."
- "Do not add or route the new backend completion operations in this PR."
- "Shared SQL helpers never commit or roll back the owning transaction themselves; the PostgreSQL commit guard retains its existing aborted-transaction rollback."
- Preserve stored UUID values and null-safe predicates, one final serialization, SQLite `BEGIN IMMEDIATE`, PostgreSQL RLS-aware `FOR UPDATE`, savepoint controls and exact suppression tuples.
- Ordinary defects get separate uniquely identified tasks/PRs; stop for security/cross-tenant failure, irreversible data loss, broken atomicity, or success before commit.
- No fail/cancel/batch extraction, caller migration, historical repair, schema/config/CI changes, generic terminal engine, or broader cleanup. Existing incoming admission, `quarantine=False`, WorkerSDK accounting, and fixture behavior remain intact.
- Do not modify the dirty primary checkout or historical foundation branch. Check live dev and identifier collisions again at execution; update/re-review this plan if the completion contract has changed.

## Review Focus

1. A callback replaced during commit must not replace an already captured observer; a gauge-binding failure must retain the metric prefix but omit tracing (Stage 1).
2. Missing/stale/terminal rows with a nonserializable result must retain current short-circuit and conversion precedence, not fail in a new constructor (Stages 1-2).
3. Optional metric preparation must stay optional, including partial prefixes, while savepoint controls/counters/core outbox remain fatal (Stages 1-2).
4. A missing counter in admin reschedule/immediate-retry must still reconcile correctly after shared-helper relocation, including real PostgreSQL (Stages 1, 3-4).
5. Missing psycopg, cold import order, and manager helper aliases must not introduce mandatory driver imports, cycles, or changed exception identity (Stages 3-4).

## Files and Boundaries

Paths below are repository-relative. Runtime creation is authorized only after plan approval.

| Create | Responsibility |
| --- | --- |
| `tldw_Server_API/tests/Jobs/_completion_characterization_support.py` | Small real-connection probes and callback recorders, not a new provisioning framework. |
| `tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py` | Paired facade behavior and fault ordering. |
| `tldw_Server_API/tests/Jobs/test_jobs_completion_contracts.py` | Shallow contracts, invariants, lazy interfaces, and property tests. |
| `tldw_Server_API/tests/Jobs/test_jobs_completion_helper_boundaries.py` | Moved helper behavior, import/alias identity, driver absence and dependency boundaries. |
| `tldw_Server_API/app/core/DB_Management/jobs_completion_contracts.py` | Data contracts and executor-free lazy policy/preparation interfaces only. |
| `tldw_Server_API/app/core/DB_Management/jobs_lifecycle_persistence.py` | Existing event insertion and missing-counter reconciliation definitions. |
| `tldw_Server_API/app/core/DB_Management/jobs_transaction_helpers.py` | Existing PostgreSQL commit guard, retaining its function-local driver import. |
| `tldw_Server_API/app/core/DB_Management/jobs_exception_policy.py` | Exact existing optional-driver construction and noncritical tuple. |

Modify only `tldw_Server_API/app/core/Jobs/manager.py` helper imports/definitions, task/plan progress, and narrowly necessary tests in the spec's shared-helper matrix. Do not create `jobs_completion_sqlite.py`, `jobs_completion_postgres.py`, a facade binding registry/adapter, or a forwarding Jobs wrapper in PR 1. No new backend completion entry point is used by production.

New contract tests use `pytest.mark.unit`; facade characterizations use
`pytest.mark.integration`; helper tests mark pure doubles/import checks as unit
and real transactions as integration. PostgreSQL cases additionally carry
`pg_jobs`. Do not put module-level `importorskip("psycopg")` in a mixed module:
SQLite/contract/import-absence tests must remain runnable without the driver.

## Stage 1: Characterize the Current Facade

**Goal:** Establish reusable behavior tests before any production move.
**Success Criteria:** New characterizations pass on the untouched baseline; paired paths use real fixture-owned databases and pin the five review-focus conditions.
**Tests:** New characterization module plus existing identity, lifecycle, admin-counter and RLS suites.
**Status:** Not Started

**Interfaces:** Consume current `JobManager.complete_job(job_id, *, result=None, worker_id=None, lease_id=None, completion_token=None, expected_uuid=None, enforce=None) -> bool`. Test support exports `record_completion_observers(manager, monkeypatch, calls: list[str]) -> None` and `record_completion_boundaries(manager, monkeypatch, calls: list[str], *, on_commit: Callable[[], None] | None = None, fail_at: str | None = None) -> None`; proxies delegate real execute/fetch/context/commit/rollback/close rather than faking successful SQL.

- [ ] Add the support/module pair. `completion_manager` is a module-local fixture parametrized with SQLite and `pytest.param("postgres", marks=pytest.mark.pg_jobs)`. SQLite uses `tmp_path`; PostgreSQL obtains the existing `jobs_pg_dsn` through `request.getfixturevalue`. New paired modules opt in with `USE_SHARED_JOBS_POSTGRES = True` and `pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]`, following the current failed-requeue tests. Do not modify global conftests or migrate historical modules.
- [ ] Record calls only after seeding a job; mock external observers/telemetry, not persistence. Wrap the first owning connection/cursor; do not accidentally wrap the pre-existing SLA policy lookup as a second completion transaction.
- [ ] Write the following binding assertions using the probes. A real processing job has valid lease/token/UUID; counters/outbox are enabled, SLA policy returns no breach, test diagnostics are disabled for these order assertions:

These snippets show each test's required assertions, not complete test bodies.
Each test creates `calls: list[str]`, seeds/acquires its job before installing
the probes, and creates an independent `reader` manager against the same
fixture-owned database. Invoke the facade with the acquired `job_id`, worker,
lease, UUID and `completion_token="binding-token"`. Assert True except in the
outside-tuple observer-error case, which uses `pytest.raises(AssertionError)`.

```python
def test_commit_time_replacement_keeps_original_callback(completion_manager, monkeypatch):
    # Arrange recorder and a commit hook that replaces manager._update_gauges.
    # Call complete_job with the real acquired identity and valid token/lease.
    assert calls == ["commit", "duration", "completed", "original_gauges", "trace", "event", "context_exit", "close"]

def test_suppressed_gauge_binding_failure_preserves_metric_prefix(completion_manager, monkeypatch):
    # Use a scoped descriptor/__getattribute__ probe after seeding, not a callback-body error.
    assert calls == ["commit", "duration", "completed", "event", "context_exit", "close"]

def test_unsuppressed_observer_failure_keeps_committed_state(completion_manager, monkeypatch):
    # A selected observer raises AssertionError, which is outside the exact tuple.
    assert reader.get_job(job_id)["status"] == "completed"
```

For each, also assert the current public result/exception and durable event/counter state; the failing observer case must assert propagation and absence of later callbacks. PostgreSQL records cursor exit separately before connection exit; extend the expected order accordingly, never hide that boundary behind a common list.

- [ ] Add parameterized tests with these exact preservation outcomes:

| Test name | Required assertion(s) |
| --- | --- |
| `test_completion_identity_short_circuits_result_work` | Missing/stale/terminal cases: `assert transforms == []`; no final serialization/observer; exact existing replay boolean. Include stored UUID `None`, `""`, and a noncanonical string with matching/mismatching expected values. |
| `test_completion_serialization_and_connection_error_precedence` | Oversize preflight fails before connect; an initial TypeError/ValueError size-probe failure is deferred; connect failure wins before final serialization; rejected nonterminal guard still performs exactly one final serialization after transformation. |
| `test_completion_preserves_status_token_and_lease_matrix` | Enforced/unenforced processing and allowed/disallowed queued-ready/queued-scheduled rows; conflicting token, lease/worker mismatch, completed matching-token replay and non-completed terminal rejection retain exact boolean/mutation/event outcomes. Reuse identity/concurrency cases rather than duplicate them. |
| `test_completion_metric_preparation_boundaries` | Truncation and duration groups fail independently; a metric clock/parser failure suppresses the current group without failing completion. Preserve encrypted `_truncated` behavior, raw tracing projection, PG manager-clock vs SQLite naive UTC sampling, and the later independent SLA sample. |
| `test_completion_policy_read_timing` | Change allowlist after transform and counters policy after metric preparation; reads happen only at current points. Early exits do not evaluate later callbacks. Observer-internal event/gauge settings still read after commit. |
| `test_completion_cleanup_and_commit_failure_order` | Applied, replay, false and primary-error paths retain context/cursor exit and close order; allowed cleanup exceptions are suppressed, outside-tuple failures retain masking; explicit commit failure emits no observers. |
| `test_completion_sla_policy_lookup_lifetime` | Lookup uses its existing separate read-only connection under the owning lock, closes it before SLA writes, preserves lookup/close suppression, and never owns completion commit. |
| `test_completion_diagnostic_failure_keeps_pg_commit_guard` | Current test-mode entry/read ordering retained; a real PostgreSQL diagnostic statement abort is not converted into successful completion by suppression. |

- [ ] Run `RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py`. Expect all paired cases pass and no substantive PG skips. Characterization is intentionally green on unchanged production; do not manufacture a production defect to obtain red. Prove selected assertions are live with a scoped test-only adversarial probe (wrong binding/order raises an assertion), and keep production unchanged.
- [ ] Run existing shared-helper matrix tests before relocation: `test_jobs_lifecycle_event_atomicity.py`, `test_jobs_lifecycle_hardening_regressions.py`, `test_jobs_dependency_acquire_counter_regressions.py`, `test_jobs_admin_retry_transaction_boundaries.py`. Add only missing cases for each current helper caller/branch. Keep ordinary discovered defects in separate tasks, not this PR.
- [ ] Commit characterization and TASK-13871/plan evidence with normal hooks: `test(jobs): characterize completion extraction boundaries (TASK-13871)`.

## Stage 2: Add Completion-Only Contracts

**Goal:** Define data-only boundaries without changing production routing or accepted input behavior.
**Success Criteria:** Direct tests prove shallow identity preservation, lazy callbacks, legal internal result states, and no driver/Jobs dependencies.
**Tests:** `test_jobs_completion_contracts.py`.
**Status:** Not Started

**Interfaces:** All types below live only in `DB_Management/jobs_completion_contracts.py`. Use frozen dataclasses, string enums, `Protocol` and `Callable`; no Pydantic validation, copying, environment reads or I/O.

```python
CompleteJobCommand(job_id: int, result: dict[str, Any] | None,
    worker_id: str | None, lease_id: str | None, completion_token: str | None,
    expected_uuid: str | None, enforce: bool, outbox_enabled: bool)
CompletionOutcome = APPLIED | IDEMPOTENT_REPLAY | NO_TRANSITION
CompletionNoTransitionReason = MISSING_ROW | IDENTITY_MISMATCH | TERMINAL_STATE | GUARD_REJECTED
CompletionOrigin = PROCESSING | QUEUED
```

Annotations do not authorize runtime input validation. A shallow command retains the original result/ID/token/UUID objects, even unusual inputs accepted until a later backend conversion. Enum values use lowercase names (`"applied"`, `"missing_row"`, etc.).

Each frozen effect has `binding_id: int`, local to one facade attempt, and only the following prepared data. IDs are not persisted, globally allocated, or used to choose an observer in the DB layer.

| Effect type | Additional fields |
| --- | --- |
| `JsonTruncatedEffect` | `job: dict[str, Any]`, `kind: Literal["result"] = "result"` |
| `DurationEffect` | `job: dict[str, Any]`, `started_at: Any`, `completed_at: Any`; preserve raw prepared values, not new datetime validation. |
| `CompletedEffect` | `job: dict[str, Any]` |
| `GaugesEffect` | `domain: Any`, `queue: Any`, `job_type: Any`; preserve keyword-call semantics. |
| `TraceEffect` | `job: dict[str, Any]`, `operation: Literal["job.complete"] = "job.complete"` |
| `SlaWarningEffect` | `job_id: int`, `error_type: str` |
| `SlaEventEffect` | `job: dict[str, Any]`, `attrs: dict[str, Any]` |
| `SlaMetricEffect` | `job: dict[str, Any]`, `kind: Literal["duration"] = "duration"` |
| `CompletionEventEffect` | `job: dict[str, Any]`, `attrs: None = None` |

`CompletionEffect` is the union of these nine types. `CompletionDurableEvent(event_type: Literal["job.sla_breached", "job.completed"], job: dict[str, Any], attrs: dict[str, Any] | None)` records inserted outbox facts only, independently of observers/outbox disabled state.

```python
CompleteJobResult(outcome: CompletionOutcome,
    reason: CompletionNoTransitionReason | None = None,
    origin: CompletionOrigin | None = None,
    locked_row: dict[str, Any] | None = None,
    effects: tuple[CompletionEffect, ...] = (),
    durable_events: tuple[CompletionDurableEvent, ...] = ())
CompleteJobResult.applied(*, origin, locked_row, effects, durable_events) -> CompleteJobResult
CompleteJobResult.replay() -> CompleteJobResult
CompleteJobResult.no_transition(reason: CompletionNoTransitionReason) -> CompleteJobResult
CompleteJobResult.public_success -> bool  # property: applied/replay True, no-transition False
```

Validate internal envelope consistency only: applied has row/origin and no reason; replay/no-transition cannot carry row/origin/effects/durable facts, and only no-transition carries a reason. Reject duplicate binding IDs and invalid effect order, not arbitrary payload values. Metric effects are optional and may be partial prefixes; never require a full duration/completed/gauge/trace group. A completed observer occurs once and last; SLA success carries event then metric, optional SLA failure carries warning instead, and neither is required without a breach. Durable facts, when present, preserve SLA-before-completion order and allow an empty tuple when outbox is disabled. All validation occurs before commit in PR 2; PR 1 does not route these types.

`CompletionPolicies` is an executor-free Protocol with these exact method signatures; it contains lazy behavior, not observer dispatch:

```python
transform_result(result: Any, domain: str) -> Any
queued_domain_allowed(domain: Any) -> bool
counters_enabled() -> bool
prepare_metric_effects(backend: str, job: dict[str, Any], result: Any) -> tuple[CompletionEffect, ...]
duration_sla_breach(job: dict[str, Any]) -> tuple[float, float] | None
prepare_lifecycle_event(event_type: str, job: dict[str, Any], attrs: dict[str, Any] | None) -> SlaEventEffect | CompletionEventEffect
prepare_sla_metric(job: dict[str, Any]) -> SlaMetricEffect
prepare_sla_warning(job_id: int, error_type: str) -> SlaWarningEffect
```

The later facade adapter owns binding/parsing/clock preparation together, so it can preserve callable-before-argument evaluation and the separate truncation/grouped-metric suppression boundaries. `prepare_metric_effects` must permit a partial prepared prefix. Do not add a separate eager duration callback that samples clocks before callback binding. Implement neither that adapter nor a registry in this foundation; test the Protocol using inert recording stubs.

- [ ] Write direct tests before the module exists. `test_command_is_shallow_and_inert` asserts `command.result is original` and unchanged ID/UUID/token identity with objects whose `__int__`, `__str__`, `__deepcopy__` and JSON conversion fail if invoked. Assert assignment raises `FrozenInstanceError` while nested data is not recursively frozen.
- [ ] Write parameterized outcome/invariant tests and Hypothesis tests: `assert replay.public_success is True`; `assert no_transition.public_success is False`; `assert applied.locked_row is row`; all legal metric-prefix lengths accepted, conflicting SLA forms/duplicate bindings/payload on replay rejected. Constructors must not call any recording policy stub. Test raw `None`, empty/noncanonical UUIDs, naive/aware/raw timestamp fields without normalization.
- [ ] Run `RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_contracts.py`; expect missing-module/import failure before implementation, not an unrelated fixture failure.
- [ ] Implement the specified contracts only, with module/class/function docstrings and type hints; no backend or manager imports. Run the same tests to green and rerun Stage 1. Add a cold-import check that importing this contracts module loads neither psycopg nor Jobs modules.
- [ ] Commit contracts/tests/plan/task notes: `refactor(jobs): add shallow completion contracts (TASK-13871)`.

## Stage 3: Relocate Shared Helpers Mechanically

**Goal:** Place existing helper SQL/commit/exception ownership in DB_Management without changing callers.
**Success Criteria:** Moved definitions and complete_job are structurally equivalent; aliases and optional-driver behavior remain compatible; every helper caller is verified.
**Tests:** `test_jobs_completion_helper_boundaries.py`, Stage 1, and the shared-helper matrix.
**Status:** Not Started

**Interfaces:** Copy these definitions without body/SQL/signature changes:

```python
# jobs_lifecycle_persistence.py
_insert_lifecycle_event(executor: Any, *, backend: str, event_type: str,
    job: dict[str, Any], attrs: dict[str, Any] | None = None) -> None
_reconcile_lifecycle_counter_row(executor: Any, *, backend: str,
    domain: Any, queue: Any, job_type: Any) -> None
# jobs_transaction_helpers.py
_commit_postgres_transaction(conn: Any, *, operation: str) -> None
# jobs_exception_policy.py
_PG_ERRORS: tuple[type[BaseException], ...]
_JOB_NONCRITICAL_EXCEPTIONS: tuple[type[BaseException], ...]
```

The policy module takes the existing `try: import psycopg`/`except ImportError` construction and tuple verbatim. At the original import-time point, manager imports those two objects and conditionally reexports `psycopg` from the policy module only when `_PG_ERRORS` is nonempty, retaining the existing present/absent binding. The commit helper retains `from psycopg.pq import TransactionStatus` inside its function. Persistence helpers depend only on their existing standard-library needs. Do not import manager, Jobs contracts, observers or metrics from any new DB module.

- [ ] Add failing import/identity tests before moving definitions: manager's three helper names and two tuple objects must be `is` their new module exports. Import failure for the not-yet-existing modules is the expected red.
- [ ] Add direct event-helper tests on real SQLite/PG transactions: attribution/attrs unchanged; owning rollback removes the row; helper never calls commit/rollback/close. Test missing-counter reconstruction with all four buckets and malformed count/error propagation. Do not snapshot all facade SQL.
- [ ] Add commit-guard tests using narrow connection doubles for normal/INERROR/commit-failure/rollback-failure, then verify real INERROR behavior with fixture-owned PostgreSQL. Assert `rollback` then RuntimeError for aborted state, original rollback error precedence, and no observer invocation.
- [ ] Add isolated-process driver-absence tests: block psycopg imports without uninstalling anything, import the exception module/SQLite path, assert `_PG_ERRORS == ()`, tuple identity and class order match baseline, and no new `TransactionStatus` import occurs until the PG guard is invoked. Test driver-present import too; do not count absence tests as PostgreSQL execution.
- [ ] Move one responsibility at a time and import it back into manager under its exact private name. Keep every caller untouched, including completion/SLA, `fail_job` terminal/retry/quarantine, `reschedule_jobs`, and `retry_now_jobs`. No new observer preparer or production completion-contract import is needed.
- [ ] After each move run the direct helper test plus Stage 1 and affected matrix tests. Preserve `_insert_lifecycle_event`'s `json.dumps(attrs or {})`, raw values, counter SQL, INERROR guard and exact exception-class identity.
- [ ] Before committing, compare `ast.dump(..., include_attributes=False)` for each moved function against execution-baseline `git show` and for `JobManager.complete_job` against the unchanged baseline. Inspect SQL strings/parameter order and imports manually; store the comparison result in task notes, not a brittle permanent facade source snapshot. Changed AST bodies require explanation/review, not automatic acceptance.
- [ ] Commit mechanical moves and evidence: `refactor(jobs): relocate lifecycle persistence helpers (TASK-13871)`.

## Stage 4: Qualify and Prepare the Foundation PR

**Goal:** Prove the foundation is independently reviewable/revertible without routing completion.
**Success Criteria:** All applicable gates pass on the candidate head; scoped source/SQL/dependency review finds no behavior change; any unrelated failure has an evidence-backed disposition, not a weakened test.
**Tests:** Full non-PG Jobs, required real PG completion/lifecycle/admin/parity/RLS, opt-in outbox/SSE, touched-scope quality/security, Docs, and hosted gates.
**Status:** Not Started

**Interfaces:** Consume the modules/tests from Stages 1-3. Produce verification artifacts, an accurate task/plan record and a foundation-only PR when separately authorized; do not produce backend operation implementations.

All commands run from this worktree after `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate`. Use a fixture-supported test cluster; no manual database provisioner or shared production database. Do not modify shared venv dependencies to force a pass.

- [ ] Run new direct/characterization tests on their own and in the combined Jobs suite so plugin/import-order regressions cannot hide behind an earlier module's registration.
- [ ] Run full non-PG: `RUN_JOBS=1 python -m pytest -q -m "not pg_jobs" tldw_Server_API/tests/Jobs --junitxml=/tmp/task13871_sqlite.xml`. Record existing skips explicitly; do not interpret a skipped relevant case as success.
- [ ] Run required PG matrix with `RUN_JOBS=1 RUN_PG_JOBS_TESTS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q -m pg_jobs` and these complete modules under `tldw_Server_API/tests/Jobs/`:

```text
test_jobs_completion_characterization.py
test_jobs_completion_helper_boundaries.py
test_jobs_completion_row_identity_atomicity.py
test_jobs_completion_idempotent_postgres.py
test_jobs_finalize_idempotency_postgres.py
test_jobs_lifecycle_event_atomicity.py
test_jobs_lifecycle_hardening_regressions.py
test_jobs_dependency_acquire_counter_regressions.py
test_jobs_admin_retry_transaction_boundaries.py
test_jobs_rls_postgres.py
test_jobs_events_outbox_postgres.py
parity/test_postgres_parity.py
```

Append `--junitxml=/tmp/task13871_postgres.xml`. Expected: zero failures and zero substantive skips, not a fixed stale count. The paired reschedule/immediate-retry modules cover missing counters; completion-only evidence is insufficient. Also run new contracts and non-PG helper tests; a `pg_jobs` selection does not select their unit cases.

- [ ] Run relevant opt-in SSE cases locally with `CI` unset and `TLDW_TEST_NO_SSE=0`: `test_jobs_events_sse_sqlite.py`, `test_jobs_sse_smoke.py`, and the enabled PG outbox module. Record actual event/heartbeat results and existing environment skips; if a relevant gate cannot execute, resolve qualification before treating the PR as ready. Do not remove existing guards or increase latency budgets to obtain green.
- [ ] Run `python -m ruff check` on all four new production modules, three new test modules, test support, manager and any narrowly modified existing tests; run `python -m black --check` on new modules/tests only (no manager-wide formatting), `python -m compileall -q` on touched Python files, and `git diff --check`. Use exact touched paths from `git diff --name-only <execution-base>...HEAD`, never include unrelated files.
- [ ] Before production moves, retain baseline Bandit JSON for manager and touched existing tests. After moves run project-venv `python -m bandit -r <touched-production-paths> -f json -o /tmp/bandit_task13871_prod.json` and a separate touched-test scan with `-s B101 -o /tmp/bandit_task13871_tests.json`. Compare semantic findings (test ID, code, severity/confidence), mapping mechanically relocated helper findings to their old definitions rather than treating line-number churn as new defects. No new findings/errors or new suppressions; record inherited warnings and exact comparison, not just the scan exit code.
- [ ] Run `python -m pytest -q tldw_Server_API/tests/Docs` with the repository-supported MkDocs toolchain. If canonical ADR/docs metadata changed, run `bash Helper_Scripts/refresh_docs_published.sh`, require a deterministic second refresh and canonical/generated parity, then rerun strict Docs checks. Never publish this private plan into the public tree.
- [ ] Perform whole-branch review: no new backend completion modules/entry points, unchanged complete_job AST and SQL, mechanically identical helper bodies, no new driver import/DB-to-Jobs dependencies, all policy/preparation/cleanup fault paths accounted for, and spec-to-test coverage including shared callers. Fix only foundation-caused regressions; create separate tasks for validated inherited ordinary defects.
- [ ] Record fresh verification and remaining blockers in TASK-13871; mark stages Complete only when actually complete. With separate PR authorization, use explicit `--repo rmusser01/tldw_server`, target current `dev`, attach the created PR to this chat, and request a new human-written Change summary. Do not reuse PR #3092's summary or treat written-spec/plan approval as merge approval. Obey current repository merge-queue and exact-head hosted/Jobs requirements; no bypass/admin merge or unrelated CI cancellation.
- [ ] Commit final plan/task evidence with normal hooks. Reverting the foundation at its own merge point restores helper definitions/imports and removes unused contracts; after routing depends on it, revert routing first. Routing gets a new task/plan only after this foundation merges.

## Plan Review and Execution Gate

This document plans PR 1 only and contains no implementation. Human review and execution selection are required before Stage 1 begins. If no subagent tools are available, use native `superpowers:executing-plans`; do not create separate user chats as substitute agents or claim independent review that did not occur.

Self-review checklist: map every PR-1 spec requirement to the four stages; verify contract names/signatures agree across stages; verify all five review-focus conditions have named tests; retain partial optional effects and pre-commit binding without routing new code; verify current helper caller inventory and real PG fixtures; recheck new identifiers before publication.
