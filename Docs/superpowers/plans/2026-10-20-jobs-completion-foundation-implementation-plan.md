# Jobs Completion Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver the strict-extraction foundation without routing production completion through a new operation.

**Architecture:** Characterize the existing facade, introduce completion-only shallow contracts, and move existing shared persistence/commit/exception helpers mechanically into DB_Management. JobManager retains its existing completion SQL, transaction contexts, policy timing, callback binding, and observer dispatch. Backend operations and facade binding adapters belong to the later routing PR, not this foundation.

**Tech Stack:** Existing Python/dataclasses/typing, pytest/Hypothesis, SQLite, psycopg, repository PostgreSQL fixtures, Ruff, Black, Bandit, MkDocs. No new dependencies.

**Spec:** [Approved completion extraction design](../specs/2026-10-04-jobs-completion-extraction-design.md), approved 2026-10-20.

**Tracking:** TASK-13878; assessment TASK-13421. Human approved the plan and selected subagent-driven execution on 2026-10-20. Stages 1-3 and final-review F1/F2 corrections are locally qualified. Stage 4 and the foundation remain In Progress for scoped controller re-review, separately authorized integration and outstanding hosted/human gates. Provisional Jobs TASK-13871 and TASK-13875 are archived because their IDs also belong to VN and Moderation records. Those unrelated records remain unchanged.

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

## Task 1: Characterize the Current Facade

### Stage 1

**Goal:** Establish reusable behavior tests before any production move.
**Success Criteria:** New characterizations pass on the untouched baseline; paired paths use real fixture-owned databases and pin the five review-focus conditions.
**Tests:** New characterization module plus existing identity, lifecycle, admin-counter and RLS suites.
**Status:** Complete

**Interfaces:** Consume current `JobManager.complete_job(job_id, *, result=None, worker_id=None, lease_id=None, completion_token=None, expected_uuid=None, enforce=None) -> bool`. Test support exports `record_completion_observers(manager, monkeypatch, calls: list[str]) -> None` and `record_completion_boundaries(manager, monkeypatch, calls: list[str], *, on_commit: Callable[[], None] | None = None, fail_at: str | None = None) -> None`; proxies delegate real execute/fetch/context/commit/rollback/close rather than faking successful SQL.

- [x] Add the support/module pair. `completion_manager` is a module-local fixture parametrized with SQLite and `pytest.param("postgres", marks=pytest.mark.pg_jobs)`. SQLite uses `tmp_path`; PostgreSQL obtains the existing `jobs_pg_dsn` through `request.getfixturevalue`. New paired modules opt in with `USE_SHARED_JOBS_POSTGRES = True` and `pytest_plugins = ["tldw_Server_API.tests._plugins.authnz_isolated_fixtures"]`, following the current failed-requeue tests. Do not modify global conftests or migrate historical modules.
- [x] Record calls only after seeding a job; mock external observers/telemetry, not persistence. Wrap the first owning connection/cursor; do not accidentally wrap the pre-existing SLA policy lookup as a second completion transaction.
- [x] Write the following binding assertions using the probes. A real processing job has valid lease/token/UUID; counters/outbox are enabled, SLA policy returns no breach, test diagnostics are disabled for these order assertions:

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

- [x] Add parameterized tests with these exact preservation outcomes:

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

- [x] Run `RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py`. Expect all paired cases pass and no substantive PG skips. Characterization is intentionally green on unchanged production; do not manufacture a production defect to obtain red. Prove selected assertions are live with a scoped test-only adversarial probe (wrong binding/order raises an assertion), and keep production unchanged.
- [x] Run existing shared-helper matrix tests before relocation: `test_jobs_lifecycle_event_atomicity.py`, `test_jobs_lifecycle_hardening_regressions.py`, `test_jobs_dependency_acquire_counter_regressions.py`, `test_jobs_admin_retry_transaction_boundaries.py`. Add only missing cases for each current helper caller/branch. Keep ordinary discovered defects in separate tasks, not this PR.
- [x] Commit characterization and TASK-13875/plan evidence with normal hooks: `test(jobs): characterize completion extraction boundaries (TASK-13875)`.

Stage 1 implementation evidence (2026-10-20; controller independent review pending):
focused command above: **138 passed, 3648 warnings**, 0 skips (164.39s);
the four affected helper modules: **242 passed, 581 warnings**, 0 skips (57.44s),
versus the untouched **232 passed** baseline. Combined with existing row-identity,
idempotency and real RLS suites: **418 passed, 4300 warnings**, 0 skips (208.86s).
Scoped late-binding/wrong-order adversarial probes: **4 passed, 134 deselected**;
each wrong dispatcher triggers the expected order assertion on both real backends.
Ruff, scoped Black, compilation and whitespace pass; test Bandit with B101 excluded
has **0 new findings**, 14 unchanged historical helper findings, 0 in new files.
Production/provisioners/conftests remain unchanged. Exact commands, reused coverage
map, self-review and pre-existing TASK-13875 identifier collision are recorded in
`.superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-1-report.md`.

Task 1 review fix round 1 (I1, TASK-13878; 2026-10-20): removed all three new
Bandit B106 suppressions using equivalent marker values/argument helpers; original
implementation commit `99f1d0cba1` and production behavior are unchanged. The two
affected paired test functions pass **20 cases, 299 warnings, 0 skips** (21.74s).
Ruff, scoped Black, compilation and whitespace pass. Fresh test Bandit compared
with execution BASE `be07b6618217743938ea395bc756cf15fb0426aa` reports identical
14 semantic findings, 0 errors, **skipped_tests 0/0 and nosec 0/0**: no new findings
or suppressions. This corrects the initial evidence, whose 3 targeted suppressions
did not satisfy the separate no-new-suppressions gate. Exact commands/results and
self-review are appended to the Task 1 report. Controller tracking repair to
TASK-13878 is retained; archived Jobs and unrelated Moderation records are not
edited. Controller owns review acceptance and the ledger.

### Task 1 Final-Review Correction F1 (2026-10-20)

Both existing early-exit tests now retain valid mappings and add genuine object-payload TypeError and circular-mapping ValueError variants. The raw UUID `None`/empty/noncanonical and matching/stale matrix is unchanged. The two tests grow from 24 to 72 paired cases (36 per backend); the full module grows from 138 to 186 (93 per backend). Original real `json.dumps` first proves each poison value would fail final encoding; facade assertions require the exact existing boolean, one size-probe call and one genuine error only, no transform/final encoding/duration or gauge/SLA preparation/observer dispatch, and unchanged independent-reader job/counter/outbox state on real SQLite and fixture-owned PG. This is intentionally green on untouched production, not a manufactured production defect or runtime red/green claim. First affected nodes plus all helpers: 102 passed, 2323 warnings in 99.58s, zero skips. Standalone complete characterization: 186 passed, 4922 warnings in 200.98s, zero skips. Evidence uses only new `/tmp/task13878_finalfix_*` logs/XML; prior Task1/Task4 evidence is retained. Exact commands/environment and test-source hashes: `/tmp/task13878_finalfix_runs.json`. Full combined non-PG/required-PG requalification is recorded under Stage 4 when complete.

## Task 2: Add Completion-Only Contracts

### Stage 2

**Goal:** Define data-only boundaries without changing production routing or accepted input behavior.
**Success Criteria:** Direct tests prove shallow identity preservation, lazy callbacks, legal internal result states, and no driver/Jobs dependencies.
**Tests:** `test_jobs_completion_contracts.py`.
**Status:** Complete

Task 2 execution starts at `21f22a88ac900e8121621f2e087733b5942d01d7` in the approved worktree. TASK-13878 is verified active; only new contracts/direct tests and this stage's evidence are owned. Governing ADR-058/ADR-078 are unchanged. Detailed commands and red/green evidence: `.superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-2-report.md`.

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

- [x] Write direct tests before the module exists. `test_command_is_shallow_and_inert` asserts `command.result is original` and unchanged ID/UUID/token identity with objects whose `__int__`, `__str__`, `__deepcopy__` and JSON conversion fail if invoked. Assert assignment raises `FrozenInstanceError` while nested data is not recursively frozen.
- [x] Write parameterized outcome/invariant tests and Hypothesis tests: `assert replay.public_success is True`; `assert no_transition.public_success is False`; `assert applied.locked_row is row`; all legal metric-prefix lengths accepted, conflicting SLA forms/duplicate bindings/payload on replay rejected. Constructors must not call any recording policy stub. Test raw `None`, empty/noncanonical UUIDs, naive/aware/raw timestamp fields without normalization.
- [x] Run `RUN_JOBS=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_contracts.py`; expect missing-module/import failure before implementation, not an unrelated fixture failure.
- [x] Implement the specified contracts only, with module/class/function docstrings and type hints; no backend or manager imports. Run the same tests to green and rerun Stage 1. Add a cold-import check that importing this contracts module loads neither psycopg nor Jobs modules.
- [x] Commit contracts/tests/plan/task notes: `refactor(jobs): add shallow completion contracts (TASK-13878)`.

Task 2 qualification (2026-10-20): expected missing-module RED (exit 2; 12 warnings), then 93 direct tests GREEN. Combined `RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_contracts.py tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py` passed 231 cases (3834 warnings, 0 skips), including all 138 unchanged paired characterization cases using the existing shared PostgreSQL fixture. Final direct-source rerun after strengthening the JSON tripwire: 93 passed, 198 warnings, 0 skips; coverage rerun: 99% (142/144 statements). Isolated `python -I` import and spawned-interpreter test load no psycopg/psycopg2/Jobs modules. Touched Ruff, Black, py_compile, whitespace and both Bandit scans pass; production has no exclusions, tests exclude only B101, with 0 findings/errors/nosec/skipped_tests in both scans. No production caller/routing, manager, characterization, fixture, unrelated task or controller-ledger changes. Broad repository fail-fast run cannot collect unchanged `tests/Workspaces/test_workspaces_api.py` because `httpx2` is unavailable (12 collection skips, 9 warnings, 1 error); no dependency installation or out-of-scope fix. Whole foundation remains In Progress and controller owns review acceptance. Full exact commands, self-review and logs are linked in the Task 2 report above.

## Task 3: Relocate Shared Helpers Mechanically

### Stage 3

**Goal:** Place existing helper SQL/commit/exception ownership in DB_Management without changing callers.
**Success Criteria:** Moved definitions and complete_job are structurally equivalent; aliases and optional-driver behavior remain compatible; every helper caller is verified.
**Tests:** `test_jobs_completion_helper_boundaries.py`, Stage 1, and the shared-helper matrix.
**Status:** Complete

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

- [x] Add failing import/identity tests before moving definitions: manager's three helper names and two tuple objects must be `is` their new module exports. Import failure for the not-yet-existing modules is the expected red.
- [x] Add direct event-helper tests on real SQLite/PG transactions: attribution/attrs unchanged; owning rollback removes the row; helper never calls commit/rollback/close. Test missing-counter reconstruction with all four buckets and malformed count/error propagation. Do not snapshot all facade SQL.
- [x] Add commit-guard tests using narrow connection doubles for normal/INERROR/commit-failure/rollback-failure, then verify real INERROR behavior with fixture-owned PostgreSQL. Assert `rollback` then RuntimeError for aborted state, original rollback error precedence, and no observer invocation.
- [x] Add isolated-process driver-absence tests: block psycopg imports without uninstalling anything, import the exception module/SQLite path, assert `_PG_ERRORS == ()`, tuple identity and class order match baseline, and no new `TransactionStatus` import occurs until the PG guard is invoked. Test driver-present import too; do not count absence tests as PostgreSQL execution.
- [x] Move one responsibility at a time and import it back into manager under its exact private name. Keep every caller untouched, including completion/SLA, `fail_job` terminal/retry/quarantine, `reschedule_jobs`, and `retry_now_jobs`. No new observer preparer or production completion-contract import is needed.
- [x] After each move run the direct helper test plus Stage 1 and affected matrix tests. Preserve `_insert_lifecycle_event`'s `json.dumps(attrs or {})`, raw values, counter SQL, INERROR guard and exact exception-class identity.
- [x] Before committing, compare `ast.dump(..., include_attributes=False)` for each moved function against execution-baseline `git show` and for `JobManager.complete_job` against the unchanged baseline. Inspect SQL strings/parameter order and imports manually; store the comparison result in task notes, not a brittle permanent facade source snapshot. Changed AST bodies require explanation/review, not automatic acceptance.
- [x] Commit mechanical moves and evidence: `refactor(jobs): relocate lifecycle persistence helpers (TASK-13878)`.

Task 3 qualification (2026-10-20): missing-module alias RED produced 3 expected failures before production moves. Persistence-only direct run: 19 passed; persistence Stage 1/matrix checkpoint: 380 passed. Commit-guard checkpoint: 405 passed, 5 policy/import cases deselected. Final source: `RUN_JOBS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q tldw_Server_API/tests/Jobs/test_jobs_completion_helper_boundaries.py tldw_Server_API/tests/Jobs/test_jobs_completion_characterization.py tldw_Server_API/tests/Jobs/test_jobs_lifecycle_event_atomicity.py tldw_Server_API/tests/Jobs/test_jobs_lifecycle_hardening_regressions.py tldw_Server_API/tests/Jobs/test_jobs_dependency_acquire_counter_regressions.py tldw_Server_API/tests/Jobs/test_jobs_admin_retry_transaction_boundaries.py` -> 410 passed, 4669 inherited warnings in 212.94s, 0 skips. This includes all 30 new direct cases, 138 unchanged paired characterization cases and 242 unchanged helper-matrix cases; eight direct native-PG transaction cases use the existing shared fixture, while four cold import subprocess cases are not PG execution. A missing required owner argument in the new subprocess test fixture was corrected without production changes; targeted import rerun passed 4 cases. Full AST equality against both be07b66182 and dc853a5dcb covers all three moved definitions/signatures/SQL, exact optional-driver construction and all 20 remaining manager function/class nodes, including complete_job and every existing caller. Manual SQL/parameter/import review is clean. Manager imports policy after the original tracing boundary and conditionally reexports psycopg; TransactionStatus stays function-local. Touched Ruff, Black on new files only, compilation and whitespace pass. Fresh supplied exact-be07 Bandit source/reports compare semantically: production 1 inherited B608, tests 14 inherited findings (B101 only excluded), 0 errors/new findings/new suppressions; skipped_tests remains 91/4 and nosec remains 0/0, with 0 findings in new files. No contract/routing/backend/caller/fixture/existing-test changes, extra races, installs, nested agents or broad rerun. ADR-058/078 unchanged. Detailed commands, red/green, correction, source/security comparisons and self-review: `.superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-3-report.md`. Foundation remains In Progress; Task 4 and controller review are separate.

### Task 3 Final-Review Correction F2 (2026-10-20)

Added only per-test classification marks in the helper boundary module: alias identities (3), narrow counter-error doubles (4), commit-guard doubles (4) and cold imports (4) are unit (15); native event transactions (12), counter reconstruction (2) and aborted-PG guard (1) are integration (15). Existing eight `pg_jobs` cases remain integration; no module-level driver skip or fixture change. Fresh supported-plugin collection selects all 30 for `unit or integration`, 15 for each class, eight for `pg_jobs` and zero for `unit and integration` (expected collection exit 5 only for the empty intersection). Parsed collection sets prove disjointness, complete union and PG inclusion. All 30 execute within the first affected 102-case run above with zero skips. Fresh touched-scope Ruff/Black/compile and exact helper/manager/optional-driver AST equivalence pass. Bandit semantic comparison to verified be07 source retains 1 production and 14 test findings, zero errors/new findings/new suppressions; tests exclude B101 only, production excludes nothing; skipped_tests 91/0 and nosec 0/0 are unchanged. New evidence: `/tmp/task13878_finalfix_quality.json`, `/tmp/task13878_finalfix_bandit_compare.log`, marker logs and the final-fix SDD report. Production remains byte-identical to frozen `0f2b649cb8688abd7132d8fad7ec7dd293baaf9b`.

## Task 4: Qualify and Prepare the Foundation PR

### Stage 4

**Goal:** Prove the foundation is independently reviewable/revertible without routing completion.
**Success Criteria:** All applicable gates pass on the candidate head; scoped source/SQL/dependency review finds no behavior change; any unrelated failure has an evidence-backed disposition, not a weakened test.
**Tests:** Full non-PG Jobs, required real PG completion/lifecycle/admin/parity/RLS, opt-in outbox/SSE, touched-scope quality/security, Docs, and hosted gates.
**Status:** In Progress (local qualification and independent reviews complete; separately authorized integration, hosted gates and human Change summary pending)

**Interfaces:** Consume the modules/tests from Stages 1-3. Produce verification artifacts, an accurate task/plan record and a foundation-only PR when separately authorized; do not produce backend operation implementations.

All commands run from this worktree after `source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate`. Use a fixture-supported test cluster; no manual database provisioner or shared production database. Do not modify shared venv dependencies to force a pass.

- [x] Run new direct/characterization tests on their own and in the combined Jobs suite so plugin/import-order regressions cannot hide behind an earlier module's registration.
- [x] Run full non-PG: `RUN_JOBS=1 python -m pytest -q -m "not pg_jobs" tldw_Server_API/tests/Jobs --junitxml=/tmp/task13878_sqlite.xml`. Record existing skips explicitly; do not interpret a skipped relevant case as success. Initial autoload invocation failed in an unchanged subprocess registration probe; final complete selection uses the existing Jobs CI environment `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and explicit `-p pytest_cov -p pytest_asyncio.plugin`, with `/tmp/task13878_sqlite_ci.xml`. See fresh qualification evidence below; no tests or assertions were removed.
- [x] Run required PG matrix with `RUN_JOBS=1 RUN_PG_JOBS_TESTS=1 TLDW_TEST_POSTGRES_REQUIRED=1 python -m pytest -q -m pg_jobs` and these complete modules under `tldw_Server_API/tests/Jobs/`:

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

Append `--junitxml=/tmp/task13878_postgres.xml`. Expected: zero failures and zero substantive skips, not a fixed stale count. The paired reschedule/immediate-retry modules cover missing counters; completion-only evidence is insufficient. Also run new contracts and non-PG helper tests; a `pg_jobs` selection does not select their unit cases.

- [x] Run relevant opt-in SSE cases locally with `CI` unset and `TLDW_TEST_NO_SSE=0`: `test_jobs_events_sse_sqlite.py`, `test_jobs_sse_smoke.py`, and the enabled PG outbox module. Record actual event/heartbeat results and existing environment skips; if a relevant gate cannot execute, resolve qualification before treating the PR as ready. Do not remove existing guards or increase latency budgets to obtain green.
- [x] Run `python -m ruff check` on all four new production modules, three new test modules, test support, manager and any narrowly modified existing tests; run `python -m black --check` on new modules/tests only (no manager-wide formatting), `python -m compileall -q` on touched Python files, and `git diff --check`. Use exact touched paths from `git diff --name-only <execution-base>...HEAD`, never include unrelated files.
- [x] Before production moves, retain baseline Bandit JSON for manager and touched existing tests. After moves run project-venv `python -m bandit -r <touched-production-paths> -f json -o /tmp/task13878_bandit_prod.json` and a separate touched-test scan with `-s B101 -o /tmp/task13878_bandit_tests.json`. Compare semantic findings (test ID, code, severity/confidence), mapping mechanically relocated helper findings to their old definitions rather than treating line-number churn as new defects. No new findings/errors or new suppressions; record inherited warnings and exact comparison, not just the scan exit code.
- [x] Run `python -m pytest -q tldw_Server_API/tests/Docs` with the repository-supported MkDocs toolchain. If canonical ADR/docs metadata changed, run `bash Helper_Scripts/refresh_docs_published.sh`, require a deterministic second refresh and canonical/generated parity, then rerun strict Docs checks. Never publish this private plan into the public tree.
- [x] Perform local whole-branch source/SQL/coverage self-review: no new backend completion modules/entry points, unchanged complete_job AST and SQL, mechanically identical helper bodies, no new driver import/DB-to-Jobs dependencies, all policy/preparation/cleanup fault paths accounted for, and spec-to-test coverage including shared callers. No foundation-caused regression identified. Inherited ordinary defects require separate controller-owned follow-up; no other-task edits are authorized here. This does not replace the controller's independent whole-branch review.
- [x] Record fresh local verification and remaining gates in TASK-13878. Local qualification is complete; foundation and Stage 4 remain In Progress for independent review and conditional integration requirements.
- [ ] With separate PR authorization, use explicit `--repo rmusser01/tldw_server`, target current `dev`, attach the created PR to this chat, and request a new human-written Change summary. Do not reuse PR #3092's summary or treat written-spec/plan approval as merge approval. Obey current repository merge-queue and exact-head hosted/Jobs requirements; no bypass/admin merge or unrelated CI cancellation. Hosted gates: NOT RUN; human Change summary: NOT PROVIDED; no merge-ready claim.
- [x] Commit final local plan/task evidence with normal hooks and explicit metadata paths only, including the controller-created TASK-13879 unchanged. Reverting the foundation at its own merge point restores helper definitions/imports and removes unused contracts; after routing depends on it, revert routing first. Routing gets a new task/plan only after this foundation merges.

### Task 4 Local Qualification Evidence (2026-10-20)

Frozen source HEAD: `0f2b649cb8688abd7132d8fad7ec7dd293baaf9b`; exact execution/security baseline: `be07b6618217743938ea395bc756cf15fb0426aa`. Activate `/Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate` before every Python/test/Bandit command. No source/test/config/CI/dependency changes. No installs, new provisioner, nested agents, whole-repository collection rerun, push, PR, merge or rebase. Approved plan and controller SDD workspace remain intact.

| Fresh Selection | Passed | Skipped | Deselected | Warnings | Pytest Time |
| --- | ---: | ---: | ---: | ---: | --- |
| Isolated contracts module | 93 | 0 | 0 | 198 | 1.80s |
| Isolated direct helper module | 30 | 0 | 0 | 413 | 27.02s |
| Isolated paired characterization module | 138 | 0 | 0 | 3652 | 174.42s |
| Full non-PG Jobs, supported CI plugin invocation | 1536 | 2 | 698 | 7559 | 87.04s |
| Complete specified required PG matrix | 231 | 0 | 231 | 4306 | 222.61s |
| Explicit SQLite SSE/smoke plus enabled PG outbox | 5 | 0 | 0 | 310 | 13.58s |
| Full Docs, final metadata scope, supported CI toolchain/plugin invocation | 212 | 0 | 0 | 441 | 27.40s |

Jobs commands set `RUN_JOBS=1 RUN_PG_JOBS_TESTS=1 TLDW_TEST_POSTGRES_REQUIRED=1`, unset `CI`, set `TLDW_TEST_NO_SSE=0`, leave `SSE_SMOKE_MAX_MS` unset, add `-q -ra` and separate `--junitxml=/tmp/task13878_<selection>.xml`. Isolated modules run in separate processes with default autoload. Full non-PG, PG and SSE use existing `.github/workflows/jobs-suite.yml` settings `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and explicit `-p pytest_cov -p pytest_asyncio.plugin`. All DB-heavy selections run serially. Exact expanded commands/exits/environment: `/tmp/task13878_runs.json`; XML/log selections: `contracts`, `helpers`, `characterization`, `sqlite_ci`, `postgres`, `sse`; full parsed counts, per-module coverage and inspected skips: `/tmp/task13878_results.json` and `/tmp/task13878_summary.log`.

Initial default-autoload full non-PG command is preserved, not hidden: `/tmp/task13878_sqlite.{log,xml}` records 1 failed, 1535 passed, 2 skipped, 698 deselected, 7659 warnings in 76.47s. Existing `test_bridge_resolution_preserves_jobs_loop_and_isolation` explicitly loads asyncio in a child pytest; setuptools autoload registers the same plugin under `asyncio` and fails before collection. Registration test, probe helper, root/AuthNZ conftests and both bridge plugins are byte-unchanged from be07. Existing Jobs CI disables autoload and explicitly loads the plugin. Diagnostic complete registration-module run: 4 passed, 23 warnings in 16.40s, no skips (`/tmp/task13878_fixture_ci.{log,xml}`); final full rerun above passes unchanged assertions. Controller accepted this invocation disposition; no source fix, new guard, skip or latency-budget adjustment. Three attempts total (original, diagnostic, full supported invocation), no further retry needed.

Both full non-PG runs skip only unchanged `test_jobs_manager.py::test_acquire_decrypts_payload` (line 358) and `::test_rotate_encryption_keys_respects_filters_sqlite` (line 1846), reason `Crypto backend unavailable; skipping encryption test`. `/tmp/task13878_skip_diagnosis.log` verifies crypto is available but both existing fixture keys decode to 33 rather than required 32 bytes; the module is byte-identical to be07. These are not completion/PG/SSE success evidence. Controller created and validated [TASK-13879](../../../backlog/tasks/task-13879%20-%20Repair-inherited-Jobs-encryption-fixture-skips.md) for separate correction; preserve that record body and do not correct fixtures/production in this foundation. Known unrelated missing-httpx2 whole-repository collection blocker remains from Task 2, not rerun.

Required PG JUnit has zero failures/errors/skips in all 12 complete selected modules: characterization 69, helpers 8, identity 9, completion idempotency 1, finalize idempotency 2, event atomicity 24, hardening 37, dependency/reschedule counters 35, admin retry boundaries 21, RLS 12, enabled outbox 2, parity 11. Full non-PG includes all 93 contract, 22 helper and 69 paired SQLite cases in combination. Separate cold import probes pass with present/absent driver and both import orders. Explicit SSE asserts initial SQLite ping plus seeded job.created with parsed attrs, PG job.created plus exactly one durable job.completed, filter/after-id behavior and unchanged 0.5-second list responsiveness assertion. All five run, zero skips.

Fresh exact touched paths from `git diff --name-only be07b6618217743938ea395bc756cf15fb0426aa...HEAD`: Ruff on 10 Python paths, Black check only 8 new paths, compileall on 10, branch/working whitespace all pass. `/tmp/task13878_quality.json` contains exact commands/exits. Fresh AST/SQL comparison matches all three relocated functions/signatures/bodies/SQL, exact optional-driver policy and all 20 remaining manager function/class nodes including complete_job and shared callers. Import AST and runtime cold probes confirm no new DB-to-Jobs dependency, eager driver/TransactionStatus import, production contract routing or backend operation. Manual SQL/parameter/exception-policy review and retained named coverage map cover every shared caller and policy/preparation/cleanup fault boundary. Independent controller whole-branch review remains separate.

Fresh touched Bandit production/test reports `/tmp/task13878_bandit_{prod,tests}.json` each exit 1 only for inherited findings. No production exclusions; tests exclude only B101. Every supplied archived scanned source byte is verified against exact be07; semantic Counters compare ID, issue, severity, confidence and actual source, never shifted lines. Production: 1 inherited B608 MEDIUM/LOW, 0 errors/new findings/new suppressions, skipped_tests 91/91, nosec 0/0. Tests: 14 inherited (10 B106 LOW/MEDIUM, 2 B105 LOW/MEDIUM, 2 B608 MEDIUM/MEDIUM), 0 errors/new findings/new suppressions, skipped_tests 0/0, nosec 0/0. Supplied wider test baseline's four suppressions are in unchanged files excluded from this exact touched scope. All new files have zero findings/suppressions; inherited manager B608 comment warnings retained. Comparison: `/tmp/task13878_bandit_compare.log`.

Shared MkDocs 1.6.1 / Material 9.7.7 / git-revision-date-localized 1.6.0 satisfy onboarding-docs CI ranges; the empty `/tmp/task13215-mkdocs` was not used. Two actual canonical refreshes match all 487 SHA-256 file hashes with zero generated changes. All three CI onboarding boundary/drift scripts pass. Initial full Docs: 212 passed/441 warnings/28.91s; initial strict build 9.46s. Final metadata-scope rerun above and `python -m mkdocs build --strict -f Docs/mkdocs.yml --site-dir /tmp/task13878_mkdocs_final_site` pass (build 9.86s); exact final commands/exits in `/tmp/task13878_final_runs.json`, logs/XML `/tmp/task13878_docs_final.{log,xml}` and `/tmp/task13878_mkdocs_final.log`. Upstream Material banner and INFO navigation/link notices retained, no MkDocs WARNING/ERROR records. Final full AST/source-freeze/whitespace/canonical task-format checks also pass. No private-plan publication or generated Docs edits. Detailed report: `.superpowers/sdd/2026-10-20-jobs-completion-foundation-implementation-plan/task-4-report.md`; local artifact/source hash manifest: `/tmp/task13878_manifest.json`.

**Local qualification: COMPLETE.** Foundation/Stage 4: **In Progress**. Remaining: controller independent whole-branch review/finishing handoff, separately authorized PR, a new human-written Change summary, exact-final-head hosted/Jobs/required gates and current merge-queue requirements. Hosted gates **NOT RUN**, human summary **NOT PROVIDED**; no integration or merge-ready approval is inferred. Preserve this approved plan for handoff.

### Task 4 Final-Fix Requalification (2026-10-20)

This is the F1/F2-only follow-up to the delivered whole-branch review, not another broad review or integration authorization. Dispatch BASE `64bef52f5ce3f6544b925362d3fea9c6946bee19`; production remains frozen at `0f2b649cb8688abd7132d8fad7ec7dd293baaf9b`. Exactly two test modules, this plan and TASK-13878 notes differ from dispatch BASE; no production/fixture/schema/config/CI, other task/archive, controller ledger or TASK-13879 edits. All prior evidence remains intact; new logs/XML/JSON/build output use `/tmp/task13878_finalfix_*` only.

| Final Test Source Selection | Passed | Skipped | Deselected | Warnings | Pytest Time |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact two affected characterization nodes plus complete helpers, first run | 102 | 0 | 0 | 2323 | 99.58s |
| Complete standalone characterization | 186 | 0 | 0 | 4922 | 200.98s |
| Full non-PG Jobs | 1560 | 2 | 722 | 7607 | 85.59s |
| Complete specified twelve-module required PG matrix | 255 | 0 | 255 | 5530 | 263.37s |
| Explicit SQLite SSE/smoke and enabled PG outbox | 5 | 0 | 0 | 310 | 13.97s |
| Full Docs | 212 | 0 | 0 | 441 | 23.86s |

All execution suites exit 0, zero failures/errors. Jobs invocations use the existing CI-supported `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -p pytest_cov -p pytest_asyncio.plugin`, `RUN_JOBS=1 RUN_PG_JOBS_TESTS=1 TLDW_TEST_POSTGRES_REQUIRED=1`, `CI` unset, `TLDW_TEST_NO_SSE=0`, `SSE_SMOKE_MAX_MS` unset; new runner also disables bytecode/cache writes. DB-heavy selections are serial. Exact expanded selections/commands/exits/environment/test SHA-256 hashes: `/tmp/task13878_finalfix_runs.json`; parsed JUnit counts, per-module coverage, actual skips and marker sets: `/tmp/task13878_finalfix_results.json`. Both test files remain hash-identical through every run. No default-autoload or whole-repository httpx2 failure is repeated or hidden.

Required PG per-module passes: characterization 93, helpers 8, row identity 9, completion idempotency 1, finalizer idempotency 2, lifecycle events 24, hardening 37, dependency/reschedule counters 35, admin retry 21, RLS 12, enabled outbox 2, parity 11. Full non-PG contains all 93 contracts, 22 non-PG helpers and 93 SQLite characterizations. Both actual skips are exactly unchanged `test_acquire_decrypts_payload` and `test_rotate_encryption_keys_respects_filters_sqlite`, reason `Crypto backend unavailable; skipping encryption test`; prior diagnosis and separate TASK-13879 disposition are retained without modifying that task. PG/SSE have zero skips. Explicit SSE/outbox exercises the existing ping/event, parsed attrs, single completion event, filter/after-id and unmodified responsiveness assertions, without duration overrides.

Marker collection: all 30 selected by `unit or integration`, 15 unit, 15 integration, disjoint, eight `pg_jobs` all integration. Empty `unit and integration` correctly selects zero (collection exit 5, not an execution failure). F1 adds 48 cases without dropping the 24 valid/raw-UUID early-exit cases, intentionally green on frozen production; no manufactured runtime defect. See Task1/Task3 correction evidence above.

Fresh touched-scope quality/security/equivalence: Ruff 10 paths, Black 8 new paths, compile 10 paths, working/branch whitespace pass. Three relocated full helper ASTs/SQL/signatures, exact optional-driver policy and all 20 remaining manager function/class nodes including complete_job/callers match be07 and dc853a5dcb. Exact production diff against frozen 0f2b is empty. No DB-to-Jobs dependency/eager driver import/routing changes. Bandit semantic findings remain 1 production/14 test inherited, zero new findings/errors/suppressions; production excludes nothing, tests B101 only; skipped_tests 91/0, nosec 0/0 unchanged. Commands/results: `/tmp/task13878_finalfix_quality.json`, new Bandit JSON/comparison logs and final equivalence/source-frozen logs.

All three Docs boundary/drift scripts pass; full Docs above and strict `python -B -m mkdocs build --strict -f Docs/mkdocs.yml --site-dir /tmp/task13878_finalfix_mkdocs_site` pass, build 8.65s. Canonical/published ADR/index bytes match and all generated Docs are unchanged, so no new refresh is required; prior deterministic-refresh evidence remains valid and preserved. Same upstream Material banner/INFO link notices, no MkDocs WARNING/ERROR records. Canonical TASK-13878 format check passes. Exact Docs commands/exits: `/tmp/task13878_finalfix_docs_runs.json`.

**F1/F2 local qualification and scoped re-review: COMPLETE.** Foundation/Stage4/TASK-13878 remain **In Progress** only for separately authorized integration gates. Hosted/exact-head/required/Jobs/merge-queue gates **NOT RUN/UNKNOWN**, human Change summary **NOT PROVIDED**. No hosted or merge-ready claim. No installs, nested implementer agents, rebase, push, PR or merge.

## Controller Handoff

Independent task reviews approved all four local stages. Whole-branch review of current dev `e521605198dd311b94d2c63416bf07db3e5e7a0a` through `64bef52f5ce3f6544b925362d3fea9c6946bee19` found only F1 (failed-preflight early-exit coverage) and F2 (helper markers). Both were validated, corrected in `d98b9ae0e90d63f8ac7c3414bff3cdf87ba3d3dd`, and accepted by a fresh scoped re-review with no residual findings or new breakage. Production remains identical to qualified source `0f2b649cb8688abd7132d8fad7ec7dd293baaf9b`; backend routing was not introduced.

Final local evidence is the final-fix table above: full non-PG 1560 passed with the two inherited skips tracked by TASK-13879; required real PG 255 passed with no skips; SSE/outbox 5 passed; Docs 212 passed; static, strict docs and semantic security-delta checks passed. The initial default-autoload failure, supported CI-style invocation, inherited warnings/findings and unavailable whole-repository httpx2 collection remain disclosed, not silently converted into success.

Controller accepts all eleven explicitly declined review boundaries: excluded future routing/lifecycle/caller/history work, characterized ordinary behavior/SLA lifetime, separately tracked fixture correction, unchanged plugin/dependency environment, present-driver-specific test scope, whole-repository/deployment/security audit limits, and genuinely pending hosted/human/PR gates. No new stop-condition defect was found. Preserve this approved plan while conditional integration remains unfinished; do not remove another agent's plan.

The plan-specific ignored SDD evidence is archived at `/tmp/task13878-sdd-evidence-d98b9ae0.tar.gz` before removing only that completed execution scratch directory. Durable decisions, exact commands/results and separate follow-up remain in this plan and TASK-13878/TASK-13879. Branch/worktree stay intact for the human's integration choice; no push, PR creation, local/shared merge or backend routing is inferred.

## Plan Review and Execution Gate

This document covers PR 1 only. Human review and subagent-driven execution selection were received before Stage 1 began. No PR/push/merge or backend-routing authorization is inferred. If no subagent tools are available, use native `superpowers:executing-plans`; do not create separate user chats as substitute agents or claim independent review that did not occur.

Self-review checklist: map every PR-1 spec requirement to the four stages; verify contract names/signatures agree across stages; verify all five review-focus conditions have named tests; retain partial optional effects and pre-commit binding without routing new code; verify current helper caller inventory and real PG fixtures; recheck new identifiers before publication.
