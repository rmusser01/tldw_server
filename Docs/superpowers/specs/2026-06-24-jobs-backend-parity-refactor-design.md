# Jobs Backend Parity Refactor Design

Date: 2026-06-24
Topic: Jobs backend parity refactor
Status: Approved design

## Objective

Refactor the Jobs core for long-term stability and pragmatic maintainability by making SQLite and Postgres behavior harder to diverge.

The first priority is not visible feature change or broad code movement. The first priority is a safety net: parity and API-compatibility tests that preserve existing REST response behavior while exposing backend-specific drift in lifecycle, admission, locking, durable events, and status mapping.

After that safety net exists, extraction should proceed one operation family at a time behind the existing `JobManager` facade.

## Design Goals

- Preserve existing REST/API response fields and status mappings as the primary compatibility boundary.
- Preserve domain-specific Jobs status and identifier mappings, including mappings documented in `Docs/Product/Job_System_Unification_Mapping_Matrix.md`.
- Keep `JobManager` as the public facade for REST endpoints, workers, and domain modules.
- Reduce SQLite/Postgres divergence through shared operation contracts and backend-specific implementations.
- Make transaction boundaries, side effects, and no-op outcomes explicit.
- Avoid schema churn in the first implementation slice unless a parity test proves it is required.
- Prefer incremental, reviewable PRs over a large rewrite.

## Non-Goals

- Do not replace all `JobManager` internals in one PR.
- Do not require existing REST endpoints, workers, or domain modules to adopt a new public API in the first slice.
- Do not use broad snapshot tests as the primary public compatibility mechanism.
- Do not make real Postgres mandatory for every local Jobs change; use it narrowly for semantics fake tests cannot prove.
- Do not move business-specific job payload ownership into the Jobs core.

## Approaches Considered

### Recommended: Parity harness, then operation extraction

First build reusable parity scenarios for high-risk Jobs behavior, then extract backend-specific operation families behind `JobManager`.

Why this is preferred:

- establishes behavioral evidence before code movement
- keeps REST and worker callers stable
- lets extraction proceed in small PRs
- directly targets the most repeated failure mode: SQLite/Postgres drift

### Alternative: Repository interface first

Define a full `JobsRepository` contract and migrate manager methods onto SQLite/Postgres repositories.

Trade-offs:

- cleaner architecture on paper
- higher upfront churn
- more likely to break public behavior before parity coverage exists
- harder to review incrementally because many methods must route through the new abstraction at once

### Alternative: Code decomposition first

Split `manager.py` by responsibility before adding broad parity tests.

Trade-offs:

- faster visible reduction in file size
- risks creating smaller files with the same implicit contracts
- may preserve or hide existing backend divergence
- gives less confidence that public behavior stayed stable

## Architecture

`JobManager` remains the public facade. Existing callers continue to use methods such as `create_job`, `acquire_next_job`, `renew_job_lease`, `complete_job`, `fail_job`, `cancel_job`, `list_jobs`, and summary/admin helpers.

Internally, the refactor introduces:

- typed operation commands and results
- a small `JobsSettings` object for environment-derived Jobs behavior
- backend-specific operation implementations
- parity test helpers under tests, not production code

`JobsSettings` must not accidentally freeze behavior that is currently dynamic. Some API paths cache `JobManager` instances by backend key, and many tests intentionally change `JOBS_*` environment variables per test. The first implementation plan must define whether each setting is snapshotted at manager construction, read per operation, or refreshed through an explicit test/runtime hook.

Proposed production layout:

```text
tldw_Server_API/app/core/Jobs/
  manager.py
  settings.py
  operations/
    contracts.py
    sqlite/
      admission.py
      lifecycle.py
    postgres/
      admission.py
      lifecycle.py
```

Proposed test layout:

```text
tldw_Server_API/tests/Jobs/parity/
  scenarios.py
  test_sqlite_parity.py
  test_postgres_parity.py
```

The first operation families should be admission and lifecycle because they carry the highest backend-parity risk:

- admission: quota, fair-share, idempotency, queue controls, create-job transaction behavior
- lifecycle: acquire, renew, complete, fail, cancel, retry, terminal transitions, stale lease behavior

Later slices can extract dependencies, durable events/outbox, query/stat/admin operations, pruning/archive, and attachments.

### Existing direct SQL boundaries

Current admin endpoints still contain direct SQLite/Postgres SQL for some Jobs operations such as SLA breach scans, batch cancel, batch reschedule, and quarantined requeue. Other status, metrics, webhook, worker, and domain-service paths also query Jobs tables directly for read-side or operational behavior. That means the current system is not purely `JobManager`-facade-backed even though the desired end state keeps persistence behavior behind core Jobs boundaries.

The implementation plan must inventory direct Jobs SQL before extraction begins and classify each path as one of:

- state-changing SQL that should move behind `JobManager` or backend operation contracts,
- read-only/status SQL that may remain as an explicitly documented read model for now,
- service/worker operational SQL that needs parity coverage or a narrower core helper,
- migration/bootstrap SQL that remains outside runtime operation extraction.

Each non-migration path should then be either:

- covered by field-level API compatibility tests and explicitly deferred, or
- moved behind `JobManager`/operation boundaries in a dedicated extraction slice.

Do not claim backend parity is complete while state-changing endpoint-owned SQL still performs Jobs operations outside the shared operation contracts. For read-only direct SQL, parity claims must state whether the path is in scope, deferred as a read model, or covered by separate status/metrics compatibility tests.

### Domain status mapping boundary

Several public domain APIs intentionally map core Jobs fields into domain-specific response contracts. Examples already documented in `Docs/Product/Job_System_Unification_Mapping_Matrix.md` include:

- media embeddings list responses where `quarantined` maps to `failed` and job ids prefer Jobs UUIDs,
- chatbooks export/import responses where `queued` maps to `pending` and `processing` maps to `in_progress`,
- prompt-studio optimization responses where `quarantined` maps to `failed` and unknown statuses fall back to `queued`.

The first implementation plan must inventory these domain-facing mappings before extraction work starts. Backend parity tests should verify core Jobs behavior, but they are not enough to prove public compatibility for adapters that intentionally translate status, id, or error fields.

## Component Boundaries

`JobManager` owns:

- public argument validation
- REST-compatible response normalization
- public status and error mapping
- command construction
- orchestration across operations
- non-transactional post-commit side effects

Backend operation modules own:

- backend-specific SQL
- backend-specific locking and transaction details
- atomic database state transitions
- durable outbox writes that must commit or roll back with the state transition
- typed operation outcomes

Backend operation modules must not import `JobManager`. They should receive explicit dependencies such as connections, settings, clock helpers, prepared serialization/encryption helpers, and tracing context.

## Data Flow

### Create and admission

1. REST or domain code calls `JobManager.create_job(...)`.
2. `JobManager` validates public arguments and builds a normalized `CreateJobCommand`.
3. `JobManager` decides security-sensitive policy such as JSON truncation, secret handling, and encryption behavior.
4. The backend-specific admission operation owns the transaction:
   - admission locks
   - quota and fair-share checks
   - queue pause/drain checks
   - idempotency lookup
   - insert or existing-row return
   - durable outbox write when the outbox row is part of the persisted job fact
5. The operation returns an `AdmissionResult` that distinguishes inserted rows from existing idempotent rows.
6. `JobManager` maps the result to today’s public row shape and performs post-commit metrics, logging, audit, and SSE/event dispatch exactly once.

### Lifecycle transitions

1. Worker or admin code calls `acquire_next_job`, `renew_job_lease`, `complete_job`, `fail_job`, `cancel_job`, or batch variants.
2. `JobManager` validates public arguments and builds a typed lifecycle command.
3. The backend-specific lifecycle operation owns the backend transaction and locking semantics.
4. The operation returns a `LifecycleResult` with:
   - whether a transition happened
   - a reason code for no-op outcomes
   - current or final job row
   - timing and label facts
   - persisted durable event facts
   - retry or terminal metadata
5. `JobManager` applies public compatibility mapping and performs non-transactional post-commit side effects exactly once.

Batch lifecycle operations should preserve current per-item behavior in the first slice unless a parity test proves the current behavior is incorrect. If a later change moves batches to all-or-nothing semantics, that must be explicit, tested, and reflected in API compatibility notes.

## Error Handling

Operation outcomes should be typed so backend behavior can be tested without parsing SQL exceptions or boolean return values.

Initial outcome families:

- `TransitionApplied`: state changed successfully.
- `NoTransition`: valid request, but no state change.
- `AdmissionRejected`: valid command rejected by queue, quota, fair-share, idempotency, dependency, or policy state.
- `BackendConflict`: safe-to-retry lock contention, serialization failure, or concurrent transition race.
- `BackendSchemaError`: missing table, missing column, or detected migration/schema drift.
- `BackendError`: unexpected DB, serialization, or operation failure.

`NoTransition` should carry a reason code such as:

- `missing`
- `wrong_status`
- `stale_lease`
- `already_terminal`
- `idempotent_existing`
- `rls_filtered`

`JobManager` owns the internal-to-public mapping table. For example, RLS filtering may remain distinguishable internally while still mapping to a public not-found response where existing API compatibility requires that behavior.

Rules:

- Public input validation stays in `JobManager`; malformed input should not become `AdmissionRejected`.
- DB operation failures roll back the transaction and do not emit metrics, audit, SSE, or non-durable events.
- Durable outbox failure rolls back when the outbox row represents the same persisted state transition.
- Post-commit logging, metrics, audit dispatch, and SSE failures do not undo committed job state, but must be logged with job id, operation, backend, and reason.
- `BackendConflict` is reserved for narrowly retryable transaction conflicts; plain SQL errors are `BackendError` or `BackendSchemaError`.

## Side Effects

The design separates durable facts from post-commit effects.

Durable facts:

- job row updates
- dependency rows
- durable `job_events` outbox rows when enabled and tied to a state transition

These belong in the backend operation transaction.

Post-commit effects:

- in-process event notification
- metrics updates
- audit dispatch
- structured logging
- SSE fanout

These belong in `JobManager` after the operation returns successfully. A lifecycle transition should emit each applicable post-commit effect exactly once.

## Testing Strategy

The first implementation slice should be test-first and compatibility-focused.

The test design should reuse existing Jobs conventions rather than invent a parallel fixture layer. Real Postgres tests should use the existing Jobs test support, including `pytest.mark.pg_jobs`, `jobs_pg_dsn`, and the unified temporary Postgres fixture imported by `tldw_Server_API/tests/Jobs/conftest.py`. Local and CI commands that need Jobs-marked tests must set `RUN_JOBS=1`; otherwise the suite-level collection hook skips `jobs`, `pg_jobs`, and `pg_jobs_stress` tests.

The test design should also inventory existing paired SQLite/Postgres tests before adding new parity files. Where existing tests already cover the same behavior separately, prefer extracting a shared scenario/helper over duplicating another test with slightly different setup.

Core parity tests should stay independent of FastAPI startup. API compatibility tests are still required for public response behavior, but they should use the existing Jobs minimal-app conventions and stay as narrow as possible because the global test fixture resets the main app lifecycle around tests. If that fixture blocks execution in a local environment, the implementation notes must separate core parity results from API-harness limitations.

### Fast parity tests

Add reusable scenario definitions under `tldw_Server_API/tests/Jobs/parity/`. Fast parity should run against SQLite and fake or adapter-backed Postgres paths where that is enough to prove control flow and public compatibility.

Coverage should include:

- create/admission
- idempotent create returning existing rows
- quota and fair-share rejection
- acquire and lease assignment
- renew with correct and stale lease data
- complete, fail, cancel, and already-terminal no-ops
- dependency cycle prevention
- durable event outbox behavior
- side-effect cardinality facts

### Real Postgres subset

Use the existing Postgres fixture/marker for behavior fake tests cannot prove:

- advisory locks
- transaction isolation
- `RETURNING` behavior
- RLS and domain allowlist filtering
- migration/schema drift
- lock contention and serialization conflicts

Real Postgres should be required for touched backend-parity areas, not for every local Jobs-only edit.

### API compatibility tests

Use field-level contract tests rather than broad response snapshots. These tests should cover:

- required response keys
- status mappings
- error code and body shape
- no removal of known public fields
- list/status/admin endpoint behavior around missing, filtered, terminal, and queued jobs

The API compatibility inventory should seed from `Docs/Product/Job_System_Unification_Mapping_Matrix.md` and then add any currently active Jobs-backed endpoints not listed there. The first PR does not need to test every domain endpoint, but it must document the mapping inventory and cover the most exposed admin/list/detail path plus at least one domain adapter with non-identity status mapping before production extraction begins.

### Operation contract tests

Once operation modules exist, test typed operation outcomes directly without going through REST. This should catch backend drift before public mapping hides it.

### Migration and old-row tolerance

Where practical, include rows that resemble older deployments:

- missing newer optional fields
- legacy result or error payload shapes
- older status/progress combinations that current readers still need to tolerate

### Quality gates

- No public response shape changes without explicit compatibility tests.
- No schema changes without SQLite and Postgres migration tests.
- No operation extraction without parity coverage for that operation family.
- No new parity test family without checking whether an equivalent SQLite/Postgres pair already exists and can be consolidated.
- Bandit must run on touched Jobs implementation paths.
- If pytest harness setup blocks direct test execution, document that separately from direct core verification.

## Rollout Plan

1. Inventory existing SQLite/Postgres Jobs test pairs, direct Jobs SQL across endpoints/services/workers/domain helpers, and domain-specific Jobs status/id mappings.
2. Define a minimum first PR: shared parity helpers plus one create/acquire/complete path, one stale/no-op lifecycle path, one idempotent-create path, field-level API contract coverage for the most exposed list/detail endpoint, and one domain adapter with non-identity status mapping.
3. Add parity scenario helpers and first fast parity coverage for admission and lifecycle.
4. Add field-level REST compatibility tests for the public endpoints most tied to Jobs status/list/admin behavior.
5. Add the narrow real-Postgres parity subset for locking, RLS, and transaction semantics using the existing `pg_jobs`/`jobs_pg_dsn` conventions.
6. Introduce `JobsSettings` and operation contract dataclasses without moving production SQL yet, including explicit snapshot/refresh rules for env-derived behavior.
7. Extract admission operations behind `JobManager`.
8. Extract lifecycle operations behind `JobManager`.
9. Reassess before extracting dependency, event/outbox, query/stat, direct runtime SQL paths, and prune/archive paths.

Each extraction PR should leave `JobManager` callable by existing REST endpoints, workers, and domain modules.

## Risks And Mitigations

### Risk: Operation modules become smaller versions of `manager.py`

Mitigation: require typed commands/results, explicit transaction ownership, and direct operation contract tests before extracting each family.

### Risk: Fake Postgres tests miss real backend semantics

Mitigation: maintain a narrow real-Postgres parity suite for advisory locks, RLS, `RETURNING`, transaction isolation, and schema drift.

### Risk: Public compatibility breaks while backend parity improves

Mitigation: preserve `JobManager` as facade and add field-level REST compatibility tests before changing internals.

### Risk: Domain adapters regress while core Jobs parity passes

Mitigation: seed the public compatibility inventory from `Docs/Product/Job_System_Unification_Mapping_Matrix.md`, add missing active Jobs-backed endpoints, and cover representative non-identity status/id mappings before production extraction.

### Risk: Side effects duplicate or disappear

Mitigation: operations return facts; `JobManager` emits post-commit effects exactly once; tests assert durable outbox and post-commit cardinality where applicable.

### Risk: Schema cleanup distracts from safety

Mitigation: avoid schema changes in the first slice unless a parity test proves they are required. Any schema change must include SQLite and Postgres migration coverage.

### Risk: Settings caching changes runtime or test behavior

Mitigation: classify every env-derived Jobs setting as construction-time, operation-time, or explicitly refreshable before introducing `JobsSettings`. Add tests for any setting whose current behavior depends on per-test monkeypatching or API dependency cache keys.

### Risk: Direct runtime SQL bypasses operation contracts

Mitigation: inventory endpoint, service, worker, and domain-helper SQL before extraction. Track each state-changing path as deferred with compatibility tests or migrated into a later operation slice. Track each read-only path as an explicit read model or cover it with status/metrics compatibility tests.

### Risk: API compatibility tests inherit unrelated app-startup instability

Mitigation: keep core parity tests independent of FastAPI startup. Keep API compatibility tests minimal, Jobs-marked, and aligned with the existing Jobs test environment. Record `RUN_JOBS=1` and any local app-lifecycle fixture blocker in verification notes.

## Acceptance Criteria

- The first implementation plan starts with parity and API compatibility tests, not production extraction.
- The public REST/API response boundary remains stable unless a later approved spec explicitly changes it.
- Operation modules use typed commands/results and do not import `JobManager`.
- Durable outbox writes tied to state transitions remain transactional.
- Post-commit side effects are facade-owned and emitted exactly once.
- Real Postgres coverage is narrow but mandatory for backend semantics fake tests cannot prove.
- Direct runtime Jobs SQL is inventoried and classified as state-changing, read-only/status, service/worker operational, or migration/bootstrap SQL.
- State-changing direct runtime Jobs SQL is either covered as a compatibility boundary or scheduled for migration behind operation contracts.
- Domain-specific Jobs status/id mappings are inventoried and either covered by field-level contract tests or explicitly deferred with rationale.
- `JobsSettings` includes explicit snapshot/refresh semantics for env-derived behavior.
