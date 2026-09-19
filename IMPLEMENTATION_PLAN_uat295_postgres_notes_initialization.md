# UAT295: PostgreSQL Notes initialization during another user's read

Task: TASK13260.232. Branch starts from PR2969 merge1dfdd819b6 on latest dev. Scope is the recorded second-user Notes bootstrap failure; all other open findings remain separate.

## Stage 1: reproduce and attribute
**Goal**: Explain why a current shared schema still requires blocking DDL for a new user.
**Success Criteria**: A real official-fixture PostgreSQL test retains a legitimate first-user read transaction and reproduces second-user initialization failure; existing no-active-reader behavior remains a control.
**Tests**: New active-reader regression and existing Notes bootstrap lifecycle controls. Preserve failure output locally.
**Status**: Complete

## Stage 2: repair the verified cause
**Goal**: Avoid unnecessary blocking shared-schema work while preserving bootstrap, migrations and tenant scope.
**Success Criteria**: The causal test passes; missing/stale schema and failed initialization remain recoverable; backend replacement cannot reuse the wrong database's readiness.
**Tests**: Real PostgreSQL concurrency, migration/reinitialization, backend cache and tenant-isolation tests; SQLite adjacent controls.
**Status**: Complete

**Design**: Record completed trailing PostgreSQL schema reconciliation as v71. Current-v71 opens always run the existing fail-closed catalog/relationship checks, with read-compatible catalog locks, then hydrate local FTS aliases without DDL/DML. Keep migration verifiers' original locks. A session advisory lock on an independently owned checkout spans the legacy v63 migration's durable commits; publish v71 only after the final reconciliation/auxiliary tables succeed in one transaction. SQLite is unchanged. This replaces the rejected process-local cache, which missed cold workers and warm catalog drift.

**Evidence so far**: Original active-reader causal tests failed2/passed2. Cache review controls failed8/passed4. First versioned lifecycle run passed61/failed1: concurrent fresh bootstrap escaped the transaction advisory lock when existing v63 committed a page. Revised lock lifetime accordingly; expanded lifecycle, migration rollback, drift, cold search, and concurrent-worker checks are in progress. Original failures remain in ignored local logs.

**Completed verification**: 66 lifecycle tests,3 lock-lifetime/cleanup tests,110 adjacent migration/FTS/backend cases and27 existing RLS/drift cases now pass across their scoped reruns (206 unique tests, no PostgreSQL skips). Five old fixture failures reproduce on untouched HEAD; repaired stale tenant-aware FTS stubs and bounded historical v59→v64 reconstruction. Two routing fakes isolate the new coordinator while real integration tests exercise its lock. Bandit0 findings/errors; no new Ruff diagnostics. Independent production review clear; final test-diff review and native acceptance are separate gates.

## Stage 3: review and targeted acceptance
**Goal**: Verify the bounded repair and record exactly what passed.
**Success Criteria**: Independent review clear, scoped static/security checks complete, targeted fresh PostgreSQL second-user Notes acceptance succeeds, tracker and task current.
**Tests**: Native/API second-user Notes initialization during ordinary first-user activity on an immutable source snapshot; no PostgreSQL skip or new full-matrix claim.
**Status**: In Progress
