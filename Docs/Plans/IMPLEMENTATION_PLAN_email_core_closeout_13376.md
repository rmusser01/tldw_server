# Core email closeout (TASK-13376)

## Stage 1: Close implementation gaps
**Goal**: Metrics, safe logging and attachment policy with integration.
**Success Criteria**: Bounded real-registry observations, no synthetic sensitive sentinels in INFO+ logs, explicit MIME allow/deny and metadata-only outcomes.
**Tests**: Agent regressions plus parser/persistence/API suites.
**Status**: Complete

## Stage 2: Sustained ingestion
**Goal**: At least 50 messages/sec on SQLite then PostgreSQL.
**Success Criteria**: Sustained authenticated HTTP windows, identity/retry/isolation and native failure behavior verified.
**Tests**: Guarded synthetic upload probe, database scope and transaction regressions; Ruff and Bandit.
**Status**: In Progress

## Stage 3: Million-message search
**Goal**: Actual 1,000,000-message representative fixture per backend.
**Success Criteria**: Complete ten-class mix, warm p50 <=250ms / p95 <=900ms; cold results and shape recorded; fix measured failures without changing semantics.
**Tests**: Bulk fixture parity/isolation, existing operator tests, guarded benchmark and correctness checks.
**Status**: In Progress

## Stage 4: Deployment and release evidence
**Goal**: Concrete local rollout/parity/auto_email/rollback evidence and reconciled release gates.
**Success Criteria**: Honest dated reports, Gmail separately deferred, single owner review-ready checklist, scoped resources removed, reviewed tested commits.
**Tests**: Full app endpoints and flag rollback on both backends, focused integration suite, Ruff/Bandit/review.
**Status**: In Progress
