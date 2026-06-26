## Stage 1: Lock The Admin Sandbox Failure Contract
**Goal**: Capture the current sandbox diagnostics failure gap with a focused Monitoring dashboard regression.
**Success Criteria**: The test requires sandbox diagnostics failures to render through the shared `RecoveryCallout`, with endpoint/status/error details kept in diagnostics.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx -t "distinguishes forbidden sandbox diagnostics from unavailable diagnostics"`
**Status**: Complete

## Stage 2: Adopt Shared Capability Recovery State
**Goal**: Replace the sandbox diagnostics generic error alert with `buildCapabilityState` and `RecoveryCallout`.
**Success Criteria**: Forbidden failures show access-denied user copy, unavailable failures stay distinct, retry remains available, and raw details appear in diagnostics.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Admin/__tests__/MonitoringDashboardPage.test.tsx`
**Status**: Complete

## Stage 3: Verify And Finalize
**Goal**: Run scoped verification and record the outcome in Backlog.
**Success Criteria**: Focused Admin tests pass, touched files lint clean or known existing warnings are recorded, whitespace checks pass, and `TASK-12043` is finalized.
**Tests**: Focused Vitest, direct ESLint on touched TS/TSX files, `git diff --check`.
**Status**: Complete
