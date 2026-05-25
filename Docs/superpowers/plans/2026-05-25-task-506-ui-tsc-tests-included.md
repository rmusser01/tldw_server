## Stage 1: Reconfirm Baseline
**Goal**: Capture the current full `apps/packages/ui` TypeScript error set with tests included.
**Success Criteria**: `tsconfig.json` still includes `src/**/*`; the failing diagnostics are grouped into production and test clusters.
**Tests**: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`.
**Status**: Complete

## Stage 2: Production Type Errors
**Goal**: Fix source-level typing drift without changing user-facing behavior.
**Success Criteria**: `ResearchWorkspace`, `Sources`, `Watchlists`, `Integrations`, and `TldwModels` production diagnostics are resolved.
**Tests**: Full `tsc` after each cluster and focused source inspection for touched contracts.
**Status**: Complete

## Stage 3: Test Contract Drift
**Goal**: Update fixtures, mocks, and helper types so tests compile against current production contracts.
**Success Criteria**: Test diagnostics are resolved through typed builders and mocks, not by excluding tests.
**Tests**: Full `tsc`; focused Vitest may be added for heavily edited test helpers.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Prove the main UI TypeScript check passes with tests included and record evidence.
**Success Criteria**: Full `tsc` and `git diff --check` pass; TASK-506 notes and DoD are updated.
**Tests**: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`; `git diff --check`.
**Status**: Complete
