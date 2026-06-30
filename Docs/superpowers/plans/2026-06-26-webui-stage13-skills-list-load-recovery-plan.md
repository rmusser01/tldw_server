# WebUI Stage 13: Skills List-Load Recovery Plan

## Stage 1: Lock Failure-State Expectations
**Goal**: Prove Skills list-load failures use the shared recovery surface with diagnostics.
**Success Criteria**: The focused Skills manager test fails until the list-load error renders as a `RecoveryCallout` with request diagnostics and a retry action.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`
**Status**: Complete

## Stage 2: Adopt Shared Recovery State
**Goal**: Replace the plain list-load error banner with `RecoveryCallout` and `buildCapabilityState`.
**Success Criteria**: Primary copy remains user-facing, raw endpoint/error details move to diagnostics, and retry still refetches the list.
**Tests**: Focused Skills manager Vitest suite.
**Status**: Complete

## Stage 3: Verify And Finalize
**Goal**: Run targeted verification and record the result in Backlog.
**Success Criteria**: Focused tests, direct lint, and whitespace checks pass; Bandit is recorded as not applicable for TS/TSX/docs-only changes.
**Tests**: Focused Vitest, direct ESLint on touched TS/TSX files, `git diff --check`.
**Status**: Complete
