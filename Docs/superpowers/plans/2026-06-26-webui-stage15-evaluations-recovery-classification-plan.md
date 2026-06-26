## Stage 1: Lock Evaluation Recovery Classification
**Goal**: Capture the Evaluations recovery helper gap with a focused regression.
**Success Criteria**: A 403 Evaluations API response renders the shared `permission_denied` state while preserving route-provided title/message and diagnostics.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationRecoveryCallout.test.tsx -t "classifies forbidden responses"`
**Status**: Complete

## Stage 2: Reuse Shared Capability Mapping
**Goal**: Make `EvaluationRecoveryCallout` derive its recovery state through `buildCapabilityState`.
**Success Criteria**: Existing component props and diagnostics labels remain compatible; status-based states no longer collapse to `unavailable`.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/Evaluations/components/__tests__/EvaluationRecoveryCallout.test.tsx`
**Status**: Complete

## Stage 3: Verify And Finalize
**Goal**: Run scoped verification and record the result in Backlog.
**Success Criteria**: Focused helper tests pass, touched files lint clean, whitespace checks pass, and `TASK-12044` is finalized.
**Tests**: Focused Vitest, direct ESLint on touched TS/TSX files, `git diff --check`.
**Status**: Complete
