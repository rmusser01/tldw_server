## Stage 1: Route Contract
**Goal**: Confirm current first-run gating behavior and define the narrow /chat exception.
**Success Criteria**: The plan records that /chat should bypass the global assistant setup modal while preserving inline setup nudges and existing character-chat onboarding behavior.
**Tests**: Focused app layout test for /chat FirstRunGate bypass.
**Status**: Complete

## Stage 2: Red Test
**Goal**: Add failing coverage for /chat first-run access and inline setup nudge availability.
**Success Criteria**: Focused tests fail against current behavior for the /chat route gate while existing nudge behavior remains asserted.
**Tests**: `bunx vitest run __tests__/app/app-layout.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundComposerNotices.first-run.test.tsx`
**Status**: Complete

## Stage 3: Minimal Implementation
**Goal**: Let /chat bypass the global FirstRunGate overlay without changing setup, settings, login, or character-chat onboarding routes.
**Success Criteria**: /chat renders through the app shell with FirstRunGate bypass enabled; setup route target logic remains unchanged for other first-run surfaces.
**Tests**: Focused app layout and first-run nudge Vitest.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Record verification and finish TASK-536.
**Success Criteria**: Focused tests pass, diff checks pass, Bandit applicability is recorded, and TASK-536 AC/DoD are updated.
**Tests**: Focused Vitest; `git diff --check`.
**Status**: Complete
