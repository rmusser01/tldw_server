# PR 2808 CI Gate Remediation Plan

Backlog task: `TASK-12020.44`

## Stage 1: Reproduce and classify failures
**Goal**: Distinguish PR regressions from inherited failures using the PR head and exact base commit.
**Success Criteria**: Each failing gate has a reproduced error and evidence-backed root cause.
**Tests**: Targeted extension Playwright tests and the shard coverage guard on both relevant revisions.
**Status**: Complete

## Stage 2: Add regression coverage
**Goal**: Encode the split-entrypoint i18n initialization contract before changing production code.
**Success Criteria**: A focused guard test fails when either active extension entrypoint omits i18n initialization.
**Tests**: `split-entrypoint-i18n.guard.test.ts` fails against the current implementation.
**Status**: Complete

## Stage 3: Apply bounded fixes
**Goal**: Fix the three observed gate failures without broadening application behavior.
**Success Criteria**: Active sidepanel/options entrypoints initialize i18n, the watchlist test targets the stable row contract, and both workspace test modules belong to every generated shard matrix.
**Tests**: Focused Vitest, both targeted Playwright scenarios, and shard coverage guard.
**Status**: Complete

## Stage 4: Verify and review
**Goal**: Confirm the patch is clean and introduces no adjacent regressions.
**Success Criteria**: Focused tests, lint/type checks for touched TypeScript, `git diff --check`, and self-review pass.
**Tests**: Relevant package checks and CI-equivalent commands.
**Status**: Complete

## Stage 5: Deliver and monitor
**Goal**: Commit and push the fix to PR #2808, then observe its replacement check set without interfering with PR #2806.
**Success Criteria**: The PR head contains the verified commit and replacement checks start from that head.
**Tests**: Read-only GitHub check inspection.
**Status**: In Progress
