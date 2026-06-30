## Stage 1: Plan and Baseline
**Goal**: Keep the Persona Visual import handoff slice isolated and scoped to #1696.
**Success Criteria**: TASK-357 has the approved plan, the worktree is on `codex/persona-visual-import-handoff-1696`, and no main-checkout files are edited.
**Tests**: `git status --short --branch`
**Status**: Complete

## Stage 2: Focused Failing Tests
**Goal**: Capture the missing import upload and handoff behavior before implementation.
**Success Criteria**: Tests fail for invalid archive copy/no POST, backend FormData key, import status copy, and selected imported draft handoff.
**Tests**: `bunx vitest run ../packages/ui/src/components/PersonaGarden/__tests__/VisualPackEditor.test.tsx`
**Status**: Complete

## Stage 3: Minimal Implementation
**Goal**: Polish the Persona Visual archive import path without changing activation or renderer scope.
**Success Criteria**: Upload uses the backend `archive` field, unsupported filenames fail locally with clear copy, preview/commit statuses are readable, and completed commits select the returned draft.
**Tests**: Focused VisualPackEditor Vitest.
**Status**: Complete

## Stage 4: Validation
**Goal**: Verify the touched scope and repo hygiene.
**Success Criteria**: Focused tests pass, `git diff --check` passes, and Bandit is run for Python changes or documented as not applicable.
**Tests**: Focused Vitest, `git diff --check`, optional backend/API/Bandit if Python changes are introduced.
**Status**: Complete

## Stage 5: Closeout
**Goal**: Leave a reviewable branch with task metadata, commit, and PR.
**Success Criteria**: TASK-357 is updated, changes are committed, and the PR references #1696/#1510.
**Tests**: Final status and verification notes.
**Status**: Complete
