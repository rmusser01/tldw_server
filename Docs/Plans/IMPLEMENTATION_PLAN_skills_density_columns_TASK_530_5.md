# Skills Density And Column Controls Implementation Plan

## Stage 1: Preference Contract And Regression Tests
**Goal**: Define the focused frontend behavior for Skills table density and optional columns.
**Success Criteria**: Tests fail before implementation for density toggling, column visibility, and restored preferences.
**Tests**: Add focused cases to `apps/packages/ui/src/components/Option/Skills/__tests__/Manager.test.tsx`.
**Status**: Complete

## Stage 2: Local Preference Helpers
**Goal**: Add small, component-local helpers for loading and saving Skills table preferences safely.
**Success Criteria**: Missing, malformed, or partial localStorage values fall back to defaults without breaking render.
**Tests**: Covered through Skills manager render/remount tests.
**Status**: Complete

## Stage 3: Dense View And Column Controls
**Goal**: Add keyboard-accessible controls that let users switch table density and show/hide optional secondary columns.
**Success Criteria**: Name and actions remain mandatory; optional description, mode, argument hint, visibility, and model invocation columns can be toggled.
**Tests**: Focused Skills manager tests verify visible columns, persisted choices, and unchanged server-backed query behavior.
**Status**: Complete

## Stage 4: Verification And Task Closeout
**Goal**: Verify the focused UI slice, update task notes, and keep unrelated work out of the branch.
**Success Criteria**: Focused Vitest passes, `git diff --check` passes, Bandit is documented as not applicable for frontend-only TypeScript, and `TASK-530.5` records verification.
**Tests**: `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx`; `git diff --check`.
**Status**: Complete

## Verification Notes
- PR #2339 review follow-up addressed hidden context sort state, AntD-class-coupled test selection, and render-phase preference initialization feedback.
- PASS: `bunx vitest run src/components/Option/Skills/__tests__/Manager.test.tsx` (22 tests).
- PASS: `bunx vitest run src/services/__tests__/tldw-api-client.boundary-slices.test.ts` (8 tests).
- PASS: `git diff --check`.
- TYPECHECK CAVEAT: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit -p tsconfig.json --pretty false` still fails on inherited baseline errors in Notes tests, `src/entries/background.ts`, and `src/services/tldw/voice-cloning.ts`; no `src/components/Option/Skills/Manager.tsx` errors remain.
- Bandit is not applicable for this frontend-only TypeScript/docs/test task.
