## Stage 1: Add Decomposition Guard
**Goal**: Capture issue #987's module split expectation before production edits.
**Success Criteria**: A focused test fails until import/export/generate panels are addressable as separate modules.
**Tests**: `ImportExportTab.decomposition.test.tsx`
**Status**: Complete

## Stage 2: Extract Panels
**Goal**: Move import, export, and generate panels out of `ImportExportTab.tsx` without behavior changes.
**Success Criteria**: `ImportExportTab.tsx` becomes a shell that renders imported panels and shared transfer summary state.
**Tests**: Existing ImportExportTab focused tests continue to pass.
**Status**: Complete

## Stage 3: Verify And Package
**Goal**: Run focused frontend checks plus repository hygiene checks.
**Success Criteria**: Focused Vitest passes, `git diff --check` is clean, and changed source scope passes Bandit where applicable.
**Tests**: Focused Vitest, diff check.
**Status**: Complete
