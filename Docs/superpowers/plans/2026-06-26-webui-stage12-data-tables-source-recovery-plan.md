# WebUI Stage 12 Data Tables Source Recovery Plan

## Stage 1: Lock Current Gap
**Goal**: Add a focused SourceSelector regression test for failed chat/document source loading.
**Success Criteria**: The test expects a shared unavailable recovery state with diagnostics and retry instead of only an empty list.
**Tests**: `bun run test:run ../packages/ui/src/components/Option/DataTables/__tests__/SourceSelector.recovery.test.tsx`
**Status**: Complete

## Stage 2: Adopt Shared Recovery State
**Goal**: Render a `StatePanel` unavailable state inside the source picker when non-RAG source queries fail.
**Success Criteria**: User-facing copy stays stable, raw error details move into diagnostics, and retry calls the existing query refetch.
**Tests**: Focused SourceSelector test.
**Status**: Complete

## Stage 3: Verification And Task Closure
**Goal**: Verify the focused slice and record completion on `TASK-12041`.
**Success Criteria**: Focused test, touched-file lint, and whitespace checks pass; Bandit is documented as not applicable for TS/TSX/docs-only changes.
**Tests**: Focused SourceSelector test, ESLint touched files, `git diff --check`.
**Status**: Complete
