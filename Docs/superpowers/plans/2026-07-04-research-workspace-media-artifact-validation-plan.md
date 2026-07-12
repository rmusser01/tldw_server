## Stage 1: Trace And Reproduce
**Goal**: Identify Research Workspace generated file/media artifact paths that can complete placeholder output.
**Success Criteria**: Focused tests fail for placeholder quiz, flashcard, audio, and data table outputs.
**Tests**: `bunx vitest run ../packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage2.test.tsx`
**Status**: Complete

## Stage 2: Shared Validation
**Goal**: Reject placeholder-only generated output in the existing artifact generation flow before persistence/downloadable artifacts are marked completed.
**Success Criteria**: Placeholder quiz, flashcard, audio, and data table outputs fail closed; valid outputs still pass.
**Tests**: Focused Stage 2 StudioPane tests.
**Status**: Complete

## Stage 3: Verify And Record
**Goal**: Run focused regression tests and record Backlog/PR evidence.
**Success Criteria**: Focused tests, diff check, and applicable security checks pass; task and branch are updated.
**Tests**: Focused Research Workspace suites plus `git diff --check`.
**Status**: Complete
