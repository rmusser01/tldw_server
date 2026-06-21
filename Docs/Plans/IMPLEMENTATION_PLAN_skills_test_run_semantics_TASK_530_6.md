# Skills Test Run Semantics Implementation Plan

## Stage 1: Red Tests For Test-Run Copy
**Goal**: Capture the safe-operations behavior expected from the Skills test-run modal and row action.
**Success Criteria**: Focused tests fail on the current passive Preview copy and non-alert error rendering.
**Tests**: `bunx vitest run src/components/Option/Skills/__tests__/SkillPreview.test.tsx src/components/Option/Skills/__tests__/Manager.test.tsx -t "test run|execution risk|alert"`
**Status**: Complete

## Stage 2: Rename Preview Semantics
**Goal**: Replace passive Preview language with explicit Test run / Run test language in row actions and the modal.
**Success Criteria**: Users can tell that the action executes the skill path rather than merely opening a read-only preview.
**Tests**: Focused SkillPreview and Manager tests.
**Status**: Complete

## Stage 3: Execution-Risk Disclosure And Error Alert
**Goal**: Add pre-run copy naming argument rendering and fork-mode model/tool risk, and expose execution failures through alert semantics.
**Success Criteria**: The risk copy is visible before execution and errors render with `role="alert"`.
**Tests**: Focused SkillPreview tests.
**Status**: Complete

## Stage 4: Verification And Closeout
**Goal**: Verify the focused frontend slice, update Backlog notes, and keep later Safe Operations work out of this PR.
**Success Criteria**: Focused Vitest passes, `git diff --check` passes, Bandit is documented as not applicable for frontend-only TypeScript, and `TASK-530.6` records verification.
**Tests**: `bunx vitest run src/components/Option/Skills/__tests__/SkillPreview.test.tsx src/components/Option/Skills/__tests__/Manager.test.tsx`; `git diff --check`.
**Status**: Complete
