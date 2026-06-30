## Stage 1: Extension Flashcards Bridge
**Goal**: Replace the auto-redirect-only sidepanel flashcards route with a small focused bridge that exposes full Flashcards and the existing selected-text capture path.
**Success Criteria**: Users can open full Flashcards directly, see how to make cards from selected text, and jump to the sidepanel chat route where the existing note quick-save `Generate flashcards` action lives.
**Tests**: Focused route/component tests for the sidepanel flashcards bridge and route registration.
**Status**: Complete

## Stage 2: Direct Documentation Refresh
**Goal**: Update the extension flashcards feature doc and WebUI/extension study guide to match the current Flashcards tab names and extension handoff.
**Success Criteria**: Docs describe `Study`, `Manage`, `Create & Import`, `Templates`, and `Scheduler`; direct extension capture docs point to selected text -> Save to Notes -> Generate flashcards.
**Tests**: Documentation review plus `git diff --check`.
**Status**: Complete

## Stage 3: Verification And Task Finalization
**Goal**: Run focused verification, record results in `TASK-513`, and leave the branch ready for review.
**Success Criteria**: Focused tests pass or known unrelated failures are documented; task acceptance criteria and final summary are updated.
**Tests**: `bunx vitest run ...`, `git diff --check`; design-system guard if the implementation introduces product-state surfaces.
**Status**: Complete
