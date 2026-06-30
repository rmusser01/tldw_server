# Flashcards Extension Generate Handoff Implementation Plan

## Stage 1: Sidepanel Generate Handoff Tests
**Goal**: Prove the extension can capture selected page text and open full Flashcards directly in the existing GeneratePanel.
**Success Criteria**:
- A sidepanel action labeled for generation is visible alongside manual capture.
- Clicking it reads the active page selection with the existing injected helper.
- The opened options URL targets `/flashcards` with `tab=importExport`, `generate=1`, selected text, page URL, and page title.
- Capture validation failures stay inline and do not clear queued manual drafts.
**Tests**:
- `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
**Status**: Complete

## Stage 2: Component Implementation
**Goal**: Add a direct Generate-from-selection action without native sidepanel LLM generation.
**Success Criteria**:
- Handoff uses the existing `buildFlashcardsGenerateRoute` service.
- Existing manual capture/save-one/save-all behavior remains unchanged.
- In-flight save state still prevents sidepanel queue changes.
**Tests**:
- `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
**Status**: Complete

## Stage 3: Source List And Task Closeout
**Goal**: Keep the master fix list, user docs, and Backlog task aligned with the completed F12 handoff slice.
**Success Criteria**:
- `Flashcards-UX-Fix-List.md` distinguishes this generated-selection handoff from still-deferred native sidepanel generation/templates/review.
- Extension flashcards docs describe queued capture, `Generate from selection`, and the deferred native generation/review boundary.
- TASK-518 records verification and final summary.
**Tests**:
- `git diff --check`
**Status**: Complete
