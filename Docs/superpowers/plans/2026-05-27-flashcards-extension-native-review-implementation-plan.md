# Flashcards Extension Native Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact native flashcard review loop to the extension sidepanel so users can review due cards without leaving the sidepanel.

**Architecture:** Reuse the existing sidepanel deck selection state and the shared Flashcards review query/mutation hooks. Keep the sidepanel review flow intentionally small: select deck, fetch next due card, reveal back, submit Again/Hard/Good/Easy, refetch next card, and report inline progress or errors.

**Tech Stack:** React, Ant Design, lucide-react, TanStack Query hooks from `apps/packages/ui/src/components/Flashcards/hooks`, Vitest + Testing Library.

---

### Task 1: Sidepanel Review Contract Tests

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`

- [x] **Step 1: Extend test hook mocks**
  Add `useReviewQuery`, `useReviewFlashcardMutation`, and review mutation/refetch mocks to the existing Flashcards hook mock.

- [x] **Step 2: Write failing native review test**
  Add a test that renders a due card, clicks `Review due card`, reveals the answer, submits `Good`, verifies `{ cardUuid, rating: 3, answerTimeMs }`, confirms no full Flashcards tab opens, and sees inline reviewed progress.

- [x] **Step 3: Write caught-up empty-state test**
  Add a test where the review query returns `null`, then verify the sidepanel says no due cards are available and still offers the full Flashcards workspace for richer review.

- [x] **Step 4: Write failure-retention test**
  Add a test where the rating mutation rejects, then verify the current card remains visible and an inline error is shown.

- [x] **Step 5: Run RED**
  Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
  Expected: FAIL because the sidepanel has no native review action yet.

### Task 2: Compact Native Review UI

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`

- [x] **Step 1: Import review hook contracts**
  Import `useReviewQuery` and `useReviewFlashcardMutation` from the existing Flashcards hooks barrel.

- [x] **Step 2: Add review state**
  Add state for review panel visibility, revealed-answer state, reviewed count, and answer-start timing.

- [x] **Step 3: Add top-level review action**
  Add a `Review due card` button alongside the existing open/capture/generate actions. Opening the panel clears stale capture/save status.

- [x] **Step 4: Render deck-aware review panel**
  Reuse the same deck availability states as saving: unavailable, loading, load error, no decks, and selected deck.

- [x] **Step 5: Render one-card review**
  Show front first, reveal answer on demand, then render rating buttons for Again/Hard/Good/Easy using the established rating values `0`, `2`, `3`, and `5`.

- [x] **Step 6: Submit and advance**
  On rating submit, call `mutateAsync({ cardUuid, rating, answerTimeMs })`, increment sidepanel reviewed count, reset reveal state, and refetch the next review card.

- [x] **Step 7: Preserve recovery**
  On mutation failure, keep the current card and answer visible and show an inline error.

- [x] **Step 8: Run GREEN**
  Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
  Expected: PASS.

### Task 3: UX Source And Docs Updates

**Files:**
- Modify: `Flashcards-UX-Fix-List.md`
- Modify: `apps/extension/docs/features/flashcards.md`
- Modify: `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `backlog/tasks/task-522 - Add-native-sidepanel-flashcard-review-loop.md`

- [x] **Step 1: Update master fix list**
  Mark F12 in-extension review complete with the limited native due-card sidepanel scope.

- [x] **Step 2: Update extension docs**
  Replace copy that says review is unavailable in-sidepanel with the new compact due-card review workflow, while keeping richer study tools in full Flashcards.

- [x] **Step 3: Update Backlog task**
  Record implementation notes, touched files, verification commands, and completion summary.

### Task 4: Final Verification And PR

**Files:**
- All changed files above.

- [x] **Step 1: Focused tests**
  Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
  Result: PASS, 44 tests after PR review fixes.

- [x] **Step 2: Related regression tests**
  Run: `bunx vitest run src/components/Flashcards/hooks/__tests__/useFlashcardQueries.review-next.test.tsx src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`
  Result: PASS, 54 tests after PR review fixes.

- [x] **Step 3: Typecheck**
  Run: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`
  Result: Known unrelated baseline failure remains in `src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx(35,3)` where `"comfortable"` is not assignable to `GalleryCardDensity`.

- [x] **Step 4: Diff hygiene**
  Run: `git diff --check`
  Expected: PASS.

- [x] **Step 5: Security check**
  Bandit is not applicable unless Python files are touched. Record the non-Python scope in the Backlog task.

- [ ] **Step 6: Commit and PR**
  Commit the task and open a draft PR with a human-written change summary explaining why this is intentionally a compact sidepanel review loop rather than a full Study redesign.
