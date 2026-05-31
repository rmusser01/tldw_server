# Flashcards Extension Capture Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the next F12 extension slice: a sidepanel repeat-capture queue with editable drafts, delete, save-one, and save-all behavior while keeping generation/templates/import/review in full Flashcards.

**Architecture:** Keep the work inside the existing `SidepanelFlashcards` route and tests. Replace the single `draft` state with a small local draft queue keyed by generated IDs, keep deck selection global for the sidepanel, and preserve failed drafts during save-all by removing only drafts that save successfully.

**Tech Stack:** React, Ant Design, lucide-react icons, WXT `browser` APIs, existing Flashcards hooks, Vitest, React Testing Library.

---

### Task 1: Add Queue Behavior Tests

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`

- [x] **Step 1: Write failing multi-capture queue test**

Add coverage proving two successful capture clicks append two editable draft cards instead of replacing the first draft. Mock `browserMocks.executeScript` with sequential selected-text values and assert both draft headings/fields are present.

- [x] **Step 2: Write failing delete/edit test**

Add coverage proving users can edit a draft, delete a different draft, and keep the edited draft in place.

- [x] **Step 3: Write failing save-one test**

Update the existing single-save expectation so saving one draft removes only that draft, leaves other unsaved drafts in the queue, and shows a success status.

- [x] **Step 4: Write failing save-all partial failure test**

Add coverage where the first save succeeds and the second save rejects. Assert the mutation ran for both valid drafts, the successful draft was removed, the failed draft remains editable, and the status uses partial-failure copy.

- [x] **Step 5: Run focused tests and verify RED**

Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`

Expected: FAIL because the route still has a single `draft`, no delete controls, and no save-all action.

### Task 2: Implement Sidepanel Queue

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`

- [x] **Step 1: Add draft queue model**

Extend `CaptureDraft` with an `id`; store `drafts: CaptureDraft[]` instead of `draft: CaptureDraft | null`.

- [x] **Step 2: Append successful captures**

Change capture success to append a new draft while keeping existing drafts. Keep the existing behavior that a failed later capture shows an error, but do not discard existing queued drafts.

- [x] **Step 3: Render queued drafts**

Render each draft as an editable section with stable labels and per-draft Front/Back fields. Keep source context visible.

- [x] **Step 4: Add delete and clear-saved mechanics**

Add a visible delete action per draft. Remove successfully saved drafts from the queue after save-one/save-all; preserve failed drafts for retry.

- [x] **Step 5: Add save-one and save-all actions**

Use `useCreateFlashcardMutation().mutateAsync` for each valid draft. Save-one operates on a single draft. Save-all attempts all valid drafts, reports success/partial/failure, removes only successes, and keeps invalid or failed drafts.

- [x] **Step 6: Preserve deck and availability states**

Keep existing deck loading/error/unavailable/no-decks copy and disabled save behavior. Save-all should be disabled when there are no valid drafts, no selectable deck, or a save is pending.

### Task 3: Update UX Source And Task Metadata

**Files:**
- Modify: `Flashcards-UX-Fix-List.md`
- Modify via Backlog.md MCP: `backlog/tasks/task-517 - Add-flashcards-extension-repeat-capture-queue.md`

- [x] **Step 1: Update master checklist wording**

Change the deferred F12 item to show repeat-capture queue and bulk save as complete while leaving templates, generated drafts, and in-extension review deferred.

- [x] **Step 2: Add implementation notes to TASK-517**

Record touched files, plan path, and the narrowed non-goals: no native LLM generation, no template application, no in-extension review.

### Task 4: Verification And Commit

**Files:**
- Verify: `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`
- Verify: `apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`
- Verify: touched docs/task files

- [x] **Step 1: Run focused sidepanel tests**

Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`

Expected: PASS.

- [x] **Step 2: Run TypeScript check if practical**

Run from `apps/packages/ui`: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`

Expected: PASS or only documented unrelated baseline failures.

- [x] **Step 3: Run diff hygiene**

Run from repo root: `git diff --check`

Expected: PASS.

- [x] **Step 4: Record Bandit applicability**

No Python files should be touched. Record “Bandit not applicable: frontend/docs/task-only changes” in TASK-517.

- [x] **Step 5: Finalize TASK-517**

Mark acceptance criteria/DoD complete, add verification results, and add final summary.

- [x] **Step 6: Commit**

Commit message: `feat: add flashcards sidepanel capture queue`
