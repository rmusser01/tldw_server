# Flashcards Extension Template Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add native template application to the extension sidepanel draft queue without adding in-extension review.

**Architecture:** Reuse the existing Flashcards template materialization modal and helper so the sidepanel does not duplicate placeholder parsing. Each sidepanel draft keeps its source provenance and queue identity while template application replaces front/back/model/notes/extra/tags for the selected draft.

**Tech Stack:** React, Ant Design, TanStack Query flashcard hooks, existing Flashcards template utilities, Vitest and Testing Library.

---

### Task 1: Sidepanel Template Application Tests

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`

- [x] **Step 1: Establish baseline**

Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
Expected: PASS before feature tests are added.

- [x] **Step 2: Write failing tests**

Add sidepanel route tests proving:
- a captured draft exposes an `Apply template` action,
- choosing a template with placeholders updates only that draft's front/back/model/notes/extra/tags,
- save payload preserves source provenance and uses the template model fields,
- generated drafts can also receive template output,
- missing templates show the existing modal empty state without clearing the draft queue.

- [x] **Step 3: Verify RED**

Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
Expected: FAIL because the sidepanel does not yet render native template application controls.

### Task 2: Sidepanel Template Application

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`

- [x] **Step 1: Reuse existing template modal**

Import `FlashcardTemplateValueModal` and render it only when a draft is selected for template application.

- [x] **Step 2: Apply materialized draft fields**

When the modal returns a template draft, update the selected sidepanel draft's `front`, `back`, `modelType`, `tags`, `notes`, and `extra`, while preserving `id`, `sourceId`, and `sourceTitle`.

- [x] **Step 3: Add per-draft action**

Add a small `Apply template` button to each draft card, disabled while saving. Keep visual density consistent with the existing remove/save controls.

- [x] **Step 4: Verify GREEN**

Run: `bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx`
Expected: PASS.

### Task 3: Docs And Source List

**Files:**
- Modify: `Flashcards-UX-Fix-List.md`
- Modify: `apps/extension/docs/features/flashcards.md`
- Modify: `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `backlog/tasks/task-521 - Add-flashcards-extension-native-template-application.md`

- [x] **Step 1: Update F12 status**

Mark native sidepanel template application complete while leaving in-extension review deferred.

- [x] **Step 2: Update docs**

Describe that captured/generated sidepanel drafts can apply existing templates before save.

- [x] **Step 3: Update task**

Record acceptance criteria, implementation notes, final summary, touched files, and verification.

### Task 4: Final Verification And PR

**Files:**
- No new production files beyond Task 2 and docs.

- [x] **Step 1: Run focused tests**

Run: `bunx vitest run src/components/Flashcards/components/__tests__/FlashcardTemplateValueModal.test.tsx src/components/Flashcards/utils/__tests__/flashcard-template-resolution.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx`
Expected: PASS.

- [x] **Step 2: Run broader sidepanel/generate handoff tests**

Run: `bunx vitest run src/services/__tests__/flashcards-generate-handoff.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts`
Expected: PASS.

- [x] **Step 3: Run static checks**

Run: `git diff --check`
Expected: PASS.

Run: `NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false`
Expected: No new sidepanel/template errors; known unrelated CharacterListContent density baseline may still fail.

- [x] **Step 4: Commit and open PR**

Commit the slice and create a draft PR against `dev`.
