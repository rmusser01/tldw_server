# Flashcards Extension Native Capture MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native extension sidepanel flashcard capture MVP for selected page text.

**Architecture:** Keep the sidepanel route as the only product surface changed. Reuse existing flashcard deck and create-card hooks, preserve the full Flashcards handoff, and keep LLM generation/templates/bulk drafting deferred.

**Tech Stack:** React, TypeScript, Ant Design, WXT browser APIs, TanStack Query-backed flashcard hooks, Vitest and Testing Library.

---

### Task 1: Native Capture UI Contract

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`

- [x] **Step 1: Write failing tests**

Add route tests that verify:
- The sidepanel offers `Capture page selection` without opening the full options tab.
- Capturing selected text opens an inline draft editor.
- The draft editor shows deck selection, Front, Back, Save card, and full-workspace continuation controls.
- Saving calls `useCreateFlashcardMutation().mutateAsync` with `deck_id`, edited front/back, `model_type: "basic"`, `is_cloze: false`, `reverse: false`, `source_ref_type: "manual"`, and the active page URL.

- [x] **Step 2: Run focused sidepanel test and verify it fails**

Run:

```bash
bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx
```

Expected: FAIL because the inline native capture editor and save behavior do not exist yet.

- [x] **Step 3: Implement minimal sidepanel native capture**

Use `useDecksQuery()` to load decks and `useCreateFlashcardMutation()` to save. Capture selected text through the existing injected helper, set draft state locally, and keep the existing full Flashcards opener. Use compact Ant Design `Select`, `Input`, and `Button` controls inside the sidepanel flow.

- [x] **Step 4: Run focused test and verify it passes**

Run:

```bash
bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx
```

Expected: PASS.

### Task 2: Error And Empty States

**Files:**
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-flashcards.tsx`

- [x] **Step 1: Write failing tests**

Add coverage for:
- No decks: save remains unavailable and the user can open full Flashcards to create a deck.
- Save failure: draft stays visible and an inline error appears.
- No selected text: user remains in place and sees the existing validation message.

- [x] **Step 2: Run focused sidepanel test and verify it fails**

Run:

```bash
bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx
```

Expected: FAIL for the new native capture recovery states.

- [x] **Step 3: Implement minimal recovery behavior**

Disable save until a deck and non-empty fields exist, preserve draft values on save failure, and show concise inline status/error copy.

- [x] **Step 4: Run focused test and verify it passes**

Run:

```bash
bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx
```

Expected: PASS.

### Task 3: Close Checklist And Verify

**Files:**
- Modify: `Flashcards-UX-Fix-List.md`
- Modify: `apps/extension/docs/features/flashcards.md`
- Modify: `Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md`
- Modify: `backlog/tasks/task-516 - Flashcards-UX-F12-native-extension-capture-MVP.md`

- [x] **Step 1: Update master checklist**

Mark F12 native extension deck-picker/edit/save as completed for the MVP and keep deferred items explicit: generated drafts, templates, bulk editing, and in-extension review.

- [x] **Step 2: Update user-facing docs**

Update the extension flashcards feature doc and WebUI flashcards study guide copies so they describe `Capture page selection`, deck selection, editable Front/Back fields, one-card save, and the remaining full-workspace handoff for generation/import/review.

- [x] **Step 3: Update Backlog task**

Record touched files, verification, non-goals, and Bandit applicability.

- [x] **Step 4: Run verification**

Run:

```bash
bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx
git diff --check
NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false
```

Result: focused sidepanel and route-registry tests passed, `git diff --check` passed, and typecheck reported only the documented unrelated baseline `CharacterListContent.design-system.test.tsx` density diagnostic. No Python files were touched, so Bandit is not applicable to this slice.
