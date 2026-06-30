# Knowledge Ready Search Source Scope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden the ready `/knowledge` search workflow for source scope, exact source selection, saved profiles, suggestions, shortcuts, presets, web fallback, and answer model selection.

**Architecture:** Treat source scope as the user's search contract. Keep category selection, exact document/note selection, saved profiles, and compact-mode controls synchronized through the existing Knowledge QA provider and context toolbar rather than one-off local state.

**Tech Stack:** React, shared Knowledge QA UI, local storage profile persistence, Vitest, Playwright.

**Backlog Task:** TASK-528.5

---

## Boundaries

- This phase assumes the route can reach ready state through TASK-528.1 and TASK-528.2 fixtures.
- Do not change backend retrieval semantics unless a frontend contract bug requires a separate backend task.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SearchBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/CompactToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/AnswerModelMenu.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/PresetSelector.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/BasicSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/AnswerPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/layout/KnowledgeQALayout.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/querySuggestions.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SearchBar.behavior.test.tsx`
- Add: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerModelMenu.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx`

## Task 1: Write Failing Scope And Profile Tests

- [x] Add tests for selecting source categories, selecting exact documents, selecting exact notes, and clearing exact scope.
- [x] Add tests for saved profile create, restore, delete, and compact-mode access.
- [x] Add tests for provider/model loading success and failure.
- [x] Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx
```

Expected: tests fail where current behavior is incomplete.

Result: initial focused run failed on exact profile scope round-trip, compact exact counts, Deep preset naming, and answer-provider loading/error states.

## Task 2: Normalize Preset Labels

- [x] Choose one user-facing label set: Fast, Balanced, Deep, Custom.
- [x] Update toolbar, settings drawer, saved profile summaries, loading hints, and tests to use the same labels.
- [x] Keep internal ids stable; `thorough` remains the internal preset id for API/provider compatibility.
- [x] Add unit coverage that fails if settings drifts back to `Thorough`.

## Task 3: Harden Exact Source Selection

- [x] Ensure document/media and notes selectors have loading, empty, retry, and API error states.
- [x] Show selected counts clearly in both detailed and compact modes.
- [x] Ensure filtering exact sources does not clear already selected items unexpectedly.
- [x] Ensure "Use all" clears exact filters without changing selected source categories.

## Task 4: Improve Search Input And Shortcuts

- [x] Validate `/` focus, Cmd/Ctrl+K new search, arrow navigation in suggestions, Enter selection, and Escape close.
- [x] Ensure shortcuts do not steal focus from text inputs, dialogs, or menus.
- [x] Keep suggestions grounded in history, source titles, or static examples.

## Task 5: Harden Web Fallback And Model Controls

- [x] Show capability-aware web fallback state and disabled reason.
- [x] Preserve user-controlled fallback state when web fallback is unavailable.
- [x] Handle provider list loading, error, server default, and manual model entry.
- [ ] Add e2e coverage for a scoped ready search with selected documents and notes.

## Task 6: Close Verification

- [x] Run targeted Vitest files listed above.
- [ ] Run route fixture e2e for ready search once TASK-528.1 fixtures exist.
- [x] Record Bandit as not applicable; only TypeScript/React files were touched.
- [ ] Update TASK-528.5 with verification results.

Verification:

```bash
bunx vitest run src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.profiles.test.tsx src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/AnswerModelMenu.test.tsx src/components/Option/KnowledgeQA/__tests__/KnowledgeContextBar.test.tsx src/components/Option/KnowledgeQA/__tests__/SearchBar.behavior.test.tsx src/components/Option/KnowledgeQA/__tests__/querySuggestions.test.ts src/components/Option/KnowledgeQA/__tests__/AnswerPanel.states.test.tsx
```

Result: 8 files, 93 tests passed.

Route fixture e2e note: not rerun in this slice because the current environment still has the previously recorded browser launch/WXT production build blockers from TASK-528.4 and TASK-306.
