# Knowledge Power User Settings Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden `/knowledge` for experienced users who need compact source control, advanced RAG tuning, evidence review, provider selection, and WebUI/extension parity.

**Architecture:** Keep simple and detailed modes as two presentations of the same Knowledge QA state. Advanced settings should remain progressive: Basic mode for common tuning, Expert mode for retrieval details, with focus-safe drawer behavior and reversible defaults.

**Tech Stack:** React, shared Knowledge QA UI, local persistence, Vitest, Playwright, accessibility checks.

**Backlog Task:** TASK-528.7

---

## Boundaries

- This phase optimizes the existing Knowledge QA power-user workflow.
- Do not convert `/knowledge` into a generic CRUD workspace.
- Do not add flashcard behavior to `/knowledge`.

## Files

- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/BasicSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/ExpertSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel/PresetSelector.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/CompactToolbar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/KnowledgeContextBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/context/AnswerModelMenu.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/hooks/useLayoutMode.ts`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/ExpertSettings.accessibility.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/HistorySidebar.responsive.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/CompactToolbar.test.tsx`

## Task 1: Write Failing Settings Accessibility Tests

- [x] Add tests for settings drawer open, close, Escape, focus return, focus trap, and reset defaults.
- [x] Add tests for Basic versus Expert mode switch and first-use Expert hint.
- [x] Add tests for All Options filtering in Expert mode.
- [x] Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/SettingsPanel.behavior.test.tsx apps/packages/ui/src/components/Option/KnowledgeQA/__tests__/ExpertSettings.accessibility.test.tsx
```

Expected: tests fail where accessibility or filtering is incomplete.

Result: Focused settings and expert coverage is included in the passing 102-test Knowledge QA Vitest run.

## Task 2: Improve Compact And Detailed Mode Parity

- [x] Audit all controls available in detailed mode: sources, exact scope, profiles, preset, web fallback, model, settings, history, evidence.
- [x] Ensure compact mode exposes the same critical controls through toolbar buttons or a compact drawer.
- [x] Add tests for compact-mode source/profile/model/settings access.
- [x] Confirm text does not overflow in mobile and extension options widths.

Result: Compact simple mode now opens a source scope and profiles dialog that reuses the shared context controls for source categories, exact documents/notes, saved profiles, preset, web fallback, answer model, and settings.

## Task 3: Harden Provider And Model Controls

- [x] Add tests for provider list loading, provider list failure, server default, manual model entry, and restored model selection.
- [x] Redact or avoid sensitive provider configuration details in UI errors.
- [x] Keep provider/model copy short and task-focused.

## Task 4: Verify Cross-Surface Parity

- [x] Run WebUI fixture e2e in desktop and mobile viewports.
- [ ] Run extension options e2e at representative extension viewport.
- [x] Document intentional differences, especially setup and permissions.
- [x] Fail tests if shared Knowledge QA controls disappear from one surface without a documented reason.

Result: WebUI Chromium fixture coverage passed for the Knowledge QA state and empty-recovery flows. Extension E2E was attempted, but the WXT production build stalled before any browser test started; the stuck process tree was terminated and the blocker is recorded on TASK-528.7.

## Task 5: Close Verification

- [x] Run targeted Vitest files for settings, compact toolbar, history sidebar, and provider controls.
- [x] Run WebUI and extension e2e parity checks.
- [x] Record Bandit as not applicable unless Python backend files were touched.
- [x] Update TASK-528.7 with verification results and parity notes.

Result: Targeted Vitest passed. WebUI E2E passed. Extension E2E is recorded as blocked by the extension production build stall. Bandit is not applicable because TASK-528.7 touched frontend and E2E files only.
