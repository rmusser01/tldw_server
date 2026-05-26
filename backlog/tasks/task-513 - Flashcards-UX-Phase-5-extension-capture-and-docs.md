---
id: TASK-513
title: Flashcards UX Phase 5 extension capture and docs
status: Done
labels:
- ux
- flashcards
- phase-5
- extension
- docs
- frontend
modified_files:
- apps/packages/ui/src/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx
- apps/packages/ui/src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts
- apps/packages/ui/src/services/__tests__/flashcards-generate-handoff.test.ts
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/GeneratePanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/ImportExport/shared.ts
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx
- apps/extension/docs/features/flashcards.md
- Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
- Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the scoped Phase 5 flashcards UX follow-up after the merged import recovery work: make the extension flashcards entry point better support the existing capture-to-card handoff and refresh directly connected flashcards documentation so it matches the current WebUI/extension workflow.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel flashcards entry point exposes clear actions for opening full Flashcards and finding the existing selected-text capture-to-flashcards handoff.
- [x] #2 Extension/WebUI flashcards docs describe the current tab names and directly connected extension capture workflow.
- [x] #3 Focused route tests cover the sidepanel flashcards entry point behavior.
- [x] #4 Verification commands are recorded in the task before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-26-flashcards-ux-phase5-extension-capture-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- PR review-fix pass: rebased on `origin/dev`; preserved selected-page capture provenance as backend-supported `manual` source references; reported tab-open fallback failures inline; captured focused input/textarea selections before falling back to `window.getSelection()`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Phase 5 extension flashcards capture bridge and PR review fixes. The sidepanel /flashcards route no longer auto-opens a tab on mount; it now offers an explicit full Flashcards action, a direct Generate from page selection action, inline failure feedback if tab/window opening is blocked, and an injected selection reader that handles focused input/textarea selections before falling back to window selection. Selected-page captures now carry manual source attribution through the generate handoff and GeneratePanel save payload, preserving the active tab URL/title without inventing an unsupported backend source type. Updated the extension feature doc to use the visible Create & Import tab label and checked the Backlog AC/DoD items. Verification: rebased on origin/dev with no conflicts; bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts src/services/__tests__/flashcards-generate-handoff.test.ts src/components/Flashcards/tabs/__tests__/ImportExportTab.import-results.test.tsx passed with 43 tests; git diff --check passed; bun run verify:design-system-state exited 0 with 225 existing allowed baseline exceptions; rg -n '/flashcards[?]tab=importExport.*selected text|selected text.*\\/flashcards[?]tab=importExport' apps/extension/docs Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md returned no matches. Bandit skipped because this task touched TypeScript/Markdown/Backlog task metadata only, not Python.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
