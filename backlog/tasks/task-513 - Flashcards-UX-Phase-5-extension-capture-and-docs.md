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
- [ ] #1 Sidepanel flashcards entry point exposes clear actions for opening full Flashcards and finding the existing selected-text capture-to-flashcards handoff.
- [ ] #2 Extension/WebUI flashcards docs describe the current tab names and directly connected extension capture workflow.
- [ ] #3 Focused route tests cover the sidepanel flashcards entry point behavior.
- [ ] #4 Verification commands are recorded in the task before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-26-flashcards-ux-phase5-extension-capture-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the Phase 5 extension flashcards capture bridge. The sidepanel /flashcards route no longer auto-opens a tab on mount; it now offers an explicit full Flashcards action and a direct Generate from page selection action that captures the active tab selection with browser.scripting and opens the existing /flashcards Create & Import generate handoff. Updated the extension feature doc and WebUI/extension study guide, including the published mirror, to use current tab names and describe the selected-text capture workflow. Verification: bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts passed with 10 tests; git diff --check passed; bun run verify:design-system-state exited 0 with existing baseline exceptions; stale Flashcards doc term grep returned no matches. Bandit skipped because this task touched TypeScript/Markdown only, not Python.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
