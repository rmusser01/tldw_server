---
id: TASK-518
title: Add flashcards extension generate-from-selection handoff
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-27 05:01
labels:
- flashcards
- extension
- ux
dependencies: []
documentation:
- Flashcards-UX-Fix-List.md
priority: medium
modified_files:
- apps/packages/ui/src/routes/sidepanel-flashcards.tsx
- apps/packages/ui/src/routes/__tests__/sidepanel-flashcards.test.tsx
- apps/extension/docs/features/flashcards.md
- Docs/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
- Docs/Published/User_Guides/WebUI_Extension/Flashcards_Study_Guide.md
- Flashcards-UX-Fix-List.md
- Docs/superpowers/plans/2026-05-27-flashcards-extension-generate-handoff-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow F12 follow-up: let the extension sidepanel capture selected page text and open full Flashcards directly in the existing GeneratePanel with source context, while keeping native sidepanel LLM generation/templates/review deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel Flashcards exposes Generate from selection next to native capture.
- [x] #2 Generate from selection captures active-page selected text through the existing browser scripting path.
- [x] #3 The generated handoff opens full Flashcards with tab=importExport, generate=1, selected text, source URL, and source title.
- [x] #4 Capture validation failures remain inline and do not clear queued manual drafts.
- [x] #5 Native sidepanel LLM generation, templates, and in-extension review remain deferred.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-flashcards-extension-generate-handoff-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-27-flashcards-extension-generate-handoff-implementation-plan.md.

Touched scope: sidepanel Flashcards route/tests, extension/WebUI flashcards docs, master UX fix list, and the implementation plan.

Implementation notes: added Generate from selection beside existing sidepanel capture, sharing the same active-page selection reader and validation messages. The action opens full Flashcards with buildFlashcardsGenerateRoute so generated drafts stay in the existing full workspace while selected text, source URL, and title are prefilled. Manual queued capture/save-one/save-all behavior remains unchanged.

Non-goals: native sidepanel LLM generation, native generation templates, and in-extension review.

Verification:
- RED: bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx initially failed because Generate from selection was missing.
- PASS: bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx passed 23 tests.
- PASS: bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts passed 29 tests.
- PASS: bunx vitest run src/services/__tests__/flashcards-generate-handoff.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts passed 33 tests.
- PARTIAL: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still reports only the unrelated CharacterListContent design-system density baseline.
- PASS: git diff --check.
- Bandit not applicable: no Python files touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the F12 extension generate-from-selection handoff. Users can select text on a page and open full Flashcards generation directly from the sidepanel with selected text, source URL, and source title prefilled. Existing native sidepanel queued basic-card capture remains intact, and richer native sidepanel generated drafts/templates/review remain deferred to the full Flashcards workspace.

Focused sidepanel, route-registry, and generate-handoff tests pass. Package-wide TypeScript still reports the unrelated pre-existing CharacterListContent density baseline; no flashcards/sidepanel TypeScript errors were reported. Bandit is not applicable because this slice touches TypeScript/docs only and no Python files.
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
