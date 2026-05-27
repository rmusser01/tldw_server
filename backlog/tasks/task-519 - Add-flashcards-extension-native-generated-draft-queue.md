---
id: TASK-519
title: Add flashcards extension native generated draft queue
status: Done
assignee: []
created_date: ''
updated_date: 2026-05-27 05:30
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
- Docs/superpowers/plans/2026-05-27-flashcards-extension-native-generated-drafts-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the next narrow F12 follow-up: let the extension sidepanel generate a small set of editable draft cards from selected page text using the existing flashcards generation mutation, while keeping native template application and in-extension review deferred.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sidepanel Flashcards exposes Generate draft cards next to native capture and full-workspace generation handoff.
- [x] #2 Generate draft cards captures active-page selected text through the existing browser scripting path.
- [x] #3 Native generation uses the existing flashcards generation mutation with compact defaults and appends normalized generated cards to the sidepanel draft queue.
- [x] #4 Generated draft saves preserve model type, cloze/reverse flags, tags, notes, extra fields, selected deck, and page source provenance.
- [x] #5 Generation failures remain inline and do not clear queued manual drafts.
- [x] #6 Native sidepanel template application and in-extension review remain deferred.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-27-flashcards-extension-native-generated-drafts-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-05-27-flashcards-extension-native-generated-drafts-implementation-plan.md.

Touched scope: sidepanel Flashcards route/tests, extension/WebUI flashcards docs, master UX fix list, and the implementation plan.

Implementation notes: added Generate draft cards beside existing capture and full-workspace generation handoff actions. The new action shares the active-page selection reader, calls useGenerateFlashcardsMutation with compact defaults, normalizes results with normalizeGeneratedCards, and appends generated drafts to the existing editable queue. Generated draft save payloads preserve model type, cloze/reverse flags, tags, notes, extra fields, selected deck, and source URL provenance. Template application and in-extension review remain deferred.

Verification:
- RED: bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx failed with the expected missing Generate draft cards action after dependency setup.
- PASS: bunx vitest run src/routes/__tests__/sidepanel-flashcards.test.tsx passed 25 tests.
- PASS: bunx vitest run src/services/__tests__/flashcards-generate-handoff.test.ts src/routes/__tests__/sidepanel-flashcards.test.tsx src/routes/__tests__/route-registry.sidepanel-flashcards.test.ts passed 35 tests.
- PASS: git diff --check.
- PARTIAL: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still reports only the unrelated CharacterListContent design-system density baseline.
- Bandit not applicable: no Python files touched.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added native sidepanel generated draft batches for selected page text. The sidepanel now offers a compact Generate draft cards action that reads the active page selection, uses the existing flashcards generation mutation, normalizes generated cards through the existing shared helper, and appends editable drafts into the existing sidepanel queue. Generated drafts preserve model type, tags, notes, extra fields, source URL provenance, and reuse save-one/save-all recovery.

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
