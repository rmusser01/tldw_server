---
id: TASK-45.44.9.4
title: Migrate FlashcardStudyAssistantPanel warning alert to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1933
modified_files:
- apps/packages/ui/src/components/Flashcards/components/FlashcardStudyAssistantPanel.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Flashcards study-assistant warning UI off AntD Alert and onto the canonical design-system Alert, with focused coverage and baseline evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 FlashcardStudyAssistantPanel warning UI renders the design-system Alert primitive instead of AntD Alert.
- [x] #2 Design-system product-state verifier passes with the FlashcardStudyAssistantPanel Alert exception removed from the baseline.
- [x] #3 Focused ReviewTab assistant coverage verifies the warning still renders and exposes the design-system Alert marker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing ReviewTab assistant test assertion for the study-assistant warning state requiring the design-system Alert marker.
2. Replace the AntD Alert import/JSX in FlashcardStudyAssistantPanel with the canonical design-system Alert primitive.
3. Remove the stale product-state baseline entry and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. The first red run exposed a stale test harness mock for recent flashcard sessions; after adding the default hook mock, the focused test failed on the intended missing data-ds-component="Alert" assertion. Production code now imports the canonical design-system Alert primitive and renders the study-assistant warning with variant="warning". PR review follow-up moved the longer assistant warning message into Alert children while keeping a short "Study assistant unavailable" title, with a focused red/green assertion. Removed the now-stale baseline entry for FlashcardStudyAssistantPanel Alert.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated FlashcardStudyAssistantPanel's study-assistant warning from AntD Alert to the design-system Alert primitive. Added focused ReviewTab assistant coverage that verifies the warning remains visible, carries the design-system Alert marker, and uses a short design-system Alert title with the longer warning message in the body. Removed the stale product-state baseline exception and verified the guard passes with 319 remaining legacy exceptions. Verification: bunx vitest run src/components/Flashcards/tabs/__tests__/ReviewTab.assistant.test.tsx; bun run verify:design-system-state; git diff --check. Bandit skipped: frontend-only TSX/test/JSON change.
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
