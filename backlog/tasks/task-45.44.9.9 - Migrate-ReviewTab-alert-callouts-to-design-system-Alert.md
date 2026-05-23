---
id: TASK-45.44.9.9
title: Migrate ReviewTab alert callouts to design-system Alert
status: Done
labels:
- design-system
- webui
- extension
- product-state
- flashcards
priority: medium
parent_task_id: TASK-45.44.9
references:
- https://github.com/rmusser01/tldw_server/issues/1666
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2004
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/ReviewTab.create-cta.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Flashcards ReviewTab onboarding and review-retry callouts off AntD Alert and onto the canonical design-system Alert while preserving empty-state guidance and retry/reload behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ReviewTab onboarding and review-retry callouts render the design-system Alert primitive instead of AntD Alert.
- [x] #2 Focused ReviewTab coverage proves both alert texts/actions remain visible and wrapped in the canonical design-system marker.
- [x] #3 Design-system product-state verifier passes with the stale ReviewTab Alert baseline entry removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add failing focused ReviewTab assertions requiring onboarding and retry callouts to render with the design-system Alert marker.
2. Replace ReviewTab AntD Alert usages with the canonical design-system Alert primitive while preserving variants, copy, data-testids, action buttons, and behavior.
3. Remove the ReviewTab Alert entry from the product-state baseline and run focused tests plus the design-system verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
TDD red/green completed. Added focused ReviewTab assertions requiring both the first-run onboarding guide and review retry alert to carry the canonical data-ds-component Alert marker; the red run failed on both missing markers. Replaced both ReviewTab AntD Alert callouts with the shared design-system Alert primitive while preserving variants, copy, data-testids, retry/reload buttons, onboarding dismissal, and doc link behavior, then removed the stale ReviewTab Alert baseline entry.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated ReviewTab onboarding and review-retry callouts from AntD Alert to the design-system Alert primitive. Focused ReviewTab coverage now verifies both callouts carry data-ds-component="Alert", and the product-state baseline no longer contains the ReviewTab Alert exception. Verification: red ReviewTab.create-cta test failed on missing design-system markers for onboarding and retry alerts; green ReviewTab.create-cta suite passed 10/10; product-state guard passed 54/54; bun run verify:design-system-state passed with 262 allowed legacy exceptions and 38 remaining Flashcards/Quiz/study-flow exceptions; baseline JSON parse reported targetRows 0; git diff --check passed. TypeScript still exits 2 on 330 existing diagnostics, with no diagnostics for ReviewTab, ReviewTab.create-cta, the baseline, or TASK-45.44.9.9. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.
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
