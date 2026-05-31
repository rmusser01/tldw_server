---
id: TASK-45.44.9.3
title: Migrate TemplatesTab load-error alert to design-system Alert
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
- apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1932
modified_files:
- apps/packages/ui/src/components/Flashcards/tabs/TemplatesTab.tsx
- apps/packages/ui/src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Flashcards TemplatesTab load-error UI off AntD Alert and onto the canonical design-system Alert, with focused coverage and baseline evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TemplatesTab load-error UI renders the design-system Alert primitive instead of AntD Alert.
- [ ] #2 Design-system product-state verifier passes with the TemplatesTab Alert exception removed from the baseline.
- [ ] #3 Focused TemplatesTab coverage verifies the load-error banner still renders and exposes the design-system Alert marker.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing TemplatesTab test for the load-error state requiring the design-system Alert marker and preserved copy.
2. Replace the AntD Alert import/JSX with the canonical design-system Alert primitive.
3. Remove the stale product-state baseline entry and run focused verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed a narrow Flashcards product-state migration slice and opened PR #1932. Verification: red test first failed on missing data-ds-component marker; `bunx vitest run src/components/Flashcards/tabs/__tests__/TemplatesTab.test.tsx` passes 15 tests; `bun run verify:design-system-state` reports 320 baseline exceptions with no stale TemplatesTab entry; `git diff --check` passes. Bandit skipped because this slice only changes frontend TSX/test/baseline JSON.
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
