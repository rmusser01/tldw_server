---
id: TASK-45.44.2.4
title: Migrate TimelineModal alerts to design-system Alert
status: Done
labels:
- design-system
- webui
- product-state
- timeline
priority: medium
parent_task_id: TASK-45.44.2
references:
- apps/packages/ui/src/components/Timeline/TimelineModal.tsx
- apps/packages/ui/src/components/Timeline/__tests__/TimelineModal.product-state.test.tsx
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/1785
documentation:
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue the Ingestion, Library, and media design-system migration by replacing TimelineModal's remaining AntD Alert product-state usage with the shared design-system Alert primitive while preserving loading, error, and empty timeline behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TimelineModal renders error and empty timeline product-state messages through the shared design-system Alert primitive instead of AntD Alert.
- [x] #2 Existing loading, graph, error, empty, and close behavior remain covered by focused tests.
- [x] #3 The product-state baseline no longer contains TimelineModal AntD Alert exceptions.
- [x] #4 Focused Vitest, design-system verifier, git diff check, and TypeScript/Bandit applicability are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add focused TimelineModal coverage proving error and empty states render through the design-system Alert primitive while loading and graph states remain distinct.
2. Run the focused test before implementation to capture the missing design-system Alert marker while TimelineModal still uses AntD Alert.
3. Replace the two AntD Alert usages with the shared Alert primitive, mapping error to variant="error" and empty to variant="info".
4. Remove the two TimelineModal AntD Alert baseline exceptions.
5. Verify focused Vitest, product-state guard/verifier, git diff check, and TypeScript/Bandit applicability.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented via TDD. The focused TimelineModal product-state test failed first on the missing design-system Alert marker for the error and empty states while loading and graph states already passed. TimelineModal now imports the shared Alert primitive for the two product-state messages, maps error to variant="error" and empty to variant="info", and keeps the existing AntD Modal and Spin mechanics unchanged. Removed the two TimelineModal AntD Alert baseline exceptions.

PR opened against dev: https://github.com/rmusser01/tldw_server/pull/1785

PR review follow-up: removed the now-redundant AntD Alert mock from the TimelineModal product-state test after the component migrated to the design-system Alert primitive. Focused TimelineModal plus product-state guard Vitest remained green.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated TimelineModal error and empty product-state alerts from AntD Alert to the shared design-system Alert primitive, added focused coverage for loading, error, empty, graph, and close behavior, and reduced the product-state baseline from 398 to 396 exceptions.

Verification: RED focused Vitest failed on missing design-system Alert markers for error and empty states; GREEN focused TimelineModal plus product-state guard Vitest passed 56 tests; bun run verify:design-system-state passed with 396 allowed legacy exceptions; baseline JSON parse passed; git diff --check passed. bunx tsc --noEmit --pretty false still exits 2 on existing repo-wide TypeScript debt, with no touched-file matches for TimelineModal, TimelineModal.product-state, or design-system-product-state-baseline. Bandit skipped because this slice touches frontend TypeScript, JSON baseline data, and Backlog metadata only.

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
