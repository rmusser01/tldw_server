---
id: TASK-45.44.9.6
title: Migrate SourceSeenDrawer error alert to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-22 19:28'
labels:
  - design-system
  - webui
  - extension
  - product-state
  - watchlists
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1666'
  - >-
    apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceSeenDrawer.tsx
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45.44.9
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Move the Watchlists SourceSeenDrawer load error UI off AntD Alert and onto the canonical design-system Alert while preserving loading, retry/context, and seen-item list behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 SourceSeenDrawer load error callout renders the design-system Alert primitive instead of AntD Alert.
- [x] #2 Existing focused SourceSeenDrawer coverage proves the error text remains visible and wrapped in the canonical design-system marker.
- [x] #3 Design-system product-state verifier passes with the stale SourceSeenDrawer Alert baseline entry removed.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add a failing SourceSeenDrawer test assertion requiring the load-error callout to render with the design-system Alert marker.
2. Replace the SourceSeenDrawer AntD Alert usage with the canonical design-system Alert primitive while preserving error copy and spacing.
3. Remove the SourceSeenDrawer Alert entry from the product-state baseline and run focused tests plus the design-system verifier.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD red/green completed. Added a failing SourceSeenDrawer error-state assertion requiring the load-error callout to be wrapped by the canonical data-ds-component Alert marker; the initial focused run failed because the AntD Alert mock rendered only the error text. Replaced the SourceSeenDrawer AntD Alert import/usage with the shared design-system Alert primitive while preserving the existing error title and spacing, and removed the stale SourceSeenDrawer Alert baseline entry.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the Watchlists SourceSeenDrawer load-error callout from AntD Alert to the design-system Alert primitive. Focused coverage now verifies the error text renders in the canonical Alert wrapper, and the product-state baseline no longer contains the SourceSeenDrawer Alert exception. Verification: red focused SourceSeenDrawer test failed on the missing design-system marker; green focused SourceSeenDrawer test passed 14/14; bun run verify:design-system-state passed with 316 allowed legacy exceptions; baseline JSON parse passed; git diff --check passed. Bandit skipped because this slice touched frontend TSX/test/JSON/Backlog markdown only.
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
