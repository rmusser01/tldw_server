---
id: TASK-45.44.3.10
title: Migrate AlertsTab alerts to design-system Alert
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-24 01:48'
labels:
  - design-system
  - webui
  - watchlists
  - product-state
dependencies: []
parent_task_id: TASK-45.44.3
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-45.44.3 by replacing Watchlists AlertsTab AntD Alert product-state callouts with the shared design-system Alert primitive, preserving boundary guidance/error copy and retry behavior, removing migrated baseline exceptions, and recording focused verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 AlertsTab boundary guidance and load-error callouts render via design-system Alert.
- [x] #2 The AlertsTab Alert baseline exceptions are removed from design-system-product-state-baseline.json.
- [x] #3 Focused AlertsTab coverage asserts the design-system Alert marker for the boundary and error paths.
- [x] #4 Design-system product-state verification passes or records existing unrelated blockers.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Extend focused AlertsTab tests to assert boundary guidance and load-error callouts render inside `[data-ds-component="Alert"]`, watching the assertions fail while AlertsTab still uses AntD Alert.
2. Migrate AlertsTab Alert usage from AntD props to the design-system Alert primitive while preserving copy and refresh behavior.
3. Remove AlertsTab Alert exceptions from the product-state baseline and run focused Vitest plus design-system verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added RED coverage in AlertsTab.test.tsx for boundary guidance and load-error callouts requiring the design-system Alert marker; focused test failed while the component still rendered the AntD Alert mock.
- Migrated AlertsTab boundary guidance to Alert variant="info" and load-error messaging to Alert variant="error" with the existing refresh action preserved.
- Removed the two AlertsTab Alert entries from design-system-product-state-baseline.json.
- Verification: bunx vitest run src/components/Option/Watchlists/AlertsTab/__tests__/AlertsTab.test.tsx --reporter=dot passed 3 tests; bunx vitest run src/design-system/__tests__/product-state-guard.test.ts --reporter=dot passed 54 tests; bun run verify:design-system-state passed with 251 total exceptions and 18 Jobs/Scheduler/Watchlists exceptions; AlertsTab baseline rows are 0; git diff --check passed.
- TypeScript: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false still exits 2 with 347 existing diagnostics; no diagnostics mention AlertsTab, the AlertsTab test, the baseline file, or this task record.
- Bandit skipped because this slice only touches TypeScript/React test files, JSON baseline metadata, and Backlog task markdown.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Migrated the AlertsTab boundary guidance and load-error callouts from AntD Alert to the shared design-system Alert primitive, preserved the refresh behavior, added focused design-system marker coverage, and removed the two AlertsTab baseline exceptions. Focused Vitest, product-state guard, design-system verifier, and whitespace checks passed; TypeScript remains blocked by existing unrelated repo-wide diagnostics with no touched-file matches; Bandit was skipped for this UI-only slice.
<!-- SECTION:FINAL_SUMMARY:END -->

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
