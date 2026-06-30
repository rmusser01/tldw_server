---
id: TASK-597
title: Reduce frontend TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-03 01:05'
labels:
  - typescript
  - frontend
  - webui
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the standalone tldw-frontend TypeScript baseline after clearing the UI package, extension, and SDK checks.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The HeaderShortcuts NavLink diagnostics are removed from the frontend typecheck without changing launcher navigation behavior.
- [x] #2 A frontend tsc verification run is recorded with the updated diagnostic count.
- [x] #3 Focused or relevant checks are run where practical.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Switched launcher rows from NavLink to Link and preserved current-route indication with explicit aria-current from HeaderShortcuts own isCurrentShortcutRoute check. Frontend tsc dropped from 14 to 12 diagnostics with no HeaderShortcuts/NavLink diagnostics remaining. UI package tsc remains clean. Bandit is not applicable for this JS/TS-only touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the standalone tldw-frontend TypeScript baseline by removing the HeaderShortcuts NavLink end prop mismatch while preserving launcher navigation and current-route semantics. Verification: frontend tsc now reports the remaining 12 e2e diagnostics; UI package tsc exits 0; HeaderShortcuts.test.tsx passed.
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
