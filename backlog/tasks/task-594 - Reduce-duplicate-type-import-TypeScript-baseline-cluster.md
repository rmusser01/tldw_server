---
id: TASK-594
title: Reduce duplicate type import TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:57'
labels:
  - typescript
  - webui
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the package-wide TypeScript baseline by removing redundant test type imports that produce duplicate identifier diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Duplicate identifier diagnostics for StudySuggestionSnapshotResponse and ChatCompletionRequest are removed from the package typecheck.
- [x] #2 Focused tests for touched files are run where practical.
- [x] #3 Verification is recorded in the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed redundant standalone type imports from the StudySuggestions hook test and TldwApiClient chat sanitization regression test. Full package tsc dropped from 7 src diagnostics to 3, with no duplicate identifier diagnostics remaining. Bandit is not applicable for this JS/TS-only touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the duplicate type import TypeScript baseline cluster by relying on the existing grouped type imports in two tests. Verification: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false leaves the remaining 3-diagnostic baseline; bunx vitest run the two touched test files passed.
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
