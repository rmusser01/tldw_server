---
id: TASK-593
title: Reduce Flashcards duplicate fixture TypeScript baseline cluster
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:56'
labels:
  - typescript
  - webui
  - flashcards
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Reduce the package-wide TypeScript baseline by removing duplicate Flashcards deck fixture properties that trigger TS1117 diagnostics.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Flashcards TS1117 duplicate object literal diagnostics are removed from the package typecheck.
- [x] #2 Focused Flashcards tests for touched fixtures are run where practical.
- [x] #3 Verification is recorded in the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Removed duplicate review_prompt_side fields from five Flashcards deck fixtures. Full package tsc dropped from 12 src diagnostics to 7, with no Flashcards TS1117 diagnostics remaining. Bandit is not applicable for this JS/TS-only touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Reduced the Flashcards TypeScript baseline cluster by removing duplicated deck fixture properties in five tests. Verification: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false leaves the remaining non-Flashcards 7-diagnostic baseline; bunx vitest run the five touched Flashcards test files passed.
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
