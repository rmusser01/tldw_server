---
id: TASK-595
title: Clear final UI package TypeScript baseline diagnostics
status: Done
assignee: []
created_date: ''
updated_date: '2026-06-02 20:59'
labels:
  - typescript
  - webui
  - tests
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Clear the remaining package-wide TypeScript diagnostics in layout, sidepanel header handoff, and setup onboarding tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The UI package typecheck reports zero TypeScript diagnostics.
- [x] #2 Focused tests for the touched files are run where practical.
- [x] #3 Verification is recorded in the task.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Typed the external shell fixture with optional setOverrides, rendered SidepanelHeaderSimple directly so its props are preserved, and annotated the completed setup fixture as FirstRunState. Full package tsc now exits 0 with empty output. Bandit is not applicable for this JS/TS-only touched scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Cleared the final UI package TypeScript diagnostics in three tests. Verification: NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false exited 0 with empty output; bunx vitest run the three touched test files passed.
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
