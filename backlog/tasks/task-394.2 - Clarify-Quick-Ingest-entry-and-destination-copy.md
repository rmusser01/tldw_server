---
id: TASK-394.2
title: Clarify Quick Ingest entry and destination copy
status: Done
assignee: []
created_date: '2026-05-16 00:42'
updated_date: '2026-05-16 02:34'
labels:
  - quick-ingest
  - ux
  - task-2
dependencies: []
documentation:
  - >-
    Docs/superpowers/plans/2026-05-16-quick-ingest-ux-remediation-implementation-plan.md
parent_task_id: TASK-394
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute implementation plan Task 2: improve first-time clarity, destination expectations, entry consistency, and accessible labels for the active Quick Ingest wizard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Entry labels and wizard copy clarify Quick Ingest purpose and where items go
- [x] #2 First-time and returning-user flows keep the same quick path with better recognition over recall
- [x] #3 Accessible labels and keyboard/focus behavior are preserved or improved
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Task 2 started on branch codex/quick-ingest-ux-remediation after Task 1 path-map review approval. Scope: first-time clarity copy, entry terminology consistency, optional review estimate cleanup, focused tests, and browser verification.

Task 2 implementation commit: e0ff4e144 fix: clarify quick ingest entry and destination. Review checkpoints passed: spec compliance approved, code quality approved with only a non-blocking note about broad Media/Knowledge assertions. Verification: UI package Vitest command passed from apps/packages/ui (2 files, 29 tests); git diff --check passed; focused Playwright browser check passed against http://127.0.0.1:18001 with seeded auth, confirming visible Quick Ingest trigger, destination copy, invalid URL validation, and Use defaults & process after valid URL. The root bunx command in the original plan was stale because Vitest 4 does not support --runInBand and root invocation lacks the UI alias/jsdom config. Bandit not run because this slice changed TypeScript/TSX frontend files only.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Clarified the active Quick Ingest entry flow by changing the header label to Quick Ingest, adding concise Add-step destination copy for Media and Knowledge, and removing the duplicate review-time tilde. Added focused active-wizard integration coverage for the first-open copy. Verified with package-local Vitest, diff check, focused Playwright browser verification, and two-stage subagent review.
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
