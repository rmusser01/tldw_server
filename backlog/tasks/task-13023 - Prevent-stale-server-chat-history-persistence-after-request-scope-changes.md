---
id: TASK-13023
title: Prevent stale server-chat history persistence after request scope changes
status: Done
assignee: []
created_date: '2026-08-22 07:08'
updated_date: '2026-08-22 07:26'
labels:
  - chat
  - scope-isolation
  - frontend
dependencies: []
references:
  - TASK-13014
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make workspace and persona server-chat history linking fail closed when the captured server/account scope is invalidated, so stale Dexie mappings, history refs, and shared server-chat UI metadata cannot commit after a scope change.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Scoped history linking rolls back Dexie mutations when the dedicated scope-invalidated signal aborts before commit.
- [x] #2 Workspace and persona create/reuse paths publish server-chat metadata and invalidate history only after scoped history linking succeeds.
- [x] #3 Unscoped callers preserve existing behavior and public call compatibility.
- [x] #4 Focused regressions cover workspace creation, persona creation/reuse, and local history ref/persistence invalidation.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Trace the existing history-linking boundary; add RED regressions; implement the minimal scoped transaction and deferred-state commit; run focused tests, typecheck, lint, extension compile, and diff verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
RED: five focused regressions reproduced stale Dexie/ref and workspace/persona publication; an additional existing-chat preflight assertion also failed before its signal was threaded.

GREEN: 81 focused/adjacent Vitest tests passed; extension compile passed; focused ESLint reported 0 errors; git diff --check passed.

Known baseline: the mixed image-event suite retains four unrelated character-stream failures; package-wide UI test typecheck retains an unrelated docs fixture error at useChatActions.service-prompts.test.tsx:1030. All five image-sync tests passed. Bandit is not applicable because the touched implementation is TypeScript only; no documentation change was needed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made server-chat history linking scope-aware: scoped Dexie work now runs transactionally and publishes refs/history state only after scope validation. Workspace, persona create/reuse, existing-chat preflight, Compare, and per-model paths pass the dedicated invalidation signal and defer shared server-chat metadata until history linking succeeds. Added RED/GREEN regressions for rollback/cache isolation and deferred workspace/persona publication.
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
