---
id: TASK-13023
title: Prevent stale server-chat history persistence after request scope changes
status: Done
assignee: []
created_date: '2026-08-22 07:08'
updated_date: '2026-08-22 15:19'
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

Independent pre-publication review of 8d83b6d3..ed57d8aa found no Critical, Important, or Minor issues and assessed the change ready to merge.

Published as draft PR #2799: https://github.com/rmusser01/tldw_server/pull/2799. The repository-required human-authored Change summary remains the only known manual merge gate at publication time.

Qodo review on PR #2799 posted two threads: remove the transaction-helper implementation assertion, and prevent an aborted/superseded useServerChatLoader load from committing local history/title/load state. Root cause: shouldCommitServerChatLoadResult does not reject an already-aborted owned controller, and the best-effort local-mirror catch can continue after cancellation.

Qodo remediation RED: 2 focused failures reproduced the gap—an aborted owned loader still passed shouldCommitServerChatLoadResult, and superseded Chat A published its title after Chat B replaced it. GREEN: the loader now rejects aborted ownership, forwards its per-load signal into scoped history linking, and checks ownership before/after the link and before final publication; the brittle transaction-helper call assertion was removed while behavior coverage remains. Verification: 84/84 focused and adjacent tests passed; extension compile passed; focused ESLint reported 0 errors (pre-existing warnings only); git diff --check passed. Bandit remains not applicable because all touched implementation is TypeScript.

Post-review rebase: rebased conflict-free onto origin/dev 424cb464a6225d71adbcd1fcedcb0a73853a2055, then reran the exact focused gates: 84/84 tests passed, extension compile passed, focused ESLint reported 0 errors, and branch diff check passed.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made server-chat history linking scope-aware: scoped Dexie work now runs transactionally and publishes refs/history state only after scope validation. Workspace, persona create/reuse, existing-chat preflight, Compare, and per-model paths pass the dedicated invalidation signal and defer shared server-chat metadata until history linking succeeds. Added RED/GREEN regressions for rollback/cache isolation and deferred workspace/persona publication.

The reviewed branch was rebased onto current dev and published as draft PR #2799.

Addressed both Qodo review findings with behavior-first coverage: removed the internal transaction-call assertion and made server-chat loading fail closed when its per-load controller is aborted, preventing superseded history linking, title publication, and loaded-state publication.
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
